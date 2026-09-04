# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The GPU launcher: the command it builds, the log it reads, the state it keeps.

Pure only, in the repo's stated sense — no subprocess, no network, no sleeping.
The clock is injected (`_clock`, as `remote_inference`'s watchdogs are), the
`Popen` is a fake, and the pump is driven line by line rather than threaded.

DELIBERATELY NOT TESTED, and it is the same list the module docstring carries:
the real `modal run`, Modal authentication, actual cold-start timing, the
tailscale relay, Modal's log-stream reconnection, and the wrappers' own env
fallback — `modal_policy*.py` import `modal` at module top and the Lab venv has
no `modal`, so those two lines are not importable from here. The argv test
below covers the half that CAN be checked from this side: that the Lab does not
pass the secret as a flag, and does pass it in the environment.
"""

from __future__ import annotations

import signal
import subprocess
import types

import pytest

from makermodslab import modal_launcher as ml, remote_inference as ri
from makermodslab.api_errors import ApiError, ErrorCode

SECRET = "test-secret"  # noqa: S105  # nosec B105 — a fixture value, not a credential


class FakeClock:
    """A controllable stand-in for time.monotonic (tests/test_session_lease.py)."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakePopen:
    """The minimum surface the launcher touches on a subprocess.

    `stdout` stays None: the pump is exercised line by line rather than
    threaded, which is what keeps these tests pure. A process whose stdout
    never closes is therefore modelled by simply never calling `_handle_exit`
    — which is exactly the wedged-pipe case the drain bound exists for.
    """

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.stdout = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode


def _plan(*, needs_tailscale: bool = True, url: str = "ws://100.64.0.1:7880") -> ml.TransportPlan:
    return ml.TransportPlan(
        url=url,
        room="mml-abcdef123456",
        api_key="APIkeyname",
        api_secret=SECRET,
        needs_tailscale=needs_tailscale,
        source="sfu" if needs_tailscale else "cloud",
    )


@pytest.fixture(autouse=True)
def _fresh_launcher(tmp_path, monkeypatch):
    """Every test starts and ends with an idle launcher and the real clock.

    It also redirects the persisted app-id record into tmp_path FOR EVERY TEST
    in this module — not just the ones that exercise it. The real file lives in
    the developer's `~/.cache/huggingface/lerobot/`, and a test that wrote
    there would be interfering with the machine's actual state.
    """
    monkeypatch.setattr(ml, "_APP_RECORD_FILE", tmp_path / "gpu_app.json")
    ml._go_idle_locked()
    ml._tail.clear()
    ml._log_path = None
    ml._message = None
    ml._hint = None
    ml._stop_outcome = ml.STATE_IDLE
    ml._code = None
    ml._drain_deadline = None
    ml._reaped = False
    yield
    ml._go_idle_locked()
    ml._tail.clear()
    ml._reaped = False


@pytest.fixture
def fake_clock(monkeypatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(ml, "_clock", clock)
    return clock


@pytest.fixture
def spawned(monkeypatch, tmp_path):
    """Make `start()` deterministic: a fake process, no pump thread, no kill.

    Returns a dict recording what each seam was handed, so a test can assert on
    the argv and the child environment without a subprocess anywhere.
    """
    calls: dict[str, object] = {"popen": [], "terminated": []}
    proc = FakePopen()

    def _popen(argv, env):
        calls["popen"].append((argv, env))
        # A test that needs a SECOND, distinct process (the orphan guard) drops
        # it here; everything else gets the one fake.
        return calls.get("popen_result") or proc

    monkeypatch.setattr(ml, "_popen", _popen)
    monkeypatch.setattr(ml, "_start_pump", lambda *a, **k: None)
    monkeypatch.setattr(ml, "_terminate_async", lambda p: calls["terminated"].append(p))
    monkeypatch.setattr(ml, "_open_log", lambda: ((tmp_path / "gpu.log").open("w"), tmp_path / "gpu.log"))
    monkeypatch.setattr(ml, "find_modal", lambda: "/usr/bin/modal")
    monkeypatch.setattr(ml, "resolve_transport_plan", _plan)
    monkeypatch.setattr(ml, "_remote_inference_is_active", lambda: False)
    calls["proc"] = proc
    return calls


# --- the argv builder --------------------------------------------------------


_ARGS = {
    "policy_hub_id": "someone/some-policy",
    "task": "Put the lego brick in the box",
    "horizon": 16,
    "fps": 30,
    "video_codec": "H264",
    "s_min": 4,
    "modal_bin": "/usr/bin/modal",
}


def test_each_engine_runs_its_own_wrapper_by_absolute_path():
    """The two servers publish different state schemas, and Portal drops a
    mismatched stream in silence — so the engine picks the FILE, not a flag."""
    sync = ml.build_argv(_plan(), **_ARGS, engine="sync")
    rtc = ml.build_argv(_plan(), **_ARGS, engine="rtc")

    assert sync[:2] == ["/usr/bin/modal", "run"]
    assert sync[2].endswith("makermodslab/drtc/modal_policy.py")
    assert rtc[2].endswith("makermodslab/drtc/modal_policy_rtc.py")
    assert sync[2].startswith("/") and rtc[2].startswith("/")


def test_the_command_is_a_list_never_a_string():
    """No shell anywhere, so no metacharacter handling anywhere either."""
    argv = ml.build_argv(_plan(), **_ARGS, engine="sync")
    assert isinstance(argv, list)
    assert all(isinstance(part, str) for part in argv)


def test_task_is_passed_when_set_and_omitted_when_empty():
    with_task = ml.build_argv(_plan(), **{**_ARGS, "task": "grab it"}, engine="sync")
    assert with_task[with_task.index("--task") + 1] == "grab it"

    without = ml.build_argv(_plan(), **{**_ARGS, "task": ""}, engine="sync")
    assert "--task" not in without


def test_s_min_is_rtc_only():
    """`modal_policy.py` has no --s-min at all: emitting it there is a Click
    usage error, not a run that falls back to a default."""
    assert "--s-min" not in ml.build_argv(_plan(), **_ARGS, engine="sync")
    rtc = ml.build_argv(_plan(), **_ARGS, engine="rtc")
    assert rtc[rtc.index("--s-min") + 1] == "4"


def test_the_room_is_always_pinned():
    """Without --livekit-room the GPU takes the room from its own Modal secret,
    which the Lab cannot read and therefore cannot check — the one mismatch
    that is invisible by construction."""
    for engine in ("sync", "rtc"):
        for tailscale in (True, False):
            argv = ml.build_argv(_plan(needs_tailscale=tailscale), **_ARGS, engine=engine)
            assert argv[argv.index("--livekit-room") + 1] == "mml-abcdef123456"


def test_tailscale_flags_ride_on_the_plan_alone():
    ts = ml.build_argv(_plan(needs_tailscale=True), **_ARGS, engine="sync")
    assert "--tailscale" in ts
    assert ts[ts.index("--livekit-url") + 1] == "ws://100.64.0.1:7880"

    cloud = ml.build_argv(_plan(needs_tailscale=False), **_ARGS, engine="sync")
    assert "--tailscale" not in cloud
    assert "--livekit-url" not in cloud


def test_the_secret_is_never_in_argv_and_always_in_the_child_env():
    """THE load-bearing assertion of this module.

    argv is world-readable in `ps` on this machine, so a regression that
    re-introduces `--livekit-api-secret` is a credential leak that nothing else
    here would catch. Both wrappers' `main()` falls back to the environment
    precisely so this flag never has to be passed."""
    plan = _plan()
    argv = ml.build_argv(plan, **_ARGS, engine="rtc")

    assert "--livekit-api-secret" not in argv
    assert "--livekit-api-key" not in argv
    assert SECRET not in " ".join(argv)

    env = ml.child_env(plan, env={"PATH": "/usr/bin"})
    assert env["LIVEKIT_API_SECRET"] == SECRET
    assert env["LIVEKIT_API_KEY"] == "APIkeyname"
    assert env["PYTHONUNBUFFERED"] == "1"


# --- which workspace pays (S3.8b) --------------------------------------------


def test_the_environment_is_a_modal_run_option_and_sits_before_the_wrapper():
    """`modal run [OPTIONS] FUNC_REF`. After the path, Click hands `--env` to
    the wrapper's own local_entrypoint — which has no such parameter — and the
    run dies on an unknown flag instead of billing the named environment."""
    argv = ml.build_argv(_plan(), **_ARGS, engine="sync", environment="staging")

    assert argv[:4] == ["/usr/bin/modal", "run", "--env", "staging"]
    assert argv[4].endswith("modal_policy.py")
    assert argv.index("--env") < argv.index("--policy-path")


def test_no_environment_means_no_flag_at_all():
    """Empty is S3.8's behaviour byte for byte: the CLI resolves the
    environment itself (MODAL_ENVIRONMENT, the active profile, the workspace
    default). Passing `--env ""` would name an environment called ""."""
    for argv in (
        ml.build_argv(_plan(), **_ARGS, engine="sync"),
        ml.build_argv(_plan(), **_ARGS, engine="rtc", environment=""),
    ):
        assert "--env" not in argv


def test_the_profile_travels_in_the_child_env_and_never_in_argv():
    """`MODAL_PROFILE` per process is what lets ONE launch bill another
    workspace without `modal profile activate` rewriting the ~/.modal.toml
    every other terminal on this machine shares."""
    plan = _plan()
    argv = ml.build_argv(plan, **_ARGS, engine="sync", environment="main")
    assert "--profile" not in argv
    assert "work-account" not in " ".join(argv)

    env = ml.child_env(plan, env={"PATH": "/usr/bin"}, profile="work-account")
    assert env["MODAL_PROFILE"] == "work-account"


def test_an_unchosen_profile_leaves_the_variable_alone():
    """Not set to "" — the CLI would read that as a profile named ""."""
    env = ml.child_env(_plan(), env={"PATH": "/usr/bin"})
    assert "MODAL_PROFILE" not in env

    inherited = ml.child_env(_plan(), env={"MODAL_PROFILE": "from-the-shell"})
    assert inherited["MODAL_PROFILE"] == "from-the-shell"


# --- precision and the GPU type (S3.8e) --------------------------------------


def test_the_precision_is_a_flag_and_only_when_it_is_set():
    """Unset is not a default this side picks — it is the dtype the checkpoint
    was saved with, and the only way to ask for that is to pass nothing."""
    argv = ml.build_argv(_plan(), **_ARGS, engine="sync", model_dtype="bfloat16")
    assert argv[argv.index("--model-dtype") + 1] == "bfloat16"
    # Where both wrappers' local_entrypoint signatures put it, so the list
    # reads in the order the function declares.
    assert argv.index("--task") < argv.index("--model-dtype") < argv.index("--horizon")

    for unset in (
        ml.build_argv(_plan(), **_ARGS, engine="sync"),
        ml.build_argv(_plan(), **_ARGS, engine="rtc", model_dtype=""),
    ):
        assert "--model-dtype" not in unset


def test_the_gpu_type_travels_in_the_child_env_because_no_flag_could_reach_it():
    """`_FN_KWARGS["gpu"]` is evaluated when `modal run` IMPORTS the wrapper on
    this machine — before Click parses anything — so the decorator is already
    built by the time a flag could be read. `DRTC_GPU` is the only channel."""
    plan = _plan()
    for engine in ("sync", "rtc"):
        argv = ml.build_argv(plan, **_ARGS, engine=engine)
        assert "--gpu" not in argv
        assert "H100" not in " ".join(argv)

    env = ml.child_env(plan, env={"PATH": "/usr/bin"}, gpu="H100")
    assert env["DRTC_GPU"] == "H100"


def test_an_unchosen_gpu_leaves_the_variable_alone():
    """Empty means "whatever the wrapper pins", which is what NOT exporting it
    gets — `DRTC_GPU=""` would have each wrapper's `or` fall through anyway,
    but a variable we set is a variable we have to explain."""
    env = ml.child_env(_plan(), env={"PATH": "/usr/bin"})
    assert "DRTC_GPU" not in env

    inherited = ml.child_env(_plan(), env={"DRTC_GPU": "from-the-shell"})
    assert inherited["DRTC_GPU"] == "from-the-shell"


def test_an_off_list_gpu_refuses_before_anything_is_spawned(spawned):
    """The allowlist is closed because the value fails somewhere nobody is
    watching: it reaches Modal at the wrapper's import, and a plausible wrong
    answer is an hour billed on hardware nobody chose."""
    with pytest.raises(ApiError) as excinfo:
        ml.start(engine="sync", policy_hub_id="someone/p", gpu="A100-40GB")

    assert excinfo.value.status_code == 400
    assert excinfo.value.code == ErrorCode.GPU_LAUNCH_FAILED
    # The message names the way out, not just the refusal.
    assert "`A100-80GB`" in excinfo.value.detail and "`H100`" in excinfo.value.detail
    assert spawned["popen"] == []


def test_an_off_list_precision_refuses_before_anything_is_spawned(spawned):
    """The wrapper refuses an unknown dtype too — but in the container, after a
    cold start. A name that cannot be right is worth a millisecond here."""
    with pytest.raises(ApiError) as excinfo:
        ml.start(engine="rtc", policy_hub_id="someone/p", model_dtype="bf16")

    assert excinfo.value.code == ErrorCode.GPU_LAUNCH_FAILED
    assert "`bfloat16`" in excinfo.value.detail
    assert spawned["popen"] == []


def test_the_two_knobs_reach_the_flag_the_env_and_the_status(spawned, fake_clock):
    ml.start(
        engine="sync",
        policy_hub_id="someone/p",
        model_dtype=" bfloat16 ",  # stripped, like the profile
        gpu=" H100 ",
    )

    argv, env = spawned["popen"][0]
    assert argv[argv.index("--model-dtype") + 1] == "bfloat16"
    assert env["DRTC_GPU"] == "H100"
    status = ml.status()
    assert status["model_dtype"] == "bfloat16"
    assert status["gpu"] == "H100"


def test_an_unchosen_knob_is_echoed_as_launched_not_as_a_non_choice(spawned, fake_clock, monkeypatch):
    """`task`'s convention, not `profile`'s: "" is a REAL answer here — the
    dtype the checkpoint saved, and the wrapper's own pin — so it is echoed as
    sent. Null means idle, and nothing else."""
    # The child inherits os.environ, so a developer shell that already exports
    # DRTC_GPU would otherwise decide the assertion below.
    monkeypatch.delenv("DRTC_GPU", raising=False)
    idle = ml.status()
    assert idle["model_dtype"] is None and idle["gpu"] is None

    ml.start(engine="sync", policy_hub_id="someone/p")
    assert ml.status()["model_dtype"] == ""
    assert ml.status()["gpu"] == ""
    _, env = spawned["popen"][0]
    assert "DRTC_GPU" not in env

    ml.stop()
    ml._handle_exit(spawned["proc"], -2)
    assert ml.status()["gpu"] is None


# --- the two listings --------------------------------------------------------

_PROFILE_ROWS = [
    {"name": "personal", "workspace": "mokuroh54", "active": False},
    {"name": "work", "workspace": "makermods", "active": True},
]
# The environment listing's real shape: a key with a SPACE in it, and `active`
# as the STRING "True" rather than a JSON boolean.
_ENV_ROWS = [
    {"name": "main", "web suffix": "", "active": "True"},
    {"name": "staging", "web suffix": "-staging", "active": "False"},
]


def test_profiles_are_parsed_with_their_workspace_and_active_flag():
    assert ml.parse_profiles(_PROFILE_ROWS) == [
        {"name": "personal", "workspace": "mokuroh54", "active": False},
        {"name": "work", "workspace": "makermods", "active": True},
    ]


def test_environments_are_parsed_through_the_string_boolean():
    """`modal environment list --json` serializes its table renderer's text, so
    `active` arrives as "True". A picker that read that as truthy-string would
    mark every environment active."""
    assert ml.parse_environments(_ENV_ROWS) == [
        {"name": "main", "active": True},
        {"name": "staging", "active": False},
    ]


@pytest.mark.parametrize(
    "payload", [None, {}, "nope", [None, 3, "x"], [{"workspace": "w"}], [{"name": "  "}]]
)
def test_a_listing_this_build_cannot_read_degrades_to_no_choices(payload):
    """Never a blank option: a row with no usable name would launch against
    something unnamed, and no picker at all is the honest answer."""
    assert ml.parse_profiles(payload) == []
    assert ml.parse_environments(payload) == []


def test_a_missing_cli_is_a_coded_body_never_a_raise(monkeypatch):
    """This sits inside a GET the panel calls on open. A failed listing costs
    the two pickers and NOTHING else — the CLI's own resolution still works, so
    Start GPU must stay live."""
    monkeypatch.setattr(ml, "find_modal", lambda: None)
    out = ml.list_targets()
    assert out["error"]["code"] == "gpu.cli_missing"
    assert "uv tool install modal" in out["error"]["message"]
    assert out["profiles"] == [] and out["environments"] == [] and out["profile"] is None


def test_the_listings_are_read_only_and_name_the_profile_they_describe(monkeypatch):
    seen: list[tuple[list[str], str]] = []

    def _fake(argv, *, profile=""):
        seen.append((argv, profile))
        return (_PROFILE_ROWS if argv[1] == "profile" else _ENV_ROWS), None

    monkeypatch.setattr(ml, "find_modal", lambda: "/usr/bin/modal")
    monkeypatch.setattr(ml, "_modal_json", _fake)

    out = ml.list_targets()
    # The active profile is what the environments were listed for when none was
    # asked for — reported so a stale list cannot read as current.
    assert out["profile"] == "work"
    assert out["error"] is None
    assert [e["name"] for e in out["environments"]] == ["main", "staging"]
    # Only `list`, only `--json`, and never `activate` / `create` / `token`.
    assert [argv[1:] for argv, _ in seen] == [
        ["profile", "list", "--json"],
        ["environment", "list", "--json"],
    ]
    assert [p for _, p in seen] == ["", ""]


def test_another_profile_s_environments_are_listed_under_the_profile_env_var(monkeypatch):
    """`modal environment list` describes ONE profile's workspace, and the env
    var is how another one is asked for — never `modal profile activate`."""
    seen: list[tuple[list[str], str]] = []

    def _fake(argv, *, profile=""):
        seen.append((argv, profile))
        return (_PROFILE_ROWS if argv[1] == "profile" else _ENV_ROWS), None

    monkeypatch.setattr(ml, "find_modal", lambda: "/usr/bin/modal")
    monkeypatch.setattr(ml, "_modal_json", _fake)

    out = ml.list_targets("personal")
    assert out["profile"] == "personal"
    assert seen[1][1] == "personal"


def test_the_profile_env_var_is_the_only_thing_a_listing_changes(monkeypatch):
    """The one place the subprocess is real enough to check: the env var goes
    in, the argv stays a fixed list, and no shell is involved."""
    captured: dict[str, object] = {}

    class _Done:
        returncode = 0
        stdout = '[{"name": "main", "active": "True"}]'
        stderr = ""

    def _run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        captured["timeout"] = kwargs["timeout"]
        captured["kwargs"] = kwargs
        return _Done()

    monkeypatch.setattr(ml.subprocess, "run", _run)
    payload, err = ml._modal_json(["/usr/bin/modal", "environment", "list", "--json"], profile="personal")

    assert err is None and payload == [{"name": "main", "active": "True"}]
    assert captured["env"]["MODAL_PROFILE"] == "personal"
    assert captured["timeout"] == ml._TARGETS_TIMEOUT_S
    assert "shell" not in captured["kwargs"]


def test_a_profile_this_machine_does_not_have_is_refused_rather_than_silently_swapped(monkeypatch):
    """Falling back to the active profile here would show a DIFFERENT
    workspace's environments under the label the operator picked."""
    monkeypatch.setattr(ml, "find_modal", lambda: "/usr/bin/modal")
    monkeypatch.setattr(ml, "_modal_json", lambda argv, *, profile="": (_PROFILE_ROWS, None))

    out = ml.list_targets("does-not-exist")
    assert out["error"]["code"] == "gpu.targets_unavailable"
    assert "does-not-exist" in out["error"]["message"]
    assert out["environments"] == []
    assert out["profile"] is None
    # The profiles still come back: picking one is what re-runs the listing.
    assert [p["name"] for p in out["profiles"]] == ["personal", "work"]


def test_an_unauthenticated_listing_names_the_one_remedy_that_differs():
    code, message = ml._listing_error("Error: token id and token secret are missing", 1)
    assert code == "gpu.unauthenticated"
    assert "modal token new" in message
    assert "never reads or writes ~/.modal.toml" in message


def test_an_unclassified_listing_failure_quotes_the_cli_s_own_last_line():
    code, message = ml._listing_error("warming up\nError: workspace lookup failed", 2)
    assert code == "gpu.targets_unavailable"
    assert "Error: workspace lookup failed" in message
    assert "exit code 2" in message


def test_output_that_is_not_json_costs_the_pickers_and_says_so(monkeypatch):
    class _Done:
        returncode = 0
        stdout = "Profile  Workspace\npersonal mokuroh54"
        stderr = ""

    monkeypatch.setattr(ml.subprocess, "run", lambda argv, **kw: _Done())
    payload, err = ml._modal_json(["/usr/bin/modal", "profile", "list", "--json"])
    assert payload is None
    assert err[0] == "gpu.targets_unavailable"
    assert "the launch still uses the CLI's own active profile" in err[1]


# --- choosing a target at launch ---------------------------------------------


def test_choosing_nothing_checks_nothing(monkeypatch):
    """The default path must not pay for a listing — and must not be able to
    fail because of one."""
    monkeypatch.setattr(ml, "list_targets", lambda profile="": pytest.fail("no listing here"))
    ml.check_target("", "   ")


def test_an_unknown_environment_refuses_before_anything_is_spawned(spawned, monkeypatch):
    monkeypatch.setattr(
        ml,
        "list_targets",
        lambda profile="": {
            "profiles": ml.parse_profiles(_PROFILE_ROWS),
            "environments": ml.parse_environments(_ENV_ROWS),
            "profile": "work",
            "error": None,
        },
    )
    with pytest.raises(ApiError) as excinfo:
        ml.start(engine="sync", policy_hub_id="someone/p", environment="prod")

    assert excinfo.value.status_code == 400
    assert excinfo.value.code == ErrorCode.GPU_LAUNCH_FAILED
    assert "`main`" in excinfo.value.detail and "`staging`" in excinfo.value.detail
    assert spawned["popen"] == []


def test_a_target_that_cannot_be_confirmed_refuses_rather_than_guesses(spawned, monkeypatch):
    """An A100-hour billed to the wrong workspace is not recoverable. A typo is
    loud ninety seconds later; someone else's real workspace is silent and
    billed — so an unconfirmable target costs a retry, not a guess."""
    monkeypatch.setattr(
        ml,
        "list_targets",
        lambda profile="": {
            "profiles": [],
            "environments": [],
            "profile": None,
            "error": {"code": "gpu.unauthenticated", "message": "Modal rejected this machine."},
        },
    )
    with pytest.raises(ApiError) as excinfo:
        ml.start(engine="sync", policy_hub_id="someone/p", profile="work")

    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "gpu.unauthenticated"
    assert "Clear the profile and environment" in excinfo.value.detail
    assert spawned["popen"] == []


def test_the_chosen_target_reaches_the_env_the_argv_and_the_status(spawned, fake_clock, monkeypatch):
    monkeypatch.setattr(
        ml,
        "list_targets",
        lambda profile="": {
            "profiles": ml.parse_profiles(_PROFILE_ROWS),
            "environments": ml.parse_environments(_ENV_ROWS),
            "profile": profile or "work",
            "error": None,
        },
    )
    ml.start(engine="sync", policy_hub_id="someone/p", profile=" work ", environment=" staging ")

    argv, env = spawned["popen"][0]
    assert env["MODAL_PROFILE"] == "work"  # stripped
    assert argv[argv.index("--env") + 1] == "staging"
    # The whole point of the feature: a running GPU says which workspace pays.
    assert ml.status()["profile"] == "work"
    assert ml.status()["environment"] == "staging"


def test_an_unchosen_target_is_null_in_the_status_not_empty(spawned, fake_clock, monkeypatch):
    """ "We did not pick" is a different fact from "we picked the empty one"."""
    # The child inherits os.environ, so a developer shell that already exports
    # MODAL_PROFILE would otherwise decide this assertion.
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    ml.start(engine="sync", policy_hub_id="someone/p")
    assert ml.status()["profile"] is None
    assert ml.status()["environment"] is None
    _, env = spawned["popen"][0]
    assert "MODAL_PROFILE" not in env


# --- the log parser ----------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "phase"),
    [
        ("[tailscale] relay 127.0.0.1:7999 -> socks5 -> 100.64.0.1:7880", "tailscale_up"),
        ("[policy] loading 'someone/some-policy' on cuda ...", "loading"),
        ("[policy] warming up model ...", "warmup"),
        ("[policy] connecting to ws://100.64.0.1:7880 as 'policy' in room 'mml-x' ...", "connecting"),
        ("[policy] connected as 'policy'; claiming control in background ...", "connected"),
        ("[policy] claimed control as policy (took over)", "claimed"),
    ],
)
def test_every_marker_maps_to_its_phase_through_decoration(line, phase):
    """Matched ANYWHERE in the line — `drtc_protocol`'s rule — because Modal
    decorates its log lines and a record can arrive without its newline."""
    assert ml.parse_phase(line) == phase
    assert ml.parse_phase(f"2026-09-03T14:02:11Z  app-1 | {line}\r") == phase


def test_an_unrecognised_line_says_nothing():
    """None, not a phase: the caller keeps the phase it had."""
    assert ml.parse_phase("[policy] warmup done in 12.4s") is None
    assert ml.parse_phase("Building image im-1234...") is None
    assert ml.parse_phase("") is None


def test_the_parser_is_a_matcher_not_a_state_machine():
    """A `connected as` arriving with no preceding warmup is still connected —
    ordering is the state machine's business, not the matcher's."""
    assert ml.parse_phase("[policy] connected as 'policy'") == "connected"


def test_readiness_is_connected_and_never_requires_claimed():
    """policy.py claims control in a BACKGROUND task and the claim is
    non-fatal, so a perfectly healthy run may never print `claimed`. Treating
    it as readiness would time out runs that are working."""
    assert ml.READY_PHASE == "connected"
    assert "connected" in ml._READY_PHASES
    assert "claimed" in ml._READY_PHASES


def test_a_ready_line_flips_the_state_without_a_claim(spawned, fake_clock):
    ml.start(engine="sync", policy_hub_id="someone/p")
    ml._handle_line(spawned["proc"], "[policy] connected as 'policy'; claiming control in background ...\n")
    assert ml.status()["state"] == "ready"
    assert ml.status()["phase"] == "connected"


# --- the binary ladder -------------------------------------------------------


def test_the_env_override_wins():
    found = ml.find_modal(
        env={ml.ENV_MODAL_BIN: "/opt/mine/modal"},
        which=lambda name: "/usr/bin/modal",
        is_file=lambda p: p == "/opt/mine/modal",
    )
    assert found == "/opt/mine/modal"


def test_a_typod_override_refuses_rather_than_falling_through():
    """sfu.find_livekit_server's rule: you must not silently run a different
    build than the one you asked for."""
    assert (
        ml.find_modal(
            env={ml.ENV_MODAL_BIN: "/opt/typo/modal"},
            which=lambda name: "/usr/bin/modal",
            is_file=lambda p: False,
        )
        is None
    )


def test_path_comes_before_the_fallback_locations():
    found = ml.find_modal(env={}, which=lambda name: "/usr/bin/modal", is_file=lambda p: True)
    assert found == "/usr/bin/modal"


def test_each_fallback_is_tried_in_order():
    for candidate in ml._FALLBACK_BINS:
        found = ml.find_modal(env={}, which=lambda name: None, is_file=lambda p, c=candidate: p == c)
        assert found == candidate


def test_all_miss_is_none():
    assert ml.find_modal(env={}, which=lambda name: None, is_file=lambda p: False) is None


# --- the state machine -------------------------------------------------------


def test_the_happy_path_walks_idle_starting_ready_stopping_idle(spawned, fake_clock):
    assert ml.status()["state"] == "idle"

    result = ml.start(engine="rtc", policy_hub_id="someone/p", horizon=50, s_min=6)
    assert result["started"] is True
    assert ml.status()["state"] == "starting"
    assert ml.status()["engine"] == "rtc"
    assert ml.status()["policy_hub_id"] == "someone/p"
    assert ml.status()["room"] == "mml-abcdef123456"

    ml._handle_line(spawned["proc"], "[policy] loading 'someone/p' on cuda ...\n")
    assert ml.status()["phase"] == "loading"
    ml._handle_line(spawned["proc"], "[policy] connected as 'policy'\n")
    assert ml.status()["state"] == "ready"

    stopped = ml.stop()
    assert stopped["state"] == "stopping"
    assert spawned["terminated"] == [spawned["proc"]]

    ml._handle_exit(spawned["proc"], -15)
    assert ml.status()["state"] == "idle"
    # The log path survives the idle transition — after a run it is the most
    # useful thing left — and a stop the operator asked for leaves no message
    # behind (a stale "Stopping…" on an idle panel reads as a stuck request).
    assert ml.status()["log_path"] is not None
    assert ml.status()["message"] is None
    assert ml.status()["code"] is None


def test_the_status_dict_always_carries_every_key(spawned, fake_clock):
    """`response_model` with no exclusion mode materializes absent optionals as
    null, so the payload must really always carry them (GpuStatusResponse)."""
    expected = {
        "state",
        "phase",
        "engine",
        "policy_hub_id",
        "room",
        "profile",
        "environment",
        "app_id",
        "task",
        "horizon",
        "fps",
        "video_codec",
        "s_min",
        "model_dtype",
        "gpu",
        "log_path",
        "started_at",
        "elapsed_s",
        "message",
        "hint",
        "code",
        "last_line",
        "idle_stop_in_s",
    }
    assert set(ml.status()) == expected
    ml.start(engine="sync", policy_hub_id="someone/p")
    assert set(ml.status()) == expected
    ml._handle_line(spawned["proc"], "[policy] connected as 'policy'\n")
    assert set(ml.status()) == expected


def test_a_second_start_is_refused(spawned, fake_clock):
    ml.start(engine="sync", policy_hub_id="someone/p")
    with pytest.raises(ApiError) as excinfo:
        ml.start(engine="sync", policy_hub_id="someone/p")
    assert excinfo.value.code == ErrorCode.GPU_ALREADY_RUNNING
    assert excinfo.value.status_code == 409


def test_a_stop_with_nothing_running_is_refused(spawned):
    with pytest.raises(ApiError) as excinfo:
        ml.stop()
    assert excinfo.value.code == ErrorCode.GPU_NOT_RUNNING


def test_a_missing_binary_refuses_before_anything_is_spawned(spawned, monkeypatch):
    monkeypatch.setattr(ml, "find_modal", lambda: None)
    with pytest.raises(ApiError) as excinfo:
        ml.start(engine="sync", policy_hub_id="someone/p")
    assert excinfo.value.code == ErrorCode.GPU_CLI_MISSING
    assert "uv tool install modal" in excinfo.value.detail
    assert spawned["popen"] == []
    assert ml.status()["state"] == "idle"


def test_an_empty_policy_id_refuses_before_anything_is_spawned(spawned):
    """`--policy-path` is required by both entrypoints; letting it through
    means a Click usage error 90s into a log the user is watching for
    cold-start progress."""
    with pytest.raises(ApiError) as excinfo:
        ml.start(engine="sync", policy_hub_id="   ")
    assert excinfo.value.code == ErrorCode.GPU_LAUNCH_FAILED
    assert spawned["popen"] == []


def test_no_tailnet_address_refuses_before_anything_is_spawned(spawned, monkeypatch):
    monkeypatch.setattr(ml, "resolve_transport_plan", lambda: _plan(url=""))
    with pytest.raises(ApiError) as excinfo:
        ml.start(engine="sync", policy_hub_id="someone/p")
    assert excinfo.value.code == ErrorCode.GPU_LAUNCH_FAILED
    assert "tailnet" in excinfo.value.detail
    assert spawned["popen"] == []


def test_the_spawn_gets_the_built_argv_and_the_credentialed_env(spawned, fake_clock):
    ml.start(engine="rtc", policy_hub_id=" someone/p ", task="", horizon=50, s_min=6)
    argv, env = spawned["popen"][0]
    assert argv[2].endswith("modal_policy_rtc.py")
    assert argv[argv.index("--policy-path") + 1] == "someone/p"  # stripped
    assert "--task" not in argv
    assert argv[argv.index("--s-min") + 1] == "6"
    assert SECRET not in " ".join(argv)
    assert env["LIVEKIT_API_SECRET"] == SECRET


# --- the two deadlines -------------------------------------------------------


def test_the_cold_start_bound_fires_at_exactly_its_timeout(spawned, fake_clock):
    ml.start(engine="sync", policy_hub_id="someone/p")
    ml._handle_line(spawned["proc"], "[policy] loading 'someone/p' on cuda ...\n")

    fake_clock.advance(ml._COLD_START_TIMEOUT_S - 1.0)
    assert ml.status()["state"] == "starting"
    assert spawned["terminated"] == []

    fake_clock.advance(1.0)
    status = ml.status()
    assert status["state"] == "stopping"
    # The last phase reached IS the diagnosis: "stuck at loading" and "stuck at
    # tailscale_up" have nothing in common as remedies.
    assert "`loading`" in status["message"]
    assert spawned["terminated"] == [spawned["proc"]]

    # And it lands in FAILED, not idle: we implement the overrun by killing the
    # group, but it is a failure and must keep its diagnosis on screen.
    ml._handle_exit(spawned["proc"], -15)
    status = ml.status()
    assert status["state"] == "failed"
    assert "`loading`" in status["message"]
    assert status["phase"] == "loading"
    assert status["code"] == "gpu.launch_failed"


def test_a_ready_gpu_is_stopped_after_the_idle_window(spawned, fake_clock):
    ml.start(engine="sync", policy_hub_id="someone/p")
    ml._handle_line(spawned["proc"], "[policy] connected as 'policy'\n")

    fake_clock.advance(ml._GPU_IDLE_STOP_S - 1.0)
    status = ml.status()
    assert status["state"] == "ready"
    assert status["idle_stop_in_s"] == pytest.approx(1.0)

    fake_clock.advance(1.0)
    status = ml.status()
    assert status["state"] == "stopping"
    assert "billing" in status["message"]
    assert spawned["terminated"] == [spawned["proc"]]

    # An automatic stop is still a clean stop — idle, with the reason kept so
    # the panel can say why the GPU is no longer there.
    ml._handle_exit(spawned["proc"], -15)
    assert ml.status()["state"] == "idle"
    assert "billing" in ml.status()["message"]
    # An automatic stop is not a failure, so it carries no code.
    assert ml.status()["code"] is None


def test_the_idle_stop_never_fires_while_a_session_is_running(spawned, fake_clock, monkeypatch):
    """A GPU driving an arm is the opposite of idle. Injected as a callable, so
    this needs no session — and it is the only thing the launcher asks about
    one."""
    ml.start(engine="sync", policy_hub_id="someone/p")
    ml._handle_line(spawned["proc"], "[policy] connected as 'policy'\n")

    monkeypatch.setattr(ml, "_remote_inference_is_active", lambda: True)
    fake_clock.advance(ml._GPU_IDLE_STOP_S * 10)
    status = ml.status()
    assert status["state"] == "ready"
    # Not a paused countdown — there is no countdown at all while a run holds
    # the arm, and the panel must not render one.
    assert status["idle_stop_in_s"] is None
    assert spawned["terminated"] == []


def test_the_idle_window_restarts_when_a_session_ends(spawned, fake_clock, monkeypatch):
    """Measured from whichever is LATER: reaching ready, or the last session's
    end."""
    ml.start(engine="sync", policy_hub_id="someone/p")
    ml._handle_line(spawned["proc"], "[policy] connected as 'policy'\n")

    active = {"value": True}
    monkeypatch.setattr(ml, "_remote_inference_is_active", lambda: active["value"])
    fake_clock.advance(ml._GPU_IDLE_STOP_S * 2)
    ml.status()

    active["value"] = False
    ml.status()  # the session just ended: the window starts here
    fake_clock.advance(ml._GPU_IDLE_STOP_S - 1.0)
    assert ml.status()["state"] == "ready"
    fake_clock.advance(1.0)
    assert ml.status()["state"] == "stopping"


def test_the_pump_also_notices_a_deadline(spawned, fake_clock):
    """The two things that already wake are the log pump and a status poll —
    deliberately not a thread of its own."""
    ml.start(engine="sync", policy_hub_id="someone/p")
    fake_clock.advance(ml._COLD_START_TIMEOUT_S)
    ml._handle_line(spawned["proc"], "Building image im-1234...\n")
    assert ml._state == "stopping"
    assert spawned["terminated"] == [spawned["proc"]]


# --- failure classification --------------------------------------------------


def test_a_nonzero_exit_before_the_room_lands_in_failed(spawned, fake_clock):
    ml.start(engine="sync", policy_hub_id="someone/p")
    ml._handle_line(spawned["proc"], "Traceback (most recent call last):\n")
    ml._handle_exit(spawned["proc"], 1)

    status = ml.status()
    assert status["state"] == "failed"
    assert "exit code 1" in status["message"]
    assert status["hint"]
    assert status["code"] == "gpu.launch_failed"


def test_an_exit_after_the_room_says_so_plainly(spawned, fake_clock):
    """It got there and then went away — a real diagnosis, and not one the
    text classifiers can improve on."""
    ml.start(engine="sync", policy_hub_id="someone/p")
    ml._handle_line(spawned["proc"], "[policy] connected as 'policy'\n")
    ml._handle_exit(spawned["proc"], 137)

    status = ml.status()
    assert status["state"] == "failed"
    assert "exited" in status["message"]
    assert status["code"] == "gpu.launch_failed"


def test_the_auth_code_reaches_the_wire(spawned, fake_clock):
    """Each classifier's code, as an SDK would read it. `gpu.unauthenticated`
    and a tailnet auth-key expiry have nothing in common as remedies, and the
    prose beside them is free to improve — the code is the contract."""
    ml.start(engine="sync", policy_hub_id="someone/p")
    ml._handle_line(spawned["proc"], "Error: could not authenticate; run `modal token new`\n")
    ml._handle_exit(spawned["proc"], 1)
    assert ml.status()["code"] == "gpu.unauthenticated"


def test_the_tailscale_code_reaches_the_wire(spawned, fake_clock):
    ml.start(engine="sync", policy_hub_id="someone/p")
    ml._handle_line(spawned["proc"], "backend error: invalid key: authkey expired\n")
    ml._handle_exit(spawned["proc"], 1)
    assert ml.status()["code"] == "gpu.launch_failed"
    assert "tailscale-auth" in ml.status()["hint"]


def test_missing_modal_auth_is_classified():
    code, _message, hint = ml.classify_failure(
        "Error: Token missing. Could not authenticate client.\nRun `modal token new` to fix.", 1
    )
    assert code == ErrorCode.GPU_UNAUTHENTICATED
    assert "modal token new" in hint
    assert "~/.modal.toml" in hint  # the Lab never reads it — say so


def test_an_expired_tailscale_auth_key_is_classified():
    """This same failure currently reaches the operator as
    `transport.no_policy`, whose hint lists three possible causes. The launcher
    can now say which one."""
    code, _message, hint = ml.classify_failure(
        "[tailscale] tailscale up --hostname=drtc --auth-key=<redacted>\n"
        "backend error: invalid key: authkey expired",
        1,
    )
    assert code == ErrorCode.GPU_LAUNCH_FAILED
    assert "tailscale-auth" in hint
    assert "--force" in hint
    assert "EPHEMERAL" in hint


def test_an_unrecognised_failure_still_names_the_exit_code_and_the_log():
    code, message, hint = ml.classify_failure("something went wrong in the container", 42)
    assert code == ErrorCode.GPU_LAUNCH_FAILED
    assert "exit code 42" in message
    assert "log" in hint


def test_the_cold_start_overrun_names_the_phase_it_reached():
    code, message, _hint = ml.classify_failure("", None, phase="loading", timed_out=True)
    assert code == ErrorCode.GPU_LAUNCH_FAILED
    assert "`loading`" in message
    assert "300s" in message

    _, no_phase, _ = ml.classify_failure("", None, phase=None, timed_out=True)
    assert "never reported a phase" in no_phase


# --- the shared resolver -----------------------------------------------------


def test_the_plan_comes_from_the_session_s_own_resolver(monkeypatch):
    """ONE credential path. The session's preflight, the transport endpoint and
    this launcher all call `remote_inference.resolve_transport()`; the two
    halves meeting in different rooms is invisible by construction, so a second
    path is not a duplication smell, it is the bug."""
    monkeypatch.setattr(
        ri,
        "resolve_transport",
        lambda: ri.ResolvedTransport(
            url="ws://127.0.0.1:7880",
            room="mml-deadbeef0000",
            api_key="APIkeyname",
            api_secret=SECRET,
            child_token="jwt",
            source="sfu",
            missing=(),
        ),
    )
    monkeypatch.setattr(ri, "sfu_modal_url", lambda: "ws://100.64.0.1:7880")

    plan = ml.resolve_transport_plan()
    # The url a CONTAINER dials, never the loopback one a local child dials.
    assert plan.url == "ws://100.64.0.1:7880"
    assert plan.needs_tailscale is True
    assert plan.room == "mml-deadbeef0000"
    assert plan.api_secret == SECRET


def test_the_cloud_path_needs_no_tailnet(monkeypatch):
    monkeypatch.setattr(
        ri,
        "resolve_transport",
        lambda: ri.ResolvedTransport(
            url="wss://x.livekit.cloud",
            room="portal-lerobot-inference",
            api_key="APIkeyname",
            api_secret=SECRET,
            child_token="",
            source="cloud",
            missing=(),
        ),
    )
    plan = ml.resolve_transport_plan()
    assert plan.url == "wss://x.livekit.cloud"
    assert plan.needs_tailscale is False


# --- the stop's own bound ----------------------------------------------------


def _arm_drain(spawned, fake_clock):
    """Reach `stopping` with the kill already returned and the pump wedged.

    The real `_terminate_and_watch` arms the deadline once `_terminate_tree`
    comes back; the fixture replaces `_terminate_async`, so the test arms it
    the same way the thread would.
    """
    ml.start(engine="sync", policy_hub_id="someone/p")
    ml._handle_line(spawned["proc"], "[policy] connected as 'policy'\n")
    ml.stop()
    assert ml.status()["state"] == "stopping"
    # The kill has returned; the pump has NOT reached EOF and never will.
    ml._drain_deadline = fake_clock() + ml._STOP_DRAIN_TIMEOUT_S


def test_a_stop_whose_log_stream_never_closes_is_still_bounded(spawned, fake_clock):
    """`stopping` must not be able to last forever.

    `_terminate_tree` already escalated to SIGKILL, so the PROCESS is gone;
    what can still hang is the stdout pipe, whose write end an un-reaped
    grandchild may hold open — leaving `readline` blocked and the pump's
    finalizer unreachable."""
    _arm_drain(spawned, fake_clock)

    fake_clock.advance(ml._STOP_DRAIN_TIMEOUT_S - 1.0)
    assert ml.status()["state"] == "stopping"

    fake_clock.advance(1.0)
    status = ml.status()
    assert status["state"] == "idle"
    # It says the KILL worked and the LISTENING stopped — "it is still running
    # somewhere" would be the wrong thing to make an operator believe.
    assert "log stream never closed" in status["message"]
    assert status["code"] is None


def test_a_forced_stop_keeps_a_failure_s_diagnosis(spawned, fake_clock):
    """A cold-start overrun that then wedges is still a FAILURE, and still
    names the phase it died at."""
    ml.start(engine="sync", policy_hub_id="someone/p")
    ml._handle_line(spawned["proc"], "[policy] loading 'someone/p' on cuda ...\n")
    fake_clock.advance(ml._COLD_START_TIMEOUT_S)
    ml.status()  # the overrun fires and asks for the kill
    ml._drain_deadline = fake_clock() + ml._STOP_DRAIN_TIMEOUT_S

    fake_clock.advance(ml._STOP_DRAIN_TIMEOUT_S)
    status = ml.status()
    assert status["state"] == "failed"
    assert "`loading`" in status["message"]  # the diagnosis survives
    assert "log stream never closed" in status["message"]
    assert status["phase"] == "loading"
    assert status["code"] == "gpu.launch_failed"


def test_a_wedged_pump_cannot_write_into_the_next_launch(spawned, fake_clock):
    """The orphan guard. Forcing the terminal state clears `_proc`, so the
    zombie thread's lines — and its eventual verdict — are dropped instead of
    landing on whatever runs next."""
    _arm_drain(spawned, fake_clock)
    stale = spawned["proc"]
    fake_clock.advance(ml._STOP_DRAIN_TIMEOUT_S)
    assert ml.status()["state"] == "idle"

    # A second launch claims the slot with its own process.
    fresh = FakePopen()
    spawned["popen_result"] = fresh
    ml.start(engine="rtc", policy_hub_id="someone/other")
    ml._handle_line(fresh, "[policy] loading 'someone/other' on cuda ...\n")
    assert ml.status()["phase"] == "loading"

    # The old pump finally wakes up. Both of its calls must be no-ops.
    ml._handle_line(stale, "[policy] connected as 'policy'\n")
    ml._handle_exit(stale, 1)
    status = ml.status()
    assert status["state"] == "starting"
    assert status["phase"] == "loading"
    assert status["policy_hub_id"] == "someone/other"


def test_the_drain_is_armed_only_after_the_kill_returns(monkeypatch, spawned, fake_clock):
    """The bound is on the DRAIN, not on the kill — `_terminate_tree` has a
    ceiling of its own and arming at the stop request would count it twice."""
    monkeypatch.setattr(ml, "_terminate_tree", lambda proc, timeout=None: None)
    ml.start(engine="sync", policy_hub_id="someone/p")
    ml._handle_line(spawned["proc"], "[policy] connected as 'policy'\n")
    ml.stop()
    assert ml._drain_deadline is None  # the stop itself arms nothing

    ml._terminate_and_watch(spawned["proc"])
    assert ml._drain_deadline == fake_clock() + ml._STOP_DRAIN_TIMEOUT_S


# --- S3.8c: the app outlives its client --------------------------------------
# The premise the whole section rests on, verified in the installed client's
# source (`modal/runner.py::_run_app`): the app-stop RPC is sent from `except
# KeyboardInterrupt` and from the exception paths, and there is NO SIGTERM
# handler. A client killed the way S3.8 killed it therefore leaves the app —
# and the A100 under it — running until Modal's heartbeat timeout notices.


class SignalledPopen:
    """A fake `modal run` client that records the signals it was sent.

    `dies_on` names the first signal it obeys ("SIGINT", "SIGTERM", or None for
    a process that ignores everything); `wait` behaves accordingly, so a test
    can assert the ORDER of the escalation without a real process, a real
    signal, or a sleep.
    """

    def __init__(self, dies_on: str | None = "SIGINT") -> None:
        self.dies_on = dies_on
        self.signals: list[str] = []
        self.returncode: int | None = None
        self.stdout = None
        self.pid = 4242

    def receive(self, name: str) -> None:
        self.signals.append(name)
        if name == self.dies_on:
            self.returncode = -2 if name == "SIGINT" else -15

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("modal", timeout or 0)
        return self.returncode


@pytest.fixture
def signalled(monkeypatch):
    """A client whose group signals are recorded, with the escalation stubbed.

    `_signal_group` (rollout's, imported by the launcher) and `_terminate_tree`
    are both replaced, so nothing here can signal a real process — the test
    asserts on what WOULD have been sent, in order.
    """
    proc = SignalledPopen()

    def _sig(p, signum):
        p.receive(signal.Signals(signum).name)
        return True

    def _tree(p, timeout=None):
        p.receive("SIGTERM")
        if p.returncode is None:
            p.receive("SIGKILL")
            p.returncode = -9

    monkeypatch.setattr(ml, "_signal_group", _sig)
    monkeypatch.setattr(ml, "_terminate_tree", _tree)
    return proc


def test_the_stop_sends_sigint_first_and_waits_for_the_client_to_disconnect(signalled, monkeypatch):
    """SIGINT is the ONLY signal that makes the client stop its Modal app, so
    it goes first and the client is given a bounded moment to act on it."""
    stops: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ml, "stop_app", lambda app_id, profile="": (stops.append((app_id, profile)), (True, "s"))[1]
    )

    assert ml._graceful_terminate(signalled) is True
    # SIGINT alone: no escalation, because the client did exit.
    assert signalled.signals == ["SIGINT"]
    # And no API call — the client's own disconnect is the authority.
    assert stops == []


def test_a_client_that_ignores_sigint_is_escalated_and_its_app_stopped_over_the_api(monkeypatch):
    proc = SignalledPopen(dies_on="SIGTERM")
    order: list[str] = []

    def _sig(p, signum):
        order.append(signal.Signals(signum).name)
        p.receive(signal.Signals(signum).name)
        return True

    def _tree(p, timeout=None):
        order.append("terminate_tree")
        p.receive("SIGTERM")

    monkeypatch.setattr(ml, "_signal_group", _sig)
    monkeypatch.setattr(ml, "_terminate_tree", _tree)

    assert ml._graceful_terminate(proc) is False
    # The existing SIGTERM->SIGKILL escalation is still there, and still comes
    # from rollout's `_terminate_tree` — this adds a step in FRONT of it.
    assert order == ["SIGINT", "terminate_tree"]


def test_an_already_dead_client_is_never_signalled_but_its_app_is_still_stopped(monkeypatch):
    """We cannot know whether a client that died on its own made the call, and
    `modal app stop` on an already-stopped app is free."""
    proc = SignalledPopen()
    proc.returncode = 1
    monkeypatch.setattr(ml, "_signal_group", lambda p, s: pytest.fail("nothing to signal"))
    monkeypatch.setattr(ml, "_terminate_tree", lambda p, timeout=None: pytest.fail("nothing to kill"))

    assert ml._graceful_terminate(proc) is False


def test_the_settle_stops_the_app_only_when_the_client_did_not(monkeypatch):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ml, "stop_app", lambda app_id, profile="": (calls.append((app_id, profile)), (True, "stopped"))[1]
    )

    ml._settle_app("ap-1", "makermods", client_exited_cleanly=True)
    assert calls == []

    ml._settle_app("ap-1", "makermods", client_exited_cleanly=False)
    assert calls == [("ap-1", "makermods")]


def test_a_stop_whose_client_was_killed_stops_the_app_with_the_launch_s_profile(
    spawned, fake_clock, monkeypatch
):
    """End to end through the real `_terminate_and_watch`: the profile that
    billed the launch is the profile the stop is made against, because an app
    id belongs to ONE workspace."""
    monkeypatch.setattr(
        ml,
        "list_targets",
        lambda profile="": {
            "profiles": [{"name": "makermods", "workspace": "w", "active": True}],
            "environments": [],
            "profile": "makermods",
            "error": None,
        },
    )
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ml, "stop_app", lambda app_id, profile="": (calls.append((app_id, profile)), (True, "stopped"))[1]
    )
    monkeypatch.setattr(ml, "_graceful_terminate", lambda proc: False)

    ml.start(engine="sync", policy_hub_id="someone/p", profile="makermods")
    ml._handle_line(spawned["proc"], "https://modal.com/apps/makermods/main/ap-QfTK2AxcfbJnnY1kLS7Y22\n")
    ml.stop()
    ml._terminate_and_watch(spawned["proc"])

    assert calls == [("ap-QfTK2AxcfbJnnY1kLS7Y22", "makermods")]
    # And the record is gone, so nothing reaps an app that is already stopped.
    assert ml.read_app_record() is None


# --- the app id --------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        # The shape the client ACTUALLY prints: Rich wraps the sentence, so
        # "View run at" is on the line before and the url is alone on its own.
        "https://modal.com/apps/makermods/main/ap-QfTK2AxcfbJnnY1kLS7Y22\n",
        "✓ Initialized. View run at https://modal.com/apps/ws/env/ap-QfTK2AxcfbJnnY1kLS7Y22",
        "  │ https://modal.com/apps/ws/env/ap-QfTK2AxcfbJnnY1kLS7Y22 │",
    ],
)
def test_the_app_id_is_read_off_the_run_url_however_it_is_wrapped(line):
    assert ml.parse_app_id(line) == "ap-QfTK2AxcfbJnnY1kLS7Y22"


@pytest.mark.parametrize(
    "line",
    [
        "[policy] connected as 'policy'",
        "Traceback: ap-something is not an app id here",
        "",
    ],
)
def test_a_line_without_a_run_url_names_no_app(line):
    assert ml.parse_app_id(line) is None


def test_the_app_id_reaches_the_status_and_the_record(spawned, fake_clock):
    ml.start(engine="sync", policy_hub_id="someone/p")
    assert ml.status()["app_id"] is None

    ml._handle_line(spawned["proc"], "https://modal.com/apps/makermods/main/ap-y9x9NllbdwdEqmvNqA8by8\n")
    assert ml.status()["app_id"] == "ap-y9x9NllbdwdEqmvNqA8by8"

    record = ml.read_app_record()
    assert record["app_id"] == "ap-y9x9NllbdwdEqmvNqA8by8"
    assert record["started_at"] is not None


def test_the_record_is_only_cleared_by_the_app_it_names(spawned, fake_clock):
    """The race with a relaunch: a stop settling late must not delete the id of
    the app that started after it."""
    ml._write_app_record("ap-new", "makermods", 1.0)
    ml._clear_app_record("ap-old")
    assert ml.read_app_record()["app_id"] == "ap-new"

    ml._clear_app_record("ap-new")
    assert ml.read_app_record() is None


def test_a_record_this_build_cannot_read_is_simply_nothing_to_reap():
    ml._APP_RECORD_FILE.write_text("{not json")
    assert ml.read_app_record() is None
    ml._APP_RECORD_FILE.write_text('{"profile": "makermods"}')
    assert ml.read_app_record() is None


# --- the orphan reaper -------------------------------------------------------


def test_the_reaper_stops_an_app_the_last_process_left_behind(monkeypatch):
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        ml, "stop_app", lambda app_id, profile="": (calls.append((app_id, profile)), (True, "stopped"))[1]
    )
    ml._write_app_record("ap-orphan", "makermods", 1_788_486_546.0)

    ml._reap_orphan_app()

    assert calls == [("ap-orphan", "makermods")]
    # The operator learns it happened, on the idle panel that would otherwise
    # say nothing at all.
    assert "ap-orphan" in ml.status()["message"]
    assert ml.read_app_record() is None


def test_the_reaper_does_nothing_without_a_record(monkeypatch):
    monkeypatch.setattr(ml, "stop_app", lambda *a, **k: pytest.fail("nothing to stop"))
    ml._reap_orphan_app()
    assert ml.status()["message"] is None


def test_the_reaper_never_touches_a_live_launch_s_app(spawned, fake_clock, monkeypatch):
    """The record belongs to the launch that is RUNNING here; stopping it would
    be the reaper killing a healthy GPU."""
    monkeypatch.setattr(ml, "stop_app", lambda *a, **k: pytest.fail("that app is ours and alive"))
    ml.start(engine="sync", policy_hub_id="someone/p")
    ml._handle_line(spawned["proc"], "https://modal.com/apps/makermods/main/ap-live\n")

    ml._reap_orphan_app()


def test_a_reap_that_fails_says_so_and_still_leaves_the_lab_launchable(spawned, fake_clock, monkeypatch):
    monkeypatch.setattr(ml, "stop_app", lambda app_id, profile="": (False, "connection refused"))
    ml._write_app_record("ap-stuck", "", 1_788_486_546.0)

    ml._reap_orphan_app()

    message = ml.status()["message"]
    assert "ap-stuck" in message and "connection refused" in message
    # The record survives a failed stop — the app may still be up, and the next
    # boot should try again.
    assert ml.read_app_record()["app_id"] == "ap-stuck"
    # And nothing about it blocks a launch.
    assert ml.start(engine="sync", policy_hub_id="someone/p")["started"] is True


def test_the_reap_is_kicked_once_per_process(monkeypatch):
    """Once per PROCESS: a poll that re-ran it would be a Modal API call per
    poll, and a second reap has nothing left to find anyway."""
    started: list[str] = []

    class _Thread:
        def __init__(self, **kwargs):
            self._name = kwargs["name"]

        def start(self):
            started.append(self._name)

    # The module's own `threading` reference, not the real module.
    monkeypatch.setattr(ml, "threading", types.SimpleNamespace(Thread=_Thread, Lock=ml.threading.Lock))

    ml.reap_orphan_app_async()
    ml.reap_orphan_app_async()

    assert started == ["modal-launcher-reap"]


# --- `modal app stop` --------------------------------------------------------


def test_the_app_stop_is_non_interactive_and_carries_the_profile(monkeypatch):
    """Without `--yes` the CLI prompts, and a server has no terminal — it would
    abort with "no interactive terminal detected"."""
    seen: dict[str, object] = {}

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def _run(argv, **kwargs):
        seen["argv"] = argv
        seen["env"] = kwargs["env"]
        seen["timeout"] = kwargs["timeout"]
        return _Done()

    monkeypatch.setattr(ml, "find_modal", lambda: "/usr/bin/modal")
    monkeypatch.setattr(ml.subprocess, "run", _run)

    assert ml.stop_app("ap-1", "makermods") == (True, "stopped")
    assert seen["argv"] == ["/usr/bin/modal", "app", "stop", "--yes", "ap-1"]
    assert seen["env"]["MODAL_PROFILE"] == "makermods"
    assert seen["timeout"] == ml._APP_STOP_TIMEOUT_S


def test_an_app_stop_failure_quotes_the_cli_and_never_raises(monkeypatch):
    class _Failed:
        returncode = 1
        stdout = ""
        stderr = "Error: App ap-1 not found in environment main\n"

    monkeypatch.setattr(ml, "find_modal", lambda: "/usr/bin/modal")
    monkeypatch.setattr(ml.subprocess, "run", lambda argv, **kw: _Failed())

    stopped, detail = ml.stop_app("ap-1")
    assert stopped is False
    assert "not found" in detail


def test_no_cli_means_no_app_stop_and_a_reason(monkeypatch):
    monkeypatch.setattr(ml, "find_modal", lambda: None)
    stopped, detail = ml.stop_app("ap-1")
    assert stopped is False
    assert "PATH" in detail


# --- the shutdown stop -------------------------------------------------------


def test_the_shutdown_stop_is_synchronous_and_reports_whether_it_did_anything(
    spawned, fake_clock, monkeypatch
):
    """Fire-and-forget is exactly wrong on the way out: a kill that outlives
    the process it runs in never happens, which is how a --reload save left an
    A100 running."""
    assert ml.stop_for_shutdown() is False  # nothing running

    watched: list[object] = []
    monkeypatch.setattr(ml, "_terminate_and_watch", lambda proc: watched.append(proc))
    ml.start(engine="sync", policy_hub_id="someone/p")

    assert ml.stop_for_shutdown() is True
    assert watched == [spawned["proc"]]  # ON THIS THREAD, not a daemon one
    assert ml.status()["state"] == "stopping"


# --- the status echo ---------------------------------------------------------


def test_the_status_echoes_the_transport_tuple_it_launched_with(spawned, fake_clock):
    """So the panel's drift warning can compare the form against the SERVER's
    record — which survives a page reload and describes a GPU another tab
    started, neither of which a tab's own memory can do."""
    idle = ml.status()
    assert idle["horizon"] is None and idle["fps"] is None and idle["s_min"] is None
    assert idle["task"] is None and idle["video_codec"] is None

    ml.start(
        engine="rtc",
        policy_hub_id="someone/p",
        task="Put the lego brick in the box",
        horizon=50,
        fps=20,
        video_codec="MJPEG",
        s_min=6,
    )
    status = ml.status()
    assert status["task"] == "Put the lego brick in the box"
    assert status["horizon"] == 50
    assert status["fps"] == 20
    assert status["video_codec"] == "MJPEG"
    assert status["s_min"] == 6

    ml.stop()
    ml._handle_exit(spawned["proc"], -2)
    assert ml.status()["horizon"] is None


def test_a_client_that_exited_on_its_own_leaves_no_record_to_reap(spawned, fake_clock):
    """A normal exit (rc >= 0) means the client ran its OWN disconnect, so its
    app is going down. Keeping the record would make the next boot announce an
    orphan it never had."""
    ml.start(engine="sync", policy_hub_id="someone/p")
    ml._handle_line(spawned["proc"], "https://modal.com/apps/makermods/main/ap-selfexit\n")
    assert ml.read_app_record() is not None

    ml._handle_exit(spawned["proc"], 1)  # an uncaught exception: modal stopped its app
    assert ml.read_app_record() is None


def test_a_client_someone_else_killed_keeps_its_record_for_the_next_boot(spawned, fake_clock):
    """rc < 0 is a SIGNAL death this launcher did not ask for (a `kill -9`), so
    no disconnect was sent and the app may well still be up."""
    ml.start(engine="sync", policy_hub_id="someone/p")
    ml._handle_line(spawned["proc"], "https://modal.com/apps/makermods/main/ap-killed\n")

    ml._handle_exit(spawned["proc"], -9)
    assert ml.read_app_record()["app_id"] == "ap-killed"


# --- classify_failure: the container's own exit message, and the tailnet
# false positive it used to hide ---------------------------------------------

_JOINED_THEN_TASK_REFUSED = """\
[tailscale] tailscale up --hostname=modal-policy --auth-key=<redacted>
2026/09/04 02:53:11 Switching ipn state NoState -> NeedsLogin (WantRunning=true, nm=false)
[tailscale] joined tailnet as modal-policy (100.118.36.94)
[tailscale] relay 127.0.0.1:7880 -> socks5 -> 100.80.250.40:7880
[policy] HF_TOKEN present; gated base models will authenticate.
Traceback (most recent call last):
  File "/root/makermodslab/drtc/policy_rtc.py", line 253, in load_policy
    raise SystemExit(
SystemExit: --task is required for a 'molmoact2' policy. It is language-conditioned.
Stopping app - uncaught exception raised in remote container: SystemExit("--task is required for a 'molmoact2' policy. It is language-conditioned.")
"""


def test_classify_failure_surfaces_the_containers_own_exit_message() -> None:
    """The 2026-09-03 case: the join succeeded, the policy server refused an
    empty --task, and the classifier blamed the auth key because the join
    block was still inside the tail."""
    code, message, hint = ml.classify_failure(_JOINED_THEN_TASK_REFUSED, 1)
    assert code == ErrorCode.GPU_LAUNCH_FAILED
    assert "policy server stopped itself" in message
    assert "--task is required for a 'molmoact2' policy" in message
    assert "SystemExit(" not in message  # the repr wrapper is unwrapped
    assert "tailnet" not in message
    assert hint is not None and "log" in hint


def test_classify_failure_does_not_blame_the_tailnet_once_it_joined() -> None:
    tail = """\
[tailscale] tailscale up --hostname=modal-policy --auth-key=<redacted>
[tailscale] joined tailnet as modal-policy (100.118.36.94)
[policy] loading 'x/y' on cuda ...
some other failure with no recognised marker
"""
    _code, message, _hint = ml.classify_failure(tail, 1)
    assert "tailnet" not in message


def test_classify_failure_still_blames_the_tailnet_when_the_join_never_happened() -> None:
    tail = """\
[tailscale] tailscale up --hostname=modal-policy --auth-key=<redacted>
2026/09/04 02:53:11 Switching ipn state NoState -> NeedsLogin (WantRunning=true, nm=false)
control: RegisterReq: got response; nodeKeyExpired=true, machineAuthorized=false
"""
    _code, message, hint = ml.classify_failure(tail, 1)
    assert "couldn't join the tailnet" in message
    assert hint is not None and "TS_AUTHKEY" in hint
