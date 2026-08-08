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
"""Tests for makermodslab.runners.hf_cloud — covers the host-side wandb credential
resolution, the pinned-lerobot spec derivation, and the cloud-boundary config
localization, plus the terminal-stage bookkeeping in HfCloudJobRunner.stop()
(which decides whether a stopped run reads as `interrupted` or as a failure,
and needs only a two-method fake). Submission, log tailing and status polling
talk to HF Jobs and are intentionally left for integration tests.

The credential tests never read the real ~/.netrc: `netrc.netrc` is
monkeypatched to a fake in every case, so the developer's own W&B key is
neither read nor able to make a test pass."""

from __future__ import annotations

import netrc
import re
import tomllib
from pathlib import Path

import pytest


def test_resolve_wandb_api_key_prefers_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.setenv("WANDB_API_KEY", "env-key-123")
    assert resolve_wandb_api_key() == "env-key-123"


def test_resolve_wandb_api_key_falls_back_to_netrc(monkeypatch: pytest.MonkeyPatch) -> None:
    """When WANDB_API_KEY is unset, the function must read the same place
    `wandb login` writes — ~/.netrc under machine api.wandb.ai."""
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    class _FakeNetrc:
        def authenticators(self, host):
            assert host == "api.wandb.ai"
            return ("login", "account", "netrc-key-456")

    monkeypatch.setattr(netrc, "netrc", lambda: _FakeNetrc())
    assert resolve_wandb_api_key() == "netrc-key-456"


def test_resolve_wandb_api_key_returns_none_when_netrc_has_no_wandb_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    class _FakeNetrc:
        def authenticators(self, host):
            return None

    monkeypatch.setattr(netrc, "netrc", lambda: _FakeNetrc())
    assert resolve_wandb_api_key() is None


def test_resolve_wandb_api_key_returns_none_when_netrc_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env var, no ~/.netrc — neither source has it, caller decides."""
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    def _raise_missing():
        raise FileNotFoundError("~/.netrc")

    monkeypatch.setattr(netrc, "netrc", _raise_missing)
    assert resolve_wandb_api_key() is None


def test_resolve_wandb_api_key_returns_none_when_netrc_parse_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    def _raise_parse():
        raise netrc.NetrcParseError("bad netrc", "~/.netrc", 1)

    monkeypatch.setattr(netrc, "netrc", _raise_parse)
    assert resolve_wandb_api_key() is None


def test_resolve_wandb_api_key_returns_none_when_password_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty password from netrc is treated as missing — the helper
    contract is 'returns the usable key or None', not 'returns whatever
    netrc happened to have'."""
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    class _FakeNetrc:
        def authenticators(self, host):
            return ("login", "account", "")

    monkeypatch.setattr(netrc, "netrc", lambda: _FakeNetrc())
    assert resolve_wandb_api_key() is None


def test_wandb_credentials_endpoint_reports_only_a_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe behind GET /system/wandb-credentials must never leak the key
    itself — the UI only needs to know whether a launch would be refused."""
    from makermodslab.runners import hf_cloud

    monkeypatch.setattr(hf_cloud, "resolve_wandb_api_key", lambda: "super-secret-key")
    payload = hf_cloud.handle_get_wandb_credentials()
    assert payload["available"] is True
    assert "super-secret-key" not in repr(payload)

    monkeypatch.setattr(hf_cloud, "resolve_wandb_api_key", lambda: None)
    assert hf_cloud.handle_get_wandb_credentials()["available"] is False


# -- pinned-lerobot spec derivation (version-skew fix) --


def _pyproject_lerobot_pin() -> str:
    """The raw lerobot dependency line from this repo's pyproject.toml."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    return next(d for d in deps if d.startswith("lerobot"))


def _spec_extras(spec: str) -> set[str]:
    m = re.match(r"lerobot\[(?P<extras>[^\]]*)\]", spec)
    assert m, f"no extras block in {spec!r}"
    return set(m.group("extras").split(","))


def test_cloud_lerobot_spec_carries_the_pyproject_pinned_ref() -> None:
    """The container install spec must reference the exact ref pinned in
    pyproject.toml — never a hardcoded second copy, never :latest."""
    from makermodslab.runners.hf_cloud import cloud_lerobot_spec

    pin = _pyproject_lerobot_pin()
    ref = pin.rsplit("@", 1)[1]  # the sha at the end of git+https://…@<sha>
    spec = cloud_lerobot_spec("act")
    assert ref in spec
    assert "latest" not in spec


def test_cloud_lerobot_spec_uses_archive_tarball_not_git() -> None:
    """A GitHub git+ pin is rewritten to the source archive tarball so pip in
    the container can install it without a git binary."""
    from makermodslab.runners.hf_cloud import cloud_lerobot_spec

    spec = cloud_lerobot_spec("act")
    assert "git+" not in spec
    # 0.6.0 pins by tag (v0.6.0), not a hex SHA, so the archive ref may contain
    # non-hex chars (v, dots). Match any non-slash ref, not just [0-9a-f].
    assert re.search(r"@ https://github\.com/.+/archive/[^/]+\.tar\.gz$", spec)


def test_cloud_lerobot_spec_drops_host_only_extras_and_adds_policy_extra() -> None:
    from makermodslab.runners.hf_cloud import cloud_lerobot_spec

    act = _spec_extras(cloud_lerobot_spec("act"))
    assert "feetech" not in act  # serial motor bus: host-only
    assert "training" in act
    assert "core_scripts" in act  # provides lerobot_train

    smolvla = _spec_extras(cloud_lerobot_spec("smolvla"))
    assert smolvla == act | {"smolvla"}

    pi0_fast = _spec_extras(cloud_lerobot_spec("pi0_fast"))
    assert pi0_fast == act | {"pi"}


def test_cloud_lerobot_spec_falls_back_to_pyproject_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running from a source tree without installed MakerMods Lab metadata must still
    derive the pin — from pyproject.toml directly."""
    from makermodslab.runners import hf_cloud

    monkeypatch.setattr(hf_cloud, "requires", lambda name: None)
    pin = _pyproject_lerobot_pin()
    ref = pin.rsplit("@", 1)[1]
    assert ref in hf_cloud.cloud_lerobot_spec("act")


# -- cloud-boundary config localization (device-leak fix) --


def _request(**overrides):
    from makermodslab.train import TrainingRequest

    return TrainingRequest(dataset_repo_id="user/ds", **overrides)


def test_localize_forces_flavor_device_over_host_detection() -> None:
    """The host's auto-detected device (mps on a Mac) must never reach a cloud
    job: GPU flavors force cuda, cpu tiers force cpu."""
    from makermodslab.runners.hf_cloud import localize_config_for_cloud
    from makermodslab.train import build_training_command

    for host_device in ("auto", "mps", "cpu", None):
        config = _request(policy_device=host_device)
        localize_config_for_cloud(config, "t4-small")
        assert config.policy_device == "cuda"
        cmd = build_training_command(config, "/tmp/out")
        assert cmd[cmd.index("--policy.device") + 1] == "cuda"

    cpu_config = _request(policy_device="mps")
    localize_config_for_cloud(cpu_config, "cpu-upgrade")
    assert cpu_config.policy_device == "cpu"


def test_localize_clears_host_local_dataset_root() -> None:
    from makermodslab.runners.hf_cloud import localize_config_for_cloud
    from makermodslab.train import build_training_command

    config = _request(dataset_root="/Users/someone/.cache/huggingface/lerobot/user/ds")
    localize_config_for_cloud(config, "t4-small")
    assert config.dataset_root is None
    assert "--dataset.root" not in build_training_command(config, "/tmp/out")


def test_localize_rejects_resume_from_host_checkpoint() -> None:
    from makermodslab.runners.hf_cloud import localize_config_for_cloud

    config = _request(
        resume=True, config_path="/host/run/checkpoints/5000/pretrained_model/train_config.json"
    )
    with pytest.raises(ValueError, match="[Rr]esum"):
        localize_config_for_cloud(config, "t4-small")


def test_localize_allows_cloud_resume_from_hub() -> None:
    """A cloud resume signals via resume_from_hub_repo (the wrapper downloads the
    checkpoint from the Hub), not a host-local config_path — so localization must
    NOT reject it. The host config_path stays unset here; the runner sets the
    container path later."""
    from makermodslab.runners.hf_cloud import localize_config_for_cloud

    config = _request(resume=True, resume_from_hub_repo="user/parent-run", resume_from_hub_step="005000")
    localize_config_for_cloud(config, "t4-small")  # no raise
    assert config.policy_device == "cuda"


def test_localize_rejects_a_resume_that_names_no_hub_checkpoint() -> None:
    """MT42's invariant at the cloud submission boundary: `resume` with nothing
    to resume FROM would fall through build_training_command's fresh-run branch
    and start over at step 0 on rented hardware while every UI surface reported a
    continuation. Refuse instead — silence is the failure mode here."""
    from makermodslab.runners.hf_cloud import localize_config_for_cloud

    config = _request(resume=True)
    with pytest.raises(ValueError, match="step 0"):
        localize_config_for_cloud(config, "t4-small")


def test_localize_rejects_local_pretrained_path_but_allows_hub_id() -> None:
    from makermodslab.runners.hf_cloud import localize_config_for_cloud

    local = _request(policy_pretrained_path="/host/checkpoints/000500/pretrained_model")
    with pytest.raises(ValueError, match="[Ff]ine-tun"):
        localize_config_for_cloud(local, "t4-small")

    hub = _request(policy_pretrained_path="user/some-model")
    localize_config_for_cloud(hub, "t4-small")  # no raise
    assert hub.policy_pretrained_path == "user/some-model"


def test_localize_allows_a_step_suffixed_hub_pretrained_ref() -> None:
    """MT2, cloud half: fine-tuning from a SPECIFIC Hub step travels as the ref
    'repo@checkpoints/<step_dir>' — the wrapper materializes it pod-side, the
    same way cloud resume already downloads its parent checkpoint. It must reach
    the container verbatim, not be rejected as a host path and not be rewritten
    here (a host path would be meaningless on the pod)."""
    from makermodslab.runners.hf_cloud import localize_config_for_cloud

    config = _request(policy_pretrained_path="user/some-model@checkpoints/003000")
    localize_config_for_cloud(config, "t4-small")  # no raise
    assert config.policy_pretrained_path == "user/some-model@checkpoints/003000"


# -- in-container installer ladder (the image's uv venv ships no pip) --


def test_install_plan_prefers_uv() -> None:
    """The lerobot-gpu image's venv is uv-created (no pip module), so uv on
    PATH must win, with --python pinning the install into this interpreter."""
    from makermodslab.runners.hf_cloud import _install_plan

    label, cmds = _install_plan("lerobot @ url", "/venv/bin/python", "/usr/local/bin/uv", True, True)
    assert label == "uv"
    assert cmds == [
        [
            "/usr/local/bin/uv",
            "pip",
            "install",
            "--python",
            "/venv/bin/python",
            "--no-cache",
            "lerobot @ url",
        ]
    ]


def test_install_plan_falls_back_to_pip_without_uv() -> None:
    from makermodslab.runners.hf_cloud import _install_plan

    label, cmds = _install_plan("spec", "/py", None, True, True)
    assert label == "pip"
    assert cmds == [["/py", "-m", "pip", "install", "--no-cache-dir", "spec"]]


def test_install_plan_bootstraps_pip_via_ensurepip_as_last_resort() -> None:
    from makermodslab.runners.hf_cloud import _install_plan

    label, cmds = _install_plan("spec", "/py", None, False, True)
    assert label == "ensurepip+pip"
    assert cmds == [
        ["/py", "-m", "ensurepip", "--upgrade"],
        ["/py", "-m", "pip", "install", "--no-cache-dir", "spec"],
    ]


def test_install_plan_reports_no_installer() -> None:
    from makermodslab.runners.hf_cloud import _install_plan

    assert _install_plan("spec", "/py", None, False, False) == (None, [])


# -- checkpoint-completeness check (partial-upload race fix) --


def test_checkpoint_step_ready_false_when_no_last_link_exists(tmp_path: Path) -> None:
    """No checkpoints/last symlink at all — e.g. before the first checkpoint
    of a run has completed — must not be mistaken for readiness."""
    from makermodslab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    step_dir.mkdir()
    assert _checkpoint_step_ready(step_dir) is False


def test_checkpoint_step_ready_false_when_last_points_to_an_earlier_step(tmp_path: Path) -> None:
    """The exact race this fixes: lerobot writes pretrained_model/config.json
    (and creates training_state/) well before the step is actually complete.
    Only checkpoints/last advancing to (or past) this step is proof the whole
    step finished — a step dir existing on its own proves nothing."""
    from makermodslab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "010000"
    step_dir.mkdir()
    (tmp_path / "005000").mkdir()
    (tmp_path / "last").symlink_to("005000")
    assert _checkpoint_step_ready(step_dir) is False


def test_checkpoint_step_ready_true_when_last_points_to_this_step(tmp_path: Path) -> None:
    from makermodslab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    step_dir.mkdir()
    (tmp_path / "last").symlink_to("005000")
    assert _checkpoint_step_ready(step_dir) is True


def test_checkpoint_step_ready_true_when_last_has_advanced_past_this_step(tmp_path: Path) -> None:
    """<=, not ==: two checkpoints can complete inside one 15s poll window, and
    a step must not go permanently unuploaded just because the poller missed
    the instant `last` pointed at it exactly."""
    from makermodslab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    step_dir.mkdir()
    (tmp_path / "010000").mkdir()
    (tmp_path / "last").symlink_to("010000")
    assert _checkpoint_step_ready(step_dir) is True


def test_checkpoint_step_ready_false_when_last_is_not_a_symlink(tmp_path: Path) -> None:
    """A plain file or directory named `last` (not the symlink lerobot writes)
    must not be misread as a target step number."""
    from makermodslab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    step_dir.mkdir()
    (tmp_path / "last").mkdir()
    assert _checkpoint_step_ready(step_dir) is False


def test_wrapper_source_inlines_the_tested_checkpoint_ready_check() -> None:
    """The wrapper's checkpoint-completeness check is _checkpoint_step_ready's
    source inlined verbatim, so the in-container upload gate is exactly the
    function the tests above exercise — and _scan_and_upload must call it
    instead of checking config.json directly."""
    import inspect

    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE, _checkpoint_step_ready

    assert inspect.getsource(_checkpoint_step_ready) in WRAPPER_SOURCE
    assert "__CHECKPOINT_READY_SOURCE__" not in WRAPPER_SOURCE  # placeholder replaced
    assert "_checkpoint_step_ready(entry)" in WRAPPER_SOURCE


def test_lerobot_last_checkpoint_symlink_matches_our_readiness_check() -> None:
    """_checkpoint_step_ready's readiness signal is lerobot's own
    checkpoints/<LAST_CHECKPOINT_LINK> symlink, which lerobot_train.py points
    at a step directory via update_last_checkpoint() strictly after
    save_checkpoint() returns for that step. A lerobot pin bump that renames
    the link or reorders those two calls should fail here in CI, not ship a
    gate that silently never passes."""
    import inspect

    from lerobot.scripts import lerobot_train
    from lerobot.utils.constants import LAST_CHECKPOINT_LINK

    assert LAST_CHECKPOINT_LINK == "last"
    src = inspect.getsource(lerobot_train)
    assert src.index("save_checkpoint(") < src.index("update_last_checkpoint(checkpoint_dir)")


# -- wrapper sanity --


def test_wrapper_source_compiles_and_launches_an_argv_list() -> None:
    """The wrapper must pass the trainer argv to Popen as a LIST (splitting a
    joined string was the bug-3 hypothesis — it is not the case and must stay
    that way) and quote its log line so spaced values read unambiguously."""
    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE

    compile(WRAPPER_SOURCE, "<hf-jobs-wrapper>", "exec")  # syntactically valid
    assert "subprocess.Popen(list(trainer_argv)" in WRAPPER_SOURCE
    assert "shlex.join(trainer_argv)" in WRAPPER_SOURCE
    assert re.search(r"^import .*\bshlex\b", WRAPPER_SOURCE, re.MULTILINE)  # imported up top


def test_wrapper_source_handles_resume_download() -> None:
    """Cloud resume: the wrapper must parse --resume-from, download the parent
    checkpoint tree, refuse when the downloaded step is incomplete, and
    pre-seed `seen` so it never re-uploads the checkpoint it just pulled
    down."""
    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE

    compile(WRAPPER_SOURCE, "<hf-jobs-wrapper>", "exec")  # still valid with the resume block
    assert "--resume-from=" in WRAPPER_SOURCE
    assert "snapshot_download" in WRAPPER_SOURCE
    assert "seen.add(step_dir)" in WRAPPER_SOURCE


def test_wrapper_source_resume_checks_weights_and_training_state_not_just_a_bare_dir() -> None:
    """C4: a plain `training_state/` is_dir() check would pass a checkpoint
    that was itself only partially uploaded before this fix existed — the
    checkpoints/last symlink used by the live watcher isn't available here
    (it's never pushed to the Hub), so the resume path checks the files it
    actually needs directly instead of trusting the directory's existence."""
    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE

    assert 'any((dest / "pretrained_model").glob("*.safetensors"))' in WRAPPER_SOURCE
    assert '(dest / "training_state" / "training_step.json").is_file()' in WRAPPER_SOURCE
    assert '(dest / "training_state" / "rng_state.safetensors").is_file()' in WRAPPER_SOURCE


def test_wrapper_source_materializes_a_step_suffixed_finetune_base() -> None:
    """MT2, container half: a --policy.pretrained_path naming a Hub STEP is
    downloaded pod-side and rewritten to that local dir before the trainer runs.

    Two properties the block must keep:
      * it pulls ONLY that step's pretrained_model/ (weights-only — fine-tuning
        needs no training_state/), and
      * it uses the snapshot cache rather than <output_dir>/checkpoints/, which
        the uploader watches — a base checkpoint copied there would be
        republished as if this run had produced it.

    Source-level assertions: the block is top-level wrapper code (like the
    resume download it mirrors), so it has no import seam to exec against. The
    argv rewrite it depends on IS unit-tested below."""
    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE

    compile(WRAPPER_SOURCE, "<hf-jobs-wrapper>", "exec")
    assert '_arg("--policy.pretrained_path")' in WRAPPER_SOURCE
    assert '_set_arg("--policy.pretrained_path", str(base_dir))' in WRAPPER_SOURCE
    assert 'allow_patterns=[f"checkpoints/{step_dir}/pretrained_model/*"]' in WRAPPER_SOURCE
    # Never staged under the watched output dir.
    assert 'base_dir = Path(local_root) / "checkpoints" / step_dir / "pretrained_model"' in WRAPPER_SOURCE


def _wrapper_argv_helpers(trainer_argv: list[str]):
    """Exec the wrapper's own `_arg` / `_set_arg` over `trainer_argv`.

    Sliced out of WRAPPER_SOURCE by name and given the globals the wrapper would
    have, so a drift between the template and these tests fails loudly instead
    of silently testing a host-side paraphrase."""
    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE

    namespace: dict = {"trainer_argv": trainer_argv}
    for name in ("_arg", "_set_arg"):
        match = re.search(rf"^def {name}\(.*?(?=^\S)", WRAPPER_SOURCE, re.MULTILINE | re.DOTALL)
        assert match, f"{name} not found in WRAPPER_SOURCE"
        exec(compile(match.group(0), "<hf-jobs-wrapper>", "exec"), namespace)  # noqa: S102
    return namespace["_arg"], namespace["_set_arg"]


def test_wrapper_set_arg_rewrites_both_argv_spellings() -> None:
    """The rewrite must hit whichever form the argv builder used, and touch
    nothing else — this is what turns a Hub ref only the pod can resolve into a
    real path for the trainer."""
    joined = ["--policy.type=act", "--policy.pretrained_path=user/repo@checkpoints/003000", "--steps=10"]
    arg, set_arg = _wrapper_argv_helpers(joined)
    assert arg("--policy.pretrained_path") == "user/repo@checkpoints/003000"
    assert set_arg("--policy.pretrained_path", "/tmp/base") is True
    assert joined == ["--policy.type=act", "--policy.pretrained_path=/tmp/base", "--steps=10"]

    split = ["--policy.pretrained_path", "user/repo@checkpoints/003000", "--steps", "10"]
    arg, set_arg = _wrapper_argv_helpers(split)
    assert arg("--policy.pretrained_path") == "user/repo@checkpoints/003000"
    assert set_arg("--policy.pretrained_path", "/tmp/base") is True
    assert split == ["--policy.pretrained_path", "/tmp/base", "--steps", "10"]


def test_wrapper_set_arg_reports_a_missing_flag() -> None:
    """A bare-repo-id fine-tune (or no fine-tune at all) leaves the argv alone;
    the wrapper treats an unexpected miss as a hard error rather than launching
    a run that trains from the wrong weights."""
    argv = ["--policy.type=act"]
    arg, set_arg = _wrapper_argv_helpers(argv)
    assert arg("--policy.pretrained_path") is None
    assert set_arg("--policy.pretrained_path", "/tmp/base") is False
    assert argv == ["--policy.type=act"]


def test_cloud_resume_argv_keeps_lineage_in_parent_repo() -> None:
    """A cloud-resume config resolves to a --config_path at the container path and
    pushes into the parent's repo (same lineage), with resume essentials only."""
    from makermodslab.train import TrainingRequest, build_training_command

    req = TrainingRequest(
        dataset_repo_id="user/ds",
        resume=True,
        steps=20000,
        policy_push_to_hub=True,
        policy_repo_id="user/parent-run",
        config_path="/tmp/makermodslab/train/checkpoints/005000/pretrained_model/train_config.json",
    )
    cmd = build_training_command(req, output_dir="/tmp/makermodslab/train")
    assert (
        "--config_path=/tmp/makermodslab/train/checkpoints/005000/pretrained_model/train_config.json" in cmd
    )
    assert cmd[cmd.index("--policy.push_to_hub") + 1] == "true"
    assert cmd[cmd.index("--policy.repo_id") + 1] == "user/parent-run"
    assert cmd[cmd.index("--resume") + 1] == "true"
    # Inherited from the checkpoint — never re-passed on resume.
    assert "--dataset.repo_id" not in cmd
    assert "--policy.type" not in cmd


def _submitted_command(config, tmp_path, monkeypatch, job_id: str = "child_run"):
    """Drive HfCloudJobRunner.start with the Hub stubbed out; return the argv it
    submitted. No token, no network, no job — the run_job stand-in records and
    the worker threads are stubbed off."""
    from unittest.mock import MagicMock

    from makermodslab.jobs import TrainingMetrics
    from makermodslab.runners.hf_cloud import HfCloudJobRunner

    monkeypatch.setattr("makermodslab.runners.hf_cloud.get_token", lambda: "hf_fake")
    monkeypatch.setattr("makermodslab.runners.hf_cloud.cached_whoami", lambda: {"name": "alice"})
    runner = HfCloudJobRunner(TrainingMetrics(), tmp_path / "log.jsonl", "t4-small")
    api = MagicMock()
    api.run_job.return_value = MagicMock(id="hfjob-1", url="https://hf/jobs/1")
    runner._api = api
    monkeypatch.setattr(runner, "_ensure_dataset_on_hub", lambda repo_id: None)
    monkeypatch.setattr(runner, "_start_worker_threads", lambda label: None)
    runner.start(job_id, config, "/host/out")
    return api.run_job.call_args.kwargs["command"]


def test_cloud_resume_from_a_cloud_parent_publishes_into_the_parents_repo(tmp_path, monkeypatch) -> None:
    """Unchanged behaviour, pinned: a cloud→cloud continuation keeps the whole
    lineage in one repo (its known cost is MT12's parent/child ambiguity)."""
    config = _request(resume=True, resume_from_hub_repo="user/parent-run", resume_from_hub_step="005000")
    command = _submitted_command(config, tmp_path, monkeypatch)

    assert config.policy_repo_id == "user/parent-run"
    assert "--resume-from=user/parent-run@checkpoints/005000" in command


def test_cloud_resume_from_an_uploaded_local_checkpoint_gets_its_own_repo(tmp_path, monkeypatch) -> None:
    """F7, local→cloud: the source repo is a private STAGING repo holding the
    local parent's uploaded checkpoint, not an output repo. Publishing into it
    would put parent and child checkpoints in one tree again (MT12's shape), so
    the run takes its own repo — while still resuming from the staged step."""
    config = _request(
        resume=True,
        resume_from_hub_repo="alice/src_checkpoints",
        resume_from_hub_step="000100",
        resume_from_uploaded_checkpoint=True,
    )
    command = _submitted_command(config, tmp_path, monkeypatch)

    assert config.policy_repo_id == "alice/child_run"
    assert "--resume-from=alice/src_checkpoints@checkpoints/000100" in command
    assert "--policy.repo_id" in command
    assert command[command.index("--policy.repo_id") + 1] == "alice/child_run"


def test_wrapper_source_inlines_the_tested_install_plan() -> None:
    """The wrapper's installer choice is _install_plan's source inlined
    verbatim, so the in-container code is exactly what the unit tests above
    exercised — uv first (shutil.which), pip / ensurepip as fallbacks."""
    import inspect

    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE, _install_plan

    assert inspect.getsource(_install_plan) in WRAPPER_SOURCE
    assert "__INSTALL_PLAN_SOURCE__" not in WRAPPER_SOURCE  # placeholder replaced
    assert 'shutil.which("uv")' in WRAPPER_SOURCE
    assert "no uv, pip, or ensurepip" in WRAPPER_SOURCE  # clear terminal message


# ---------------------------------------------------------------------------
# Job-timeout precedence: request value wins (normalised to seconds), else the
# HF_JOB_TIMEOUT fallback constant.
# ---------------------------------------------------------------------------


def test_resolve_job_timeout_falls_back_to_constant_when_unset() -> None:
    from makermodslab.runners.hf_cloud import HF_JOB_TIMEOUT, resolve_job_timeout
    from makermodslab.train import TrainingRequest

    config = TrainingRequest(dataset_repo_id="x")
    assert config.hf_job_timeout is None
    assert resolve_job_timeout(config) == HF_JOB_TIMEOUT  # string passthrough, not seconds


def test_hf_job_timeout_constant_is_single_unit_and_covers_measured_runs() -> None:
    """The fallback is handed to run_job as a raw string, and run_job's parser is
    only `float(timeout[:-1]) * factor[timeout[-1]]` — a single unit suffix. A
    compound "improvement" like "1d12h" would not survive that, so pin the shape
    as well as the budget: it must clear the longest run we have measured
    (SmolVLA 15k steps at 2.24 s/step on a10g-small ≈ 8.8h)."""
    from makermodslab.runners.hf_cloud import HF_JOB_TIMEOUT

    assert isinstance(HF_JOB_TIMEOUT, str)
    factors = {"s": 1, "m": 60, "h": 3600, "d": 3600 * 24}  # run_job's own table
    assert HF_JOB_TIMEOUT[-1] in factors
    seconds = float(HF_JOB_TIMEOUT[:-1]) * factors[HF_JOB_TIMEOUT[-1]]  # no ValueError
    assert seconds == 24 * 3600
    assert seconds > 8.8 * 3600


def test_resolve_job_timeout_uses_request_value_normalised_to_seconds() -> None:
    """An explicit request value wins over the constant and is converted to an
    int of seconds — run_job's own str parser only handles a single unit, so
    compound forms like "3h30m" must be pre-resolved here."""
    from makermodslab.runners.hf_cloud import resolve_job_timeout
    from makermodslab.train import TrainingRequest

    assert resolve_job_timeout(TrainingRequest(dataset_repo_id="x", hf_job_timeout="45m")) == 2700
    assert resolve_job_timeout(TrainingRequest(dataset_repo_id="x", hf_job_timeout="3h30m")) == 12600
    assert resolve_job_timeout(TrainingRequest(dataset_repo_id="x", hf_job_timeout="2h")) == 7200


# --- stop(): distinguishing a cancel from a crash ---------------------------
#
# returncode() collapses every non-COMPLETED stage to 1, so the registry
# classifies cloud runs on terminal_stage() instead. These cover stop()'s own
# decisions only — no submission, no threads, no network — because the stage
# stop() leaves behind is what decides whether a stopped run reads as
# `interrupted` or as a failed model.


class _FakeStatus:
    def __init__(self, stage, message=None):
        self.stage = stage
        self.message = message


class _FakeJobInfo:
    def __init__(self, stage, message=None):
        self.status = _FakeStatus(stage, message)


class _FakeJobsApi:
    """Just the two calls stop() makes."""

    def __init__(self, *, cancel_raises=False, inspect=None, inspect_raises=False):
        self._cancel_raises = cancel_raises
        self._inspect = inspect
        self._inspect_raises = inspect_raises
        self.cancelled = []
        self.inspected = []

    def cancel_job(self, job_id):
        self.cancelled.append(job_id)
        if self._cancel_raises:
            raise RuntimeError("404 job already ended")

    def inspect_job(self, job_id):
        self.inspected.append(job_id)
        if self._inspect_raises:
            raise RuntimeError("network down")
        return self._inspect


def _runner_with(api, tmp_path, *, stage=None):
    from makermodslab.jobs import TrainingMetrics
    from makermodslab.runners.hf_cloud import HfCloudJobRunner

    runner = HfCloudJobRunner(TrainingMetrics(), tmp_path / "log.jsonl", "a10g-small")
    runner._api = api
    runner._hf_job_id = "job-1"
    runner._terminal_status = stage
    return runner


def test_stop_records_canceled_stage(tmp_path) -> None:
    api = _FakeJobsApi()
    runner = _runner_with(api, tmp_path)

    runner.stop()

    assert api.cancelled == ["job-1"]
    assert runner.terminal_stage() == "CANCELED"
    assert runner.is_running() is False
    # No corrective lookup needed when the cancel was accepted.
    assert api.inspected == []


def test_stop_does_not_overwrite_a_stage_the_poller_already_saw(tmp_path) -> None:
    """_set_terminal is idempotent, and that is what keeps a run which
    finished before the stop landed reported as COMPLETED."""
    api = _FakeJobsApi()
    runner = _runner_with(api, tmp_path, stage="COMPLETED")

    runner.stop()

    assert runner.terminal_stage() == "COMPLETED"
    assert runner.returncode() == 0


def test_stop_adopts_the_real_stage_when_cancel_is_refused(tmp_path) -> None:
    """cancel_job refusing usually means the job had ALREADY ended, so the
    pre-set CANCELED describes a run that finished on its own. Re-read it."""
    api = _FakeJobsApi(cancel_raises=True, inspect=_FakeJobInfo("COMPLETED"))
    runner = _runner_with(api, tmp_path)

    runner.stop()

    assert api.inspected == ["job-1"]
    assert runner.terminal_stage() == "COMPLETED"
    assert runner.returncode() == 0


def test_stop_adopts_an_error_stage_and_its_message(tmp_path) -> None:
    api = _FakeJobsApi(cancel_raises=True, inspect=_FakeJobInfo("ERROR", "Job timeout"))
    runner = _runner_with(api, tmp_path)

    runner.stop()

    assert runner.terminal_stage() == "ERROR"
    assert runner.terminal_message() == "Job timeout"


def test_stop_keeps_canceled_when_the_stage_cannot_be_re_read(tmp_path) -> None:
    """An unreachable Hub leaves CANCELED standing: our cancel is already out,
    so it's the best available account of the run."""
    api = _FakeJobsApi(cancel_raises=True, inspect_raises=True)
    runner = _runner_with(api, tmp_path)

    runner.stop()

    assert runner.terminal_stage() == "CANCELED"


def test_stop_keeps_canceled_when_the_job_is_still_running(tmp_path) -> None:
    """cancel_job can also fail transiently. A non-terminal stage is no
    evidence that the run ended on its own, so don't adopt it."""
    api = _FakeJobsApi(cancel_raises=True, inspect=_FakeJobInfo("RUNNING"))
    runner = _runner_with(api, tmp_path)

    runner.stop()

    assert runner.terminal_stage() == "CANCELED"


def test_stop_is_a_noop_before_submission(tmp_path) -> None:
    api = _FakeJobsApi()
    runner = _runner_with(api, tmp_path)
    runner._hf_job_id = None

    runner.stop()

    assert api.cancelled == []
    assert runner.terminal_stage() is None


# ---------------------------------------------------------------------------
# MT47: the log tail must not go permanently silent mid-run.
# ---------------------------------------------------------------------------


def test_is_replayed_skips_a_repeated_line_but_not_a_new_one(tmp_path) -> None:
    """The replay de-dupe, as a pure function. Content-based, so it is correct
    whether a reconnect replays the whole log, part of it, or none of it — the
    three cases the old positional counter could not tell apart."""
    runner = _runner_with(_FakeJobsApi(), tmp_path)

    assert runner._is_replayed("first") is False
    assert runner._is_replayed("second") is False
    # The reconnect replays both, then delivers something genuinely new.
    assert runner._is_replayed("first") is True
    assert runner._is_replayed("second") is True
    assert runner._is_replayed("third") is False


def test_is_replayed_window_is_bounded_and_forgets_the_oldest(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Memory is bounded, and a line that falls out of the window is re-emitted
    rather than suppressed. That direction is deliberate: a duplicated log line
    is cosmetic, whereas the scheme this replaced failed the other way and went
    mute for the rest of the run."""
    from collections import deque

    runner = _runner_with(_FakeJobsApi(), tmp_path)
    runner._recent_lines = deque(maxlen=3)
    runner._recent_line_set = set()

    for line in ("a", "b", "c"):
        assert runner._is_replayed(line) is False
    assert runner._is_replayed("a") is True  # still inside the window

    runner._is_replayed("d")  # evicts "a"
    assert len(runner._recent_lines) == 3
    assert runner._recent_line_set == set(runner._recent_lines)
    assert runner._is_replayed("a") is False  # aged out ⇒ re-emitted, not dropped


def _drain(runner) -> list[str]:
    from queue import Empty as _Empty

    out = []
    while True:
        try:
            out.append(runner._log_queue.get_nowait().message)
        except _Empty:
            return out


def _run_tail_briefly(runner, monkeypatch, *, silence=30.0, seconds=1.0) -> list[str]:
    """Drive _tail_loop on a thread, then stop it from THIS thread.

    The stop always comes from the test, never from inside the fake stream:
    the loop consumes its reader through a queue, so a fake that set the event
    mid-yield would race the consumer and make the assertion meaningless.
    """
    import threading
    import time

    from makermodslab.runners import hf_cloud

    monkeypatch.setattr(hf_cloud, "_TAIL_CLEAN_END_WAIT_S", 0.01)
    monkeypatch.setattr(hf_cloud, "_TAIL_RECONNECT_BACKOFF_S", 0.01)
    monkeypatch.setattr(hf_cloud, "_TAIL_SILENCE_TIMEOUT_S", silence)

    thread = threading.Thread(target=runner._tail_loop, daemon=True)
    thread.start()
    time.sleep(seconds)
    lines = _drain(runner)
    runner._stop_event.set()
    thread.join(timeout=5)
    return lines


def test_tail_loop_keeps_delivering_after_a_reconnect_replays_less_than_it_had(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MT47, the exact regression. A reconnect that replays FEWER lines than
    were already processed used to leave a per-connection counter permanently
    behind a cross-connection total, so every later line was skipped — the job
    ran to completion while progress and charts froze at the drop.
    """
    import time

    calls = {"n": 0}
    api = _FakeJobsApi()

    def fetch_job_logs(job_id, follow):
        calls["n"] += 1
        if calls["n"] == 1:
            yield from [f"L{i}" for i in range(1, 7)]
            raise RuntimeError("SSE dropped mid-run")
        # HF replays only a short tail, then the run continues.
        yield from ["L5", "L6", "L7", "L8", "L9"]
        while True:
            time.sleep(0.02)

    api.fetch_job_logs = fetch_job_logs
    runner = _runner_with(api, tmp_path)

    lines = _run_tail_briefly(runner, monkeypatch)

    assert calls["n"] == 2, "expected exactly one reconnect"
    # The new lines arrive...
    assert [x for x in lines if x in ("L7", "L8", "L9")] == ["L7", "L8", "L9"]
    # ...and the replayed prefix is not teed a second time.
    assert lines == ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]


def test_tail_loop_reconnects_when_a_connection_stops_yielding(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of MT47. `fetch_job_logs(follow=True)` can block inside a
    single read forever — no line, no StopIteration, no exception — and a plain
    `for` over it cannot even observe _stop_event, because the loop body never
    runs again. The silence timeout is what notices."""
    import time

    calls = {"n": 0}
    api = _FakeJobsApi()

    def fetch_job_logs(job_id, follow):
        calls["n"] += 1
        if calls["n"] == 1:
            yield "L1"
            while True:  # half-open socket
                time.sleep(0.02)
        else:
            yield "L2"  # the reconnect recovers the stream
            while True:
                time.sleep(0.02)

    api.fetch_job_logs = fetch_job_logs
    runner = _runner_with(api, tmp_path)

    lines = _run_tail_briefly(runner, monkeypatch, silence=0.3, seconds=2.0)

    assert calls["n"] >= 2, "a silent connection must be abandoned and retried"
    assert lines == ["L1", "L2"]


def test_tail_loop_does_not_reconnect_while_a_connection_is_still_talking(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The silence timer must measure silence, not connection age — otherwise a
    long run would be torn down and re-replayed on a fixed cadence."""
    import time

    calls = {"n": 0}
    api = _FakeJobsApi()

    def fetch_job_logs(job_id, follow):
        calls["n"] += 1
        for i in range(40):
            yield f"L{i}"
            time.sleep(0.05)

    api.fetch_job_logs = fetch_job_logs
    runner = _runner_with(api, tmp_path)

    _run_tail_briefly(runner, monkeypatch, silence=0.3, seconds=1.0)

    assert calls["n"] == 1
