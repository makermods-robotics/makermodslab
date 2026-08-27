"""Presence: the pure half only.

Per the repo's test policy, the writer thread and every Hub call are
deliberately out of scope. What IS tested is the logic a wrong answer from
which would show the user something false: which runs get published, and how
much of a silent device's last payload we still believe.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from makermodslab import presence


def _rec(
    job_id="act_cube_2026-08-01_12-00-00",
    *,
    runner="local",
    state="running",
    started_at=1000.0,
    ended_at=None,
    current_step=42,
    steps=1000,
):
    return SimpleNamespace(
        id=job_id,
        job_number=7,
        name=job_id,
        display_name=None,
        state=state,
        runner=runner,
        started_at=started_at,
        ended_at=ended_at,
        metrics=SimpleNamespace(current_step=current_step),
        config=SimpleNamespace(steps=steps, policy_type="act", dataset_repo_id="u/d"),
    )


# --- build_payload --------------------------------------------------------


def test_payload_carries_a_running_local_run():
    payload = presence.build_payload([_rec()], now=2000.0, label="desktop", dev_id="dev-1")
    assert payload["schema"] == presence.PRESENCE_SCHEMA
    assert payload["device_label"] == "desktop"
    assert payload["device_id"] == "dev-1"
    (run,) = payload["runs"]
    assert run["state"] == "running"
    assert run["current_step"] == 42
    assert run["total_steps"] == 1000
    assert run["dataset_repo_id"] == "u/d"


@pytest.mark.parametrize("runner", ["hf_cloud", "imported"])
def test_payload_publishes_only_local_runs(runner):
    # A cloud run is already visible from every machine via HF Jobs; publishing
    # it here too would give one run two rows in one library.
    payload = presence.build_payload([_rec(runner=runner)], now=2000.0, label="d", dev_id="x")
    assert payload["runs"] == []


def test_payload_keeps_a_just_finished_run_so_the_finish_is_seen():
    rec = _rec(state="done", ended_at=1900.0)
    payload = presence.build_payload([rec], now=2000.0, label="d", dev_id="x")
    assert [r["state"] for r in payload["runs"]] == ["done"]


def test_payload_drops_a_long_finished_run():
    rec = _rec(state="done", ended_at=1000.0)
    now = 1000.0 + presence.TERMINAL_GRACE_S + 1
    assert presence.build_payload([rec], now=now, label="d", dev_id="x")["runs"] == []


def test_payload_publishes_nothing_beyond_the_declared_fields():
    # This file goes to the Hub, so the field list is a privacy surface, not a
    # convenience. A new field must be a deliberate choice, not a leak.
    payload = presence.build_payload([_rec()], now=2000.0, label="d", dev_id="x")
    assert set(payload) == {"schema", "device_id", "device_label", "updated_at", "runs"}
    assert set(payload["runs"][0]) == {
        "job_id",
        "job_number",
        "name",
        "display_name",
        "state",
        "current_step",
        "total_steps",
        "policy_type",
        "dataset_repo_id",
        "started_at",
        "ended_at",
    }


def test_payload_is_json_serializable():
    payload = presence.build_payload([_rec()], now=2000.0, label="d", dev_id="x")
    assert json.loads(json.dumps(payload))["runs"][0]["job_id"]


def test_has_active_runs_gates_the_keepalive():
    assert presence.has_active_runs(presence.build_payload([_rec()], now=1.0, label="d", dev_id="x"))
    idle = presence.build_payload([_rec(state="done", ended_at=1.0)], now=2.0, label="d", dev_id="x")
    assert not presence.has_active_runs(idle)


# --- liveness -------------------------------------------------------------


def test_liveness_boundaries():
    now = 10_000.0
    assert presence.classify_liveness(now, now=now) == "live"
    assert presence.classify_liveness(now - presence.STALE_AFTER_S, now=now) == "live"
    assert presence.classify_liveness(now - presence.STALE_AFTER_S - 1, now=now) == "unknown"
    assert presence.classify_liveness(now - presence.PRESUMED_STOPPED_AFTER_S, now=now) == "unknown"
    assert (
        presence.classify_liveness(now - presence.PRESUMED_STOPPED_AFTER_S - 1, now=now) == "presumed_stopped"
    )
    assert presence.classify_liveness(None, now=now) == "presumed_stopped"


def test_a_silent_device_never_still_reports_running():
    # THE rule. A machine unplugged mid-run never wrote a goodbye, so its last
    # payload says "running" forever. The UI must not keep claiming that.
    payload = presence.build_payload([_rec()], now=0.0, label="d", dev_id="x")
    now = presence.STALE_AFTER_S + 100
    projected = presence.project_device(payload, last_seen=0.0, now=now)
    assert projected["liveness"] == "unknown"
    assert [r["state"] for r in projected["runs"]] == ["unknown"]


def test_a_silent_device_is_never_reported_as_failed():
    # We observed a silence, not a failure. Inventing "failed" would send
    # someone to a machine to debug a run that is probably fine.
    payload = presence.build_payload([_rec()], now=0.0, label="d", dev_id="x")
    projected = presence.project_device(payload, last_seen=0.0, now=presence.PRESUMED_STOPPED_AFTER_S + 100)
    assert projected["liveness"] == "presumed_stopped"
    assert all(r["state"] != "failed" for r in projected["runs"])


def test_a_live_device_keeps_its_states_untouched():
    payload = presence.build_payload([_rec()], now=100.0, label="d", dev_id="x")
    projected = presence.project_device(payload, last_seen=100.0, now=120.0)
    assert projected["liveness"] == "live"
    assert [r["state"] for r in projected["runs"]] == ["running"]


# --- identity / settings --------------------------------------------------


def test_device_id_is_stable_and_minted_once(tmp_path, monkeypatch):
    path = tmp_path / "device_id.txt"
    monkeypatch.setattr(presence, "DEVICE_ID_FILE", str(path))
    first = presence.device_id()
    assert first and presence.device_id() == first


def test_device_id_remints_rather_than_raising_on_a_corrupt_file(tmp_path, monkeypatch):
    path = tmp_path / "device_id.txt"
    path.write_text("   \n")
    monkeypatch.setattr(presence, "DEVICE_ID_FILE", str(path))
    assert presence.device_id().strip()


def test_settings_default_to_publishing_on(tmp_path, monkeypatch):
    monkeypatch.setattr(presence, "PRESENCE_SETTINGS_FILE", str(tmp_path / "presence.json"))
    settings = presence.load_settings()
    assert settings["enabled"] is True
    assert settings["label"]


def test_settings_round_trip_and_survive_corruption(tmp_path, monkeypatch):
    path = tmp_path / "presence.json"
    monkeypatch.setattr(presence, "PRESENCE_SETTINGS_FILE", str(path))
    presence.save_settings(enabled=False, label="desktop")
    assert presence.load_settings()["enabled"] is False
    assert presence.load_settings()["label"] == "desktop"
    path.write_text("{not json")
    assert presence.load_settings()["enabled"] is True  # falls back to the default


def test_repo_and_file_naming():
    assert presence.presence_repo_id("alice") == "alice/makermodslab-presence"
    assert presence.device_file_path("dev-1") == "devices/dev-1.json"


# --- the publisher ---------------------------------------------------------
#
# Not a network test: the Hub client is INJECTED. What is exercised here is the
# write-decision logic, which is where this feature's two worst bugs lived — the
# event path silently never firing, and the shutdown goodbye losing a race with
# the keepalive. Both were invisible to the pure-payload tests above.


class _FakeApi:
    """Records uploads instead of performing them."""

    def __init__(self, fail_with: Exception | None = None):
        self.uploads: list[dict] = []
        self.repos: list[str] = []
        self.fail_with = fail_with

    def create_repo(self, **kwargs):
        self.repos.append(kwargs.get("repo_id"))

    def upload_file(self, **kwargs):
        if self.fail_with:
            raise self.fail_with
        self.uploads.append(
            {
                "path": kwargs["path_in_repo"],
                "payload": json.loads(kwargs["path_or_fileobj"].decode()),
            }
        )


class _FakeRegistry:
    def __init__(self, records):
        self._records = records

    def list(self, limit=10, *, with_checkpoints=True):
        # Presence must never ask for checkpoint counts: each one is a Hub call.
        assert with_checkpoints is False, "presence must not trigger checkpoint counting"
        return self._records


@pytest.fixture
def publisher(tmp_path, monkeypatch):
    monkeypatch.setattr(presence, "DEVICE_ID_FILE", str(tmp_path / "device_id.txt"))
    monkeypatch.setattr(presence, "PRESENCE_SETTINGS_FILE", str(tmp_path / "presence.json"))
    presence.reset_device_id_cache()
    monkeypatch.setattr(presence, "cached_whoami", lambda **kw: {"name": "alice"})

    def _make(records, api=None):
        api = api or _FakeApi()
        pub = presence.PresencePublisher(_FakeRegistry(records), api_factory=lambda: api)
        return pub, api

    return _make


def test_a_finished_run_is_published_even_though_nothing_is_active(publisher):
    # THE regression. When the last active run ends there is nothing left to
    # keep alive, so the keepalive declines to write — the terminal state has to
    # ride the EVENT path. That path was dead: the loop cleared the wake event
    # before the write-decision read it, so a finish was never published and the
    # other machine watched "running" decay into "unknown" instead.
    import time as _time

    # Inside TERMINAL_GRACE_S, so the finish is still on the board to be seen.
    pub, api = publisher([_rec(state="done", ended_at=_time.time() - 10)])
    pub.mark_dirty()
    pub._publish_once(dirty=pub._take_dirty())
    assert len(api.uploads) == 1
    assert api.uploads[0]["payload"]["runs"][0]["state"] == "done"


def test_an_idle_device_writes_nothing_without_an_event(publisher):
    # The other half of the same decision: no event and nothing active means no
    # commit at all. This is the "zero writes when idle" budget.
    pub, api = publisher([_rec(state="done", ended_at=1.0)])
    pub._publish_once(dirty=False)
    assert api.uploads == []


def test_the_keepalive_writes_only_while_a_run_is_active(publisher):
    pub, api = publisher([_rec(state="running")])
    pub._publish_once(dirty=False)
    assert len(api.uploads) == 1
    # A second immediate pass is inside the keepalive interval — no new commit.
    pub._publish_once(dirty=False)
    assert len(api.uploads) == 1


def test_the_payload_and_its_filename_carry_the_same_device_id(publisher):
    # If these disagree the device cannot recognize its own file on the board,
    # and lists its own runs back to itself as somebody else's.
    pub, api = publisher([_rec(state="running")])
    pub._publish_once(dirty=True)
    upload = api.uploads[0]
    assert upload["path"] == f"devices/{upload['payload']['device_id']}.json"


def test_a_change_during_a_publish_is_not_swallowed(publisher):
    # mark_dirty landing between the flag being taken and the next pass must
    # survive: the run that just started would otherwise wait a full keepalive.
    pub, _ = publisher([_rec(state="running")])
    pub.mark_dirty()
    assert pub._take_dirty() is True
    pub.mark_dirty()
    assert pub._take_dirty() is True  # the second event is still pending
    assert pub._take_dirty() is False


def test_a_forbidden_token_disables_publishing_permanently(publisher):
    # A read-only token can never succeed. Retrying forever while the UI's
    # toggle still reads "on" is the silent failure default-on must not become.
    pub, _ = publisher([_rec()])
    exc = Exception("forbidden")
    exc.response = SimpleNamespace(status_code=403)
    pub._note_failure(exc)
    assert pub.status()["disabled_reason"] == "forbidden"


def test_a_transient_failure_does_not_disable_publishing(publisher, monkeypatch):
    pub, _ = publisher([_rec()])
    monkeypatch.setattr(pub._stop, "wait", lambda timeout=None: True)  # no real sleep
    pub._note_failure(RuntimeError("connection reset"))
    assert pub.status()["disabled_reason"] is None


def test_publishing_is_skipped_when_signed_out(publisher, monkeypatch):
    pub, api = publisher([_rec(state="running")])
    monkeypatch.setattr(presence, "cached_whoami", lambda **kw: None)
    pub._publish_once(dirty=True)
    assert api.uploads == []


def test_sharing_turned_off_stops_the_writer(publisher):
    pub, api = publisher([_rec(state="running")])
    presence.save_settings(enabled=False)
    pub._wake.set()
    pub._stop.set()
    pub._loop()
    assert api.uploads == []


# --- board reading ---------------------------------------------------------


def test_a_malformed_presence_file_is_skipped_not_rendered():
    # Everything on the board crossed the network from another machine: a
    # half-written commit or a foreign file must not reach React.
    assert presence._sanitize_payload({"schema": 999}) is None
    assert presence._sanitize_payload("not a dict") is None
    assert presence._sanitize_payload({"schema": presence.PRESENCE_SCHEMA, "runs": "nope"})["runs"] == []
    mixed = presence._sanitize_payload(
        {"schema": presence.PRESENCE_SCHEMA, "runs": [{"job_id": "a"}, "junk", 7]}
    )
    assert mixed["runs"] == [{"job_id": "a"}]
    assert mixed["device_label"] == "unknown device"
