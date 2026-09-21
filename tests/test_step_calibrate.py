# Copyright 2026 MakerMods. All rights reserved.
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
"""TB6a — the generic step-wizard calibration manager (makermodslab.step_calibrate).

The zero-pose manager became a family-agnostic wizard: the family runs its
procedure on the manager's worker thread through a ``CalibrationUI`` whose
``step`` publishes a step and BLOCKS until the user confirms, and the manager
owns everything a family must not (the mutex gate, the name-collision check,
connect through ``open_for_calibration``, writing the returned calibration,
the record write-back, the session-events phases, and torque release +
disconnect on every exit). Driven end to end with a fake ``steps`` family
and fake devices — no bus is ever opened.

House rules: no sleeps — the worker thread is observed through the
session-events seam (a Condition, not polling); ``tmp_lerobot_home`` for
every record and library dir; nothing here touches hardware.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from makermodslab import session_events, sessions
from makermodslab.api_errors import ApiError, ErrorCode
from makermodslab.arms import registry
from makermodslab.utils import config as cfg
from tests.mocks import make_arm_family, scratch_registry

POSE_A = "Fold the arm against the base — then confirm."
POSE_B = "Now open the gripper fully — then confirm."


@pytest.fixture(autouse=True)
def _fresh_tracker():
    """The tracker is a process singleton subscribed to the seam; a claim a
    failing (RED) manager leaves behind must not leak into the next test."""
    sessions.tracker.reset()
    yield
    sessions.tracker.reset()


class _Phases:
    """Every session_changed event the manager emits, waitable without sleeps."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self._cond = threading.Condition()

    def __call__(self, event: dict) -> None:
        with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    def wait_for(self, phase: str, *, count: int = 1, timeout: float = 5.0) -> None:
        with self._cond:
            ok = self._cond.wait_for(
                lambda: sum(1 for e in self.events if e["session"].get("phase") == phase) >= count,
                timeout=timeout,
            )
        assert ok, (
            f"never saw phase {phase!r} x{count}; got {[e['session'].get('phase') for e in self.events]}"
        )

    def wait_released(self, timeout: float = 5.0) -> None:
        with self._cond:
            ok = self._cond.wait_for(
                lambda: any(e["session"].get("active") is False for e in self.events), timeout=timeout
            )
        assert ok, f"never saw a release; got {self.events}"


@pytest.fixture
def phases():
    seen = _Phases()
    session_events.subscribe(seen)
    yield seen
    session_events.unsubscribe(seen)


class _FakeBus:
    """A torque-off bus: what read_positions' fallback reads and what the
    manager writes the calibration into when the bus can hold it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def sync_read(self, register: str) -> dict[str, float]:
        assert register == "Present_Position"
        self.calls.append(("sync_read", register))
        return {"j1": 12.5, "j2": -3.0}

    def write_calibration(self, calibration: dict) -> None:
        self.calls.append(("write_calibration", dict(calibration)))


class _FakeDevice:
    """What a fake family's open_for_calibration hands back."""

    def __init__(self, with_bus: bool = True) -> None:
        self.bus = _FakeBus() if with_bus else None
        self.calibration: dict | None = None
        self.calibration_fpath = "/fake/calibration.json"
        self.saved = 0
        self.disconnected = 0

    def _save_calibration(self) -> None:
        self.saved += 1

    def disconnect(self) -> None:
        self.disconnected += 1


class _Harness:
    """A registered fake ``steps`` family wired to record everything the
    manager asks of it, plus the device it hands out."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, family_id: str = "nine") -> None:
        from makermodslab.arms.base import CalibrationAborted

        self.device = _FakeDevice()
        self.opened: list[tuple] = []
        self.released: list[tuple] = []
        self.calibrated: list[tuple] = []
        self.aborted: list[str] = []
        self.observed: list[tuple] = []
        self.result = {"j1": "cal-j1", "j2": "cal-j2"}
        harness = self

        def open_for_calibration(device_type, port, config_id):
            harness.opened.append((device_type, port, config_id))
            return harness.device

        def calibrate(device, device_type, ui):
            harness.calibrated.append((device, device_type))
            try:
                ui.step(POSE_A, live_positions=True)
                ui.step(POSE_B)
            except CalibrationAborted as exc:
                harness.aborted.append(str(exc))
                raise
            ui.message("Setting zero…")
            from makermodslab.step_calibrate import step_calibration_manager

            status = step_calibration_manager.get_status()
            harness.observed.append((status.status, status.message, status.step))
            return dict(harness.result)

        def release_torque(device, label="device"):
            harness.released.append((device, label))
            return []

        scratch_registry(monkeypatch)
        self.family = make_arm_family(
            family_id,
            calibrate=calibrate,
            open_for_calibration=open_for_calibration,
            release_torque=release_torque,
        )
        registry.register(self.family, provided_by="ext")


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_lerobot_home: Path):
    from makermodslab import step_calibrate

    h = _Harness(monkeypatch)
    yield h
    if step_calibrate.step_calibration_is_active():
        step_calibrate.step_calibration_manager.stop()


def _request(**overrides):
    from makermodslab.step_calibrate import StepCalibrationRequest

    fields = {"device_type": "robot", "port": "/dev/nine0", "config_file": "unit", "arm_type": "nine"}
    fields.update(overrides)
    return StepCalibrationRequest(**fields)


# ---------------------------------------------------------------------------
# The happy path, step by step
# ---------------------------------------------------------------------------


def test_the_wizard_runs_the_families_steps_to_completion(harness: _Harness, phases: _Phases) -> None:
    """start → the family's first ``ui.step`` publishes ``awaiting_step`` 1 with
    the family's own text and live positions → complete_step → step 2 (no
    positions) → complete_step → the family's return value is written through
    the device exactly as the zero flow wrote it (``calibration`` set, the
    bus's ``write_calibration``, ``_save_calibration``) → ``completed`` →
    torque released, then disconnected."""
    from makermodslab.step_calibrate import step_calibration_is_active, step_calibration_manager

    manager = step_calibration_manager
    cfg.save_robot_record(
        "bot9", {"arm_type": "nine", "mode": "single", "follower_port": "/dev/old"}, allow_create=True
    )

    result = manager.start(_request(robot_name="bot9"))
    assert result["success"] is True, result
    assert step_calibration_is_active() is True

    phases.wait_for("awaiting_step", count=1)
    status = manager.get_status()
    assert status.status == "awaiting_step"
    assert status.calibration_active is True
    assert status.step == 1
    assert status.total_steps == status.step, "the step count is unknown up front: the wizard shows 'Step N'"
    assert status.message == POSE_A
    assert status.image_url is None
    assert status.live_positions is True
    # Read through the family's read_positions (base default: the bus's
    # Present_Position sync_read on a torque-off bus).
    assert status.current_positions == {"j1": 12.5, "j2": -3.0}
    assert harness.opened == [("robot", "/dev/nine0", "unit")]
    assert harness.calibrated == [(harness.device, "robot")]

    assert manager.complete_step()["success"] is True
    phases.wait_for("awaiting_step", count=2)
    status = manager.get_status()
    assert status.status == "awaiting_step"
    assert status.step == 2
    assert status.message == POSE_B
    assert status.live_positions is False
    assert status.current_positions is None, "positions are read only while a live-positions step is up"

    assert manager.complete_step(2)["success"] is True
    phases.wait_released()
    status = manager.get_status()
    assert status.status == "completed"
    assert status.calibration_active is False
    assert step_calibration_is_active() is False

    # ui.message() after the last step: a transient message, status `saving`.
    assert harness.observed == [("saving", "Setting zero…", 2)]
    # The family's dict was written exactly as the zero flow wrote it.
    assert harness.device.calibration == harness.result
    assert ("write_calibration", harness.result) in harness.device.bus.calls
    assert harness.device.saved == 1
    # Release BEFORE disconnect, and on the follower (device_type "robot").
    assert [d for d, _label in harness.released] == [harness.device]
    assert harness.device.disconnected == 1
    # The record's slot points at what was just written.
    record = cfg.get_robot_record("bot9")
    assert record["follower_port"] == "/dev/nine0"
    assert record["follower_config"] == "unit"

    kinds = [(e["session"]["kind"], e["session"].get("phase")) for e in phases.events]
    assert kinds[0] == ("calibration", "connecting")
    assert ("calibration", "awaiting_step") in kinds
    assert kinds[-1] == ("calibration", "completed")


def test_a_leader_run_is_never_torque_released(harness: _Harness, phases: _Phases) -> None:
    """The leader is human-held and never energized by this flow; releasing
    torque on it is at best a no-op and on a Star leader a write to servos the
    family never enabled. Disconnect only."""
    from makermodslab.step_calibrate import step_calibration_manager

    manager = step_calibration_manager
    assert manager.start(_request(device_type="teleop", port="/dev/star0"))["success"] is True
    phases.wait_for("awaiting_step", count=1)
    assert harness.opened == [("teleop", "/dev/star0", "unit")]
    manager.complete_step()
    phases.wait_for("awaiting_step", count=2)
    manager.complete_step(2)
    phases.wait_released()

    assert manager.get_status().status == "completed"
    assert harness.released == []
    assert harness.device.disconnected == 1


# ---------------------------------------------------------------------------
# Stop and timeout
# ---------------------------------------------------------------------------


def test_stop_mid_step_aborts_inside_the_family_and_releases_the_follower(
    harness: _Harness, phases: _Phases
) -> None:
    """A stop while ``ui.step`` is blocking raises ``CalibrationAborted`` INSIDE
    the family's ``calibrate`` (so a family can clean up its own state), the
    run finishes ``idle`` — not ``error`` — and the follower is torque-released
    then disconnected."""
    from makermodslab.step_calibrate import step_calibration_is_active, step_calibration_manager

    manager = step_calibration_manager
    assert manager.start(_request())["success"] is True
    phases.wait_for("awaiting_step", count=1)

    result = manager.stop()
    assert result["success"] is True
    phases.wait_released()

    assert step_calibration_is_active() is False
    assert manager.get_status().status == "idle"
    assert len(harness.aborted) == 1
    assert [d for d, _label in harness.released] == [harness.device]
    assert harness.device.disconnected == 1
    assert harness.device.saved == 0, "an aborted run writes nothing"


def test_stop_mid_step_on_a_leader_disconnects_without_a_release(harness: _Harness, phases: _Phases) -> None:
    from makermodslab.step_calibrate import step_calibration_manager

    manager = step_calibration_manager
    assert manager.start(_request(device_type="teleop", port="/dev/star0"))["success"] is True
    phases.wait_for("awaiting_step", count=1)
    manager.stop()
    phases.wait_released()

    assert manager.get_status().status == "idle"
    assert harness.released == []
    assert harness.device.disconnected == 1


def test_a_step_that_is_never_confirmed_times_out_as_an_error(
    harness: _Harness, phases: _Phases, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-step timeout (``_POSE_TIMEOUT_S``, 15 min in production) bounds
    how long a forgotten tab can hold the bus. Injected tiny here; the run
    finishes ``error`` — a timeout is not a user stop — with the device
    released and disconnected."""
    from makermodslab import step_calibrate

    monkeypatch.setattr(step_calibrate, "_POSE_TIMEOUT_S", 0.01)
    manager = step_calibrate.step_calibration_manager
    assert manager.start(_request())["success"] is True
    phases.wait_released()

    status = manager.get_status()
    assert status.status == "error"
    assert status.calibration_active is False
    assert status.error, "the timeout is reported as the run's error"
    assert len(harness.aborted) == 1
    assert [d for d, _label in harness.released] == [harness.device]
    assert harness.device.disconnected == 1
    assert harness.device.saved == 0


def test_the_production_timeout_is_still_fifteen_minutes() -> None:
    from makermodslab import step_calibrate

    assert step_calibrate._POSE_TIMEOUT_S == 15 * 60.0


# ---------------------------------------------------------------------------
# Start refusals: kind and arm type
# ---------------------------------------------------------------------------


def test_start_refuses_a_range_sweep_family_naming_the_kind(tmp_lerobot_home: Path) -> None:
    """An SO-101 is calibrated by a range sweep — this manager has no
    procedure to run for it. Same coded 400 the sessions surface uses for the
    mirror case (auto-calibration on a CAN arm), and the detail names the
    kind so the reader knows which flow to run instead."""
    from makermodslab.step_calibrate import step_calibration_is_active, step_calibration_manager

    with pytest.raises(ApiError) as excinfo:
        step_calibration_manager.start(_request(arm_type="so101", port="/dev/null-f"))
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == ErrorCode.ROBOT_NOT_READY
    assert "range_sweep" in excinfo.value.detail
    assert step_calibration_is_active() is False


def test_start_refuses_a_panel_family_naming_the_kind(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from makermodslab.step_calibrate import step_calibration_is_active, step_calibration_manager

    scratch_registry(monkeypatch)
    registry.register(
        make_arm_family("paneled", calibration_kind="panel", calibration_panel_url="/api/v1/ext/p/static/cal")
    )
    with pytest.raises(ApiError) as excinfo:
        step_calibration_manager.start(_request(arm_type="paneled"))
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == ErrorCode.ROBOT_NOT_READY
    assert "panel" in excinfo.value.detail
    assert step_calibration_is_active() is False


def test_start_refuses_an_unknown_arm_type(tmp_lerobot_home: Path) -> None:
    from makermodslab.step_calibrate import step_calibration_is_active, step_calibration_manager

    with pytest.raises(ApiError) as excinfo:
        step_calibration_manager.start(_request(arm_type="nope"))
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == ErrorCode.ROBOT_ARM_TYPE_UNAVAILABLE
    assert "'nope'" in excinfo.value.detail
    assert step_calibration_is_active() is False


def test_the_request_keeps_the_zero_requests_fields_and_maker_default() -> None:
    """Same fields as the zero request it replaces (sessions builds one from
    a record); ``arm_type`` still defaults to maker so a request built before
    the Metal arm existed is unchanged, and any registered id rides it."""
    from makermodslab.step_calibrate import StepCalibrationRequest

    request = StepCalibrationRequest(device_type="robot", port="/dev/x", config_file="c")
    assert request.arm_type == "maker"
    assert (request.robot_name, request.overwrite, request.arm) == (None, False, "left")
    assert StepCalibrationRequest(
        device_type="robot", port="/dev/x", config_file="c", arm_type="so101_twin"
    ).arm_type == ("so101_twin")


def test_the_status_shape_reads_as_one_client_shape() -> None:
    """The fields the wizard renders, with their idle defaults; ``awaiting_pose``
    is gone (the status string ``awaiting_step`` carries that fact)."""
    from dataclasses import asdict

    from makermodslab.step_calibrate import StepCalibrationStatus

    status = asdict(StepCalibrationStatus())
    assert status == {
        "calibration_active": False,
        "status": "idle",
        "device_type": None,
        "error": None,
        "message": "",
        "step": 0,
        "total_steps": 1,
        "current_positions": None,
        "recorded_ranges": None,
        "image_url": None,
        "live_positions": False,
    }


def test_the_old_module_and_names_are_gone() -> None:
    """The rename is complete: no ``zero_calibrate`` module and no ``zero_*``
    aliases left to drift out of step with the manager."""
    from pathlib import Path

    import makermodslab
    from makermodslab import step_calibrate

    # A filesystem check, not an import: the dev venv's editable finder falls
    # back to the main checkout for a submodule missing from a worktree, so
    # `import makermodslab.zero_calibrate` can succeed there for the wrong
    # reason. The package directory is unambiguous.
    package_dir = Path(makermodslab.__file__).parent
    assert not (package_dir / "zero_calibrate.py").exists()
    assert (package_dir / "step_calibrate.py").exists()

    assert not hasattr(step_calibrate, "ZeroCalibrationManager")
    assert not hasattr(step_calibrate, "zero_calibration_manager")
    assert not hasattr(step_calibrate, "zero_pose_instructions")


def test_a_confirm_for_another_step_is_refused(harness: _Harness, phases: _Phases) -> None:
    """A duplicate Next that reaches the server after the family has moved on
    must not confirm the step it never saw: with a multi-step family that is
    a hardware write in an unconfirmed pose. The client names the step it is
    confirming and any other number is refused, nothing set."""
    from makermodslab.step_calibrate import step_calibration_manager as manager

    assert manager.start(_request())["success"]
    phases.wait_for("awaiting_step")
    assert manager.get_status().step == 1

    stale = manager.complete_step(2)
    assert stale["success"] is False
    assert "not the current step" in stale["message"]
    assert manager.get_status().status == "awaiting_step"

    assert manager.complete_step(1)["success"]
    phases.wait_for("awaiting_step", count=2)
    assert manager.get_status().step == 2
    assert manager.complete_step(1)["success"] is False
    assert manager.complete_step(2)["success"]
    phases.wait_released()


def test_the_confirm_route_passes_the_step_number_through(client, harness: _Harness, phases: _Phases) -> None:
    """The legacy confirm route grew an optional body; a bodiless call (the
    range-sweep flow, older clients) still confirms, a mismatched step is
    refused with the manager's message."""
    from makermodslab.step_calibrate import step_calibration_manager as manager

    assert manager.start(_request())["success"]
    phases.wait_for("awaiting_step")
    refused = client.post("/api/v1/complete-calibration-step", json={"step": 7}).json()
    assert refused["success"] is False
    assert manager.get_status().step == 1
    assert client.post("/api/v1/complete-calibration-step").json()["success"]
    phases.wait_for("awaiting_step", count=2)
    assert client.post("/api/v1/complete-calibration-step", json={"step": 2}).json()["success"]
    phases.wait_released()


def test_a_stop_after_the_last_confirm_still_writes_the_calibration(
    harness: _Harness, phases: _Phases, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the family has returned, its motors already hold the new zero
    (RobStride set-zero and FashionStar origin persist in the motor). A stop
    that lands in that window is honoured a moment late: the file is written
    so the hardware and the library agree, and the run ends as completed."""
    from makermodslab.step_calibrate import step_calibration_manager as manager

    gate = threading.Event()
    past_last_wait = threading.Event()
    original = harness.family.calibrate

    def slow_calibrate(device, device_type, ui):
        result = original(device, device_type, ui)
        # Every ui.step has returned: the worker is past its last wait and the
        # procedure is done. Hold here so the stop lands in exactly that
        # window (set any earlier and ui.step honours it, correctly).
        past_last_wait.set()
        assert gate.wait(timeout=5.0)
        return result

    monkeypatch.setattr(harness.family, "calibrate", slow_calibrate)
    assert manager.start(_request())["success"]
    phases.wait_for("awaiting_step")
    assert manager.complete_step()["success"]
    phases.wait_for("awaiting_step", count=2)
    assert manager.complete_step(2)["success"]
    assert past_last_wait.wait(timeout=5.0)
    # Request a stop without joining (stop() would wait for the worker, which
    # waits on us).
    manager.stop_requested = True
    gate.set()
    phases.wait_released()
    assert harness.device.calibration == harness.result
    assert harness.device.saved == 1
    assert manager.get_status().status == "completed"


def test_a_bodiless_confirm_past_step_one_is_refused(harness: _Harness, phases: _Phases) -> None:
    """The bodiless confirm is the one-step contract older clients speak; on
    step 2 it cannot be told from a stale duplicate of step 1's click, so a
    multi-step run requires the confirm to name its step."""
    from makermodslab.step_calibrate import step_calibration_manager as manager

    assert manager.start(_request())["success"]
    phases.wait_for("awaiting_step")
    assert manager.complete_step()["success"]
    phases.wait_for("awaiting_step", count=2)
    refused = manager.complete_step()
    assert refused["success"] is False
    assert "must name the step" in refused["message"]
    assert manager.get_status().step == 2
    assert manager.complete_step(2)["success"]
    phases.wait_released()


def test_a_stop_that_loses_the_race_to_completion_reports_completed(
    harness: _Harness, phases: _Phases, monkeypatch: pytest.MonkeyPatch
) -> None:
    """stop() joins the worker; when the worker finished on its own in the
    meantime — past its last confirm, file written, record repointed — the
    reply and the status say completed, not "stopped": a caller reading
    "stopped" would believe the library is unchanged."""
    from makermodslab.step_calibrate import step_calibration_manager as manager

    gate = threading.Event()
    past_last_wait = threading.Event()
    original = harness.family.calibrate

    def slow_calibrate(device, device_type, ui):
        result = original(device, device_type, ui)
        past_last_wait.set()
        assert gate.wait(timeout=5.0)
        return result

    monkeypatch.setattr(harness.family, "calibrate", slow_calibrate)
    assert manager.start(_request())["success"]
    phases.wait_for("awaiting_step")
    assert manager.complete_step()["success"]
    phases.wait_for("awaiting_step", count=2)
    assert manager.complete_step(2)["success"]
    assert past_last_wait.wait(timeout=5.0)

    # stop() will join the worker, which is parked on the gate: release it
    # from another thread once the join has started.
    threading.Timer(0.05, gate.set).start()
    reply = manager.stop()
    assert reply["success"] is True
    assert "completed" in reply["message"].lower()
    assert manager.get_status().status == "completed"
    assert harness.device.calibration == harness.result
    assert harness.device.disconnected == 1


# ---------------------------------------------------------------------------
# Mutual exclusion
# ---------------------------------------------------------------------------


def test_calibration_is_active_ors_the_step_manager() -> None:
    """The whole reason this flow reuses the `calibration` session kind: every
    reciprocal check calls calibrate.calibration_is_active(), so widening that
    one function enrolls the wizard with no new robot.busy.* discriminant."""
    from makermodslab import calibrate, step_calibrate

    assert calibrate.calibration_is_active() is False
    try:
        step_calibrate.step_calibration_manager.status.calibration_active = True
        assert step_calibrate.step_calibration_is_active() is True
        assert calibrate.calibration_is_active() is True
    finally:
        step_calibrate.step_calibration_manager.status.calibration_active = False
    assert calibrate.calibration_is_active() is False


def test_start_refuses_while_another_feature_owns_the_bus(
    tmp_lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from makermodslab import step_calibrate, teleoperate

    monkeypatch.setattr(teleoperate, "teleoperation_active", True)
    result = step_calibrate.step_calibration_manager.start(_request(arm_type="maker", port="/dev/can0"))
    assert result["success"] is False
    assert result["code"] == ErrorCode.ROBOT_BUSY_TELEOPERATION
    assert step_calibrate.step_calibration_is_active() is False


# ---------------------------------------------------------------------------
# The legacy routes serve the step manager where they served the zero one
# ---------------------------------------------------------------------------


def test_calibration_status_for_the_sweep_flow_carries_the_wizard_defaults(client) -> None:
    """One client shape reads both flows: the SO-101 sweep's payload gains
    ``image_url: null`` and ``live_positions: false`` (as it gained
    ``awaiting_pose`` before, which is gone)."""
    resp = client.get("/api/v1/calibration-status")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["calibration_active"] is False
    assert payload["image_url"] is None
    assert payload["live_positions"] is False
    assert "awaiting_pose" not in payload


def test_the_legacy_routes_drive_a_live_step_calibration(client, harness: _Harness, phases: _Phases) -> None:
    """``/calibration-status``, ``/complete-calibration-step`` and
    ``/stop-calibration`` reach the step manager while it is live."""
    from makermodslab.step_calibrate import step_calibration_manager

    assert step_calibration_manager.start(_request())["success"] is True
    phases.wait_for("awaiting_step", count=1)

    payload = client.get("/api/v1/calibration-status").json()
    assert payload["status"] == "awaiting_step"
    assert payload["step"] == 1
    assert payload["message"] == POSE_A
    assert payload["live_positions"] is True
    assert payload["image_url"] is None
    assert payload["current_positions"] == {"j1": 12.5, "j2": -3.0}

    assert client.post("/api/v1/complete-calibration-step").json()["success"] is True
    phases.wait_for("awaiting_step", count=2)
    assert client.get("/api/v1/calibration-status").json()["step"] == 2

    assert client.post("/api/v1/stop-calibration").json()["success"] is True
    phases.wait_released()
    assert client.get("/api/v1/calibration-status").json()["calibration_active"] is False
