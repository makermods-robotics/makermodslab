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
#
"""The ``calibration`` namespace: watch a calibration and advance its steps.

STARTING a calibration is a leased session — ``client.sessions.calibrate()``
or ``client.sessions.auto_calibrate()`` — and stopping it is the session
stop. This namespace is the other half: the observe-and-advance surface the
session start has no way to expose.

Which half you need depends on the arm family (``client.system.arms()``
reports ``calibration.kind`` per family):

* ``steps`` (Maker, Metal) — the family publishes one instruction at a time
  and BLOCKS until it is confirmed. Poll :meth:`status` until
  ``awaiting_step``, relay ``message`` to the human, and call
  :meth:`complete_step` once the arm is physically posed. The CAN families
  publish exactly one step (pose the arm at its zero, torque off throughout).
  ``client.flows.calibrate_zero()`` wraps that loop.
* ``range_sweep`` (SO-101, manual) — the human sweeps every joint through its
  range while ``recorded_ranges`` fills in. That is a continuous visual loop
  and the web UI is the better tool; :meth:`status` lets an agent narrate it,
  but prefer auto-calibration below.
* SO-101 auto-calibration drives the arm itself with no human in the loop:
  start the session, then poll :meth:`auto_batch_status` to a terminal state.
  ``client.flows.auto_calibrate()`` wraps that.

PROVISIONAL surface: these routes are untagged and untyped server-side, so
their errors arrive as plain ApiError and the models here are hand-mirrored
against the handlers' dataclasses rather than generated from response models.
They are pinned in the coverage ratchet's UNTAGGED_NAMESPACES register and
graduate when the server tags them.
"""

from __future__ import annotations

from typing import Any, ClassVar

from makermodslab_sdk._operations import operation
from makermodslab_sdk.resources._base import RecordList, Resource, SdkModel

#: ``status`` values that mean the run is over, either way.
TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "error"})

#: ``status`` values that mean the wizard is waiting for a human confirmation.
AWAITING_STEP = "awaiting_step"


class CalibrationStatus(SdkModel):
    """One shape for both calibration flows (server.py ``calibration_status``).

    The step wizard and the SO-101 sweep publish field-compatible statuses, so
    one model reads either. ``image_url`` and ``live_positions`` are the step
    wizard's own and default (null / False) for the sweep; ``recorded_ranges``
    is the sweep's own and stays null for the wizard.

    ``status`` walks idle → connecting → ``awaiting_step`` (wizard) or
    ``recording`` (sweep) → saving → completed | error, with ``stopping`` on
    the way out. ``message`` is the instruction to show a human while
    ``awaiting_step`` — server prose, relay it verbatim.
    """

    calibration_active: bool = False
    status: str = "idle"
    device_type: str | None = None
    error: str | None = None
    message: str = ""
    step: int = 0
    #: Unknown up front for a steps family: while running it equals ``step``,
    #: so render "Step N", never "N of M".
    total_steps: int = 1
    current_positions: dict[str, float] | None = None
    #: The sweep's per-motor ``{min, max, current}``; null for the wizard.
    recorded_ranges: dict[str, dict[str, float]] | None = None
    image_url: str | None = None
    live_positions: bool = False

    @property
    def awaiting_step(self) -> bool:
        """True while the wizard is blocked on a human confirming this step."""
        return self.status == AWAITING_STEP

    @property
    def finished(self) -> bool:
        """True once the run reached ``completed`` or ``error``."""
        return self.status in TERMINAL_STATUSES


class StepConfirmation(SdkModel):
    """The answer to :meth:`complete_step`.

    ``success=False`` is a soft refusal with the reason in ``message`` — no
    calibration active, the wrong status, or a ``step`` that is no longer the
    one on screen — not an HTTP error.
    """

    success: bool
    message: str = ""


class AutoCalibrationArmStatus(SdkModel):
    """One arm of a batch auto-calibration (a superset of the single status)."""

    active: bool = False
    #: idle | running | stopping | completed | failed | stopped.
    status: str = "idle"
    message: str = ""
    error: str | None = None
    logs: list[str] = []
    name: str = ""
    port: str = ""
    device_type: str = ""
    arm: str = "left"


class AutoCalibrationStatus(SdkModel):
    """The single-arm auto-calibration manager's state."""

    active: bool = False
    status: str = "idle"
    message: str = ""
    error: str | None = None
    logs: list[str] = []

    @property
    def finished(self) -> bool:
        """True once this arm stopped running, however it ended."""
        return self.status in ("completed", "failed", "stopped")


class AutoCalibrationBatchStatus(SdkModel):
    """Per-arm status plus overall counts for a batch auto-calibration.

    A leased ``auto_calibration`` session always routes through the batch
    manager (even for one arm), so this is the status that flow polls.
    ``logs`` is the combined, per-arm-prefixed stream; each arm also carries
    its own.
    """

    active: bool = False
    arms: list[AutoCalibrationArmStatus] = []
    total: int = 0
    completed: int = 0
    failed: int = 0
    logs: list[str] = []

    @property
    def finished(self) -> bool:
        """True once no arm is still running and every arm has an outcome."""
        return not self.active and self.total > 0 and self.completed + self.failed >= self.total


class CalibrationConfig(SdkModel):
    """One saved calibration file in a library."""

    name: str
    filename: str = ""
    size: int = 0
    #: Epoch seconds.
    modified: float = 0.0


class CalibrationConfigList(RecordList):
    """A device type's calibration library.

    ``success=False`` carries the reason in ``message`` (still HTTP 200) —
    an invalid device type, or a library that could not be read.
    """

    success: bool

    RECORDS_FIELD: ClassVar[str] = "configs"

    configs: list[CalibrationConfig] = []
    device_type: str | None = None
    message: str | None = None


class CalibrationResource(Resource):
    """``client.calibration`` — watch a calibration run and advance its steps.

    Example:
        >>> with client.sessions.calibrate("bench", device_type="robot") as s:
        ...     status = client.calibration.status(arm_type="metal")
        ...     if status.awaiting_step:
        ...         print(status.message)  # relay to the human
        ...         # …the human poses the arm and confirms…
        ...         client.calibration.complete_step(step=status.step)
    """

    @operation("calibration_status")
    def status(self, arm_type: str | None = None) -> CalibrationStatus:
        """The live calibration status, from whichever flow is running.

        Pass ``arm_type`` to keep a terminal wizard result visible after the
        session released — without it, a finished run reads back as idle.

        Example:
            >>> client.calibration.status(arm_type="metal").status
            'awaiting_step'
        """
        params = {"arm_type": arm_type} if arm_type is not None else None
        return CalibrationStatus.model_validate(
            self._transport.request(
                "GET", "/api/v1/calibration-status", params=params, action="Get calibration status"
            )
        )

    @operation("complete_calibration_step")
    def complete_step(self, step: int | None = None) -> StepConfirmation:
        """Confirm the step currently on screen — the human has posed the arm.

        NEVER call this without a human having actually done the step. The
        confirmation is what tells the server the arm is in the pose it is
        about to record as zero; confirming an unposed arm writes a wrong
        calibration that then corrupts every later session silently.

        Pass ``step`` (from :meth:`status`) so a confirmation that arrives
        after the family moved on is refused rather than applied to the next
        step — with a multi-step family that would be a hardware write in the
        wrong pose.

        Example:
            >>> client.calibration.complete_step(step=1).success
            True
        """
        body: dict[str, Any] = {} if step is None else {"step": step}
        return StepConfirmation.model_validate(
            self._transport.request(
                "POST",
                "/api/v1/complete-calibration-step",
                json=body,
                action="Complete calibration step",
            )
        )

    @operation("auto_calibration_status")
    def auto_status(self) -> AutoCalibrationStatus:
        """The SINGLE-arm auto-calibration manager's state.

        A leased ``auto_calibration`` session routes through the batch
        manager — use :meth:`auto_batch_status` for one started that way.
        """
        return AutoCalibrationStatus.model_validate(
            self._transport.request(
                "GET", "/api/v1/auto-calibration-status", action="Get auto-calibration status"
            )
        )

    @operation("auto_calibration_batch_status")
    def auto_batch_status(self) -> AutoCalibrationBatchStatus:
        """Per-arm status and counts for a batch auto-calibration — the one a
        leased ``auto_calibration`` session reports through.

        The arm drives ITSELF here (a vendored subprocess under torque, which
        writes servo EEPROM); there is no human in the loop, so this is a
        plain poll-to-terminal. ``client.flows.auto_calibrate()`` wraps it.

        Example:
            >>> batch = client.calibration.auto_batch_status()
            >>> batch.completed, batch.failed, batch.total
            (2, 0, 2)
        """
        return AutoCalibrationBatchStatus.model_validate(
            self._transport.request(
                "GET",
                "/api/v1/auto-calibration-batch-status",
                action="Get batch auto-calibration status",
            )
        )

    @operation("get_calibration_configs")
    def configs(
        self, device_type: str, *, arm_type: str = "so101", leader_kind: str | None = None
    ) -> CalibrationConfigList:
        """The saved calibrations for one device slot.

        ``device_type`` is ``"robot"`` (follower) or ``"teleop"`` (leader).
        Libraries are per arm type and never merged, so ``arm_type`` picks
        which one — an unknown id refuses with ``robot.arm_type.unavailable``
        rather than falling back to the SO-101 library. ``leader_kind``
        selects among a family's leaders where it has more than one.

        List before calibrating: the step manager refuses a name that already
        exists unless the session was started with ``overwrite=True``.

        Example:
            >>> [c.name for c in client.calibration.configs("robot", arm_type="metal").configs]
            ['bench_follower']
        """
        params: dict[str, Any] = {"arm_type": arm_type}
        if leader_kind is not None:
            params["leader_kind"] = leader_kind
        return CalibrationConfigList.model_validate(
            self._transport.request(
                "GET",
                f"/api/v1/calibration-configs/{device_type}",
                params=params,
                action="List calibration configs",
            )
        )
