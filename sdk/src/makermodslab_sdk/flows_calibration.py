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
"""Calibration compositions over the session start and the status surface.

Two flows, because the two procedures differ in who does the work:

* :meth:`CalibrationFlows.auto_calibrate` — SO-101 auto-calibration. A
  vendored subprocess drives the arm under torque and writes servo EEPROM;
  nobody touches it. Composes ``sessions.auto_calibrate`` and
  ``calibration.auto_batch_status`` into a plain poll-to-terminal.
* :meth:`CalibrationFlows.calibrate_zero` — the CAN families' step wizard.
  The human poses the arm and confirms; the flow only relays. Composes
  ``robots.get``, ``sessions.calibrate``, ``calibration.status`` and
  ``calibration.complete_step``.

The SO-101 MANUAL sweep has no flow on purpose: the human sweeps every joint
while watching ``recorded_ranges`` fill in, which is a continuous visual loop
the web UI serves better. Use auto-calibration, or watch
``client.calibration.status()`` and narrate.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from makermodslab_sdk.errors import MakerModsError
from makermodslab_sdk.resources.calibration import (
    AutoCalibrationBatchStatus,
    CalibrationStatus,
)
from makermodslab_sdk.resources.sessions import EndedSessionInfo

if TYPE_CHECKING:
    from makermodslab_sdk.client import Client


class CalibrationFlowError(MakerModsError):
    """A calibration run failed, was refused, or ended without saving."""

    def __init__(self, message: str, *, session_id: str, robot: str) -> None:
        super().__init__(message)
        self.session_id = session_id
        self.robot = robot


class CalibrationFlowTimeout(CalibrationFlowError, TimeoutError):  # noqa: N818 - matches builtins.TimeoutError
    """The run outlived its budget; this flow's session was stopped on exit."""


class StepNotConfirmedError(CalibrationFlowError):
    """The ``confirm`` callback declined, so nothing was written.

    Declining is a valid answer — the arm was not in the pose, so confirming
    would have recorded a wrong zero. The session is stopped and no
    calibration is saved.
    """


@dataclass(frozen=True)
class AutoCalibrationFlowResult:
    """Every arm's outcome plus the matching session end."""

    session: EndedSessionInfo
    robot: str
    status: AutoCalibrationBatchStatus

    @property
    def completed(self) -> int:
        return self.status.completed

    @property
    def failed(self) -> int:
        return self.status.failed

    def summary(self) -> str:
        """One line per arm, naming what failed where."""
        lines = [f"{self.robot}: {self.completed} of {self.status.total} arms calibrated"]
        for arm in self.status.arms:
            detail = f" — {arm.error}" if arm.error else ""
            lines.append(f"- {arm.name or arm.port} [{arm.device_type}/{arm.arm}] {arm.status}{detail}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ZeroCalibrationResult:
    """The finished step wizard: its session end and terminal status."""

    session: EndedSessionInfo
    robot: str
    device_type: str
    status: CalibrationStatus
    steps_confirmed: int


class CalibrationFlows:
    """Calibration sequences; ``client.flows`` is the public entry."""

    def __init__(self, client: Client) -> None:
        self._client = client

    def auto_calibrate(
        self,
        robot: str,
        *,
        arms: list[dict[str, Any]],
        motor_power: int | None = None,
        overwrite: bool | None = None,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
        on_progress: Callable[[AutoCalibrationBatchStatus], None] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> AutoCalibrationFlowResult:
        """Run SO-101 auto-calibration to completion — no human in the loop.

        THE ARM MOVES ITSELF: a vendored subprocess drives it under torque and
        writes servo EEPROM. Make sure it has clear space before calling.

        Composes ``sessions.auto_calibrate`` (a leased session, so an abandoned
        process is safety-stopped) and ``calibration.auto_batch_status``.
        ``arms`` is the same 1-4 slot list the session takes, e.g.
        ``[{"device_type": "robot"}, {"device_type": "teleop"}]``; every slot
        runs concurrently. A per-arm failure does NOT raise — it lands in the
        result, because "the follower calibrated and the leader did not" is a
        real outcome an agent should report rather than a raise.

        Raises CalibrationFlowTimeout if the batch outlives ``timeout``;
        ``sleep_fn``/``clock`` are test seams.

        Example:
            >>> result = client.flows.auto_calibrate("bench", arms=[{"device_type": "robot"}])
            >>> print(result.summary())
            bench: 1 of 1 arms calibrated
            - bench_follower [robot/left] completed
        """
        if timeout < 0 or poll_interval <= 0:
            raise ValueError("timeout must be nonnegative and poll_interval positive")
        if not arms:
            raise ValueError("auto_calibrate needs at least one arm slot")

        started_at = clock()
        with self._client.sessions.auto_calibrate(
            robot, arms=arms, motor_power=motor_power, overwrite=overwrite
        ) as session:
            while True:
                status = self._client.calibration.auto_batch_status()
                if on_progress is not None:
                    on_progress(status)
                if status.finished:
                    elapsed = max(0.0, clock() - started_at)
                    ended = session.wait(
                        timeout=max(0.0, timeout - elapsed),
                        poll_interval=poll_interval,
                        sleep_fn=sleep_fn,
                        clock=clock,
                    )
                    return AutoCalibrationFlowResult(ended, robot, status)
                elapsed = max(0.0, clock() - started_at)
                if elapsed >= timeout:
                    raise CalibrationFlowTimeout(
                        f"Auto-calibration of {robot!r} is still running after {elapsed:g}s "
                        f"(timeout={timeout:g}); its session is being stopped. "
                        "Next step: client.calibration.auto_batch_status() shows each arm's logs.",
                        session_id=session.id,
                        robot=robot,
                    )
                sleep_fn(min(poll_interval, timeout - elapsed))

    def calibrate_zero(
        self,
        robot: str,
        *,
        device_type: str,
        confirm: Callable[[CalibrationStatus], bool],
        arm: str | None = None,
        port: str | None = None,
        config_file: str | None = None,
        overwrite: bool | None = None,
        timeout: float = 300.0,
        poll_interval: float = 1.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> ZeroCalibrationResult:
        """Run a CAN family's zero-pose wizard, relaying each step to a human.

        ``confirm`` IS REQUIRED AND HAS NO DEFAULT, deliberately. It receives
        the live status — ``status.message`` is the instruction to show, and
        ``status.image_url`` illustrates the pose — and must return True only
        once a human has PHYSICALLY put the arm in that pose. Confirming an
        unposed arm records a wrong zero, which then silently corrupts every
        session that uses this calibration. Returning False declines: the
        session stops, nothing is saved, and StepNotConfirmedError is raised.

        Composes ``robots.get``, ``sessions.calibrate``, ``calibration.status``
        and ``calibration.complete_step``. Torque stays off throughout (the arm is
        posed by hand), so there is nothing to return to rest. The Maker and
        Metal families publish exactly one step today; this loop handles more.

        ``timeout`` bounds the SERVER's work — connecting, applying each
        confirmed step, saving — and NOT the human: the clock is paused while
        ``confirm`` runs, so posing the arm may take as long as it takes.
        After every confirmed step the status is read again before any
        timeout verdict, so a step the server accepted and finished is never
        reported as a timeout. Raises CalibrationFlowTimeout (and stops the
        session) once the server's own time exceeds ``timeout``;
        ``sleep_fn``/``clock`` are test seams.

        Example:
            >>> def ask(status):
            ...     print(status.message)  # "Fold the arm and close the gripper…"
            ...     return input("done? ") == "y"
            >>> result = client.flows.calibrate_zero("bench", device_type="robot", confirm=ask)
            >>> result.status.status
            'completed'
        """
        if timeout < 0 or poll_interval <= 0:
            raise ValueError("timeout must be nonnegative and poll_interval positive")

        started_at = clock()
        # Time spent inside confirm() — the human posing the arm — is not
        # charged against ``timeout``, which bounds the server's work only.
        human_time = 0.0
        confirmed = 0
        last_step: int | None = None
        # The wizard's session releases the moment the zero is written, and
        # the server reads a released wizard back as idle unless the status
        # call names the family — without it this loop never sees "completed".
        arm_type = self._client.robots.get(robot).arm_type
        with self._client.sessions.calibrate(
            robot,
            device_type=device_type,
            arm=arm,
            port=port,
            config_file=config_file,
            overwrite=overwrite,
        ) as session:
            while True:
                status = self._client.calibration.status(arm_type=arm_type)
                if status.finished:
                    if status.status == "error":
                        raise CalibrationFlowError(
                            f"Calibration of {robot!r} ({device_type}) failed: "
                            f"{status.error or status.message or 'no detail'}. "
                            "Next step: client.calibration.status() holds the terminal detail.",
                            session_id=session.id,
                            robot=robot,
                        )
                    elapsed = max(0.0, clock() - started_at - human_time)
                    ended = session.wait(
                        timeout=max(0.0, timeout - elapsed),
                        poll_interval=poll_interval,
                        sleep_fn=sleep_fn,
                        clock=clock,
                    )
                    return ZeroCalibrationResult(ended, robot, device_type, status, confirmed)
                if status.awaiting_step and status.step != last_step:
                    asked_at = clock()
                    posed = confirm(status)
                    human_time += max(0.0, clock() - asked_at)
                    if not posed:
                        raise StepNotConfirmedError(
                            f"Step {status.step} of {robot!r} ({device_type}) was declined, so no "
                            "calibration was written and the session is being stopped. "
                            "Next step: re-run once the arm can be posed as the step describes.",
                            session_id=session.id,
                            robot=robot,
                        )
                    answer = self._client.calibration.complete_step(step=status.step)
                    if not answer.success:
                        raise CalibrationFlowError(
                            f"Step {status.step} of {robot!r} was refused: {answer.message}. "
                            "Next step: client.calibration.status() shows the phase that accepts "
                            "a confirmation.",
                            session_id=session.id,
                            robot=robot,
                        )
                    confirmed += 1
                    last_step = status.step
                    # Re-read before judging the budget: the step may already
                    # have finished the wizard, and a timeout must quote the
                    # status the server holds now, not the pre-confirm one.
                    continue
                elapsed = max(0.0, clock() - started_at - human_time)
                if elapsed >= timeout:
                    raise CalibrationFlowTimeout(
                        f"Calibration of {robot!r} is still {status.status!r} after {elapsed:g}s "
                        f"(timeout={timeout:g}); its session is being stopped. "
                        "Next step: client.calibration.status() shows the step it was waiting on.",
                        session_id=session.id,
                        robot=robot,
                    )
                sleep_fn(min(poll_interval, timeout - elapsed))
