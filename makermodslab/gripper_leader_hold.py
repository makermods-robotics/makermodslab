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
"""Leader-side haptic gripper hold for a MOTORIZED Star Arm 102 leader.

A layer ON TOP OF Metal's follower-side gripper limiter (``gripper_soft_limit``):
when the operator is squeezing the leader gripper harder than the Metal follower
can physically close (an object in the follower's jaws), briefly energize the
*leader* gripper servo rigid so the operator feels the wall and can't stall the
follower by over-squeezing. Local teleop / record only, opt-in per robot record,
single-arm Metal only.

**Why a pulsed hold.** The rigid hold is immovable by hand (verified on the real
purple arm — see the ``reference_fashionstar_servo_hold_gotcha`` memory), so the
operator cannot signal "let me open now" against it. Instead the hold pulses:
~0.4 s rigid, then ~0.1 s released while we watch the operator's true hand
position. Pull the handle open during that window and the hold lets go; keep
squeezing and it re-engages. The pulsing itself is the "you're at the wall" cue.

**Frame consistency.** Every ``set_angle`` is ``multi_turn=True`` with a nonzero
``interval_ms``, and the hold target is read back with ``read_angle(...,
multi_turn=True)``. Mixing an instant ``multi_turn=False`` write with a
``multi_turn=True`` read does not engage holding torque AND poisons subsequent
reads so they echo the commanded value — a fake perfect hold. See the memory.

**Follower gripper convention (Metal):** ``gripper.pos`` is in motor degrees,
0 = closed, increasing = open. Over-squeezing means the operator commands a
*smaller* value than the follower reaches, so ``present - commanded`` grows.
"""

import logging
import math
import time

from .utils.config import get_robot_record, validate_gripper_leader_hold_gap

logger = logging.getLogger(__name__)

# Defaults, all overridable for tests. Tuned from the spike on the real arm:
# read round-trip ~5 ms, "a few Hz" re-assert holds rigid.
_ENGAGE_DEBOUNCE_S = 0.15
_HOLD_PULSE_S = 0.4
_PROBE_WINDOW_S = 0.1
_HOLD_INTERVAL_MS = 150
_ASSERT_HZ = 7.0
# Hysteresis: engage at gap_deg, release once the follower gets within this.
_RELEASE_GAP_DEG = 2.0
# Operator opening the handle this far during a probe releases the hold.
_PROBE_RELEASE_DEG = 3.0

_MONITORING = "monitoring"
_HOLDING = "holding"
_PROBING = "probing"


class GripperLeaderHold:
    """State machine driving one leader gripper servo. One instance per session.

    ``step(commanded, present)`` is called once per control action (both values
    in the follower's ``gripper.pos`` degree space); ``present`` may be ``None``
    when a fresh follower reading was not available this tick. ``release()`` is
    called on every teardown path — the family's leader torque release is a
    no-op on the motorless Star leader, so this is the only thing that unlocks
    the servo.
    """

    def __init__(
        self,
        servo_bus,
        gripper_id,
        gap_deg,
        *,
        release_gap_deg=_RELEASE_GAP_DEG,
        probe_release_deg=_PROBE_RELEASE_DEG,
        engage_debounce_s=_ENGAGE_DEBOUNCE_S,
        hold_pulse_s=_HOLD_PULSE_S,
        probe_window_s=_PROBE_WINDOW_S,
        hold_interval_ms=_HOLD_INTERVAL_MS,
        assert_hz=_ASSERT_HZ,
        clock=time.monotonic,
    ):
        self._bus = servo_bus
        self._gid = gripper_id
        self._gap_deg = gap_deg
        self._release_gap_deg = release_gap_deg
        self._probe_release_deg = probe_release_deg
        self._engage_debounce_s = engage_debounce_s
        self._hold_pulse_s = hold_pulse_s
        self._probe_window_s = probe_window_s
        self._hold_interval_ms = hold_interval_ms
        self._assert_period_s = 1.0 / assert_hz

        self._clock = clock
        self.phase = _MONITORING
        self._gap_since = None
        self._hold_raw = None
        self._hold_commanded = None
        self._last_assert = 0.0
        self._pulse_since = 0.0
        self._probe_since = 0.0

    # -- public API -------------------------------------------------------

    def step(self, commanded, present, now=None):
        if now is None:
            now = self._clock()
        if not math.isfinite(commanded) or (present is not None and not math.isfinite(present)):
            return

        if self.phase == _MONITORING:
            self._step_monitoring(commanded, present, now)
        elif self.phase == _HOLDING:
            self._step_holding(commanded, present, now)
        else:
            self._step_probing(commanded, present, now)

    def reset(self):
        """Drop any in-progress engage/hold without touching the servo.

        Used when the follower is not yet tracking the leader (its slow startup
        sync), where the position gap is expected and must not trip a hold.
        """
        if self.phase != _MONITORING:
            self._safe_unlock()
        self.phase = _MONITORING
        self._gap_since = None
        self._hold_raw = None
        self._hold_commanded = None

    def release(self):
        """Unlock the servo. Idempotent, never raises."""
        self._safe_unlock()
        self.phase = _MONITORING
        self._gap_since = None

    # -- phases ----------------------------------------------------------

    def _step_monitoring(self, commanded, present, now):
        if present is None:
            self._gap_since = None
            return
        if present - commanded >= self._gap_deg:
            if self._gap_since is None:
                self._gap_since = now
            elif now - self._gap_since >= self._engage_debounce_s:
                self._engage(commanded, now)
        else:
            self._gap_since = None

    def _step_holding(self, commanded, present, now):
        if present is not None and present - commanded <= self._release_gap_deg:
            self.release()
            return
        self._assert_hold(now)
        if now - self._pulse_since >= self._hold_pulse_s:
            self._safe_unlock()
            self.phase = _PROBING
            self._probe_since = now

    def _step_probing(self, commanded, present, now):
        # Leader gripper is limp here, so ``commanded`` is the operator's real hand.
        if commanded - self._hold_commanded >= self._probe_release_deg:
            self.release()
            return
        if present is not None and present - commanded <= self._release_gap_deg:
            self.release()
            return
        if now - self._probe_since >= self._probe_window_s:
            self._engage(self._hold_commanded, now)

    # -- helpers -------------------------------------------------------

    def _engage(self, commanded, now):
        try:
            self._hold_raw = float(self._bus.read_angle(self._gid, multi_turn=True).raw_deg)
        except Exception:
            logger.warning("Gripper leader hold: could not read the leader gripper angle", exc_info=True)
            self.release()
            return
        self._hold_commanded = commanded
        self.phase = _HOLDING
        self._pulse_since = now
        self._last_assert = -math.inf
        self._assert_hold(now)

    def _assert_hold(self, now):
        if now - self._last_assert < self._assert_period_s:
            return
        try:
            self._bus.set_angle(
                self._gid, self._hold_raw, multi_turn=True, interval_ms=self._hold_interval_ms
            )
        except Exception:
            logger.warning("Gripper leader hold: re-assert failed; releasing", exc_info=True)
            self.release()
            return
        self._last_assert = now

    def _safe_unlock(self):
        try:
            self._bus.unlock(self._gid)
        except Exception:
            logger.warning("Gripper leader hold: unlock failed", exc_info=True)


class _PresentGripper:
    """Rate-limited cache of the follower gripper's present ``gripper.pos``.

    The Damiao bus caches this from every MIT feedback frame, so a ``sync_read``
    is cheap, but the teleop loop runs far faster than the follower gripper
    moves — one read every ``min_interval_s`` is plenty and keeps the added bus
    traffic bounded. Any read failure yields ``None`` (the state machine then
    holds its position rather than acting on a stale gap).
    """

    def __init__(self, robot, *, clock, min_interval_s):
        self._robot = robot
        self._clock = clock
        self._min_interval_s = min_interval_s
        self._last_read = -math.inf
        self._value = None

    def read(self):
        now = self._clock()
        if now - self._last_read < self._min_interval_s:
            return self._value
        self._last_read = now
        try:
            self._value = float(self._robot.bus.sync_read("Present_Position")["gripper"])
        except Exception:
            logger.debug("Gripper leader hold: follower position read failed", exc_info=True)
            self._value = None
        return self._value


def install_gripper_leader_hold(
    robot, teleop_device, family, gap_deg, *, clock=time.monotonic, present_poll_s=0.05
):
    """Wrap ``robot.send_action`` so each control action drives the leader hold.

    Returns the controller (also stashed on ``robot._gripper_leader_hold`` for
    the teardown paths, which must call ``.release()`` — the arm family's leader
    torque release is a no-op on the motorless Star leader). Returns ``None``
    when the feature is disabled. Install AFTER ``install_gripper_soft_limit``
    so this wrapper sees the operator's raw gripper request, not the clamped one.
    Single-arm Metal only for now.
    """
    gap_deg = validate_gripper_leader_hold_gap(gap_deg)
    if gap_deg is None:
        return None
    if not getattr(family, "supports_gripper_leader_hold", False):
        raise ValueError("Gripper leader hold is only supported for the Metal arm")
    if hasattr(robot, "left_arm") or hasattr(robot, "right_arm") or hasattr(teleop_device, "left_arm"):
        raise ValueError("Gripper leader hold is not yet supported for bimanual arms")

    servo_bus = teleop_device.bus
    gripper_id = teleop_device.config.joint_ids["gripper"]
    controller = GripperLeaderHold(servo_bus, gripper_id, gap_deg, clock=clock)
    present = _PresentGripper(robot, clock=clock, min_interval_s=present_poll_s)
    original_send = robot.send_action

    def send_action(action, _send=original_send):
        sent = _send(action)
        try:
            if not getattr(robot, "_synced", True):
                # The follower's slow startup sync opens a position gap that is
                # not the operator squeezing anything — don't let it engage.
                controller.reset()
            else:
                commanded = action.get("gripper.pos")
                if commanded is None and isinstance(sent, dict):
                    commanded = sent.get("gripper.pos")
                if commanded is not None:
                    controller.step(float(commanded), present.read())
        except Exception:
            logger.warning("Gripper leader hold: step failed; releasing", exc_info=True)
            controller.release()
        return sent

    robot.send_action = send_action
    robot._gripper_leader_hold = controller
    return controller


def resolve_request_gripper_leader_hold(request):
    """Legacy named starts inherit the saved gap when the field is omitted."""
    from .api_errors import ApiError, ErrorCode
    from .arms import registry

    value = request.gripper_leader_hold_gap_deg
    if "gripper_leader_hold_gap_deg" not in request.model_fields_set and request.robot_name:
        record = get_robot_record(request.robot_name)
        if record is not None:
            value = record.get("gripper_leader_hold_gap_deg")
    try:
        value = validate_gripper_leader_hold_gap(value)
        if value is not None and not registry.get(request.arm_type).supports_gripper_leader_hold:
            raise ValueError("Gripper leader hold is only supported for the Metal arm")
    except ValueError as exc:
        raise ApiError(status_code=400, detail=str(exc), code=ErrorCode.REQUEST_VALIDATION) from exc
    request.gripper_leader_hold_gap_deg = value
