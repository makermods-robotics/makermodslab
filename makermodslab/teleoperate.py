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

import contextlib
import logging
import math
import threading
import time
from typing import Any

from pydantic import BaseModel

from lerobot.robots.bi_so_follower import BiSOFollower
from lerobot.robots.so_follower import SO101Follower
from lerobot.teleoperators.bi_so_leader import BiSOLeader
from lerobot.teleoperators.so_leader import SO101Leader
from lerobot.utils.errors import DeviceNotConnectedError

from .arm_identity import verify_devices
from .motor_power import clear_goal_velocity, reset_torque_limit
from .rest_pose import RETURN_CEILING_S, capture_rest_pose, return_to_rest_pose
from .utils.devices import _force_close_device_resources
from .utils.errors import classify_outcome, format_exception, friendly_hint
from .utils.robot_factory import build_bimanual_configs, build_single_configs

logger = logging.getLogger(__name__)

# sts3215 motor resolution; lerobot's DEGREES normalization uses (resolution - 1).
_STS3215_MAX_RES = 4095

# General URDF mapping. Rather than chase a per-joint "zero tick" (fragile — the
# URDF's zero pose isn't reachable/meaningful for every joint), affinely map each
# motor's *calibrated* travel [range_min, range_max] onto its URDF joint limits
# [lower, upper]. Both ends are known per arm: the motor range comes from
# calibration, the limits from so101_new_calib.urdf. So the on-screen model
# tracks the real arm across its full range, with no clamping and no per-arm tick
# constants. The only per-joint fact that can't be derived is `sign`: whether
# motor-increasing maps to URDF-increasing (+1) or -decreasing (-1). Limits are
# in radians (URDF units); the gripper is RANGE_0_100 (0-100), not degrees.
_SO101_URDF_JOINTS = {
    # motor_name: (urdf_joint, lower_rad, upper_rad, sign)
    "shoulder_pan": ("Rotation", -1.91986, 1.91986, +1),
    "shoulder_lift": ("Pitch", -1.74533, 1.74533, +1),
    "elbow_flex": ("Elbow", -1.74533, 1.57080, +1),
    "wrist_flex": ("Wrist_Pitch", -1.65806, 1.65806, +1),
    "wrist_roll": ("Wrist_Roll", -2.79253, 2.79253, +1),
    "gripper": ("Jaw", -0.174533, 1.74533, +1),
}


def _motor_fraction(motor_name: str, value: float, cal) -> float | None:
    """Position of ``value`` within the motor's calibrated travel, as a 0..1 fraction.

    Returns ``None`` when a body (DEGREES) joint has no calibration to define its
    range, so the caller can fall back instead of guessing.
    """
    if motor_name == "gripper":
        # RANGE_0_100 norm mode: the observation already is a 0-100 percentage.
        return value / 100.0
    if cal is None:
        return None
    # DEGREES norm mode: value = (ticks - mid) * 360 / max_res, i.e. symmetric
    # about 0 across the calibrated range, so 0.5 sits at the midpoint.
    full_range_deg = (cal.range_max - cal.range_min) * 360.0 / _STS3215_MAX_RES
    if full_range_deg <= 0:
        return None
    return 0.5 + value / full_range_deg


# Global variables for teleoperation state
teleoperation_active = False
teleoperation_thread: threading.Thread | None = None
current_robot = None
current_teleop = None
# Set by the worker's cleanup when torque disable/disconnect failed, i.e. an arm
# may be left energized (rigid). Cleared on start; surfaced in the stop response
# and the /teleoperation-status payload so the frontend can warn the user.
last_cleanup_error: str | None = None
# Terminal error taxonomy of the most recent session (the in-process twin of
# rollout's exited payload; complements last_cleanup_error, which stays the
# raw "torque may still be enabled" safety text). `last_session_outcome` is
# "ok" | "ran_with_warning" | "failed" (None before any session ends);
# `last_session_error` is the error behind it — the caught loop exception
# formatted as "Type: message" (the worker's catch site holds the actual
# exception object; no log forensics), or the cleanup problem text when only
# teardown tripped. Classified by catch site: a mid-loop exception is a failed
# session; a user-requested stop whose cleanup complained (e.g. a gripper
# overload on torque disable) ran fine — a warning. Cleared on start.
last_session_outcome: str | None = None
last_session_error: str | None = None
# Guards the start path; the worker owns disconnect so stop() does not race.
_state_lock = threading.Lock()

# Grace period between a *user-initiated* stop and the torque release, used by
# RECORDING only (makermodslab/record.py imports hold_torque_release_grace; recording
# has no return-to-start — its last pose may be deliberate — so the timed hold
# gives the operator a window to get a hand on the arm). Teleoperation stops
# no longer hold: the servos already hold their last goal on their own in
# position mode, so the stop drives the follower straight back to its
# session-start pose (see makermodslab/rest_pose.py) — same behavior as the
# auto-calibration stop — then releases. Error paths (an exception in the
# control loop — e.g. an unplugged bus) skip everything and attempt the
# release immediately: you can't hold what you can't reach.
TORQUE_RELEASE_GRACE_S = 5.0
# True while the worker is driving the follower back to its rest pose (and on
# through the release). The session is no longer "active" (the control loop
# has exited) but the serial ports are still held and the arms are still
# energized — surfaced in the status payload so the UI isn't lying about the
# arm's state.
releasing = False
# Cuts the post-stop return short: set by a second stop request ("release
# now") or by a new start request that needs the serial ports. Cleared on start.
_release_now = threading.Event()

# --- Follower power telemetry ---
# Present_Current (register 69, 2 bytes, read-only; absent from the pinned
# sign-magnitude encoding table, so reads come back as plain magnitudes) is
# the servo's MEASURED winding current, 6.5 mA per LSB per Feetech's STS3215
# memory table — the honest signal for validating the motor-power cap.
# Present_Load (register 60) is a signed PWM duty in 0.1% units, i.e.
# commanded effort, not measured draw — deliberately not used. Sampled at
# ~1 Hz inside the teleop loop: one sync_read per second per FOLLOWER bus,
# piggybacked on the broadcast tick so it adds no extra loop overhead. If a
# firmware revision leaves the register at 0, the summary shows all-zero
# peaks — that is the register being unpopulated, not zero current.
_PRESENT_CURRENT_REGISTER = "Present_Current"
_CURRENT_MA_PER_UNIT = 6.5
_CURRENT_SAMPLE_INTERVAL_S = 1.0


class PowerTelemetry:
    """Per-motor peak/mean Present_Current (mA) across a teleop session.

    Gives an objective A/B for the motor-power cap: run once at 100% and once
    at 30% against the same manual resistance and compare the logged peaks —
    no subjective feel test needed. Sampling and summarizing never raise.
    """

    def __init__(self) -> None:
        self.peak_ma: dict[str, float] = {}
        self.latest_ma: dict[str, float] = {}
        self._sum_ma: dict[str, float] = {}
        self._n: dict[str, int] = {}

    def sample(self, bus, prefix: str = "") -> None:
        """One Present_Current sync_read on a follower bus; never raises."""
        try:
            raw = bus.sync_read(_PRESENT_CURRENT_REGISTER, normalize=False)
        except Exception as e:
            logger.debug(f"Power telemetry sample failed: {e}")
            return
        for motor, value in raw.items():
            ma = abs(float(value)) * _CURRENT_MA_PER_UNIT
            key = f"{prefix}{motor}"
            self.latest_ma[key] = round(ma, 1)
            self.peak_ma[key] = max(self.peak_ma.get(key, 0.0), ma)
            self._sum_ma[key] = self._sum_ma.get(key, 0.0) + ma
            self._n[key] = self._n.get(key, 0) + 1

    def summary(self) -> str | None:
        """One INFO-ready line of per-motor peaks/means, or None if no samples."""
        if not self._n:
            return None
        parts = [
            f"{motor} peak {self.peak_ma[motor]:.0f}mA / mean {self._sum_ma[motor] / self._n[motor]:.0f}mA"
            for motor in self.peak_ma
        ]
        return f"power telemetry: {'; '.join(parts)}"


def hold_torque_release_grace(
    release_now: threading.Event,
    grace_s: float = TORQUE_RELEASE_GRACE_S,
    label: str = "arms",
) -> bool:
    """Keep the arm(s) energized for up to ``grace_s`` seconds before release.

    "Holding" is just *not disabling torque yet* — the servos hold their last
    goal on their own in position mode. Waits on ``release_now`` so a second
    stop press (or a new start that needs the ports) can cut the hold short.
    Returns True when cut short, False when the full grace elapsed.

    This only *delays* the release; the caller must still run its
    force_disable_torque + disconnect cleanup afterwards, unconditionally.
    """
    logger.info(
        "Holding torque on the %s for up to %.0f s — guide the arm to a rest position "
        "(stop again to release now)",
        label,
        grace_s,
    )
    cut_short = release_now.wait(timeout=grace_s)
    logger.info("Grace hold on the %s finished: %s", label, "release-now" if cut_short else "elapsed")
    return cut_short


def finish_pending_release(timeout: float = 10.0) -> bool:
    """Cut a pending torque-release grace short and wait for its cleanup.

    Called by start paths (teleoperation and recording) so a start arriving
    during the grace window releases the arms and frees the serial ports
    immediately instead of failing port-busy for the rest of the grace.
    Returns True when no teleoperation worker is running afterwards (the ports
    are free as far as teleoperation is concerned); False when a live session
    is running or the worker did not exit in time.
    """
    global teleoperation_thread

    worker = teleoperation_thread
    if worker is None or not worker.is_alive():
        return True
    if teleoperation_active:
        # A live session, not a pending release — the caller's mutex check
        # will report "already active".
        return False
    _release_now.set()
    worker.join(timeout=timeout)
    if worker.is_alive():
        return False
    teleoperation_thread = None
    return True


# Bound on how long stop_and_wait() waits for the graceful return-to-rest +
# torque release to finish on its own before forcing an immediate release.
# return_to_rest_pose's own RETURN_CEILING_S already bounds the return motion
# itself (and bimanual arms return concurrently, not sequentially — see
# _return_followers_to_rest — so this doesn't need to account for two arms in
# series); the margin covers the torque-release writes, disconnect, and
# thread-join overhead that follow it.
_STOP_AND_WAIT_TIMEOUT_S = RETURN_CEILING_S + 5.0


def stop_and_wait(timeout: float = _STOP_AND_WAIT_TIMEOUT_S) -> None:
    """Stop teleoperation and block until the worker has actually released
    the arm(s), for a caller with no UI to poll and no "press Stop again"
    gesture available (shutdown).

    handle_stop_teleoperation()'s first call is deliberately fire-and-forget:
    it flips teleoperation_active and returns immediately so the request
    thread isn't blocked on the return-to-rest motion, relying on a status
    poll or a second Stop press to observe or force completion. Neither
    exists at shutdown. This composes both: trigger the stop, then join the
    worker — giving it its normal grace to finish the graceful return-to-rest
    and release on its own, and only forcing an immediate release (the same
    "second stop" mechanism a human pressing Stop twice would trigger) if it
    hasn't finished within the bound. A no-op when idle.
    """
    if teleoperation_active:
        handle_stop_teleoperation()
    worker = teleoperation_thread
    if worker is None or not worker.is_alive():
        return
    worker.join(timeout=timeout)
    if not worker.is_alive():
        return
    logger.warning(
        "Teleoperation worker did not finish its graceful release within %.0fs; forcing release now",
        timeout,
    )
    _release_now.set()
    worker.join(timeout=5.0)
    if worker.is_alive():
        logger.warning("Teleoperation worker still alive after forcing release; giving up the wait")


def _return_one_follower_to_rest(bus, pose: dict, abort_event: threading.Event) -> None:
    """Drive one follower bus back to its captured pose; log start and outcome.

    The per-arm body of _return_followers_to_rest, run on its own thread so
    the two bimanual followers return concurrently. return_to_rest_pose never
    raises, but guard anyway so one arm's failure can never take down the
    thread (and thus block its join) before the outcome is logged.
    """
    port = getattr(bus, "port", None) or "unknown port"
    label = f"follower arm on {port}"
    logger.info(f"Rest-pose return starting for the {label}")
    try:
        _arrived, reason = return_to_rest_pose(bus, pose, abort_event=abort_event, label=label)
        logger.info(f"Rest-pose return finished for the {label}: {reason}")
    except Exception as e:
        # return_to_rest_pose is documented never-raises; this is belt-and-braces
        # so a surprise failure on one arm can't prevent the other's thread from
        # being joined or the wrapper from returning to run the torque release.
        logger.warning(f"Rest-pose return errored for the {label}: {e}")


def _return_followers_to_rest(rest_poses: list[tuple], abort_event: threading.Event) -> None:
    """Drive every follower bus back to its captured session-start pose, at once.

    Runs immediately before the torque release on a NORMAL stop only (no timed
    hold: the servos hold their last goal on their own until the return goals
    land). Best-effort: each bus's outcome is logged (returned | settled |
    stalled | ceiling | cut-short | no-pose | comm-error) and every failure
    falls through to the unconditional torque release. NEVER called with a
    leader bus — the leader is human-held with torque off; driving it would
    fight the user's hand.

    Each follower is its own serial bus on its own USB port (no shared bus),
    so the returns run CONCURRENTLY: one thread per (bus, pose), all joined
    before this returns. Bimanual arms therefore land at the same time instead
    of one-after-the-other. The shared ``abort_event`` (a second stop /
    release-now) cuts every arm's return short promptly; the wrapper still
    returns only after all per-arm threads have wound down, because the
    downstream torque-release ordering depends on this having finished. A
    single-arm session is the same shape — one thread, joined — preserving the
    existing single-arm timing and semantics.
    """
    threads = [
        threading.Thread(
            target=_return_one_follower_to_rest,
            args=(bus, pose, abort_event),
            name=f"rest-return-{getattr(bus, 'port', None) or i}",
            daemon=True,
        )
        for i, (bus, pose) in enumerate(rest_poses)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


class TeleoperateRequest(BaseModel):
    leader_port: str
    follower_port: str
    leader_config: str
    follower_config: str
    # Bimanual: the primary pair above is the LEFT arm; these add the right arm.
    mode: str = "single"
    right_leader_port: str = ""
    right_follower_port: str = ""
    right_leader_config: str = ""
    right_follower_config: str = ""
    # Robot record name — used only as the BiSO staging base id (bimanual). It
    # decides the on-disk staging dir, not which calibration drives which arm.
    # Blank/invalid falls back to DEFAULT_BIMANUAL_BASE.
    robot_name: str = ""
    # Escape hatch for the arm-identity guard (see makermodslab/arm_identity.py):
    # when true, start even if the connected arms don't match their calibrations.
    skip_identity_check: bool = False


def get_joint_positions_from_robot(robot, prefix: str = "", calibration=None) -> dict[str, float]:
    """
    Extract current joint positions from the robot and convert to URDF joint format.

    Args:
        robot: The robot instance (SO101Follower, or a BiSO* arm).
        prefix: Motor-key prefix in the observation. "" for single-arm; for a
            bimanual BiSO robot pass "left_"/"right_" to pull one arm.
        calibration: Calibration dict to use for the URDF correction. Defaults to
            ``robot.calibration``; for a BiSO robot pass the sub-arm's calibration.

    Returns:
        Dictionary mapping URDF joint names to radian values
    """
    try:
        observation = robot.get_observation()
        calibration = calibration if calibration is not None else (getattr(robot, "calibration", None) or {})

        joint_positions: dict[str, float] = {}
        debug_rows = []
        for motor_name, (urdf_joint_name, lower, upper, sign) in _SO101_URDF_JOINTS.items():
            # Single-arm uses "<motor>.pos"; a bimanual BiSO robot prefixes both
            # arms ("left_<motor>.pos"/"right_<motor>.pos"). Callers pass the
            # prefix for bimanual; when unset, fall back to the left arm so a
            # single-arm caller still gets something from a BiSO robot.
            motor_key = f"{prefix}{motor_name}.pos"
            if motor_key not in observation and not prefix and f"left_{motor_name}.pos" in observation:
                motor_key = f"left_{motor_name}.pos"
            if motor_key not in observation:
                logger.warning(f"Motor {motor_key} not found in observation")
                joint_positions[urdf_joint_name] = 0.0
                continue

            value = observation[motor_key]
            frac = _motor_fraction(motor_name, value, calibration.get(motor_name))
            if frac is None:
                # No calibration for a DEGREES joint — render the raw normalized
                # value as degrees so we still show *something* (uncalibrated).
                urdf_rad = value * math.pi / 180.0
            else:
                # Affinely map the calibrated range onto the URDF limits; the
                # clamp guards the rare case of driving slightly past calibration.
                frac = min(1.0, max(0.0, frac))
                if sign < 0:
                    frac = 1.0 - frac
                urdf_rad = lower + frac * (upper - lower)

            joint_positions[urdf_joint_name] = urdf_rad
            debug_rows.append(
                f"{motor_name:14s} raw={value:+8.2f} → {urdf_joint_name:11s} = {urdf_rad * 180.0 / math.pi:+8.2f}°"
            )

        # Throttled debug print (~once per second at 20 Hz broadcast).
        now = time.time()
        if now - getattr(get_joint_positions_from_robot, "_last_log", 0) > 1.0:
            get_joint_positions_from_robot._last_log = now
            logger.info("[joint-debug]\n  " + "\n  ".join(debug_rows))

        return joint_positions

    except Exception as e:
        logger.error(f"Error getting joint positions: {e}")
        return {urdf[0]: 0.0 for urdf in _SO101_URDF_JOINTS.values()}


def _device_buses(device) -> list:
    """The motor bus(es) of a robot/teleop device.

    A single-arm device exposes ``.bus``; a bimanual BiSO device exposes
    ``left_arm``/``right_arm`` sub-arms which each carry their own bus.
    """
    if device is None:
        return []
    arms = [
        arm
        for arm in (getattr(device, "left_arm", None), getattr(device, "right_arm", None))
        if arm is not None
    ]
    targets = arms if arms else [device]
    return [target.bus for target in targets if getattr(target, "bus", None) is not None]


def _device_ports(device) -> str:
    """Comma-separated serial port(s) of a device, for error messages."""
    ports = [str(bus.port) for bus in _device_buses(device) if getattr(bus, "port", None)]
    return ", ".join(ports) if ports else "unknown port"


def _bus_has_a_responding_motor(bus) -> bool:
    """True when at least one motor on ``bus`` answers a ping.

    Used only to choose the *wording* of the alarm after the torque-disable
    pass below has already failed on every motor on this bus — it never gates
    whether that pass runs. A degraded-but-recoverable bus (a brownout that
    recovers, EMI, a long cable, the moments right after the control loop died
    on a comms error) can fail every zero-retry ping while a retried write
    would have landed, so the probe must not be allowed to veto the write.

    ``is_connected`` can't stand in for this: it's literally
    ``port_handler.is_open`` (lerobot ``motors_bus.py``), and
    ``MotorsBus._connect`` calls ``openPort()`` *before* ``_handshake()`` and
    does not close the port when the handshake fails — so an unpowered,
    browned-out, or wrong-baud arm leaves ``is_connected`` True on a bus that
    no motor is listening on.

    ``ping`` is the right probe: it is a read (so it works whatever the torque
    state is) and it returns ``None`` rather than raising on comm failure.
    Short-circuits on the first motor that answers.

    Defaults to True (assume the bus is alive, keep the loud alarm) for
    anything that can't be probed: a bus with no ``motors``, a bus without a
    ``ping`` method, or a test double that models neither.
    """
    motors = getattr(bus, "motors", None) or {}
    ping = getattr(bus, "ping", None)
    if not motors or not callable(ping):
        return True
    for motor in motors:
        try:
            if ping(motor) is not None:
                return True
        except Exception:
            continue
    return False


def force_disable_torque(device, label: str = "device") -> list[str]:
    """Explicitly disable torque on every motor of a device, motor by motor.

    Belt-and-braces step to run *before* ``device.disconnect()``. lerobot's
    disconnect does disable torque itself, but any exception on the way there
    leaves the arm energized: one motor's failed write aborts the disable for
    all remaining motors (and skips closing the port), and the error is easy
    to swallow on a cleanup path. Going motor by motor means one bad motor
    can't leave the other joints locked.

    Returns a list of problem descriptions — empty when torque was disabled on
    every motor. Each problem is also logged at ERROR level naming the port.
    """
    problems: list[str] = []
    for bus in _device_buses(device):
        # A bus whose port never opened at all (the device node was missing or
        # busy, so openPort() itself raised) has nothing to disable — writing
        # to a closed port just produces a false "TORQUE MAY STILL BE ENABLED"
        # alarm. Note this covers *only* that case: is_connected is
        # port_handler.is_open, so it stays True when the port opened and the
        # handshake then failed. The unpowered/wrong-baud arm is caught by the
        # liveness probe below, not here. Test doubles that don't model
        # connection state default to True so torque-only tests are unaffected.
        if not getattr(bus, "is_connected", True):
            continue
        # The Dynamixel SDK port handler can be left flagged "in use" after a
        # failed read/write in the control loop. LeRobot's normal
        # bus.disconnect() clears this before disabling torque; mirror that here
        # because this belt-and-braces path deliberately bypasses disconnect()
        # to disable motors one by one.
        port_handler = getattr(bus, "port_handler", None)
        if port_handler is not None:
            with contextlib.suppress(Exception):
                port_handler.clearPort()
            with contextlib.suppress(Exception):
                port_handler.is_using = False
        port = getattr(bus, "port", None) or "unknown port"
        # Always attempt the disable, whatever the liveness probe below would
        # say: a degraded-but-recoverable bus can fail every zero-retry ping
        # while the retried write here still lands, and skipping the write on
        # a failed probe would leave a genuinely energized arm rigid. The
        # probe is only consulted after a failure, to pick the wording.
        failed: list[str] = []
        for motor in getattr(bus, "motors", None) or {}:
            try:
                bus.disable_torque(motor, num_retry=5)
            except Exception as e:
                failed.append(f"{motor}: {e}")
        if failed:
            if _bus_has_a_responding_motor(bus):
                # Something is listening, so the failed writes are a real
                # "this joint may stay rigid" condition.
                message = (
                    f"TORQUE MAY STILL BE ENABLED on {port} ({label}; failed motors — {'; '.join(failed)}). "
                    "The arm can stay rigid; unplug its power to release it."
                )
                logger.error(message)
            else:
                # Nothing on this bus is listening: the port opened but no
                # motor answers (arm unpowered, browned-out servos, wrong
                # baud, or a valid-but-wrong serial device). Reporting that as
                # "TORQUE MAY STILL BE ENABLED — unplug its power" is actively
                # misleading on an arm that has no power. Say what was
                # actually observed, and keep the rigid-arm advice as a
                # conditional rather than an assertion.
                message = (
                    f"No motor answered on {port} ({label}) — the torque disable failed on every motor. "
                    "Check the arm's power and USB cable. If the arm is rigid, unplug its power to release it."
                )
                logger.warning(message)
            problems.append(message)
    return problems


def _safe_disconnect(device, label: str = "device") -> str | None:
    """Disconnect a robot/teleop device, swallowing (but logging) any error.

    Used on cleanup paths so one device's failure can't leave the other
    holding its serial port open. Returns an error message when the disconnect
    failed — lerobot's disconnect is also what releases motor torque, so a
    failure here can leave the arm energized (rigid) — or None on success.
    """
    if device is None:
        return None
    try:
        device.disconnect()
        return None
    except Exception as e:
        message = (
            f"Failed to disconnect the {label} ({_device_ports(device)}): {e}. "
            "TORQUE MAY STILL BE ENABLED — the arm can stay rigid; unplug its power to release it."
        )
        logger.error(message)
        # Last resort (upstream): force-close the serial port(s) so a failed
        # disconnect can't leave the port handle open until process exit.
        # Buses live on sub-arms for bimanual BiSO devices, so hit those too.
        for target in (device, getattr(device, "left_arm", None), getattr(device, "right_arm", None)):
            if target is not None:
                _force_close_device_resources(target, logger)
        return message


def _cleanup_after_setup_failure(
    robot, teleop_device, robot_label: str = "follower arm", teleop_label: str = "leader arm"
) -> str | None:
    """Force-disable torque and disconnect both devices after a setup failure.

    Shared by the single-arm and bimanual setup paths: either device may
    already be configured (energized) by the time a *later* setup step
    raises, so skipping the explicit disable — or dropping _safe_disconnect's
    error text — would leave an arm energized with no warning surfaced.
    Returns the joined problem text, or None when cleanup was clean.
    """
    problems = force_disable_torque(robot, robot_label)
    problems += force_disable_torque(teleop_device, teleop_label)
    for device, label in ((robot, robot_label), (teleop_device, teleop_label)):
        error = _safe_disconnect(device, label)
        if error:
            problems.append(error)
    return " ".join(problems) if problems else None


def force_disconnect_partial(device, label: str = "device") -> list[str]:
    """Release a device that may be only *partially* connected.

    ``device.disconnect()`` is unusable after a failed ``connect()``. lerobot
    guards it with ``@check_if_not_connected``, and a robot's ``is_connected``
    is all-or-nothing (``bus.is_connected and all(cam.is_connected ...)``), so
    when ``connect()`` opens the bus and then dies on a camera, the cameras
    *after* the failing one in iteration order were never opened — the guard
    sees ``is_connected == False`` and ``disconnect()`` raises
    ``DeviceNotConnectedError`` immediately, releasing **nothing**. The bus
    stays open and every already-opened camera keeps its background read thread
    running, holding the OS device busy for the rest of the process. The next
    connect attempt then dies on the leaked bus with the misleading
    "FeetechMotorsBus is already connected", and its cameras time out against
    the leaked read threads — the failure is self-perpetuating until restart
    (diagnosed 2026-07-29 on the front/wrist rig).

    Go component by component instead, each independently guarded, so a broken
    camera can't strand the bus and vice versa. Safe on a fully connected,
    partially connected, or already disconnected device.

    Torque, motor-by-motor, before any of that: on a camera failure torque was
    never enabled (configure() runs after the cameras), but a failure later in
    connect() can leave it on. lerobot's bus.disconnect() does disable torque
    itself, but — same reasoning as force_disable_torque's docstring — one
    motor's failed write there aborts the disable for every motor after it on
    that bus, leaving them energized *and* skipping closePort(). So
    force_disable_torque disables each motor independently first, and the
    disconnect() below is then called with disable_torque=False: the work is
    already done, and re-running it only reintroduces the abort-before-close
    failure mode. Same ordering as _cleanup_after_setup_failure uses for the
    same "torque may already be enabled after an incomplete setup" situation.
    force_disable_torque skips any bus whose port never opened and any bus
    where no motor answers a ping, so this is safe to call unconditionally.

    Returns the list of problem descriptions, empty when teardown was clean,
    so callers can surface a partial teardown instead of silently discarding
    it. Note this is the raw list, unlike _cleanup_after_setup_failure, which
    joins its problems into a single ``str | None`` for its one caller —
    leaving them separate here lets a caller count, filter, or join them as
    it needs.
    """
    problems: list[str] = []
    if device is None:
        return problems
    problems += force_disable_torque(device, label)
    # Cameras next: they're the usual point of failure, and their read threads
    # are what keep the device busy for the next attempt. ``.cameras`` covers
    # both sides on a bimanual BiSO device (it merges the sub-arms' dicts).
    for name, cam in (getattr(device, "cameras", None) or {}).items():
        try:
            cam.disconnect()
        except DeviceNotConnectedError:
            pass  # never opened (or already released) — nothing to do
        except Exception as e:
            message = f"Could not release {label} camera {name}: {e}"
            logger.warning(message)
            problems.append(message)
    for bus in _device_buses(device):
        try:
            # disable_torque=False: force_disable_torque above already
            # disabled every motor independently, and lerobot's default
            # disconnect(disable_torque=True) would re-run the same pass with
            # num_retry=5 — where the *first* unresponsive motor raises before
            # closePort(), leaking the port and appending a second, misleading
            # "could not release the bus" problem about a bus we then close by
            # hand below. Same reasoning and same call as rollout.py,
            # motor_power.py, identify.py and auto_calibrate.py.
            if getattr(bus, "is_connected", True):
                bus.disconnect(disable_torque=False)
        except Exception as e:
            port = getattr(bus, "port", None) or "unknown port"
            message = f"Could not release {label} bus on {port}: {e}"
            logger.warning(message)
            problems.append(message)
            # Last resort (same fallback utils/devices.py's
            # _force_close_device_resources uses): force the port handle
            # closed so a failed disconnect can't leave it open until process
            # exit. Scoped per-bus (not delegated to that device-level
            # helper) so bimanual sub-arm buses each get their own fallback.
            port_handler = getattr(bus, "port_handler", None)
            if port_handler is not None:
                with contextlib.suppress(Exception):
                    port_handler.clearPort()
                with contextlib.suppress(Exception):
                    port_handler.is_using = False
                try:
                    port_handler.closePort()
                    logger.info(f"Force-closed {label} bus port on {port} after disconnect failure")
                except Exception as close_exc:
                    close_message = f"Failed to force-close {label} bus port on {port}: {close_exc}"
                    logger.warning(close_message)
                    problems.append(close_message)
    return problems


def _connect_bimanual(request: TeleoperateRequest):
    """Build, connect, and configure a bimanual leader+follower pair.

    Each side is a lerobot BiSO* device wrapping two SO101 arms (left = the
    primary leader/follower pair, right = the right_* pair). lerobot loads each
    sub-arm's calibration as "<base>_left/right.json" from a single dir, with no
    way to point left/right at differently named library files, so we stage the
    four arbitrarily-named library calibrations into per-device dirs under that
    convention and point BiSO at those. Returns
    (robot, teleop_device, identity_warnings) connected, or raises after
    disconnecting any device. The staging copy fails fast with a clear per-slot
    error if any library file is missing (before connect() drops into
    interactive recalibration, which would hang this thread).
    """
    robot_config, teleop_config = build_bimanual_configs(request)

    robot = BiSOFollower(robot_config)
    teleop_device = BiSOLeader(teleop_config)

    try:
        # Connect each of the four buses, naming the one that fails.
        for arm, label, port in (
            (robot.left_arm, "left follower", request.follower_port),
            (robot.right_arm, "right follower", request.right_follower_port),
            (teleop_device.left_arm, "left leader", request.leader_port),
            (teleop_device.right_arm, "right leader", request.right_leader_port),
        ):
            try:
                arm.bus.connect()
            except Exception as e:
                raise RuntimeError(
                    f"Could not connect to the {label} arm on {port}. "
                    "Make sure it's plugged in and powered on, then try again."
                ) from e

        # Arm-identity guard: all four arms, read-only, BEFORE write_calibration
        # below can stamp a wrong file into a swapped arm's EEPROM. The sub-arm
        # ids are BiSO staging aliases, so pass the real library stems (in
        # arm-iteration order: left follower, right follower, left leader, right
        # leader) for the identity comparison.
        identity_warnings = verify_devices(
            ((robot, "follower"), (teleop_device, "leader")),
            skip=request.skip_identity_check,
            config_names=[
                request.follower_config,
                request.right_follower_config,
                request.leader_config,
                request.right_leader_config,
            ],
        )

        # Each sub-arm auto-loaded its calibration in __init__ (id=<base>_side);
        # register it on the bus, then configure both sides. Teleop opens no
        # cameras: it consumes no frames (only motor positions drive the URDF
        # viewer), so lerobot gets no cameras and the browser handles any display.
        for arm in (robot.left_arm, robot.right_arm, teleop_device.left_arm, teleop_device.right_arm):
            arm.bus.write_calibration(arm.calibration)
        robot.configure()
        teleop_device.configure()
        # Stock session torque (RAM Torque_Limit re-seeded from EEPROM) —
        # followers only, never the human-held leader. Clears any torque cap a
        # previous auto-calibration left in RAM; a failed write degrades to
        # the previous limit and is surfaced as a warning.
        identity_warnings += reset_torque_limit(robot, "follower arms")
        # Clear any leftover Goal_Velocity speed cap a previous arm-driving
        # feature stamped in RAM (auto-cal fold/unfold=1000, rest-pose return=400);
        # followers only, never the human-held leader. See makermodslab/motor_power.py.
        identity_warnings += clear_goal_velocity(robot, "follower arms")
        logger.info("Successfully connected to both bimanual arms")
        return robot, teleop_device, identity_warnings
    except Exception as e:
        # robot.configure() above may already have energized both follower
        # sub-arms before a later step (e.g. teleop_device.configure()) failed
        # — mirror the single-arm path's belt-and-braces cleanup instead of a
        # bare disconnect, and stash the result on the exception so the caller
        # (which has no robot/teleop_device of its own to clean up) can surface
        # it as a warning instead of losing it.
        e.cleanup_error = _cleanup_after_setup_failure(robot, teleop_device, "follower arms", "leader arms")
        raise


def handle_start_teleoperation(request: TeleoperateRequest, websocket_manager=None) -> dict[str, Any]:
    """Handle start teleoperation request.

    Connects to both arms *synchronously* so that a connection failure (arm
    unplugged, port busy, power off) is reported back to the caller, rather than
    dying silently in the worker thread while the API has already claimed
    success. Only the teleoperation loop runs in the background thread.
    """
    global teleoperation_active, teleoperation_thread, current_robot, current_teleop, last_cleanup_error
    global releasing, last_session_outcome, last_session_error

    from . import (
        auto_calibrate as _auto_calibrate,
        calibrate as _calibrate,
        record as _record,
        replay as _replay,
        rollout as _rollout,
        wiggle as _wiggle,
    )

    # A previous session (teleop or recording) may still be holding torque for
    # its release grace — cut it short so this start doesn't fail on a busy
    # serial port. Best effort: failures are surfaced by the checks below.
    finish_pending_release()
    _record.finish_pending_release()

    with _state_lock:
        if teleoperation_active:
            return {"success": False, "message": "Teleoperation is already active"}
        if teleoperation_thread is not None and teleoperation_thread.is_alive():
            # A stopped session's worker is still releasing the arms (grace was
            # cut short above but the cleanup hasn't finished yet).
            return {
                "success": False,
                "message": "The arms from the previous session are still being released. "
                "Try again in a few seconds.",
            }
        if _record.recording_active:
            return {"success": False, "message": "Recording is currently active. Stop it first."}
        if _rollout.inference_active:
            return {"success": False, "message": "Inference is currently active. Stop it first."}
        if _calibrate.calibration_is_active():
            return {"success": False, "message": "Calibration is currently active. Stop it first."}
        if _auto_calibrate.auto_calibration_is_active():
            return {"success": False, "message": "Auto-calibration is currently active. Stop it first."}
        if _wiggle.wiggle_active:
            return {
                "success": False,
                "message": "A gripper wiggle is currently in progress. Wait for it to finish.",
            }
        if _replay.replay_active:
            return {
                "success": False,
                "message": "Replay is currently active. Stop it first.",
            }
        # Per-session state reset, under the same lock that claims the active
        # flag: a stale _release_now from a previous session's double-stop
        # would otherwise cut EVERY later grace/return short until the server
        # restarts (regression-tested in tests/test_teleoperate.py).
        teleoperation_active = True
        last_cleanup_error = None
        last_session_outcome = None
        last_session_error = None
        releasing = False
        _release_now.clear()

    robot = None
    teleop_device = None
    try:
        logger.info(
            f"Starting teleoperation with leader port: {request.leader_port}, follower port: {request.follower_port}"
        )

        if request.mode == "bimanual":
            robot, teleop_device, identity_warnings = _connect_bimanual(request)
        else:
            # Create robot and teleop configs (stages calibration files).
            robot_config, teleop_config = build_single_configs(request)

            # Connect synchronously. If either device fails to connect, clean up the
            # other (so its serial port is released) and report the error — do NOT
            # leave the caller thinking teleoperation started.
            logger.info("Initializing robot and teleop device...")
            robot = SO101Follower(robot_config)
            teleop_device = SO101Leader(teleop_config)

            # Connect each arm separately so the error names which one failed and
            # tells the user what to do, instead of a generic "failed to start".
            logger.info("Connecting to follower arm...")
            try:
                robot.bus.connect()
            except Exception as e:
                raise RuntimeError(
                    f"Could not connect to the follower arm on {request.follower_port}. "
                    "Make sure it's plugged in and powered on, then try again."
                ) from e

            logger.info("Connecting to leader arm...")
            try:
                teleop_device.bus.connect()
            except Exception as e:
                raise RuntimeError(
                    f"Could not connect to the leader arm on {request.leader_port}. "
                    "Make sure it's plugged in and powered on, then try again."
                ) from e

            # Arm-identity guard: read-only check that each connected arm matches
            # its assigned calibration — swapped leader/follower ports still
            # connect fine but would drive the arms with the wrong calibration.
            # Must run BEFORE write_calibration below stamps the (possibly wrong)
            # file into the servos' EEPROM. Raises on mismatch; the except path
            # below disconnects both devices and surfaces the message.
            identity_warnings = verify_devices(
                ((robot, "follower"), (teleop_device, "leader")), skip=request.skip_identity_check
            )

            # Write calibration to motors' memory
            logger.info("Writing calibration to motors...")
            robot.bus.write_calibration(robot.calibration)
            teleop_device.bus.write_calibration(teleop_device.calibration)

            # Configure motors. Teleop opens no cameras: it consumes no frames
            # (only motor positions drive the URDF viewer), so lerobot gets no
            # cameras and the browser handles any camera display.
            logger.info("Configuring motors...")
            robot.configure()
            teleop_device.configure()
            # Stock session torque (RAM Torque_Limit re-seeded from EEPROM) —
            # follower only, never the human-held leader. Clears any torque
            # cap a previous auto-calibration left in RAM; a failed write
            # degrades to the previous limit and is surfaced as a warning.
            identity_warnings += reset_torque_limit(robot, "follower arm")
            # Clear any leftover Goal_Velocity speed cap a previous arm-driving
            # feature stamped in RAM (auto-cal fold/unfold=1000, rest-pose
            # return=400); follower only, never the human-held leader. See
            # makermodslab/motor_power.py.
            identity_warnings += clear_goal_velocity(robot, "follower arm")
            logger.info("Successfully connected to both devices")

        current_robot = robot
        current_teleop = teleop_device

        # Capture the follower's rest pose now — after connect/identity guard,
        # before the loop moves anything — so a normal stop can drive it back
        # to where the user left it. Followers only, NEVER the human-held
        # leader. The gripper is excluded: at stop time it may be holding an
        # object, and returning it to its (likely open) starting width would
        # drop the object mid-return.
        follower_rest_poses = [
            (bus, {m: v for m, v in capture_rest_pose(bus).items() if m != "gripper"})
            for bus in _device_buses(robot)
        ]

        # Stream the arms in the background; the worker owns disconnect so stop()
        # does not race the serial bus from the request thread.
        # A bimanual BiSO robot exposes left_arm/right_arm; broadcast both arms'
        # joints so the frontend can drive two 3D viewers.
        is_bimanual = hasattr(robot, "left_arm") and hasattr(robot, "right_arm")

        # Power telemetry: ~1 Hz Present_Current samples per follower bus (see
        # PowerTelemetry). Followers only — the leader's torque is off.
        telemetry = PowerTelemetry()
        telemetry_targets = list(
            zip(_device_buses(robot), ["left_", "right_"] if is_bimanual else [""], strict=False)
        )

        def teleoperation_worker():
            global teleoperation_active, current_robot, current_teleop, last_cleanup_error, releasing
            global last_session_outcome, last_session_error

            logger.info("Starting teleoperation loop...")
            stopped_normally = False
            loop_error: str | None = None
            try:
                last_broadcast_time = 0
                last_current_sample_time = 0.0
                broadcast_interval = 0.05  # 20 FPS

                while teleoperation_active:
                    action = teleop_device.get_action()
                    robot.send_action(action)

                    current_time = time.time()
                    if current_time - last_broadcast_time >= broadcast_interval:
                        try:
                            # Piggyback the ~1 Hz current sample on the
                            # broadcast tick — no extra loop overhead, one
                            # sync_read per second per follower bus.
                            if current_time - last_current_sample_time >= _CURRENT_SAMPLE_INTERVAL_S:
                                for bus, prefix in telemetry_targets:
                                    telemetry.sample(bus, prefix)
                                last_current_sample_time = current_time
                            if is_bimanual:
                                joint_positions = get_joint_positions_from_robot(
                                    robot, prefix="left_", calibration=robot.left_arm.calibration
                                )
                            else:
                                joint_positions = get_joint_positions_from_robot(robot)
                            joint_data = {
                                "type": "joint_update",
                                "joints": joint_positions,
                                "timestamp": current_time,
                            }
                            if is_bimanual:
                                joint_data["joints_right"] = get_joint_positions_from_robot(
                                    robot, prefix="right_", calibration=robot.right_arm.calibration
                                )
                            if telemetry.latest_ma:
                                # Instantaneous follower current (mA), ~1 Hz
                                # fresh; the frontend is free to ignore it.
                                joint_data["follower_currents_ma"] = dict(telemetry.latest_ma)
                            if websocket_manager and websocket_manager.active_connections:
                                websocket_manager.broadcast_joint_data_sync(joint_data)
                            last_broadcast_time = current_time
                        except Exception as e:
                            logger.error(f"Error broadcasting joint data: {e}")

                    time.sleep(0.001)
                # The loop exited because a stop request cleared the active
                # flag — a *user-initiated* stop, so the graceful landing applies.
                stopped_normally = True
            except Exception as e:
                logger.error(f"Error during teleoperation loop: {e}")
                # In-process error taxonomy: this catch site holds the actual
                # exception object, so format it straight into the status
                # payload (no log forensics). A mid-loop exception means the
                # session died on its own — a real failure, classified below.
                loop_error = format_exception(e)
            finally:
                telemetry_summary = telemetry.summary()
                if telemetry_summary:
                    logger.info(telemetry_summary)
                if stopped_normally and not _release_now.is_set():
                    # User-initiated stop: no timed hold — the servos hold
                    # their last goal on their own — so drive the follower(s)
                    # straight back to their session-start pose, then release
                    # (same behavior as the auto-calibration stop). A second
                    # stop (release-now) skips/aborts the return; error exits
                    # skip this — the bus may be gone, release ASAP.
                    releasing = True
                    _return_followers_to_rest(follower_rest_poses, _release_now)
                # Belt and braces: disable torque explicitly before disconnect.
                # disconnect() disables torque too, but if it fails partway the
                # error is swallowed here and the arm stays energized (rigid) —
                # so make the disable explicit, and make any failure loud.
                problems = force_disable_torque(robot, "follower arm")
                problems += force_disable_torque(teleop_device, "leader arm")
                for device, label in ((robot, "follower arm"), (teleop_device, "leader arm")):
                    error = _safe_disconnect(device, label)
                    if error:
                        problems.append(error)
                last_cleanup_error = " ".join(problems) if problems else None
                # Catch-site classification: the loop exception (if any) is the
                # session's error and means "failed" — the loop was cut short.
                # A user-requested stop (stopped_normally) whose cleanup alone
                # complained ran fine: "ran_with_warning" with the cleanup text
                # as the error. A clean stop is "ok".
                last_session_error = loop_error or last_cleanup_error
                last_session_outcome = classify_outcome(stopped_normally, last_session_error)
                logger.info("Teleoperation stopped")
                teleoperation_active = False
                releasing = False
                current_robot = None
                current_teleop = None

        teleoperation_thread = threading.Thread(
            target=teleoperation_worker, name="teleoperation-worker", daemon=True
        )
        teleoperation_thread.start()

        response = {
            "success": True,
            "message": "Teleoperation started successfully",
            "leader_port": request.leader_port,
            "follower_port": request.follower_port,
        }
        # Warn-but-allow identity findings (EEPROM offsets differ from the file,
        # e.g. a saved config assigned without recalibrating).
        if identity_warnings:
            response["warning"] = " ".join(identity_warnings)
        return response

    except Exception as e:
        # Connection (or setup) failed before the loop started: release any
        # device that did open, reset state, and surface the error. Mirrors
        # the worker's normal cleanup below — robot.configure() above may
        # already have written Torque_Enable=1 on the real follower servos
        # before this fired, so skipping the explicit disable (or dropping
        # _safe_disconnect's error text) would leave the arm energized with
        # no warning surfaced. The bimanual path's _connect_bimanual already
        # ran this same cleanup itself — it owns robot/teleop_device before
        # this scope ever sees them — and stashed the result on the
        # exception; reuse that instead of dropping it (running it again here
        # would be a no-op anyway, since robot/teleop_device are still None
        # for that path).
        if hasattr(e, "cleanup_error"):
            last_cleanup_error = e.cleanup_error
        else:
            last_cleanup_error = _cleanup_after_setup_failure(robot, teleop_device)
        teleoperation_active = False
        current_robot = None
        current_teleop = None
        logger.error(f"Failed to start teleoperation: {e}")
        # str(e) is already a user-facing message for the connection failures
        # raised above; the toast title supplies the "error starting" context.
        response = {"success": False, "message": str(e)}
        if last_cleanup_error:
            response["warning"] = last_cleanup_error
        return response


def handle_stop_teleoperation() -> dict[str, Any]:
    """Handle stop teleoperation request.

    First stop: signals the worker via `teleoperation_active = False`. The
    control loop exits and the worker drives the follower(s) straight back to
    their session-start pose (no timed hold — the servos hold their last goal
    on their own in position mode), then releases torque — reported
    immediately via `releasing: true` rather than blocking through the return.
    Second stop during the return: aborts it (release now) and waits for the
    cleanup so any release problem is surfaced. The worker owns the disconnect
    call either way, so this never races the serial bus from the request
    thread.
    """
    global teleoperation_active, teleoperation_thread

    worker = teleoperation_thread
    if teleoperation_active:
        logger.info("Stop teleoperation triggered from web interface")
        teleoperation_active = False

        if worker is None or not worker.is_alive():
            # No worker is holding the arms (it already exited, or a start
            # never spawned one) — nothing to hold torque with, so report the
            # cleanup outcome directly, as before the grace period existed.
            teleoperation_thread = None
            if last_cleanup_error:
                return {
                    "success": True,
                    "message": "Teleoperation stopped, but releasing the arms reported a problem",
                    "warning": last_cleanup_error,
                }
            return {"success": True, "message": "Teleoperation stopped successfully"}

        return {
            "success": True,
            # The port is still held and the arm is still energized while the
            # return runs; the status endpoint reports the same flag.
            "releasing": True,
            "message": (
                "Teleoperation stopped — the arm returns to its starting position, "
                "then goes limp. Press Stop again to release it now."
            ),
        }

    if worker is not None and worker.is_alive():
        # Second stop while the return (or the release cleanup) is still
        # running: release immediately and wait so problems can be surfaced.
        logger.info("Second stop during the rest-pose return — releasing the arms now")
        _release_now.set()
        worker.join(timeout=5.0)
        if worker.is_alive():
            logger.warning("Teleoperation worker did not exit within 5s")
            return {
                "success": True,
                "message": "Release requested, but the worker has not shut down yet",
                "warning": (
                    "The teleoperation worker did not shut down within 5s, so the arms may not have "
                    "been released. If an arm stays rigid, unplug its power to release it."
                ),
            }
        teleoperation_thread = None
        # The worker has exited, so its cleanup already ran; if disabling
        # torque or disconnecting failed, tell the caller — the arm may still
        # be energized.
        if last_cleanup_error:
            return {
                "success": True,
                "message": "Arms released, but the release reported a problem",
                "warning": last_cleanup_error,
            }
        return {"success": True, "message": "Arms released"}

    return {"success": False, "message": "No teleoperation session is active"}


def handle_teleoperation_status() -> dict[str, Any]:
    """Handle teleoperation status request"""
    message = (
        "Returning the arm to its rest position…"
        if releasing
        else "Teleoperation status retrieved successfully"
    )
    return {
        "teleoperation_active": teleoperation_active,
        "available_controls": {
            "stop_teleoperation": teleoperation_active,
        },
        # True during the post-stop rest-pose return: the session is over but
        # the arm is still energized and the port is still held.
        "releasing": releasing,
        # Non-None when the last session's cleanup could not release an arm
        # (torque may still be enabled); cleared when a new session starts.
        "last_cleanup_error": last_cleanup_error,
        # Terminal error taxonomy of the most recent session (in-process twin
        # of rollout's exited payload): `outcome` classifies it (ok |
        # ran_with_warning | failed, None before any session ends), `error` is
        # the text behind it, `hint` a plain-language headline. A mid-loop
        # death is "failed"; a user stop whose cleanup alone complained is
        # "ran_with_warning" — the frontend styles it amber, not red.
        "outcome": last_session_outcome,
        "error": last_session_error,
        "hint": friendly_hint(last_session_error),
        "message": message,
    }
