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
"""Port detection for the Maker arm — the CAN/UART counterpart to identify.py.

``identify.py`` finds an SO-101 by opening a Feetech bus on every candidate
port and watching motor id 1's ``Present_Position`` while the user swings the
arm. None of that transfers: a Maker follower answers RobStride frames on a CAN
adapter, and its Star Arm 102 leader answers FashionStar frames on a UART
bridge. Opening a Feetech bus on either finds nothing.

Two detection modes, because the Maker pair needs less work than the SO-101 to
tell apart:

* **probe** (``probe_maker_ports``) — the two halves of a Maker rig speak
  DIFFERENT protocols on different adapters, so simply asking each port "do you
  answer RobStride?" and "do you answer FashionStar?" identifies which is the
  follower and which is the leader, with no gesture from the user at all. This
  is the common single-arm case and it is instant.

* **motion** (``identify_maker_arm_by_motion``) — probing cannot separate the
  LEFT from the RIGHT arm of a bimanual rig: both arms ship with identical CAN
  ids and identical servo ids, so both answer the same way. There the user has
  to tell us which is which by moving one, exactly as on the SO-101, and this
  watches shoulder-pan across the candidate ports of ONE device type.

Both modes are strictly READ-ONLY. No torque is enabled, no register or EEPROM
is written, and no zero is set. Reading a position does not energize an idle
arm on either bus.
"""

import asyncio
import contextlib
import logging
import time

from .utils.config import find_available_ports

logger = logging.getLogger(__name__)

# Motor/servo ids of the shoulder-pan joint on each side of a Maker rig. Both
# come from lerobot's own defaults (MakerFollowerConfig.motor_can_ids and
# RebotArm102LeaderConfig.joint_ids) — note they differ: the CAN motors are
# 1-indexed and the FashionStar servos are 0-indexed.
_FOLLOWER_PAN_CAN_ID = 1
_LEADER_PAN_SERVO_ID = 0

# Classic CAN at 1 Mbps, no CAN FD — MakerFollowerConfig.can_bitrate's default
# and what the arm's motors are configured for.
_CAN_BITRATE = 1_000_000

# The leader's UART speed, from RebotArm102LeaderConfig.baudrate.
_LEADER_BAUDRATE = 1_000_000

# A deliberate left-right swing must move the joint at least this far, in
# DEGREES, both above and below where it started. Both Maker buses report
# degrees natively (unlike the SO-101's raw 0-4095 encoder ticks), so this is
# not a tick threshold. ~10 degrees each way is comfortably past hand tremor
# and gravity sag but still an easy gesture.
_SWING_THRESHOLD_DEG = 10.0

# Each port is sampled at roughly this rate (one round-robin sweep per period).
# Slower than identify.py's 15 Hz: a CAN read waits for its reply frame, so
# sweeping too fast just queues requests behind each other.
_POLL_INTERVAL_S = 1.0 / 10.0

# How long the user has to perform the gesture.
_IDENTIFY_TIMEOUT_S = 20.0

# Per-port budget for a probe. A port with nothing on it fails fast (no reply
# to the first frame); this only bounds the pathological case of an adapter
# that opens but never answers.
_PROBE_TIMEOUT_S = 4.0

_NO_MOTION_MESSAGE = (
    "No motion detected within 20s — make sure the arm is powered and swing its base left and right."
)


def swing_detected(
    baseline: float, min_seen: float, max_seen: float, threshold: float = _SWING_THRESHOLD_DEG
) -> bool:
    """True when observed angles swing BOTH ways past ``threshold`` degrees.

    Requiring both directions rejects what a single-sided check would accept.
    That matters more here than on the SO-101: a Maker follower has no brakes,
    so a torque-off arm SAGS under gravity — a steady one-way drift that a
    one-sided threshold would eventually read as a deliberate gesture.
    """
    return (max_seen - baseline) >= threshold and (baseline - min_seen) >= threshold


# --- follower (RobStride over CAN) -------------------------------------------


def _open_follower_bus(port: str):
    """Open a CAN bus carrying just the shoulder-pan motor, and read its angle.

    Only ONE motor is put on the bus rather than all seven: the handshake pings
    every motor it knows about and raises if any is missing, so probing with
    the full set would reject an otherwise-healthy arm that has one motor
    powered down. One motor answering is enough to say "a Maker follower is
    here".

    The Motor is built exactly the way ``MakerFollower.__init__`` builds its
    own, and for the same reasons: the model string comes from lerobot's
    ``MOTOR_MODELS`` table (shoulder-pan is an "O0"), and ``recv_id`` /
    ``motor_type_str`` are set because RobStride MIT feedback carries the motor
    id in payload byte 0. The bus flags likewise mirror the follower's —
    classic CAN at 1 Mbps, no CAN FD — because a probe that negotiated
    differently from the real session could answer for a bus the session then
    cannot open.

    Raises on any failure (port busy, no adapter, nothing answering).
    """
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.robstride import RobstrideMotorsBus
    from lerobot.robots.maker_follower.maker_follower import MOTOR_MODELS

    model = MOTOR_MODELS["shoulder_pan"]
    motor = Motor(_FOLLOWER_PAN_CAN_ID, model, MotorNormMode.DEGREES)
    motor.recv_id = _FOLLOWER_PAN_CAN_ID
    motor.motor_type_str = model

    bus = RobstrideMotorsBus(
        port=port,
        motors={"shoulder_pan": motor},
        # "slcan" explicitly rather than the "auto" default: auto picks slcan
        # only for a port starting "/dev/", so a Windows "COM5" would fall
        # through to socketcan, which does not exist off Linux.
        can_interface="slcan",
        use_can_fd=False,
        bitrate=_CAN_BITRATE,
        data_bitrate=None,
    )
    try:
        bus.connect()
        angle = float(bus.read("Present_Position", "shoulder_pan"))
    except Exception:
        _release_follower_bus(bus)
        raise
    return bus, angle


def _release_follower_bus(bus) -> None:
    """Close a CAN bus without disabling torque.

    ``disconnect()`` defaults to disable_torque=True, which WRITES to the
    motors. That would break the read-only guarantee, and on a misidentified
    port it could drop a torqued arm that some other session is holding — the
    same hazard identify.py's ``_release_bus`` avoids.
    """
    with contextlib.suppress(Exception):
        bus.disconnect(disable_torque=False)


def _read_follower_angle(bus) -> float:
    return float(bus.read("Present_Position", "shoulder_pan"))


# --- leader (FashionStar over UART) ------------------------------------------


def _open_leader_bus(port: str):
    """Open the leader's servo bus and read its shoulder-pan angle.

    The Star Arm 102 is an encoder-only arm — there is no torque to enable or
    disable here at all, which makes this the safest probe of the two.
    """
    from motorbridge_smart_servo import FashionStarServo

    bus = FashionStarServo(port, baudrate=_LEADER_BAUDRATE)
    try:
        if not bus.ping(_LEADER_PAN_SERVO_ID):
            raise RuntimeError(f"No FashionStar servo answered id {_LEADER_PAN_SERVO_ID}")
        angle = _read_leader_angle(bus)
    except Exception:
        _release_leader_bus(bus)
        raise
    return bus, angle


def _release_leader_bus(bus) -> None:
    with contextlib.suppress(Exception):
        bus.close()


def _read_leader_angle(bus) -> float:
    """Shoulder-pan angle in degrees from a FashionStar monitor frame.

    ``angle_deg``, not ``angle`` — the same field lerobot's own
    ``RebotArm102Leader._read_raw_positions`` reads. A ServoMonitor also
    carries voltage/current/turn, none of which matter here.
    """
    monitors = bus.sync_monitor([_LEADER_PAN_SERVO_ID])
    monitor = monitors.get(_LEADER_PAN_SERVO_ID) if isinstance(monitors, dict) else None
    if monitor is None:
        raise RuntimeError("No monitor frame from the leader's shoulder-pan servo")
    return float(monitor.angle_deg)


# --- the two detection modes --------------------------------------------------

_OPENERS = {
    "robot": (_open_follower_bus, _release_follower_bus, _read_follower_angle),
    "teleop": (_open_leader_bus, _release_leader_bus, _read_leader_angle),
}


def _probe_sync(ports: list[str]) -> dict:
    """Classify each port by which protocol answers on it.

    Tries the LEADER (UART) probe first and the follower (CAN) probe second,
    stopping at the first that answers — a port cannot be both. The order is
    for speed, not correctness: a FashionStar ping that goes unanswered fails
    in milliseconds, whereas the RobStride handshake retries every motor with
    its own timeout, so probing CAN-first spends seconds on each port that
    turns out to be the leader.
    """
    follower_ports: list[str] = []
    leader_ports: list[str] = []
    unknown: list[str] = []

    for port in ports:
        found = None
        for device_type in ("teleop", "robot"):
            opener, releaser, _ = _OPENERS[device_type]
            try:
                bus, _angle = opener(port)
            except Exception as e:
                logger.debug(f"maker probe: {port} is not a {device_type}: {e}")
                continue
            releaser(bus)
            found = device_type
            break
        if found == "robot":
            follower_ports.append(port)
        elif found == "teleop":
            leader_ports.append(port)
        else:
            unknown.append(port)

    return {
        "success": bool(follower_ports or leader_ports),
        "follower_ports": follower_ports,
        "leader_ports": leader_ports,
        "unknown_ports": unknown,
        "message": _probe_message(follower_ports, leader_ports),
    }


def _probe_message(follower_ports: list[str], leader_ports: list[str]) -> str:
    if not follower_ports and not leader_ports:
        return (
            "No Maker arm hardware answered on any port. Check that the CAN adapter and the "
            "leader's USB cable are plugged in, the arm is powered, and its motors are in MIT mode."
        )
    parts = []
    if follower_ports:
        parts.append(f"follower on {', '.join(follower_ports)}")
    if leader_ports:
        parts.append(f"leader on {', '.join(leader_ports)}")
    return "Found " + "; ".join(parts) + "."


def _identify_sync(ports: list[str], device_type: str, timeout_s: float = _IDENTIFY_TIMEOUT_S) -> dict:
    """Watch shoulder-pan on all `ports` of one device type until one swings.

    Blocking; run in a worker thread. Ports that fail to open are skipped
    (and reported), not fatal — one of them is usually the OTHER half of the
    rig, which speaks a protocol this opener does not.
    """
    opener, releaser, reader = _OPENERS[device_type]
    buses: dict[str, object] = {}
    skipped: list[str] = []
    try:
        baselines: dict[str, float] = {}
        min_seen: dict[str, float] = {}
        max_seen: dict[str, float] = {}
        for port in ports:
            try:
                bus, baseline = opener(port)
            except Exception as e:
                logger.info(f"maker identify: skipping {port}: {e}")
                skipped.append(port)
                continue
            buses[port] = bus
            baselines[port] = min_seen[port] = max_seen[port] = baseline

        if not buses:
            return {
                "success": False,
                "message": (
                    "Could not open any Maker arm port — is another feature (teleop, recording) "
                    "using them, or is the arm unpowered?"
                ),
                "skipped": skipped,
            }

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for port, bus in buses.items():
                try:
                    angle = reader(bus)
                except Exception:
                    continue  # transient read glitch; keep watching
                min_seen[port] = min(min_seen[port], angle)
                max_seen[port] = max(max_seen[port], angle)
                if swing_detected(baselines[port], min_seen[port], max_seen[port]):
                    return {
                        "success": True,
                        "port": port,
                        "message": f"Detected motion on {port}.",
                        "skipped": skipped,
                    }
            time.sleep(_POLL_INTERVAL_S)

        return {"success": False, "message": _NO_MOTION_MESSAGE, "skipped": skipped}
    finally:
        for bus in buses.values():
            releaser(bus)


def _candidate_ports(ports: list[str] | None) -> list[str]:
    candidates = [p.strip() for p in (ports or []) if p and p.strip()]
    if not candidates:
        candidates = find_available_ports()
    return list(dict.fromkeys(candidates))  # dedupe, keep order


async def probe_maker_ports(ports: list[str] | None = None) -> dict:
    """Identify which ports carry a Maker follower and which carry a leader.

    No user gesture needed — the two halves answer different protocols. Returns
    ``{"success", "follower_ports", "leader_ports", "unknown_ports",
    "message"}``; logical failures are reported rather than raised so the
    endpoint stays HTTP 200 like the other hardware handlers.
    """
    candidates = _candidate_ports(ports)
    if not candidates:
        return {
            "success": False,
            "follower_ports": [],
            "leader_ports": [],
            "unknown_ports": [],
            "message": "No serial ports detected — plug in the arm and try again.",
        }
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_probe_sync, candidates),
            timeout=_PROBE_TIMEOUT_S * len(candidates) * 2 + 5.0,
        )
    except TimeoutError:
        return {
            "success": False,
            "follower_ports": [],
            "leader_ports": [],
            "unknown_ports": candidates,
            "message": "Timed out probing the ports. Unplug anything unrelated and try again.",
        }
    except Exception as e:
        logger.exception("Maker port probe failed")
        return {
            "success": False,
            "follower_ports": [],
            "leader_ports": [],
            "unknown_ports": candidates,
            "message": f"Failed to probe for Maker arm hardware: {e}",
        }


async def identify_maker_arm_by_motion(device_type: str, ports: list[str] | None = None) -> dict:
    """Watch for a hand gesture to tell one Maker arm from its twin.

    ``device_type`` is "robot" (the CAN follower) or "teleop" (the UART
    leader) — unlike the SO-101 the two need different bus drivers, so the
    caller has to say which side it is asking about.
    """
    if device_type not in _OPENERS:
        return {
            "success": False,
            "message": "device_type must be 'teleop' or 'robot'",
            "skipped": [],
        }
    candidates = _candidate_ports(ports)
    if not candidates:
        return {
            "success": False,
            "message": "No serial ports detected — plug in the arm and try again.",
            "skipped": [],
        }
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_identify_sync, candidates, device_type),
            timeout=_IDENTIFY_TIMEOUT_S + 5.0,
        )
    except TimeoutError:
        return {"success": False, "message": _NO_MOTION_MESSAGE, "skipped": []}
    except Exception as e:
        logger.exception("Maker identify-arm failed")
        return {"success": False, "message": f"Failed to identify the arm: {e}", "skipped": []}
