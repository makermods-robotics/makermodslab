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
"""
Touch-to-identify: the inverse of wiggle-to-find-port (makermodslab/wiggle.py). Instead
of DRIVING a motor so the user can spot the arm, the user physically swings an
arm's shoulder-pan joint left and right by hand, and we watch raw
Present_Position on motor id 1 across every candidate port to find the one that
sees the motion. Strictly READ-ONLY: no register writes, no torque changes, no
EEPROM dependence — reading position does not energize an idle (torque-off) arm.
"""

import asyncio
import logging
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from .hardware_lease import (
    HardwareLeaseHeld,
    HardwareLeaseToken,
    hardware_lease_registry,
    held_response,
    safe_hardware_receipt,
)
from .hardware_recovery_identity import hardware_recovery_identity
from .utils.config import find_available_ports

logger = logging.getLogger(__name__)

# A deliberate left-right swing must move the raw position at least this many
# encoder ticks (of 4096 per turn, so ~10°) BOTH above and below the baseline.
_SWING_THRESHOLD_TICKS = 120
# Each port is sampled at roughly this rate (one round-robin sweep per period).
_POLL_INTERVAL_S = 1.0 / 15.0
# How long the user has to perform the gesture.
_IDENTIFY_TIMEOUT_S = 20.0

_NO_MOTION_MESSAGE = (
    "No motion detected within 20s — make sure the arm is powered and swing its base left and right."
)


def swing_detected(
    baseline: int, min_seen: int, max_seen: int, threshold: int = _SWING_THRESHOLD_TICKS
) -> bool:
    """True when observed positions swing BOTH above and below the baseline by
    at least `threshold` ticks (in any order).

    Requiring both directions rejects the false positives a single-sided check
    would accept: a bump against the arm, or slow drift, moves the position one
    way — only a deliberate left-right gesture crosses the threshold on both
    sides of where the joint started.
    """
    return (max_seen - baseline) >= threshold and (baseline - min_seen) >= threshold


def _open_shoulder_pan_bus(port: str) -> tuple[FeetechMotorsBus, int]:
    """Open a bus with just the shoulder-pan motor (id 1) and read the raw
    baseline position. Raises on any failure (busy port, unplugged, no motor)."""
    bus = FeetechMotorsBus(
        port=port,
        motors={"shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)},
    )
    try:
        bus.connect()
        baseline = bus.sync_read("Present_Position", "shoulder_pan", normalize=False)["shoulder_pan"]
    except Exception:
        _release_bus(bus)
        raise
    return bus, baseline


def _release_bus(bus: FeetechMotorsBus) -> str | None:
    """Close a bus without ever writing to the servos.

    `disconnect()` defaults to disable_torque=True, which WRITES Torque_Enable —
    that would break the read-only guarantee (and could drop a torqued arm if a
    port was misidentified). Never raises: called on every exit path, including
    after a failed connect where the port was never opened.
    """
    try:
        bus.disconnect(disable_torque=False)
    except Exception as exc:
        return str(exc)
    return None


def _identify_arm_sync(ports: list[str], timeout_s: float = _IDENTIFY_TIMEOUT_S) -> dict:
    """Watch shoulder-pan raw positions on all `ports` until one swings both
    ways past the threshold, or `timeout_s` elapses. Blocking; run in a worker
    thread. Ports that fail to open are skipped (reported), not fatal.
    """
    buses: dict[str, FeetechMotorsBus] = {}
    skipped: list[str] = []
    try:
        baselines: dict[str, int] = {}
        min_seen: dict[str, int] = {}
        max_seen: dict[str, int] = {}
        for port in ports:
            try:
                bus, baseline = _open_shoulder_pan_bus(port)
            except Exception as e:
                # Busy (teleop running), unplugged mid-scan, or not an arm.
                logger.info(f"identify: skipping {port}: {e}")
                skipped.append(port)
                continue
            buses[port] = bus
            baselines[port] = baseline
            min_seen[port] = baseline
            max_seen[port] = baseline

        if not buses:
            return {
                "success": False,
                "message": "Could not open any arm port — is another feature (teleop, recording) using them?",
                "skipped": skipped,
            }

        # Even with a single open port this is still a useful confirmation
        # gesture, so run the watch loop regardless of how many ports opened.
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for port, bus in buses.items():
                try:
                    pos = bus.sync_read("Present_Position", "shoulder_pan", normalize=False)["shoulder_pan"]
                except Exception:
                    continue  # transient read glitch; keep watching
                min_seen[port] = min(min_seen[port], pos)
                max_seen[port] = max(max_seen[port], pos)
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
        close_errors = [error for bus in buses.values() if (error := _release_bus(bus))]
        if close_errors:
            raise RuntimeError("Could not disconnect identify-arm bus: " + "; ".join(close_errors))


def _identify_and_release(candidates: list[str], lease_token: HardwareLeaseToken) -> dict:
    error: Exception | None = None
    try:
        return _identify_arm_sync(candidates)
    except Exception as exc:
        error = exc
        raise
    finally:
        if hardware_lease_registry.is_token_current(lease_token):
            if error is not None and "disconnect" in str(error).lower():
                hardware_lease_registry.mark_unresolved(lease_token, str(error))
            else:
                hardware_lease_registry.release(
                    lease_token,
                    safe_hardware_receipt(
                        "read-only identify buses closed",
                        torque_off=None,
                        torque_not_applicable=True,
                    ),
                )


async def identify_arm_by_motion(ports: list[str] | None = None) -> dict:
    """
    Run the read-only motion watch in a worker thread. `ports` defaults to all
    detected arm ports (same enumeration as the port dropdown). Returns a result
    dict ({"success": bool, "message": str, "port": str?, "skipped": [...]}) —
    logical failures are reported, not raised, so the endpoint stays HTTP 200
    like the other feature handlers (wiggle included).
    """
    candidates = [p.strip() for p in (ports or []) if p and p.strip()]
    if not candidates:
        candidates = find_available_ports()
    candidates = list(dict.fromkeys(candidates))  # dedupe, keep order
    if not candidates:
        return {
            "success": False,
            "message": "No arm ports detected — plug in an arm and try again.",
            "skipped": [],
        }
    try:
        lease_token = hardware_lease_registry.claim(
            "identify",
            "local:feetech-identify",
            recovery=hardware_recovery_identity(
                "so101",
                target_ports=candidates,
            ),
        )
    except HardwareLeaseHeld as exc:
        return held_response(exc)
    try:
        # The sync loop enforces its own deadline and returns a friendly
        # message; wait_for is a backstop (slightly longer, so the graceful
        # path wins) in case a serial read wedges.
        return await asyncio.wait_for(
            asyncio.to_thread(_identify_and_release, candidates, lease_token),
            timeout=_IDENTIFY_TIMEOUT_S + 5.0,
        )
    except TimeoutError:
        return {"success": False, "message": _NO_MOTION_MESSAGE, "skipped": []}
    except Exception as e:
        logger.exception("Identify-arm failed")
        return {"success": False, "message": f"Failed to identify the arm: {e}", "skipped": []}
