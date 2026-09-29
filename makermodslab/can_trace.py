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
"""TEMPORARY developer trace of every Damiao CAN frame. Off unless opted in.

Enable for one server run with ``MAKERMODSLAB_CAN_TRACE=1 makermodslab --dev``.
Every DamiaoMotorsBus (the Metal follower, and an energized Metal leader) then
wraps its python-can bus in :class:`CanTrace`, which:

* appends one JSON line per frame, both directions, decoded, to
  ``MAKERMODSLAB_HOME/logs/can_trace/<time>_<port>.jsonl``;
* logs to the terminal, immediately, any motor status other than
  disabled/enabled (8 overvoltage, 9 UNDERVOLTAGE, 10 overcurrent, 11/12
  overtemperature, 13 comm lost, 14 overload), any enabled -> disabled drop,
  any motor that stops replying for more than ``SILENCE_MS`` while it is being
  commanded, and any adapter exception;
* logs a once-per-second health line per motor: replies/commands, peak
  |torque| (tracks current draw), peak velocity and max temperatures.

The Damiao motors do not report their supply voltage; the undervoltage status
(9), a torque spike just before a fault, or every motor going silent at once
is how a brown-out shows up here. Remove this module when the investigation
is done.
"""

import contextlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

TRACE_ENV = "MAKERMODSLAB_CAN_TRACE"
SILENCE_MS = 150.0
SUMMARY_INTERVAL_S = 1.0
FLUSH_INTERVAL_S = 0.5

STATUS_NAMES = {
    0: "disabled",
    1: "enabled",
    8: "OVERVOLTAGE",
    9: "UNDERVOLTAGE",
    10: "OVERCURRENT",
    11: "MOS_OVERTEMP",
    12: "ROTOR_OVERTEMP",
    13: "COMM_LOST",
    14: "OVERLOAD",
}
SIMPLE_COMMANDS = {0xFC: "enable", 0xFD: "disable", 0xFE: "set_zero", 0xFB: "clear_error"}


def enabled() -> bool:
    return os.environ.get(TRACE_ENV, "").strip() not in ("", "0", "false", "False")


def _trace_path(port: str) -> Path:
    from .utils.config import MAKERMODSLAB_HOME

    root = Path(MAKERMODSLAB_HOME) / "logs" / "can_trace"
    root.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(port))
    return root / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}.jsonl"


class _MotorStats:
    def __init__(self) -> None:
        self.commands = 0
        self.replies = 0
        self.peak_torque = 0.0
        self.peak_velocity = 0.0
        self.max_t_mos = 0
        self.max_t_rotor = 0
        self.statuses: set[int] = set()


class CanTrace:
    """A python-can bus proxy that records and watches every frame of one Damiao bus."""

    def __init__(self, inner, damiao_bus) -> None:
        self._inner = inner
        self._bus = damiao_bus
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self.path = _trace_path(getattr(damiao_bus, "port", "can"))
        self._file = self.path.open("a", buffering=1 << 16)
        self._last_flush = self._t0
        self._last_summary = self._t0
        self._send_ids = {}
        for name in damiao_bus.motors:
            with contextlib.suppress(Exception):
                self._send_ids[damiao_bus._get_motor_id(name)] = name
        self._recv_ids = dict(getattr(damiao_bus, "_recv_id_to_motor", {}))
        self._stats = {name: _MotorStats() for name in damiao_bus.motors}
        self._status: dict[str, int] = {}
        self._last_reply: dict[str, float] = {}
        self._first_unanswered: dict[str, float] = {}
        self._silent: set[str] = set()
        logger.warning("[can-trace] recording %s -> %s", getattr(damiao_bus, "port", "?"), self.path)
        self._write(
            {"event": "open", "port": str(getattr(damiao_bus, "port", "")), "motors": list(self._stats)}
        )

    # -- python-can surface ---------------------------------------------------

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def send(self, msg, timeout=None):
        now = time.monotonic()
        try:
            result = self._inner.send(msg) if timeout is None else self._inner.send(msg, timeout)
        except Exception as exc:
            self._alarm("tx_error", None, f"adapter send failed: {type(exc).__name__}: {exc}", now)
            raise
        self._on_tx(msg, now)
        return result

    def recv(self, timeout=None):
        try:
            msg = self._inner.recv(timeout)
        except Exception as exc:
            self._alarm(
                "rx_error", None, f"adapter receive failed: {type(exc).__name__}: {exc}", time.monotonic()
            )
            raise
        if msg is not None:
            self._on_rx(msg, time.monotonic())
        return msg

    def shutdown(self):
        with self._lock:
            self._summary(time.monotonic(), force=True)
            self._write({"event": "close"})
            with contextlib.suppress(Exception):
                self._file.close()
        logger.warning("[can-trace] closed %s", self.path)
        return self._inner.shutdown()

    # -- decoding -------------------------------------------------------------

    def _motor_type(self, motor):
        return self._bus._motor_types[motor]

    def _decode_command(self, motor, data):
        from lerobot.motors.damiao.tables import MIT_KD_RANGE, MIT_KP_RANGE, MOTOR_LIMIT_PARAMS

        if len(data) == 8 and data[:7] == b"\xff" * 7:
            return {"cmd": SIMPLE_COMMANDS.get(data[7], f"0x{data[7]:02X}")}
        pmax, vmax, tmax = MOTOR_LIMIT_PARAMS[self._motor_type(motor)]
        u = self._bus._uint_to_float
        q = (data[0] << 8) | data[1]
        dq = (data[2] << 4) | (data[3] >> 4)
        kp = ((data[3] & 0x0F) << 8) | data[4]
        kd = (data[5] << 4) | (data[6] >> 4)
        tau = ((data[6] & 0x0F) << 8) | data[7]
        return {
            "cmd": "mit",
            "pos": round(float(u(q, -pmax, pmax, 16)) * 57.29577951308232, 3),
            "vel": round(float(u(dq, -vmax, vmax, 12)) * 57.29577951308232, 2),
            "kp": round(float(u(kp, *MIT_KP_RANGE, 12)), 2),
            "kd": round(float(u(kd, *MIT_KD_RANGE, 12)), 3),
            "tau_ff": round(float(u(tau, -tmax, tmax, 12)), 3),
        }

    def _on_tx(self, msg, now):
        data = bytes(msg.data)
        with self._lock:
            row = {"t": round((now - self._t0) * 1000, 2), "dir": "tx", "id": f"0x{msg.arbitration_id:03X}"}
            motor = self._send_ids.get(msg.arbitration_id)
            if motor is not None:
                row["motor"] = motor
                try:
                    row.update(self._decode_command(motor, data))
                except Exception:
                    row["raw"] = data.hex()
                self._stats[motor].commands += 1
                self._first_unanswered.setdefault(motor, now)
                self._check_silence(motor, now)
            else:
                row["raw"] = data.hex()
            self._write(row)
            self._housekeeping(now)

    def _on_rx(self, msg, now):
        data = bytes(msg.data)
        with self._lock:
            row = {"t": round((now - self._t0) * 1000, 2), "dir": "rx", "id": f"0x{msg.arbitration_id:03X}"}
            if getattr(msg, "is_error_frame", False):
                row["error_frame"] = True
                self._alarm("error_frame", None, f"CAN error frame {data.hex()}", now, locked=True)
            motor = self._recv_ids.get(msg.arbitration_id)
            if motor is not None and len(data) == 8 and not (data[1] == 0 and data[2] in (0x33, 0x55, 0xAA)):
                status = data[0] >> 4
                pos, vel, torque, t_mos, t_rotor = self._bus._decode_motor_state(
                    data, self._motor_type(motor)
                )
                row.update(
                    motor=motor,
                    status=status,
                    status_name=STATUS_NAMES.get(status, "?"),
                    pos=round(float(pos), 3),
                    vel=round(float(vel), 2),
                    torque=round(float(torque), 3),
                    t_mos=t_mos,
                    t_rotor=t_rotor,
                )
                stats = self._stats[motor]
                stats.replies += 1
                stats.peak_torque = max(stats.peak_torque, abs(float(torque)))
                stats.peak_velocity = max(stats.peak_velocity, abs(float(vel)))
                stats.max_t_mos = max(stats.max_t_mos, t_mos)
                stats.max_t_rotor = max(stats.max_t_rotor, t_rotor)
                stats.statuses.add(status)
                self._on_status(motor, status, now, float(torque))
                self._last_reply[motor] = now
                self._first_unanswered.pop(motor, None)
                if motor in self._silent:
                    self._silent.discard(motor)
                    logger.warning("[can-trace] %s is replying again", motor)
            elif motor is not None:
                row.update(motor=motor, param_reply=data.hex())
            else:
                row["raw"] = data.hex()
            self._write(row)
            self._housekeeping(now)

    # -- watching -------------------------------------------------------------

    def _on_status(self, motor, status, now, torque):
        previous = self._status.get(motor)
        self._status[motor] = status
        if status == previous:
            return
        if status not in (0, 1):
            self._alarm(
                "fault",
                motor,
                f"{motor} FAULT {STATUS_NAMES.get(status, status)} (torque {torque:+.2f} Nm)",
                now,
                locked=True,
                level=logging.ERROR,
            )
        elif previous == 1 and status == 0:
            self._alarm(
                "disabled",
                motor,
                f"{motor} went from enabled to DISABLED",
                now,
                locked=True,
                level=logging.ERROR,
            )
        elif previous is not None and previous not in (0, 1):
            self._alarm(
                "recovered",
                motor,
                f"{motor} status back to {STATUS_NAMES.get(status, status)}",
                now,
                locked=True,
                level=logging.WARNING,
            )

    def _check_silence(self, motor, now):
        started = self._first_unanswered.get(motor)
        if started is None or motor in self._silent:
            return
        if (now - started) * 1000 >= SILENCE_MS and self._stats[motor].commands > 1:
            self._silent.add(motor)
            quiet = sorted(self._silent)
            everyone = len(quiet) == len(self._stats)
            self._alarm(
                "silence",
                motor,
                f"{motor} has not replied for {(now - started) * 1000:.0f} ms while commanded"
                + (
                    " -- ALL motors silent (power loss or CAN link down?)"
                    if everyone
                    else f" (silent: {quiet})"
                ),
                now,
                locked=True,
                level=logging.ERROR,
            )

    def _alarm(self, kind, motor, text, now, locked=False, level=logging.ERROR):
        logger.log(level, "[can-trace] %s", text)
        row = {"t": round((now - self._t0) * 1000, 2), "event": kind, "motor": motor, "detail": text}
        if locked:
            self._write(row)
            self._flush(now, force=True)
        else:
            with self._lock:
                self._write(row)
                self._flush(now, force=True)

    def _housekeeping(self, now):
        self._summary(now)
        self._flush(now)

    def _summary(self, now, force=False):
        if not force and now - self._last_summary < SUMMARY_INTERVAL_S:
            return
        self._last_summary = now
        parts = []
        rows = {}
        for motor, s in self._stats.items():
            status = self._status.get(motor)
            parts.append(
                f"{motor}:{s.replies}/{s.commands} tq{s.peak_torque:.1f} v{s.peak_velocity:.0f} "
                f"T{s.max_t_mos}/{s.max_t_rotor} st{STATUS_NAMES.get(status, status)}"
            )
            rows[motor] = {
                "replies": s.replies,
                "commands": s.commands,
                "peak_torque": round(s.peak_torque, 3),
                "peak_velocity": round(s.peak_velocity, 1),
                "max_t_mos": s.max_t_mos,
                "max_t_rotor": s.max_t_rotor,
                "statuses": sorted(s.statuses),
            }
            self._stats[motor] = _MotorStats()
        if any(r["commands"] or r["replies"] for r in rows.values()):
            logger.info("[can-trace] 1s: %s", " | ".join(parts))
        self._write({"t": round((now - self._t0) * 1000, 2), "event": "summary", "motors": rows})

    def _write(self, row):
        with contextlib.suppress(Exception):
            self._file.write(json.dumps(row, separators=(",", ":")) + "\n")

    def _flush(self, now, force=False):
        if force or now - self._last_flush >= FLUSH_INTERVAL_S:
            self._last_flush = now
            with contextlib.suppress(Exception):
                self._file.flush()


def wrap(damiao_bus) -> None:
    """Wrap ``damiao_bus.canbus`` in a trace when developer tracing is on."""
    inner = getattr(damiao_bus, "canbus", None)
    if not enabled() or inner is None or isinstance(inner, CanTrace):
        return
    try:
        damiao_bus.canbus = CanTrace(inner, damiao_bus)
    except Exception:
        logger.exception("[can-trace] could not start the trace; continuing without it")
