"""CAN journal and post-return fault read; no extra queries during playback."""

from __future__ import annotations

import json
import time
from collections import deque

FAULT_BITS = {
    0: "motor_overtemperature",
    1: "driver_chip",
    2: "undervoltage",
    3: "overvoltage",
    4: "phase_b_overcurrent",
    5: "phase_c_overcurrent",
    7: "encoder_uncalibrated",
    8: "hardware_identification",
    9: "position_initialization",
    14: "stall_overload",
    16: "phase_a_overcurrent",
}
MANUAL = "https://github.com/RobStride/Product_Information/blob/main/Product%20Literature/RS02/RS02User%20Manual260713.pdf"


def decode_feedback(data):
    """Keep raw bits and interpret the documented MIT status layout (July 2026)."""
    if len(data) != 8:
        return None
    word = int.from_bytes(data[6:8], "big")
    return {
        "raw_hex": bytes(data).hex(),
        "raw_temperature_word": word,
        "winding_temperature_c": (word & 0xFFF) / 10,
        "mode_bits": data[6] >> 6,
        "fault_flag": bool(data[6] & 0x20),
        "warning_flag": bool(data[6] & 0x10),
        "board_temperature_c": None,
        "board_temperature_status": "not exposed by this MIT feedback/driver",
        "interpretation": "MIT feedback per RS02/RS00 manual; firmware revision not read back",
    }


class ThermalDiagnostics:
    """Install before robot.connect; journal all transport traffic after bus handshake.

    Uses the existing bus owner, never a second receive thread. A latched firmware
    fault blocks subsequent fault-clear requests; manual recovery remains explicit.
    """

    def __init__(self, bus):
        self.bus = getattr(bus, "_base", bus)
        self.latest = {}
        self.faults = {}
        self.recent = deque(maxlen=20000)
        self.log = None
        self.clear_requests = 0
        self.events = deque(maxlen=1000)
        self.fault_status_reads = {}
        self.pending_fault_read = None
        self.bus.thermal_diagnostics = self
        original_connect = self.bus.connect
        original_decode = self.bus._decode_motor_state

        def connect(*args, **kwargs):
            result = original_connect(*args, **kwargs)
            transport = self.bus.canbus
            original_recv, original_send = transport.recv, transport.send

            def recv(*args, **kwargs):
                msg = original_recv(*args, **kwargs)
                if msg is not None:
                    self.record("rx", msg)
                return msg

            def send(msg, *args, **kwargs):
                data = bytes(msg.data)
                clearing = data == bytes([255] * 7 + [251])
                if clearing and self.faults:
                    self.record("blocked_fault_clear", msg)
                    raise RuntimeError("Fault clear blocked to preserve firmware fault evidence")
                if clearing:
                    self.clear_requests += 1
                self.record("tx", msg)
                return original_send(msg, *args, **kwargs)

            transport.recv, transport.send = recv, send
            return result

        def decode(data):
            name = self.bus._id_to_name.get(data[0]) if data else None
            if len(data) == 8 and name:
                if self.is_fault_frame(data):
                    raise RuntimeError("Firmware fault frame; see diagnostics journal")
                # Newer firmware uses the upper nibble for mode/fault/warning.
                # Preserve it in the raw journal, pass only temperature to the
                # legacy decoder, which otherwise misreads flags as thousands C.
                clean = bytearray(data)
                clean[6] &= 0x0F
                return original_decode(clean)
            return original_decode(data)

        self.bus.connect = connect
        self.bus._decode_motor_state = decode

    @staticmethod
    def is_fault_frame(data):
        # Same shape heuristic as the installed driver. Preserve ambiguity in
        # the label; no extra fault-read request is sent during control.
        return len(data) == 8 and any(data[1:5]) and not any(data[5:8])

    def record(self, direction, msg):
        stamp = time.time()
        data = bytes(msg.data)
        row = {
            "timestamp": stamp,
            "can_timestamp": msg.timestamp,
            "direction": direction,
            "arbitration_id": msg.arbitration_id,
            "extended_id": msg.is_extended_id,
            "error_frame": msg.is_error_frame,
            "data_hex": data.hex(),
        }
        name = self.bus._id_to_name.get(data[0]) if data else None
        if direction == "rx" and name and len(data) == 8 and not msg.is_extended_id:
            if self.is_fault_frame(data) or (name == self.pending_fault_read and not any(data[5:8])):
                bits = int.from_bytes(data[1:5], "little")
                item = {
                    "timestamp": stamp,
                    "fault_word": bits,
                    "fault_names": [v for k, v in FAULT_BITS.items() if bits & (1 << k)],
                    "classification": "possible MIT fault-status frame (driver shape heuristic)",
                    "raw_hex": data.hex(),
                }
                if bits:
                    self.faults[name] = item
                row["fault_status"] = item
            else:
                item = {"timestamp": stamp, **decode_feedback(data)}
                self.latest[name] = item
                if item["fault_flag"]:
                    self.faults[name] = item
                if item["fault_flag"] or item["warning_flag"]:
                    self.events.append({"actuator": name, **item})
        if self.log:
            self.log.write(json.dumps(row) + "\n")
        else:
            self.recent.append(row)

    def open(self, root):
        root.mkdir(parents=True, exist_ok=True)
        self.log = (root / "can_frames.jsonl").open("a", buffering=65536)
        for row in self.recent:
            self.log.write(json.dumps(row) + "\n")
        self.recent.clear()
        self.log.flush()

    def snapshot(self):
        if self.log:
            self.log.flush()
        return {
            "latest_feedback": dict(self.latest),
            "latched_firmware_faults": dict(self.faults),
            "warning_fault_events": list(self.events),
            "fault_clear_requests_observed": self.clear_requests,
            "fault_status_reads": dict(self.fault_status_reads),
            "board_temperature_c": None,
            "board_temperature_status": "not available in current MIT stream",
            "firmware_protection_threshold_readback": None,
            "manual": MANUAL,
            "capture_scope": "raw RX/TX after initial bus handshake, including return and disconnect",
        }

    def read_fault_status(self):
        """Bounded post-return read using F_CMD=0; NEVER the F_CMD=255 clear.

        Called only once the trial has verified rest, not in the playback loop.
        Older firmware may not support this; preserve the reply or timeout instead
        of mistaking a normal status response for a fault word.
        """
        import can

        for name in self.bus.motors:
            target = self.bus._get_motor_id(name)
            recv_id = self.bus._get_motor_recv_id(name)
            try:
                self.pending_fault_read = name
                self.bus.canbus.send(
                    can.Message(arbitration_id=target, data=[255] * 6 + [0, 251], is_extended_id=False)
                )
                deadline = time.monotonic() + 0.04
                item = {"status": "no fault-status response", "fault_word": None}
                while time.monotonic() < deadline:
                    msg = self.bus.canbus.recv(timeout=max(0, deadline - time.monotonic()))
                    if msg is None:
                        break
                    data = bytes(msg.data)
                    if msg.is_extended_id or len(data) != 8 or data[0] != recv_id:
                        continue
                    if any(data[5:8]):
                        item["other_response_hex"] = data.hex()
                        continue
                    bits = int.from_bytes(data[1:5], "little")
                    item = {
                        "status": "fault-status response",
                        "timestamp": time.time(),
                        "raw_hex": data.hex(),
                        "fault_word": bits,
                        "fault_names": [v for k, v in FAULT_BITS.items() if bits & (1 << k)],
                        "request_hex": "ffffffffffff00fb",
                    }
                    break
                self.fault_status_reads[name] = item
            except Exception as exc:
                self.fault_status_reads[name] = {"status": "read unavailable", "error": str(exc)}
            finally:
                self.pending_fault_read = None

    def close(self):
        if self.log:
            self.log.close()
            self.log = None
