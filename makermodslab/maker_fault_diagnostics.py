"""Bounded CAN fault evidence for Maker teleoperation recovery."""

from __future__ import annotations

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


class MakerFaultDiagnostics:
    """Install before robot.connect; journal all transport traffic after bus handshake.

    Uses the existing bus owner, never a second receive thread. A latched firmware
    fault blocks subsequent fault-clear requests; manual recovery remains explicit.
    """

    def __init__(self, bus):
        self.bus = getattr(bus, "_base", bus)
        self.latest = {}
        self.faults = {}
        self.recent = deque(maxlen=20000)
        self.clear_requests = 0
        self.events = deque(maxlen=1000)
        self.pending_fault_read = None
        self.bus.maker_fault_diagnostics = self
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
            if len(data) in (5, 8) and name:
                if len(data) == 5 or self.is_fault_frame(data):
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
        # Legacy firmware sends ID + uint32. Padded replies retain the installed
        # driver's shape heuristic; preserve that ambiguity in the label.
        return len(data) == 5 or (len(data) == 8 and any(data[1:5]) and not any(data[5:8]))

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
        # MIT parameter acknowledgements (0x300/0x400 + motor id) carry a
        # register index in data[0], not a motor id. Their zero-filled payloads
        # can otherwise look like a fault-status frame for an unrelated joint.
        parameter_reply = (msg.arbitration_id & 0x700) in (0x300, 0x400)
        if (
            direction == "rx"
            and name
            and len(data) in (5, 8)
            and not msg.is_extended_id
            and not parameter_reply
        ):
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
        self.recent.append(row)

    def snapshot(self):
        return {
            "latest_feedback": dict(self.latest),
            "latched_firmware_faults": dict(self.faults),
            "warning_fault_events": list(self.events),
            "fault_clear_requests_observed": self.clear_requests,
            "board_temperature_c": None,
            "board_temperature_status": "not available in current MIT stream",
            "firmware_protection_threshold_readback": None,
            "manual": MANUAL,
            "capture_scope": "bounded CAN history captured when teleoperation faults",
        }
