from types import SimpleNamespace

import pytest

from makermodslab.maker_fault_diagnostics import MakerFaultDiagnostics, decode_feedback


def frame(data):
    return SimpleNamespace(
        data=bytes(data), timestamp=123.0, arbitration_id=253, is_extended_id=False, is_error_frame=False
    )


def test_temperature_bits_do_not_include_mode_fault_or_warning():
    # 100 C with running mode and both flags, versus old temperature-only packet.
    new = decode_feedback(bytes([1, 128, 0, 0, 0, 1, 0xB3, 0xE8]))
    old = decode_feedback(bytes([1, 128, 0, 0, 0, 1, 0x03, 0xE8]))
    assert new["winding_temperature_c"] == old["winding_temperature_c"] == 100
    assert new["fault_flag"] and new["warning_flag"] and new["mode_bits"] == 2
    assert not old["fault_flag"]
    assert new["board_temperature_c"] is None


def test_raw_capture_fault_latch_and_no_automatic_fault_clear():
    sent = []
    messages = [frame([1, 128, 0, 0, 0, 1, 0xA3, 0xE8])]
    transport = SimpleNamespace(send=lambda msg: sent.append(msg), recv=lambda: messages.pop())
    decoded = []
    bus = SimpleNamespace(
        connect=lambda: None,
        canbus=transport,
        _id_to_name={1: "shoulder_lift"},
        _decode_motor_state=lambda d: decoded.append(bytes(d)),
    )
    diag = MakerFaultDiagnostics(bus)
    bus.connect()
    msg = transport.recv()
    bus._decode_motor_state(msg.data)
    assert decoded[0][6] == 3
    assert diag.faults["shoulder_lift"]["winding_temperature_c"] == 100
    with pytest.raises(RuntimeError, match="Fault clear blocked"):
        transport.send(frame([255] * 7 + [251]))
    assert not sent
    diag.snapshot()
    assert any("a3e8" in row["data_hex"] for row in diag.recent)


def test_fault_status_word_preserved_without_decoding_as_temperature():
    bus = SimpleNamespace(
        connect=lambda: None, _id_to_name={1: "shoulder_lift"}, _decode_motor_state=lambda d: None
    )
    diag = MakerFaultDiagnostics(bus)
    data = bytes([1, 1, 64, 0, 0, 0, 0, 0])
    diag.record("rx", frame(data))
    assert diag.faults["shoulder_lift"]["fault_names"] == ["motor_overtemperature", "stall_overload"]
    assert not diag.latest
    with pytest.raises(RuntimeError, match="Firmware fault frame"):
        bus._decode_motor_state(data)


def test_five_byte_legacy_fault_word_is_latched():
    bus = SimpleNamespace(
        connect=lambda: None, _id_to_name={7: "gripper"}, _decode_motor_state=lambda d: None
    )
    diag = MakerFaultDiagnostics(bus)
    data = bytes.fromhex("0706000000")
    diag.record("rx", frame(data))
    assert diag.faults["gripper"]["fault_word"] == 6
    assert diag.faults["gripper"]["fault_names"] == ["driver_chip", "undervoltage"]
    assert not diag.latest
    with pytest.raises(RuntimeError, match="Firmware fault frame"):
        bus._decode_motor_state(data)
