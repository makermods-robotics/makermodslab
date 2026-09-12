from collections import deque
from types import SimpleNamespace

import can
import pytest

from makermodslab import gs_usb_transport as ids


def frame():
    return can.Message(arbitration_id=1, is_extended_id=False, data=b"\xff" * 6 + b"\x00\xfb")


class UsbRaw:
    def __init__(self):
        self.writes = []
        self.controls = []
        self.reads = deque()
        self.read_timeouts = []

    def write(self, endpoint, data, timeout):
        self.writes.append((endpoint, data, timeout))
        return len(data)

    def read(self, endpoint, size, timeout):
        self.read_timeouts.append(timeout)
        value = self.reads.popleft()
        if isinstance(value, Exception):
            raise value
        return value

    def ctrl_transfer(self, *args):
        self.controls.append(args)


@pytest.fixture
def gs_device(monkeypatch):
    pytest.importorskip("gs_usb")
    import usb.util

    raw = UsbRaw()
    device = SimpleNamespace(
        serial_number="ABC",
        gs_usb=raw,
        device_capability=SimpleNamespace(fclk_can=48000000, feature=16),
        set_timing=lambda *args: None,
    )
    monkeypatch.setattr(usb.util, "claim_interface", lambda *args: None)
    disposed = []
    monkeypatch.setattr(ids, "dispose_usb", disposed.append)
    return device, raw, disposed


def test_gs_bounded_write_no_message_mutation_and_exact_close(gs_device):
    device, raw, disposed = gs_device
    bus = ids.BoundedGsUsb(device, 1000000, 0.3)
    msg = frame()
    original = bytes(msg.data)
    bus.send(msg, 0.0001)
    assert raw.writes[0][0] == 2
    assert raw.writes[0][2] == 1
    assert bytes(msg.data) == original
    assert raw.default_timeout == 300
    bus.shutdown()
    bus.shutdown()
    assert len(raw.controls) == 2
    assert disposed == [raw]


def test_gs_receive_timeout_rounding_echo_and_errors(gs_device):
    import usb.core
    from gs_usb.gs_usb_frame import GS_USB_NONE_ECHO_ID, GsUsbFrame

    device, raw, _ = gs_device
    bus = ids.BoundedGsUsb(device, 1000000, 0.3)
    packet = GsUsbFrame(can_id=0xFD, data=bytes([1]) + bytes(7))
    raw.reads.append(packet.pack(True))
    assert bus._recv_internal(0.00001) == (None, False)
    packet.echo_id = GS_USB_NONE_ECHO_ID
    raw.reads.append(packet.pack(True))
    assert bus.recv(0).is_rx
    raw.reads.append(usb.core.USBTimeoutError("timeout"))
    assert bus._recv_internal(0.3) == (None, False)
    raw.reads.append(usb.core.USBError("unplugged"))
    with pytest.raises(usb.core.USBError):
        bus.recv(0.3)
    assert raw.read_timeouts == [1, 1, 300, 300]
    bus.shutdown()


def test_gs_partial_send_failure_and_failed_start_cleanup(gs_device, monkeypatch):
    device, raw, disposed = gs_device
    bus = ids.BoundedGsUsb(device, 1000000, 0.3)
    monkeypatch.setattr(raw, "write", lambda *args, **kwargs: 1)
    with pytest.raises(ids.DiagnosticError, match="may have been transmitted"):
        bus.send(frame(), 0.3)
    bus.shutdown()

    def fail(*args):
        raise OSError("start failed")

    monkeypatch.setattr(raw, "ctrl_transfer", fail)
    with pytest.raises(OSError):
        ids.BoundedGsUsb(device, 1000000, 0.3)
    assert disposed == [raw, raw]


def test_gs_rejects_fd_flag_with_short_payload(gs_device):
    from gs_usb.gs_usb_frame import GsUsbFrame

    device, raw, _ = gs_device
    bus = ids.BoundedGsUsb(device, 1000000, 0.3)
    packet = GsUsbFrame(can_id=0xFD, data=bytes([1]) + bytes(7))
    packet.flags = 2  # GS_CAN_FLAG_FD, even though DLC fits classic CAN.
    raw.reads.append(packet.pack(True))
    with pytest.raises(ids.DiagnosticError, match="CAN-FD"):
        bus.recv(0.3)
    bus.shutdown()


def test_discovery_absent_preserves_serial_listing(monkeypatch):
    monkeypatch.setattr(ids, "discover_gs_usb", lambda: (_ for _ in ()).throw(ids.DiagnosticError("missing")))
    assert ids.available_gs_usb_ports() == []


def test_tokens_stable_unique_and_disposed(monkeypatch):
    devices = [SimpleNamespace(serial_number=s, gs_usb=object()) for s in ["B", "A", "B"]]
    disposed = []
    monkeypatch.setattr(ids, "discover_gs_usb", lambda: devices)
    monkeypatch.setattr(ids, "dispose_usb", disposed.append)
    assert ids.available_gs_usb_ports() == ["gs_usb:A"]
    assert len(disposed) == 3


def test_hook_no_global_bus_patch_and_cleanup(monkeypatch):
    from lerobot.motors.robstride import RobstrideMotorsBus
    from makermodslab import maker_can

    maker_can.install()
    original_can = can.interface.Bus
    hook = RobstrideMotorsBus.connect
    maker_can.install()
    assert RobstrideMotorsBus.connect is hook
    assert can.interface.Bus is original_can
    closed = []
    usb = SimpleNamespace(shutdown=lambda: closed.append(True))
    monkeypatch.setattr(ids, "open_gs_usb", lambda *a: usb)
    fake = SimpleNamespace(
        port="gs_usb:A",
        is_connected=False,
        use_can_fd=False,
        bitrate=1000000,
        _handshake=lambda: (_ for _ in ()).throw(ValueError("bad handshake")),
    )
    with pytest.raises(ValueError, match="bad handshake"):
        hook(fake)
    assert closed == [True]
    assert not fake._is_connected and fake.canbus is None


def test_probe_only_fault_query_and_requires_rx(monkeypatch):
    sent = []
    bus = SimpleNamespace(
        send=lambda msg, **kw: sent.append(msg),
        recv=lambda **kw: can.Message(arbitration_id=0xFD, is_extended_id=False, data=b"\x01" + bytes(4)),
        shutdown=lambda: None,
    )
    monkeypatch.setattr(ids, "open_gs_usb", lambda p: bus)
    assert ids.probe_maker("gs_usb:A")
    assert len(sent) == 1
    assert bytes(sent[0].data) == b"\xff" * 6 + b"\x00\xfb"


def test_gs_probe_skips_leader_and_metal(monkeypatch):
    from makermodslab import maker_ports

    calls = []
    monkeypatch.setattr(ids, "probe_maker", lambda p: calls.append(p) or True)
    assert maker_ports._probe_sync(["gs_usb:A"])["follower_ports"] == ["gs_usb:A"]
    assert maker_ports._probe_sync(["gs_usb:A"], "metal")["unknown_ports"] == ["gs_usb:A"]
    assert calls == ["gs_usb:A"]


def test_token_record_roundtrip(tmp_path, monkeypatch):
    from makermodslab.utils import config

    monkeypatch.setattr(config, "ROBOTS_PATH", str(tmp_path))
    assert config.save_robot_record("usb_arm", {"arm_type": "maker", "follower_port": "gs_usb:ABC"})
    assert config.get_robot_record("usb_arm")["follower_port"] == "gs_usb:ABC"


@pytest.mark.asyncio
async def test_leader_motion_filters_usb_tokens(monkeypatch):
    from makermodslab import maker_ports

    seen = []
    monkeypatch.setattr(
        maker_ports, "_identify_sync", lambda ports, *a: seen.extend(ports) or {"success": True}
    )
    result = await maker_ports.identify_maker_arm_by_motion("teleop", ["gs_usb:A", "COM3"])
    assert result["success"]
    assert seen == ["COM3"]


def test_discovery_cleanup_failure_does_not_hide_serials(monkeypatch):
    monkeypatch.setattr(ids, "discover_gs_usb", lambda: [SimpleNamespace(serial_number="A", gs_usb=object())])
    monkeypatch.setattr(ids, "dispose_usb", lambda dev: (_ for _ in ()).throw(OSError("cleanup")))
    assert ids.available_gs_usb_ports() == ["gs_usb:A"]
