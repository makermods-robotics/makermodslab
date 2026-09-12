# Copyright 2026 MakerMods contributors. All rights reserved.
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
"""Exact-serial candleLight transport; optional USB imports stay lazy."""

import logging
import math
import sys
from pathlib import Path

import can

GS_USB_INSTALL = "Install makermodslab[gs-usb] and libusb (macOS: brew install libusb)."


class DiagnosticError(RuntimeError):
    """USB transport could not complete an operation."""


def discover_gs_usb():
    """Descriptor-only discovery; never call GsUsb.start/scan/reset."""
    try:
        import usb.core
        from gs_usb.gs_usb import GsUsb
        from usb.backend import libusb1

        backend = libusb1.get_backend()
        if backend is None and sys.platform == "darwin":
            for path in (
                "/opt/homebrew/opt/libusb/lib/libusb-1.0.dylib",
                "/usr/local/opt/libusb/lib/libusb-1.0.dylib",
            ):
                if Path(path).is_file():
                    backend = libusb1.get_backend(find_library=lambda _, path=path: path)
                    if backend is not None:
                        break
        if backend is None:
            raise DiagnosticError(f"libusb backend not found. {GS_USB_INSTALL}")
        return [
            GsUsb(dev)
            for dev in usb.core.find(find_all=True, custom_match=GsUsb.is_gs_usb_device, backend=backend)
        ]
    except DiagnosticError:
        raise
    except Exception as exc:
        raise DiagnosticError(f"USB enumeration failed: {exc}. {GS_USB_INSTALL}") from exc


def dispose_usb(raw):
    import usb.util

    usb.util.dispose_resources(raw)


def select_gs_usb(serial):
    devices = discover_gs_usb()
    matches = []
    try:
        for dev in devices:
            try:
                if dev.serial_number == serial:
                    matches.append(dev)
            except Exception:
                continue
        if len(matches) != 1:
            raise DiagnosticError(
                f"Expected exactly one gs_usb adapter with serial {serial!r}; found {len(matches)}. Run ports; check USB permissions and descriptor availability."
            )
        return matches[0]
    finally:
        for dev in devices:
            if len(matches) != 1 or dev is not matches[0]:
                dispose_usb(dev.gs_usb)


class BoundedGsUsb(can.BusABC):
    """gs-usb 0.3.1 frame codec with bounded PyUSB I/O on one selected handle.

    python-can 4.6.1's gs_usb send ignores its timeout and shutdown may rescan
    by index. Upstream GsUsb.read also swallows all USB errors. Avoid all three.
    Sources: https://python-can.readthedocs.io/en/stable/interfaces/gs_usb.html
    https://github.com/jxltom/gs_usb/blob/master/gs_usb/gs_usb.py
    No USB reset, automatic kernel detach, threads or index-based discovery.
    """

    def __init__(self, device, bitrate=1_000_000, timeout=0.1):
        super().__init__(channel=device.serial_number)
        self.io_timeout = timeout
        import can
        import usb.util
        from gs_usb.constants import GS_CAN_MODE_HW_TIMESTAMP
        from gs_usb.gs_usb_structures import DeviceMode

        self.device = device
        self.raw = device.gs_usb
        self.closed = False
        self.claimed = False
        self.started = False
        self.raw.default_timeout = self.milliseconds(timeout)
        try:
            usb.util.claim_interface(self.raw, 0)
            self.claimed = True
            timing = can.BitTiming.from_sample_point(
                f_clock=device.device_capability.fclk_can, bitrate=bitrate, sample_point=87.5
            )
            device.set_timing(1, timing.tseg1 - 1, timing.tseg2, timing.sjw, timing.brp)
            device.device_flags = GS_CAN_MODE_HW_TIMESTAMP & device.device_capability.feature
            # gs_usb MODE request 2, START mode 1. Unlike GsUsb.start, no USB reset.
            self.started = True  # Cleanup even if the start transfer fails partway.
            self.raw.ctrl_transfer(0x41, 2, 0, 0, DeviceMode(1, device.device_flags).pack())
        except Exception as exc:
            try:
                self.shutdown()
            except Exception as cleanup_error:
                exc.add_note(f"Selected USB adapter cleanup also failed: {cleanup_error}")
            raise

    @staticmethod
    def milliseconds(timeout):
        return max(1, math.ceil(timeout * 1000))

    def send(self, msg, timeout=None):
        timeout = self.io_timeout if timeout is None else timeout
        from gs_usb.constants import GS_CAN_MODE_HW_TIMESTAMP
        from gs_usb.gs_usb_frame import GsUsbFrame

        if msg.is_extended_id or msg.is_remote_frame or msg.is_error_frame or msg.is_fd or msg.dlc != 8:
            raise DiagnosticError("This gs_usb utility sends only classic 8-byte standard data frames")
        packet = GsUsbFrame(can_id=msg.arbitration_id, data=bytes(msg.data))
        data = packet.pack(bool(self.device.device_flags & GS_CAN_MODE_HW_TIMESTAMP))
        count = self.raw.write(0x02, data, timeout=self.milliseconds(timeout))
        if count != len(data):
            raise DiagnosticError("Incomplete gs_usb write; command may have been transmitted")

    def _recv_internal(self, timeout):
        timeout = self.io_timeout if timeout is None else timeout
        import can
        import usb.core
        from gs_usb.constants import GS_CAN_MODE_HW_TIMESTAMP
        from gs_usb.gs_usb_frame import GS_USB_NONE_ECHO_ID, GsUsbFrame

        packet = GsUsbFrame()
        timestamps = bool(self.device.device_flags & GS_CAN_MODE_HW_TIMESTAMP)
        try:
            data = self.raw.read(0x81, packet.__sizeof__(timestamps), timeout=self.milliseconds(timeout))
        except usb.core.USBTimeoutError:
            return None, False
        GsUsbFrame.unpack_into(packet, data, timestamps)
        if packet.channel != 0 or packet.can_dlc > 8 or packet.flags:
            raise DiagnosticError(
                "Unexpected gs_usb channel or CAN-FD frame; only classic channel 0 is supported"
            )
        if packet.echo_id != GS_USB_NONE_ECHO_ID:
            return None, False
        return can.Message(
            arbitration_id=packet.arbitration_id,
            is_extended_id=packet.is_extended_id,
            is_remote_frame=packet.is_remote_frame,
            is_error_frame=packet.is_error_frame,
            dlc=packet.can_dlc,
            data=bytes(packet.data[: packet.can_dlc]),
            is_rx=True,
        ), False

    def shutdown(self):
        from gs_usb.gs_usb_structures import DeviceMode

        if self.closed:
            return
        self.closed = True
        super().shutdown()
        try:
            if self.started:
                self.raw.ctrl_transfer(0x41, 2, 0, 0, DeviceMode(0, 0).pack())
        finally:
            dispose_usb(self.raw)


def available_gs_usb_ports():
    """Descriptor-only tokens; unavailable optional dependencies do not break serial discovery."""
    try:
        devices = discover_gs_usb()
    except DiagnosticError as exc:
        logging.getLogger(__name__).debug("gs_usb discovery unavailable: %s", exc)
        return []
    serials = []
    for device in devices:
        try:
            serial = device.serial_number
            if serial and not any(c.isspace() for c in serial):
                serials.append(serial)
        except Exception:
            logging.getLogger(__name__).debug("Cannot read USB serial", exc_info=True)
        finally:
            try:
                dispose_usb(device.gs_usb)
            except Exception:
                logging.getLogger(__name__).debug("USB descriptor cleanup failed", exc_info=True)
    return sorted("gs_usb:" + serial for serial in set(serials) if serials.count(serial) == 1)


def open_gs_usb(port, bitrate=1_000_000):
    if not port.startswith("gs_usb:") or not port[7:] or any(c.isspace() for c in port[7:]):
        raise DiagnosticError("Expected gs_usb:<exact USB serial>")
    try:
        return BoundedGsUsb(select_gs_usb(port[7:]), bitrate)
    except Exception as exc:
        raise DiagnosticError(
            f"Cannot open {port}: {exc}. Close other CAN software; check USB permissions. {GS_USB_INSTALL}"
        ) from exc


def probe_maker(port):
    """Non-clearing MIT fault query; adapter presence alone is not a motor reply.

    RS00/RS02 manual 260713 section 6.7: FF*6 00 FB, host 0xFD,
    motor ID then little-endian fault word (firmware uses 5 or padded 8 bytes).
    """
    import time

    bus = open_gs_usb(port)
    try:
        for motor in range(1, 8):
            bus.send(
                can.Message(arbitration_id=motor, is_extended_id=False, data=b"\xff" * 6 + b"\x00\xfb"),
                timeout=0.1,
            )
            deadline = time.monotonic() + 0.1
            while time.monotonic() < deadline:
                msg = bus.recv(timeout=max(0, deadline - time.monotonic()))
                if msg is None:
                    continue
                if (
                    msg.is_rx
                    and not msg.is_extended_id
                    and not msg.is_error_frame
                    and not msg.is_remote_frame
                    and not msg.is_fd
                    and msg.arbitration_id == 0xFD
                    and msg.dlc in (5, 8)
                    and len(msg.data) == msg.dlc
                    and msg.data[0] == motor
                    and (msg.dlc == 5 or bytes(msg.data[5:]) == b"\x00" * 3)
                ):
                    return True
        return False
    finally:
        bus.shutdown()
