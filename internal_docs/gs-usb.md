# candleLight / gs_usb Maker adapters

Install `uv sync --extra gs-usb`; macOS also needs `brew install libusb`.
Refresh ports in Robot Configuration, choose the Maker follower slot, and select
`gs_usb · <USB serial>`. The saved value is `gs_usb:<USB serial>`; USB bus/address
changes do not alter it. Star leaders, SO-101 and Metal slots use their existing ports.

Listing reads USB descriptors only. Detect sends non-clearing MIT fault queries
to IDs 1–7; a valid reply confirms communication, not the arm model. Multiple
adapters can be assigned by serial; hand-motion port detection is unavailable
for gs_usb. Normal user-started robot sessions retain the driver's existing
handshake and torque behavior. Transport support has mock validation only;
teleoperation and recording over this transport require hardware validation.

Close other CAN programs before connecting. On Linux, if the kernel owns the
USB device use its configured SocketCAN interface; this transport does not
detach kernel drivers. Missing optional dependencies leave serial discovery
available. On macOS the normal Homebrew libusb locations are checked.

Source: python-can 4.6.1 gs_usb backend and gs-usb 0.3.1.
https://python-can.readthedocs.io/en/stable/interfaces/gs_usb.html
