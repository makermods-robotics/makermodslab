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
"""Narrow compatibility hooks for the pinned CAN drivers' port-only configuration."""

from functools import wraps


def _install_gripper_open_limit():
    """Narrow the Maker gripper's open end without editing the pinned driver.

    Every gripper path (follower clamp, holding controller, wiggle, remote CAN)
    reads ``config.joint_limits["gripper"]``, so bounding it at construction
    covers them all. The bimanual sub-arm base has no ``__post_init__``, hence
    wrapping ``__init__``.
    """
    from lerobot.robots.maker_follower import MakerFollowerConfig, MakerFollowerConfigBase

    from .gripper_settings import MAKER_GRIPPER_OPEN_LIMIT_DEG

    for cls in (MakerFollowerConfigBase, MakerFollowerConfig):
        original = cls.__init__
        if getattr(original, "_makermodslab_gripper_limit", False):
            continue

        def make(original):
            @wraps(original)
            def init(self, *args, **kwargs):
                original(self, *args, **kwargs)
                low, high = self.joint_limits["gripper"]
                self.joint_limits = {
                    **self.joint_limits,
                    "gripper": (max(low, MAKER_GRIPPER_OPEN_LIMIT_DEG), high),
                }

            init._makermodslab_gripper_limit = True
            return init

        cls.__init__ = make(original)


def _install_gs_usb_connect(bus_cls, label, wrap_errors=None, trace=False):
    """Route ``bus_cls.connect`` through the exact-serial gs_usb transport.

    Only a ``gs_usb:<serial>`` port is diverted; every other port reaches the
    driver's own connect untouched. A failed handshake closes the USB handle
    before raising. That matters most on Damiao, whose handshake IS the enable
    command: the stock driver leaves the half-open bus behind, and
    ``torque.de_energize_can_bus`` could not reopen an adapter this process
    still has claimed to broadcast the disable. ``wrap_errors`` re-raises as
    the driver's own failure type, so callers that catch it keep working.
    """
    original = bus_cls.connect
    if getattr(original, "_makermodslab_gs_usb", False):
        return

    @wraps(original)
    def connect(self, handshake=True):
        if not isinstance(self.port, str) or not self.port.startswith("gs_usb:"):
            result = original(self, handshake=handshake)
            if trace:
                from . import can_trace

                can_trace.wrap(self)
            return result
        from .gs_usb_transport import open_gs_usb

        if self.is_connected:
            raise RuntimeError(f"{label} bus is already connected")
        if self.use_can_fd:
            raise ValueError(f"{label} gs_usb supports classic CAN only")
        try:
            bus = open_gs_usb(self.port, self.bitrate)
        except Exception as exc:
            if wrap_errors is None:
                raise
            raise wrap_errors(f"Failed to connect to CAN bus: {exc}") from exc
        self.canbus = bus
        self._is_connected = True
        if trace:
            from . import can_trace

            can_trace.wrap(self)
        try:
            if handshake:
                self._handshake()
        except BaseException as exc:
            opened = self.canbus if self.canbus is not None else bus  # the trace wrapper, when on
            self._is_connected = False
            self.canbus = None
            try:
                opened.shutdown()
            except Exception as cleanup:
                exc.add_note(f"USB cleanup also failed: {cleanup}")
            if wrap_errors is None or not isinstance(exc, Exception) or isinstance(exc, wrap_errors):
                raise
            raise wrap_errors(f"Failed to connect to CAN bus: {exc}") from exc

    connect._makermodslab_gs_usb = True
    bus_cls.connect = connect


def install():
    from lerobot.motors.damiao import DamiaoMotorsBus
    from lerobot.motors.robstride import RobstrideMotorsBus

    _install_gripper_open_limit()
    _install_gs_usb_connect(RobstrideMotorsBus, "RobStride")
    # The Metal follower and the gravity-compensated Metal leader both open
    # DamiaoMotorsBus, so this one hook covers either side on a gs_usb adapter.
    # trace: the TEMPORARY opt-in frame trace (can_trace.py, MAKERMODSLAB_CAN_TRACE=1).
    _install_gs_usb_connect(DamiaoMotorsBus, "Damiao", wrap_errors=ConnectionError, trace=True)
