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
"""Narrow compatibility hook for the pinned Maker driver's port-only configuration."""

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


def install():
    from lerobot.motors.robstride import RobstrideMotorsBus

    _install_gripper_open_limit()

    original = RobstrideMotorsBus.connect
    if getattr(original, "_makermodslab_gs_usb", False):
        return

    @wraps(original)
    def connect(self, handshake=True):
        if not isinstance(self.port, str) or not self.port.startswith("gs_usb:"):
            return original(self, handshake=handshake)
        from .gs_usb_transport import open_gs_usb

        if self.is_connected:
            raise RuntimeError("RobStride bus is already connected")
        if self.use_can_fd:
            raise ValueError("Maker gs_usb supports classic CAN only")
        bus = open_gs_usb(self.port, self.bitrate)
        self.canbus = bus
        self._is_connected = True
        try:
            if handshake:
                self._handshake()
        except BaseException as exc:
            self._is_connected = False
            self.canbus = None
            try:
                bus.shutdown()
            except Exception as cleanup:
                exc.add_note(f"USB cleanup also failed: {cleanup}")
            raise

    connect._makermodslab_gs_usb = True
    RobstrideMotorsBus.connect = connect
