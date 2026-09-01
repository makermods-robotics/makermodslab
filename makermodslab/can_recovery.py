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
"""De-energize a CAN arm after a crash — the recovery no other path can run.

Every normal stop path releases torque on its way out. What none of them can
reach is an arm whose PROCESS died: a SIGKILL or power loss leaves Damiao
motors holding their last MIT command indefinitely (RobStride at least times
out), and the next server process starts with no device object, no session,
and no cleanup owing — just a rigid arm on a port. This module is the
deliberate, user-invoked answer: open the named follower bus WITHOUT the
energizing handshake, broadcast the disable, close.

Deliberately NOT a session. It must be usable exactly when session state is
wrecked, so it takes no lease, emits no session events, and appears in no
STARTABLE_KINDS — the same tier as wiggle, but removing energy instead of
adding it. The one gate it does respect is ``sessions._held_by()``: while a
live session is driving the hardware, a de-energize would fight it
mid-motion, so the request is refused with the same 409 the start paths use.

SO-101 arms are refused at the schema level: a Feetech arm goes limp on its
own when its process dies, so there is nothing to recover, and pointing a CAN
de-energize at a serial port would be nonsense.
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel

from .api_errors import ApiError, ErrorCode
from .torque import de_energize_can_device

logger = logging.getLogger(__name__)


class ReleaseCanTorqueRequest(BaseModel):
    # Only the CAN families: an SO-101 has nothing to recover (see module
    # docstring), so "so101" is rejected by the schema rather than no-opped.
    arm_type: Literal["maker", "metal"]
    # The follower's CAN adapter port (the leader has no torque to release).
    port: str


def _build_follower_device(arm_type: str, port: str):
    """The follower device whose bus gets de-energized.

    Built through the same factory helpers the calibration flows use, with a
    throwaway id — construction does no device I/O, and the calibration file
    (if any) is irrelevant to a torque release.
    """
    from lerobot.robots import make_robot_from_config

    from .utils.robot_factory import maker_follower_config, metal_follower_config

    builder = metal_follower_config if arm_type == "metal" else maker_follower_config
    return make_robot_from_config(builder(port, "recovery"))


def handle_release_can_torque(request: ReleaseCanTorqueRequest) -> dict[str, Any]:
    """Open the named CAN bus without the handshake and disable torque.

    Returns ``{"success", "message", "problems"}``; success means every
    disable landed. Refuses with 409 session.held while any feature's active
    flag holds the hardware.
    """
    from .sessions import _held_by

    holder = _held_by()
    if holder is not None:
        raise ApiError(
            status_code=409,
            detail=f"The robot hardware is held by an active {holder} session. Stop it first.",
            code=ErrorCode.SESSION_HELD,
            details={"holder": {"kind": holder, "session_id": None}},
        )

    family = "Metal" if request.arm_type == "metal" else "Maker"
    logger.info(f"Releasing torque on the {family} follower at {request.port} (crash recovery)")
    device = _build_follower_device(request.arm_type, request.port)
    problems = de_energize_can_device(device, f"{family} follower arm")
    if problems:
        return {
            "success": False,
            "message": f"Torque release on {request.port} reported problems.",
            "problems": problems,
        }
    return {
        "success": True,
        "message": f"Torque released on the {family} follower at {request.port}. The arm is limp.",
        "problems": [],
    }
