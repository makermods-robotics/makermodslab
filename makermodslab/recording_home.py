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
"""Fixed zero-pose reset between CAN recording episodes."""

import logging
import math

from .maker_rest_pose import capture_maker_pose, maker_follower_arms, return_maker_arms_to_rest

logger = logging.getLogger(__name__)


class _Cancelled:
    def __init__(self, cancelled):
        self.cancelled = cancelled

    def is_set(self):
        return self.cancelled()


def return_recording_home(robot, speed_deg_s, cancelled, *, hold=None):
    """Return (arrived, targets); never bypass limits or command the gripper.

    Build and validate both arms' targets before commanding either. Failure
    ends recording through its graceful cleanup, rather than permitting the
    next capture or dropping torque here.
    """
    targets = []
    try:
        if cancelled():
            return False, targets
        for device, _label in maker_follower_arms(robot):
            present = capture_maker_pose(device)
            joints = set(device.bus.motors) - {"gripper"}
            if not joints or not joints <= present.keys():
                raise ValueError("Cannot read every arm joint for the zero return")
            pose = {}
            for motor in joints:
                low, high = device.config.joint_limits[motor]
                if not all(math.isfinite(v) for v in (low, high, present[motor])) or low > high:
                    raise ValueError(f"Invalid position or soft limits for {motor}")
                # Some calibrated limits stop a few degrees before literal zero.
                pose[motor] = max(low, min(high, 0.0))
            targets.append((device, pose))
        logger.info("Returning recording followers to zero at %.1f deg/s", speed_deg_s)
        verdicts = return_maker_arms_to_rest(
            targets,
            abort_event=_Cancelled(cancelled),
            target_label="zero within the joint limits",
            speed_deg_s=speed_deg_s,
            ensure_target=True,
        )
        if cancelled() or len(verdicts) != len(targets) or not all(ok for ok, _ in verdicts):
            logger.warning("Episode zero return did not complete; ending recording: %s", verdicts)
            return False, targets
        if hold is not None:
            hold(targets)
        if cancelled():
            return False, targets
        logger.info("Episode zero return complete; holding until the next episode")
        return True, targets
    except Exception:
        logger.exception("Episode zero return failed; ending recording")
        return False, targets
