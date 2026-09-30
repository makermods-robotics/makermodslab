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
"""Preserve Maker fault evidence and healthy-joint commands without shaping targets.

Ordinary tracking forwards the latest leader action unchanged. Position lag is
not a fault; the existing driver retains its startup alignment and joint limits.
"""

import json
import logging
import math
import time
from pathlib import Path
from uuid import uuid4

from .maker_fault_diagnostics import MakerFaultDiagnostics
from .utils.config import MAKERMODSLAB_HOME

logger = logging.getLogger(__name__)
MAX_FEEDBACK_AGE_S = 0.25


class MakerTeleopSafety:
    def __init__(self, arm):
        self.arm = arm
        self.diagnostics = MakerFaultDiagnostics(arm.bus)
        self.send = arm.send_action
        self.previous = {}
        self.last_requested = {}
        self.saved_error = None
        arm.send_action = self.send_action
        arm.teleop_safety = self

    def check_feedback(self, names, *, require_all_healthy=True):
        faults = self.diagnostics.faults
        relevant = faults if require_all_healthy else {name: faults[name] for name in names if name in faults}
        if relevant:
            raise RuntimeError(f"Maker firmware fault: {relevant}")
        times = self.diagnostics.bus.last_feedback_time
        now = time.time()
        for name in names:
            stamp = times.get(name)
            if stamp is None or not 0 <= now - stamp <= MAX_FEEDBACK_AGE_S:
                raise RuntimeError(f"Maker {name}: feedback missing or older than 250 ms")

    def save_failure(self, exc):
        if self.saved_error is None:
            root = Path(MAKERMODSLAB_HOME) / "logs" / "teleoperation" / uuid4().hex
            try:
                root.mkdir(parents=True)
                (root / "diagnostics.json").write_text(json.dumps(self.diagnostics.snapshot(), indent=2))
                (root / "can_frames.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in self.diagnostics.recent)
                )
                self.saved_error = f"{str(exc).rstrip('.')}. CAN evidence: {root}"
            except OSError:
                logger.exception("Could not save Maker teleoperation CAN evidence")
                self.saved_error = str(exc)
            logger.error(self.saved_error)
        return RuntimeError(self.saved_error)

    def send_action(self, action):
        try:
            names = [key[:-4] for key in action if key.endswith(".pos")]
            # No extra observation read, slew cap, target-lead cap, or lag trip.
            # The driver's original startup alignment and joint limits still apply.
            for name in names:
                if not math.isfinite(action[f"{name}.pos"]):
                    raise ValueError("Non-finite Maker teleoperation target")
            self.last_requested = dict(action)
            self.check_feedback(names)
            # Retain the requested command even if a later joint faults mid-write,
            # so healthy joints can still receive their last goal.
            self.previous = {key: value for key, value in action.items() if key.endswith(".pos")}
            result = self.send(action)
            self.check_feedback(names)
            self.previous = {key: value for key, value in result.items() if key.endswith(".pos")}
            return result
        except Exception as exc:
            raise self.save_failure(exc) from exc

    def hold(self):
        """Refresh healthy joints' fixed commands; never clear or re-enable a fault."""
        healthy = {
            key: value
            for key, value in self.previous.items()
            if key[:-4] not in self.diagnostics.faults and key != "gripper.pos"
        }
        # Refresh only already-commanded goals, even after a slow/failed return
        # made feedback stale. A stale timestamp must not permanently stop the
        # commands that keep a brake-less arm energized. One failed joint must
        # not prevent servicing the remaining ones.
        for key, value in healthy.items():
            if key[:-4] in self.diagnostics.faults:
                continue
            try:
                self.send({key: value})
                self.check_feedback([key[:-4]], require_all_healthy=False)
            except Exception as exc:
                self.save_failure(exc)

    def recovery_drive(self):
        return MakerRecoveryDrive(self)


class MakerRecoveryDrive:
    """Land healthy arm joints independently of a stopped/faulted gripper.

    Use fresh feedback already received with MIT commands, avoiding the normal
    all-joint observation read (which includes the gripper and can clear faults
    in the pinned driver). Tracking still uses the ordinary follower path.
    """

    def __init__(self, guard):
        self.guard = guard
        self.bus = guard.arm.bus

    def get_observation(self):
        names = [name for name in self.bus.motors if name != "gripper"]
        self.guard.check_feedback(names, require_all_healthy=False)
        return {
            f"{name}.pos": self.bus._last_known_states[name]["position"] + self.guard.arm._turn_offset[name]
            for name in names
        }

    def send_action(self, action):
        names = [key[:-4] for key in action if key.endswith(".pos")]
        if "gripper" in names or any(not math.isfinite(action[f"{name}.pos"]) for name in names):
            raise ValueError("Invalid Maker recovery target")
        self.guard.check_feedback(names, require_all_healthy=False)
        # A later joint can fail after earlier writes in this action succeeded.
        # Retain this landing step even when no complete result comes back.
        self.guard.previous.update(action)
        result = self.guard.send(action)
        # Hold the last landing target if the return fails partway; never go
        # back to the leader's old, potentially distant target.
        self.guard.previous.update(result)
        self.guard.check_feedback(names, require_all_healthy=False)
        return result
