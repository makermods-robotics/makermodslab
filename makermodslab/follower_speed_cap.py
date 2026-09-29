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
"""Bound how fast, and how hard, a CAN follower's commanded target may move.

Once the pinned follower's startup sync finishes it forwards every leader
position straight to the motors. A CAN trace of a Metal arm that died
mid-teleop showed the whole motor supply dropping out and every motor
rebooting together, right after the big joints were driven at up to 321 deg/s
and then braked at ~1300 deg/s^2: a supply trip from current draw and/or the
energy braking motors push back into it.

So the TARGET each joint is sent follows a per-joint speed AND acceleration
limit. Speeding up is bounded, and so is braking: the target starts slowing
early enough to stop on the leader's position instead of overshooting it
(the classic ``v <= sqrt(2 * a * distance)`` profile).

Everything is time-based, not per-tick, so it means the same whatever rate the
control loop runs at, and a gap (a paused recording, a slow tick) cannot bank
motion: the elapsed time counted per call is bounded.
"""

import math
import time
from collections.abc import Callable

# Longest interval one call may claim. A loop that stalls for a second must
# not be allowed a full second's worth of travel in its next command.
MAX_STEP_INTERVAL_S = 0.05


class FollowerSpeedCap:
    """Wrap one follower arm's ``send_action`` with per-joint speed/acceleration limits.

    The first command for a joint passes through untouched: the pinned
    follower's own startup sync owns the initial approach. After that each
    ``<joint>.pos`` target is advanced from the last target the driver
    actually accepted (its returned action, so joint-limit clipping and startup
    sync stay the reference) with its velocity bounded by ``max_deg_s`` and
    its change in velocity by ``max_deg_s2``. Every other caller of
    ``send_action`` on the arm — rest returns, re-alignment, episode home —
    goes through the same limits, which sit far above their own rates.

    Joints named in ``exempt`` (bare motor names, e.g. ``"gripper"``) are
    forwarded untouched.
    """

    def __init__(
        self,
        arm,
        max_deg_s: float,
        max_deg_s2: float = math.inf,
        clock: Callable[[], float] = time.monotonic,
        exempt: tuple[str, ...] = (),
    ) -> None:
        if not math.isfinite(max_deg_s) or max_deg_s <= 0:
            raise ValueError("The follower speed cap must be a positive number of degrees per second")
        if math.isnan(max_deg_s2) or max_deg_s2 <= 0:
            raise ValueError(
                "The follower acceleration cap must be a positive number of degrees per second^2"
            )
        self.arm = arm
        self.max_deg_s = float(max_deg_s)
        self.max_deg_s2 = float(max_deg_s2)
        self._clock = clock
        self._exempt = {f"{name}.pos" for name in exempt}
        self._send = arm.send_action
        self._last: dict[str, float] = {}
        self._velocity: dict[str, float] = {}
        self._last_time: float | None = None
        arm.send_action = self.send_action
        arm.follower_speed_cap = self

    def _advance(self, key: str, previous: float, goal: float, dt: float) -> float:
        error = goal - previous
        velocity = self._velocity.get(key, 0.0)
        change = self.max_deg_s2 * dt
        # Fastest speed toward the goal that can still brake to a stop on it,
        # counted in whole ticks (each shedding ``change`` of speed): the
        # continuous sqrt(2*a*d) brakes a tick too late and ends in a jolt.
        # Unlimited acceleration is spelled out: inf * 0 would be NaN.
        if math.isinf(self.max_deg_s2):
            braking_limit = self.max_deg_s
        else:
            braking_limit = 0.5 * (-change + math.sqrt(change * change + 8.0 * abs(error) * change / dt))
        wanted = math.copysign(min(self.max_deg_s, braking_limit), error)
        velocity += max(-change, min(change, wanted - velocity))
        velocity = max(-self.max_deg_s, min(self.max_deg_s, velocity))
        move = velocity * dt
        # Never step past the goal while heading toward it.
        if (error >= 0 and move > error) or (error <= 0 and move < error):
            move = error
        return previous + move

    def send_action(self, action: dict) -> dict:
        now = self._clock()
        elapsed = MAX_STEP_INTERVAL_S if self._last_time is None else now - self._last_time
        dt = min(max(elapsed, 0.0), MAX_STEP_INTERVAL_S)
        capped = dict(action)
        for key, value in action.items():
            previous = self._last.get(key)
            if (
                previous is None
                or not key.endswith(".pos")
                or key in self._exempt
                or not math.isfinite(value)
            ):
                continue
            capped[key] = self._advance(key, previous, value, dt) if dt > 0 else previous
        result = self._send(capped)
        self._last_time = now
        accepted = result if isinstance(result, dict) else capped
        for key, value in accepted.items():
            if not (key.endswith(".pos") and isinstance(value, (int, float)) and math.isfinite(value)):
                continue
            previous = self._last.get(key)
            # The velocity the target really moved at (clipping included) is
            # what the next tick accelerates from.
            self._velocity[key] = (float(value) - previous) / dt if previous is not None and dt > 0 else 0.0
            self._last[key] = float(value)
        return result


def install_follower_speed_cap(
    arms, max_deg_s: float, max_deg_s2: float = math.inf, exempt: tuple[str, ...] = ()
) -> list[FollowerSpeedCap]:
    """Limit each ``(arm, label)`` pair's commanded motion; returns the installed caps.

    An object without a callable ``send_action`` cannot be commanded at all,
    so there is nothing to bound and it is left as it is.
    """
    return [
        FollowerSpeedCap(arm, max_deg_s, max_deg_s2, exempt=exempt)
        for arm, _label in arms
        if callable(getattr(arm, "send_action", None))
    ]
