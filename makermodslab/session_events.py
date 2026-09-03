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
"""Session event bus: a tiny seam between the robot-driving feature modules
and the WebSocket ConnectionManager.

Feature modules call :func:`notify_session_changed` at their REAL state
transitions — right after a session's active flag is claimed, on final
release after cleanup, and at cheap intermediate phases — so every client of
the shared ``/ws/joint-data`` socket (the UI on any page, future SDK/remote
UIs) learns that session state moved and can refetch the relevant status
endpoint. server.py wires :func:`set_notifier` to the manager at import;
tests inject fakes.

The event is a droppable HINT, exactly like the ``jobs_changed`` broadcast:
consumers refetch on it and never trust the payload as state, so a missed or
dropped event is self-healing (every page already polls). That is also why
this module must never raise into a caller — a broadcast hiccup on a cleanup
path must not be the thing that leaves an arm energized.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

# The eight robot-driving features of the mutual-exclusion state model (see
# CLAUDE.md "State model & mutual exclusion"), matching the `robot.busy.*`
# error-code discriminants minus `releasing` — releasing is a transitional
# *phase* of a session, not a session kind of its own — and minus `training`,
# which holds the machine but is not a robot session.
SESSION_KINDS = frozenset(
    {
        "teleoperation",
        "recording",
        "inference",
        "remote_inference",
        "replay",
        "calibration",
        "auto_calibration",
        "wiggle",
    }
)

_notifier: Callable[[dict], None] | None = None

# In-process consumers of the same events, beside (not instead of) the WS
# notifier. The session identity tracker (sessions.py) lives here; anything
# else that needs to OBSERVE transitions without owning them can join. Kept
# separate from `_notifier` so tests that rewire the broadcast (set_notifier)
# never detach identity tracking by accident.
_subscribers: list[Callable[[dict], None]] = []


def set_notifier(cb: Callable[[dict], None] | None) -> None:
    """Install the callable that receives each session_changed event dict.

    server.py passes ``manager.notify_session_changed`` at import; tests pass
    fakes; ``None`` unwires (notify becomes a no-op)."""
    global _notifier
    _notifier = cb


def subscribe(cb: Callable[[dict], None]) -> None:
    """Add an in-process subscriber that receives every session_changed event.

    Subscribers are delivered to BEFORE the WS notifier, so a state-keeping
    subscriber (the session tracker) has committed the transition by the time
    any client acts on the broadcast hint. Idempotent: subscribing the same
    callable twice keeps a single entry."""
    if cb not in _subscribers:
        _subscribers.append(cb)


def unsubscribe(cb: Callable[[dict], None]) -> None:
    """Remove a subscriber; unknown callables are ignored."""
    if cb in _subscribers:
        _subscribers.remove(cb)


def notify_session_changed(kind: str, active: bool, phase: str | None = None) -> None:
    """Broadcast that a feature's session state changed.

    Builds the ``session_changed`` event and hands it to every subscriber and
    then the wired notifier. Never raises: an unknown ``kind`` is logged and
    dropped (a typo'd call site must not take down a hardware flow — the
    whitelist keeps the wire vocabulary in lockstep with the mutual-exclusion
    model), an unwired notifier is a no-op, and each consumer's exception is
    swallowed loudly and independently — one consumer blowing up must starve
    neither the other consumers nor the hardware flow that emitted the event.
    """
    if kind not in SESSION_KINDS:
        logger.error(f"Dropped session_changed event with unknown kind {kind!r} (active={active})")
        return
    subscribers = list(_subscribers)
    cb = _notifier
    if cb is None and not subscribers:
        return
    event = {
        "type": "session_changed",
        "session": {"kind": kind, "active": active, "phase": phase},
        "timestamp": time.time(),
    }
    for sub in subscribers:
        try:
            sub(event)
        except Exception as e:
            logger.warning(f"session_changed subscriber failed for {kind} (active={active}): {e}")
    if cb is None:
        return
    try:
        cb(event)
    except Exception as e:
        # A broadcast hiccup must never break the hardware flow that emitted
        # it — the event is only a refetch hint, and polling still covers it.
        logger.warning(f"session_changed notifier failed for {kind} (active={active}): {e}")
