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

"""Process-wide retry default for motor-bus position reads.

lerobot's SO-10x follower/leader call ``bus.sync_read("Present_Position")`` with
the default ``num_retry=0``, so a SINGLE missed reply kills the whole session
with ``[TxRxResult] There is no status packet!``. Replies do get missed: on
hosts where the arm serial adapters share a USB bus with streaming cameras
(the Jetson rig, and the RTX station — cameras and CH34x adapters on the same
hubs), isochronous camera traffic occasionally delays a bulk serial reply past
the packet timeout. Position reads are idempotent, and a retry costs only the
packet timeout (single-digit ms at 1 Mbps) *when a read actually glitched* —
invisible at a 30 Hz loop.

IMPORTING THIS MODULE APPLIES THE PATCH. That is the whole point: a
monkeypatch on a class default only affects the interpreter that ran it, so
every process that drives a bus has to import it for itself.

That is also why this lives here rather than in record.py, where it started.
Recording runs IN the FastAPI server, so the server had the retries and the two
subprocesses that also drive arms — ``eval_runner`` and ``dagger_runner`` — did
not. Both inherited lerobot's zero-retry default, and both died outright on a
single dropped packet. Coaching made that visible because it reads TWO buses
per tick (follower observation plus leader action) while two cameras stream,
which is exactly the traffic pattern the comment above describes.
"""

from __future__ import annotations

from lerobot.motors.motors_bus import MotorsBus

# Attempts beyond the first. Two extra tries covers the observed single-packet
# glitches without masking a genuinely disconnected arm, which fails all three.
BUS_SYNC_READ_RETRIES = 2

# The true original, recovered rather than assumed.
#
# This line runs again on every re-import — uvicorn --reload does it, and so
# does `importlib.reload`. If it simply read `MotorsBus.sync_read`, the second
# run would capture OUR OWN patch as "the original" and every motor read would
# recurse until the stack blew. The guard below used to be the only defence,
# and it does not help: it prevents re-assignment, not re-capture, so the
# damage was already done by the time it was evaluated.
#
# So the patch carries a reference to what it wrapped, and we prefer that over
# whatever is currently installed.
_installed = MotorsBus.sync_read
_original_sync_read = getattr(_installed, "_bus_retry_original", _installed)


def _sync_read_with_default_retries(
    self, data_name, motors=None, *, normalize=True, num_retry=BUS_SYNC_READ_RETRIES
):
    """`MotorsBus.sync_read` with a non-zero default `num_retry`.

    An explicit `num_retry` from any caller still wins — this only changes the
    default that lerobot's own robot/teleop classes rely on."""
    return _original_sync_read(self, data_name, motors, normalize=normalize, num_retry=num_retry)


# Stashed so a later re-import can find its way back to lerobot's own method
# instead of wrapping this one.
_sync_read_with_default_retries._bus_retry_original = _original_sync_read

# Safe to assign unconditionally now that `_original_sync_read` is guaranteed to
# be lerobot's method and never a previous copy of this patch. Re-importing
# swaps one equivalent wrapper for another.
MotorsBus.sync_read = _sync_read_with_default_retries
