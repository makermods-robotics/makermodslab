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

_original_sync_read = MotorsBus.sync_read


def _sync_read_with_default_retries(
    self, data_name, motors=None, *, normalize=True, num_retry=BUS_SYNC_READ_RETRIES
):
    """`MotorsBus.sync_read` with a non-zero default `num_retry`.

    An explicit `num_retry` from any caller still wins — this only changes the
    default that lerobot's own robot/teleop classes rely on."""
    return _original_sync_read(self, data_name, motors, normalize=normalize, num_retry=num_retry)


# Idempotent: the server re-imports its modules under uvicorn --reload, and
# re-patching an already-patched method would make `_original_sync_read` point
# at the patch and recurse.
if MotorsBus.sync_read.__name__ != "_sync_read_with_default_retries":
    MotorsBus.sync_read = _sync_read_with_default_retries
