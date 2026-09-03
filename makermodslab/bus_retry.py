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

"""Process-wide retry defaults for motor-bus reads AND single-register writes.

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

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import MotorsBus, SerialMotorsBus

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


# --- writes -----------------------------------------------------------------
#
# Reads were patched first because a dropped read was the failure we had seen.
# The logs since say the real problem is WRITES: across every session ever
# recorded on this rig, the bus has failed 13 times, and every single one was an
# unretried write —
#
#     7 x  Failed to write 'Lock'
#     6 x  Failed to write 'Torque_Enable'
#
# — against exactly one `sync_read` failure, which predates the patch above.
# Both of those come from `FeetechMotorsBus.enable_torque`, which issues two
# blocking writes per motor (twelve for a six-motor arm), each demanding a status
# packet, each raising `ConnectionError` on a single missed reply. It accepts a
# `num_retry` and passes it through, but every caller in lerobot takes the
# default of zero.
#
# That is what makes a takeover glide fail: `teleop_smooth_move_to`'s first
# statement is `teleop.enable_torque()`, so twelve unretried writes stand between
# the operator pressing space and the leader moving. One lost status packet there
# aborted the whole handover — and because the leader then never moved, the
# takeover offset came out at 113 degrees and walked the FOLLOWER across the
# workspace instead.
#
# Retrying is safe here in a way it would not be everywhere: these are register
# SETS (`Torque_Enable`, `Lock`, `Goal_Position`), so writing the same value
# twice is idempotent, and the common failure is a landed write whose status
# packet was lost — the retry simply re-asserts a value that is already correct.
#
# `sync_write` is deliberately NOT patched. It does not ask for a status packet
# at all (so it cannot fail this way), it carries the 30Hz control-loop traffic
# where an extra round trip is a real cost, and it has never once appeared in a
# failure. Retrying reads and config writes is cheap insurance; retrying the
# stream that drives the arm is not.
_installed_write = MotorsBus.write
_original_write = getattr(_installed_write, "_bus_retry_original", _installed_write)


def _write_with_default_retries(
    self, data_name, motor, value, *, normalize=True, num_retry=BUS_SYNC_READ_RETRIES
):
    """`MotorsBus.write` with a non-zero default `num_retry`.

    An explicit `num_retry` from any caller still wins — this only changes the
    default that lerobot's own robot/teleop classes rely on."""
    return _original_write(self, data_name, motor, value, normalize=normalize, num_retry=num_retry)


_write_with_default_retries._bus_retry_original = _original_write

MotorsBus.write = _write_with_default_retries


# --- torque toggles ---------------------------------------------------------
#
# Patching `write`'s DEFAULT is not enough, and the first version of this change
# shipped believing it was. `FeetechMotorsBus.enable_torque` takes its own
# `num_retry: int = 0` and passes it to `write` EXPLICITLY:
#
#     def enable_torque(self, motors=None, num_retry: int = 0):
#         for motor in self._get_motors_list(motors):
#             self.write("Torque_Enable", motor, ..., num_retry=num_retry)
#             self.write("Lock", motor, 1, num_retry=num_retry)
#
# — and an explicit argument beats a patched default, exactly as the note on
# `_write_with_default_retries` says. So the retries never reached the twelve
# writes that stand between the operator pressing space and the leader moving.
# The proof arrived within minutes of the traceback fix landing, from a real
# session: `Failed to write 'Torque_Enable' on id_=3 with '1' after 1 tries` —
# ONE try, on a build that was supposed to make three. Every `Failed to write`
# ever recorded on this rig says `after 1 tries`; not one says 3.
#
# So the toggles get their own default too. Both directions: a failed
# `disable_torque` leaves an arm rigid that the operator was told is free, which
# is the worse of the two outcomes.
def _patched_toggle(cls, name):
    installed = vars(cls)[name]
    original = getattr(installed, "_bus_retry_original", installed)

    def toggle(self, motors=None, num_retry=BUS_SYNC_READ_RETRIES):
        return original(self, motors, num_retry=num_retry)

    toggle._bus_retry_original = original
    toggle.__name__ = name
    return toggle


# Patched on EVERY class in the chain that defines its own, not just the base.
# `FeetechMotorsBus` overrides `enable_torque`/`disable_torque`, so patching
# `MotorsBus` alone leaves the SO-101's actual bus untouched — which is the
# second way this change was silently inert, caught only by reading the
# resolved signature off the concrete class rather than the one we patched.
for _cls in (MotorsBus, SerialMotorsBus, FeetechMotorsBus):
    for _name in ("enable_torque", "disable_torque"):
        if _name in vars(_cls):
            setattr(_cls, _name, _patched_toggle(_cls, _name))
