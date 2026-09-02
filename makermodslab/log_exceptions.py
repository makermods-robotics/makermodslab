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

"""Make `logger.exception` actually print the exception again.

lerobot's `init_logging` (`lerobot/utils/utils.py`) installs a formatter and then
REPLACES its bound `format` method with a plain function:

    formatter = logging.Formatter()
    formatter.format = custom_format

`custom_format` renders level, timestamp, a 15-character slice of
`pathname:lineno`, and the message. It never looks at `record.exc_info` — and
appending the traceback is the one thing stock `logging.Formatter.format` does
that a message-only function cannot. So in every process that calls
`init_logging()`, **`logger.exception(...)` is byte-for-byte identical to
`logger.error(...)`**: the traceback is not truncated by the log pump or lost in
a pipe, it is never rendered at all.

That cost a full day. A takeover glide failed on a real session, the runner
logged "Could not glide the leader toward the follower" through
`logger.exception`, and the exception itself was simply absent from the log. The
consequence was not cosmetic: because the leader never moved, the takeover
offset came out at 113 degrees and the decay walked the FOLLOWER that far across
the workspace. Diagnosing it took reasoning from timing and from thirteen
unrelated failures in other sessions, when one line would have named it.

This wraps whatever formatter `init_logging` left behind, keeping its output
exactly as-is and appending the traceback when a record carries one. Wrapping
rather than replacing is deliberate: the lerobot format is what makes our logs
greppable against lerobot's own, and a session log is read side by side with
upstream's lines.

Call `restore_traceback_rendering()` AFTER `init_logging()`. It is idempotent —
a second call finds its own marker and does nothing, so a re-imported or
re-initialised module cannot stack wrappers.
"""

from __future__ import annotations

import logging

_MARKER = "_makermodslab_renders_tracebacks"


def restore_traceback_rendering() -> None:
    """Re-attach traceback rendering to every root handler's formatter."""
    for handler in logging.getLogger().handlers:
        formatter = handler.formatter
        if formatter is None or getattr(formatter, _MARKER, False):
            continue

        inner = formatter.format

        def _with_traceback(record: logging.LogRecord, _inner=inner, _fmt=formatter) -> str:
            text = _inner(record)
            if record.exc_info:
                # Cache on the record exactly as stock `Formatter.format` does,
                # so two handlers do not each pay to format the same traceback.
                if not record.exc_text:
                    record.exc_text = _fmt.formatException(record.exc_info)
                if record.exc_text:
                    text = f"{text}\n{record.exc_text}"
            return text

        formatter.format = _with_traceback
        setattr(formatter, _MARKER, True)
