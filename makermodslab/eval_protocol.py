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

"""Line protocol spoken between the eval orchestrator and the eval runner.

Its own module, deliberately free of any heavy import, because BOTH ends need
it and they live on opposite sides of a dependency wall:

  - the orchestrator (`makermodslab.rollout`) is imported by the FastAPI server at
    boot and must NOT drag in `lerobot.rollout` (torch, datasets, …);
  - the runner (`makermodslab.eval_runner`) is exactly the process that does.

Restating the strings in both modules would be a silent-drift bug the first
time one side renames a marker — the orchestrator would simply stop
recognising episode boundaries and every episode would look like a crash.

Commands — orchestrator → runner stdin, one bare word per line:
    EPISODE   run one rollout, up to the configured `--duration`
    STOP      end the RUNNING episode early (clean; not a kill)
    QUIT      return to the initial pose, disconnect, exit

Events — runner → stdout, one line each, carrying a grep-able prefix so the
orchestrator's log pump can pick them out of lerobot's own INFO chatter (the
runner's stderr is merged into the same pipe):
    MAKERMODSLAB-EVAL READY
    MAKERMODSLAB-EVAL EPISODE_STARTED
    MAKERMODSLAB-EVAL EPISODE_ENDED reason=duration
    MAKERMODSLAB-EVAL EPISODE_ENDED reason=stopped
    MAKERMODSLAB-EVAL ERROR <message, whitespace collapsed to one line>
    MAKERMODSLAB-EVAL BYE

The runner emits NO event for an episode cut short by QUIT: aborting a session
deliberately leaves the in-flight episode unscored (see `_abort_eval_locked`),
so reporting an end there would race the abort into scoring a verdict the user
never asked for.
"""

from __future__ import annotations

# --- Commands (stdin) -------------------------------------------------------
CMD_EPISODE = "EPISODE"
CMD_STOP = "STOP"
CMD_QUIT = "QUIT"

# --- Events (stdout) --------------------------------------------------------
# Prefix chosen to be unmistakable in a log full of lerobot INFO lines and
# unlikely to appear in a traceback.
EVENT_PREFIX = "MAKERMODSLAB-EVAL"

EVENT_READY = "READY"
EVENT_EPISODE_STARTED = "EPISODE_STARTED"
EVENT_EPISODE_ENDED = "EPISODE_ENDED"
EVENT_ERROR = "ERROR"
EVENT_BYE = "BYE"

# `EPISODE_ENDED reason=` values. `stopped` is the user's "task succeeded"
# button, `duration` is the episode running out its clock — the orchestrator
# maps them onto the success/failure verdicts.
REASON_DURATION = "duration"
REASON_STOPPED = "stopped"


def format_event(event: str, payload: str = "") -> str:
    """Render one protocol line.

    The payload's whitespace is collapsed so a multi-line exception message can
    never split one event across several lines — the reader is line-oriented and
    a wrapped traceback would otherwise be read as several unknown events."""
    body = " ".join(str(payload).split())
    return f"{EVENT_PREFIX} {event} {body}".rstrip()


def parse_event(line: str) -> tuple[str, str] | None:
    """`(event, payload)` for a protocol line, or None when the line isn't one.

    Matches the prefix anywhere in the line rather than only at the start: the
    runner's logging handler writes to the same merged pipe, and a log record
    flushed without its trailing newline would otherwise swallow the event that
    follows it on the wire. A line that merely mentions the prefix can't be
    produced by the runner (it only ever writes it via `format_event`)."""
    idx = line.find(EVENT_PREFIX)
    if idx < 0:
        return None
    rest = line[idx + len(EVENT_PREFIX) :].strip()
    if not rest:
        return None
    event, _, payload = rest.partition(" ")
    return event, payload.strip()


def parse_episode_end_reason(payload: str) -> str:
    """Pull `reason=<value>` out of an EPISODE_ENDED payload.

    Returns "" when the payload carries no recognisable reason, which the
    orchestrator treats as "ended, cause unknown" rather than guessing a
    verdict — a renamed/absent reason must not silently score an episode."""
    for token in payload.split():
        key, sep, value = token.partition("=")
        if sep and key == "reason":
            return value
    return ""
