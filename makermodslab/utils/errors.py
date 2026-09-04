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

"""Source-agnostic error-text helpers shared by the robot-driving features.

Turns a raw error string into a plain-language, actionable hint, and knows
which errors mean "the run actually worked, only shutdown/cleanup complained".

Deliberately free of any rollout/recording/teleop specifics: the input is just
text, so every feature can reuse it regardless of where the text came from.
Rollout mines it out of a failed subprocess's log tail (see rollout.py);
recording/teleop run in-process and will pass the message of a caught exception
(`str(exc)`) instead — same functions, different source. Training is a third
source: jobs.py mines a finished run's log the same way rollout does, local and
cloud alike (see is_out_of_memory).
"""

from __future__ import annotations

# Errors that mean the policy/robot actually ran and only shutdown/cleanup
# tripped — e.g. disabling torque on a gripper still holding an object.
# Connection-loss errors are deliberately excluded: a mid-run disconnect is a
# real failure, not a noisy-cleanup warning.
CLEANUP_MARKERS: tuple[str, ...] = ("overload", "torque_enable")


def is_cleanup_error(error_text: str | None) -> bool:
    """True when the error text matches a known shutdown/cleanup-only failure
    (see CLEANUP_MARKERS). Case-insensitive; None/empty text is False.

    Callers use this to decide whether a non-zero/raised failure that happened
    *after* the run got going is a real failure or just noisy teardown."""
    if not error_text:
        return False
    low = error_text.lower()
    return any(marker in low for marker in CLEANUP_MARKERS)


# An allocator running out of memory, keyed on what each backend actually
# prints: PyTorch's CUDA allocator raises "torch.OutOfMemoryError: CUDA out of
# memory", ROCm and MPS have their own wording, and a host-RAM allocation
# failure surfaces through DefaultCPUAllocator. Deliberately NOT keyed on a
# bare "killed": the Linux OOM killer's SIGKILL leaves no text in the log at
# all, so that case is recognised from the exit code by the caller.
OOM_MARKERS: tuple[str, ...] = (
    "cuda out of memory",
    "hip out of memory",
    "mps backend out of memory",
    "outofmemoryerror",
    "cuda error: out of memory",
    "defaultcpuallocator: can't allocate memory",
)


def is_out_of_memory(error_text: str | None) -> bool:
    """True when the error text is an allocator running out of memory (see
    OOM_MARKERS). Case-insensitive; None/empty text is False.

    Training's most common hard failure and, until this existed, its most
    silent one: the trainer dies on the first step, the platform reports only
    that the process exited non-zero, and the actual cause sits in a log the
    user has to go find and read."""
    if not error_text:
        return False
    low = error_text.lower()
    return any(marker in low for marker in OOM_MARKERS)


def format_exception(exc: BaseException, limit: int = 500) -> str:
    """Format a caught exception as a short "Type: message" line for a status
    payload, truncated to `limit` characters.

    The in-process features (recording, teleoperation) hold the actual
    exception object at their catch sites, so — unlike rollout's subprocess
    log forensics — the error text comes straight from here."""
    text = f"{type(exc).__name__}: {exc}".strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text


def classify_outcome(work_completed: bool, error_text: str | None) -> str:
    """ok | ran_with_warning | failed, for an IN-PROCESS session's catch site.

    `work_completed` is the caller's phase flag: True when the session's real
    work was already done when the error was raised (episodes saved for
    recording; the teleop loop ran and the user requested the stop) — then a
    raised/reported error means only teardown/cleanup tripped (e.g. disabling
    torque on a gripper still holding an object), which is a warning, not a
    failed session. An error before/at any other phase (setup, mid-episode,
    mid-loop) is a real failure. No error means the session was fine.

    The catch-site structure is the classifier — deliberately NOT the
    CLEANUP_MARKERS text match (that's rollout's fallback, which only knows a
    subprocess's log tail, not where the failure was raised)."""
    if not error_text:
        return "ok"
    return "ran_with_warning" if work_completed else "failed"


def friendly_hint(error_text: str | None) -> str | None:
    """A plain-language, actionable headline for the common SO-101 and
    training failures, or None when the text doesn't match a known pattern.

    Pure text → hint: pass a subprocess log-tail snippet or the message of a
    caught exception — nothing here is coupled to how the text was obtained."""
    if not error_text:
        return None
    low = error_text.lower()
    # Checked first: an OOM traceback often carries a device/allocator string
    # that would otherwise trip one of the connection branches below, and it
    # is unambiguous when it matches.
    if is_out_of_memory(error_text):
        return (
            "The GPU ran out of memory. Turn on mixed precision (AMP), lower the batch size, "
            "or run on a larger GPU."
        )
    # lerobot's own `require_package(pkg, extra=...)` message, which names the
    # extra in a form no other error uses: "'scipy' is required but not
    # installed. Install it with: pip install 'lerobot[molmoact2]'". Matched on
    # the bracketed extra so the hint keeps working whichever of the extra's
    # three packages is the missing one.
    if "lerobot[molmoact2]" in low:
        return (
            "This MolmoAct2 checkpoint needs an optional package that isn't installed here. "
            "Install it with `pip install 'lerobot[molmoact2]'` (transformers + peft + scipy), "
            "then start the run again."
        )
    # MolmoAct2 is the one policy whose checkpoint can be internally unrunnable:
    # `inference_action_mode` has no default (the policy raises on None), and a
    # checkpoint trained for one head cannot be run with the other. MakerMods
    # Lab fills a MISSING mode in from the checkpoint's own training mode
    # (rollout._rollout_cli_args), so what reaches here is a real head mismatch.
    #
    # Keyed on the raised sentences, NOT on the bare field name: lerobot logs
    # the effective policy config, so `inference_action_mode` alone appears in
    # the log tail of runs that failed for entirely unrelated reasons.
    if (
        "cannot run continuous inference" in low
        or "cannot run discrete inference" in low
        or "requires `inference_action_mode` to be set explicitly" in low
    ):
        return (
            "This MolmoAct2 checkpoint can't run the action head it was asked for. It was "
            "trained for discrete or continuous actions, not both — pick a checkpoint whose "
            "action_mode matches, or re-export it with action_mode='both'."
        )
    if "libtorchcodec" in low or "library not loaded: @rpath/libavutil" in low:
        return (
            "The trainer's video decoder (torchcodec) couldn't load its FFmpeg libraries on this "
            "machine. Install ffmpeg (macOS: brew install ffmpeg; Ubuntu: sudo apt install ffmpeg) "
            "or retry — newer MakerMods Lab falls back to the built-in pyav decoder automatically."
        )
    if "overload" in low or "torque_enable" in low:
        return (
            "A motor overloaded — usually the gripper holding an object too hard. Release the object / "
            "open the gripper and power-cycle the arm before trying again."
        )
    if "missing motor ids" in low or "motor check failed" in low:
        return (
            "A follower motor isn't responding (often the gripper, id 6). If a policy was holding an object "
            "it likely overloaded — remove it, power-cycle the arm, then try teleoperation first."
        )
    # Servo bus comms: lerobot's motors_bus raises these as ConnectionError with
    # a "Failed to <read|write|sync read|sync write> ... [TxRxResult] ..." body
    # when a servo doesn't answer or answers with a corrupt packet (arm powered
    # down mid-session, a loose daisy-chain link, a half-seated cable). Keyed on
    # the bus-specific phrasing, and placed before the Hub branches because the
    # exception TYPE name is the same "ConnectionError" a download failure has.
    # `in_download_step` is rollout's own prefix for anything raised inside the
    # model fetch — the arm has not been touched yet, so the generic
    # read/write wording below cannot be about a servo there.
    in_download_step = "failed to download the model" in low
    if not in_download_step and (
        "txrxresult" in low
        or "incorrect status packet" in low
        or "failed to sync read" in low
        or "failed to sync write" in low
        or "failed to write" in low
        or "failed to read" in low
    ):
        return (
            "A motor stopped answering on the servo bus — usually the arm lost power, or a servo "
            "cable/daisy-chain link is loose. Power-cycle the arm, re-seat the cables, then try "
            "teleoperation before inference."
        )
    # Hub model-download failures (snapshot_download, before the arm is ever
    # touched). Keyed on hub-specific tokens so a network/404/disk error while
    # fetching a checkpoint isn't mistaken for an arm-connection problem below.
    if "no space left" in low or "disk quota exceeded" in low:
        return "Ran out of disk space downloading the model — free up space in the Hugging Face cache and try again."
    if (
        "repository not found" in low
        or "repositorynotfound" in low
        or "gatedrepo" in low
        or "gated repo" in low
        or ("404" in low and ("huggingface" in low or "hf.co" in low or "repo" in low))
    ):
        return "Couldn't find the model on the Hub — check the repo id, and that you have access if it's private or gated."
    # The trigger tokens must be HUB-specific. A bare "connectionerror" is not:
    # lerobot's motors bus raises ConnectionError for every serial failure, and
    # the type name itself contains "connect", so keying on it labelled arm-side
    # startup crashes "couldn't download the model". `in_download_step` (rollout's
    # own prefix) is the one marker that says the failure really was the fetch.
    if ("huggingface.co" in low or "hf.co" in low or "max retries" in low or in_download_step) and (
        "connect" in low or "reach" in low or "retries" in low or "timed out" in low or "timeout" in low
    ):
        return "Couldn't download the model — check your internet connection, then confirm the repo id."
    if "could not connect" in low or "failed to connect" in low or "not connected" in low:
        return "Couldn't connect to the arm — make sure it's plugged in, powered on, and on the right port."
    if "frame is too old" in low or "no frame" in low or "frame timeout" in low:
        return (
            "A camera can't keep up — frames are arriving too slowly. Lower its resolution/FPS, "
            "set FOURCC=MJPG, and close other heavy apps, then try again."
        )
    if "failed to set capture_" in low or "actual_width" in low or "actual_height" in low:
        return (
            "A camera didn't come up in the configured resolution — open camera settings and click Auto. "
            "If it keeps failing the same way (even without unplugging anything), the camera's OS-level "
            "session is likely stuck — restart MakerMods Lab to clear it."
        )
    # Same read-back failure as the resolution case above, from lerobot's fps
    # negotiation step instead (_validate_fps). Both are driven by free-form
    # numbers in the same camera-settings panel, so both can be a permanent
    # "this device can't do that" rather than turbulence. Without this branch
    # an unsupported fps was the one camera misconfiguration that got retried
    # (record.py's _is_transient_camera_error matches it) and then surfaced
    # with no guidance at all — strictly worse than the resolution case.
    if "failed to set fps" in low or "actual_fps" in low:
        return "A camera doesn't support the configured frame rate — open camera settings and click Auto."
    if "permission" in low and ("port" in low or "com" in low):
        return "Couldn't open the serial port — close anything else using it, or run `makermodslab --stop`."
    return None
