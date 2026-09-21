"""The blocking poll loop behind the namespaces' ``wait_for_*`` ergonomics.

The server's transfer managers (the dataset/model DownloadManager, record.py's
UploadManager) are single-slot state machines polled through a ``*_status``
operation: ``state`` walks idle -> running -> done/error and ``repo_id`` names
the one operation in the slot. This loop turns that into a blocking wait.

Timeouts are measured in *virtual* time — the sum of the intervals handed to
``sleep_fn`` — so tests inject a recording fake and stay deterministic without
ever sleeping (house style), while the default ``time.sleep`` gives real
callers real pacing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from makermodslab_sdk.errors import MakerModsError

__all__ = ["OperationFailedError", "WaitTimeoutError", "wait_for_repo_operation"]


class WaitTimeoutError(MakerModsError):
    """The wait budget elapsed while the operation was still running.

    The operation itself keeps running server-side — poll the ``*_status``
    operation, or call the same ``wait_for_*`` again to keep waiting.
    """


class OperationFailedError(MakerModsError):
    """The waited-on server-side operation failed, or was never the one
    running — the message carries the server's error text."""


def wait_for_repo_operation(
    poll: Callable[[], Any],
    *,
    repo_id: str,
    describe: str,
    status_call: str,
    timeout: float,
    poll_interval: float,
    sleep_fn: Callable[[float], None],
) -> Any:
    """Poll ``poll()`` until the slot reports ``repo_id`` done; return that status.

    ``describe`` names the operation in error prose ("dataset download");
    ``status_call`` is the literal polling call an agent should make next
    ("client.datasets.download_status()").
    """
    elapsed = 0.0
    while True:
        status = poll()
        if status.state == "idle":
            raise OperationFailedError(
                f"{describe} of {repo_id}: nothing is running ({status_call} reports state='idle') — "
                f"start the {describe} first, then wait."
            )
        if status.repo_id != repo_id:
            raise OperationFailedError(
                f"{describe} of {repo_id}: the server's single {describe} slot holds "
                f"{status.repo_id!r} (state={status.state!r}) — the {describe} of {repo_id} was never "
                f"started, or another one replaced its result. Start it and wait again."
            )
        if status.state == "error":
            error_text = getattr(status, "error", None) or status.message or "no error detail"
            raise OperationFailedError(f"{describe} of {repo_id} failed: {error_text}")
        if status.state == "done":
            return status
        if elapsed >= timeout:
            raise WaitTimeoutError(
                f"{describe} of {repo_id} still running after {timeout:g}s — it continues server-side; "
                f"poll {status_call} or call this wait again."
            )
        sleep_fn(poll_interval)
        elapsed += poll_interval
