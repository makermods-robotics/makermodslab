"""The ``recording`` namespace: controls for a LIVE recording session.

The session itself starts through ``client.sessions.record(...)``; these are
the in-flight extras that grew beyond the legacy flow routes. ``task`` set
here applies to the episode currently being recorded (the
``per_episode_task`` recording option enables per-episode prompts).
"""

from __future__ import annotations

from urllib.parse import quote

from makermodslab_sdk._operations import operation
from makermodslab_sdk.resources._base import Resource, SdkModel


class RecordingControlResult(SdkModel):
    """The control verbs' shared answer — ``success=False`` carries the
    reason in ``message`` (e.g. no recording is running), not an HTTP error."""

    success: bool
    message: str


class RecordingResource(Resource):
    """``client.recording`` — live-recording extras.

    Example:
        >>> with client.sessions.record(
        ...     "bench", dataset_repo_id="u/d", single_task="pick", per_episode_task=True
        ... ) as s:
        ...     client.recording.set_episode_task("pick the red cube")
    """

    @operation("recording_episode_task")
    def set_episode_task(self, task: str) -> RecordingControlResult:
        """Set the task prompt for the episode being recorded right now
        (needs the session started with ``per_episode_task=True``)."""
        return RecordingControlResult.model_validate(
            self._transport.request(
                "POST",
                "/api/v1/recording-episode-task",
                json={"task": task},
                action="Set recording episode task",
            )
        )

    @operation("recording_camera_preview")
    def camera_preview_url(self, camera_name: str) -> str:
        """The live JPEG preview URL for one of the recording session's
        cameras.

        Deliberately returns the URL instead of reading the stream — it is
        unbounded, and an agent must never be handed an endless read (the
        ``sample_joints`` rule). Point a browser/<img> at it.
        """
        return f"{self._transport.base_url}/api/v1/recording-preview/{quote(camera_name, safe='')}"
