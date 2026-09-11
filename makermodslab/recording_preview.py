"""Read-only previews tapped from recording observations; never opens hardware.

Keep only the latest sampled RGB images. JPEG encoding runs in HTTP workers,
not in the control loop. Short snapshot requests avoid exhausting a browser's
per-host connection pool when a robot has many cameras.
"""

import logging
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RecordingPreview:
    def __init__(self):
        self._lock = threading.Lock()
        self._active = False
        self._frames = {}
        self._last_sample = float("-inf")
        self.joint_notifier: Callable[[dict], None] | None = None

    def start(self):
        with self._lock:
            self._frames = {}
            self._last_sample = float("-inf")
            self._active = True

    def stop(self):
        with self._lock:
            self._active = False
            self._frames = {}

    def publish(self, observation: dict) -> bool:
        """Copy at most 10 Hz; camera backends may reuse their RGB buffers."""
        with self._lock:
            now = time.monotonic()
            if not self._active or now - self._last_sample < 0.1:
                return False
            self._last_sample = now
            self._frames = {
                name: frame.copy()
                for name, frame in observation.items()
                if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3
            }
            return True

    def jpeg(self, camera_name: str) -> bytes | None:
        with self._lock:
            frame = self._frames.get(camera_name)
        if frame is None:
            return None
        # Preserve the whole image and aspect ratio; never crop task footage.
        height, width = frame.shape[:2]
        scale = min(1.0, 640 / width, 480 / height)
        if scale < 1:
            frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70]
        )
        return encoded.tobytes() if ok else None


recording_preview = RecordingPreview()


def observation_tap(robot, family):
    """Forward already-read observations to cameras and the usual joint socket."""
    from .teleoperate import get_can_joint_data, get_joint_positions_from_robot

    bimanual = hasattr(robot, "left_arm") and hasattr(robot, "right_arm")
    warned = False

    def publish(observation):
        nonlocal warned
        try:
            if not recording_preview.publish(observation):
                return
            notify = recording_preview.joint_notifier
            if notify is None:
                return
            if not family.uses_feetech_bus:
                data = get_can_joint_data(robot, family, bimanual, time.time(), observation=observation)
            else:
                data = {"type": "joint_update", "timestamp": time.time()}
                sides = (
                    (("left_", "", robot.left_arm), ("right_", "_right", robot.right_arm))
                    if bimanual
                    else (("", "", robot),)
                )
                for prefix, suffix, arm in sides:
                    data[f"joints{suffix}"] = get_joint_positions_from_robot(
                        robot, prefix, getattr(arm, "calibration", None), observation=observation
                    )
            notify(data)
        except Exception:
            # Display failures must never interrupt dataset writes or control.
            if not warned:
                logger.warning("Recording preview unavailable", exc_info=True)
                warned = True

    return publish
