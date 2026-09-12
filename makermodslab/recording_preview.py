"""Read-only previews tapped from recording observations; never opens hardware.

Sample only cameras requested in the last two seconds. JPEG encoding runs in HTTP workers,
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
    # Requests renew demand; browser polling can pause briefly without churn.
    _VIEWER_TIMEOUT = 2.0
    _SAMPLE_INTERVAL = 0.1

    def __init__(self):
        self._lock = threading.Lock()
        self._active = False
        self._cameras = {}
        self._last_sample = float("-inf")
        self.joint_notifier: Callable[[dict], None] | None = None

    def start(self):
        with self._lock:
            self._cameras = {}
            self._last_sample = float("-inf")
            self._active = True

    def stop(self):
        with self._lock:
            self._active = False
            self._cameras = {}

    def _expire(self, now):
        for name in list(self._cameras):
            if now - self._cameras[name]["requested"] >= self._VIEWER_TIMEOUT:
                del self._cameras[name]

    def publish(self, observation: dict) -> bool:
        """Sample demanded cameras; return independent 10 Hz joint tick readiness."""
        with self._lock:
            now = time.monotonic()
            self._expire(now)
            if not self._active or now - self._last_sample < self._SAMPLE_INTERVAL:
                return False
            self._last_sample = now
            for name, camera in self._cameras.items():
                # Keep one snapshot until HTTP encoding consumes it. Replacing
                # pending frames would copy pixels no viewer ever sees.
                if camera["frame"] is not None:
                    continue
                frame = observation.get(name)
                camera["frame"] = (
                    frame.copy()
                    if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3
                    else None
                )
                camera["jpeg"] = None
            return True

    def jpeg(self, camera_name: str) -> bytes | None:
        with self._lock:
            if not self._active:
                return None
            now = time.monotonic()
            self._expire(now)
            camera = self._cameras.get(camera_name)
            if camera is None:
                camera = {"requested": now, "frame": None, "jpeg": None, "encoding": threading.Lock()}
                self._cameras[camera_name] = camera
            camera["requested"] = now
        # Serialize only this camera's HTTP consumers. The control loop never
        # waits for a JPEG encode, and all viewers reuse the same cached bytes.
        with camera["encoding"]:
            with self._lock:
                if self._cameras.get(camera_name) is not camera:
                    return None
                if camera["jpeg"] is not None:
                    return camera["jpeg"]
                frame = camera["frame"]
            if frame is None:
                return None
            try:
                encoded = self._encode(frame)
            except Exception:
                # Drop a failed snapshot so a later sample can recover. Keep
                # the error visible and let the per-camera lock release.
                with self._lock:
                    if camera["frame"] is frame:
                        camera["frame"] = None
                raise
            with self._lock:
                if self._cameras.get(camera_name) is not camera:
                    return None
                if camera["frame"] is frame:
                    camera["jpeg"] = encoded
                    camera["frame"] = None  # Also free the slot if encoding returned None.
            return encoded

    @staticmethod
    def _encode(frame):
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
