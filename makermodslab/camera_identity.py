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
"""Camera identity → cv2 index translation (macOS).

``cv2.VideoCapture(index)`` resolves the index against AVFoundation's
*in-process* device list, which is snapshotted the first time this process
touches AVFoundation and never refreshes — device-connection notifications
are delivered on a thread that needs an active NSRunLoop, which uvicorn
doesn't run. ``/available-cameras`` therefore enumerates in a *fresh
subprocess* (:func:`list_cameras_fresh`), but that yields indices in the
fresh device order, which diverges from this process's order whenever
cameras were plugged/unplugged after startup. Opening by such an index then
silently hits the wrong physical device — e.g. the built-in webcam instead
of a robot camera, or two identical robot cameras swapped — poisoning
recordings in a way no preview reveals.

The stable link between the index spaces is AVFoundation's ``uniqueID``
(for USB cameras Apple composes it from locationID+VID+PID, so it names the
physical port). Robot records store it per camera; before any index is
trusted it is re-anchored to the uniqueID:

- :func:`resolve_cv2_index` — for code opening cameras IN THIS PROCESS
  (recording): walks the same in-process device list cv2 walks (video +
  muxed devices, uniqueID-sorted — mirrors OpenCV's cap_avfoundation_mac.mm).
- :func:`resolve_fresh_index` — for code handing indices to a FRESH
  SUBPROCESS (the lerobot-rollout inference runner): resolves against a
  fresh enumeration, which is what the subprocess's own cv2 will see.

A device attached after this process started is invisible to in-process
AVFoundation entirely; resolution returns None for that case and callers
must fail loudly (telling the user to restart MakerMods Lab) rather than open
whatever now sits at the stale index.
"""

import json
import logging
import platform
import subprocess
import sys

logger = logging.getLogger(__name__)

_AVF_DEVICE_TYPE_NAMES = (
    "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    "AVCaptureDeviceTypeExternalUnknown",  # macOS < 14
    "AVCaptureDeviceTypeExternal",  # macOS >= 14
    "AVCaptureDeviceTypeContinuityCamera",  # macOS >= 14
    "AVCaptureDeviceTypeDeskViewCamera",  # macOS >= 13
)

# Runs in a fresh Python — see list_cameras_fresh for why. Mirrors OpenCV's
# macOS enumeration: video + muxed devices sorted by uniqueID
# (cap_avfoundation_mac.mm), so the returned index matches what
# cv2.VideoCapture will open in a process started now.
_AVF_ENUM_SCRIPT = """
import json, objc
from Foundation import NSBundle
bundle = NSBundle.bundleWithPath_("/System/Library/Frameworks/AVFoundation.framework")
bundle.load()
types = []
for name in (
    "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    "AVCaptureDeviceTypeExternalUnknown",   # macOS < 14
    "AVCaptureDeviceTypeExternal",          # macOS >= 14
    "AVCaptureDeviceTypeContinuityCamera",  # macOS >= 14
    "AVCaptureDeviceTypeDeskViewCamera",    # macOS >= 13
):
    loaded = {}
    try:
        objc.loadBundleVariables(bundle, loaded, [(name, b"@")])
    except objc.error:
        continue
    if loaded.get(name) is not None:
        types.append(loaded[name])
cls = objc.lookUpClass("AVCaptureDeviceDiscoverySession")
devs = []
for mt in ("vide", "muxx"):
    devs.extend(cls.discoverySessionWithDeviceTypes_mediaType_position_(types, mt, 0).devices() or [])
devs.sort(key=lambda d: d.uniqueID())
print(json.dumps([
    {"index": i, "name": str(d.localizedName()), "unique_id": str(d.uniqueID())}
    for i, d in enumerate(devs)
]))
"""


def list_cameras_fresh() -> list[dict]:
    """Enumerate macOS cameras in a fresh Python subprocess.

    AVFoundation's in-process device cache doesn't refresh on USB
    hotplug. Both the deprecated ``+devicesWithMediaType:`` and a
    long-lived ``AVCaptureDeviceDiscoverySession`` go stale, because
    device-connection notifications are delivered via
    ``NSNotificationCenter`` on a thread that needs an active
    ``NSRunLoop`` — uvicorn workers don't run one. A fresh subprocess
    re-initializes AVFoundation, which reads IOKit's live device state
    at startup.

    Returns ``[{"index", "name", "unique_id"}, ...]``; empty on failure.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", _AVF_ENUM_SCRIPT],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("AVFoundation enumeration subprocess failed: %s", e)
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        logger.warning("AVFoundation enumeration returned invalid JSON: %s", e)
        return []


def list_cameras_in_process() -> list[dict] | None:
    """This process's camera list, in cv2 open order.

    Returns ``[{"index", "name", "unique_id"}, ...]`` reflecting the same
    (possibly stale) AVFoundation state cv2.VideoCapture resolves indices
    against, or None when identity can't be established (non-macOS, PyObjC
    unavailable, AVFoundation query failure).
    """
    if platform.system() != "Darwin":
        return None
    try:
        import objc
        from Foundation import NSBundle

        bundle = NSBundle.bundleWithPath_("/System/Library/Frameworks/AVFoundation.framework")
        bundle.load()
        types = []
        for name in _AVF_DEVICE_TYPE_NAMES:
            loaded = {}
            try:
                objc.loadBundleVariables(bundle, loaded, [(name, b"@")])
            except objc.error:
                continue
            if loaded.get(name) is not None:
                types.append(loaded[name])
        cls = objc.lookUpClass("AVCaptureDeviceDiscoverySession")
        devs = []
        for media_type in ("vide", "muxx"):
            session = cls.discoverySessionWithDeviceTypes_mediaType_position_(types, media_type, 0)
            devs.extend(session.devices() or [])
        devs.sort(key=lambda d: d.uniqueID())
        return [
            {"index": i, "name": str(d.localizedName()), "unique_id": str(d.uniqueID())}
            for i, d in enumerate(devs)
        ]
    except Exception as e:
        logger.warning("In-process AVFoundation enumeration failed: %s", e)
        return None


def resolve_in_enumeration(
    cameras: list[dict] | None, unique_id: str | None, fallback_index: int
) -> int | None:
    """Resolve ``unique_id`` against an enumeration ``cameras`` list.

    - No unique_id, or identity unavailable (``cameras`` is None/empty):
      ``fallback_index`` — legacy trust-the-index behavior.
    - unique_id found: its enumerated index.
    - unique_id verifiably absent: None — callers must error out instead of
      opening a different physical camera.
    """
    if not unique_id or not cameras:
        return fallback_index
    for cam in cameras:
        if cam.get("unique_id") == unique_id:
            if cam["index"] != fallback_index:
                logger.info(
                    "Camera %s: resolved cv2 index %d differs from stored index %d "
                    "(device set changed since the index was stored)",
                    unique_id,
                    cam["index"],
                    fallback_index,
                )
            return cam["index"]
    logger.warning("Camera %s not present in enumeration (unplugged or moved?)", unique_id)
    return None


def resolve_cv2_index(unique_id: str | None, fallback_index: int) -> int | None:
    """The index THIS process's cv2 opens for ``unique_id`` (recording path).

    Falls back to ``fallback_index`` when identity is unavailable; returns
    None when the camera is verifiably absent from the in-process device
    list (attached after this process started — only a restart makes it
    reachable, so callers must fail loudly).
    """
    if not unique_id:
        return fallback_index
    cameras = list_cameras_in_process()
    if cameras is None:
        return fallback_index
    return resolve_in_enumeration(cameras, unique_id, fallback_index)


def resolve_fresh_index(unique_id: str | None, fallback_index: int) -> int | None:
    """The index a FRESHLY STARTED process's cv2 opens for ``unique_id``
    (inference-subprocess path). Same contract as :func:`resolve_cv2_index`.
    """
    if not unique_id:
        return fallback_index
    if platform.system() != "Darwin":
        return fallback_index
    cameras = list_cameras_fresh()
    if not cameras:
        return fallback_index
    return resolve_in_enumeration(cameras, unique_id, fallback_index)
