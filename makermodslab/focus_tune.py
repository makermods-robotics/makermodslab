"""Auto-tune and lock UVC focus for the arm cameras (macOS only).

The SO-101 rig's USB cameras ship with UVC autofocus enabled, and over a
low-texture workspace the AF hunts — whole recorded episodes come out
defocused (see the 2026-07-30 eraser_place analysis: 41/50 episodes blurry).
AVFoundation exposes no focus controls, so this module shells out to the
`uvc-util` binary (https://github.com/jtfrey/uvc-util) for the USB control
writes, while measuring sharpness through the same cv2 capture path the
recorder uses.

For each requested camera it disables autofocus, sweeps `focus-abs`
coarse-then-fine while scoring frames by variance of Laplacian, and locks
the sharpest value. UVC settings reset when a camera is replugged, so this
is meant to be re-run from the UI whenever focus looks soft.

Single global session, same reciprocal-exclusion pattern as the other
robot-driving modules: refuses to start while recording / teleoperation /
inference is active (those flows own the cameras).
"""

import logging
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .camera_preview import camera_preview_manager

logger = logging.getLogger(__name__)

UVC_UTIL_ENV = "MAKERMODSLAB_UVC_UTIL"
_UVC_UTIL_BUILD_HINT = (
    "uvc-util binary not found. Clone https://github.com/jtfrey/uvc-util, build it with "
    "`cc -o uvc-util -framework IOKit -framework Foundation uvc-util.m UVCController.m "
    "UVCType.m UVCValue.m` from its src/ directory, and either put it on PATH or point "
    f"the {UVC_UTIL_ENV} environment variable at it."
)
_UVC_UTIL_CANDIDATES = (
    Path.home() / "Documents/MakerMods/uvc-util/src/uvc-util",
    Path.home() / "uvc-util/src/uvc-util",
)

FOCUS_MIN = 0
FOCUS_MAX = 1023
COARSE_STEP = 64
FINE_STEP = 8
SETTLE_S = 0.35  # minimum settle: exposure/gain adaptation after any change
# Extra settle proportional to lens travel distance. Without this, the first
# sweep sample scores frames taken through the lens's PREVIOUS position (it
# hasn't finished traveling), which credited a stale-sharp frame to
# focus-abs=0 on the rig's wrist camera and locked it fully defocused.
LENS_TRAVEL_FULL_S = 1.2
FRAMES_PER_VALUE = 3
VERIFY_CANDIDATES = 3
VERIFY_FRAMES = 5
# Scoring region per camera, as (x0, y0, x1, y1) frame fractions. A full-frame
# score can prefer focusing on textured background far behind the workspace
# when the table itself is featureless, so each camera scores only where its
# subject actually lives:
#  - default: central crop (the workspace, for third-person views);
#  - any camera whose name contains "wrist": the bottom-center window where
#    the gripper sits. Rig-specific hardcode (this rig's wrist mount puts the
#    jaws bottom-center of frame) — adjust the fractions if the mount changes.
DEFAULT_ROI = (0.2, 0.2, 0.8, 0.8)
WRIST_ROI = (0.30, 0.55, 0.70, 0.95)


def _roi_for(camera_name: str) -> tuple[float, float, float, float]:
    return WRIST_ROI if "wrist" in camera_name.lower() else DEFAULT_ROI


_UVC_TIMEOUT_S = 10

# Rows of `uvc-util -d`, e.g.:
# 0            0x1e45:0x0209  0x01113000   1.00         USB Camera
_DEVICE_ROW = re.compile(r"^\s*(\d+)\s+0x([0-9a-fA-F]+):0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)\s")

_state_lock = threading.Lock()
focus_tune_active: bool = False
_tune_thread: threading.Thread | None = None
_status: dict[str, Any] = {"active": False, "cameras": [], "error": None}


def find_uvc_util() -> str | None:
    """Locate the uvc-util binary: env override, PATH, then known build spots."""
    env = os.environ.get(UVC_UTIL_ENV)
    if env and Path(env).is_file():
        return env
    on_path = shutil.which("uvc-util")
    if on_path:
        return on_path
    for candidate in _UVC_UTIL_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return None


def parse_uvc_devices(text: str) -> list[dict[str, int]]:
    """Parse `uvc-util -d` output into {uvc_index, vid, pid, location_id} rows."""
    devices = []
    for line in text.splitlines():
        m = _DEVICE_ROW.match(line)
        if m:
            devices.append(
                {
                    "uvc_index": int(m.group(1)),
                    "vid": int(m.group(2), 16),
                    "pid": int(m.group(3), 16),
                    "location_id": int(m.group(4), 16),
                }
            )
    return devices


def match_uvc_index(unique_id: str, devices: list[dict[str, int]]) -> int | None:
    """Map an AVFoundation uniqueID to a uvc-util device index.

    For USB cameras Apple composes uniqueID as locationID + VendorID +
    ProductID (hex concatenation), which is exactly the identity uvc-util's
    device table exposes. Built-in cameras use UUID-style ids and match
    nothing, which is correct — they have no UVC controls.
    """
    uid = unique_id.lower().removeprefix("0x").lstrip("0")
    for dev in devices:
        composed = f"{dev['location_id']:08x}{dev['vid']:04x}{dev['pid']:04x}".lstrip("0")
        if composed == uid:
            return dev["uvc_index"]
    return None


def _uvc_run(uvc_util: str, args: list[str]) -> str:
    result = subprocess.run(
        [uvc_util, *args],
        capture_output=True,
        text=True,
        timeout=_UVC_TIMEOUT_S,
        check=True,
    )
    return result.stdout


def _uvc_set(uvc_util: str, uvc_index: int, control: str, value: int) -> None:
    _uvc_run(uvc_util, ["-I", str(uvc_index), "-s", f"{control}={value}"])


def _sharpness(
    cap: cv2.VideoCapture,
    roi: tuple[float, float, float, float],
    frames: int = FRAMES_PER_VALUE,
) -> float:
    """Median variance-of-Laplacian over the camera's scoring region.

    Scoring a region instead of the full frame is what makes the sweep focus
    at the SUBJECT's depth: a full-frame score let the wrist camera lock
    focus-abs=0 because the far background (top of frame) scored sharp while
    the gripper zone went blurry. The 3x3 Gaussian suppresses sensor noise,
    which otherwise adds focus-independent variance over the featureless
    white table.
    """
    x0, y0, x1, y1 = roi
    vals = []
    for _ in range(frames):
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        crop = gray[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)]
        crop = cv2.GaussianBlur(crop, (3, 3), 0)
        vals.append(cv2.Laplacian(crop, cv2.CV_64F).var())
    return float(np.median(vals)) if vals else 0.0


def _set_and_settle(
    cap: cv2.VideoCapture, uvc_util: str, uvc_index: int, value: int, prev_value: int | None
) -> None:
    """Move the lens and wait long enough that captured frames reflect it."""
    _uvc_set(uvc_util, uvc_index, "focus-abs", value)
    travel = abs(value - prev_value) if prev_value is not None else FOCUS_MAX - FOCUS_MIN
    time.sleep(SETTLE_S + LENS_TRAVEL_FULL_S * travel / (FOCUS_MAX - FOCUS_MIN))
    for _ in range(2):  # flush frames buffered before the lens moved
        cap.read()


def _sweep(
    cap: cv2.VideoCapture,
    uvc_util: str,
    uvc_index: int,
    values: list[int],
    cam_status: dict[str, Any],
    roi: tuple[float, float, float, float],
) -> tuple[int, float]:
    best_v, best_s = values[0], -1.0
    prev: int | None = None
    for v in values:
        _set_and_settle(cap, uvc_util, uvc_index, v, prev)
        prev = v
        s = _sharpness(cap, roi)
        cam_status["points"].append([v, round(s, 1)])
        cam_status["current_value"] = v
        if s > best_s:
            best_v, best_s = v, s
            cam_status["best_value"] = v
            cam_status["best_sharpness"] = round(s, 1)
    return best_v, best_s


def _tune_one_camera(uvc_util: str, cam_status: dict[str, Any]) -> None:
    cv2_index = cam_status["camera_index"]
    uvc_index = cam_status["uvc_index"]
    cap = cv2.VideoCapture(cv2_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cam_status["phase"] = "error"
        cam_status["error"] = (
            "Could not open the camera. Another app may be holding it (a preview "
            "stream that didn't release, or an active robot session)."
        )
        return
    try:
        roi = _roi_for(cam_status["name"])
        _uvc_set(uvc_util, uvc_index, "auto-focus", 0)
        cam_status["phase"] = "coarse"
        coarse = list(range(FOCUS_MIN, FOCUS_MAX + 1, COARSE_STEP))
        v1, _ = _sweep(cap, uvc_util, uvc_index, coarse, cam_status, roi)
        cam_status["phase"] = "fine"
        fine = list(range(max(FOCUS_MIN, v1 - COARSE_STEP), min(FOCUS_MAX, v1 + COARSE_STEP) + 1, FINE_STEP))
        v2, s2 = _sweep(cap, uvc_util, uvc_index, fine, cam_status, roi)

        # Verify pass: re-measure the top sweep candidates with generous
        # settle and more frames. Guards against a single spurious sweep
        # sample (scene motion, exposure transient) deciding the lock.
        cam_status["phase"] = "verify"
        by_sharpness = sorted(cam_status["points"], key=lambda p: -p[1])
        candidates: list[int] = []
        for v, _s in by_sharpness:
            if all(abs(v - c) > FINE_STEP for c in candidates):
                candidates.append(v)
            if len(candidates) == VERIFY_CANDIDATES:
                break
        best_v, best_s = v2, -1.0
        prev = fine[-1]
        for v in candidates:
            _set_and_settle(cap, uvc_util, uvc_index, v, prev)
            prev = v
            cam_status["current_value"] = v
            s = _sharpness(cap, roi, frames=VERIFY_FRAMES)
            if s > best_s:
                best_v, best_s = v, s

        _set_and_settle(cap, uvc_util, uvc_index, best_v, prev)
        cam_status["phase"] = "done"
        cam_status["locked_value"] = best_v
        cam_status["best_value"] = best_v
        cam_status["best_sharpness"] = round(best_s, 1)
        logger.info(
            "Focus locked: camera %s (cv2 %d) at focus-abs=%d (sharpness %.1f)",
            cam_status["name"],
            cv2_index,
            best_v,
            best_s,
        )
    except (subprocess.SubprocessError, OSError) as e:
        cam_status["phase"] = "error"
        cam_status["error"] = f"uvc-util control write failed: {e}"
    finally:
        cap.release()


def _tune_worker(uvc_util: str) -> None:
    global focus_tune_active
    try:
        for cam_status in _status["cameras"]:
            if cam_status.get("error"):
                continue  # skipped at mapping time (no UVC identity / no focus control)
            _tune_one_camera(uvc_util, cam_status)
    except Exception as e:  # defensive: a worker death must not wedge the active flag
        logger.exception("Focus tune worker crashed")
        _status["error"] = str(e)
    finally:
        with _state_lock:
            focus_tune_active = False
            _status["active"] = False


def _busy_with() -> str | None:
    """Name the robot flow currently holding the cameras, if any."""
    from . import record, rollout, teleoperate

    if record.recording_active:
        return "recording"
    if teleoperate.teleoperation_active:
        return "teleoperation"
    if rollout.inference_active:
        return "inference"
    return None


def handle_start_focus_tune(
    cameras: list[dict[str, Any]], avf_cameras: list[dict[str, Any]]
) -> dict[str, Any]:
    """Start a background focus-tune session over the given cv2 camera indices.

    ``avf_cameras`` is the AVFoundation enumeration (cv2-ordered, with
    ``unique_id``) from the caller — the mapping to uvc-util device indices
    happens here so a hotplug between enumeration and click is caught.
    """
    global focus_tune_active, _tune_thread

    if platform.system() != "Darwin":
        return {"success": False, "message": "Focus tuning is only supported on macOS (uvc-util)."}
    uvc_util = find_uvc_util()
    if uvc_util is None:
        return {"success": False, "message": _UVC_UTIL_BUILD_HINT}
    if not cameras:
        return {"success": False, "message": "No cameras to tune — add a camera first."}

    busy = _busy_with()
    if busy is not None:
        return {"success": False, "message": f"Cannot tune focus while {busy} is active."}

    try:
        devices = parse_uvc_devices(_uvc_run(uvc_util, ["-d"]))
    except (subprocess.SubprocessError, OSError) as e:
        return {"success": False, "message": f"uvc-util could not list devices: {e}"}
    unique_by_index = {c["index"]: c.get("unique_id", "") for c in avf_cameras}

    cam_statuses = []
    for cam in cameras:
        cv2_index = cam["camera_index"]
        status: dict[str, Any] = {
            "camera_index": cv2_index,
            "name": cam.get("name") or f"Camera {cv2_index}",
            "uvc_index": None,
            "phase": "pending",
            "points": [],
            "current_value": None,
            "best_value": None,
            "best_sharpness": None,
            "locked_value": None,
            "error": None,
        }
        uvc_index = match_uvc_index(unique_by_index.get(cv2_index, ""), devices)
        if uvc_index is None:
            status["phase"] = "error"
            status["error"] = (
                "No UVC identity for this camera (built-in cameras and virtual "
                "cameras have no focus control)."
            )
        else:
            try:
                controls = _uvc_run(uvc_util, ["-I", str(uvc_index), "-c"])
            except (subprocess.SubprocessError, OSError) as e:
                controls = ""
                status["phase"] = "error"
                status["error"] = f"Could not query camera controls: {e}"
            if status["error"] is None and "focus-abs" not in controls:
                status["phase"] = "error"
                status["error"] = "This camera does not expose a focus control."
            status["uvc_index"] = uvc_index
        cam_statuses.append(status)

    if all(s["error"] is not None for s in cam_statuses):
        return {
            "success": False,
            "message": "None of the selected cameras are focus-tunable: "
            + "; ".join(f"{s['name']}: {s['error']}" for s in cam_statuses),
        }

    with _state_lock:
        if focus_tune_active:
            return {"success": False, "message": "A focus tune is already running."}
        focus_tune_active = True
        _status["active"] = True
        _status["error"] = None
        _status["cameras"] = cam_statuses

    # Backend camera previews hold the cv2 devices this sweep is about to open,
    # and a preview still holding an index makes _tune_one_camera's
    # cv2.VideoCapture fail outright. Same handover as recording and inference
    # (record.handle_start_recording): released only after `focus_tune_active`
    # is True, so /camera-preview 409s and no client can re-acquire a device in
    # the gap between this release and the sweep's first open.
    camera_preview_manager.stop_all()

    _tune_thread = threading.Thread(target=_tune_worker, args=(uvc_util,), daemon=True)
    _tune_thread.start()

    return {"success": True, "message": f"Focus tune started for {len(cam_statuses)} camera(s)."}


def get_focus_tune_status() -> dict[str, Any]:
    with _state_lock:
        return {
            "active": _status["active"],
            "error": _status["error"],
            "cameras": [dict(c) for c in _status["cameras"]],
        }
