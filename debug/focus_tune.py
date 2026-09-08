"""Lock UVC focus at the sharpest value for each attached arm camera.

Sweeps focus-abs (coarse then fine) while measuring frame sharpness
(variance of Laplacian), then sets each camera to its best value.
Assumes autofocus is already disabled (uvc-util -I <n> -s auto-focus=0).

Run from the user's own terminal (needs macOS camera permission):

    .venv/bin/python debug/focus_tune.py

NOTE: these cameras forget UVC settings on replug/reboot — re-run this
(or at least the auto-focus=0 + focus-abs=<best> sets) at session start.

cv2 index <-> uvc-util index: NOT the same order. uvc-util lists devices
in IOKit discovery order, which varies with plug history (verified live:
cv2 0 mapped to uvc-util 1 on 2026-07-30). cv2's AVFoundation order sorts
by uniqueID string, and for USB cameras Apple composes uniqueID as
locationID+VID+PID — so the correct mapping is derived below by composing
each uvc-util device's uniqueID and sorting, never by assuming index order.
Assumes the USB cameras occupy the first cv2 indices (they do here: their
"0x..." uniqueIDs sort before the built-in camera's UUID-style id).
"""

import re
import subprocess
import sys
import time

import cv2
import numpy as np

UVC_UTIL = "/Users/mokuroh54/Documents/MakerMods/uvc-util/src/uvc-util"
_DEVICE_ROW = re.compile(
    r"^\s*(\d+)\s+0x([0-9a-fA-F]+):0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)\s"
)


def cv2_to_uvc_index_map() -> dict[int, int]:
    """Map cv2 camera indices to uvc-util device indices by USB identity."""
    out = subprocess.run([UVC_UTIL, "-d"], check=True, capture_output=True, text=True).stdout
    devices = []
    for line in out.splitlines():
        m = _DEVICE_ROW.match(line)
        if m:
            uvc_index, vid, pid, loc = (
                int(g, b) for g, b in zip(m.groups(), (10, 16, 16, 16), strict=True)
            )
            devices.append((f"0x{(loc << 32) | (vid << 16) | pid:x}", uvc_index))
    devices.sort()  # cv2's AVFoundation order = uniqueID string sort
    return {cv2_idx: uvc_idx for cv2_idx, (_, uvc_idx) in enumerate(devices)}
FOCUS_MIN, FOCUS_MAX = 0, 1023
COARSE_STEP = 64
FINE_STEP = 8
SETTLE_S = 0.35  # let the lens physically move + exposure settle
FRAMES_PER_VALUE = 3


def set_focus(uvc_index: int, value: int) -> None:
    subprocess.run(
        [UVC_UTIL, "-I", str(uvc_index), "-s", f"focus-abs={value}"],
        check=True, capture_output=True,
    )


def sharpness(cap: cv2.VideoCapture) -> float:
    vals = []
    for _ in range(FRAMES_PER_VALUE):
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vals.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.median(vals)) if vals else 0.0


def sweep(cap: cv2.VideoCapture, uvc_index: int, values: list[int]) -> tuple[int, float]:
    best_v, best_s = values[0], -1.0
    for v in values:
        set_focus(uvc_index, v)
        time.sleep(SETTLE_S)
        for _ in range(2):  # flush stale buffered frames
            cap.read()
        s = sharpness(cap)
        print(f"  focus-abs={v:4d}  sharpness={s:8.1f}")
        if s > best_s:
            best_v, best_s = v, s
    return best_v, best_s


def main() -> None:
    index_map = cv2_to_uvc_index_map()
    print(f"cv2 -> uvc-util index map: {index_map}")
    for cv2_idx, uvc_idx in index_map.items():
        print(f"\n== cv2 camera {cv2_idx} (uvc-util -I {uvc_idx})")
        cap = cv2.VideoCapture(cv2_idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print(f"  ERROR: could not open cv2 camera {cv2_idx} "
                  "(camera permission? another process holding it?)")
            sys.exit(1)
        subprocess.run(
            [UVC_UTIL, "-I", str(uvc_idx), "-s", "auto-focus=0"],
            check=True, capture_output=True,
        )
        coarse = list(range(FOCUS_MIN, FOCUS_MAX + 1, COARSE_STEP))
        v1, _ = sweep(cap, uvc_idx, coarse)
        lo = max(FOCUS_MIN, v1 - COARSE_STEP)
        hi = min(FOCUS_MAX, v1 + COARSE_STEP)
        fine = list(range(lo, hi + 1, FINE_STEP))
        v2, s2 = sweep(cap, uvc_idx, fine)
        set_focus(uvc_idx, v2)
        cap.release()
        print(f"  -> LOCKED cv2 camera {cv2_idx} at focus-abs={v2} (sharpness {s2:.1f})")
    print("\nDone. Settings reset on replug/reboot — re-run at session start.")


if __name__ == "__main__":
    main()
