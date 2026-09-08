"""Diagnose and optionally fix wrist-camera frame stalls: auto-exposure fps collapse.

The symptom: `TimeoutError: OpenCVCamera(N) latest frame is too old: ~50Xms
(max allowed: 500 ms)` on ONE camera, intermittently — a UVC camera whose
auto-exposure extends the exposure time beyond the frame period in dim light
can only deliver a frame per exposure, so 30fps silently collapses toward
~2fps (≈500ms frame age). Pose-dependent shadowing (the arm over the wrist
camera's view) makes it intermittent.

Run from your own terminal (needs macOS camera permission), with NO makerlab
recording/inference session holding the cameras:

    .venv/bin/python debug/exposure_probe.py           # measure only
    .venv/bin/python debug/exposure_probe.py --lock    # also lock exposure

Measures each USB camera's real frame-delivery intervals and brightness, reads
its current exposure state via uvc-util, and prints a verdict. With --lock it
disables auto-exposure and pins exposure-time within the 30fps frame budget
(exposure-time-abs is in 100µs units; 30fps needs <= 333).

Like the focus lock: UVC settings die on replug/reboot — re-run per session.
Locked exposure changes image brightness; add bench light rather than raising
exposure past the frame budget, and spot-check the next dataset/eval frames.

cv2 index <-> uvc-util index mapping is by composed uniqueID, never by index
order (see debug/focus_tune.py for the full story).
"""

import argparse
import re
import subprocess
import time

import cv2
import numpy as np

UVC_UTIL = "/Users/mokuroh54/Documents/MakerMods/uvc-util/src/uvc-util"
_DEVICE_ROW = re.compile(
    r"^\s*(\d+)\s+0x([0-9a-fA-F]+):0x([0-9a-fA-F]+)\s+0x([0-9a-fA-F]+)\s"
)

PROBE_SECONDS = 8.0
# 30fps budget in exposure-time-abs units (100µs): 333 ≈ 33.3ms. Leave headroom.
LOCK_EXPOSURE_ABS = 300
STALL_MS = 100.0  # any inter-frame gap beyond this is already a 3x miss at 30fps


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


def uvc_get(uvc_index: int, control: str) -> str:
    r = subprocess.run(
        [UVC_UTIL, "-I", str(uvc_index), "-g", control], capture_output=True, text=True
    )
    return r.stdout.strip() or r.stderr.strip()


def uvc_set(uvc_index: int, setting: str) -> bool:
    r = subprocess.run(
        [UVC_UTIL, "-I", str(uvc_index), "-s", setting], capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"    (uvc-util -s {setting} failed: {r.stderr.strip() or r.stdout.strip()})")
    return r.returncode == 0


def probe(cv2_idx: int) -> tuple[list[float], float]:
    """Read frames for PROBE_SECONDS; return (inter-frame gaps ms, mean brightness)."""
    cap = cv2.VideoCapture(cv2_idx, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    gaps, brightness = [], []
    try:
        for _ in range(5):  # warmup / let exposure settle
            cap.read()
        last = time.perf_counter()
        deadline = last + PROBE_SECONDS
        while time.perf_counter() < deadline:
            ok, frame = cap.read()
            now = time.perf_counter()
            if not ok:
                continue
            gaps.append((now - last) * 1e3)
            last = now
            brightness.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
    finally:
        cap.release()
    return gaps, float(np.mean(brightness)) if brightness else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", action="store_true", help="disable auto-exposure and pin exposure-time")
    args = ap.parse_args()

    mapping = cv2_to_uvc_index_map()
    if not mapping:
        raise SystemExit("no USB cameras found via uvc-util")

    for cv2_idx, uvc_idx in sorted(mapping.items()):
        print(f"\n== cv2 index {cv2_idx}  (uvc-util -I {uvc_idx})")
        print(f"  auto-exposure-mode: {uvc_get(uvc_idx, 'auto-exposure-mode')}")
        print(f"  exposure-time-abs:  {uvc_get(uvc_idx, 'exposure-time-abs')}")

        gaps, bright = probe(cv2_idx)
        if not gaps:
            print("  NO FRAMES — camera held by another process, or dead")
            continue
        g = np.array(gaps)
        stalls = int((g > STALL_MS).sum())
        print(
            f"  frames={len(g)}  effective_fps={1e3 / g.mean():5.1f}  "
            f"gap p50={np.percentile(g, 50):6.1f}ms  p95={np.percentile(g, 95):6.1f}ms  "
            f"max={g.max():6.1f}ms  gaps>{STALL_MS:.0f}ms={stalls}  brightness={bright:5.1f}/255"
        )
        if g.max() > 400:
            print(
                "  VERDICT: delivery stalls at the read_latest death zone — with low "
                "brightness this is auto-exposure fps collapse; add light and/or --lock."
            )
        elif g.mean() > 40:
            print("  VERDICT: running below 30fps — watch it, likely exposure creeping up.")
        else:
            print("  VERDICT: healthy 30fps delivery in the current light.")

        if args.lock:
            print(f"  locking: auto-exposure-mode=1, exposure-time-abs={LOCK_EXPOSURE_ABS}")
            uvc_set(uvc_idx, "auto-exposure-mode=1")  # 1 = manual per UVC spec
            uvc_set(uvc_idx, f"exposure-time-abs={LOCK_EXPOSURE_ABS}")
            time.sleep(0.3)
            gaps2, bright2 = probe(cv2_idx)
            if gaps2:
                g2 = np.array(gaps2)
                print(
                    f"  after lock: effective_fps={1e3 / g2.mean():5.1f}  "
                    f"max_gap={g2.max():6.1f}ms  brightness={bright2:5.1f}/255"
                    + ("  (image darkened — add bench light)" if bright2 < bright * 0.7 else "")
                )


if __name__ == "__main__":
    main()
