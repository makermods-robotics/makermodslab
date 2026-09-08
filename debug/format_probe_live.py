"""Open each camera via cv2 (same path lerobot uses), measure real delivered
fps, and run debug/format_probe MID-STREAM so the activeFormat reading and the
frame rate come from the same open.

Discriminates the 2fps hypotheses in one shot:
  - fps ~2 + ACTIVE `yuvs`  -> raw wire format landed (hypothesis 2)
  - fps ~2 + ACTIVE `420v`  -> wire fine; suspect auto-exposure (hypothesis 1,
    see debug/exposure_probe.py) or a software cap (hypothesis 3)
  - fps ~30 here but 2fps in the app -> the cap lives in the app path, not
    the camera/OS layer

Run from your camera-authorized terminal, with NO makerlab session or browser
preview holding the cameras:

    .venv/bin/python debug/format_probe_live.py                 # 640x480
    .venv/bin/python debug/format_probe_live.py --width 1280 --height 720

Compile the probe first if debug/format_probe doesn't exist:
    swiftc -O debug/format_probe.swift -o debug/format_probe
"""

import argparse
import pathlib
import subprocess
import time

import cv2

PROBE = pathlib.Path(__file__).with_name("format_probe")
MEASURE_SECONDS = 8.0
PROBE_AT_SECONDS = 3.0  # fire mid-stream, after format negotiation settles


def stream_and_probe(index: int, width: int, height: int) -> None:
    print(f"\n=== camera index {index} ===")
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("  open failed (index may not exist — done)")
        cap.release()
        raise SystemExit(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, _ = cap.read()
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"  first read ok={ok}, negotiated {actual[0]}x{actual[1]} "
          f"(requested {width}x{height})")
    if not ok:
        cap.release()
        return

    start = time.time()
    frames = 0
    probed = False
    intervals = []
    last = start
    while time.time() - start < MEASURE_SECONDS:
        if cap.read()[0]:
            now = time.time()
            frames += 1
            intervals.append(now - last)
            last = now
        if not probed and time.time() - start >= PROBE_AT_SECONDS:
            probed = True
            print("  --- activeFormat mid-stream ---")
            subprocess.run([str(PROBE)], check=False)
            print("  -------------------------------")
    elapsed = time.time() - start
    fps = frames / elapsed
    worst = max(intervals) * 1000 if intervals else float("nan")
    print(f"  delivered: {frames} frames / {elapsed:.1f}s = {fps:.1f} fps "
          f"(worst gap {worst:.0f} ms)")
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--indexes", type=int, nargs="*", default=[0, 1])
    args = parser.parse_args()
    if not PROBE.exists():
        raise SystemExit("debug/format_probe not built — run: "
                         "swiftc -O debug/format_probe.swift -o debug/format_probe")
    for index in args.indexes:
        stream_and_probe(index, args.width, args.height)


if __name__ == "__main__":
    main()
