"""Camera concurrency probe — run from YOUR terminal (camera-authorized), with
lelab and all preview pages stopped, all three cameras plugged in.

Answers ONE question with a controlled experiment: does the THIRD concurrently
open camera fail no matter which camera it is (systemic — USB2 isochronous
bandwidth reservations cap ~2 streams per bus), or does one specific unit fail
wherever it sits in the order (that unit / its cable)?

Run:  .venv/bin/python camera_probe.py
"""

import sys
import threading
import time

import cv2

# lelab.camera_enumeration was deleted in the camera revert (8d7634d); the
# equivalent enumerator lives in lelab.server. Shim keeps call sites intact.
from lelab import server as _lelab_server


class camera_enumeration:
    list_cameras = staticmethod(_lelab_server._avfoundation_cameras_in_cv2_order)

FIRST_FRAME_PATIENCE = 15.0  # generous: cold starts observed up to ~5s
MEASURE_S = 4.0


def open_cam(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None, "OPEN FAILED"
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    t0 = time.monotonic()
    while time.monotonic() - t0 < FIRST_FRAME_PATIENCE:
        ok, _ = cap.read()
        if ok:
            return cap, f"first frame after {time.monotonic() - t0:.1f}s"
    return cap, f"NO FRAME in {FIRST_FRAME_PATIENCE:.0f}s"


def measure(caps: dict):
    """Concurrent fps per open capture over MEASURE_S seconds."""
    counts = dict.fromkeys(caps, 0)
    stop = time.monotonic() + MEASURE_S

    def run(label, cap):
        while time.monotonic() < stop:
            ok, _ = cap.read()
            if ok:
                counts[label] += 1

    threads = [
        threading.Thread(target=run, args=(k, c), daemon=True) for k, c in caps.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(MEASURE_S + 5)
    return {k: v / MEASURE_S for k, v in counts.items()}


def main() -> None:
    cams = [
        c
        for c in camera_enumeration.list_cameras()
        if str(c.get("unique_id", "")).startswith("0x")
    ]
    print("USB cameras found:", [(c["index"], c["unique_id"]) for c in cams])
    if len(cams) < 3:
        print("Need all 3 cameras connected — only", len(cams), "enumerate. Fix that first.")
        sys.exit(1)
    ids = {c["unique_id"]: c["index"] for c in cams}
    labels = list(ids)

    print("\n== SOLO baselines (each camera alone) ==")
    for lab in labels:
        cap, note = open_cam(ids[lab])
        if cap is not None and "NO FRAME" not in note and "FAILED" not in note:
            fps = measure({lab: cap})[lab]
            print(f"  {lab}: {note}, {fps:.1f} fps solo")
        else:
            print(f"  {lab}: {note}")
        if cap is not None:
            cap.release()
        time.sleep(2)

    for order_name, order in [("forward", labels), ("REVERSED", list(reversed(labels)))]:
        print(f"\n== TRIPLE, open order {order_name}: {' -> '.join(order)} ==")
        open_caps: dict = {}
        for lab in order:
            cap, note = open_cam(ids[lab])
            print(f"  open {lab}: {note}")
            if cap is not None:
                open_caps[lab] = cap
        fps = measure(open_caps)
        for lab in order:
            print(f"  concurrent {lab}: {fps.get(lab, 0.0):.1f} fps")
        for cap in open_caps.values():
            cap.release()
        time.sleep(3)

    print(
        "\nHow to read it: all three healthy solo + the LAST-opened camera dead in"
        "\nBOTH orders (two different units!) = positional -> bus bandwidth"
        "\nreservation cap, hardware fine. One unique_id dead in every position"
        "\n(even opening first) = that unit/cable. Everything streaming in both"
        "\ntriples = the cap isn't binding at this negotiated format -> suspect"
        "\nthe app's standing-stream count or open-order interactions instead."
    )


if __name__ == "__main__":
    main()
