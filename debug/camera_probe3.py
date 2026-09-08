"""Probe 3: lelab-lifecycle behaviors, tested directly against cv2/AVFoundation.

Probes 1-2 proved: 3 concurrent streams fine, enumeration subprocesses fine.
This one tests what's left of lelab's structure, isolated:

  A. opens from worker THREADS (lelab's readers open there; probes 1-2 opened
     on the main thread)
  B. rapid release->reopen CHURN of one device at varying gaps (tile retries /
     refcount cycling), counting opens that succeed but deliver no frame
  C. a SECOND concurrent capture of an already-streaming device in the same
     process (crossed paths / stacked opens), watching both
  D. paced reads at 15 fps (lelab reads at half the camera rate), plus the
     longest single read() stall seen anywhere

Run from your camera-authorized terminal, lelab stopped:

    .venv/bin/python camera_probe3.py
"""

import threading
import time

import cv2

# lelab.camera_enumeration was deleted in the camera revert (8d7634d); the
# equivalent enumerator lives in lelab.server. Shim keeps call sites intact.
from lelab import server as _lelab_server


class camera_enumeration:
    list_cameras = staticmethod(_lelab_server._avfoundation_cameras_in_cv2_order)

FIRST_FRAME_PATIENCE = 10.0


def open_and_first_frame(index: int):
    """Open + configure + wait for first frame. Returns (cap, seconds|None)."""
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None, None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    t0 = time.monotonic()
    while time.monotonic() - t0 < FIRST_FRAME_PATIENCE:
        ok, _ = cap.read()
        if ok:
            return cap, time.monotonic() - t0
    return cap, None


def main() -> None:
    cams = [
        c
        for c in camera_enumeration.list_cameras()
        if str(c.get("unique_id", "")).startswith("0x")
    ]
    if len(cams) < 3:
        print("need 3 cameras,", len(cams), "found")
        return
    idx = [c["index"] for c in cams]
    uid = {c["index"]: c["unique_id"] for c in cams}

    print("== A. worker-thread opens (lelab reader style) ==")
    results: dict[int, str] = {}

    def worker_open(i: int) -> None:
        cap, ff = open_and_first_frame(i)
        if cap is None:
            results[i] = "OPEN FAILED"
        elif ff is None:
            results[i] = "NO FRAME"
        else:
            n = 0
            t0 = time.monotonic()
            while time.monotonic() - t0 < 2.0:
                ok, _ = cap.read()
                n += ok
            results[i] = f"first frame {ff:.1f}s, {n / 2.0:.0f} fps"
        if cap is not None:
            cap.release()

    for i in idx:  # sequential like the funnel, but each open on a fresh thread
        t = threading.Thread(target=worker_open, args=(i,), daemon=True)
        t.start()
        t.join(20)
        print(f"  {uid[i]}: {results.get(i, 'thread hung')}")
        time.sleep(1)

    print("\n== B. release->reopen churn (same device, varying gap) ==")
    victim = idx[0]
    for gap in (0.0, 0.25, 1.0):
        failures = 0
        slowest = 0.0
        for _ in range(8):
            cap, ff = open_and_first_frame(victim)
            if cap is None or ff is None:
                failures += 1
            else:
                slowest = max(slowest, ff)
            if cap is not None:
                cap.release()
            time.sleep(gap)
        print(f"  gap {gap:.2f}s: {failures}/8 no-frame opens, slowest first frame {slowest:.1f}s")

    print("\n== C. second concurrent capture of a streaming device ==")
    cap1, ff1 = open_and_first_frame(idx[1])
    print(f"  primary open: {'ok' if ff1 is not None else 'NO FRAME'}")
    keep = {"n": 0, "run": True}

    def keep_reading() -> None:
        while keep["run"]:
            ok, _ = cap1.read()
            keep["n"] += ok

    t = threading.Thread(target=keep_reading, daemon=True)
    t.start()
    time.sleep(1)
    n_before = keep["n"]
    cap2, ff2 = open_and_first_frame(idx[1])  # second capture, same device
    time.sleep(2)
    n_after = keep["n"]
    primary_fps = (n_after - n_before) / 3.0
    print(
        f"  second open: {'ok, first frame %.1fs' % ff2 if ff2 is not None else 'NO FRAME'};"
        f" primary during overlap: {primary_fps:.0f} fps"
    )
    if cap2 is not None:
        cap2.release()
    time.sleep(2)
    n_final = keep["n"]
    print(f"  primary after second released: {(n_final - n_after) / 2.0:.0f} fps")
    keep["run"] = False
    t.join(3)
    if cap1 is not None:
        cap1.release()

    print("\n== D. paced reads at 15 fps (lelab's TARGET_FPS) + worst read stall ==")
    cap, ff = open_and_first_frame(idx[2])
    if cap is None or ff is None:
        print("  open failed")
    else:
        got = 0
        worst = 0.0
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10.0:
            r0 = time.monotonic()
            ok, _ = cap.read()
            dt = time.monotonic() - r0
            worst = max(worst, dt)
            got += ok
            time.sleep(max(0.0, (1 / 15.0) - dt))
        print(f"  paced: {got / 10.0:.1f} reads/s delivered, worst single read() {worst * 1000:.0f} ms")
        cap.release()

    print(
        "\nRead: any NO FRAME in A/B/C/D reproduces lelab's black tile outside"
        "\nlelab -> that lifecycle behavior is the trigger. All clean -> the bug"
        "\nis in the manager's own state machine and I audit/instrument that."
    )


if __name__ == "__main__":
    main()
