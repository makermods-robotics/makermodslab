"""Probe 2: does the AVFoundation enumeration subprocess disturb live captures?

camera_probe.py proved 3 concurrent direct captures run at ~30 fps each. lelab
differs structurally in ONE big way: it spawns a fresh AVFoundation enumeration
subprocess on every /available-cameras call and (cache permitting) every
capture open. This probe holds 3 live captures and measures per-SECOND fps
while firing those same enumeration subprocesses, then races an open against
one. Run from your camera-authorized terminal, lelab stopped:

    .venv/bin/python camera_probe2.py
"""

import threading
import time

import cv2

# lelab.camera_enumeration was deleted in the camera revert (8d7634d); the
# equivalent enumerator lives in lelab.server. Shim keeps call sites intact.
from lelab import server as _lelab_server


class camera_enumeration:
    list_cameras = staticmethod(_lelab_server._avfoundation_cameras_in_cv2_order)

MEASURE_QUIET_S = 4
STORM_ENUMS = 6


class Counter:
    """Per-second frame counter for one capture, read on a dedicated thread."""

    def __init__(self, label: str, cap: cv2.VideoCapture) -> None:
        self.label = label
        self.cap = cap
        self.buckets: dict[int, int] = {}
        self.t0 = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.t0 = time.monotonic()
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, _ = self.cap.read()
            if ok:
                self.buckets[int(time.monotonic() - self.t0)] = (
                    self.buckets.get(int(time.monotonic() - self.t0), 0) + 1
                )

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(3)

    def report(self, start_s: int, end_s: int) -> str:
        vals = [self.buckets.get(s, 0) for s in range(start_s, end_s)]
        if not vals:
            return "n/a"
        return f"min {min(vals)} / avg {sum(vals) / len(vals):.0f} fps-per-second"


def main() -> None:
    cams = [
        c
        for c in camera_enumeration.list_cameras()
        if str(c.get("unique_id", "")).startswith("0x")
    ]
    if len(cams) < 3:
        print("need 3 cameras,", len(cams), "found")
        return
    counters: list[Counter] = []
    for c in cams:
        cap = cv2.VideoCapture(c["index"], cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok = False
        t0 = time.monotonic()
        while time.monotonic() - t0 < 15 and not ok:
            ok, _ = cap.read()
        print(f"open {c['unique_id']}: {'ok' if ok else 'NO FRAME'}")
        counters.append(Counter(c["unique_id"], cap))

    for ctr in counters:
        ctr.start()

    print(f"\n== quiet baseline ({MEASURE_QUIET_S}s, all 3 streaming) ==")
    time.sleep(MEASURE_QUIET_S)
    for ctr in counters:
        print(f"  {ctr.label}: {ctr.report(0, MEASURE_QUIET_S)}")

    print(f"\n== enumeration storm ({STORM_ENUMS} subprocesses back-to-back) ==")
    storm_start = int(time.monotonic() - counters[0].t0)
    for i in range(STORM_ENUMS):
        t = time.monotonic()
        n = len(camera_enumeration.list_cameras())
        print(f"  enum {i + 1}: {n} cameras in {time.monotonic() - t:.1f}s")
    storm_end = int(time.monotonic() - counters[0].t0) + 1
    time.sleep(1)
    for ctr in counters:
        print(f"  {ctr.label} during storm: {ctr.report(storm_start, storm_end)}")

    print(f"\n== quiet again ({MEASURE_QUIET_S}s) ==")
    quiet2_start = int(time.monotonic() - counters[0].t0)
    time.sleep(MEASURE_QUIET_S)
    for ctr in counters:
        print(f"  {ctr.label}: {ctr.report(quiet2_start, quiet2_start + MEASURE_QUIET_S)}")

    print("\n== open-vs-enumeration race (close one cam, reopen DURING an enum) ==")
    victim = counters[0]
    victim.stop()
    victim.cap.release()
    time.sleep(1)
    for attempt in range(3):
        enum_thread = threading.Thread(
            target=camera_enumeration.list_cameras, daemon=True
        )
        enum_thread.start()  # enumeration in flight...
        cap = cv2.VideoCapture(cams[0]["index"], cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok = False
        t0 = time.monotonic()
        while time.monotonic() - t0 < 15 and not ok:
            ok, _ = cap.read()
        print(
            f"  race {attempt + 1}: open {'ok' if ok else 'NO FRAME'} "
            f"(first frame {time.monotonic() - t0:.1f}s)"
        )
        enum_thread.join(15)
        cap.release()
        time.sleep(2)

    for ctr in counters[1:]:
        ctr.stop()
        ctr.cap.release()

    print(
        "\nRead: if per-second fps holds ~30 through the storm and races open"
        "\nfine, enumeration is innocent -> the bug is in lelab's stream"
        "\nlifecycle itself. If fps craters during the storm or raced opens get"
        "\nNO FRAME, the enumeration subprocess is the disruptor and lelab"
        "\nmust stop enumerating while captures are open."
    )


if __name__ == "__main__":
    main()
