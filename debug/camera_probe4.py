"""Probe 4: cameras + serial together — the one combination every kill shares.

Probe 1 (2026-07-13) proved 3 concurrent 480p captures run clean at 30 fps with
NO serial traffic. Every in-app kill happens once 30 Hz serial joins the
cameras (and the connect-phase 'Lock' write failures are the same interference
seen from the serial side). Safari reproduced the kill, eliminating Chrome.
This probe reproduces the app's bus load with NOTHING else — no browser, no
recording machinery, no encoding:

  phase A (10 s): 3 cameras only               — expect 30/30/30 (probe-1 baseline)
  phase B (40 s): 3 cameras + 4 buses reading
                  Present_Position at ~30 Hz   — the recording-session load
  phase C (10 s): cameras only again           — does a stalled camera recover?

Prints a per-second fps table per camera, serial read error counts per port,
and flags any camera whose frames stop (the in-app failure is camera 0 =
lowest-locationID unit going silent within ~10 s of phase-B-like load).

Serial reads are read-only (Present_Position); torque is never touched — safe
with the arms resting. Run from your camera-authorized terminal, makerlab
stopped, all app tabs closed:

    .venv/bin/python camera_probe4.py
"""

import glob
import sys
import threading
import time

import cv2

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

# lelab.camera_enumeration was deleted in the camera revert (8d7634d); the
# equivalent enumerator lives in lelab.server.
from lelab import server as _lelab_server

PHASE_A_S = 10
PHASE_B_S = 40
PHASE_C_S = 10
SERIAL_HZ = 30
FIRST_FRAME_PATIENCE = 15.0

SO101_MOTORS = {
    f"m{i}": Motor(i, "sts3215", MotorNormMode.RANGE_0_100) for i in range(1, 7)
}


class CamCounter:
    """Reader thread + per-second frame buckets for one capture."""

    def __init__(self, label: str, cap: cv2.VideoCapture) -> None:
        self.label = label
        self.cap = cap
        self.buckets: dict[int, int] = {}
        self.read_failures = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                self.buckets[int(time.monotonic())] = (
                    self.buckets.get(int(time.monotonic()), 0) + 1
                )
            else:
                self.read_failures += 1
                time.sleep(0.01)


class SerialHammer:
    """~30 Hz Present_Position sync_read loop on one bus, counting errors."""

    def __init__(self, port: str) -> None:
        self.port = port
        self.bus = FeetechMotorsBus(port=port, motors=dict(SO101_MOTORS))
        self.reads = 0
        self.errors = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        period = 1.0 / SERIAL_HZ
        while not self.stop.is_set():
            t0 = time.monotonic()
            try:
                self.bus.sync_read("Present_Position", normalize=False, num_retry=0)
                self.reads += 1
            except Exception:
                self.errors += 1
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)


def open_cam(index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    t0 = time.monotonic()
    while time.monotonic() - t0 < FIRST_FRAME_PATIENCE:
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
    cap.release()
    return None


def fps_row(counters: list[CamCounter], second: int) -> str:
    return "  ".join(f"{c.label}:{c.buckets.get(second, 0):>3}" for c in counters)


def main() -> None:
    cams = [
        c
        for c in _lelab_server._avfoundation_cameras_in_cv2_order()
        if str(c.get("unique_id", "")).startswith("0x")
    ]
    print("USB cameras:", [(c["index"], c["unique_id"]) for c in cams])
    if len(cams) < 3:
        print("Need all 3 cameras connected.")
        sys.exit(1)

    ports = sorted(glob.glob("/dev/tty.usbmodem*"))
    print("Serial ports:", ports)
    if len(ports) < 4:
        print("Expected 4 arm serial ports; found", len(ports), "- probing with what's there.")

    counters: list[CamCounter] = []
    for c in cams:
        cap = open_cam(c["index"])
        if cap is None:
            print(f"camera {c['index']} ({c['unique_id']}): OPEN FAILED — aborting")
            sys.exit(1)
        counters.append(CamCounter(f"cam{c['index']}", cap))
        print(f"camera {c['index']} ({c['unique_id']}): open, first frame OK")

    for ctr in counters:
        ctr.thread.start()

    def watch(label: str, seconds: int) -> None:
        start = int(time.monotonic()) + 1
        time.sleep(1)
        for s in range(seconds):
            time.sleep(1)
            print(f"  [{label} t={s + 1:>2}s] {fps_row(counters, start + s)}")

    print(f"\n== phase A: cameras only ({PHASE_A_S}s) ==")
    watch("A", PHASE_A_S)

    print(f"\n== phase B: cameras + serial @ {SERIAL_HZ} Hz x {len(ports)} buses ({PHASE_B_S}s) ==")
    hammers: list[SerialHammer] = []
    for p in ports:
        try:
            h = SerialHammer(p)
            h.bus.connect(handshake=False)
            h.thread.start()
            hammers.append(h)
            print(f"  serial up: {p}")
        except Exception as e:
            print(f"  serial FAILED to open {p}: {e}")
    watch("B", PHASE_B_S)
    for h in hammers:
        h.stop.set()
    for h in hammers:
        h.thread.join(2)
        try:
            h.bus.disconnect(disable_torque=False)
        except Exception:
            pass

    print(f"\n== phase C: cameras only again ({PHASE_C_S}s) ==")
    watch("C", PHASE_C_S)

    for ctr in counters:
        ctr.stop.set()
    print("\n== summary ==")
    for ctr in counters:
        print(f"  {ctr.label}: cap.read() hard failures: {ctr.read_failures}")
        ctr.cap.release()
    for h in hammers:
        print(f"  {h.port}: reads={h.reads} errors={h.errors}")
    print(
        "\nHow to read it: a camera at 30 fps in phase A that stalls (0 fps or"
        "\nhard read failures) during phase B = serial traffic interferes with"
        "\nthat camera's capture — the in-app killer, reproduced minimally."
        "\nSerial errors piling up while cameras stream = the same interference"
        "\nseen from the serial side (the connect-phase 'Lock' failures)."
        "\nEverything clean for all 60s = the interaction needs the recording"
        "\nprocess itself (image writer / GIL / timing), not the bus."
    )


if __name__ == "__main__":
    main()
