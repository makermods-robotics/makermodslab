#!/usr/bin/env python
"""Experiment: does a live NSRunLoop let this process's AVFoundation device
snapshot refresh on USB camera hotplug/replug?

Background: the makermodslab server's in-process AVFoundation device list is
snapshotted at first touch and never refreshes (device-connection
notifications need an active NSRunLoop, which uvicorn doesn't run). That's
why a hot-plugged camera needs a server restart, and why a same-port replug
yields a stale device whose cap.read() blocks forever (blank preview). This
script tests whether keeping a runloop alive fixes both — if it does, the
server fix is a tiny daemon thread and the restart requirement disappears.

Run each mode from the same terminal you normally run `makermodslab` in (it
has camera permission), with the server STOPPED (it holds the cameras):

    .venv/bin/python camera_runloop_experiment.py --runloop none     # control
    .venv/bin/python camera_runloop_experiment.py --runloop thread   # candidate fix
    .venv/bin/python camera_runloop_experiment.py --runloop main     # fallback variant

Protocol per run: start it, wait for two "read ok" lines, unplug the watched
camera, wait ~10 s, plug it back into the SAME port, watch another ~20 s,
Ctrl-C. Two signals to watch:

  - "DEVICE LIST CHANGED" lines on unplug/replug -> the snapshot refreshes.
  - the read probe returning to "ok" after the replug -> stale-device blank
    preview is cured.

Expected control result (--runloop none): no list change on unplug, probe
stuck on open-failed/TIMEOUT after replug — today's bug, reproduced. If
--runloop thread shows both signals recovering, the fix is proven. Restart
the script between trials (a timed-out probe leaks its wedged capture).
"""

import argparse
import threading
import time
import urllib.error
import urllib.request

import cv2
import objc
from Foundation import NSDate, NSDefaultRunLoopMode, NSRunLoop

from makermodslab.camera_identity import list_cameras_in_process

# Generous: a USB camera's first frame after a replug can take several seconds
# to arrive; only a read still stuck past this is the wedge signature.
PROBE_TIMEOUT_S = 8.0
CYCLE_S = 3.0


def fmt(cams: list[dict]) -> str:
    return ", ".join(f"[{c['index']}] {c['name']} {c['unique_id']}" for c in cams) or "(none)"


def probe_read(index: int) -> str:
    """Open cv2 `index` and read one frame under a watchdog thread."""
    result: dict = {}

    def work():
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            result["r"] = "open-failed"
            return
        ok, frame = cap.read()
        result["r"] = "ok" if ok and frame is not None else "blank"
        cap.release()

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(PROBE_TIMEOUT_S)
    if t.is_alive():
        return "TIMEOUT (read wedged — capture leaked; restart script before the next trial)"
    return result.get("r", "no-result")


def start_runloop_thread() -> None:
    """Dedicated thread: first AVFoundation touch, then keep its runloop alive."""
    ready = threading.Event()

    def run():
        with objc.autorelease_pool():
            list_cameras_in_process()
        ready.set()
        rl = NSRunLoop.currentRunLoop()
        while True:
            with objc.autorelease_pool():
                busy = rl.runMode_beforeDate_(NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(1.0))
            if not busy:  # no input sources attached yet — don't spin
                time.sleep(0.1)

    t = threading.Thread(target=run, daemon=True, name="avf-runloop")
    t.start()
    if not ready.wait(10):
        raise SystemExit("runloop thread failed to initialize AVFoundation")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runloop", choices=["none", "thread", "main"], default="none")
    ap.add_argument(
        "--watch",
        default="USB Camera",
        help="probe the first camera whose name contains this substring",
    )
    args = ap.parse_args()

    # A running server (or an open browser preview tab) holds the cameras and
    # makes every probe read blank, masking the experiment's result entirely.
    try:
        urllib.request.urlopen(  # noqa: S310  # nosec B310 — fixed localhost URL, no user input
            "http://localhost:8000/available-cameras", timeout=1
        )
        print(
            "WARNING: a makermodslab server is RUNNING on :8000. Stop it and close "
            "any browser tabs with preview tiles before trusting these results — "
            "its captures make every probe read blank.",
            flush=True,
        )
    except (urllib.error.URLError, OSError):
        pass

    pump = None
    if args.runloop == "thread":
        start_runloop_thread()
    elif args.runloop == "main":
        with objc.autorelease_pool():
            list_cameras_in_process()  # first AVFoundation touch on the main thread
        rl = NSRunLoop.currentRunLoop()

        def pump(seconds: float) -> None:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                with objc.autorelease_pool():
                    busy = rl.runMode_beforeDate_(
                        NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.25)
                    )
                if not busy:
                    time.sleep(0.1)

    print(
        f"runloop={args.runloop}; watching first camera matching {args.watch!r}; Ctrl-C to stop",
        flush=True,
    )
    last_ids: list[str] | None = None
    while True:
        with objc.autorelease_pool():
            cams = list_cameras_in_process() or []
        ids = [c["unique_id"] for c in cams]
        if ids != last_ids:
            last_ids = ids
            print(f"{time.strftime('%H:%M:%S')} DEVICE LIST CHANGED: {fmt(cams)}", flush=True)
        # Probe EVERY match, not just the first: the two robot cameras are
        # identically named, and unplugging the second twin while probing only
        # the first would show a healthy "ok" forever.
        targets = [c for c in cams if args.watch.lower() in c["name"].lower()]
        if not targets:
            print(
                f"{time.strftime('%H:%M:%S')} probe: no camera matching "
                f"{args.watch!r} in the in-process list",
                flush=True,
            )
        for target in targets:
            started = time.monotonic()
            outcome = probe_read(target["index"])
            print(
                f"{time.strftime('%H:%M:%S')} probe [{target['index']}] "
                f"{target['unique_id']}: read {outcome} "
                f"({time.monotonic() - started:.1f}s)",
                flush=True,
            )
        if pump is not None:
            pump(CYCLE_S)
        else:
            time.sleep(CYCLE_S)


if __name__ == "__main__":
    main()
