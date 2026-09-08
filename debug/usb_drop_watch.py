"""Live USB drop watcher — wiggle each cable and the guilty connector confesses.

Polls the IOUSB registry ~4x/second and prints the instant any camera, serial
adapter, or hub on the rig detaches or (re)attaches, with wall-clock timestamps.
sessionID changes are detected too (a drop-and-reattach faster than one poll).

Background (2026-07-13): camera 0 (locationID 0x1133) was proven to drop off
the bus mid-recording — its IOUSB sessionID (= mach ticks at attach) converted
to 13:51:38.125, the exact second of the session kill — and again at 14:05:06.
Every software theory was eliminated first (probes 1+4 clean, Safari repro,
encoders off). Trigger correlates with arm motion; suspect cable/connector.

Run in a spare terminal (no camera permission needed, safe alongside anything):

    python3 usb_drop_watch.py

Then wiggle one cable at a time — camera cables at both ends, hub uplinks too.
Ctrl-C to stop.
"""

import datetime
import re
import subprocess
import time

POLL_S = 0.25
WATCH = re.compile(
    r"(USB Camera|KD-USB Cameras|USB (?:Single )?Serial|USB[23]\.[12] Hub)@([0-9a-f]+)", re.I
)


def snapshot() -> dict[str, int]:
    """{'USB Camera@01133000': sessionID, ...} for every watched device."""
    out = subprocess.run(
        ["ioreg", "-p", "IOUSB", "-l", "-w0"], capture_output=True, text=True
    ).stdout
    devices: dict[str, int] = {}
    for chunk in re.split(r"\+\-o ", out):
        m = WATCH.match(chunk)
        if not m:
            continue
        sid = re.search(r"\"sessionID\" = (\d+)", chunk)
        if sid:
            devices[f"{m.group(1)}@{m.group(2)}"] = int(sid.group(1))
    return devices


def main() -> None:
    prev = snapshot()
    print(f"watching {len(prev)} devices; wiggle away (Ctrl-C to stop)")
    for name in sorted(prev):
        print(f"  {name}")
    while True:
        time.sleep(POLL_S)
        cur = snapshot()
        now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        for name in prev.keys() - cur.keys():
            print(f"{now}  ❌ DETACHED: {name}")
        for name in cur.keys() - prev.keys():
            print(f"{now}  ✅ attached: {name}")
        for name in prev.keys() & cur.keys():
            if prev[name] != cur[name]:
                print(f"{now}  🔄 DROPPED+reattached between polls: {name}")
        prev = cur


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")
