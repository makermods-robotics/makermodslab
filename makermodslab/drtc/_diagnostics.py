"""Local timing diagnostics; no Portal imports or changes to wire timestamps."""

import threading
import time
from collections import Counter, deque
from contextlib import contextmanager


@contextmanager
def startup_stage(name):
    started = time.perf_counter()
    print(f"[startup] {name}: starting", flush=True)
    try:
        yield
    except BaseException:
        print(f"[startup] {name}: failed after {time.perf_counter() - started:.1f}s", flush=True)
        raise
    else:
        print(f"[startup] {name}: done in {time.perf_counter() - started:.1f}s", flush=True)


def parameter_summary(policy):
    """Count loaded parameter elements by actual dtype/device, not config."""
    counts = Counter()
    for parameter in policy.parameters():
        counts[f"{parameter.dtype}@{parameter.device}"] += parameter.numel()
    return ", ".join(f"{key}={value:,}" for key, value in sorted(counts.items())) or "no parameters"


class CameraTimingMonitor:
    """Observe existing OpenCV buffers without reading cameras or touching motors.

    The timestamp is AFTER the driver read, not sensor exposure. Polling may
    miss updates, so observed_hz is a lower bound, never a certified sensor FPS.
    A five-second window and 200 Hz polling keep memory and work bounded.
    Cameras without these buffer attributes are reported as unavailable.
    """

    def __init__(self, cameras):
        self.cameras = dict(cameras)
        self.samples = {name: deque(maxlen=2000) for name in cameras}
        self.shapes = {}
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, name="camera-timing", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.done.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=0.2)

    def sample(self):
        for name, camera in self.cameras.items():
            frame_lock = getattr(camera, "frame_lock", None)
            if frame_lock is None or not frame_lock.acquire(blocking=False):
                continue
            try:
                stamp = getattr(camera, "latest_timestamp", None)
                shape = getattr(getattr(camera, "latest_frame", None), "shape", None)
            finally:
                frame_lock.release()
            if stamp is None or shape is None:
                continue
            with self.lock:
                samples = self.samples[name]
                if not samples or stamp > samples[-1]:
                    samples.append(stamp)
                self.shapes[name] = tuple(shape)

    def _run(self):
        while not self.done.is_set():
            self.sample()
            self.done.wait(0.005)

    def summary(self, now=None):
        now = time.perf_counter() if now is None else now
        parts = []
        with self.lock:
            for name, samples in self.samples.items():
                if not samples:
                    parts.append(f"{name}:unavailable")
                    continue
                recent = [stamp for stamp in samples if stamp >= now - 5]
                rate = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
                shape = "x".join(str(x) for x in self.shapes[name][:2][::-1])
                parts.append(
                    f"{name}:size={shape},buffer_age_ms={(now - samples[-1]) * 1000:.1f},"
                    f"observed_hz={rate:.1f}"
                )
        return " | ".join(parts) or "unavailable"
