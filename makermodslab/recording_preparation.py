# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 MakerMods. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare actual episode encoders before starting the recording clock.

The worker loop and streaming setup are adapted from the SHA-pinned LeRobot
video_utils._CameraEncoderThread/StreamingVideoEncoder (eaab69339). The fork
has no preparation hook: opening threads alone defers codec creation until
frame zero. This local extension opens the SAME contexts used by the episode,
without encoding dummy frames or touching dataset counters/stats/timestamps.
Codec open removes known lazy initialization; first-encode costs and achievable
camera/encoder throughput still need hardware measurement.
"""

import contextlib
import logging
import queue
import shutil
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from lerobot.datasets.video_utils import StreamingVideoEncoder, _CameraEncoderThread, quantize_depth

logger = logging.getLogger(__name__)


def image_dimensions(image):
    shape = image.shape
    if len(shape) == 3 and shape[0] in (1, 3):
        shape = (shape[1], shape[2], shape[0])
    if len(shape) not in (2, 3) or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"Invalid camera frame shape for recording: {image.shape}")
    return int(shape[1]), int(shape[0])


class _PreparedCameraEncoder(_CameraEncoderThread):
    def __init__(self, *, dimensions, **kwargs):
        super().__init__(**kwargs)
        self.dimensions = dimensions
        self.ready = threading.Event()
        self.preparation_error = None

    def run(self):
        from lerobot.datasets.compute_stats import RunningQuantileStats, auto_downsample_height_width

        container = None
        output_stream = None
        stats_tracker = RunningQuantileStats()
        frame_count = 0
        try:
            container = av.open(str(self.video_path), "w")
            output_stream = container.add_stream(
                self.video_encoder.vcodec,
                self.fps,
                options=self.video_encoder.get_codec_options(self.encoder_threads, as_strings=True),
            )
            output_stream.pix_fmt = self.video_encoder.pix_fmt
            output_stream.width, output_stream.height = self.dimensions
            output_stream.time_base = Fraction(1, self.fps)
            output_stream.codec_context.open()
            container.start_encoding()
            self.ready.set()

            while not self.stop_event.is_set():
                try:
                    frame_data = self.frame_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if frame_data is None:
                    break
                if isinstance(frame_data, np.ndarray):
                    if frame_data.ndim == 3 and frame_data.shape[0] in (1, 3):
                        frame_data = frame_data.transpose(1, 2, 0)
                    if not self.is_depth and frame_data.dtype != np.uint8:
                        frame_data = (frame_data * 255).astype(np.uint8)
                if frame_data.ndim == 2:
                    frame_data = frame_data[..., None]
                if image_dimensions(frame_data) != self.dimensions:
                    raise ValueError("Camera dimensions changed after episode preparation")
                if not self.is_depth:
                    video_frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                else:
                    video_frame = quantize_depth(
                        frame_data,
                        depth_min=self.video_encoder.depth_min,
                        depth_max=self.video_encoder.depth_max,
                        shift=self.video_encoder.shift,
                        use_log=self.video_encoder.use_log,
                        video_backend=self.video_encoder.video_backend,
                    )
                video_frame.pts = frame_count
                video_frame.time_base = Fraction(1, self.fps)
                packet = output_stream.encode(video_frame)
                if packet:
                    container.mux(packet)
                img_chw = frame_data.transpose(2, 0, 1)
                img_downsampled = auto_downsample_height_width(img_chw)
                channels = img_downsampled.shape[0]
                stats_tracker.update(img_downsampled.transpose(1, 2, 0).reshape(-1, channels))
                frame_count += 1
            if frame_count:
                packet = output_stream.encode()
                if packet:
                    container.mux(packet)
            container.close()
            container = None
            self.result_queue.put(("ok", stats_tracker.get_statistics() if frame_count >= 2 else None))
        except Exception as exc:
            self.preparation_error = exc
            logger.exception("Episode encoder failed: %s", self.video_path)
            self.result_queue.put(("error", str(exc)))
        finally:
            if container is not None:
                with contextlib.suppress(Exception):
                    container.close()
            self.ready.set()  # Failure also releases the readiness waiter.


class PreparedStreamingVideoEncoder(StreamingVideoEncoder):
    def __init__(self, original):
        super().__init__(
            fps=original.fps,
            rgb_encoder=original._rgb_encoder,
            depth_encoder=original._depth_encoder,
            queue_maxsize=original.queue_maxsize,
            encoder_threads=original._encoder_threads,
        )
        self._prepared_signature = None
        self._preserve_failure = False

    def prepare(self, images, temp_dir, depth_video_keys, cancelled, timeout_s=15.0):
        if self._episode_active:
            raise RuntimeError("Previous episode encoder is still active")
        # Validate all cameras before creating any avoidable output artifacts.
        dimensions = {key: image_dimensions(image) for key, image in images.items()}
        if not images:
            return not cancelled()
        if not Path(temp_dir).is_dir():
            raise ValueError("Recording dataset directory does not exist")
        if cancelled():
            return False
        signature = (tuple(images), tuple(depth_video_keys), Path(temp_dir))
        self._preserve_failure = False
        self._dropped_frames.clear()
        self._episode_active = True
        try:
            for key in images:
                frame_queue = queue.Queue(maxsize=self.queue_maxsize)
                result_queue = queue.Queue(maxsize=1)
                stop = threading.Event()
                path = Path(tempfile.mkdtemp(dir=temp_dir)) / f"{key.replace('/', '_')}_streaming.mp4"
                worker = _PreparedCameraEncoder(
                    dimensions=dimensions[key],
                    video_path=path,
                    fps=self.fps,
                    video_encoder=self._depth_encoder if key in depth_video_keys else self._rgb_encoder,
                    frame_queue=frame_queue,
                    result_queue=result_queue,
                    stop_event=stop,
                    encoder_threads=self._encoder_threads,
                )
                self._frame_queues[key] = frame_queue
                self._result_queues[key] = result_queue
                self._stop_events[key] = stop
                self._video_paths[key] = path
                self._threads[key] = worker
                worker.start()
            deadline = time.monotonic() + timeout_s
            for worker in self._threads.values():
                while not worker.ready.wait(timeout=0.05):
                    if cancelled():
                        self.cancel_episode()
                        return False
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Timed out preparing episode video encoders")
                if worker.preparation_error is not None:
                    raise RuntimeError(
                        "Could not prepare episode video encoder"
                    ) from worker.preparation_error
            if cancelled():
                self.cancel_episode()
                return False
            self._prepared_signature = signature
            return True
        except Exception:
            self._preserve_failure = True
            self.cancel_episode()
            raise

    def start_episode(self, video_keys, temp_dir, depth_video_keys=None):
        # DatasetWriter still calls this for frame zero. Consume preparation
        # exactly once, without closing and recreating the ready contexts.
        signature = (tuple(video_keys), tuple(depth_video_keys or []), Path(temp_dir))
        if self._episode_active and signature == self._prepared_signature:
            self._prepared_signature = None
            return
        raise RuntimeError("Episode encoder was not prepared before frame zero")

    def close(self, preserve_failure=False):
        self._preserve_failure = self._preserve_failure or preserve_failure
        super().close()

    def cancel_episode(self):
        if not self._episode_active:
            return
        for event in self._stop_events.values():
            event.set()
        deadline = time.monotonic() + 1.0
        for key, worker in self._threads.items():
            if worker.ident is not None:
                worker.join(timeout=max(0.0, deadline - time.monotonic()))
            if worker.is_alive():
                logger.warning(
                    "Encoder still exiting; preserving temporary output: %s", self._video_paths[key]
                )
            elif not self._preserve_failure:
                shutil.rmtree(self._video_paths[key].parent, ignore_errors=True)
        self._cleanup()
        self._episode_active = False
        self._prepared_signature = None


def install_episode_preparation(dataset):
    """Return our encoder, or None for non-streaming/custom writer paths."""
    writer = getattr(dataset, "writer", None)
    original = getattr(writer, "_streaming_encoder", None)
    if type(original) is not StreamingVideoEncoder:
        return None
    if original._episode_active:
        raise RuntimeError("Cannot replace an encoder during an active episode")
    prepared = PreparedStreamingVideoEncoder(original)
    writer._streaming_encoder = prepared
    original.close()
    return prepared


def prepare_recording_episode(robot, dataset, encoder, realign, cancelled, observation_processor=None):
    """Prepare camera capture/codec context and align before any recorded frame."""
    from lerobot.utils.feature_utils import build_dataset_frame

    if cancelled():
        return False
    observation = None
    if encoder is not None or getattr(robot, "cameras", None):
        observation = robot.get_observation()
        if observation_processor is not None:
            observation = observation_processor(observation)
    if cancelled():
        return False
    if encoder is not None:
        frame = build_dataset_frame(dataset.features, observation, prefix="observation")
        images = {key: frame[key] for key in dataset.meta.video_keys}
        if not encoder.prepare(images, dataset.root, list(dataset.meta.depth_keys), cancelled):
            return False
    # Camera/codec preparation is a holding gap: chase the current leader LAST,
    # so its movement during initialization cannot become a first-frame jump.
    return not (cancelled() or realign() is False or cancelled())
