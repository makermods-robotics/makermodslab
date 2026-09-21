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
"""Readiness/lifecycle seams with fake codecs and workers; no thread starts."""

import gc
import queue
import threading
import weakref
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from makermodslab import recording_preparation as prep


class FakeWorker:
    fail_start = False
    fail_open = False
    pending = False

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.ready = threading.Event()
        self.preparation_error = None
        self.ident = None
        self.alive = False

    def start(self):
        if self.fail_start:
            raise RuntimeError("thread creation failed")
        self.ident = 1
        self.alive = True
        if self.fail_open:
            self.preparation_error = RuntimeError("codec open failed")
        if not self.pending:
            self.ready.set()

    def is_alive(self):
        return self.alive

    def join(self, timeout):
        assert self.ident is not None
        self.alive = False


@pytest.fixture
def encoder(monkeypatch):
    monkeypatch.setattr(prep, "_PreparedCameraEncoder", FakeWorker)
    original = SimpleNamespace(
        fps=30, _rgb_encoder=object(), _depth_encoder=object(), queue_maxsize=4, _encoder_threads=2
    )
    return prep.PreparedStreamingVideoEncoder(original)


def test_preparation_preserves_actual_workers_for_frame0_and_resets_next_episode(encoder, tmp_path):
    images = {
        "observation.images.left_cam": np.zeros((8, 10, 3), np.uint8),
        "observation.images.right_cam": np.zeros((8, 10, 3), np.uint8),
    }
    encoder._dropped_frames = {"old": 4}
    assert encoder.prepare(images, tmp_path, [], lambda: False)
    workers = dict(encoder._threads)
    assert encoder._dropped_frames == {}
    assert all(q.empty() for q in encoder._frame_queues.values())
    encoder.start_episode(list(images), tmp_path)
    assert encoder._threads == workers
    with pytest.raises(RuntimeError, match="not prepared"):
        encoder.start_episode(list(images), tmp_path)
    encoder.cancel_episode()
    assert encoder.prepare(images, tmp_path, [], lambda: False)
    assert all(encoder._threads[key] is not worker for key, worker in workers.items())
    encoder.cancel_episode()


@pytest.mark.parametrize("failure", ["fail_start", "fail_open"])
def test_preparation_failure_preserves_original_error_and_ends_workers(
    encoder, tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(FakeWorker, failure, True)
    with pytest.raises(RuntimeError):
        encoder.prepare({"camera": np.zeros((8, 8, 3), np.uint8)}, tmp_path, [], lambda: False)
    assert not encoder._episode_active
    assert not encoder._threads
    assert list(tmp_path.iterdir())  # Failure artifacts preserved for diagnosis.


def test_cancel_and_timeout_are_bounded_and_do_not_feed_frames(encoder, tmp_path, monkeypatch):
    monkeypatch.setattr(FakeWorker, "pending", True)
    calls = iter([False, True])
    assert not encoder.prepare({"camera": np.zeros((8, 8, 3), np.uint8)}, tmp_path, [], lambda: next(calls))
    assert not encoder._episode_active
    assert not list(tmp_path.iterdir())
    with pytest.raises(TimeoutError):
        encoder.prepare({"camera": np.zeros((8, 8, 3), np.uint8)}, tmp_path, [], lambda: False, timeout_s=0)
    assert not encoder._episode_active


def test_invalid_shape_creates_no_artifact(encoder, tmp_path):
    with pytest.raises(ValueError):
        encoder.prepare({"camera": np.zeros((0, 8, 3), np.uint8)}, tmp_path, [], lambda: False)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("layout", ["hwc", "chw", "strided", "float_hwc", "float_chw", "depth", "queued_hwc"])
def test_worker_opens_actual_context_before_dequeue_without_dummy_frames(monkeypatch, tmp_path, layout):
    # Real RGB PyAV conversion with a fake output stream; no encoding or thread.
    depth = layout == "depth"
    events = []
    encoded = []
    stream = SimpleNamespace(codec_context=SimpleNamespace(open=lambda: events.append("codec_open")))

    def encode(frame=None):
        if frame is not None:
            encoded.append(frame)
        return []

    stream.encode = encode
    container = SimpleNamespace(
        add_stream=lambda *a, **k: stream,
        start_encoding=lambda: events.append("header"),
        close=lambda: None,
        mux=lambda _: None,
    )
    monkeypatch.setattr(prep.av, "open", lambda *a, **k: container)

    def reject_pil(*args, **kwargs):
        raise AssertionError("RGB encoding must not round-trip through PIL")

    monkeypatch.setattr(Image, "fromarray", reject_pil)
    monkeypatch.setattr(prep, "quantize_depth", lambda *a, **k: SimpleNamespace())
    config = SimpleNamespace(vcodec="fake", pix_fmt="fake", get_codec_options=lambda *a, **k: {})
    if depth:
        from lerobot.configs.video import DepthEncoderConfig

        config = object.__new__(DepthEncoderConfig)
        config.vcodec = "fake"
        config.pix_fmt = "fake"
        config.get_codec_options = lambda *a, **k: {}
    q = queue.Queue()
    expected = []
    queued_buffers = []
    for offset in (0, 19):
        rgb = (np.arange(8 * 10 * 3).reshape(8, 10, 3) + offset).astype(np.uint8)
        expected.append(rgb)
        frame = rgb
        if layout.startswith("float"):
            frame = rgb.astype(np.float32) / 255
        if layout.endswith("chw"):
            frame = frame.transpose(2, 0, 1).copy()
        elif layout == "strided":
            backing = np.zeros((8, 20, 3), dtype=np.uint8)
            backing[:, ::2] = rgb
            frame = backing[:, ::2]
            assert not frame.flags.c_contiguous
        elif depth:
            frame = np.full((8, 10), offset, np.uint16)
        if layout == "queued_hwc":
            original = SimpleNamespace(
                fps=30,
                _rgb_encoder=config,
                _depth_encoder=object(),
                queue_maxsize=4,
                _encoder_threads=2,
            )
            encoder = prep.PreparedStreamingVideoEncoder(original)
            encoder._episode_active = True
            encoder._threads["camera"] = SimpleNamespace(is_alive=lambda: True)
            encoder._frame_queues["camera"] = q
            encoder.feed_frame("camera", frame)  # Company's real protective copy.
            queued = q.queue[-1]
            assert not np.shares_memory(queued, frame)
            queued_buffers.append((weakref.ref(queued), queued.ctypes.data))
            expected[-1] = rgb.copy()
            frame[:] = 0  # Reusing the producer buffer cannot change queued RGB.
            del queued
            encoder._episode_active = False  # No background worker was started.
        else:
            q.put(frame)
    q.put(None)
    original_get = q.get

    def get(*a, **k):
        events.append("dequeue")
        return original_get(*a, **k)

    q.get = get
    result = queue.Queue()
    worker = prep._PreparedCameraEncoder(
        dimensions=(10, 8),
        video_path=tmp_path / "video.mp4",
        fps=30,
        video_encoder=config,
        frame_queue=q,
        result_queue=result,
        stop_event=threading.Event(),
    )
    worker.run()
    assert worker.preparation_error is None
    assert events[:3] == ["codec_open", "header", "dequeue"]
    assert [frame.pts for frame in encoded] == [0, 1]
    assert all(frame.time_base == Fraction(1, 30) for frame in encoded)
    if not depth:
        for frame, rgb in zip(encoded, expected, strict=True):
            assert frame.format.name == "rgb24"
            np.testing.assert_array_equal(frame.to_ndarray(format="rgb24"), rgb)
    if layout == "queued_hwc":
        gc.collect()
        for video_frame, (backing, address) in zip(encoded, queued_buffers, strict=True):
            assert backing() is not None  # Fake encoder retains AV frames after run().
            assert video_frame.planes[0].buffer_ptr == address
    status, stats = result.get_nowait()
    assert status == "ok"
    assert stats["count"].item() == 160  # Exactly two real 8x10 frames.


def test_prepare_uses_processed_camera_frame_then_aligns_without_dataset_write(monkeypatch, tmp_path):
    calls = []
    raw = np.zeros((8, 10, 3), np.uint8)
    processed = np.ones((8, 10, 3), np.uint8)
    robot = SimpleNamespace(cameras={"left_cam": object()}, get_observation=lambda: {"left_cam": raw})
    dataset = SimpleNamespace(
        root=tmp_path,
        features={
            "observation.images.left_cam": {
                "dtype": "video",
                "shape": (8, 10, 3),
                "names": ["height", "width", "channels"],
            }
        },
        meta=SimpleNamespace(video_keys=["observation.images.left_cam"], depth_keys=[]),
    )

    def prepare(images, *args):
        calls.append("codec")
        assert images["observation.images.left_cam"] is processed
        return True

    encoder = SimpleNamespace(prepare=prepare)

    def align():
        calls.append("align")
        return True

    assert prep.prepare_recording_episode(
        robot,
        dataset,
        encoder,
        align,
        lambda: False,
        observation_processor=lambda obs: {"left_cam": processed},
    )
    assert calls == ["codec", "align"]


def test_nonstreaming_and_custom_encoders_are_unchanged():
    for existing in (None, object()):
        dataset = SimpleNamespace(writer=SimpleNamespace(_streaming_encoder=existing))
        assert prep.install_episode_preparation(dataset) is None
        assert dataset.writer._streaming_encoder is existing


def test_real_dataset_writer_keeps_frame0_camera_action_alignment(encoder, tmp_path):
    from lerobot.datasets.dataset_writer import DatasetWriter
    from lerobot.utils.constants import DEFAULT_FEATURES

    keys = ["observation.images.left_cam", "observation.images.right_cam"]
    features = {**DEFAULT_FEATURES, "action": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]}}
    for key in keys:
        features[key] = {"dtype": "video", "shape": (8, 10, 3), "names": ["height", "width", "channels"]}
    meta = SimpleNamespace(features=features, total_episodes=0, video_keys=keys, depth_keys=[], fps=30)
    writer = DatasetWriter(
        meta=meta,
        root=tmp_path,
        rgb_encoder=object(),
        depth_encoder=object(),
        streaming_encoder=encoder,
        encoder_threads=None,
        batch_encoding_size=1,
    )
    warmup = {key: np.zeros((8, 10, 3), np.uint8) for key in keys}
    assert encoder.prepare(warmup, tmp_path, [], lambda: False)
    assert writer.episode_buffer["size"] == 0
    assert writer.episode_buffer["timestamp"] == []
    assert all(q.empty() for q in encoder._frame_queues.values())
    real = {key: np.full((8, 10, 3), index + 11, np.uint8) for index, key in enumerate(keys)}
    writer.add_frame({**real, "action": np.array([23], np.float32), "task": "pick"})
    assert writer.episode_buffer["frame_index"] == [0]
    assert writer.episode_buffer["timestamp"] == [0.0]
    assert writer.episode_buffer["size"] == 1
    assert writer.episode_buffer["action"][0].item() == 23
    for key in keys:
        np.testing.assert_array_equal(encoder._frame_queues[key].get_nowait(), real[key])
    encoder.cancel_episode()


def test_rgb_frame_shares_storage_and_keeps_numpy_alive():
    array = np.full((8, 10, 3), 47, dtype=np.uint8)
    backing = weakref.ref(array)
    frame = prep._rgb_video_frame(array)
    assert frame.planes[0].buffer_ptr == array.ctypes.data
    array[0, 0] = [5, 11, 23]
    np.testing.assert_array_equal(frame.to_ndarray(format="rgb24")[0, 0], [5, 11, 23])
    del array
    gc.collect()
    assert backing() is not None
    np.testing.assert_array_equal(frame.to_ndarray(format="rgb24")[0, 0], [5, 11, 23])
    del frame
    gc.collect()
    assert backing() is None


@pytest.mark.parametrize("layout", ["strided", "chw_transposed", "negative_stride"])
def test_rgb_frame_falls_back_for_unpacked_layout(monkeypatch, layout):
    array = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    if layout == "strided":
        array = array[:, ::2]
    elif layout == "chw_transposed":
        array = array.transpose(2, 0, 1).copy().transpose(1, 2, 0)
    else:
        array = array[::-1]
    assert not array.flags.c_contiguous
    real_from_ndarray = prep.av.VideoFrame.from_ndarray

    def reject_buffer(*args, **kwargs):
        raise AssertionError("Unpacked layout must use copying constructor")

    monkeypatch.setattr(
        prep.av,
        "VideoFrame",
        SimpleNamespace(
            from_numpy_buffer=reject_buffer,
            from_ndarray=real_from_ndarray,
        ),
    )
    frame = prep._rgb_video_frame(array)
    np.testing.assert_array_equal(frame.to_ndarray(format="rgb24"), array)
    assert frame.planes[0].buffer_ptr != array.ctypes.data


def test_rgb_frame_supports_pyav_without_buffer_api(monkeypatch):
    array = np.full((8, 10, 3), 29, np.uint8)
    monkeypatch.setattr(
        prep.av,
        "VideoFrame",
        SimpleNamespace(
            from_ndarray=prep.av.VideoFrame.from_ndarray,
        ),
    )
    frame = prep._rgb_video_frame(array)
    np.testing.assert_array_equal(frame.to_ndarray(format="rgb24"), array)
    assert frame.planes[0].buffer_ptr != array.ctypes.data


def test_rgb_frame_does_not_hide_buffer_constructor_errors(monkeypatch):
    def fail(*args, **kwargs):
        raise MemoryError("allocation failed")

    monkeypatch.setattr(prep.av, "VideoFrame", SimpleNamespace(from_numpy_buffer=fail))
    with pytest.raises(MemoryError, match="allocation failed"):
        prep._rgb_video_frame(np.zeros((8, 10, 3), np.uint8))
