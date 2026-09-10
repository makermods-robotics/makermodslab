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

import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

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


@pytest.mark.parametrize("depth", [False, True])
def test_worker_opens_actual_context_before_dequeue_without_dummy_frames(monkeypatch, tmp_path, depth):
    # Run the worker body synchronously with fake PyAV; no service or thread.
    events = []
    encoded = []
    stream = SimpleNamespace(codec_context=SimpleNamespace(open=lambda: events.append("codec_open")))

    def encode(frame=None):
        if frame is not None:
            encoded.append((frame.pts, frame.time_base))
        return []

    stream.encode = encode
    container = SimpleNamespace(
        add_stream=lambda *a, **k: stream,
        start_encoding=lambda: events.append("header"),
        close=lambda: None,
        mux=lambda _: None,
    )
    monkeypatch.setattr(prep.av, "open", lambda *a, **k: container)
    monkeypatch.setattr(prep.av, "VideoFrame", SimpleNamespace(from_image=lambda _: SimpleNamespace()))
    monkeypatch.setattr(prep, "quantize_depth", lambda *a, **k: SimpleNamespace())
    config = SimpleNamespace(vcodec="fake", pix_fmt="fake", get_codec_options=lambda *a, **k: {})
    if depth:
        from lerobot.configs.video import DepthEncoderConfig

        config = object.__new__(DepthEncoderConfig)
        config.vcodec = "fake"
        config.pix_fmt = "fake"
        config.get_codec_options = lambda *a, **k: {}
    q = queue.Queue()
    shape = (8, 10) if depth else (8, 10, 3)
    q.put(np.zeros(shape, np.uint16 if depth else np.uint8))
    q.put(np.ones(shape, np.uint16 if depth else np.uint8))
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
    assert [pts for pts, _ in encoded] == [0, 1]
    assert all(float(tb) == 1 / 30 for _, tb in encoded)
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
