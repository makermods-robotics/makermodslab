"""Network validation and observation pacing without hardware or wall-clock waits."""
import pytest
from pydantic import ValidationError

from makermodslab.drtc._observation_rate import ObservationRateLimiter
from makermodslab.schemas.remote_network import GpuNetworkOptions
from makermodslab.schemas.sessions import RemoteInferenceOptions


def test_send_cap_leaves_control_ticks_available_and_does_not_catch_up():
    limiter = ObservationRateLimiter(5)
    sends, controls = [], []
    for tick in range(20):
        now = tick / 20
        controls.append(now)
        if limiter.ready(now):
            assert limiter.ready(now)  # Checking alone does not consume a send.
            limiter.sent(now)
            sends.append(now)
    assert len(controls) == 20
    assert sends == [0, .2, .4, .6, .8]
    assert limiter.ready(10)
    limiter.sent(10)
    assert not limiter.ready(10.05)


def test_automatic_does_not_add_a_send_limit():
    limiter = ObservationRateLimiter()
    for tick in range(20):
        assert limiter.ready(tick / 20)
        limiter.sent(tick / 20)


@pytest.mark.parametrize('hz', [-1, float('nan'), float('inf')])
def test_invalid_limiter_rate(hz):
    with pytest.raises(ValueError):
        ObservationRateLimiter(hz)


@pytest.mark.parametrize('options', [
    {'camera_send_hz': -1}, {'camera_send_hz': 21}, {'camera_send_hz': float('nan')},
    {'engine': 'sync', 'camera_send_hz': 5}, {'video_quality': 0},
    {'video_quality': 101}, {'video_quality': 50.5}, {'video_bitrate_kbps': 255},
    {'video_bitrate_kbps': 20001}, {'latency_k': -1}, {'latency_k': float('inf')},
])
def test_invalid_network_options(options):
    with pytest.raises(ValidationError):
        RemoteInferenceOptions.model_validate({'policy_ref': 'hub:someone/p', 'engine': 'rtc', 'fps': 20, **options})


def test_valid_send_cap_preserves_control_fps():
    value = RemoteInferenceOptions(policy_ref='hub:someone/p', engine='rtc', fps=20, camera_send_hz=5)
    assert value.fps == 20
    assert value.camera_send_hz == 5


@pytest.mark.parametrize('options', [{'region': 'invalid'}, {'tolerance': 0}, {'tolerance': float('nan')}])
def test_invalid_gpu_network_options(options):
    with pytest.raises(ValidationError):
        GpuNetworkOptions(**options)
