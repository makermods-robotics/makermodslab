"""Duration limits must budget execution, even after a long setup."""

import pytest

from makermodslab.drtc._run_timer import RunTimer


def test_setup_and_easing_do_not_consume_the_duration():
    now = [10.0]
    timer = RunTimer(60, clock=lambda: now[0])
    now[0] += 300  # connection, first inference and easing
    assert timer.elapsed_s == 0
    assert not timer.expired
    assert timer.start()
    now[0] += 59.9
    assert not timer.expired
    assert not timer.start()  # subsequent commands cannot renew the duration
    now[0] += 0.1
    assert timer.elapsed_s == pytest.approx(60)
    assert timer.expired


def test_an_unbounded_run_does_not_expire():
    now = [0.0]
    timer = RunTimer(0, clock=lambda: now[0])
    timer.start()
    now[0] = 10000
    assert timer.elapsed_s == 10000
    assert not timer.expired
