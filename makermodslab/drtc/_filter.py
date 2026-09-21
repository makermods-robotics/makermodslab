"""Causal Butterworth low-pass for the per-tick action stream.

A Butterworth low-pass has a maximally flat passband (no ripple), unity DC
gain, and monotonic roll-off above the cutoff: joint-space motion below
``cutoff_hz`` passes essentially untouched while chunk-seam discontinuities
(LWW merges, starvation-hold jumps, uncorrelated-chunk splices) are
attenuated at ``6*order`` dB/octave.

This is the *causal* (online) form — one sample in, one sample out per
control tick — so unlike ``scipy.signal.filtfilt`` it adds phase lag: group
delay in the passband is roughly ``order / (2*pi*cutoff_hz)`` seconds (e.g.
order 2 at 5 Hz -> ~64 ms ~= 2 ticks at 30 fps). Pick the cutoff as high as
still kills the artifacts; lowering it trades responsiveness for smoothness.

Implementation: cascade of RBJ-cookbook low-pass biquads (bilinear transform,
prewarped so the -3 dB point lands exactly on ``cutoff_hz``), one per analog
Butterworth pole pair with Q_k = 1/(2 sin(theta_k)), theta_k = pi(2k+1)/(2n);
odd orders add one first-order section. Sections run Direct Form II
transposed, vectorized across action dimensions. State is seeded to DC
steady-state on the first sample so the filter starts *at* the arm's current
command instead of transiting from zero.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Section:
    """One filter section (biquad or first-order), DF-II transposed."""

    b: np.ndarray  # feedforward [b0, b1, b2]
    a: np.ndarray  # feedback [a1, a2] (a0 normalized to 1)
    z: np.ndarray | None = None  # state, shape (2, dims); None until seeded

    def seed(self, x: np.ndarray) -> None:
        # DC steady state for input x (unity DC gain => y == x):
        #   z1 = (1 - b0) x,  z2 = (b2 - a2) x
        b0, _, b2 = self.b
        _, a2 = self.a
        self.z = np.stack([(1.0 - b0) * x, (b2 - a2) * x])

    def step(self, x: np.ndarray) -> np.ndarray:
        b0, b1, b2 = self.b
        a1, a2 = self.a
        z1, z2 = self.z
        y = b0 * x + z1
        self.z[0] = b1 * x - a1 * y + z2
        self.z[1] = b2 * x - a2 * y
        return y


def _biquad_lowpass(cutoff_hz: float, fs: float, q: float) -> _Section:
    """RBJ cookbook low-pass biquad, -3 dB exactly at cutoff_hz."""
    w0 = 2.0 * np.pi * cutoff_hz / fs
    cw, alpha = np.cos(w0), np.sin(w0) / (2.0 * q)
    a0 = 1.0 + alpha
    b = np.array([(1.0 - cw) / 2.0, 1.0 - cw, (1.0 - cw) / 2.0]) / a0
    a = np.array([-2.0 * cw, 1.0 - alpha]) / a0
    return _Section(b=b, a=a)


def _first_order_lowpass(cutoff_hz: float, fs: float) -> _Section:
    """Bilinear-transform first-order low-pass (for odd total orders)."""
    k = np.tan(np.pi * cutoff_hz / fs)
    b0 = k / (1.0 + k)
    return _Section(
        b=np.array([b0, b0, 0.0]),
        a=np.array([(k - 1.0) / (k + 1.0), 0.0]),
    )


@dataclass
class ButterworthLowpass:
    """Online n-th order Butterworth low-pass over a vector signal.

    Call :meth:`filter` once per control tick with the raw command vector;
    it returns the smoothed vector. The first call is passed through
    unchanged (state seeds there). ``reset()`` re-seeds on the next sample —
    use it after a gap in the tick stream (filter state assumes uniform fs).
    """

    cutoff_hz: float
    fs: float
    order: int = 2
    _sections: list[_Section] = field(init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.cutoff_hz < self.fs / 2.0:
            raise ValueError(f"cutoff_hz must be in (0, fs/2)={self.fs / 2.0:g}, got {self.cutoff_hz}")
        if self.order < 1:
            raise ValueError(f"order must be >= 1, got {self.order}")
        n = self.order
        # Butterworth pole-pair Qs: theta_k = pi(2k+1)/(2n), Q = 1/(2 sin theta).
        self._sections = [
            _biquad_lowpass(self.cutoff_hz, self.fs, 1.0 / (2.0 * np.sin(np.pi * (2 * k + 1) / (2 * n))))
            for k in range(n // 2)
        ]
        if n % 2:
            self._sections.append(_first_order_lowpass(self.cutoff_hz, self.fs))

    def filter(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64)
        if self._sections[0].z is None:
            for s in self._sections:
                s.seed(y)
            return y.copy()
        for s in self._sections:
            y = s.step(y)
        return y

    def reset(self) -> None:
        for s in self._sections:
            s.z = None
