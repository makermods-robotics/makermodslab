"""Pure NTP-style monotonic clock negotiation for remote teleoperation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

DEFAULT_SAMPLE_COUNT = 16
DEFAULT_UNCERTAINTY_CEILING_NS = 50_000_000
DEFAULT_SCHEDULER_MARGIN_NS = 1_000_000


class ClockSyncError(ValueError):
    """Clock evidence is malformed or too uncertain for robot motion."""


def _monotonic_ns(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ClockSyncError(f"{name} must be a non-negative monotonic timestamp")
    return value


@dataclass(frozen=True)
class ClockSample:
    """One four-timestamp exchange, expressed in the two hosts' monotonic epochs."""

    operator_send_ns: int
    robot_receive_ns: int
    robot_send_ns: int
    operator_receive_ns: int

    def __post_init__(self) -> None:
        t0 = _monotonic_ns(self.operator_send_ns, "operator_send_ns")
        t1 = _monotonic_ns(self.robot_receive_ns, "robot_receive_ns")
        t2 = _monotonic_ns(self.robot_send_ns, "robot_send_ns")
        t3 = _monotonic_ns(self.operator_receive_ns, "operator_receive_ns")
        if t3 < t0:
            raise ClockSyncError("operator clock moved backwards during the probe")
        if t2 < t1:
            raise ClockSyncError("robot clock moved backwards during the probe")
        if self.round_trip_ns < 0:
            raise ClockSyncError("clock probe has a negative network round trip")

    @property
    def offset_ns(self) -> int:
        """Best integer estimate of robot monotonic minus operator monotonic."""
        numerator = (self.robot_receive_ns - self.operator_send_ns) + (
            self.robot_send_ns - self.operator_receive_ns
        )
        # Floor division is deterministic for the possible half-nanosecond case.
        return numerator // 2

    @property
    def round_trip_ns(self) -> int:
        return (self.operator_receive_ns - self.operator_send_ns) - (
            self.robot_send_ns - self.robot_receive_ns
        )

    def uncertainty_ns(self, scheduler_margin_ns: int = DEFAULT_SCHEDULER_MARGIN_NS) -> int:
        if scheduler_margin_ns < 0:
            raise ClockSyncError("scheduler margin must be non-negative")
        return (self.round_trip_ns + 1) // 2 + scheduler_margin_ns

    def to_dict(self) -> dict[str, int]:
        return {
            "operator_send_ns": self.operator_send_ns,
            "robot_receive_ns": self.robot_receive_ns,
            "robot_send_ns": self.robot_send_ns,
            "operator_receive_ns": self.operator_receive_ns,
        }

    @classmethod
    def from_dict(cls, value: object) -> ClockSample:
        expected = {
            "operator_send_ns",
            "robot_receive_ns",
            "robot_send_ns",
            "operator_receive_ns",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ClockSyncError("clock sample has missing or extra fields")
        return cls(**value)


@dataclass(frozen=True)
class FrozenClockMapping:
    """An immutable clock mapping selected before a robot session is opened."""

    robot_minus_operator_ns: int
    uncertainty_ns: int
    selected_round_trip_ns: int
    sample_count: int

    def operator_to_robot_ns(self, operator_ns: int) -> int:
        return _monotonic_ns(operator_ns, "operator_ns") + self.robot_minus_operator_ns

    def robot_interval_for_operator_ns(self, operator_ns: int) -> tuple[int, int]:
        estimate = self.operator_to_robot_ns(operator_ns)
        return estimate - self.uncertainty_ns, estimate + self.uncertainty_ns

    def public(self) -> dict[str, int]:
        return {
            "robot_minus_operator_ns": self.robot_minus_operator_ns,
            "uncertainty_ns": self.uncertainty_ns,
            "selected_round_trip_ns": self.selected_round_trip_ns,
            "sample_count": self.sample_count,
        }


def select_clock_mapping(
    samples: Iterable[ClockSample],
    *,
    required_samples: int = DEFAULT_SAMPLE_COUNT,
    scheduler_margin_ns: int = DEFAULT_SCHEDULER_MARGIN_NS,
    uncertainty_ceiling_ns: int = DEFAULT_UNCERTAINTY_CEILING_NS,
) -> FrozenClockMapping:
    """Select the lowest-RTT valid sample without averaging away jitter."""
    if required_samples < 1:
        raise ClockSyncError("required sample count must be positive")
    if not 0 <= scheduler_margin_ns <= uncertainty_ceiling_ns:
        raise ClockSyncError("clock scheduler margin exceeds the uncertainty ceiling")
    collected = tuple(samples)
    if len(collected) < required_samples:
        raise ClockSyncError(f"clock synchronization requires {required_samples} valid samples")
    selected = min(collected, key=lambda sample: sample.round_trip_ns)
    uncertainty = selected.uncertainty_ns(scheduler_margin_ns)
    if uncertainty > uncertainty_ceiling_ns:
        raise ClockSyncError("clock uncertainty exceeds the session-open ceiling")
    return FrozenClockMapping(
        robot_minus_operator_ns=selected.offset_ns,
        uncertainty_ns=uncertainty,
        selected_round_trip_ns=selected.round_trip_ns,
        sample_count=len(collected),
    )


class ClockDriftMonitor:
    """Validate fresh probes while keeping the active session mapping frozen."""

    def __init__(
        self,
        mapping: FrozenClockMapping,
        *,
        uncertainty_ceiling_ns: int = DEFAULT_UNCERTAINTY_CEILING_NS,
        drift_tolerance_ns: int = 2_000_000,
    ) -> None:
        if uncertainty_ceiling_ns < mapping.uncertainty_ns or drift_tolerance_ns < 0:
            raise ClockSyncError("invalid active-session clock budget")
        self.mapping = mapping
        self.uncertainty_ceiling_ns = uncertainty_ceiling_ns
        self.drift_tolerance_ns = drift_tolerance_ns

    def validate(self, samples: Iterable[ClockSample]) -> FrozenClockMapping:
        candidate = select_clock_mapping(
            samples,
            uncertainty_ceiling_ns=self.uncertainty_ceiling_ns,
        )
        allowed_delta = self.mapping.uncertainty_ns + candidate.uncertainty_ns + self.drift_tolerance_ns
        if abs(candidate.robot_minus_operator_ns - self.mapping.robot_minus_operator_ns) > allowed_delta:
            raise ClockSyncError("clock drift exceeds the active-session budget")
        # Return the original mapping deliberately. Active sessions never remap in place.
        return self.mapping
