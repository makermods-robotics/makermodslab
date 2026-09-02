"""Deterministic follower and recording doubles; never opens hardware."""

from __future__ import annotations

from collections.abc import Mapping


class SimulatedFollower:
    def __init__(self, joint_names: tuple[str, ...], *, initial: float = 0.0) -> None:
        self.joint_names = joint_names
        self.positions = {joint: float(initial) for joint in joint_names}
        self.connected = False
        self.stop_reasons: list[str] = []
        self.fail_next_execute = False

    def connect(self) -> None:
        if self.connected:
            raise RuntimeError("simulated follower already connected")
        self.connected = True

    def observe(self) -> Mapping[str, float]:
        if not self.connected:
            raise RuntimeError("simulated follower is disconnected")
        return dict(self.positions)

    def execute(self, positions: Mapping[str, float]) -> Mapping[str, float]:
        if not self.connected:
            raise RuntimeError("simulated follower is disconnected")
        if self.fail_next_execute:
            self.fail_next_execute = False
            raise OSError("injected follower write failure")
        if set(positions) != set(self.joint_names):
            raise ValueError("simulated action changed joint membership")
        self.positions = {joint: float(positions[joint]) for joint in self.joint_names}
        return dict(self.positions)

    def stop(self, reason: str) -> Mapping[str, object]:
        self.stop_reasons.append(reason)
        return {
            "disable_requested": True,
            "hardware_stop_completed": True,
            "torque_off_confirmed": True,
            "verification": "simulation",
        }

    def close(self) -> Mapping[str, object]:
        self.connected = False
        return {
            "close_completed": True,
        }


class InMemoryRecorder:
    def __init__(self, *, fail: bool = False) -> None:
        self.events: list[dict[str, object]] = []
        self.fail = fail

    def __call__(self, event: Mapping[str, object]) -> bool:
        if self.fail:
            return False
        self.events.append(dict(event))
        return True
