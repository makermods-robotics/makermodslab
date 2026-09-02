"""JSON-line peers that exercise the production remote runtime services.

This module is test-only. Importing it opens no listener or device; its CLI
entrypoint is the explicit test action that enables a loopback runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from makermodslab.hardware_lease import HardwareLeaseRegistry
from makermodslab.remote_teleop.adapters.common import RawLeaderSample
from makermodslab.remote_teleop.calibration_identity import CalibrationIdentity
from makermodslab.remote_teleop.commissioning import CommissioningRecord, CommissioningStore
from makermodslab.remote_teleop.config import RemoteRoleConfigStore
from makermodslab.remote_teleop.executor import JointLimit
from makermodslab.remote_teleop.operator_service import RemoteOperatorService
from makermodslab.remote_teleop.robot_service import PreparedRobotProfile, RemoteRobotService
from makermodslab.remote_teleop.simulation import SimulatedFollower

JOINTS = ("shoulder_pan", "shoulder_lift")
UNITS = ("degree", "degree")


@dataclass(frozen=True)
class _LoopbackRobotConfig:
    """Test-only dataclass compatible with the service's ephemeral-port replace()."""

    node_id: str
    robot_name: str
    bind_address: str
    control_port: int
    udp_port: int
    tls_certificate_path: str
    tls_private_key_path: str
    leader_calibration_id: str
    leader_calibration_digest: str
    action_rate_hz: int
    action_watchdog_ms: int
    first_action_deadline_ms: int
    control_deadline_ms: int
    browser_deadline_ms: int
    max_velocity_per_s: float
    max_acceleration_per_s2: float
    recording_enabled: bool


def _profile() -> PreparedRobotProfile:
    return PreparedRobotProfile(
        follower_config=object(),
        follower_calibration=CalibrationIdentity("process-follower-calibration", "c" * 64),
        leader_calibration=CalibrationIdentity("process-leader-calibration", "b" * 64),
        rig_id="process-so101-pair",
        rig_digest="a" * 64,
        joint_names=JOINTS,
        units=UNITS,
        limits={joint: JointLimit(-90.0, 90.0, 120.0, 600.0) for joint in JOINTS},
        limits_digest="d" * 64,
        device_identity_digest="e" * 64,
        follower_port="/dev/test-runtime-follower",
    )


def _write(message: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(message, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _read() -> dict[str, object] | None:
    raw = sys.stdin.readline()
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get("command"), str):
        raise ValueError("peer command must be a JSON object with a command")
    return value


def _registry_status(registry: HardwareLeaseRegistry) -> dict[str, object]:
    snapshot = registry.snapshot()
    return {
        "held": snapshot.held,
        "state": snapshot.state,
        "kind": snapshot.kind,
        "owner": snapshot.owner,
        "generation": snapshot.generation,
    }


class RuntimeFollower(SimulatedFollower):
    """Safe deterministic follower with the same status shape as the process adapter."""

    @property
    def child_status(self) -> dict[str, object]:
        return {
            "device": {"digest": "e" * 64},
            "stop_receipt": {
                "hardware_stop_completed": True,
                "torque_off_confirmed": True,
                "fault": None,
            },
        }


class RuntimeLeader:
    joint_schema = SimpleNamespace(action_keys=JOINTS, units=UNITS)

    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.sequence = 0

    def connect(self) -> None:
        if self.connected:
            raise RuntimeError("runtime test leader is already connected")
        self.connected = True

    def read(self) -> RawLeaderSample:
        if not self.connected or self.closed:
            raise RuntimeError("runtime test leader is closed")
        self.sequence += 1
        position = min(15.0, float(self.sequence))
        return RawLeaderSample(
            {"shoulder_pan": position, "shoulder_lift": -position},
            time.monotonic_ns(),
        )

    def close(self) -> None:
        self.closed = True
        self.connected = False


class RobotRuntimePeer:
    def __init__(self, state_root: Path, certificate: Path, private_key: Path) -> None:
        self.profile = _profile()
        self.follower = RuntimeFollower(JOINTS)
        self.registry = HardwareLeaseRegistry()
        config_store = RemoteRoleConfigStore(state_root)
        CommissioningStore(state_root).save(CommissioningRecord.from_profile(self.profile))
        self.service = RemoteRobotService(
            config_store=config_store,
            registry=self.registry,
            profile_builder=lambda _config: self.profile,
            follower_factory=lambda *_args: self.follower,
            serial_binding_resolver=lambda ports: dict.fromkeys(
                ports, "test-runtime-physical-adapter"
            ),
            allow_loopback=True,
        )
        # Production config refuses loopback and ephemeral ports. The process
        # test deliberately injects those two test-only values after config
        # validation so the production refusal remains covered and unchanged.
        config = _LoopbackRobotConfig(
            node_id="process-robot",
            robot_name="test-only-so101-follower",
            bind_address="127.0.0.1",
            control_port=0,
            udp_port=0,
            tls_certificate_path=str(certificate),
            tls_private_key_path=str(private_key),
            leader_calibration_id=self.profile.leader_calibration.calibration_id,
            leader_calibration_digest=self.profile.leader_calibration.digest,
            action_rate_hz=50,
            action_watchdog_ms=350,
            first_action_deadline_ms=1500,
            control_deadline_ms=1000,
            browser_deadline_ms=2000,
            max_velocity_per_s=120.0,
            max_acceleration_per_s2=600.0,
            recording_enabled=False,
        )
        self.service._load_robot_config = lambda: config

    def start(self) -> dict[str, object]:
        status = self.service.enable()
        pairing = self.service.open_pairing_window()
        listener = status["listener"]
        assert isinstance(listener, dict)
        return {
            "control_port": listener["control_port"],
            "pairing_token": pairing["pairing_token"],
            "certificate_fingerprint": pairing["certificate_fingerprint"],
        }

    def status(self) -> dict[str, object]:
        return {
            "service": self.service.status(),
            "registry": _registry_status(self.registry),
            "follower": {
                "connected": self.follower.connected,
                "positions": dict(self.follower.positions),
                "stop_reasons": list(self.follower.stop_reasons),
            },
        }

    def close(self) -> None:
        self.service.shutdown()


class OperatorRuntimePeer:
    def __init__(
        self,
        state_root: Path,
        control_uri: str,
        fingerprint: str,
    ) -> None:
        self.registry = HardwareLeaseRegistry()
        self.leader = RuntimeLeader()
        self.service = RemoteOperatorService(
            config_store=RemoteRoleConfigStore(state_root),
            registry=self.registry,
            leader_factory=lambda *_args: self.leader,
            leader_port_resolver=lambda _config: "/dev/test-runtime-leader",
        )
        config = SimpleNamespace(
            node_id="process-operator",
            robot_id="process-robot",
            leader_robot_name="test-only-so101-leader",
            control_uri=control_uri,
            certificate_fingerprint=fingerprint,
            action_rate_hz=50,
        )
        self.service._load_config = lambda: config

    def start(self, pairing_token: str) -> dict[str, object]:
        paired = self.service.pair(pairing_token, "runtime process test")
        status = self.service.start()
        return {"paired": paired, "status": status}

    def status(self) -> dict[str, object]:
        return {
            "service": self.service.status(),
            "registry": _registry_status(self.registry),
            "leader": {
                "connected": self.leader.connected,
                "closed": self.leader.closed,
                "reads": self.leader.sequence,
            },
        }

    def close(self) -> None:
        self.service.shutdown()


def _robot_main(args: argparse.Namespace) -> None:
    peer = RobotRuntimePeer(
        Path(args.state_root),
        Path(args.certificate),
        Path(args.private_key),
    )
    try:
        _write({"event": "ready", **peer.start()})
        while command := _read():
            if command["command"] == "status":
                _write({"event": "status", "status": peer.status()})
            elif command["command"] == "shutdown":
                _write({"event": "shutdown"})
                return
            else:
                raise ValueError(f"unsupported robot runtime command: {command['command']}")
    finally:
        peer.close()


def _operator_main(args: argparse.Namespace) -> None:
    peer = OperatorRuntimePeer(Path(args.state_root), args.control_uri, args.fingerprint)
    try:
        started = peer.start(args.pairing_token)
        _write({"event": "ready", **started})
        while command := _read():
            if command["command"] == "status":
                _write({"event": "status", "status": peer.status()})
            elif command["command"] == "browser_heartbeat":
                _write({"event": "browser_heartbeat", **peer.service.browser_heartbeat()})
            elif command["command"] == "shutdown":
                _write({"event": "shutdown"})
                return
            else:
                raise ValueError(f"unsupported operator runtime command: {command['command']}")
    finally:
        peer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="role", required=True)
    robot = subparsers.add_parser("robot")
    robot.add_argument("--state-root", required=True)
    robot.add_argument("--certificate", required=True)
    robot.add_argument("--private-key", required=True)
    operator = subparsers.add_parser("operator")
    operator.add_argument("--state-root", required=True)
    operator.add_argument("--control-uri", required=True)
    operator.add_argument("--fingerprint", required=True)
    operator.add_argument("--pairing-token", required=True)
    args = parser.parse_args()
    if args.role == "robot":
        _robot_main(args)
    else:
        _operator_main(args)


if __name__ == "__main__":
    main()
