"""JSON-line subprocess peers for the hardware-free remote teleoperation trial.

This helper is intentionally test-only. Starting it is the explicit act that
opens loopback listeners; importing it opens no socket and no device.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import ssl
import sys
import threading
import time
from pathlib import Path

from makermodslab.remote_teleop.clock_sync import ClockSample
from makermodslab.remote_teleop.contracts import ActionSample, SessionSpec, encode_action
from makermodslab.remote_teleop.control_client import PinnedControlClient
from makermodslab.remote_teleop.control_server import (
    RobotControlCallbacks,
    RobotControlProtocol,
    RobotSessionProfile,
    SessionOpenResult,
    TlsControlServer,
)
from makermodslab.remote_teleop.executor import JointLimit, RemoteExecutor
from makermodslab.remote_teleop.pairing import (
    OperatorCredentialVault,
    PairingAuthority,
    RobotCredentialStore,
    certificate_sha256_fingerprint,
)
from makermodslab.remote_teleop.simulation import SimulatedFollower
from makermodslab.remote_teleop.transport import UdpActionReceiver, UdpFaultProxy, UdpSessionSender
from makermodslab.remote_teleop.watchdog import (
    RobotLivenessWatchdog,
    WatchdogDeadlines,
    WatchdogRunner,
)

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64


def _spec() -> SessionSpec:
    return SessionSpec(
        source_id="process-operator",
        rig_id="process-so101-pair",
        rig_digest=_DIGEST_A,
        leader_calibration_id="process-leader-calibration",
        leader_calibration_digest=_DIGEST_B,
        follower_calibration_id="process-follower-calibration",
        follower_calibration_digest=_DIGEST_C,
        joint_names=("shoulder_pan", "shoulder_lift"),
        units=("degree", "degree"),
    )


def _write(message: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(message, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


async def _read() -> dict[str, object] | None:
    raw = await asyncio.to_thread(sys.stdin.readline)
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get("command"), str):
        raise ValueError("peer command must be a JSON object with a command")
    return value


class RobotPeer:
    def __init__(self, state_root: Path, certificate: Path, private_key: Path) -> None:
        self.state_root = state_root
        self.certificate = certificate
        self.private_key = private_key
        self.spec = _spec()
        self.credentials = RobotCredentialStore(state_root / "robot-credentials")
        self.pairing = PairingAuthority(self.credentials)
        self.follower = SimulatedFollower(self.spec.joint_names)
        limits = {joint: JointLimit(-90.0, 90.0, 120.0, 600.0) for joint in self.spec.joint_names}
        # The independent liveness watchdog owns the process-trial deadlines.
        # Keep the executor watchdog longer so every trip uses the same safe-stop path.
        self.executor = RemoteExecutor(
            self.follower,
            limits,
            tick_hz=50,
            watchdog_ns=2_000_000_000,
            first_action_deadline_ns=2_000_000_000,
            mode="simulation",
        )
        self.receiver = UdpActionReceiver("127.0.0.1", 0, self._action, allow_loopback=True)
        self.watchdog = RobotLivenessWatchdog(
            self._safe_stop,
            deadlines=WatchdogDeadlines(
                action_ns=350_000_000,
                first_action_ns=1_500_000_000,
                control_ns=1_500_000_000,
                browser_ns=2_000_000_000,
            ),
        )
        self.watchdog_runner = WatchdogRunner(self.watchdog, poll_interval_s=0.01)
        self.control: TlsControlServer | None = None
        self._tick_stop = threading.Event()
        self._tick_thread: threading.Thread | None = None
        self._stop_lock = threading.RLock()
        self.last_receipt: dict[str, object] | None = None

    def _action(self, raw: bytes) -> None:
        self.executor.submit_datagram(raw)
        self.watchdog.mark_action()

    def _safe_stop(self, reason: str) -> dict[str, object]:
        with self._stop_lock:
            self.receiver.stop_dispatch()
            self.watchdog.disarm()
            receipt = self.executor.stop(reason)
            self.last_receipt = receipt
            return receipt

    def _open_session(self, spec, mapping, _credential_id) -> SessionOpenResult:
        grant = self.executor.open_session(
            spec,
            clock_offset_ns=mapping.robot_minus_operator_ns,
            clock_uncertainty_ns=mapping.uncertainty_ns,
        )
        self.receiver.begin_session(
            session_id=grant.session_id,
            executor_generation=grant.executor_generation,
            key_id=grant.key_id,
            key=grant.action_key,
        )
        self.watchdog.arm()
        return SessionOpenResult(grant, "127.0.0.1", self.receiver.bound_port or 0)

    def _stop_session(self, session_id: str, generation: int, reason: str) -> dict[str, object]:
        grant = self.executor.authority.grant
        if grant is None or (grant.session_id, grant.executor_generation) != (session_id, generation):
            raise RuntimeError("stale process-trial STOP")
        return self._safe_stop(reason)

    def _heartbeat(
        self, _session_id: str, _generation: int, process_live: bool, browser_live: bool
    ) -> dict[str, object]:
        self.watchdog.mark_control(
            operator_process_live=process_live,
            browser_live=browser_live,
        )
        status = self.watchdog.status()
        remaining = status.get("control_remaining_ms")
        return {"watchdog_remaining_ms": remaining if isinstance(remaining, float) else 0}

    def _tick(self) -> None:
        interval = 1.0 / self.executor.tick_hz
        while not self._tick_stop.wait(interval):
            with contextlib.suppress(Exception):
                self.executor.tick()

    def status(self) -> dict[str, object]:
        return {
            "executor": self.executor.status(),
            "watchdog": self.watchdog.status(),
            "udp": self.receiver.status(),
            "follower": {
                "connected": self.follower.connected,
                "positions": dict(self.follower.positions),
                "stop_reasons": list(self.follower.stop_reasons),
            },
            "last_receipt": self.last_receipt,
        }

    async def start(self) -> dict[str, object]:
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(self.certificate, self.private_key)
        callbacks = RobotControlCallbacks(
            session_profile=lambda _credential_id: RobotSessionProfile(self.spec, "d" * 64),
            open_session=self._open_session,
            stop_session=self._stop_session,
            session_status=lambda _session, _generation: self.executor.status(),
            heartbeat=self._heartbeat,
            udp_probe_status=lambda _session, _generation: bool(self.receiver.status()["endpoint_bound"]),
            control_lost=lambda _session, _generation, reason: self._safe_stop(reason),
        )
        self.receiver.start()
        self.watchdog_runner.start()
        self._tick_stop.clear()
        self._tick_thread = threading.Thread(
            target=self._tick,
            name="remote-process-trial-executor",
            daemon=True,
        )
        self._tick_thread.start()
        self.control = TlsControlServer(
            "127.0.0.1",
            0,
            tls,
            lambda: RobotControlProtocol(
                robot_id="process-robot",
                credentials=self.credentials,
                pairing=self.pairing,
                callbacks=callbacks,
            ),
            allow_loopback=True,
            heartbeat_deadline_s=1.5,
        )
        port = await self.control.start()
        pairing = self.pairing.open_local_window(
            local_request=True,
            robot_address="127.0.0.1",
            control_port=port,
            certificate_fingerprint=certificate_sha256_fingerprint(self.certificate),
        )
        return {"control_port": port, "pairing_token": pairing.pairing_token}

    async def close(self) -> None:
        self._safe_stop("process_trial_shutdown")
        self._tick_stop.set()
        if self._tick_thread is not None:
            self._tick_thread.join(timeout=1.0)
        self.watchdog_runner.close()
        self.receiver.close()
        if self.control is not None:
            await self.control.close()


class OperatorPeer:
    def __init__(
        self,
        state_root: Path,
        control_uri: str,
        fingerprint: str,
        *,
        drop: set[int],
        delay: set[int],
        duplicate: set[int],
    ) -> None:
        self.state_root = state_root
        self.vault = OperatorCredentialVault(state_root / "operator-credentials")
        self.control_uri = control_uri
        self.fingerprint = fingerprint
        self.client = PinnedControlClient(control_uri, fingerprint)
        self.spec = _spec()
        self.grant = None
        self.sender: UdpSessionSender | None = None
        self.proxy: UdpFaultProxy | None = None
        self.drop = drop
        self.delay = delay
        self.duplicate = duplicate
        self.sequence = 0
        self._heartbeat_enabled = True
        self._action_loop_enabled = False
        self._browser_live = True
        self._background: asyncio.Task[None] | None = None

    async def connect(self, pairing_token: str | None) -> dict[str, object]:
        await self.client.connect()
        credential = self.vault.get("process-robot")
        hello = await self.client.hello("process-operator", credential)
        if not hello["authenticated"]:
            if not pairing_token:
                raise RuntimeError("pairing token required for first process-trial connection")
            credential = await self.client.pair(pairing_token, "process operator")
            self.vault.put("process-robot", credential)
        profile = await self.client.profile()
        _mapping, samples = await self.client.synchronize_clocks()
        self.grant = await self.client.open_session(self.spec, samples)
        self.proxy = UdpFaultProxy(
            "127.0.0.1",
            0,
            self.grant.udp_host,
            self.grant.udp_port,
            allow_loopback=True,
            drop_numbers=self.drop,
            delay_numbers=self.delay,
            duplicate_numbers=self.duplicate,
        )
        proxy_port = self.proxy.start()
        self.sender = UdpSessionSender("127.0.0.1", proxy_port)
        try:
            self.sender.prove_endpoint(
                session_id=self.grant.session_id,
                executor_generation=self.grant.executor_generation,
                key_id=self.grant.key_id,
                key=self.grant.action_key(),
            )
        except Exception as exc:
            raise RuntimeError(f"UDP proof failed; proxy={self.proxy.status()}") from exc
        if not await self.client.udp_probe_status(self.grant.key_id):
            raise RuntimeError("robot did not bind the authenticated UDP endpoint")
        self._background = asyncio.create_task(self._background_loop())
        return {
            "authenticated": True,
            "session_id": self.grant.session_id,
            "generation": self.grant.executor_generation,
            "profile": profile,
        }

    async def _background_loop(self) -> None:
        while True:
            try:
                if self._heartbeat_enabled:
                    await self.client.heartbeat(browser_live=self._browser_live)
                if self._action_loop_enabled:
                    self.send_action()
            except asyncio.CancelledError:
                raise
            except Exception:
                return
            await asyncio.sleep(0.05)

    def send_action(self, *, skew_ns: int = 0) -> int:
        if self.grant is None or self.sender is None:
            raise RuntimeError("operator process-trial session is not active")
        self.sequence += 1
        now = time.monotonic_ns() + skew_ns
        sample = ActionSample(
            session_id=self.grant.session_id,
            source_id=self.spec.source_id,
            executor_generation=self.grant.executor_generation,
            rig_id=self.spec.rig_id,
            rig_digest=self.spec.rig_digest,
            leader_calibration_id=self.spec.leader_calibration_id,
            leader_calibration_digest=self.spec.leader_calibration_digest,
            follower_calibration_id=self.spec.follower_calibration_id,
            follower_calibration_digest=self.spec.follower_calibration_digest,
            sequence=self.sequence,
            source_monotonic_ns=now,
            expires_at_source_monotonic_ns=now + 100_000_000,
            joint_names=self.spec.joint_names,
            units=self.spec.units,
            positions=(float(self.sequence), float(-self.sequence)),
        )
        self.sender.send(encode_action(sample, key_id=self.grant.key_id, key=self.grant.action_key()))
        return self.sequence

    def save_old_action(self) -> None:
        if self.grant is None:
            raise RuntimeError("operator process-trial session is not active")
        self.sequence += 1
        now = time.monotonic_ns()
        sample = ActionSample(
            session_id=self.grant.session_id,
            source_id=self.spec.source_id,
            executor_generation=self.grant.executor_generation,
            rig_id=self.spec.rig_id,
            rig_digest=self.spec.rig_digest,
            leader_calibration_id=self.spec.leader_calibration_id,
            leader_calibration_digest=self.spec.leader_calibration_digest,
            follower_calibration_id=self.spec.follower_calibration_id,
            follower_calibration_digest=self.spec.follower_calibration_digest,
            sequence=self.sequence,
            source_monotonic_ns=now,
            expires_at_source_monotonic_ns=now + 100_000_000,
            joint_names=self.spec.joint_names,
            units=self.spec.units,
            positions=(1.0, -1.0),
        )
        path = self.state_root / "old-action.bin"
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(encode_action(sample, key_id=self.grant.key_id, key=self.grant.action_key()))
        path.chmod(0o600)

    def replay_old_action(self) -> None:
        if self.sender is None:
            raise RuntimeError("operator process-trial session is not active")
        self.sender.send((self.state_root / "old-action.bin").read_bytes())

    def status(self) -> dict[str, object]:
        return {
            "proxy": self.proxy.status() if self.proxy is not None else None,
            "sequence": self.sequence,
            "heartbeat_enabled": self._heartbeat_enabled,
            "action_loop_enabled": self._action_loop_enabled,
        }

    async def duplicate_session(self) -> str:
        credential = self.vault.get("process-robot")
        assert credential is not None
        duplicate = PinnedControlClient(self.control_uri, self.fingerprint)
        try:
            await duplicate.connect()
            hello = await duplicate.hello("process-operator-duplicate", credential)
            assert hello["authenticated"] is True
            await duplicate.profile()
            _mapping, samples = await duplicate.synchronize_clocks()
            await duplicate.open_session(self.spec, samples)
        except Exception as exc:
            return type(exc).__name__
        finally:
            await duplicate.close()
        raise RuntimeError("duplicate process-trial session was accepted")

    async def clock_drift(self) -> bool:
        if self.grant is None:
            raise RuntimeError("operator process-trial session is not active")
        _, samples = await self.client.synchronize_clocks()
        shifted = [
            ClockSample(
                sample.operator_send_ns,
                sample.robot_receive_ns + 100_000_000,
                sample.robot_send_ns + 100_000_000,
                sample.operator_receive_ns,
            )
            for sample in samples
        ]
        return await self.client.check_clock(shifted)

    async def close(self) -> None:
        if self._background is not None:
            self._background.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._background
        await self.client.close()
        if self.sender is not None:
            self.sender.close()
        if self.proxy is not None:
            self.proxy.close()


def _numbers(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item}


async def _robot_main(args: argparse.Namespace) -> None:
    peer = RobotPeer(Path(args.state_root), Path(args.certificate), Path(args.private_key))
    try:
        _write({"event": "ready", **await peer.start()})
        while command := await _read():
            kind = command["command"]
            if kind == "status":
                _write({"event": "status", "status": peer.status()})
            elif kind == "shutdown":
                _write({"event": "shutdown"})
                return
            else:
                raise ValueError(f"unsupported robot peer command: {kind}")
    finally:
        await peer.close()


async def _operator_main(args: argparse.Namespace) -> None:
    peer = OperatorPeer(
        Path(args.state_root),
        args.control_uri,
        args.fingerprint,
        drop=_numbers(args.drop),
        delay=_numbers(args.delay),
        duplicate=_numbers(args.duplicate),
    )
    _write({"event": "ready"})
    try:
        while command := await _read():
            kind = command["command"]
            if kind == "connect":
                _write({"event": "connected", **await peer.connect(command.get("pairing_token"))})
            elif kind == "send_action":
                sequence = peer.send_action(skew_ns=int(command.get("skew_ns", 0)))
                _write({"event": "action_sent", "sequence": sequence})
            elif kind == "save_old_action":
                peer.save_old_action()
                _write({"event": "old_action_saved"})
            elif kind == "replay_old_action":
                peer.replay_old_action()
                _write({"event": "old_action_replayed"})
            elif kind == "start_actions":
                peer._action_loop_enabled = True
                _write({"event": "actions_started"})
            elif kind == "pause_actions":
                peer._action_loop_enabled = False
                _write({"event": "actions_paused"})
            elif kind == "pause_control":
                peer._heartbeat_enabled = False
                _write({"event": "control_paused"})
            elif kind == "browser_loss":
                await peer.client.heartbeat(browser_live=False)
                _write({"event": "browser_loss_sent"})
            elif kind == "process_loss":
                await peer.client.heartbeat(browser_live=True)
                session_id, generation = peer.client._require_session()
                from makermodslab.remote_teleop.control_protocol import ControlMessage

                response = await peer.client.request(
                    ControlMessage(
                        "heartbeat",
                        peer.client._request_id(),
                        {
                            "operator_monotonic_ns": time.monotonic_ns(),
                            "operator_process_live": False,
                            "browser_live": True,
                        },
                        session_id,
                        generation,
                    )
                )
                _write({"event": "process_loss_sent", "response": response.message_type})
            elif kind == "network_down":
                assert peer.proxy is not None
                peer.proxy.set_forwarding(False)
                _write({"event": "network_down"})
            elif kind == "flush_reordered":
                assert peer.proxy is not None
                peer.proxy.flush_delayed()
                _write({"event": "reordered_flushed"})
            elif kind == "duplicate_session":
                _write({"event": "duplicate_refused", "error": await peer.duplicate_session()})
            elif kind == "status":
                _write({"event": "status", "status": peer.status()})
            elif kind == "clock_drift":
                _write({"event": "clock_checked", "valid": await peer.clock_drift()})
            elif kind == "stop":
                receipt = await peer.client.stop("process_trial_operator_stop")
                _write({"event": "stop_ack", "receipt": receipt})
            elif kind == "shutdown":
                _write({"event": "shutdown"})
                return
            else:
                raise ValueError(f"unsupported operator peer command: {kind}")
    finally:
        await peer.close()


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
    operator.add_argument("--drop", default="")
    operator.add_argument("--delay", default="")
    operator.add_argument("--duplicate", default="")
    args = parser.parse_args()
    asyncio.run(_robot_main(args) if args.role == "robot" else _operator_main(args))


if __name__ == "__main__":
    main()
