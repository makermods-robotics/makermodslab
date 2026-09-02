"""Killable process boundary for the live SO-101 follower adapter.

The parent side implements ``FollowerDriver`` for ``RemoteExecutor``.  The
child is the only process that constructs or owns ``SO101FollowerDriver`` and
therefore the only process that opens the serial device.  Killing a stuck
child is *not* evidence that Feetech torque is off; every such path latches an
unknown-torque fault.
"""

from __future__ import annotations

import math
import multiprocessing
import threading
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

from ...servo_health.service import ServoHealthService, servo_health_service
from .lerobot_follower import SO101FollowerDriver

WORKER_PROTOCOL_VERSION = "makermodslab.follower-worker.v1"
SO101_ACTION_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
_OPERATIONS = frozenset({"connect", "observe", "execute", "stop", "close"})


class FollowerWorkerError(RuntimeError):
    """The hardware worker failed or violated its bounded protocol."""


class FollowerWorkerTimeoutError(FollowerWorkerError):
    """The hardware worker exceeded a parent-enforced operation deadline."""


@dataclass(frozen=True)
class WorkerTimeouts:
    connect_s: float = 15.0
    observe_s: float = 0.25
    execute_s: float = 0.25
    stop_s: float = 0.25
    close_s: float = 0.5
    terminate_s: float = 0.1

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.01 <= value <= 60:
                raise ValueError(f"{name} must be in [0.01s,60s]")

    def for_operation(self, operation: str) -> float:
        return {
            "connect": self.connect_s,
            "observe": self.observe_s,
            "execute": self.execute_s,
            "stop": self.stop_s,
            "close": self.close_s,
        }[operation]


def _make_so101_follower_driver(
    config: object,
    expected_calibration_id: str | None,
    expected_calibration_digest: str | None,
    expected_serial_binding: str | None,
) -> SO101FollowerDriver:
    return SO101FollowerDriver(
        config,
        expected_calibration_id=expected_calibration_id,
        expected_calibration_digest=expected_calibration_digest,
        expected_serial_binding=expected_serial_binding,
    )


AdapterFactory = Callable[[object, str | None, str | None, str | None], object]


def _request(request_id: int, operation: str, payload: Mapping[str, object]) -> dict[str, object]:
    if operation not in _OPERATIONS:
        raise ValueError("unsupported follower worker operation")
    return {
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "request_id": request_id,
        "operation": operation,
        "payload": dict(payload),
    }


def _response(
    request_id: int,
    *,
    ok: bool,
    result: object = None,
    error_code: str | None = None,
) -> dict[str, object]:
    return {
        "protocol_version": WORKER_PROTOCOL_VERSION,
        "request_id": request_id,
        "ok": ok,
        "result": result,
        "error_code": error_code,
    }


def _validate_request(value: object) -> tuple[int, str, dict[str, object]]:
    if not isinstance(value, dict) or set(value) != {
        "protocol_version",
        "request_id",
        "operation",
        "payload",
    }:
        raise ValueError("invalid follower worker request shape")
    if value["protocol_version"] != WORKER_PROTOCOL_VERSION:
        raise ValueError("unsupported follower worker protocol")
    request_id = value["request_id"]
    operation = value["operation"]
    payload = value["payload"]
    if not isinstance(request_id, int) or isinstance(request_id, bool) or request_id < 1:
        raise ValueError("invalid follower worker request id")
    if not isinstance(operation, str) or operation not in _OPERATIONS:
        raise ValueError("invalid follower worker operation")
    if not isinstance(payload, dict):
        raise ValueError("invalid follower worker payload")
    return request_id, operation, payload


def _validate_response(value: object, request_id: int) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "protocol_version",
        "request_id",
        "ok",
        "result",
        "error_code",
    }:
        raise FollowerWorkerError("invalid follower worker response shape")
    if value["protocol_version"] != WORKER_PROTOCOL_VERSION or value["request_id"] != request_id:
        raise FollowerWorkerError("follower worker response correlation failed")
    if not isinstance(value["ok"], bool):
        raise FollowerWorkerError("invalid follower worker response status")
    if value["error_code"] is not None and not isinstance(value["error_code"], str):
        raise FollowerWorkerError("invalid follower worker error code")
    return value


def _worker_main(
    connection: Connection,
    config: object,
    expected_calibration_id: str | None,
    expected_calibration_digest: str | None,
    expected_serial_binding: str | None,
    adapter_factory: AdapterFactory,
) -> None:
    adapter: object | None = None
    adapter_closed = False
    try:
        while True:
            try:
                request_id, operation, payload = _validate_request(connection.recv())
            except (EOFError, OSError):
                return
            except Exception:
                # An uncorrelated malformed request cannot receive a trustworthy
                # response. Exit and let the parent report a protocol fault.
                return

            try:
                if operation == "connect":
                    if adapter is not None:
                        raise RuntimeError("adapter already constructed")
                    adapter = adapter_factory(
                        config,
                        expected_calibration_id,
                        expected_calibration_digest,
                        expected_serial_binding,
                    )
                    adapter.connect()
                    result: object = {
                        "joint_names": list(adapter.joint_names),
                        "status": dict(adapter.status),
                        "servo_health": (
                            adapter.sample_health()
                            if callable(getattr(adapter, "sample_health", None))
                            else None
                        ),
                    }
                elif adapter is None:
                    raise RuntimeError("adapter is not connected")
                elif operation == "observe":
                    result = {
                        "positions": dict(adapter.observe()),
                        "servo_health": (
                            adapter.sample_health()
                            if callable(getattr(adapter, "sample_health", None))
                            else None
                        ),
                    }
                elif operation == "execute":
                    if set(payload) != {"positions"} or not isinstance(payload["positions"], dict):
                        raise ValueError("invalid execute payload")
                    result = dict(adapter.execute(payload["positions"]))
                elif operation == "stop":
                    if set(payload) != {"reason"} or not isinstance(payload["reason"], str):
                        raise ValueError("invalid stop payload")
                    result = dict(adapter.stop(payload["reason"]))
                elif operation == "close":
                    if payload:
                        raise ValueError("invalid close payload")
                    result = dict(adapter.close())
                    adapter_closed = True
                else:  # pragma: no cover - _validate_request owns this invariant
                    raise ValueError("unsupported operation")
            except Exception as exc:
                with suppress(BrokenPipeError, EOFError, OSError):
                    connection.send(
                        _response(
                            request_id,
                            ok=False,
                            error_code=f"{operation}_{type(exc).__name__}",
                        )
                    )
                if operation in {"connect", "close"}:
                    return
                continue

            try:
                connection.send(_response(request_id, ok=True, result=result))
            except (BrokenPipeError, EOFError, OSError):
                return
            if operation == "close":
                return
    finally:
        if adapter is not None and not adapter_closed:
            with suppress(Exception):
                adapter.stop("worker_channel_closed")
            with suppress(Exception):
                adapter.close()
        connection.close()


class SO101FollowerProcessDriver:
    """Parent-side, timeout-bounded ``FollowerDriver`` implementation."""

    def __init__(
        self,
        config: object,
        *,
        joint_names: tuple[str, ...] = SO101_ACTION_KEYS,
        expected_calibration_id: str | None = None,
        expected_calibration_digest: str | None = None,
        expected_serial_binding: str | None = None,
        timeouts: WorkerTimeouts | None = None,
        process_context: Any | None = None,
        adapter_factory: AdapterFactory = _make_so101_follower_driver,
        health_service: ServoHealthService = servo_health_service,
    ) -> None:
        if tuple(joint_names) != SO101_ACTION_KEYS:
            raise ValueError("live follower worker supports the exact single-arm SO-101 action schema")
        if (expected_calibration_id is None) != (expected_calibration_digest is None):
            raise ValueError("expected follower calibration id and digest must be provided together")
        self.config = config
        self.joint_names = tuple(joint_names)
        self.expected_calibration_id = expected_calibration_id
        self.expected_calibration_digest = expected_calibration_digest
        self.expected_serial_binding = expected_serial_binding
        self.timeouts = timeouts or WorkerTimeouts()
        self._context = process_context or multiprocessing.get_context("spawn")
        self._adapter_factory = adapter_factory
        self._health_service = health_service
        self._health_key = "remote_teleoperation:follower"
        self._lock = threading.RLock()
        self._connection: Connection | None = None
        self._process: Any | None = None
        self._next_request_id = 0
        self._connected = False
        self._fault_lockout = False
        self._terminated_without_close = False
        self._child_status: dict[str, object] | None = None
        self.last_stop_receipt: dict[str, object] | None = None
        self.last_close_receipt: dict[str, object] | None = None

    @property
    def worker_pid(self) -> int | None:
        with self._lock:
            return None if self._process is None else self._process.pid

    @property
    def fault_lockout(self) -> bool:
        with self._lock:
            return self._fault_lockout

    @property
    def child_status(self) -> dict[str, object] | None:
        with self._lock:
            return None if self._child_status is None else dict(self._child_status)

    @property
    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "worker_running": self._process is not None and self._process.is_alive(),
                "connected": self._connected,
                "fault_lockout": self._fault_lockout,
                "child": None if self._child_status is None else dict(self._child_status),
                "stop_receipt": (None if self.last_stop_receipt is None else dict(self.last_stop_receipt)),
                "close_receipt": (None if self.last_close_receipt is None else dict(self.last_close_receipt)),
            }

    def connect(self) -> None:
        with self._lock:
            if self._process is not None:
                raise RuntimeError("follower worker already exists")
            if self._fault_lockout:
                raise RuntimeError("follower worker is in fault lockout")
            parent, child = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=_worker_main,
                args=(
                    child,
                    self.config,
                    self.expected_calibration_id,
                    self.expected_calibration_digest,
                    self.expected_serial_binding,
                    self._adapter_factory,
                ),
                name="makermodslab-so101-follower",
                daemon=True,
            )
            self._connection = parent
            self._process = process
            try:
                process.start()
            except Exception:
                parent.close()
                child.close()
                self._connection = None
                self._process = None
                self._fault_lockout = True
                raise
            child.close()
            try:
                result = self._invoke_locked("connect", {})
                if not isinstance(result, Mapping):
                    raise FollowerWorkerError("connect response is not a mapping")
                child_joints = result.get("joint_names")
                child_status = result.get("status")
                if child_joints != list(self.joint_names) or not isinstance(child_status, Mapping):
                    raise FollowerWorkerError("child SO-101 action schema does not match the parent")
                self._child_status = dict(child_status)
                self._publish_health(result.get("servo_health"))
                self._connected = True
            except Exception:
                self._fault_lockout = True
                self._terminate_locked("connect_failed")
                raise

    def observe(self) -> Mapping[str, float]:
        with self._lock:
            if not self._connected:
                raise FollowerWorkerError("follower worker is not connected for observation")
            result = self._invoke_locked("observe", {})
            if not isinstance(result, Mapping) or set(result) != {"positions", "servo_health"}:
                raise FollowerWorkerError("observe response has an invalid envelope")
            self._publish_health(result["servo_health"])
            return self._positions(result["positions"], "observe")

    def execute(self, positions: Mapping[str, float]) -> Mapping[str, float]:
        with self._lock:
            if not self._connected:
                raise FollowerWorkerError("follower worker dispatch is halted")
            if tuple(positions) != self.joint_names:
                raise ValueError("positions must match the exact ordered SO-101 action schema")
            result = self._invoke_locked("execute", {"positions": dict(positions)})
            return self._positions(result, "execute")

    def stop(self, reason: str) -> Mapping[str, object]:
        with self._lock:
            if not isinstance(reason, str) or not reason:
                raise ValueError("stop reason must be non-empty")
            # Parent-local revocation is immediate even while the child works
            # on the physical torque-disable request.
            self._connected = False
            if self._process is None or self._connection is None:
                receipt = self._unknown_stop_receipt(
                    reason,
                    disable_requested=False,
                    fault=(
                        "follower_worker_terminated_without_torque_evidence"
                        if self._terminated_without_close
                        else None
                    ),
                    worker_terminated=self._terminated_without_close,
                )
                self.last_stop_receipt = receipt
                return receipt
            try:
                result = self._invoke_locked("stop", {"reason": reason})
                if not isinstance(result, Mapping):
                    raise FollowerWorkerError("stop response is not a mapping")
                receipt = dict(result)
                if (
                    receipt.get("hardware_stop_completed") is not True
                    or receipt.get("torque_off_confirmed") is not True
                    or receipt.get("fault") not in {None, ""}
                ):
                    self._fault_lockout = True
                    receipt["fault_lockout"] = True
                else:
                    receipt["fault_lockout"] = False
                receipt["worker_terminated"] = False
            except FollowerWorkerTimeoutError:
                # _invoke_locked has already bounded/terminated the worker.
                receipt = self._unknown_stop_receipt(
                    reason,
                    disable_requested=True,
                    fault="follower_worker_unresponsive",
                    worker_terminated=True,
                )
            except FollowerWorkerError:
                terminated = self._process is None
                receipt = self._unknown_stop_receipt(
                    reason,
                    disable_requested=True,
                    fault=("follower_worker_unavailable" if terminated else "follower_worker_stop_failed"),
                    worker_terminated=terminated,
                )
            self.last_stop_receipt = receipt
            return receipt

    def close(self) -> Mapping[str, object]:
        with self._lock:
            if self._process is None or self._connection is None:
                completed = not self._terminated_without_close
                receipt = {
                    "close_requested": True,
                    "close_completed": completed,
                    "fault": None if completed else "worker_terminated_before_close",
                    "fault_lockout": self._fault_lockout,
                }
                self.last_close_receipt = receipt
                return receipt
            try:
                result = self._invoke_locked("close", {})
                if not isinstance(result, Mapping):
                    raise FollowerWorkerError("close response is not a mapping")
                receipt = dict(result)
                completed = receipt.get("close_completed") is True
                self._finish_graceful_worker_locked()
                if not completed:
                    self._fault_lockout = True
                    receipt["fault_lockout"] = True
            except (FollowerWorkerTimeoutError, FollowerWorkerError):
                self._fault_lockout = True
                if self._process is not None:
                    self._terminate_locked("close_failed")
                receipt = {
                    "close_requested": True,
                    "close_completed": False,
                    "fault": "follower_worker_close_unresponsive",
                    "fault_lockout": True,
                }
            self._connected = False
            self._health_service.detach(self._health_key)
            self.last_close_receipt = receipt
            return receipt

    def _publish_health(self, snapshot: object) -> None:
        if snapshot is None:
            return
        if not isinstance(snapshot, Mapping):
            raise FollowerWorkerError("servo-health snapshot is not a mapping")
        try:
            self._health_service.publish(self._health_key, snapshot)
        except (RuntimeError, ValueError) as exc:
            raise FollowerWorkerError("servo-health snapshot was refused") from exc

    def _positions(self, result: object, operation: str) -> dict[str, float]:
        if not isinstance(result, Mapping) or tuple(result) != self.joint_names:
            raise FollowerWorkerError(f"{operation} response changed the SO-101 action schema")
        clean: dict[str, float] = {}
        for key, value in result.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FollowerWorkerError(f"{operation} response contains a non-numeric position")
            number = float(value)
            if not math.isfinite(number):
                raise FollowerWorkerError(f"{operation} response contains a non-finite position")
            clean[key] = number
        return clean

    def _invoke_locked(self, operation: str, payload: Mapping[str, object]) -> object:
        connection = self._connection
        process = self._process
        if connection is None or process is None:
            raise FollowerWorkerError("follower worker is unavailable")
        self._next_request_id += 1
        request_id = self._next_request_id
        try:
            connection.send(_request(request_id, operation, payload))
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._fault_lockout = True
            self._terminate_locked(f"{operation}_send_failed")
            raise FollowerWorkerError(f"follower worker {operation} send failed") from exc

        timeout = self.timeouts.for_operation(operation)
        if not connection.poll(timeout):
            self._fault_lockout = True
            self._terminate_locked(f"{operation}_timeout")
            raise FollowerWorkerTimeoutError(f"follower worker {operation} exceeded {timeout:.3f}s")
        try:
            response = _validate_response(connection.recv(), request_id)
        except (EOFError, OSError, FollowerWorkerError) as exc:
            self._fault_lockout = True
            self._terminate_locked(f"{operation}_response_failed")
            raise FollowerWorkerError(f"follower worker {operation} response failed") from exc
        if response["ok"] is not True:
            code = response["error_code"] or "operation_failed"
            if operation in {"connect", "stop", "close"}:
                self._fault_lockout = True
            raise FollowerWorkerError(f"follower worker {operation} failed ({code})")
        return response["result"]

    def _terminate_locked(self, reason: str) -> None:
        process = self._process
        connection = self._connection
        self._connected = False
        self._fault_lockout = True
        self._terminated_without_close = True
        if process is not None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=self.timeouts.terminate_s)
            if process.is_alive():
                process.kill()
                process.join(timeout=self.timeouts.terminate_s)
            with suppress(OSError, ValueError):
                process.close()
        if connection is not None:
            connection.close()
        self._process = None
        self._connection = None
        self._health_service.detach(self._health_key)
        self._child_status = {"state": "terminated", "reason": reason}

    def _finish_graceful_worker_locked(self) -> None:
        process = self._process
        connection = self._connection
        if process is not None:
            process.join(timeout=self.timeouts.terminate_s)
            if process.is_alive():
                # Close already returned, so torque/close evidence remains what
                # the child reported; termination only bounds process cleanup.
                process.terminate()
                process.join(timeout=self.timeouts.terminate_s)
            with suppress(OSError, ValueError):
                process.close()
        if connection is not None:
            connection.close()
        self._process = None
        self._connection = None
        self._health_service.detach(self._health_key)
        self._terminated_without_close = False

    def _unknown_stop_receipt(
        self,
        reason: str,
        *,
        disable_requested: bool,
        fault: str | None,
        worker_terminated: bool = False,
    ) -> dict[str, object]:
        if fault is not None:
            self._fault_lockout = True
        return {
            "reason": reason,
            "disable_requested": disable_requested,
            "hardware_stop_completed": False,
            "torque_off_confirmed": None,
            "verification": "worker_unavailable_no_torque_readback",
            "fault": fault,
            "fault_lockout": self._fault_lockout,
            "worker_terminated": worker_terminated,
        }
