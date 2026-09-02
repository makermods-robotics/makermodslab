"""Local API facade for explicitly configured remote-teleoperation roles."""

from __future__ import annotations

import threading
from typing import Any

from ..schemas.arms import RemoteCommissionBody, RemoteConfigurationBody
from .config import OperatorRoleConfig, RemoteRoleConfigStore, RobotRoleConfig
from .operator_service import RemoteOperatorService
from .robot_service import RemoteRobotService
from .service import RemoteSimulationService, remote_simulation_service


class RemoteTeleoperationRuntimeService:
    """Own both dormant role runtimes; expose only the configured one.

    Constructing this facade does not create a thread, listener, socket, or
    hardware adapter. Configuration is durable, but runtime enablement is not.
    """

    def __init__(
        self,
        *,
        config_store: RemoteRoleConfigStore | None = None,
        robot: RemoteRobotService | None = None,
        operator: RemoteOperatorService | None = None,
        simulation: RemoteSimulationService = remote_simulation_service,
    ) -> None:
        self.config_store = config_store or RemoteRoleConfigStore()
        self.robot = robot or RemoteRobotService(config_store=self.config_store)
        self.operator = operator or RemoteOperatorService(config_store=self.config_store)
        self.simulation = simulation
        self._lock = threading.RLock()

    def _active(self) -> bool:
        return bool(self.robot.status()["runtime_enabled"] or self.operator.status()["runtime_enabled"])

    def configure(self, body: RemoteConfigurationBody) -> dict[str, Any]:
        with self._lock:
            if self._active():
                raise RuntimeError("disable remote teleoperation before changing its configuration")
            self.robot.assert_no_durable_fault()
            lease = self.robot.registry.snapshot()
            if lease.held and lease.state in {"unresolved", "recovering"}:
                raise RuntimeError("resolve the retained hardware fault before changing roles")
            if body.role == "robot":
                assert body.robot is not None
                self.config_store.save_robot(RobotRoleConfig(**body.robot.model_dump()))
                # Even apparently identical form data may refer to a changed
                # calibration file or device contract. Never carry a physical
                # commissioning proof through a robot configuration save.
                self.robot.commissioning.invalidate()
            else:
                assert body.operator is not None
                self.config_store.save_operator(OperatorRoleConfig(**body.operator.model_dump()))
        return self.status()

    def commission(self, body: RemoteCommissionBody) -> dict[str, Any]:
        loaded = self.config_store.load()
        if loaded is None or loaded[0] != "robot":
            raise ValueError("configure this host as Remote robot first")
        self.robot.commission(body.model_dump())
        return self.status()

    def recover_fault(self, body: RemoteCommissionBody) -> dict[str, Any]:
        loaded = self.config_store.load()
        if loaded is None or loaded[0] != "robot":
            raise ValueError("configure this host as Remote robot first")
        self.robot.recover_fault(body.model_dump())
        return self.status()

    def enable(self) -> dict[str, Any]:
        with self._lock:
            self.robot.assert_no_durable_fault()
            loaded = self.config_store.load()
            if loaded is None:
                raise ValueError("configure a remote teleoperation role first")
            if loaded[0] == "robot":
                self.robot.enable()
            else:
                self.operator.start()
        return self.status()

    def clear_configuration(self) -> dict[str, Any]:
        with self._lock:
            if self._active():
                raise RuntimeError("disable remote teleoperation before removing its configuration")
            self.robot.assert_no_durable_fault()
            lease = self.robot.registry.snapshot()
            if lease.held and lease.state in {"unresolved", "recovering"}:
                raise RuntimeError("resolve the retained hardware fault before removing configuration")
            self.config_store.clear()
        return self.status()

    def disable(self) -> dict[str, Any]:
        with self._lock:
            robot_enabled = bool(self.robot.status()["runtime_enabled"])
            operator_state = str(self.operator.status()["state"])
            if robot_enabled:
                self.robot.disable()
            if operator_state not in {"idle", "fault"}:
                self.operator.stop("remote_role_disabled")
        return self.status()

    def open_pairing_window(self) -> dict[str, object]:
        loaded = self.config_store.load()
        if loaded is None or loaded[0] != "robot":
            raise ValueError("configure this host as Remote robot first")
        return self.robot.open_pairing_window()

    def pair(self, pairing_token: str, operator_label: str) -> dict[str, object]:
        loaded = self.config_store.load()
        if loaded is None or loaded[0] != "operator":
            raise ValueError("configure this host as Remote operator first")
        return self.operator.pair(pairing_token, operator_label)

    def browser_heartbeat(self) -> dict[str, object]:
        return self.operator.browser_heartbeat()

    def stop(self, reason: str) -> dict[str, Any]:
        loaded = self.config_store.load()
        if loaded is None:
            raise ValueError("configure a remote teleoperation role first")
        if loaded[0] == "robot":
            self.robot.local_stop(reason)
        else:
            self.operator.stop(reason)
        return self.status()

    def revoke(self, credential_id: str) -> dict[str, Any]:
        loaded = self.config_store.load()
        if loaded is None or loaded[0] != "robot":
            raise ValueError("configure this host as Remote robot first")
        if not self.robot.revoke_credential(credential_id):
            raise KeyError("remote operator credential not found")
        return self.status()

    def status(self) -> dict[str, Any]:
        configured = self.config_store.public()
        role = configured["role"]
        if role == "robot":
            runtime = self.robot.status()
        elif role == "operator":
            runtime = self.operator.status()
        else:
            runtime = None
        simulation = self.simulation.status()
        lease = self.robot.registry.snapshot()
        state = (
            str(runtime["state"])
            if runtime is not None
            else str(simulation["state"])
            if simulation["state"] != "idle"
            else "unconfigured"
        )
        return {
            "protocol_version": "makermodslab.remote-teleoperation-api.v1",
            "configured": configured["configured"],
            "role": role,
            "config": configured["config"],
            "runtime_enabled": bool(runtime and runtime["runtime_enabled"]),
            "live_hardware_enabled": bool(runtime and runtime["live_hardware_enabled"]),
            "state": state,
            "runtime": runtime,
            "simulation": simulation,
            "commissioning": self.robot.commissioning_status(),
            "durable_fault": self.robot.fault_status(),
            "hardware_registry": {
                "held": lease.held,
                "state": lease.state,
                "kind": lease.kind,
                "owner": lease.owner,
                "generation": lease.generation,
                "pending_unresolved": lease.pending_unresolved,
                "pending_kind": lease.pending_kind,
                "pending_owner": lease.pending_owner,
            },
        }

    def shutdown(self) -> None:
        self.operator.shutdown()
        self.robot.shutdown()


remote_teleoperation_runtime = RemoteTeleoperationRuntimeService()
