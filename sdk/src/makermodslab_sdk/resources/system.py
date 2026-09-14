"""The ``system`` namespace: health/identity, auth, discovery, extras, updates.

Response models mirror makermodslab/schemas/system.py; SdkModel keeps them
``extra="allow"`` so an older SDK against a newer server never breaks.
"""

from __future__ import annotations

from typing import Any

from makermodslab_sdk._operations import operation
from makermodslab_sdk.resources._base import Resource, SdkModel


class HealthCapabilities(SdkModel):
    serves_ui: bool
    accepts_jobs: bool


class Health(SdkModel):
    """Node identity + capability document (GET /api/v1/health)."""

    status: str
    message: str
    version: str
    instance_id: str
    capabilities: HealthCapabilities


class CameraInfo(SdkModel):
    index: int
    name: str
    available: bool
    unique_id: str | None = None


class AvailableCameras(SdkModel):
    status: str
    cameras: list[CameraInfo]
    message: str | None = None


class AvailablePorts(SdkModel):
    status: str
    ports: list[str] | None = None
    message: str | None = None


class HfAuthStatus(SdkModel):
    authenticated: bool
    username: str | None = None
    orgs: list[str] = []
    writable_namespaces: list[str] = []
    login_command: str = ""


class HfLoginResult(SdkModel):
    authenticated: bool
    username: str
    orgs: list[str] = []
    login_command: str = ""


class PolicyOptimizerDefaults(SdkModel):
    defaults: dict[str, Any]
    available: dict[str, Any]


class RobotPort(SdkModel):
    status: str
    saved_port: str | None = None
    default_port: str = ""


class SupplyVoltage(SdkModel):
    success: bool
    voltage: float | None = None
    message: str | None = None


class ExtraStatus(SdkModel):
    available: bool
    install_hint: str


class PolicyExtraStatus(SdkModel):
    policy_type: str
    needs_extra: bool
    available: bool
    package: str
    install_target: str
    install_hint: str


class InstallStart(SdkModel):
    started: bool
    message: str


class InstallStatus(SdkModel):
    state: str
    error: str | None = None
    logs: list[dict[str, Any]] = []


class UpdateStatus(SdkModel):
    update_available: bool
    can_auto_update: bool = False
    commits_behind: int | None = None
    current_commit: str | None = None
    latest_commit: str | None = None
    compare_url: str | None = None
    update_command: str | None = None


class UpdateResult(SdkModel):
    success: bool
    message: str
    output: str = ""


class SystemResource(Resource):
    """``client.system`` — server identity and machine-level operations.

    Example:
        >>> client.system.health().version
        '0.1.0'
    """

    @operation("health_check")
    def health(self) -> Health:
        """The server's identity and capability document.

        Use it to check the server is up and which node this is —
        ``instance_id`` is the stable 32-hex identity peers recognize this
        machine by, and ``capabilities`` says what it will do (serve the UI,
        accept training jobs, …).

        Example:
            >>> h = client.system.health()
            >>> h.status, h.version, h.capabilities.accepts_jobs
            ('ok', '0.1.0', True)
        """
        return Health.model_validate(
            self._transport.request("GET", "/api/v1/health", action="Get server health")
        )

    @operation("get_available_cameras")
    def available_cameras(self) -> AvailableCameras:
        """Cameras the server machine can see (probes devices — takes a moment).

        Example:
            >>> [c.name for c in client.system.available_cameras().cameras]
            ['FaceTime HD Camera']
        """
        return AvailableCameras.model_validate(
            self._transport.request("GET", "/api/v1/available-cameras", action="List available cameras")
        )

    @operation("get_available_ports")
    def available_ports(self) -> AvailablePorts:
        """Serial ports the server machine can see (where arms plug in).

        Example:
            >>> client.system.available_ports().ports
            ['/dev/tty.usbmodem123', '/dev/tty.usbmodem456']
        """
        return AvailablePorts.model_validate(
            self._transport.request("GET", "/api/v1/available-ports", action="List available ports")
        )

    @operation("hf_auth_status")
    def hf_auth_status(self) -> HfAuthStatus:
        """Whether the SERVER machine is logged in to the Hugging Face Hub,
        and which namespaces it can push to.

        Example:
            >>> s = client.system.hf_auth_status()
            >>> s.authenticated, s.username
            (True, 'someuser')
        """
        return HfAuthStatus.model_validate(
            self._transport.request("GET", "/api/v1/hf-auth-status", action="Get HF auth status")
        )

    @operation("hf_auth_login")
    def hf_login(self, token: str) -> HfLoginResult:
        """Log the SERVER machine in to the Hugging Face Hub with a token.

        Example:
            >>> client.system.hf_login(token="hf_...").authenticated
            True
        """
        return HfLoginResult.model_validate(
            self._transport.request("POST", "/api/v1/hf-auth/login", json={"token": token}, action="HF login")
        )

    @operation("get_policy_optimizer_defaults")
    def policy_optimizer_defaults(self) -> PolicyOptimizerDefaults:
        """Per-policy-type optimizer presets for training jobs.

        Example:
            >>> client.system.policy_optimizer_defaults().defaults["act"]["lr"]
            1e-05
        """
        return PolicyOptimizerDefaults.model_validate(
            self._transport.request(
                "GET", "/api/v1/policy-optimizer-defaults", action="Get optimizer defaults"
            )
        )

    @operation("get_robot_port")
    def robot_port(self, robot_type: str) -> RobotPort:
        """The last-used serial port for ``"leader"`` or ``"follower"``.

        Example:
            >>> client.system.robot_port("follower").saved_port
            '/dev/tty.usbmodem123'
        """
        return RobotPort.model_validate(
            self._transport.request("GET", f"/api/v1/robot-port/{robot_type}", action="Get saved robot port")
        )

    @operation("supply_voltage")
    def supply_voltage(self, port: str | None = None) -> SupplyVoltage:
        """Servo-bus supply voltage (talks to the hardware on ``port``, or the
        saved follower port when omitted).

        Example:
            >>> client.system.supply_voltage().voltage
            12.1
        """
        params = {"port": port} if port is not None else None
        return SupplyVoltage.model_validate(
            self._transport.request(
                "GET", "/api/v1/supply-voltage", params=params, action="Read supply voltage"
            )
        )

    # --- optional-extra installs (training / wandb / per-policy) -------------

    @operation("get_training_extra")
    def training_extra(self) -> ExtraStatus:
        """Whether the local-training optional dependency set is installed."""
        return ExtraStatus.model_validate(
            self._transport.request("GET", "/api/v1/system/training-extra", action="Get training extra")
        )

    @operation("install_training_extra")
    def install_training_extra(self) -> InstallStart:
        """Start installing the local-training extra (async; poll
        ``training_extra_install_status()``)."""
        return InstallStart.model_validate(
            self._transport.request(
                "POST", "/api/v1/system/training-extra/install", action="Install training extra"
            )
        )

    @operation("install_training_extra_status")
    def training_extra_install_status(self) -> InstallStatus:
        """Progress of the training-extra install (``state`` ends at
        ``"succeeded"``/``"failed"``)."""
        return InstallStatus.model_validate(
            self._transport.request(
                "GET",
                "/api/v1/system/training-extra/install-status",
                action="Training extra install status",
            )
        )

    @operation("get_wandb_extra")
    def wandb_extra(self) -> ExtraStatus:
        """Whether the Weights & Biases optional dependency is installed."""
        return ExtraStatus.model_validate(
            self._transport.request("GET", "/api/v1/system/wandb-extra", action="Get wandb extra")
        )

    @operation("install_wandb_extra")
    def install_wandb_extra(self) -> InstallStart:
        """Start installing the wandb extra (async; poll
        ``wandb_extra_install_status()``)."""
        return InstallStart.model_validate(
            self._transport.request(
                "POST", "/api/v1/system/wandb-extra/install", action="Install wandb extra"
            )
        )

    @operation("install_wandb_extra_status")
    def wandb_extra_install_status(self) -> InstallStatus:
        """Progress of the wandb-extra install."""
        return InstallStatus.model_validate(
            self._transport.request(
                "GET", "/api/v1/system/wandb-extra/install-status", action="Wandb extra install status"
            )
        )

    @operation("get_policy_extra")
    def policy_extra(self, policy_type: str) -> PolicyExtraStatus:
        """Whether the optional dependency for a policy type (e.g. ``"pi0"``)
        is installed; ``needs_extra`` is False for built-in policies."""
        return PolicyExtraStatus.model_validate(
            self._transport.request(
                "GET", f"/api/v1/system/policy-extra/{policy_type}", action="Get policy extra"
            )
        )

    @operation("install_policy_extra")
    def install_policy_extra(self, policy_type: str) -> InstallStart:
        """Start installing a policy type's extra (async; poll
        ``policy_extra_install_status(policy_type)``)."""
        return InstallStart.model_validate(
            self._transport.request(
                "POST",
                f"/api/v1/system/policy-extra/{policy_type}/install",
                action="Install policy extra",
            )
        )

    @operation("install_policy_extra_status")
    def policy_extra_install_status(self, policy_type: str) -> InstallStatus:
        """Progress of a policy extra's install."""
        return InstallStatus.model_validate(
            self._transport.request(
                "GET",
                f"/api/v1/system/policy-extra/{policy_type}/install-status",
                action="Policy extra install status",
            )
        )

    # --- app updates ---------------------------------------------------------

    @operation("update_check")
    def update_check(self) -> UpdateStatus:
        """Whether a newer MakerMods Lab is available for the server.

        Example:
            >>> u = client.system.update_check()
            >>> u.update_available, u.commits_behind
            (False, 0)
        """
        return UpdateStatus.model_validate(
            self._transport.request("GET", "/api/v1/system/update-check", action="Check for updates")
        )

    @operation("run_update")
    def update(self) -> UpdateResult:
        """Run the server's self-update (the server restarts if it succeeds —
        expect the connection to drop afterwards)."""
        return UpdateResult.model_validate(
            self._transport.request("POST", "/api/v1/system/update", action="Run server update")
        )

    @operation("restart_server")
    def restart(self) -> RestartResult:
        """Restart the server process in place (same version). Refused with a
        409 while a live session is driving hardware. Expect the connection to
        drop, then poll ``health()`` until it answers again."""
        return RestartResult.model_validate(
            self._transport.request("POST", "/api/v1/system/restart", action="Restart server")
        )

    # --- CAN arms (Maker / Metal) --------------------------------------------

    @operation("release_can_torque")
    def release_can_torque(self, arm_type: str, port: str) -> ReleaseCanTorqueResult:
        """SAFETY HAMMER for the CAN arms: de-energize every motor on the bus.

        A Damiao (Metal) handshake IS the enable command, so a failed connect
        or a killed process can leave motors energized and holding — this
        reopens the bus without handshaking and broadcasts the disable.
        Deliberately not a session (it must work when session state is
        wrecked), but refused with a 409 while a live session is driving.

        Example:
            >>> client.system.release_can_torque("metal", "/dev/tty.usbmodemCAN1").success
            True
        """
        return ReleaseCanTorqueResult.model_validate(
            self._transport.request(
                "POST",
                "/api/v1/arms/release-torque",
                json={"arm_type": arm_type, "port": port},
                action="Release CAN torque",
            )
        )

    @operation("identify_maker_arm")
    def identify_maker_arm(
        self,
        device_type: str,
        *,
        arm_type: str | None = None,
        ports: list[str] | None = None,
    ) -> MakerIdentifyResult:
        """Find which port holds a CAN-family arm of the given role
        (``device_type``: "teleop" = leader, "robot" = follower) by probing —
        the CAN arms have no hand-motion detection. ``ports`` narrows the
        probe; omitted, every candidate port is tried.

        Example:
            >>> client.system.identify_maker_arm("robot", arm_type="maker").port
            '/dev/tty.usbmodemCAN1'
        """
        body: dict[str, Any] = {"device_type": device_type}
        if arm_type is not None:
            body["arm_type"] = arm_type
        if ports is not None:
            body["ports"] = ports
        return MakerIdentifyResult.model_validate(
            self._transport.request(
                "POST", "/api/v1/maker/identify-arm", json=body, action="Identify CAN arm"
            )
        )

    @operation("probe_maker_arm_ports")
    def probe_maker_arm_ports(
        self, *, arm_type: str | None = None, ports: list[str] | None = None
    ) -> MakerProbeResult:
        """Classify serial ports into CAN leader / follower / unknown in one
        sweep (the batch sibling of ``identify_maker_arm``).

        Example:
            >>> client.system.probe_maker_arm_ports(arm_type="maker").follower_ports
            ['/dev/tty.usbmodemCAN1']
        """
        body: dict[str, Any] = {}
        if arm_type is not None:
            body["arm_type"] = arm_type
        if ports is not None:
            body["ports"] = ports
        return MakerProbeResult.model_validate(
            self._transport.request(
                "POST", "/api/v1/maker/probe-ports", json=body, action="Probe CAN arm ports"
            )
        )


class RestartResult(SdkModel):
    """POST /api/v1/system/restart and the nodes restart proxy."""

    restarting: bool
    message: str


class ReleaseCanTorqueResult(SdkModel):
    success: bool
    message: str
    problems: list[str] = []


class MakerIdentifyResult(SdkModel):
    success: bool
    message: str
    port: str | None = None
    skipped: list[str] = []


class MakerProbeResult(SdkModel):
    success: bool
    follower_ports: list[str] = []
    leader_ports: list[str] = []
    unknown_ports: list[str] = []
    message: str = ""
