"""The ``remote`` namespace: remote teleoperation over the Lab's bundled SFU.

Two sides, one namespace: a STATION hosts its robot (session kind
``hosting``, or full station mode via ``makermodslab --host``) and publishes
state + camera frames into a LiveKit room; an OPERATOR machine's leader
drives it (session kind ``remote_teleoperation``, started with
``client.sessions.remote_teleoperate(robot, station=...)``). These are the
status and control routes around those sessions.

Semantics worth knowing: the station's arm is *parked* (torque off at rest,
streaming, listening) until an operator takes the single seat, then
*engaged* (following). ``home()`` parks it mid-session (holds until
``engage()``); ending the operator session releases the seat and parks. The
verb routes answer ``{success, message}`` — ``success=False`` carries the
reason, not an HTTP error.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from makermodslab_sdk._operations import operation
from makermodslab_sdk.resources._base import Resource, SdkModel


class HostingStatus(SdkModel):
    """GET /api/v1/hosting — the station side's wire contract while live
    (``hosting`` is the descriptor: room, codec, fps, motors, cameras,
    ``phase``, ``active_operator``, ``station_mode``)."""

    hosting_active: bool
    hosting: dict[str, Any] | None = None
    releasing: bool = False
    last_cleanup_error: str | None = None
    outcome: str | None = None
    error: str | None = None
    hint: str | None = None
    message: str = ""


class StationStatus(SdkModel):
    """GET /api/v1/station — station mode's supervisor state. ``hostable``
    lists the robot records this machine could host."""

    station_mode: bool
    robot: str | None = None
    hostable: list[str] = []
    hosting_active: bool = False
    phase: str | None = None


class RemoteTeleoperationStatus(SdkModel):
    """GET /api/v1/remote-teleoperation — the operator side's live state.
    ``station_phase`` is parked/engaging/engaged/parking as the station
    reports it; ``cameras`` are re-streamed names for ``camera_url``."""

    remote_teleoperation_active: bool
    station: dict[str, Any] | None = None
    station_phase: str | None = None
    room: str | None = None
    cameras: list[str] = []
    metrics: dict[str, Any] | None = None
    last_cleanup_error: str | None = None
    outcome: str | None = None
    error: str | None = None
    hint: str | None = None
    message: str = ""


class RemoteCommandResult(SdkModel):
    """The home/engage verbs' shared answer — check ``success``."""

    success: bool
    message: str


class RemoteResource(Resource):
    """``client.remote`` — remote-teleoperation status and controls.

    Example:
        >>> client.remote.station().station_mode
        True
        >>> with client.sessions.remote_teleoperate("laptop-leader", station=peer_id) as s:
        ...     client.remote.home()  # park the far arm; Engage resumes
        ...     client.remote.engage()
    """

    @operation("get_hosting_status")
    def hosting_status(self) -> HostingStatus:
        """The STATION side's hosting state (run against the station's API).

        Example:
            >>> h = client.remote.hosting_status()
            >>> h.hosting_active, (h.hosting or {}).get("phase")
            (True, 'parked')
        """
        return HostingStatus.model_validate(
            self._transport.request("GET", "/api/v1/hosting", action="Get hosting status")
        )

    @operation("get_station_status")
    def station(self) -> StationStatus:
        """Station mode's supervisor state (mode on/off, the hosted robot,
        what else is hostable, the live phase)."""
        return StationStatus.model_validate(
            self._transport.request("GET", "/api/v1/station", action="Get station status")
        )

    @operation("set_station_robot")
    def set_station_robot(self, robot: str | None) -> StationStatus:
        """Pick (or clear, with None) the robot station mode hosts — the
        choice is remembered across restarts. Refuses with session.held
        while an operator is driving."""
        return StationStatus.model_validate(
            self._transport.request(
                "PUT", "/api/v1/station/robot", json={"robot": robot}, action="Set station robot"
            )
        )

    @operation("get_remote_teleoperation_status")
    def teleoperation_status(self) -> RemoteTeleoperationStatus:
        """The OPERATOR side's live state (run against the operator's API)."""
        return RemoteTeleoperationStatus.model_validate(
            self._transport.request(
                "GET", "/api/v1/remote-teleoperation", action="Get remote teleoperation status"
            )
        )

    @operation("remote_teleoperation_home")
    def home(self) -> RemoteCommandResult:
        """Park the station's arm mid-session (return-to-rest, torque off);
        it holds parked until ``engage()``."""
        return RemoteCommandResult.model_validate(
            self._transport.request(
                "POST", "/api/v1/remote-teleoperation/home", action="Remote teleoperation home"
            )
        )

    @operation("remote_teleoperation_engage")
    def engage(self) -> RemoteCommandResult:
        """Re-engage after a ``home()`` — the station soft-starts from the
        follower's present pose."""
        return RemoteCommandResult.model_validate(
            self._transport.request(
                "POST", "/api/v1/remote-teleoperation/engage", action="Remote teleoperation engage"
            )
        )

    @operation("get_remote_teleoperation_camera")
    def camera_url(self, name: str) -> str:
        """The MJPEG re-stream URL for one of the station's cameras (names
        from ``teleoperation_status().cameras``).

        Deliberately returns the URL instead of reading the stream: it is
        endless, and an agent must never be handed an unbounded read (the
        same rule as ``sample_joints``). Point a browser/<img> at it, or
        fetch single frames with your own bounded reader.
        """
        return f"{self._transport.base_url}/api/v1/remote-teleoperation/camera/{quote(name, safe='')}"
