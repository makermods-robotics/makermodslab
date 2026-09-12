"""The ``robots`` namespace: the saved robot records everything else starts
from (sessions start by robot NAME; the record holds ports, arm layout,
calibration names, cameras, motor power).

PROVISIONAL surface: these routes are untagged and untyped server-side at
this snapshot (slated for a redesign), so unlike the tagged namespaces their
errors arrive as plain ApiError with legacy ``{status, message}`` bodies and
no codes. The SDK exposes them anyway — the web UI can manage records, so
UI-parity (and beyond) demands the SDK can too. When the server tags them,
this namespace graduates into the coverage ratchet's tagged set.

Record semantics worth knowing (server.py upsert_robot):
- ``mode`` ("single" | "bimanual") is FIXED AT CREATION — changing it on an
  existing record refuses with 409; create a new robot instead.
- Assigning one serial port, or one calibration config, to two arms of the
  same robot refuses with 409 naming the conflict.
- ``update`` on a missing record is a deliberate no-op success (the server's
  deletion-during-calibration edge case), NOT an error — check ``.robot``
  for None if you need to know.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from makermodslab_sdk._operations import operation
from makermodslab_sdk.resources._base import Resource, SdkModel


class Robot(SdkModel):
    """One saved robot record. Every field is optional server-side (records
    are free-form merged dicts); unknown keys ride along via extra="allow"."""

    mode: str | None = None  # "single" | "bimanual"
    leader_port: str | None = None
    follower_port: str | None = None
    right_leader_port: str | None = None
    right_follower_port: str | None = None
    leader_config: str | None = None
    follower_config: str | None = None
    right_leader_config: str | None = None
    right_follower_config: str | None = None
    cameras: Any = None
    motor_power: int | None = None


class RobotsList(SdkModel):
    """GET /api/v1/robots. NOTE: an internal listing failure is a 200 with
    status="error" and message set (legacy shape) — check status."""

    status: str
    robots: list[dict[str, Any]] = []
    message: str | None = None


class RobotEnvelope(SdkModel):
    """The {status, robot} shape of the single-record routes. ``robot`` is
    None on update's no-op path (record didn't exist)."""

    status: str
    robot: dict[str, Any] | None = None


def _robot_path(name: str, suffix: str = "") -> str:
    return f"/api/v1/robots/{quote(name, safe='')}{suffix}"


class RobotsResource(Resource):
    """``client.robots`` — create, edit, and inspect saved robot records.

    Example:
        >>> [r["name"] for r in client.robots.list().robots]
        ['bench']
        >>> client.robots.get("bench").follower_port
        '/dev/tty.usbmodem123'
    """

    @operation("get_robots")
    def list(self) -> RobotsList:
        """All saved robot records (each a dict including its ``name``).

        Example:
            >>> client.robots.list().robots
            [{'name': 'bench', 'mode': 'single', ...}]
        """
        return RobotsList.model_validate(
            self._transport.request("GET", "/api/v1/robots", action="List robots")
        )

    @operation("get_robot")
    def get(self, name: str) -> Robot:
        """One record by exact name (404 when it doesn't exist).

        Example:
            >>> client.robots.get("bench").mode
            'single'
        """
        body = self._transport.request("GET", _robot_path(name), action=f"Get robot {name!r}")
        return Robot.model_validate(body["robot"])

    @operation("upsert_robot")
    def create(self, name: str, **fields: Any) -> Robot:
        """Create a NEW record (409 if the name exists). ``mode`` is fixed
        here forever — pass mode="bimanual" now or never.

        Example:
            >>> client.robots.create(
            ...     "bench",
            ...     mode="single",
            ...     leader_port="/dev/tty.usbmodemA",
            ...     follower_port="/dev/tty.usbmodemB",
            ... ).mode
            'single'
        """
        body = self._transport.request(
            "POST",
            _robot_path(name) + "?create=true",
            json=fields,
            action=f"Create robot {name!r}",
        )
        return Robot.model_validate(body["robot"])

    @operation("upsert_robot")
    def update(self, name: str, **fields: Any) -> Robot | None:
        """Merge ``fields`` into an existing record (ports, configs, cameras,
        motor_power, …). Changing ``mode`` refuses (409); a missing record is
        a no-op success returning None — the server's deliberate
        deletion-during-calibration behavior.

        Example:
            >>> client.robots.update("bench", motor_power=60).motor_power
            60
        """
        body = self._transport.request(
            "POST", _robot_path(name), json=fields, action=f"Update robot {name!r}"
        )
        robot = body.get("robot")
        return Robot.model_validate(robot) if robot is not None else None

    @operation("rename_robot")
    def rename(self, name: str, new_name: str) -> Robot | None:
        """Rename a record (calibration files are unaffected — they're keyed
        by config name, not robot name)."""
        body = self._transport.request(
            "POST",
            _robot_path(name, "/rename"),
            json={"new_name": new_name},
            action=f"Rename robot {name!r}",
        )
        robot = body.get("robot")
        return Robot.model_validate(robot) if robot is not None else None

    @operation("delete_robot")
    def delete(self, name: str) -> None:
        """Delete a record (404 when it doesn't exist). The record only —
        calibration files and datasets stay."""
        self._transport.request("DELETE", _robot_path(name), action=f"Delete robot {name!r}")
