"""``client.describe()`` — one call that answers "where do things stand?".

The first thing to run against an unfamiliar server: one fan-out over the
read-only orientation endpoints (health, current session, recent jobs, peer
nodes), each section guarded so one failing corner never hides the rest.
Agents get their bearings from this instead of four exploratory calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from makermodslab_sdk.errors import MakerModsError
from makermodslab_sdk.resources._base import SdkModel
from makermodslab_sdk.resources.jobs import Job
from makermodslab_sdk.resources.nodes import Node
from makermodslab_sdk.resources.sessions import EndedSessionInfo, SessionInfo
from makermodslab_sdk.resources.system import Health

if TYPE_CHECKING:
    from makermodslab_sdk.client import Client


class ServerSnapshot(SdkModel):
    """The composite orientation document ``client.describe()`` returns.

    Any section a request failed for is None/empty here and explained in
    ``errors`` (section name → the error text, remediation included).
    """

    health: Health | None = None
    session: SessionInfo | None = None
    last_ended: EndedSessionInfo | None = None
    running_jobs: list[Job] = []
    recent_jobs: list[Job] = []
    nodes: list[Node] = []
    errors: dict[str, str] = {}

    def summary(self) -> str:
        """A compact human/agent-readable rendering of the snapshot."""
        lines: list[str] = []
        if self.health is not None:
            capabilities = self.health.capabilities
            lines.append(
                f"server: {self.health.status} v{self.health.version} "
                f"(node {self.health.instance_id[:8]}…, accepts_jobs={capabilities.accepts_jobs})"
            )
        if self.session is not None:
            lease = self.session.lease
            lines.append(
                f"session: {self.session.kind} on robot {self.session.robot!r} "
                f"(id {self.session.id}, phase {self.session.phase}, "
                f"{'lease ' + str(lease.timeout_s) + 's' if lease else 'no lease'})"
            )
        else:
            ended = (
                f" (last: {self.last_ended.kind}, reason {self.last_ended.reason})"
                if self.last_ended is not None
                else ""
            )
            lines.append(f"session: none — the robot is free{ended}")
        if self.running_jobs:
            for job in self.running_jobs:
                metrics = job.metrics
                lines.append(
                    f"job running: #{job.job_number} {job.display_name or job.name} "
                    f"step {metrics.current_step}/{metrics.total_steps}"
                )
        else:
            lines.append(f"jobs: none running ({len(self.recent_jobs)} recent)")
        lines.append(
            "nodes: "
            + (
                ", ".join(f"{n.name or n.url} [{n.status}]" for n in self.nodes)
                if self.nodes
                else "none registered"
            )
        )
        for section, error in self.errors.items():
            lines.append(f"({section} unavailable: {error})")
        return "\n".join(lines)


def snapshot(client: Client) -> ServerSnapshot:
    """Build the composite snapshot (the body of ``client.describe()``)."""
    result = ServerSnapshot()
    try:
        result.health = client.system.health()
    except MakerModsError as exc:  # ConnectionFailedError propagates meaningfully
        # ... except that with no server at all, EVERY section would fail the
        # same way; let the very first failure speak once instead of four times.
        raise exc from None
    try:
        current = client.sessions.current()
        result.session = current.session
        result.last_ended = current.last_ended
    except MakerModsError as exc:
        result.errors["session"] = str(exc)
    try:
        jobs = client.jobs.list(limit=10).jobs
        result.recent_jobs = jobs
        result.running_jobs = [job for job in jobs if job.state == "running"]
    except MakerModsError as exc:
        result.errors["jobs"] = str(exc)
    try:
        result.nodes = client.nodes.list()
    except MakerModsError as exc:
        result.errors["nodes"] = str(exc)
    return result
