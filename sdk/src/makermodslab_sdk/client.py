from __future__ import annotations

import warnings

import httpx

from makermodslab_sdk._transport import DEFAULT_TIMEOUT, Transport
from makermodslab_sdk.errors import ApiError
from makermodslab_sdk.resources import (
    DatasetsResource,
    JobsResource,
    ModelsResource,
    NodesResource,
    Resource,
    SessionsResource,
    SystemResource,
)

# The oldest server this SDK release is known to work against (the app's
# version from the repo-root pyproject at the time the SDK was cut).
MIN_SUPPORTED_SERVER_VERSION = (0, 1, 0)

# tag -> namespace class. ONE line per namespace, kept alphabetical — parallel
# tracks each add exactly their own line, so merges never collide here.
RESOURCE_CLASSES: dict[str, type[Resource]] = {
    "datasets": DatasetsResource,
    "jobs": JobsResource,
    "models": ModelsResource,
    "nodes": NodesResource,
    "sessions": SessionsResource,
    "system": SystemResource,
}


class CompatibilityWarning(UserWarning):
    """The server looks older than this SDK targets (warn-only, never fatal)."""


def _parse_version(version: str) -> tuple[int, ...] | None:
    parts: list[int] = []
    for piece in version.split("."):
        leading = ""
        for ch in piece:
            if not ch.isdigit():
                break
            leading += ch
        if not leading:
            break
        parts.append(int(leading))
    return tuple(parts) if parts else None


class Client:
    """Agent-first client for a MakerMods Lab server.

    Example:
        >>> from makermodslab_sdk import Client
        >>> client = Client("http://localhost:8000")
        >>> client.system.health().status
        'ok'

    Namespaces mirror the API tags — ``client.system`` and ``client.sessions``
    today; ``datasets``, ``models``, ``jobs`` and ``nodes`` arrive with their
    tracks.
    Every method's docstring carries a usage example, and every error names
    the next call to make; when something fails, read the exception text.

    The first request lazily fetches ``/api/v1/health`` and warns (never
    fails) when the server predates what this SDK supports — pass
    ``check_compatibility=False`` to skip that handshake.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.Client | None = None,
        check_compatibility: bool = True,
    ) -> None:
        self._transport = Transport(
            base_url,
            timeout=timeout,
            http_client=http_client,
            on_first_request=self._verify_server_compatibility if check_compatibility else None,
        )
        self.datasets = DatasetsResource(self._transport)
        self.jobs = JobsResource(self._transport)
        self.models = ModelsResource(self._transport)
        self.nodes = NodesResource(self._transport)
        self.sessions = SessionsResource(self._transport)
        self.system = SystemResource(self._transport)

    @property
    def base_url(self) -> str:
        return self._transport.base_url

    def _verify_server_compatibility(self) -> None:
        """First-request handshake: warn when /api/v1/health is missing or old.

        Warn-only on purpose — the flat/legacy surface is shrink-only, so an
        older server mostly still works; a hard failure here would help nobody.
        Connection-level errors propagate: the real request would hit them too.
        """
        try:
            health = self.system.health()
        except ApiError:
            warnings.warn(
                f"could not verify server compatibility: {self.base_url} did not answer "
                "GET /api/v1/health — the server may predate the v1 API this SDK targets",
                CompatibilityWarning,
                stacklevel=4,
            )
            return
        version = _parse_version(health.version)
        if version is not None and version < MIN_SUPPORTED_SERVER_VERSION:
            minimum = ".".join(str(n) for n in MIN_SUPPORTED_SERVER_VERSION)
            warnings.warn(
                f"server {health.version} at {self.base_url} is older than the minimum this SDK "
                f"supports ({minimum}) — some calls may fail with 404s; update the server",
                CompatibilityWarning,
                stacklevel=4,
            )

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<makermodslab_sdk.Client {self.base_url} namespaces={sorted(RESOURCE_CLASSES)}>"

    # --- Realtime (WebSocket; needs the [realtime] extra) -------------------
    # The server's one WS (/api/v1/ws/joint-data) carries joint telemetry plus
    # control events that are droppable REFETCH HINTS, never state — see
    # realtime.py's module docstring for the contract. Imported lazily so the
    # base SDK stays dependency-light.

    def _realtime(self):
        """Import the realtime module and fail fast if the extra is missing."""
        from makermodslab_sdk import realtime

        realtime.require_websockets()
        return realtime

    def events(self, *, kinds=None):
        """Stream typed realtime events (open-ended generator; blocks).

        Yields ``JointData`` telemetry frames and the control-event hints
        (``JobsChanged``/``JobProgress``/``SessionChanged``); unknown messages
        arrive as ``UnknownEvent``. Control events mean "refetch now" — never
        treat their payload as state. ``kinds`` filters to event class(es).

        Example:
            >>> from makermodslab_sdk.realtime import SessionChanged
            >>> for event in client.events(kinds=SessionChanged):
            ...     print(event.session.kind, event.session.active)  # then refetch

        Needs the realtime extra: pip install "makermodslab-sdk[realtime]".
        For a bounded read of joint telemetry, use ``sample_joints``.
        """
        return self._realtime().events(self.base_url, kinds=kinds)

    def sample_joints(self, duration_s: float = 2.0, *, max_frames: int | None = None, clock=None):
        """Collect joint frames for up to ``duration_s`` seconds; returns a LIST.

        The agent-friendly read: always returns (never an open-ended
        iterator) — after ``duration_s`` seconds or ``max_frames`` frames,
        whichever comes first. An empty list means no hardware flow is
        streaming right now, which is an answer, not an error.

        Example:
            >>> frames = client.sample_joints(duration_s=1.0, max_frames=10)
            >>> frames[-1].joints if frames else "arm idle"

        ``clock`` is a test seam (defaults to time.monotonic); leave it unset.
        Needs the realtime extra: pip install "makermodslab-sdk[realtime]".
        """
        import time

        return self._realtime().sample_joints(
            self.base_url,
            duration_s,
            max_frames=max_frames,
            clock=time.monotonic if clock is None else clock,
        )

    def stream_joints(self):
        """Stream ``JointData`` frames as an open-ended generator (blocks).

        The human/dashboard variant — it runs for as long as the socket stays
        open and yields ~20 frames/s while an arm is being driven. Agents
        should normally use ``sample_joints`` (bounded) instead.

        Example:
            >>> for frame in client.stream_joints():  # doctest: +SKIP
            ...     print(frame.joints)

        Needs the realtime extra: pip install "makermodslab-sdk[realtime]".
        """
        realtime = self._realtime()
        return realtime.events(self.base_url, kinds=realtime.JointData)
