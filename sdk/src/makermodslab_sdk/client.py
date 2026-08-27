from __future__ import annotations

import warnings

import httpx

from makermodslab_sdk._transport import DEFAULT_TIMEOUT, Transport
from makermodslab_sdk.errors import ApiError
from makermodslab_sdk.resources import DatasetsResource, ModelsResource, Resource, SystemResource

# The oldest server this SDK release is known to work against (the app's
# version from the repo-root pyproject at the time the SDK was cut).
MIN_SUPPORTED_SERVER_VERSION = (0, 1, 0)

# tag -> namespace class. ONE line per namespace, kept alphabetical — parallel
# tracks each add exactly their own line, so merges never collide here.
RESOURCE_CLASSES: dict[str, type[Resource]] = {
    "datasets": DatasetsResource,
    "models": ModelsResource,
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

    Namespaces mirror the API tags — ``client.system`` today; ``datasets``,
    ``models``, ``jobs``, ``nodes`` and ``sessions`` arrive with their tracks.
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
        self.models = ModelsResource(self._transport)
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
