"""The ``system`` namespace: health/identity, auth, discovery, extras, updates.

Response models mirror makermodslab/schemas/system.py. SDK models are
``extra="allow"`` everywhere — an older SDK against a newer server must keep
working, and the extra keys stay readable on the object.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from makermodslab_sdk._operations import operation
from makermodslab_sdk.resources._base import Resource


class SdkModel(BaseModel):
    """Base for all SDK response models: never reject server additions."""

    model_config = ConfigDict(extra="allow")


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
