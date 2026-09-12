from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from makermodslab_sdk._transport import Transport


class Resource:
    """Base for the namespace clients (one per API tag); holds the shared Transport."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport


class SdkModel(BaseModel):
    """Base for all SDK response models: never reject server additions.

    ``extra="allow"`` everywhere — an older SDK against a newer server must
    keep working, and the extra keys stay readable on the object.
    """

    model_config = ConfigDict(extra="allow")
