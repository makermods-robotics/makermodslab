"""Agent-first Python SDK for the MakerMods Lab robot server.

Quickstart:

    >>> from makermodslab_sdk import Client
    >>> client = Client("http://localhost:8000")
    >>> client.system.health().status
    'ok'

Everything hangs off ``Client``; namespaces mirror the server's API tags.
When a call fails, the exception text names the next call to make — read it.
"""

from makermodslab_sdk.client import Client, CompatibilityWarning
from makermodslab_sdk.errors import (
    ApiError,
    ConnectionFailedError,
    InvalidRequestError,
    MakerModsError,
    NotFoundError,
    RobotBusyError,
    SessionHeldError,
)
from makermodslab_sdk.resources.sessions import SessionLostError

__version__ = "0.0.1"

__all__ = [
    "ApiError",
    "Client",
    "CompatibilityWarning",
    "ConnectionFailedError",
    "InvalidRequestError",
    "MakerModsError",
    "NotFoundError",
    "RobotBusyError",
    "SessionHeldError",
    "SessionLostError",
    "__version__",
]
