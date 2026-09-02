"""Provider-neutral split-host teleoperation primitives.

Nothing in this package opens hardware. A commissioned robot adapter is the
only object allowed to implement :class:`FollowerDriver`.
"""

from .authority import RemoteSessionAuthority, SessionAuthorityError
from .contracts import (
    PROTOCOL_VERSION,
    ActionContractError,
    ActionSample,
    SessionSpec,
    decode_action,
    encode_action,
)
from .executor import FollowerDriver, JointLimit, RemoteExecutor

__all__ = [
    "PROTOCOL_VERSION",
    "ActionContractError",
    "ActionSample",
    "FollowerDriver",
    "JointLimit",
    "RemoteExecutor",
    "RemoteSessionAuthority",
    "SessionAuthorityError",
    "SessionSpec",
    "decode_action",
    "encode_action",
]
