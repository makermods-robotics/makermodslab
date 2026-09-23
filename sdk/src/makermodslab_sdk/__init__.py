"""Agent-first Python SDK for the MakerMods Lab robot server.

Quickstart:

    >>> from makermodslab_sdk import Client
    >>> client = Client("http://localhost:8000")
    >>> client.system.health().status
    'ok'

Everything hangs off ``Client``; namespaces mirror the server's API tags.
When a call fails, the exception text names the next call to make — read it.
``client.docs()`` (or ``python -m makermodslab_sdk.docs``) is the built-in
progressive-disclosure manual: index -> namespace card -> method detail.
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
from makermodslab_sdk.flows_hardware import (
    ArmDiscoveryCapability,
    CameraAssignment,
    HardwareContext,
    HardwareNextAction,
    HardwareObservation,
    HardwareSectionError,
    PortAssignment,
    SavedRobotHardware,
)
from makermodslab_sdk.flows_recording import RecordingFlowError, RecordingFlowResult, RecordingFlowTimeout
from makermodslab_sdk.flows_remote_inference import (
    RemoteInferenceFlowError,
    RemoteInferenceRun,
    RemoteInferenceStartupTimeout,
)
from makermodslab_sdk.flows_training import PublishWaitTimeout, TrainAndPublishResult, TrainingFlowError
from makermodslab_sdk.resources._waiting import OperationFailedError, WaitTimeoutError
from makermodslab_sdk.resources.jobs import JobWaitTimeout, TrainingOptions
from makermodslab_sdk.resources.sessions import SessionLostError, SessionStoppedError, SessionWaitTimeout

__version__ = "0.0.1"

__all__ = [
    "ApiError",
    "ArmDiscoveryCapability",
    "CameraAssignment",
    "Client",
    "CompatibilityWarning",
    "ConnectionFailedError",
    "InvalidRequestError",
    "HardwareContext",
    "HardwareNextAction",
    "HardwareObservation",
    "HardwareSectionError",
    "JobWaitTimeout",
    "MakerModsError",
    "NotFoundError",
    "OperationFailedError",
    "PublishWaitTimeout",
    "PortAssignment",
    "RecordingFlowError",
    "RecordingFlowResult",
    "RecordingFlowTimeout",
    "RemoteInferenceFlowError",
    "RemoteInferenceRun",
    "RemoteInferenceStartupTimeout",
    "RobotBusyError",
    "SessionHeldError",
    "SessionLostError",
    "SessionStoppedError",
    "SessionWaitTimeout",
    "SavedRobotHardware",
    "TrainAndPublishResult",
    "TrainingOptions",
    "TrainingFlowError",
    "WaitTimeoutError",
    "__version__",
]
