# Copyright 2026 MakerMods. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""The ``flows`` layer: short, deterministic sequences built from SDK primitives.

Start here when the whole sequence fits; each method names the underlying
``client.system``, ``client.robots``, ``client.sessions``, ``client.jobs`` and
``client.models`` calls so an agent can drop down a level for a custom
workflow. Flows are client-side composition, not new server operations.
Timeouts and partial results are explicit; no flow retries a side-effecting
start automatically.
"""

from __future__ import annotations

from makermodslab_sdk.flows_calibration import CalibrationFlows
from makermodslab_sdk.flows_hardware import HardwareFlows
from makermodslab_sdk.flows_recording import RecordingFlows
from makermodslab_sdk.flows_remote_inference import RemoteInferenceFlows
from makermodslab_sdk.flows_training import TrainingFlows


class Flows(CalibrationFlows, HardwareFlows, RecordingFlows, RemoteInferenceFlows, TrainingFlows):
    """``client.flows`` — hardware context and common multi-call sequences."""
