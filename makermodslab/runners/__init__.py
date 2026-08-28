# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Job runner implementations.

The local runner currently lives in app/jobs.py for historical reasons; this
package is for newer / out-of-process runners. They all satisfy the JobRunner
Protocol declared in app/jobs.py.
"""

from .hf_cloud import HfCloudJobRunner
from .lan_node import LanNodeJobRunner

__all__ = ["HfCloudJobRunner", "LanNodeJobRunner"]
