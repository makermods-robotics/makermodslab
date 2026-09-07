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
"""Arm families and their registry. Importing this package registers the
three built-in families (SO-101, Maker, Metal) in that order.

Import registry for lookups; import ArmFamily to define a family.
"""

from . import registry
from .base import ArmFamily
from .maker import MAKER
from .metal import METAL
from .so101 import SO101

for _family in (SO101, MAKER, METAL):
    registry.register(_family)

__all__ = ["MAKER", "METAL", "SO101", "ArmFamily", "registry"]
