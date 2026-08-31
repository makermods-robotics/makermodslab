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

"""Response schemas for the /api/v1 surface, one module per route tag.

Each model describes the dict a handler ALREADY returns — adding one to a
route's ``response_model=`` must never change the bytes on the wire. Two
FastAPI behaviors make an unfaithful model a silent regression: undeclared
keys are FILTERED out of the response, and declared-but-absent optional
fields are MATERIALIZED as null. Model every always-present key as required;
handle sometimes-absent keys with ``response_model_exclude_none=True`` on the
route (only safe when the dict never carries a legitimate null).
"""
