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

"""Response models for the "sfu" route group (the bundled LiveKit server's
token broker; see makermodslab/sfu.py). Shape authority: server.py
issue_sfu_token."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

__all__ = ["SfuTokenResponse"]


class SfuTokenResponse(BaseModel):
    """server.py issue_sfu_token — everything a Portal participant needs for
    `connect(url, token)`, plus the room/identity the token was scoped to
    (echoed back because the server fills in defaults the caller may have
    omitted) and the token's expiry as epoch seconds."""

    url: str
    token: str
    room: str
    identity: str
    role: Literal["robot", "operator", "viewer"]
    expires_at: int
