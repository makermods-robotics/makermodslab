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

"""/health as the node identity + capability document.

This payload doubles as the node-registry verify handshake (Phase 1): a peer
discovered via static config or the Tailscale API is confirmed by fetching its
/api/v1/health and reading version, instance_id, and capabilities.
"""

from __future__ import annotations

import re


def test_health_reports_identity_and_capabilities(client):
    body = client.get("/api/v1/health").json()
    # Legacy fields untouched — the frontend's reachability probe reads these.
    assert body["status"] == "ok"
    assert isinstance(body["message"], str)

    from makermodslab.__version__ import __version__

    assert body["version"] == __version__
    assert re.fullmatch(r"[0-9a-f]{32}", body["instance_id"])
    caps = body["capabilities"]
    assert isinstance(caps["serves_ui"], bool)
    assert caps["accepts_jobs"] is True


def test_instance_id_is_stable_across_requests(client):
    a = client.get("/api/v1/health").json()["instance_id"]
    b = client.get("/health").json()["instance_id"]  # same identity on both mounts
    assert a == b


def test_instance_id_persists_and_regenerates(tmp_path, monkeypatch):
    from makermodslab.utils import config as cfg

    path = tmp_path / "instance_id.txt"
    monkeypatch.setattr(cfg, "INSTANCE_ID_FILE", str(path))
    monkeypatch.setattr(cfg, "_instance_id_cache", None)

    first = cfg.get_instance_id()
    assert path.read_text().strip() == first
    assert cfg.get_instance_id() == first  # cached

    # A wiped cache dir (fresh install) mints a new identity.
    monkeypatch.setattr(cfg, "_instance_id_cache", None)
    path.unlink()
    assert cfg.get_instance_id() != first


def test_no_ui_env_disables_frontend(monkeypatch):
    from makermodslab import server

    monkeypatch.delenv("MAKERMODSLAB_NO_UI", raising=False)
    assert server.ui_enabled() == server.FRONTEND_DIST.exists()
    monkeypatch.setenv("MAKERMODSLAB_NO_UI", "1")
    assert server.ui_enabled() is False


def test_health_reports_gpu_when_torch_sees_one(client, monkeypatch):
    """capabilities.gpu comes from the torch probe (cached per process): a
    display-ready {name, vram} dict, or the key absent when there is no
    accelerator. torch is already resident in the server process (the lerobot
    import chain pulls it in), so the probe costs nothing."""
    from makermodslab.utils import system as sysmod

    monkeypatch.setattr(sysmod, "_gpu_cache", sysmod._GPU_UNPROBED)
    monkeypatch.setattr(
        sysmod, "_probe_gpu_uncached", lambda: {"name": "NVIDIA GeForce RTX 4090", "vram": "24GB"}
    )
    caps = client.get("/api/v1/health").json()["capabilities"]
    assert caps["gpu"] == {"name": "NVIDIA GeForce RTX 4090", "vram": "24GB"}


def test_health_omits_gpu_when_probe_finds_none(client, monkeypatch):
    from makermodslab.utils import system as sysmod

    monkeypatch.setattr(sysmod, "_gpu_cache", sysmod._GPU_UNPROBED)
    monkeypatch.setattr(sysmod, "_probe_gpu_uncached", lambda: None)
    caps = client.get("/api/v1/health").json()["capabilities"]
    assert "gpu" not in caps


def test_gpu_probe_is_cached(monkeypatch):
    from makermodslab.utils import system as sysmod

    calls = []
    monkeypatch.setattr(sysmod, "_gpu_cache", sysmod._GPU_UNPROBED)
    monkeypatch.setattr(
        sysmod, "_probe_gpu_uncached", lambda: calls.append(1) or {"name": "x", "vram": "1GB"}
    )
    assert sysmod.probe_gpu() == {"name": "x", "vram": "1GB"}
    assert sysmod.probe_gpu() == {"name": "x", "vram": "1GB"}
    assert len(calls) == 1
