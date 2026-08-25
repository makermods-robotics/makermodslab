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

"""API contract guards: the committed OpenAPI snapshot, and the route ratchets
that drive the /api/v1 migration.

Ratchet pattern: each allowlist below is asserted by EQUALITY, not subset.
Adding a route that widens a list fails the test (put new surface on /api/v1
instead); migrating a route off a list also fails until the entry is deleted
here — so every migration commit's diff includes shrinking its allowlist, and
finished progress can't silently regress.
"""

from __future__ import annotations

import json

from fastapi.routing import APIRoute
from starlette.routing import WebSocketRoute

from makermodslab.scripts.export_openapi import SNAPSHOT_PATH, generate_spec


def api_surface() -> set[str]:
    """Every routable operation as '<METHOD> <path>' ('WS <path>' for websockets).

    Plain starlette Routes (the /docs and /openapi.json machinery) and Mounts
    (the SPA static files) are deliberately excluded — they aren't API surface.
    """
    from makermodslab.server import app

    pairs: set[str] = set()
    for r in app.routes:
        if isinstance(r, APIRoute):
            for m in sorted(r.methods - {"HEAD", "OPTIONS"}):
                pairs.add(f"{m} {r.path}")
        elif isinstance(r, WebSocketRoute):
            pairs.add(f"WS {r.path}")
    return pairs


def test_openapi_snapshot_is_fresh():
    """docs/api/openapi.json must match the live app.

    Compares parsed documents (not text) so the assertion is about the API
    surface, with formatting left to the canonical writer. Regenerate with:
        uv run python -m makermodslab.scripts.export_openapi
    """
    assert SNAPSHOT_PATH.exists(), (
        "docs/api/openapi.json is missing; run: uv run python -m makermodslab.scripts.export_openapi"
    )
    committed = json.loads(SNAPSHOT_PATH.read_text())
    assert committed == generate_spec(), (
        "docs/api/openapi.json is stale; run: uv run python -m makermodslab.scripts.export_openapi"
    )


# The pre-/api/v1 surface, frozen at the start of the migration. This list only
# ever SHRINKS: new endpoints belong under /api/v1, and a legacy route retired
# from the flat mount must be deleted here in the same commit.
LEGACY_ROUTES: frozenset[str] = frozenset(
    [
        "DELETE /calibration-configs/{device_type}/{config_name}",
        "DELETE /datasets/custom",
        "DELETE /datasets/hide",
        "DELETE /jobs/hub/models/{repo_id:path}",
        "DELETE /jobs/{job_id}",
        "DELETE /models/custom",
        "DELETE /models/hide",
        "DELETE /robots/{name}",
        "GET /auto-calibration-batch-status",
        "GET /auto-calibration-status",
        "GET /available-cameras",
        "GET /available-ports",
        "GET /calibration-configs/{device_type}",
        "GET /calibration-configs/{device_type}/{config_name}/download",
        "GET /calibration-status",
        "GET /camera-preview/{index}",
        "GET /datasets",
        "GET /datasets/download-status",
        "GET /datasets/episode-joints",
        "GET /datasets/episode-video",
        "GET /datasets/episodes",
        "GET /datasets/hub-settings",
        "GET /datasets/hub-status",
        "GET /datasets/info",
        "GET /datasets/merge/status",
        "GET /health",
        "GET /hf-auth-status",
        "GET /inference-log",
        "GET /inference-status",
        "GET /jobs",
        "GET /jobs/hub",
        "GET /jobs/runners/hardware",
        "GET /jobs/{job_id}",
        "GET /jobs/{job_id}/checkpoints",
        "GET /jobs/{job_id}/checkpoints/{step}/download",
        "GET /jobs/{job_id}/checkpoints/{step}/policy-config",
        "GET /jobs/{job_id}/log-file",
        "GET /jobs/{job_id}/logs",
        "GET /jobs/{job_id}/metrics-history",
        "GET /models",
        "GET /models/download-status",
        "GET /models/info",
        "GET /policy-optimizer-defaults",
        "GET /recording-log",
        "GET /recording-status",
        "GET /replay-status",
        "GET /robot-port/{robot_type}",
        "GET /robots",
        "GET /robots/{name}",
        "GET /supply-voltage",
        "GET /system/policy-extra/{policy_type}",
        "GET /system/policy-extra/{policy_type}/install-status",
        "GET /system/training-extra",
        "GET /system/training-extra/install-status",
        "GET /system/update-check",
        "GET /system/wandb-extra",
        "GET /system/wandb-extra/install-status",
        "GET /teleoperation-status",
        "GET /upload-status",
        "POST /calibration-configs/{device_type}/upload",
        "POST /calibration-configs/{device_type}/{config_name}/rename",
        "POST /complete-calibration-step",
        "POST /datasets/custom",
        "POST /datasets/download",
        "POST /datasets/hide",
        "POST /datasets/import",
        "POST /datasets/merge",
        "POST /datasets/rename",
        "POST /datasets/tags",
        "POST /datasets/visibility",
        "POST /delete-dataset",
        "POST /hf-auth/login",
        "POST /identify-arm",
        "POST /inference-episode-stop",
        "POST /inference-next-episode",
        "POST /jobs/hub/jobs/{job_id}/dismiss",
        "POST /jobs/import",
        "POST /jobs/training",
        "POST /jobs/{job_id}/rename",
        "POST /jobs/{job_id}/stop",
        "POST /models/custom",
        "POST /models/delete",
        "POST /models/download",
        "POST /models/hide",
        "POST /models/import",
        "POST /models/upload",
        "POST /move-arm",
        "POST /open-calibration-folder",
        "POST /recording-exit-early",
        "POST /recording-pause",
        "POST /recording-rerecord-episode",
        "POST /recording-resume",
        "POST /robots/{name}",
        "POST /robots/{name}/rename",
        "POST /start-auto-calibration",
        "POST /start-auto-calibration-batch",
        "POST /start-calibration",
        "POST /start-inference",
        "POST /start-recording",
        "POST /start-replay",
        "POST /stop-auto-calibration",
        "POST /stop-auto-calibration-batch",
        "POST /stop-calibration",
        "POST /stop-inference",
        "POST /stop-recording",
        "POST /stop-replay",
        "POST /stop-teleoperation",
        "POST /system/policy-extra/{policy_type}/install",
        "POST /system/training-extra/install",
        "POST /system/update",
        "POST /system/wandb-extra/install",
        "POST /upload-dataset",
        "POST /wiggle",
        "WS /ws/joint-data",
    ]
)


def test_no_new_routes_outside_api_v1():
    legacy = {p for p in api_surface() if not p.split(" ", 1)[1].startswith("/api/v1")}
    added = legacy - LEGACY_ROUTES
    removed = LEGACY_ROUTES - legacy
    assert legacy == LEGACY_ROUTES, (
        f"Flat (non-/api/v1) route surface changed.\n"
        f"  Added (new endpoints belong under /api/v1): {sorted(added)}\n"
        f"  Removed (delete retired routes from LEGACY_ROUTES): {sorted(removed)}"
    )
