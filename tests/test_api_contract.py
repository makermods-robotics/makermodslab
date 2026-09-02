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


def _walk_routes(routes, prefix: str = ""):
    """Yield (prefixed_path, route) over a route table, traversing includes.

    FastAPI >= 0.138 registers include_router() calls lazily as _IncludedRouter
    entries that carry the prefix in their include_context instead of
    flattening prefixed APIRoute copies into app.routes; older versions
    flatten. Handle both so a lerobot-driven fastapi bump can't blind the
    contract tests. Plain starlette Routes (the /docs machinery) and Mounts
    (the SPA static files) fall through — they aren't API surface.
    """
    for r in routes:
        if isinstance(r, APIRoute | WebSocketRoute):
            yield prefix + r.path, r
        elif type(r).__name__ == "_IncludedRouter":
            sub_prefix = prefix + (r.include_context.prefix or "")
            yield from _walk_routes(r.original_router.routes, sub_prefix)


def api_surface() -> set[str]:
    """Every routable operation as '<METHOD> <path>' ('WS <path>' for websockets)."""
    from makermodslab.server import app

    pairs: set[str] = set()
    for path, r in _walk_routes(app.routes):
        if isinstance(r, APIRoute):
            for m in sorted(r.methods - {"HEAD", "OPTIONS"}):
                pairs.add(f"{m} {path}")
        else:
            pairs.add(f"WS {path}")
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


# Operations that exist ONLY under /api/v1. The flat mount is frozen
# (LEGACY_ROUTES above), so all new surface lands versioned from day one and
# registers here. NOT a ratchet: unlike the shrink-only lists, this set GROWS —
# each entry documents a deliberate v1-only addition, and the parity test
# checks every entry actually exists so retired surface can't linger.
V1_ONLY_ROUTES: frozenset[str] = frozenset(
    [
        # Node registry (multi-node): static/manual peer source.
        "GET /api/v1/nodes/{instance_id}/jobs/queue",
        "DELETE /api/v1/nodes/{instance_id}",
        "GET /api/v1/nodes",
        # Workload proxy: the peer's own typed jobs listing, passed through.
        "GET /api/v1/nodes/{instance_id}/jobs",
        "POST /api/v1/nodes",
        # Maker arm port detection (maker_ports.py). No flat mirror: the flat
        # surface only ever shrinks, and the SO-101's /identify-arm cannot
        # serve these — a Maker rig answers RobStride over CAN and FashionStar
        # over UART, neither of which a Feetech bus can open.
        "POST /api/v1/maker/identify-arm",
        "POST /api/v1/maker/probe-ports",
        # CAN crash recovery (can_recovery.py): de-energize a follower whose
        # process died holding torque. Not a session (see the module
        # docstring), and no flat mirror for the same only-shrinks reason.
        "POST /api/v1/arms/release-torque",
        # Peer-job drill-in proxies: record + incremental log tail (GET, any
        # HTTP failure = node.unreachable) and forwarded stop/delete (the
        # peer's own coded refusals pass through with THEIR status and body).
        "GET /api/v1/nodes/{instance_id}/jobs/{job_id}",
        "GET /api/v1/nodes/{instance_id}/jobs/{job_id}/logs",
        "POST /api/v1/nodes/{instance_id}/jobs/{job_id}/stop",
        "DELETE /api/v1/nodes/{instance_id}/jobs/{job_id}",
        # Environment proxies: the peer's own policy-extra status / install /
        # install-progress (the pip subprocess runs on the PEER), plus the
        # forwarded self-restart that makes a just-installed environment
        # reachable without a shell on the node.
        "GET /api/v1/nodes/{instance_id}/policy-extra/{policy_type}",
        "GET /api/v1/nodes/{instance_id}/policy-extra/{policy_type}/install-status",
        "POST /api/v1/nodes/{instance_id}/policy-extra/{policy_type}/install",
        "POST /api/v1/nodes/{instance_id}/restart",
        # Self-restart (the peer half of the proxy above): re-exec in place,
        # guarded by the busy matrix + the training queue.
        "POST /api/v1/system/restart",
        # Local training queue (PR #83): the machine's plan, in run order, and
        # the whole-list reorder that goes with it.
        "GET /api/v1/jobs/queue",
        "POST /api/v1/jobs/queue/reorder",
        # Sessions: identity + server-side robot resolution (sessions.py).
        "GET /api/v1/sessions/current",
        "POST /api/v1/sessions",
        "POST /api/v1/sessions/{session_id}/heartbeat",
        "POST /api/v1/sessions/{session_id}/stop",
        # Episode curation (PR #84): which episodes of a dataset a training
        # run is launched with. Read/replace only — never deletes an episode.
        "GET /api/v1/datasets/excluded-episodes",
        "PUT /api/v1/datasets/excluded-episodes",
    ]
)


def test_v1_mirrors_legacy_surface():
    """Every flat operation must also be mounted under /api/v1, and every v1
    operation is either the mirror of a flat one or registered in
    V1_ONLY_ROUTES.

    The dual mount serves the shipped frontend (flat) and the versioned surface
    (v1) from the same router; new surface skips the flat mount entirely.
    """
    surface = api_surface()
    legacy = {p for p in surface if not p.split(" ", 1)[1].startswith("/api/v1")}
    v1_full = {p for p in surface if p.split(" ", 1)[1].startswith("/api/v1/")}

    missing = V1_ONLY_ROUTES - v1_full
    assert not missing, f"V1_ONLY_ROUTES lists operations that don't exist: {sorted(missing)}"

    v1 = set()
    for op in v1_full - V1_ONLY_ROUTES:
        method, path = op.split(" ", 1)
        v1.add(f"{method} {path[len('/api/v1') :]}")
    assert v1 == legacy, (
        f"flat and /api/v1 surfaces drifted (new v1-only surface belongs in V1_ONLY_ROUTES).\n"
        f"  only flat: {sorted(legacy - v1)}\n"
        f"  only v1:   {sorted(v1 - legacy)}"
    )


# Routes whose v1 response is a file, a stream, or no body at all — never a
# JSON document, so a Pydantic response_model does not apply to them. Kept out
# of the shrinking list so "fully typed" can be reached without lying about
# these.
RESPONSE_MODEL_EXEMPT: frozenset[str] = frozenset(
    [
        # 204 No Content: a successful job delete has no body to model.
        "DELETE /api/v1/jobs/{job_id}",
        # 204 No Content: the forwarded peer-job delete mirrors the peer's own.
        "DELETE /api/v1/nodes/{instance_id}/jobs/{job_id}",
        # Raw Response: calibration JSON served as an attachment download.
        "GET /api/v1/calibration-configs/{device_type}/{config_name}/download",
        # StreamingResponse: MJPEG camera preview stream.
        "GET /api/v1/camera-preview/{index}",
        # FileResponse: episode MP4 (Range-request video playback).
        "GET /api/v1/datasets/episode-video",
        # Raw Response: checkpoint zip served as an attachment download.
        "GET /api/v1/jobs/{job_id}/checkpoints/{step}/download",
    ]
)

# Every v1 operation still served without a declared response_model. This list
# only ever SHRINKS: typing a route group deletes its entries in the same
# commit that adds the models (see the ratchet note at the top of the file).
# Only the /api/v1 mount is counted so entries stay stable while the flat
# mount is retired.
UNTYPED_V1_ROUTES: frozenset[str] = frozenset(
    [
        "DELETE /api/v1/calibration-configs/{device_type}/{config_name}",
        "DELETE /api/v1/robots/{name}",
        "GET /api/v1/auto-calibration-batch-status",
        "GET /api/v1/auto-calibration-status",
        "GET /api/v1/calibration-configs/{device_type}",
        "GET /api/v1/calibration-status",
        "GET /api/v1/inference-log",
        "GET /api/v1/inference-status",
        "GET /api/v1/recording-log",
        "GET /api/v1/recording-status",
        "GET /api/v1/replay-status",
        "GET /api/v1/robots",
        "GET /api/v1/robots/{name}",
        "GET /api/v1/teleoperation-status",
        "POST /api/v1/calibration-configs/{device_type}/upload",
        "POST /api/v1/calibration-configs/{device_type}/{config_name}/rename",
        "POST /api/v1/complete-calibration-step",
        "POST /api/v1/identify-arm",
        "POST /api/v1/inference-episode-stop",
        "POST /api/v1/inference-next-episode",
        "POST /api/v1/move-arm",
        "POST /api/v1/open-calibration-folder",
        "POST /api/v1/recording-exit-early",
        "POST /api/v1/recording-pause",
        "POST /api/v1/recording-rerecord-episode",
        "POST /api/v1/recording-resume",
        "POST /api/v1/robots/{name}",
        "POST /api/v1/robots/{name}/rename",
        "POST /api/v1/start-auto-calibration",
        "POST /api/v1/start-auto-calibration-batch",
        "POST /api/v1/start-calibration",
        "POST /api/v1/start-inference",
        "POST /api/v1/start-recording",
        "POST /api/v1/start-replay",
        "POST /api/v1/stop-auto-calibration",
        "POST /api/v1/stop-auto-calibration-batch",
        "POST /api/v1/stop-calibration",
        "POST /api/v1/stop-inference",
        "POST /api/v1/stop-recording",
        "POST /api/v1/stop-replay",
        "POST /api/v1/stop-teleoperation",
        "POST /api/v1/wiggle",
    ]
)


def _v1_json_operations():
    """(op, route) for every v1 APIRoute operation, '<METHOD> /api/v1/...'."""
    from makermodslab.server import app

    for path, r in _walk_routes(app.routes):
        if not isinstance(r, APIRoute) or not path.startswith("/api/v1/"):
            continue
        for m in sorted(r.methods - {"HEAD", "OPTIONS"}):
            yield f"{m} {path}", r


def test_untyped_v1_routes_ratchet():
    """Every v1 JSON operation must either declare a response_model or appear
    in UNTYPED_V1_ROUTES; file/stream routes live in RESPONSE_MODEL_EXEMPT."""
    ops = dict(_v1_json_operations())
    unknown_exempt = RESPONSE_MODEL_EXEMPT - ops.keys()
    assert not unknown_exempt, f"RESPONSE_MODEL_EXEMPT lists retired routes: {sorted(unknown_exempt)}"

    untyped = {op for op, r in ops.items() if op not in RESPONSE_MODEL_EXEMPT and r.response_model is None}
    added = untyped - UNTYPED_V1_ROUTES
    removed = UNTYPED_V1_ROUTES - untyped
    assert untyped == UNTYPED_V1_ROUTES, (
        f"Untyped v1 route surface changed.\n"
        f"  Added (new routes must declare a response_model): {sorted(added)}\n"
        f"  Removed (delete newly-typed routes from UNTYPED_V1_ROUTES): {sorted(removed)}"
    )


def test_v1_operation_ids_are_clean_and_unique():
    """v1 operation ids are the handler function names — the names an SDK
    generator will emit as client methods — and must therefore be unique."""
    from makermodslab.server import app

    spec = app.openapi()
    v1_ids: list[str] = []
    for path, ops in spec["paths"].items():
        if not path.startswith("/api/v1"):
            continue
        v1_ids.extend(
            op["operationId"] for op in ops.values() if isinstance(op, dict) and "operationId" in op
        )
    assert v1_ids, "no /api/v1 operations in the spec yet"
    dupes = {i for i in v1_ids if v1_ids.count(i) > 1}
    assert not dupes, f"duplicate v1 operation ids (rename the handler functions): {sorted(dupes)}"
    assert all("_api_v1_" not in i for i in v1_ids), (
        "v1 operation ids must come from the clean generator (function names), "
        "not FastAPI's default name_path_method mangling"
    )
