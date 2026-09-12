"""Coverage ratchet: every tagged /api/v1 operation is implemented or planned.

Equality-asserted in both directions, house style (tests/test_api_contract.py):
implementing an operation without removing it from PLANNED fails, and so does
an operation disappearing from the snapshot. PLANNED only ever SHRINKS within
a namespace; operations newly landing in the snapshot (a staging merge) are
added here consciously in the merging commit. Each parallel track edits only
its own tag's frozenset — merges never collide.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from makermodslab_sdk import Client
from makermodslab_sdk.client import RESOURCE_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = json.loads((REPO_ROOT / "docs" / "api" / "openapi.json").read_text())
HTTP_METHODS = {"get", "post", "put", "delete", "patch"}

PLANNED: dict[str, frozenset[str]] = {
    "datasets": frozenset(),
    "inference": frozenset(),
    "jobs": frozenset(),
    "models": frozenset(),
    "nodes": frozenset(),
    "sessions": frozenset(),
    "system": frozenset(),
}


def tagged_operations() -> dict[str, set[str]]:
    """tag -> operationIds for every tagged /api/v1 route in the snapshot."""
    ops: dict[str, set[str]] = {}
    for path, methods in SNAPSHOT["paths"].items():
        if not path.startswith("/api/v1"):
            continue
        for method, op in methods.items():
            if method not in HTTP_METHODS:
                continue
            for tag in op.get("tags", []):
                ops.setdefault(tag, set()).add(op["operationId"])
    return ops


def implemented_operations(cls: type) -> set[str]:
    return {attr._operation_id for name in dir(cls) if hasattr((attr := getattr(cls, name)), "_operation_id")}


def test_every_tagged_operation_is_implemented_or_planned():
    ops = tagged_operations()
    assert set(PLANNED) == set(ops), (
        "tag set drifted — a new tag in the snapshot needs a PLANNED entry (and eventually a namespace)"
    )
    for tag in sorted(ops):
        implemented = implemented_operations(RESOURCE_CLASSES[tag]) if tag in RESOURCE_CLASSES else set()
        planned = set(PLANNED[tag])
        stale = planned - ops[tag]
        assert stale == set(), f"{tag}: PLANNED entries no longer in the snapshot: {sorted(stale)}"
        assert implemented == ops[tag] - planned, (
            f"{tag}: implemented ops must equal snapshot minus PLANNED.\n"
            f"  missing (in snapshot, not implemented, not planned): "
            f"{sorted(ops[tag] - planned - implemented)}\n"
            f"  shrink PLANNED for (implemented but still planned): "
            f"{sorted(implemented & planned)}\n"
            f"  unknown (implemented but not in snapshot): {sorted(implemented - ops[tag])}"
        )


def test_no_duplicate_operation_ids_across_namespaces():
    seen: dict[str, str] = {}
    for tag, cls in RESOURCE_CLASSES.items():
        for op in implemented_operations(cls):
            assert op not in seen, f"{op} implemented in both {seen[op]} and {tag}"
            seen[op] = tag


def test_client_exposes_every_registered_namespace():
    http = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        base_url="http://mock",
    )
    with Client("http://mock", http_client=http, check_compatibility=False) as client:
        for tag, cls in RESOURCE_CLASSES.items():
            assert isinstance(getattr(client, tag), cls), tag


# --- provisional (untagged) namespaces ---------------------------------------

# Surfaces over v1 routes the server hasn't tagged/typed yet (they predate the
# schema migration). Equality-pinned like everything else; when the server
# tags them, the namespace joins the tagged flow above and its entry here is
# DELETED in the same commit.
UNTAGGED_NAMESPACES: dict[str, frozenset[str]] = {
    "robots": frozenset(
        {
            "delete_robot",
            "get_robot",
            "get_robots",
            "rename_robot",
            "upsert_robot",
        }
    ),
}


def all_v1_operation_ids() -> set[str]:
    ops: set[str] = set()
    for path, methods in SNAPSHOT["paths"].items():
        if not path.startswith("/api/v1"):
            continue
        for method, op in methods.items():
            if method in HTTP_METHODS:
                ops.add(op["operationId"])
    return ops


def test_untagged_namespaces_are_registered_and_real():
    tags = set(tagged_operations())
    untagged_registered = set(RESOURCE_CLASSES) - tags
    assert untagged_registered == set(UNTAGGED_NAMESPACES), (
        "every RESOURCE_CLASSES key must be a snapshot tag or a registered untagged namespace"
    )
    v1_ops = all_v1_operation_ids()
    for namespace, expected in UNTAGGED_NAMESPACES.items():
        implemented = implemented_operations(RESOURCE_CLASSES[namespace])
        assert implemented == set(expected), namespace
        ghosts = expected - v1_ops
        assert ghosts == set(), f"{namespace}: ops not present anywhere in the snapshot: {sorted(ghosts)}"
