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
"""Listing shapes: `for x in result` must yield RECORDS, never field tuples.

A pydantic model iterates as ``(field_name, value)`` pairs, so a listing that
returns an envelope hands ``for robot in client.robots.list()`` a tuple and
fails at runtime with ``'tuple' object has no attribute 'name'``. An agent
reads the method name, writes the obvious loop, and only finds out on live
hardware. These tests make that shape a build-time contract instead.

Detection is dynamic, not a hand-kept list: any method returning a list, or
returning a model whose ONLY list field holds records, is a listing. A model
with two record lists (HubJobs) is ambiguous and deliberately excluded — no
single iteration contract exists for it.
"""

from __future__ import annotations

import typing

import pytest
from makermodslab_sdk.client import RESOURCE_CLASSES
from makermodslab_sdk.resources._base import SdkModel
from pydantic import BaseModel


def _records_field(model: type) -> tuple[str, object] | None:
    """(field, item type) for the single list field, or None when not a listing.

    Item type is returned whatever it is, so a listing whose records are raw
    dicts is still DETECTED and then fails the typed-records assertion — being
    untyped must not be a way to slip past this contract.
    """
    if not (isinstance(model, type) and issubclass(model, SdkModel)):
        return None
    hits = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if typing.get_origin(annotation) is not list:
            continue
        args = typing.get_args(annotation)
        hits.append((name, args[0] if args else None))
    return hits[0] if len(hits) == 1 else None


# Method names that PROMISE a collection. The trap is triggered by the name:
# an agent writes `for x in client.robots.list()` because "list" says so. A
# detail object that merely contains a list (datasets.info carrying .tasks) is
# not a listing and is deliberately absent.
COLLECTION_METHOD_NAMES = frozenset(
    {
        "list",
        "list_hub",
        "queue",
        "job_queue",
        "jobs",
        "checkpoints",
        "configs",
        "policies",
        "arms",
        "episodes",
    }
)


def _promises_a_collection(name: str, returns: object) -> bool:
    if typing.get_origin(returns) is list:
        return True
    if name in COLLECTION_METHOD_NAMES:
        return True
    # A type named ...List promises one too, whatever the method is called.
    return isinstance(returns, type) and returns.__name__.endswith("List")


def listing_methods() -> list[tuple[str, type, str, object]]:
    """(label, owning class, method name, resolved return type) per listing."""
    found = []
    for tag, cls in sorted(RESOURCE_CLASSES.items()):
        for name, member in sorted(vars(cls).items()):
            if name.startswith("_") or not callable(member):
                continue
            try:
                hints = typing.get_type_hints(member)
            except Exception:  # pragma: no cover - unresolvable forward ref
                continue
            returns = hints.get("return")
            if returns is None or not _promises_a_collection(name, returns):
                continue
            if typing.get_origin(returns) is list or _records_field(returns) is not None:
                found.append((f"{tag}.{name}", cls, name, returns))
    return found


def test_collection_name_register_has_no_stale_entries():
    live = {name for cls in RESOURCE_CLASSES.values() for name in vars(cls) if not name.startswith("_")}
    assert live >= COLLECTION_METHOD_NAMES, (
        f"names no method uses any more: {sorted(COLLECTION_METHOD_NAMES - live)}"
    )


def test_listings_were_actually_discovered():
    labels = {label for label, *_ in listing_methods()}
    # Spot-check the ones an agent reaches for first; a rename that hides a
    # listing from this sweep should fail here rather than silently pass.
    assert {"robots.list", "datasets.list", "jobs.list", "system.arms"} <= labels


@pytest.mark.parametrize(
    "label,cls,name,returns", listing_methods(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_listing_iterates_as_records(label, cls, name, returns):
    if typing.get_origin(returns) is list:
        (item,) = typing.get_args(returns)
        assert isinstance(item, type) and issubclass(item, SdkModel), (
            f"{label} returns list[{item}] — a listing must hand back typed records, "
            "not raw dicts, so attribute access works"
        )
        return
    field, item = _records_field(returns)
    # The invariant is behavioral: iteration must not be pydantic's field-pair
    # walk. Inheriting it from RecordList counts; redefining it counts.
    assert returns.__iter__ is not BaseModel.__iter__, (
        f"{label} returns the envelope {returns.__name__}, which inherits pydantic's "
        f"(field, value) iteration — `for x in client.{label}()` would yield tuples. "
        f"Give it list-like iteration over .{field}, or return a plain list."
    )
    assert isinstance(item, type) and issubclass(item, SdkModel), (
        f"{label} hands back {returns.__name__}.{field} as list[{item}] — records must be typed "
        'so attribute access works; a raw dict forces r["key"] and hides the shape'
    )
    sentinel = object()
    envelope = returns.model_construct(**{field: [sentinel]})
    assert list(envelope) == [sentinel], f"{label}: iterating {returns.__name__} must yield its .{field}"
    assert len(envelope) == 1, f"{label}: len({returns.__name__}) must count its .{field}"
    assert envelope[0] is sentinel, f"{label}: {returns.__name__}[0] must index its .{field}"


def test_card_shows_the_return_type_of_every_listing():
    """Tier 1 must say what a listing hands back; eliding it once cost a bug."""
    from makermodslab_sdk.docs import namespace_card

    for label, _cls, name, _returns in listing_methods():
        tag = label.split(".")[0]
        line = next(row for row in namespace_card(tag).splitlines() if row.startswith(f"- {name}("))
        assert " -> " in line, f"{label}'s card line hides its return type: {line!r}"
