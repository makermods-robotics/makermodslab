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
"""Resilience + performance of the three startup Hub listings.

Covers, with the Hub FULLY MOCKED (never a real network call):

* CRASH FIX — a GFW-killed TLS connection (``httpx.ConnectError``) from one
  author is swallowed and the listing returns the other authors' results,
  instead of propagating and 500ing the endpoint. Verified for both the
  ``/datasets`` (``list_user_datasets``) and ``/jobs/hub`` (model listing) paths.
* CALL REDUCTION — the ``/jobs/hub`` model listing makes ONE ``list_models`` call
  per author (previously two), filtering the ``lerobot`` tag client-side.
* TTL CACHE — a repeated listing within the TTL reuses the cached result (no
  second fan-out), and a mutation invalidates the cache so the next call
  re-fetches.
"""

from __future__ import annotations

import datetime as _dt
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

import makermodslab.datasets as ds
import makermodslab.server as server_mod


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _fake_dataset(repo_id: str, last_modified: _dt.datetime | None = None, private: bool = False):
    return SimpleNamespace(id=repo_id, last_modified=last_modified, private=private)


def _fake_model(repo_id: str, last_modified=None, private: bool = False, tags=None):
    return SimpleNamespace(id=repo_id, last_modified=last_modified, private=private, tags=list(tags or []))


def _whoami(name: str, orgs: list[str] | None = None) -> dict:
    return {"name": name, "orgs": [{"name": o} for o in (orgs or [])]}


# --------------------------------------------------------------------------- #
# CRASH FIX — list_user_datasets swallows a per-author connection error        #
# --------------------------------------------------------------------------- #
def test_list_user_datasets_swallows_connect_error_returns_other_authors() -> None:
    """A GFW-killed TLS connection (httpx.ConnectError) for ONE author must be
    caught so the listing still returns the OTHER authors' datasets — never a
    propagated exception that would 500 /datasets."""
    good = _fake_dataset("makermods/pick", _dt.datetime(2026, 7, 3, tzinfo=_dt.UTC))

    def _list_datasets(author=None, filter=None, limit=None):
        if author == "makermods":
            return [good]
        # The blocked-author case: the GFW cut the TLS connection mid-handshake.
        raise httpx.ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING]")

    api = MagicMock()
    api.list_datasets.side_effect = _list_datasets

    with (
        patch("makermodslab.datasets.cached_whoami", return_value=_whoami("makermods", orgs=["blockedorg"])),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        result = ds.list_user_datasets()  # must NOT raise

    assert [d["repo_id"] for d in result] == ["makermods/pick"]


def test_list_user_datasets_all_authors_fail_returns_empty() -> None:
    """If EVERY author's call is blocked, the listing degrades to empty rather
    than raising."""
    api = MagicMock()
    api.list_datasets.side_effect = httpx.ConnectError("blocked")

    with (
        patch("makermodslab.datasets.cached_whoami", return_value=_whoami("makermods", orgs=["org"])),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        assert ds.list_user_datasets() == []


def test_list_user_datasets_hfhub_http_error_still_swallowed() -> None:
    """The original guard (HfHubHTTPError) is preserved alongside the new
    transport-error guard."""
    from huggingface_hub.errors import HfHubHTTPError

    good = _fake_dataset("alice/aloha", _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC))

    def _list_datasets(author=None, filter=None, limit=None):
        if author == "alice":
            return [good]
        raise HfHubHTTPError("500 server error")

    api = MagicMock()
    api.list_datasets.side_effect = _list_datasets

    with (
        patch("makermodslab.datasets.cached_whoami", return_value=_whoami("alice", orgs=["org"])),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        result = ds.list_user_datasets()

    assert [d["repo_id"] for d in result] == ["alice/aloha"]


def test_list_user_datasets_dedups_across_authors() -> None:
    """Parallelization must preserve the original dedup semantics: a repo_id
    surfaced by two authors appears once."""
    shared = _fake_dataset("shared/ds", _dt.datetime(2026, 5, 1, tzinfo=_dt.UTC))
    api = MagicMock()
    api.list_datasets.return_value = [shared]

    with (
        patch("makermodslab.datasets.cached_whoami", return_value=_whoami("a", orgs=["b", "c"])),
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
    ):
        result = ds.list_user_datasets()

    assert [d["repo_id"] for d in result] == ["shared/ds"]


# --------------------------------------------------------------------------- #
# CRASH FIX + CALL REDUCTION — /jobs/hub model listing                         #
# --------------------------------------------------------------------------- #
def _patch_hub(monkeypatch, api, name="makermods", orgs=None):
    monkeypatch.setattr(server_mod, "cached_whoami", lambda: _whoami(name, orgs))
    monkeypatch.setattr(server_mod, "shared_hf_api", lambda: api)


def test_list_hub_jobs_swallows_model_connect_error(client, monkeypatch) -> None:
    """A GFW-killed connection while listing ONE author's models must not 500
    /jobs/hub; the other author's models still come through."""
    good = _fake_model(
        "makermods/act_run_2026-07-03_10-00-00",
        last_modified=_dt.datetime(2026, 7, 3, tzinfo=_dt.UTC),
        tags=["lerobot"],
    )

    def _list_models(author=None, filter=None, limit=None, expand=None):
        if author == "makermods":
            return [good]
        raise httpx.ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING]")

    api = MagicMock()
    api.list_jobs.return_value = []
    api.list_models.side_effect = _list_models
    _patch_hub(monkeypatch, api, name="makermods", orgs=["blockedorg"])

    resp = client.get("/jobs/hub")
    assert resp.status_code == 200  # not a 500
    assert [m["repo_id"] for m in resp.json()["models"]] == [good.id]


def test_list_hub_jobs_makes_one_model_call_per_author(client, monkeypatch) -> None:
    """CALL REDUCTION: exactly ONE list_models call per author (was two), and
    never with the redundant filter='lerobot' second pass."""
    m = _fake_model("makermods/act_run_2026-07-03_10-00-00", tags=["lerobot"])
    api = MagicMock()
    api.list_jobs.return_value = []
    api.list_models.return_value = [m]
    _patch_hub(monkeypatch, api, name="makermods", orgs=["org1", "org2"])

    resp = client.get("/jobs/hub")
    assert resp.status_code == 200

    # 3 authors (user + 2 orgs) -> exactly 3 calls, one per author.
    assert api.list_models.call_count == 3
    # No call passes filter='lerobot' — the tag is filtered client-side now.
    for call in api.list_models.call_args_list:
        assert call.kwargs.get("filter") is None


def test_list_hub_jobs_client_side_tag_and_suffix_filter(client, monkeypatch) -> None:
    """One unfiltered call, filtered client-side: a repo qualifies via the
    lerobot tag OR the run-repo timestamp suffix; a foreign personal model with
    neither is excluded."""
    tagged_no_suffix = _fake_model("makermods/some-model", tags=["lerobot"])
    run_repo_no_tag = _fake_model("makermods/smolvla_x_2026-07-03_09-15-57")
    personal = _fake_model("makermods/my-llm")  # no tag, no suffix
    api = MagicMock()
    api.list_jobs.return_value = []
    api.list_models.return_value = [tagged_no_suffix, run_repo_no_tag, personal]
    _patch_hub(monkeypatch, api, name="makermods")

    resp = client.get("/jobs/hub")
    ids = {m["repo_id"] for m in resp.json()["models"]}
    assert tagged_no_suffix.id in ids
    assert run_repo_no_tag.id in ids
    assert personal.id not in ids


# --------------------------------------------------------------------------- #
# TTL CACHE — datasets listing                                                 #
# --------------------------------------------------------------------------- #
def test_dataset_listing_cache_reuses_within_ttl(tmp_lerobot_home) -> None:
    """A second list_all_datasets() within the TTL reuses the cache — the
    underlying Hub fan-out runs only once."""
    ds.invalidate_dataset_listing_cache()
    calls = {"n": 0}

    def _hub():
        calls["n"] += 1
        return [{"repo_id": "makermods/pick", "last_modified": "2026-07-03T00:00:00+00:00", "private": False}]

    with patch("makermodslab.datasets.list_user_datasets", side_effect=_hub):
        first = ds.list_all_datasets()
        second = ds.list_all_datasets()

    assert calls["n"] == 1  # cached: the Hub listing ran once, not twice
    assert first == second


def test_dataset_listing_cache_invalidated_by_mutation(tmp_lerobot_home) -> None:
    """After a mutation invalidates the cache, the next list_all_datasets()
    re-fetches (and reflects the new Hub state immediately)."""
    ds.invalidate_dataset_listing_cache()
    responses = [
        [{"repo_id": "makermods/old", "last_modified": "2026-07-01T00:00:00+00:00", "private": False}],
        [{"repo_id": "makermods/new", "last_modified": "2026-07-03T00:00:00+00:00", "private": False}],
    ]
    calls = {"n": 0}

    def _hub():
        r = responses[min(calls["n"], len(responses) - 1)]
        calls["n"] += 1
        return r

    # Pins/hidden ids live outside tmp_lerobot_home, so the developer's real
    # saved-custom/hidden entries would leak into the exact-equality asserts.
    with (
        patch("makermodslab.datasets.list_user_datasets", side_effect=_hub),
        patch("makermodslab.datasets.get_saved_custom_datasets", return_value=[]),
        patch("makermodslab.datasets.get_hidden_datasets", return_value=set()),
    ):
        first = ds.list_all_datasets()
        assert [d["repo_id"] for d in first] == ["makermods/old"]

        # A mutation the user just performed must reflect immediately.
        ds.invalidate_dataset_listing_cache()
        second = ds.list_all_datasets()

    assert calls["n"] == 2
    assert [d["repo_id"] for d in second] == ["makermods/new"]


def test_set_dataset_visibility_invalidates_listing_cache() -> None:
    """set_dataset_visibility (settings agent's mutation) must invalidate the
    listing cache so a visibility flip shows up on the next /datasets load."""
    api = MagicMock()
    with (
        patch("makermodslab.datasets.shared_hf_api", return_value=api),
        patch("makermodslab.datasets.invalidate_dataset_listing_cache") as inv,
    ):
        ds.set_dataset_visibility("makermods/pick", private=True)
    inv.assert_called_once()


def test_set_dataset_tags_invalidates_listing_cache() -> None:
    """set_dataset_tags must invalidate the listing cache."""
    with (
        patch("makermodslab.datasets.metadata_update"),
        patch("makermodslab.datasets.invalidate_dataset_listing_cache") as inv,
    ):
        ds.set_dataset_tags("makermods/pick", ["robotics"])
    inv.assert_called_once()


# --------------------------------------------------------------------------- #
# TTL CACHE — /jobs/hub                                                        #
# --------------------------------------------------------------------------- #
def test_hub_jobs_cache_reuses_within_ttl(client, monkeypatch) -> None:
    """A second /jobs/hub within the TTL reuses the cache — no second fan-out to
    the Hub."""
    server_mod.invalidate_hub_jobs_cache()
    m = _fake_model("makermods/act_run_2026-07-03_10-00-00", tags=["lerobot"])
    api = MagicMock()
    api.list_jobs.return_value = []
    api.list_models.return_value = [m]
    _patch_hub(monkeypatch, api, name="makermods")

    client.get("/jobs/hub")
    first_calls = api.list_models.call_count
    client.get("/jobs/hub")
    assert api.list_models.call_count == first_calls  # second call served from cache


def test_hub_jobs_cache_invalidated_by_model_delete(client, monkeypatch) -> None:
    """Deleting a Hub model invalidates the /jobs/hub cache so the next listing
    re-fetches instead of showing the deleted repo."""
    server_mod.invalidate_hub_jobs_cache()
    m = _fake_model("makermods/act_run_2026-07-03_10-00-00", tags=["lerobot"])
    api = MagicMock()
    api.list_jobs.return_value = []
    api.list_models.return_value = [m]
    api.delete_repo.return_value = None
    _patch_hub(monkeypatch, api, name="makermods")

    client.get("/jobs/hub")
    before = api.list_models.call_count

    resp = client.delete("/jobs/hub/models/makermods/act_run_2026-07-03_10-00-00")
    assert resp.status_code == 200

    client.get("/jobs/hub")
    # The delete dropped the cache, so the second listing re-fanned-out.
    assert api.list_models.call_count > before


# --------------------------------------------------------------------------- #
# TIMEOUT — the OVERALL fan-out deadline bounds a hung author (both sites).     #
# The shared HfApi httpx client has timeout=None, so this budget is the only    #
# timeout in the stack: a blackholed connection must be abandoned + named,      #
# never left to stall the endpoint.                                             #
# --------------------------------------------------------------------------- #
def test_fan_out_model_authors_bounds_a_hung_author(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """server._fan_out_model_authors twin of the datasets test: with the budget
    shrunk to 0.2s, a fast author returns while a hung author (blocked on an
    Event never set during the call) is abandoned by the deadline — the call
    returns fast, carries only the fast author's result, and logs a warning
    naming the hung author."""
    monkeypatch.setattr(server_mod, "_HUB_MODEL_FANOUT_TIMEOUT_S", 0.2)

    # Never set DURING the call; released in the finally so the leaked worker
    # thread exits and pytest terminates cleanly.
    release = threading.Event()

    def call(author: str) -> list[str]:
        if author == "fast":
            return [f"model-for-{author}"]
        release.wait(timeout=30)  # the hung author
        return ["late"]

    try:
        start = time.monotonic()
        with caplog.at_level(logging.WARNING):
            result = server_mod._fan_out_model_authors(["fast", "hung"], call)
        elapsed = time.monotonic() - start

        # Bounded by the 0.2s budget, not the 30s the hung worker would take.
        assert elapsed < 3.0
        # Only the finished author's result survives.
        assert result == [["model-for-fast"]]
        # The timeout warning names the author that didn't finish.
        timeout_logs = [r.getMessage() for r in caplog.records if "exceeded" in r.getMessage()]
        assert timeout_logs, "expected a fan-out timeout warning"
        assert any("hung" in msg for msg in timeout_logs)
    finally:
        release.set()
