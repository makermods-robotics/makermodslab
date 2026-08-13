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
"""Tests for makermodslab.utils.hf_auth — whoami caching."""

from __future__ import annotations

from unittest.mock import patch

from huggingface_hub.errors import LocalTokenNotFoundError


def test_invalidate_whoami_cache_clears_cached_value() -> None:
    from makermodslab.utils import hf_auth

    # cached_whoami() delegates to _WHOAMI_API.whoami(cache=True), which stores
    # results in _WHOAMI_API._whoami_cache keyed by token. Patching the whole
    # whoami() method would bypass the cache logic, so we patch _inner_whoami
    # (the actual HTTP call) instead — the real caching code then runs around it.
    # get_token() must return a truthy value so cached_whoami() doesn't short-circuit.
    with (
        patch("makermodslab.utils.hf_auth.get_token", return_value="hf_fake_token"),
        patch.object(hf_auth._WHOAMI_API, "_inner_whoami", return_value={"name": "alice"}) as spy,
    ):
        # Clear any pre-existing cache entry.
        hf_auth.invalidate_whoami_cache()

        first = hf_auth.cached_whoami()
        second = hf_auth.cached_whoami()
        assert first == {"name": "alice"}
        assert second == {"name": "alice"}
        # Cached: only one upstream call.
        assert spy.call_count == 1

        hf_auth.invalidate_whoami_cache()

        third = hf_auth.cached_whoami()
        assert third == {"name": "alice"}
        # After invalidation, the next call hits whoami again.
        assert spy.call_count == 2


def test_cached_whoami_fail_on_error_swallows_by_default() -> None:
    from makermodslab.utils import hf_auth

    with (
        patch("makermodslab.utils.hf_auth.get_token", return_value="hf_fake_token"),
        patch.object(hf_auth._WHOAMI_API, "whoami", side_effect=RuntimeError("boom")),
    ):
        assert hf_auth.cached_whoami() is None


def test_cached_whoami_fail_on_error_true_reraises_when_token_present() -> None:
    """A token IS present but the whoami call fails — fail_on_error=True must
    surface that as an exception instead of collapsing it into the same None
    as "no token", so a caller gating a Hub mutation on identity can fail
    closed on the transient case rather than treating it as logged-out."""
    from makermodslab.utils import hf_auth

    with (
        patch("makermodslab.utils.hf_auth.get_token", return_value="hf_fake_token"),
        patch.object(hf_auth._WHOAMI_API, "whoami", side_effect=RuntimeError("boom")),
    ):
        try:
            hf_auth.cached_whoami(fail_on_error=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected cached_whoami(fail_on_error=True) to re-raise")


def test_cached_whoami_fail_on_error_true_still_returns_none_without_a_token() -> None:
    """No token is the genuinely unauthenticated case — fail_on_error must not
    turn that into an error too, or a logged-out, never-uploaded rename would
    start failing on a Hub error it can do nothing about."""
    from makermodslab.utils import hf_auth

    with patch("makermodslab.utils.hf_auth.get_token", return_value=None):
        assert hf_auth.cached_whoami(fail_on_error=True) is None


def test_canonical_writable_namespace_matches_case_insensitively() -> None:
    from makermodslab.utils import hf_auth

    info = {"name": "makermods", "orgs": [{"name": "myorg", "roleInGroup": "write"}]}
    assert hf_auth.canonical_writable_namespace(info, "MyOrg") == "myorg"
    assert hf_auth.canonical_writable_namespace(info, "MAKERMODS") == "makermods"


def test_canonical_writable_namespace_returns_none_when_not_writable() -> None:
    from makermodslab.utils import hf_auth

    info = {"name": "makermods", "orgs": [{"name": "readonlyorg", "roleInGroup": "read"}]}
    assert hf_auth.canonical_writable_namespace(info, "readonlyorg") is None
    assert hf_auth.canonical_writable_namespace(info, "lerobot") is None


def test_handle_hf_auth_status_returns_dict() -> None:
    from makermodslab.utils import hf_auth

    # handle_hf_auth_status() calls the module-level whoami() directly.
    with patch("makermodslab.utils.hf_auth.whoami", return_value={"name": "alice", "orgs": []}):
        hf_auth.invalidate_whoami_cache()
        result = hf_auth.handle_hf_auth_status()
        assert isinstance(result, dict)
        # Real return shape: {"authenticated": bool, "username": ..., "orgs": ..., "login_command": ...}
        assert result["authenticated"] is True
        assert result["username"] == "alice"
        assert "orgs" in result
        assert "login_command" in result


def test_handle_hf_auth_status_writable_namespaces_role_filtered() -> None:
    from makermodslab.utils import hf_auth

    # whoami() returns orgs with a per-org "roleInGroup". Only admin/write orgs
    # (plus the user's own account) should land in writable_namespaces; a
    # read-only org is excluded. The existing "orgs" field must still list ALL
    # orgs regardless of role.
    fake = {
        "name": "alice",
        "orgs": [
            {"name": "acme-admin", "roleInGroup": "admin"},
            {"name": "acme-write", "roleInGroup": "write"},
            {"name": "acme-read", "roleInGroup": "read"},
            {"name": "acme-contrib", "roleInGroup": "contributor"},
        ],
    }
    with patch("makermodslab.utils.hf_auth.whoami", return_value=fake):
        hf_auth.invalidate_whoami_cache()
        result = hf_auth.handle_hf_auth_status()

    # Own account is always writable; admin + write orgs included; read and
    # contributor excluded (contributor cannot create repos in the org).
    assert result["writable_namespaces"] == ["alice", "acme-admin", "acme-write"]
    # The existing orgs field is unchanged: every org, role-agnostic.
    assert result["orgs"] == ["acme-admin", "acme-write", "acme-read", "acme-contrib"]


def test_handle_hf_auth_status_writable_namespaces_no_orgs() -> None:
    from makermodslab.utils import hf_auth

    with patch("makermodslab.utils.hf_auth.whoami", return_value={"name": "bob", "orgs": []}):
        hf_auth.invalidate_whoami_cache()
        result = hf_auth.handle_hf_auth_status()

    # No orgs: only the user's own namespace is writable.
    assert result["writable_namespaces"] == ["bob"]


def test_handle_hf_auth_status_writable_namespaces_unauthenticated() -> None:
    from makermodslab.utils import hf_auth

    with patch("makermodslab.utils.hf_auth.whoami", side_effect=LocalTokenNotFoundError("no token")):
        hf_auth.invalidate_whoami_cache()
        result = hf_auth.handle_hf_auth_status()

    assert result["authenticated"] is False
    assert result["writable_namespaces"] == []


def test_hf_hub_offline_detects_offline_env(monkeypatch) -> None:
    from makermodslab.utils import hf_auth

    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("HF_HUB_OFFLINE", truthy)
        assert hf_auth.hf_hub_offline() is True


def test_hf_hub_offline_false_when_unset_or_zero(monkeypatch) -> None:
    from makermodslab.utils import hf_auth

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    assert hf_auth.hf_hub_offline() is False
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert hf_auth.hf_hub_offline() is False
    monkeypatch.setenv("HF_HUB_OFFLINE", "")
    assert hf_auth.hf_hub_offline() is False


def test_hf_hub_offline_honours_legacy_transformers_var(monkeypatch) -> None:
    from makermodslab.utils import hf_auth

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    assert hf_auth.hf_hub_offline() is True
