"""Second adversarial batch for PR 61: user-typed repo names, queue accounting,
and where errors actually surface."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.test_pr61_adversarial import (  # noqa: F401 - fixtures
    _hub_api,
    _no_card,
    _reset_model_cache,
    _seed_multi,
    registry,
)


def test_invalid_user_typed_repo_name_gets_a_useful_error(registry) -> None:
    """PublishToHubRow's repo-name Input is free text with no validation.
    A name the Hub can't accept should come back as an actionable 400, not a
    generic 'the Hub rejected the upload'."""
    from huggingface_hub.errors import HFValidationError

    from makermodslab.models import ModelError, upload_local_model

    _seed_multi(registry, "bad_name", ["100"])
    api = _hub_api()
    api.create_repo.side_effect = HFValidationError(
        "Repo id must use alphanumeric chars or '-', '_', '.'; got 'My Run!!'"
    )
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=api),
        pytest.raises(ModelError) as ei,
    ):
        upload_local_model("bad_name", repo_id="My Run!!", steps=[100])
    print("\nINVALID-NAME -> status:", ei.value.status, "| message:", ei.value.message)
    assert ei.value.status == 400


def test_manager_total_is_wrong_for_duplicate_steps(registry) -> None:
    """start() trusts len(steps); _resolve_upload_steps dedupes. The dialog
    renders 'Uploading N of TOTAL' from this."""
    from makermodslab.models import ModelUploadManager

    _seed_multi(registry, "dup", ["100", "200"])
    api = _hub_api()
    mgr = ModelUploadManager()
    with (
        patch("makermodslab.models.hf_hub_offline", return_value=False),
        patch("makermodslab.models.cached_whoami", return_value={"name": "user", "orgs": []}),
        patch("makermodslab.models.shared_hf_api", return_value=api),
        patch("makermodslab.models.metadata_update"),
        _no_card(),
    ):
        started = mgr.start("dup", None, [100, 100, 200])
        print("\nDUP start total:", mgr.get_status()["total"], "| started:", started["started"])
        mgr._thread.join(timeout=10)
    print("DUP final:", mgr.get_status())


def test_offline_publish_surfaces_only_via_status(registry) -> None:
    """POST /models/upload 200s even when the publish cannot possibly work."""
    from makermodslab.models import ModelUploadManager

    _seed_multi(registry, "off", ["100"])
    mgr = ModelUploadManager()
    with patch("makermodslab.models.hf_hub_offline", return_value=True):
        r = mgr.start("off", None, [100])
        mgr._thread.join(timeout=10)
    print("\nOFFLINE start returned:", r)
    print("OFFLINE status:", mgr.get_status())


def test_unknown_model_id_also_only_surfaces_via_status(registry) -> None:
    from makermodslab.models import ModelUploadManager

    mgr = ModelUploadManager()
    with patch("makermodslab.models.hf_hub_offline", return_value=False):
        r = mgr.start("does-not-exist", None, [100])
        mgr._thread.join(timeout=10)
    print("\nUNKNOWN start returned:", r)
    print("UNKNOWN status:", mgr.get_status())
