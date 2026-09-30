"""The remediation table stays honest: every key is a real server error code,
the codes agents hit most all have next steps, and family fallback works."""

from __future__ import annotations

from makermodslab_sdk.errors import (
    FAMILY_REMEDIATIONS,
    REMEDIATIONS,
    suggestion_for,
)


def server_codes() -> set[str]:
    # Server package is a test-only dependency: the SDK itself never imports it.
    from makermodslab.api_errors import ErrorCode

    return {str(code) for code in ErrorCode}


def test_every_remediation_key_is_a_real_server_code():
    unknown = set(REMEDIATIONS) - server_codes()
    assert unknown == set(), f"REMEDIATIONS keys that aren't server codes: {sorted(unknown)}"


def test_every_family_key_prefixes_a_real_server_code():
    codes = server_codes()
    for family in FAMILY_REMEDIATIONS:
        assert any(code.startswith(family + ".") for code in codes), family


def test_high_traffic_codes_all_carry_remediations():
    must_have = {
        "request.validation",
        "robot.not_found",
        "hardware.identity_mismatch",
        "hub.unauthenticated",
        "session.held",
        "session.not_found",
        "session.lease_expired",
        "job.not_found",
    }
    assert must_have <= set(REMEDIATIONS)


def test_family_fallback_and_exact_precedence():
    # Unlisted family member falls back to the family text …
    assert suggestion_for("robot.busy.inference") == FAMILY_REMEDIATIONS["robot.busy"]
    # … but an exact entry wins over its family.
    assert suggestion_for("robot.busy.releasing") == REMEDIATIONS["robot.busy.releasing"]
    assert suggestion_for("robot.busy.releasing") != FAMILY_REMEDIATIONS["robot.busy"]
    assert suggestion_for(None) is None
    assert suggestion_for("hub.upload_failed") is None
