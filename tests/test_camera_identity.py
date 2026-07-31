"""Unit tests for makerlab/camera_identity.py — uniqueID → cv2 index re-anchoring.

Pure-logic tests only: the AVFoundation enumerations (in-process and fresh
subprocess) are monkeypatched, never touched. What matters is the resolution
contract every camera-opening call site relies on:

  - no unique_id / identity unavailable  → fall back to the stored index
  - unique_id found                      → the enumerated index wins
  - unique_id verifiably absent          → None (callers must fail loudly)
"""

from __future__ import annotations

import pytest


def _enum(*pairs: tuple[int, str]) -> list[dict]:
    return [{"index": i, "name": "USB Camera", "unique_id": uid} for i, uid in pairs]


def test_resolve_in_enumeration_no_unique_id_trusts_fallback() -> None:
    from makerlab.camera_identity import resolve_in_enumeration

    assert resolve_in_enumeration(_enum((0, "0xa"), (1, "0xb")), None, 1) == 1
    assert resolve_in_enumeration(_enum((0, "0xa")), "", 0) == 0


def test_resolve_in_enumeration_unavailable_list_trusts_fallback() -> None:
    from makerlab.camera_identity import resolve_in_enumeration

    assert resolve_in_enumeration(None, "0xa", 3) == 3
    assert resolve_in_enumeration([], "0xa", 3) == 3


def test_resolve_in_enumeration_reanchors_to_enumerated_index() -> None:
    from makerlab.camera_identity import resolve_in_enumeration

    # The stored index says 0, but the device now enumerates at 1 (a twin
    # camera swap — the exact failure that recorded crossed front/wrist
    # streams): the uniqueID wins.
    assert resolve_in_enumeration(_enum((0, "0xb"), (1, "0xa")), "0xa", 0) == 1


def test_resolve_in_enumeration_absent_device_returns_none() -> None:
    from makerlab.camera_identity import resolve_in_enumeration

    assert resolve_in_enumeration(_enum((0, "0xa")), "0xdead", 0) is None


def test_resolve_cv2_index_identity_unavailable_trusts_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from makerlab import camera_identity

    monkeypatch.setattr(camera_identity, "list_cameras_in_process", lambda: None)
    assert camera_identity.resolve_cv2_index("0xa", 2) == 2


def test_resolve_cv2_index_reanchors_and_flags_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    from makerlab import camera_identity

    monkeypatch.setattr(camera_identity, "list_cameras_in_process", lambda: _enum((0, "0xb"), (1, "0xa")))
    assert camera_identity.resolve_cv2_index("0xa", 0) == 1
    assert camera_identity.resolve_cv2_index("0xdead", 0) is None
    # No unique_id short-circuits before enumeration.
    assert camera_identity.resolve_cv2_index(None, 5) == 5


def test_resolve_fresh_index_non_macos_never_enumerates(monkeypatch: pytest.MonkeyPatch) -> None:
    from makerlab import camera_identity

    def boom() -> list[dict]:
        raise AssertionError("fresh enumeration must not run off-macOS")

    monkeypatch.setattr(camera_identity, "list_cameras_fresh", boom)
    monkeypatch.setattr(camera_identity.platform, "system", lambda: "Linux")
    assert camera_identity.resolve_fresh_index("0xa", 4) == 4


def test_resolve_fresh_index_reanchors_on_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    from makerlab import camera_identity

    monkeypatch.setattr(camera_identity.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(camera_identity, "list_cameras_fresh", lambda: _enum((0, "0xb"), (1, "0xa")))
    assert camera_identity.resolve_fresh_index("0xa", 0) == 1
    assert camera_identity.resolve_fresh_index("0xdead", 0) is None
