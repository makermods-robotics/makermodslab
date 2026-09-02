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
"""Tests for makermodslab.camera_identity — uniqueID → cv2 index resolution.

The in-process AVFoundation enumeration is always patched: the host's real
camera list would make these machine-dependent, and on macOS enumerating
external cameras needs camera permission.
"""

from __future__ import annotations

import sys
import types

import pytest

import makermodslab.camera_identity as camera_identity
from makermodslab.camera_identity import (
    identify_cv2_index,
    list_cameras_in_process,
    resolve_cv2_index,
)

# A two-camera in-process view: "uid-A" sorted ahead of "uid-B", which is what
# a camera attached mid-session does to the device already streaming.
TWO_CAMERAS = [
    {"index": 0, "name": "Robot Cam", "unique_id": "uid-A"},
    {"index": 1, "name": "Robot Cam", "unique_id": "uid-B"},
]


@pytest.fixture
def enumeration(monkeypatch: pytest.MonkeyPatch):
    """Patch the in-process device list; returns a setter taking the list (or
    None for "identity unavailable" — non-macOS, PyObjC missing, query error)."""

    def _set(cameras: list[dict] | None) -> None:
        monkeypatch.setattr(camera_identity, "list_cameras_in_process", lambda: cameras)

    return _set


# ---------------------------------------------------------------------------
# resolve_cv2_index — index only (unchanged contract)
# ---------------------------------------------------------------------------


def test_resolve_returns_the_in_process_index_not_the_callers(enumeration) -> None:
    """The caller's index comes from a fresh-subprocess enumeration; this
    process may open the same device at a different number."""
    enumeration(TWO_CAMERAS)
    assert resolve_cv2_index("uid-B", 0) == 1


def test_resolve_returns_none_for_a_device_this_process_cannot_see(enumeration) -> None:
    enumeration(TWO_CAMERAS)
    assert resolve_cv2_index("uid-missing", 0) is None


def test_resolve_without_a_unique_id_trusts_the_index_without_enumerating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom() -> list[dict] | None:
        raise AssertionError("enumeration must not be needed to trust an index")

    monkeypatch.setattr(camera_identity, "list_cameras_in_process", _boom)
    assert resolve_cv2_index(None, 2) == 2
    assert resolve_cv2_index("", 2) == 2


def test_resolve_falls_back_to_the_index_when_identity_is_unavailable(enumeration) -> None:
    enumeration(None)
    assert resolve_cv2_index("uid-B", 3) == 3


# ---------------------------------------------------------------------------
# identify_cv2_index — index + the key a caller can cache the device under
# ---------------------------------------------------------------------------


def test_identify_returns_the_resolved_index_and_the_identity(enumeration) -> None:
    enumeration(TWO_CAMERAS)
    assert identify_cv2_index("uid-B", 0) == (1, "uid-B")


def test_identify_backfills_the_identity_when_the_caller_supplied_none(enumeration) -> None:
    """The uniqueID is optional on the wire (the frontend's BackendCameraStream
    takes `uniqueId?`). Without the backfill, one client would key a device by
    its uniqueID and another the SAME device by an int, and a shared,
    single-handle resource would be opened twice."""
    enumeration(TWO_CAMERAS)
    assert identify_cv2_index(None, 0) == (0, "uid-A")
    assert identify_cv2_index("", 1) == (1, "uid-B")


def test_identify_invents_no_identity_for_an_index_that_is_not_there(enumeration) -> None:
    enumeration(TWO_CAMERAS)
    assert identify_cv2_index(None, 7) == (7, None)


def test_identify_returns_no_identity_when_enumeration_is_unavailable(enumeration) -> None:
    """Nothing to key by: the caller falls back to the index, exactly as it
    behaved before identity existed. (These platforms have no live device
    list either, so indices do not renumber underneath the caller.)"""
    enumeration(None)
    assert identify_cv2_index("uid-B", 3) == (3, None)
    assert identify_cv2_index(None, 3) == (3, None)


def test_identify_returns_none_for_a_device_this_process_cannot_see(enumeration) -> None:
    """Same fail-loudly contract as resolve_cv2_index: never silently hand back
    whatever else now sits at the caller's index."""
    enumeration(TWO_CAMERAS)
    assert identify_cv2_index("uid-missing", 0) is None


# ---------------------------------------------------------------------------
# None vs [] — a deliberate fork, pinned here so it stays a decision
# ---------------------------------------------------------------------------


def test_an_empty_enumeration_is_not_the_same_answer_as_no_enumeration() -> None:
    """The distinction the whole module rests on, asserted side by side.

    ``[]`` means the enumeration ran and this machine has no cameras, so a
    requested device is definitively absent and the resolvers must fail loudly
    (None) rather than open whatever sits at the caller's index. ``None`` means
    the enumeration could not run at all, where the caller's index is the best
    answer available and is trusted. The integration branch's variant of
    resolve_in_enumeration collapses both into the index; if these two
    assertions ever have to change together, that collapse has been merged in
    by accident. See the MERGE NOTE in camera_identity.resolve_in_enumeration.
    """
    assert camera_identity.resolve_in_enumeration([], "uid-A", 3) is None
    assert camera_identity.resolve_in_enumeration(None, "uid-A", 3) == 3


def test_resolve_fails_loudly_when_the_machine_has_no_cameras(enumeration) -> None:
    enumeration([])
    assert resolve_cv2_index("uid-A", 0) is None
    assert resolve_cv2_index("uid-A", 7) is None


def test_resolve_trusts_the_index_when_the_enumeration_could_not_run(enumeration) -> None:
    enumeration(None)
    assert resolve_cv2_index("uid-A", 0) == 0
    assert resolve_cv2_index("uid-A", 7) == 7


def test_identify_carries_the_same_fork(enumeration) -> None:
    """identify_cv2_index inherits both sides: absent-on-an-empty-machine is a
    hard failure, un-enumerable is index-with-no-identity."""
    enumeration([])
    assert identify_cv2_index("uid-A", 0) is None
    assert identify_cv2_index(None, 0) == (0, None)  # nothing to backfill from
    enumeration(None)
    assert identify_cv2_index("uid-A", 0) == (0, None)


# ---------------------------------------------------------------------------
# list_cameras_in_process — the guarantee the fork above depends on: every
# "could not ask" path returns None, never an empty list.
#
# These drive the real function with a stand-in for the slice of PyObjC it
# uses, so no real AVFoundation call is ever made. That is unavoidably
# mock-shaped: it pins the CONTROL FLOW (which failures short-circuit, and
# that a discovery session is never run with an empty type list), not
# AVFoundation's real behavior.
# ---------------------------------------------------------------------------

ALL_TYPE_NAMES = {name: object() for name in camera_identity._AVF_DEVICE_TYPE_NAMES}


class FakeObjCError(Exception):
    """Stand-in for ``objc.error``."""


class FakeAVDevice:
    def __init__(self, unique_id: str, name: str) -> None:
        self._unique_id = unique_id
        self._name = name

    def uniqueID(self) -> str:  # noqa: N802 — Cocoa's camelCase API
        return self._unique_id

    def localizedName(self) -> str:  # noqa: N802 — Cocoa's camelCase API
        return self._name


class FakeAVFoundation:
    """The slice of PyObjC/AVFoundation that list_cameras_in_process() touches."""

    def __init__(self, bundle_loads: bool, constants: dict, devices_by_media: dict) -> None:
        self._bundle_loads = bundle_loads
        self._constants = constants
        self._devices_by_media = devices_by_media
        self.discovery_calls: list[tuple[list, str]] = []

    # --- Foundation.NSBundle -------------------------------------------------
    def bundleWithPath_(self, path: str):  # noqa: N802 — Cocoa's camelCase API
        return self

    def load(self) -> bool:
        return self._bundle_loads

    # --- objc ----------------------------------------------------------------
    def loadBundleVariables(self, bundle, out: dict, spec: list) -> None:  # noqa: N802
        ((name, _encoding),) = spec
        if name not in self._constants:
            raise FakeObjCError(name)
        out[name] = self._constants[name]

    def lookUpClass(self, name: str):  # noqa: N802 — Cocoa's camelCase API
        outer = self

        class _DiscoverySession:
            @staticmethod
            def discoverySessionWithDeviceTypes_mediaType_position_(  # noqa: N802
                device_types, media_type: str, position: int
            ):
                outer.discovery_calls.append((list(device_types), media_type))
                return types.SimpleNamespace(devices=lambda: outer._devices_by_media.get(media_type))

        return _DiscoverySession


@pytest.fixture
def fake_avfoundation(monkeypatch: pytest.MonkeyPatch):
    """Install the stand-in and pretend we're on macOS; returns the installer."""

    def _install(
        *,
        bundle_loads: bool = True,
        constants: dict | None = None,
        devices_by_media: dict | None = None,
    ) -> FakeAVFoundation:
        fake = FakeAVFoundation(
            bundle_loads,
            ALL_TYPE_NAMES if constants is None else constants,
            {"vide": [], "muxx": []} if devices_by_media is None else devices_by_media,
        )
        objc_module = types.ModuleType("objc")
        objc_module.error = FakeObjCError
        objc_module.loadBundleVariables = fake.loadBundleVariables
        objc_module.lookUpClass = fake.lookUpClass
        foundation_module = types.ModuleType("Foundation")
        foundation_module.NSBundle = fake
        monkeypatch.setitem(sys.modules, "objc", objc_module)
        monkeypatch.setitem(sys.modules, "Foundation", foundation_module)
        monkeypatch.setattr(camera_identity.platform, "system", lambda: "Darwin")
        return fake

    return _install


def test_enumeration_unavailable_when_the_framework_does_not_load(fake_avfoundation) -> None:
    """An unloaded framework resolves no device-type constants, so the natural
    outcome is "no devices" — which must not be reported as an empty machine."""
    fake = fake_avfoundation(bundle_loads=False)
    assert list_cameras_in_process() is None
    assert fake.discovery_calls == []


def test_enumeration_unavailable_when_no_device_type_constant_resolves(fake_avfoundation) -> None:
    """Every constant lookup failing means the lookup itself is broken (macOS
    renamed them?). A discovery session must never run with an empty type
    list: it can only match nothing, and that nothing would be a lie."""
    fake = fake_avfoundation(constants={})
    assert list_cameras_in_process() is None
    assert fake.discovery_calls == []


def test_enumeration_unavailable_when_discovery_answers_nothing(fake_avfoundation) -> None:
    """nil from every query is a failure to answer, not an answer of "none"."""
    fake_avfoundation(devices_by_media={"vide": None, "muxx": None})
    assert list_cameras_in_process() is None


def test_one_answering_query_is_enough_to_trust_an_empty_result(fake_avfoundation) -> None:
    """A single query answering (with an empty array) is a real enumeration."""
    fake_avfoundation(devices_by_media={"vide": [], "muxx": None})
    assert list_cameras_in_process() == []


def test_a_machine_with_no_cameras_reports_an_empty_list_not_none(fake_avfoundation) -> None:
    """The other half of the guarantee: a genuine empty machine must be
    distinguishable from a failed enumeration, or the fork above is moot."""
    fake_avfoundation()
    result = list_cameras_in_process()
    assert result is not None
    assert result == []


def test_devices_are_indexed_in_unique_id_order(fake_avfoundation) -> None:
    """cv2 assigns macOS camera indices by sorting on uniqueID; the returned
    index is that position, not discovery order."""
    fake_avfoundation(
        devices_by_media={
            "vide": [FakeAVDevice("uid-B", "Cam B"), FakeAVDevice("uid-A", "Cam A")],
            "muxx": [],
        }
    )
    assert list_cameras_in_process() == [
        {"index": 0, "name": "Cam A", "unique_id": "uid-A"},
        {"index": 1, "name": "Cam B", "unique_id": "uid-B"},
    ]


def test_a_missing_version_gated_constant_is_not_a_failure(fake_avfoundation) -> None:
    """Individual names are expected to miss — they are version-gated (macOS
    <14 has no AVCaptureDeviceTypeExternal). Only ALL of them missing is a
    broken lookup."""
    survivor = "AVCaptureDeviceTypeBuiltInWideAngleCamera"
    fake = fake_avfoundation(
        constants={survivor: ALL_TYPE_NAMES[survivor]},
        devices_by_media={"vide": [FakeAVDevice("uid-A", "Cam A")], "muxx": []},
    )
    assert list_cameras_in_process() == [{"index": 0, "name": "Cam A", "unique_id": "uid-A"}]
    assert fake.discovery_calls[0][0] == [ALL_TYPE_NAMES[survivor]]


def test_non_macos_has_no_in_process_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(camera_identity.platform, "system", lambda: "Linux")
    assert list_cameras_in_process() is None


# ---------------------------------------------------------------------------
# list_cameras_in_subprocess — the OTHER index space
#
# A fresh subprocess re-reads IOKit's live device state, so this is the
# ordering a newly spawned child (a preview probe, `lerobot-rollout`) will
# index against. It is NOT this process's ordering, which went stale at the
# first AVFoundation touch. The subprocess is always faked here: really
# spawning one would depend on the host's cameras and on camera permission.
# ---------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


@pytest.fixture
def fake_enum_subprocess(monkeypatch: pytest.MonkeyPatch):
    """Patch the enumeration subprocess; returns a setter taking either the
    stdout string to hand back or an exception instance to raise."""

    def _set(outcome) -> None:
        def _run(*args, **kwargs):
            if isinstance(outcome, BaseException):
                raise outcome
            return _FakeCompleted(outcome)

        monkeypatch.setattr(camera_identity.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(camera_identity.subprocess, "run", _run)

    return _set


def test_subprocess_enumeration_returns_the_parsed_device_list(fake_enum_subprocess) -> None:
    from makermodslab.camera_identity import list_cameras_in_subprocess

    fake_enum_subprocess(
        '[{"index": 0, "name": "Cam A", "unique_id": "uid-A"},'
        ' {"index": 1, "name": "Cam B", "unique_id": "uid-B"}]'
    )
    assert list_cameras_in_subprocess() == [
        {"index": 0, "name": "Cam A", "unique_id": "uid-A"},
        {"index": 1, "name": "Cam B", "unique_id": "uid-B"},
    ]


def test_subprocess_enumeration_reports_an_empty_machine_as_an_empty_list(
    fake_enum_subprocess,
) -> None:
    """An answered enumeration finding nothing is a fact about the machine, and
    callers act on it (refusing a start). It must not read as a failure."""
    from makermodslab.camera_identity import list_cameras_in_subprocess

    fake_enum_subprocess("[]")
    assert list_cameras_in_subprocess() == []


def test_subprocess_enumeration_failure_is_none_not_empty(fake_enum_subprocess) -> None:
    """A subprocess that never ran knows nothing about the device set. Reporting
    [] would make every caller conclude "no cameras attached"."""
    import subprocess as _subprocess

    from makermodslab.camera_identity import list_cameras_in_subprocess

    fake_enum_subprocess(_subprocess.SubprocessError("boom"))
    assert list_cameras_in_subprocess() is None


def test_subprocess_enumeration_invalid_json_is_none(fake_enum_subprocess) -> None:
    from makermodslab.camera_identity import list_cameras_in_subprocess

    fake_enum_subprocess("not json at all")
    assert list_cameras_in_subprocess() is None


def test_subprocess_enumeration_is_none_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    """The script is PyObjC/AVFoundation-only. Elsewhere there is no answer to
    give — and, again, "no answer" is not "no cameras"."""
    from makermodslab.camera_identity import list_cameras_in_subprocess

    monkeypatch.setattr(camera_identity.platform, "system", lambda: "Linux")
    called = []
    monkeypatch.setattr(
        camera_identity.subprocess, "run", lambda *a, **k: called.append(a) or _FakeCompleted("[]")
    )
    assert list_cameras_in_subprocess() is None
    assert called == []


# ---------------------------------------------------------------------------
# The enumeration SCRIPT itself, executed for real
#
# Every test above fakes `subprocess.run`, so none of them ever runs the script
# body — and the whole None-vs-[] contract lives inside that body. The child is
# the only thing that can tell "could not ask" from "asked, saw nothing", so a
# child that prints `[]` when the ask failed makes the parent's careful
# distinction cosmetic: `_verify_camera_identities` would read a PyObjC/TCC
# failure as a bare machine and refuse every inference start.
#
# These tests therefore spawn the REAL script with fake `objc`/`Foundation`
# modules shadowing PyObjC on PYTHONPATH, so each failure mode can be induced
# on any host (and without camera permission) while the code under test is the
# shipped script, character for character.
# ---------------------------------------------------------------------------

_FAKE_OBJC = '''
import json, os

class error(Exception):
    """Stands in for objc.error, which the script catches per-constant."""

def loadBundleVariables(bundle, into, names):
    wanted = [n for n in os.environ.get("FAKE_TYPES", "").split(",") if n]
    for name, _enc in names:
        if name in wanted:
            into[name] = "avf-type:" + name

class _Device:
    def __init__(self, raw):
        self._raw = raw

    def uniqueID(self):
        return self._raw["unique_id"]

    def localizedName(self):
        return self._raw["name"]

class _Session:
    def __init__(self, types, media_type):
        self._types = types
        self._media_type = media_type

    def devices(self):
        # No device types means a query that can only match nothing — which is
        # what real AVFoundation returns, not an error. Modelling it faithfully
        # is the point: the script cannot be allowed to pass that off as an
        # empty machine.
        if not self._types:
            return []
        # Muxed devices are a separate (usually empty) query; only the video
        # query is scripted, so a `null` here models exactly one query failing
        # to answer while the other answers honestly.
        raw = os.environ.get("FAKE_DEVICES_" + self._media_type, "[]")
        if raw == "null":
            return None
        return [_Device(d) for d in json.loads(raw)]

class _DiscoverySession:
    @staticmethod
    def discoverySessionWithDeviceTypes_mediaType_position_(types, media_type, position):
        return _Session(types, media_type)

def lookUpClass(name):
    return _DiscoverySession
'''

_FAKE_FOUNDATION = """
import os

class _Bundle:
    def load(self):
        return os.environ.get("FAKE_BUNDLE_LOAD", "1") == "1"

class NSBundle:
    @staticmethod
    def bundleWithPath_(path):
        return _Bundle()
"""


@pytest.fixture
def run_real_enum_script(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Run `list_cameras_in_subprocess` for real against a fake AVFoundation.

    Only PyObjC is faked (via PYTHONPATH shadowing); the script body, the spawn
    and the parent's parsing are all the shipped code."""
    import os
    import subprocess as _subprocess

    (tmp_path / "objc.py").write_text(_FAKE_OBJC)
    (tmp_path / "Foundation.py").write_text(_FAKE_FOUNDATION)

    def _set(**env) -> list[dict] | None:
        real_run = _subprocess.run

        def _run(cmd, **kwargs):
            kwargs["env"] = {
                **os.environ,
                "PYTHONPATH": str(tmp_path),
                **{k: str(v) for k, v in env.items()},
            }
            return real_run(cmd, **kwargs)

        monkeypatch.setattr(camera_identity.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(camera_identity.subprocess, "run", _run)
        return camera_identity.list_cameras_in_subprocess()

    return _set


def test_real_script_reports_a_genuinely_empty_machine_as_an_empty_list(run_real_enum_script) -> None:
    """The control: an ask that succeeded and saw nothing still returns [], so
    the failure signalling below cannot be blamed for over-triggering."""
    assert run_real_enum_script(FAKE_TYPES="AVCaptureDeviceTypeExternal") == []


def test_real_script_returns_the_devices_it_saw(run_real_enum_script) -> None:
    assert run_real_enum_script(
        FAKE_TYPES="AVCaptureDeviceTypeExternal",
        FAKE_DEVICES_vide='[{"name": "Wrist", "unique_id": "uid-B"},'
        ' {"name": "Front", "unique_id": "uid-A"}]',
    ) == [
        {"index": 0, "name": "Front", "unique_id": "uid-A"},
        {"index": 1, "name": "Wrist", "unique_id": "uid-B"},
    ]


def test_real_script_signals_failure_when_no_device_type_constant_resolves(
    run_real_enum_script,
) -> None:
    """A macOS point release renaming the AVCaptureDeviceType constants leaves
    the script with an empty type list, which can only ever match nothing. That
    nothing must not reach the parent as [] — `_verify_camera_identities` would
    then declare every bound camera missing and 400 every start, telling the
    user to re-select cameras that are sitting right there."""
    assert run_real_enum_script(FAKE_TYPES="") is None


def test_real_script_signals_failure_when_the_framework_does_not_load(run_real_enum_script) -> None:
    """Without AVFoundation loaded no constant resolves and no device can be
    found — a failure to ask, not an empty machine."""
    assert run_real_enum_script(FAKE_BUNDLE_LOAD="0", FAKE_TYPES="AVCaptureDeviceTypeExternal") is None


def test_real_script_signals_failure_when_no_discovery_query_answered(run_real_enum_script) -> None:
    """`devices()` returning nil is a query that did not answer. Flattening it
    with `or []` is exactly what turns a failed query into an empty machine."""
    assert (
        run_real_enum_script(
            FAKE_TYPES="AVCaptureDeviceTypeExternal",
            FAKE_DEVICES_vide="null",
            FAKE_DEVICES_muxx="null",
        )
        is None
    )


def test_real_script_trusts_one_answering_query(run_real_enum_script) -> None:
    """Mirrors list_cameras_in_process's `answered` flag: the muxed query going
    nil is normal, and one query answering is enough to trust the result."""
    assert run_real_enum_script(
        FAKE_TYPES="AVCaptureDeviceTypeExternal",
        FAKE_DEVICES_vide='[{"name": "Front", "unique_id": "uid-A"}]',
        FAKE_DEVICES_muxx="null",
    ) == [{"index": 0, "name": "Front", "unique_id": "uid-A"}]
