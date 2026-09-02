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
"""Camera identity → cv2 index translation, in both of macOS's index spaces.

``cv2.VideoCapture(index)`` resolves the index against AVFoundation's
*in-process* device list, which is snapshotted the first time this process
touches AVFoundation and never refreshes — device-connection notifications
are delivered on a thread that needs an active NSRunLoop, which uvicorn
doesn't run. ``/available-cameras`` therefore enumerates in a *fresh
subprocess* (:func:`list_cameras_in_subprocess`), but that yields indices in
the fresh device order, which diverges from this process's order whenever
cameras were plugged/unplugged after startup. Opening by such an index then
silently hits the wrong physical device — e.g. the built-in webcam instead of
a robot camera, poisoning previews AND recordings. Which enumeration is
correct depends on WHO opens the camera: this process (previews →
:func:`identify_cv2_index`) or a freshly spawned child (`lerobot-rollout` →
:func:`list_cameras_in_subprocess`).

What links the two index spaces is AVFoundation's ``uniqueID`` — a better
handle than a bare index, but read what it actually is before trusting it.
Measured on this rig it is the USB **locationID** plus a per-model constant,
so it names a **(model, port) pair, not a physical unit**: move a camera to
another port and its uniqueID changes; plug a different camera of the same
model into that port and it inherits the old one. It is not a serial number,
and it does not follow the hardware. So it is exactly strong enough to catch
"the indices renumbered underneath us" and NOT strong enough to prove the
device in front of the lens is the one the user chose.

:func:`resolve_cv2_index` maps a camera's uniqueID to the index cv2 will
actually open *in this process*, by walking the same in-process device list
cv2 walks (video + muxed devices, uniqueID-sorted — mirrors OpenCV's
cap_avfoundation_mac.mm). A device attached after startup is invisible to
in-process AVFoundation entirely — and, worse, a camera replugged into the
SAME port comes back with the same uniqueID (it is derived from the port), so
it stays *present* in the stale list as a dead device object that opens "successfully" and then never produces a frame
(silently blank previews, reads that can block forever). Resolution returns
None only for the verifiably-absent case; callers must fail loudly (telling
the user to restart MakerMods Lab) rather than open whatever now sits at the
stale index.

Both failure modes disappear while :func:`pump_avfoundation_runloop` runs:
AVFoundation queues its device-cache updates on the main dispatch queue,
which only the MAIN thread's runloop drains, and uvicorn's asyncio loop —
though it runs on the main thread — never pumps it. Pumping briefly on a
timer keeps the in-process snapshot live (hardware-verified 2026-08-07 via
camera_runloop_experiment.py: a background-thread runloop does NOT refresh
the cache; pumping the main thread's does).

A live device list means indices genuinely renumber *at runtime*, so any
per-camera state a caller caches must be keyed by identity, not by index: a
handle opened for the device that was index 0 stays bound to that device after
another camera sorts ahead of it and becomes index 0. Callers that cache
therefore use :func:`identify_cv2_index`, which returns the index to open
*and* the key to file it under.
"""

import asyncio
import json
import logging
import platform
import subprocess
import sys
import threading

logger = logging.getLogger(__name__)

_AVF_DEVICE_TYPE_NAMES = (
    "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    "AVCaptureDeviceTypeExternalUnknown",  # macOS < 14
    "AVCaptureDeviceTypeExternal",  # macOS >= 14
    "AVCaptureDeviceTypeContinuityCamera",  # macOS >= 14
    "AVCaptureDeviceTypeDeskViewCamera",  # macOS >= 13
)


def list_cameras_in_process() -> list[dict] | None:
    """This process's camera list, in cv2 open order.

    Returns ``[{"index", "name", "unique_id"}, ...]`` reflecting the same
    (possibly stale) AVFoundation state cv2.VideoCapture resolves indices
    against.

    None and ``[]`` are **different answers**, and callers rely on that:

    - None — the enumeration could not be performed at all (non-macOS, PyObjC
      unavailable, the framework failing to load, no device-type constant
      resolving, AVFoundation answering nothing). Nothing is known about the
      device set, so callers fall back to trusting the index they were given.
    - ``[]`` — the enumeration *was* performed and found zero cameras. A
      requested device is then definitively absent, and callers must fail
      loudly rather than open whatever sits at the index.

    Keeping those apart takes deliberate care, because the natural failure of
    almost every step here is "no devices found" rather than an exception: an
    unloaded framework resolves no device-type constants, and a discovery
    session handed an empty type list matches nothing by construction. Each
    such step therefore returns None explicitly instead of falling through to
    an empty result that would be indistinguishable from a real, empty
    machine. (Camera permission is the one case not caught here: a process
    denied macOS camera access enumerates only built-ins, so a Mac without a
    built-in camera reports ``[]`` truthfully — it asked, and AVFoundation
    showed it nothing. It cannot open those devices through cv2 either, so
    failing loudly stays the right answer.)
    """
    if platform.system() != "Darwin":
        return None
    try:
        import objc
        from Foundation import NSBundle

        bundle = NSBundle.bundleWithPath_("/System/Library/Frameworks/AVFoundation.framework")
        if not bundle.load():
            # Without the framework loaded no device-type constant below can
            # resolve, and the discovery session would report an empty device
            # set. That is a failure to ask, not an empty machine.
            logger.warning("AVFoundation framework did not load — camera identity unavailable")
            return None
        types = []
        for name in _AVF_DEVICE_TYPE_NAMES:
            loaded = {}
            try:
                objc.loadBundleVariables(bundle, loaded, [(name, b"@")])
            except objc.error:
                continue
            if loaded.get(name) is not None:
                types.append(loaded[name])
        if not types:
            # Never run a discovery session with an empty type list: it can
            # only match nothing, and that nothing would be a lie. Individual
            # names are expected to miss (they are version-gated); all of them
            # missing means the lookup itself is broken.
            logger.warning(
                "No AVFoundation camera device-type constants resolved (renamed by macOS?) — "
                "camera identity unavailable"
            )
            return None
        cls = objc.lookUpClass("AVCaptureDeviceDiscoverySession")
        devs = []
        answered = False
        for media_type in ("vide", "muxx"):
            session = cls.discoverySessionWithDeviceTypes_mediaType_position_(types, media_type, 0)
            found = session.devices()
            # nil (a query that did not answer) is tracked apart from an empty
            # array (a query that answered "none"); flattening both with
            # `or []` is what would let a failed query read as an empty
            # machine. One query answering is enough to trust the result.
            if found is None:
                continue
            answered = True
            devs.extend(found)
        if not answered:
            logger.warning("AVFoundation discovery answered nothing — camera identity unavailable")
            return None
        devs.sort(key=lambda d: d.uniqueID())
        return [
            {"index": i, "name": str(d.localizedName()), "unique_id": str(d.uniqueID())}
            for i, d in enumerate(devs)
        ]
    except Exception as e:
        logger.warning("In-process AVFoundation enumeration failed: %s", e)
        return None


# Runs in a fresh Python — see list_cameras_in_subprocess for why.
# Mirrors OpenCV's macOS enumeration: video + muxed devices sorted by
# uniqueID (cap_avfoundation_mac.mm), so the returned index matches what
# cv2.VideoCapture will open.
_AVF_ENUM_SCRIPT = """
import json, objc
from Foundation import NSBundle
bundle = NSBundle.bundleWithPath_("/System/Library/Frameworks/AVFoundation.framework")
bundle.load()
types = []
for name in (
    "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    "AVCaptureDeviceTypeExternalUnknown",   # macOS < 14
    "AVCaptureDeviceTypeExternal",          # macOS >= 14
    "AVCaptureDeviceTypeContinuityCamera",  # macOS >= 14
    "AVCaptureDeviceTypeDeskViewCamera",    # macOS >= 13
):
    loaded = {}
    try:
        objc.loadBundleVariables(bundle, loaded, [(name, b"@")])
    except objc.error:
        continue
    if loaded.get(name) is not None:
        types.append(loaded[name])
cls = objc.lookUpClass("AVCaptureDeviceDiscoverySession")
devs = []
for mt in ("vide", "muxx"):
    devs.extend(cls.discoverySessionWithDeviceTypes_mediaType_position_(types, mt, 0).devices() or [])
devs.sort(key=lambda d: d.uniqueID())
print(json.dumps([
    {"index": i, "name": str(d.localizedName()), "unique_id": str(d.uniqueID())}
    for i, d in enumerate(devs)
]))
"""


def list_cameras_in_subprocess() -> list[dict] | None:
    """A FRESHLY spawned process's camera list, in cv2 open order (macOS).

    The other index space from :func:`list_cameras_in_process`, and the two
    diverge the moment a camera is plugged or unplugged. AVFoundation's
    in-process device cache doesn't refresh on USB hotplug — both the
    deprecated ``+devicesWithMediaType:`` and a long-lived
    ``AVCaptureDeviceDiscoverySession`` go stale, because device-connection
    notifications are delivered via ``NSNotificationCenter`` on a thread that
    needs an active ``NSRunLoop``, which uvicorn workers don't run. A fresh
    subprocess re-initializes AVFoundation, which reads IOKit's live device
    state at startup.

    This is therefore the ordering a NEWLY SPAWNED CHILD will index against:
    ``/available-cameras`` (so the list the user picks from is live), and any
    camera a child process opens — notably the ``lerobot-rollout`` subprocess
    inference spawns. Anything opened *inside the server process* must use
    :func:`identify_cv2_index` instead; mixing the two silently opens the
    wrong physical device.

    None and ``[]`` are **different answers**, exactly as in
    :func:`list_cameras_in_process`:

    - None — the enumeration could not be performed (non-macOS, the subprocess
      failing to run, unparseable output). Nothing is known about the device
      set, so callers must fall back to trusting the index they were given.
    - ``[]`` — the enumeration ran and found zero cameras; a requested device
      is then definitively absent.

    Collapsing the two would turn a PyObjC hiccup into "no cameras attached",
    which for the verification caller means refusing every start.
    """
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            [sys.executable, "-c", _AVF_ENUM_SCRIPT],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("AVFoundation enumeration subprocess failed: %s", e)
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        logger.warning("AVFoundation enumeration returned invalid JSON: %s", e)
        return None


def resolve_in_enumeration(
    cameras: list[dict] | None, unique_id: str | None, fallback_index: int
) -> int | None:
    """Position of ``unique_id`` in an in-process enumeration, else None.

    Shared by :func:`resolve_cv2_index` and :func:`identify_cv2_index` so both
    report an index shift, and a verifiably-absent device, identically. Both
    inputs are nullable so a caller may hand an enumeration straight through
    without pre-checking it.
    """
    # MERGE NOTE — the integration branch's variant of this function guards
    # `if not unique_id or not cameras`, which sends an EMPTY list to
    # fallback_index as well. This branch guards only `cameras is None`, and
    # the difference is deliberate: list_cameras_in_process() guarantees that
    # None and [] are different answers (see its docstring — every "could not
    # ask" path returns None explicitly, precisely so that [] can be trusted).
    #   None -> nothing is known about the device set; the caller's index is
    #           the best answer available, so trust it.
    #   []   -> the enumeration ran and found no cameras, so the requested
    #           device is definitively absent. Return None and let the caller
    #           fail loudly, which is this module's whole contract.
    # Collapsing [] into fallback_index would open whatever unrelated device
    # sits at that number and turn the preview endpoint's honest 503 into a
    # wrong-camera stream. Whoever resolves this conflict: the two bodies are
    # not interchangeable, and this branch's None/[] guarantee is what makes
    # falling through on [] correct here.
    if not unique_id or cameras is None:
        return fallback_index
    for cam in cameras:
        if cam.get("unique_id") == unique_id:
            if cam["index"] != fallback_index:
                logger.info(
                    "Camera %s: in-process cv2 index %d differs from enumerated index %d "
                    "(device set changed since MakerMods Lab started)",
                    unique_id,
                    cam["index"],
                    fallback_index,
                )
            return cam["index"]
    logger.warning(
        "Camera %s is not visible to this process (attached after MakerMods Lab started?) — "
        "restart MakerMods Lab to use it.",
        unique_id,
    )
    return None


def resolve_cv2_index(unique_id: str | None, fallback_index: int) -> int | None:
    """The index THIS process's cv2 opens for ``unique_id``.

    - No unique_id, or identity unavailable (non-macOS / query failure):
      ``fallback_index`` — legacy trust-the-index behavior.
    - unique_id found in the in-process list: its position there (which can
      differ from the fresh-subprocess enumeration index the caller has).
    - unique_id verifiably absent: None — the device attached after this
      process started; only a restart makes it reachable. Callers must error
      out instead of opening a different physical camera.

    Callers that additionally *cache* something per camera want
    :func:`identify_cv2_index`, which returns the same index plus the identity
    to key that cache by — an index alone is not a stable cache key.
    """
    if not unique_id:
        return fallback_index
    cameras = list_cameras_in_process()
    if cameras is None:
        return fallback_index
    return resolve_in_enumeration(cameras, unique_id, fallback_index)


def identify_cv2_index(unique_id: str | None, fallback_index: int) -> tuple[int, str | None] | None:
    """``(index to open, identity key)`` for a camera — :func:`resolve_cv2_index` plus identity.

    Same contract as :func:`resolve_cv2_index` for the index, including the
    None return for a verifiably-absent device (callers must fail loudly). The
    second element is the camera's uniqueID, or None when this process cannot
    establish identity at all (non-macOS / PyObjC missing / query failure) —
    the caller then has nothing better than the index to key by.

    A caller who supplies no ``unique_id`` gets one **backfilled** from the
    in-process enumeration. That is load-bearing, not cosmetic: the identity is
    optional on the wire (the frontend's ``BackendCameraStream`` takes
    ``uniqueId?``), so without the backfill one client would key a device by
    its uniqueID while another keyed the *same* device by an int, and a shared,
    single-handle resource would be opened twice.
    """
    cameras = list_cameras_in_process()
    if cameras is None:
        # No identity to be had: legacy trust-the-index behavior, and the
        # caller keys by the index — exactly what it did before identity
        # existed. On these platforms the in-process device list is not
        # live-refreshed either, so indices do not renumber underneath us.
        return fallback_index, None
    if not unique_id:
        for cam in cameras:
            if cam["index"] == fallback_index:
                return fallback_index, cam["unique_id"]
        # Nothing at that index in this process's view. Don't invent an
        # identity; the open itself will fail loudly enough.
        return fallback_index, None
    resolved = resolve_in_enumeration(cameras, unique_id, fallback_index)
    if resolved is None:
        return None
    return resolved, unique_id


async def pump_avfoundation_runloop(interval_s: float = 0.5, pump_s: float = 0.05) -> None:
    """Keep this process's AVFoundation camera snapshot live (macOS).

    Run as an asyncio background task from server startup. It must live in the
    main thread's event loop — AVFoundation delivers device connect/disconnect
    cache updates via the main dispatch queue, and only the main thread's
    runloop drains that queue (a background thread's runloop verifiably does
    not — see the module docstring). Each cycle blocks the loop for at most
    ``pump_s`` (~50 ms) every ``interval_s``, then yields back to asyncio.

    No-op on non-macOS or when PyObjC is unavailable; exits with a warning if
    scheduled off the main thread; cancels cleanly on shutdown.
    """
    if platform.system() != "Darwin":
        return
    if threading.current_thread() is not threading.main_thread():
        logger.warning(
            "AVFoundation runloop pump scheduled off the main thread — "
            "hotplug refresh disabled (cache updates only drain on the main runloop)"
        )
        return
    try:
        import objc
        from Foundation import NSDate, NSDefaultRunLoopMode, NSRunLoop
    except ImportError:
        logger.warning("PyObjC unavailable — AVFoundation hotplug refresh disabled")
        return
    # First AVFoundation touch from the main thread, so the snapshot exists
    # before the first pump and is anchored where the updates get drained.
    list_cameras_in_process()
    runloop = NSRunLoop.currentRunLoop()
    logger.info("AVFoundation runloop pump running — camera hotplug/replug is live")
    while True:
        with objc.autorelease_pool():
            runloop.runMode_beforeDate_(NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(pump_s))
        await asyncio.sleep(interval_s)
