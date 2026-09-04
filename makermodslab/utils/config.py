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

import contextlib
import json
import logging
import os
import platform
import re
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

RobotSide = Literal["leader", "follower"]

# ---------------------------------------------------------------------------
# Where MakerMods Lab keeps ITS OWN state.
#
# lerobot owns ``~/.cache/huggingface/lerobot``: datasets, models, the
# calibration libraries its device classes read, and training outputs (local
# policies live there because they ARE models). Everything that is MakerMods
# Lab's rather than lerobot's — robot records, saved ports, UI bookkeeping,
# node identity, the bimanual staging area, and (next) extensions — lives under
# this root instead, so a user finds the app's files under the app's name and
# a lerobot cache wipe does not take the robot setup with it.
#
# ``MAKERMODSLAB_HOME`` overrides the root (containers, a shared machine, and
# the test suite, which points it at a tmp dir before anything is imported).
# An override also switches OFF the legacy migration below: whoever set it is
# pointing at a place they chose, and silently moving old files there would
# be a surprise — the test suite relies on exactly that to never touch a
# developer's real state.
# ---------------------------------------------------------------------------
LEGACY_STATE_ROOT = os.path.expanduser("~/.cache/huggingface/lerobot")


def resolve_makermodslab_home(env: Mapping[str, str] | None = None) -> str:
    """The MakerMods Lab state root: ``$MAKERMODSLAB_HOME`` or ``~/.makermods/makermodslab``."""
    env = os.environ if env is None else env
    override = env.get("MAKERMODSLAB_HOME")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.expanduser(os.path.join("~", ".makermods", "makermodslab"))


MAKERMODSLAB_HOME = resolve_makermodslab_home()
HOME_IS_OVERRIDDEN = bool(os.environ.get("MAKERMODSLAB_HOME"))

# Define the calibration config paths (shared between features). These stay
# under lerobot's cache: lerobot's device classes read their calibration from
# there, and the library IS lerobot calibration data.
CALIBRATION_BASE_PATH_TELEOP = os.path.expanduser("~/.cache/huggingface/lerobot/calibration/teleoperators")
CALIBRATION_BASE_PATH_ROBOTS = os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots")
LEADER_CONFIG_PATH = os.path.join(CALIBRATION_BASE_PATH_TELEOP, "so_leader")
FOLLOWER_CONFIG_PATH = os.path.join(CALIBRATION_BASE_PATH_ROBOTS, "so_follower")

# The hardware families a robot record can describe. "so101" is the SO-101
# leader/follower pair (Feetech STS3215 over USB serial); "maker" is the Maker
# Arm v1 — a 7-DOF RobStride CAN follower driven by a Star Arm 102 (reBot 102)
# leader on FashionStar UART servos. The two share no bus protocol, no
# calibration procedure and no port-detection method, so the arm type is the
# discriminant every hardware path branches on.
ArmType = Literal["so101", "maker", "metal"]
ARM_TYPES: tuple[str, ...] = ("so101", "maker", "metal")
DEFAULT_ARM_TYPE = "so101"

# lerobot derives a device's calibration directory from the device CLASS's
# `name` attribute (Robot.__init__ / Teleoperator.__init__ ->
# HF_LEROBOT_CALIBRATION/<robots|teleoperators>/<name>). These constants must
# therefore match those class names EXACTLY — "so_leader"/"so_follower" for
# the SO-101 pair, "rebot_102_leader" for the Star Arm 102 leader, and
# "maker_follower"/"metal_follower" for the CAN followers. Renaming a device
# class upstream silently strands a whole library here.
#
# The LEADER path is shared by the Maker AND Metal arms: their leader presets
# (`rebot_102_leader_maker` / `rebot_102_leader_metal`) are config-only
# variants of the one RebotArm102Leader class, and the class name is what
# picks the directory. That sharing is why default_slot_config_name below
# mints per-arm-type ids — the two presets carry different direction and range
# mappings even though the physical leader zero pose is shared, so a name
# collision would silently reuse calibration metadata for the wrong follower.
MAKER_LEADER_CONFIG_PATH = os.path.join(CALIBRATION_BASE_PATH_TELEOP, "rebot_102_leader")
MAKER_FOLLOWER_CONFIG_PATH = os.path.join(CALIBRATION_BASE_PATH_ROBOTS, "maker_follower")
METAL_FOLLOWER_CONFIG_PATH = os.path.join(CALIBRATION_BASE_PATH_ROBOTS, "metal_follower")


def normalize_arm_type(value: object) -> str:
    """Coerce any stored/received arm_type to a known one, defaulting to so101.

    Unknown values fall back rather than raising for the same reason
    ``clamp_motor_power`` does: a corrupted or future-dated record must never
    make a robot unopenable. so101 is the safe default — it is what every
    record written before the Maker arm existed implicitly is.
    """
    return value if value in ARM_TYPES else DEFAULT_ARM_TYPE


# Each arm type owns a SEPARATE calibration library: a Maker zero-pose
# calibration is meaningless to an SO-101 and vice versa, and lerobot would not
# look for it in the other directory anyway. Nothing merges the two listings.
#
# Both resolvers read the module-level path globals at CALL time rather than
# capturing them in a lookup table at import time, so a test (or an install
# with a relocated cache) that monkeypatches LEADER_CONFIG_PATH still steers
# every caller — a frozen table would silently ignore the patch.


def leader_config_path_for(arm_type: object = DEFAULT_ARM_TYPE) -> str:
    """The calibration library dir holding this arm type's LEADER configs.

    Maker and Metal share one library (both leaders are the Star Arm 102 —
    same device class, different joint-mapping preset); the per-arm-type
    separation there is carried by the minted config NAMES instead
    (default_slot_config_name).
    """
    if normalize_arm_type(arm_type) in ("maker", "metal"):
        return MAKER_LEADER_CONFIG_PATH
    return LEADER_CONFIG_PATH


def follower_config_path_for(arm_type: object = DEFAULT_ARM_TYPE) -> str:
    """The calibration library dir holding this arm type's FOLLOWER configs."""
    normalized = normalize_arm_type(arm_type)
    if normalized == "maker":
        return MAKER_FOLLOWER_CONFIG_PATH
    if normalized == "metal":
        return METAL_FOLLOWER_CONFIG_PATH
    return FOLLOWER_CONFIG_PATH


def default_slot_config_name(record_name: str, mode: object, arm: str, arm_type: object) -> str:
    """The default calibration id for a robot record's empty slot.

    SO-101 keeps its historical defaults ("<name>", "<name>_<arm>" bimanual).
    The CAN families mint the arm type into the name ("<name>_metal",
    "<name>_metal_<arm>") because their Star-leader calibrations live in ONE
    shared directory while the presets' zero poses differ — an unsuffixed
    default would let a Maker robot and a Metal robot silently share a zero
    that is wrong for one of them. Followers get the same suffix purely for
    consistency (their libraries are already separate).

    Only a default: a slot that already names a calibration keeps it.
    """
    normalized = normalize_arm_type(arm_type)
    base = record_name if normalized == DEFAULT_ARM_TYPE else f"{record_name}_{normalized}"
    return f"{base}_{arm}" if mode == "bimanual" else base


# Define port storage path
PORT_CONFIG_PATH = os.path.join(MAKERMODSLAB_HOME, "ports")
LEADER_PORT_FILE = os.path.join(PORT_CONFIG_PATH, "leader_port.txt")
FOLLOWER_PORT_FILE = os.path.join(PORT_CONFIG_PATH, "follower_port.txt")

# Robot config records (per-robot JSON metadata)
ROBOTS_PATH = os.path.join(MAKERMODSLAB_HOME, "robots")

# Staging root for bimanual (BiSO) sessions. lerobot's BiSO devices take ONE
# calibration_dir + ONE base id and load each sub-arm as "<base>_left.json" /
# "<base>_right.json" — there is no way to point left/right at differently named
# library files. To free bimanual calibration from that naming constraint, we
# alias: the user-facing library keeps arbitrary names, and at session start we
# COPY the four selected library files into per-device staging dirs under this
# root as "<base>_left.json"/"<base>_right.json" for lerobot to load. The copy is
# unconditional every session (see stage_bimanual_calibrations) so a recalibrated
# library file always refreshes its stale staging alias.
MAKERMODSLAB_BISO_STAGING_PATH = os.path.join(MAKERMODSLAB_HOME, "biso_staging")

# Fallback base id when a bimanual start request carries no robot name (older
# frontends). Filesystem-safe and stable; a single unnamed bimanual robot reuses
# the same staging dir harmlessly since the copy is unconditional.
DEFAULT_BIMANUAL_BASE = "bimanual"

# Hub-job ids the user dismissed from the jobs UI (JSON list of strings). The
# HF Jobs API has no delete — a finished job stays in list_jobs() indefinitely
# — so hiding a dead run from the untracked list must be persisted locally.
DISMISSED_HUB_JOBS_FILE = os.path.join(MAKERMODSLAB_HOME, "dismissed_hub_jobs.json")

# Hub dataset repo ids the user typed straight into the picker and chose to keep
# ("Use org/name"). They aren't in the user's own namespace listing and have no
# local copy, so they'd vanish after selection unless we persist them here and
# fold them back into the merged /datasets listing.
SAVED_CUSTOM_DATASETS_FILE = os.path.join(MAKERMODSLAB_HOME, "saved_custom_datasets.json")

# Hub MODEL repo ids the user pinned via the "Add model" chooser — the models
# mirror of SAVED_CUSTOM_DATASETS_FILE (same rationale: a foreign-namespace repo
# with no local copy vanishes from the /models listing unless persisted here).
SAVED_CUSTOM_MODELS_FILE = os.path.join(MAKERMODSLAB_HOME, "saved_custom_models.json")

# Hub dataset/model repo ids the user removed from their pickers ("hidden").
# Hiding NEVER touches the Hub repo — it only filters the merged listing, so a
# repo the user's own namespace listing keeps returning stays gone until they
# re-add it (re-pinning auto-unhides). Persisted like the dismissed hub jobs
# (JSON list on disk, a set in memory).
SAVED_HIDDEN_DATASETS_FILE = os.path.join(MAKERMODSLAB_HOME, "hidden_datasets.json")
SAVED_HIDDEN_MODELS_FILE = os.path.join(MAKERMODSLAB_HOME, "hidden_models.json")

# Per-dataset episode indices the user excluded from training (curation, not
# deletion — the episode stays on disk and in every listing/upload, it's just
# left out of the --dataset.episodes subset a training run is launched with).
# JSON object keyed by repo_id -> list[int], unlike the flat repo-id lists
# above, since the thing being persisted is per-dataset state, not membership
# in one shared collection.
EXCLUDED_EPISODES_FILE = os.path.join(MAKERMODSLAB_HOME, "excluded_episodes.json")

# Stable per-install identity, minted on first read. The node registry uses it
# to recognize a peer across restarts and address changes (a machine's IP or
# MagicDNS name can change; its instance id doesn't).
INSTANCE_ID_FILE = os.path.join(MAKERMODSLAB_HOME, "instance_id.txt")

# The node registry's saved peer list: [{"url": ..., "name": ...}, ...]. Only
# url + name are persisted — identity (instance_id/version/capabilities) is
# deliberately NOT: a peer is re-verified against its live /api/v1/health on
# load/probe, so stale identity can never be served from disk.
NODES_FILE = os.path.join(MAKERMODSLAB_HOME, "nodes.json")

# Tag stamped on every dataset pushed to the Hub from MakerMods Lab, so we can later
# query the Hub for MakerMods Lab-produced datasets and compute usage metrics.
MAKERMODSLAB_TAG = "MakerModsLab"

# Tags injected onto EVERY dataset (and, where the trainer supports it, every
# policy) pushed to the Hub from MakerMods Lab. These are the org/product tags used to
# discover MakerMods / OpenBooth artifacts on the Hub. MAKERMODSLAB_TAG is kept too so
# existing MakerMods Lab usage queries keep working. This list is the single source of
# truth — add to it here rather than sprinkling literals at push sites.
REQUIRED_HUB_TAGS = ["makermods", "openbooth", MAKERMODSLAB_TAG]


def with_makermodslab_tag(tags: list[str] | None) -> list[str]:
    """Return `tags` with the REQUIRED_HUB_TAGS appended (deduped, order preserved).

    Despite the historical name, this appends every tag in REQUIRED_HUB_TAGS
    (currently "makermods", "openbooth", and MAKERMODSLAB_TAG), so every Hub push made
    through this funnel carries the org/product tags. Caller-supplied tags come
    first and are never duplicated.
    """
    out = list(tags or [])
    for tag in REQUIRED_HUB_TAGS:
        if tag not in out:
            out.append(tag)
    return out


# State that versions before the MAKERMODSLAB_HOME split wrote beside lerobot's
# files: (name under LEGACY_STATE_ROOT, this module's attribute holding the new
# path). The attribute is looked up AT CALL TIME so a redirected constant (the
# test fixtures) is honoured. Calibration libraries and training outputs are
# deliberately absent — they stay where lerobot reads them.
_LEGACY_STATE_ENTRIES: tuple[tuple[str, str], ...] = (
    ("ports", "PORT_CONFIG_PATH"),
    ("robots", "ROBOTS_PATH"),
    ("makermodslab_biso", "MAKERMODSLAB_BISO_STAGING_PATH"),
    ("dismissed_hub_jobs.json", "DISMISSED_HUB_JOBS_FILE"),
    ("saved_custom_datasets.json", "SAVED_CUSTOM_DATASETS_FILE"),
    ("saved_custom_models.json", "SAVED_CUSTOM_MODELS_FILE"),
    ("hidden_datasets.json", "SAVED_HIDDEN_DATASETS_FILE"),
    ("hidden_models.json", "SAVED_HIDDEN_MODELS_FILE"),
    ("excluded_episodes.json", "EXCLUDED_EPISODES_FILE"),
    ("instance_id.txt", "INSTANCE_ID_FILE"),
    ("nodes.json", "NODES_FILE"),
)


def _remove_path(path: str) -> None:
    """Best-effort removal of a file, symlink or directory tree."""
    if os.path.islink(path) or os.path.isfile(path):
        with contextlib.suppress(OSError):
            os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _move_entry(src: str, dst: str) -> bool:
    """Move ``src`` to ``dst`` without ever leaving a half-written ``dst``.

    ``shutil.move`` is a rename on one filesystem but copy-then-delete across
    two — and ``~/.cache/huggingface`` symlinked onto a big external drive is
    a common lerobot setup, which puts the two roots on different volumes. A
    copy that dies half-way (disk full, one unreadable file) would leave a
    partial ``dst`` that the destination-wins rule then treats as the live
    state forever. So the move lands in a sibling ``<dst>.migrating`` first
    and is renamed into place only once complete; on failure the sibling is
    removed and ``src`` is untouched (``shutil.move`` deletes the source only
    after a full copy).
    """
    staging = dst + ".migrating"
    _remove_path(staging)
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, staging)
        os.replace(staging, dst)
    except OSError as exc:
        logger.warning("Could not migrate %s -> %s: %s", src, dst, exc)
        _remove_path(staging)
        return False
    return True


def _merge_dir(src: str, dst: str) -> tuple[int, int]:
    """Move the entries of legacy dir ``src`` that ``dst`` lacks; keep the rest.

    Returns (moved, left). ``src`` is removed once nothing is left in it.
    """
    moved = left = 0
    for name in sorted(os.listdir(src)):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if os.path.lexists(d):
            left += 1
        elif _move_entry(s, d):
            moved += 1
        else:
            left += 1
    if left == 0:
        with contextlib.suppress(OSError):
            os.rmdir(src)
    return moved, left


def migrate_legacy_state(legacy_root: str | None = None) -> list[str]:
    """Move MakerMods Lab state written beside lerobot's cache into MAKERMODSLAB_HOME.

    One-shot and idempotent. A FILE entry moves only when nothing exists at
    the new path: a destination that already exists is the live state and
    wins, so an old version run after the split cannot clobber newer files on
    the next upgrade, and a second call is a no-op. A DIRECTORY entry that
    exists at both places is merged name by name under the same rule — the
    new location's directories get created empty by ordinary reads
    (``list_robot_records`` makes ``robots/`` on every listing), so a
    new → old → new round-trip would otherwise strand every robot record the
    old version wrote in between. Whatever is left behind is named in one
    WARNING per start, so a user can find it. A failed move is logged and
    skipped; the app then starts with that entry at its defaults rather than
    refusing to start. Returns the destinations written.

    The caller decides WHEN this runs (server startup, before the first read
    of any entry — every reader here is lazy) and whether it runs at all
    (never under a ``MAKERMODSLAB_HOME`` override; see HOME_IS_OVERRIDDEN).
    """
    root = LEGACY_STATE_ROOT if legacy_root is None else legacy_root
    written: list[str] = []
    left_behind: list[str] = []
    for legacy_name, attr in _LEGACY_STATE_ENTRIES:
        src = os.path.join(root, legacy_name)
        dst = globals()[attr]
        if not os.path.lexists(src):
            continue
        if not os.path.lexists(dst):
            if _move_entry(src, dst):
                written.append(dst)
            continue
        if os.path.isdir(src) and not os.path.islink(src) and os.path.isdir(dst):
            moved, left = _merge_dir(src, dst)
            if moved:
                written.append(dst)
            if left:
                left_behind.append(src)
        else:
            left_behind.append(src)
    if written:
        logger.info(
            "Moved %d MakerMods Lab state entries from %s to %s", len(written), root, MAKERMODSLAB_HOME
        )
    if left_behind:
        logger.warning(
            "Legacy MakerMods Lab state left in place because a newer copy exists under %s: %s",
            MAKERMODSLAB_HOME,
            ", ".join(left_behind),
        )
    return written


def _atomic_write_text(path: str, content: str) -> None:
    """Write to <path>.tmp then os.replace, so a crash mid-write never leaves
    a half-written file on disk."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


def load_saved_nodes() -> list[dict[str, str | None]]:
    """The saved peer rows from NODES_FILE, each ``{"url": str, "name": str|None}``.

    Missing, corrupt, or wrong-shaped content yields [] (an empty registry is
    always a safe starting point); rows without a string url are dropped.
    """
    try:
        with open(NODES_FILE) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    rows: list[dict[str, str | None]] = []
    for row in data:
        if isinstance(row, dict) and isinstance(row.get("url"), str):
            name = row.get("name")
            rows.append({"url": row["url"], "name": name if isinstance(name, str) else None})
    return rows


def save_saved_nodes(rows: list[dict[str, str | None]]) -> None:
    """Persist the peer rows (url + name only) to NODES_FILE atomically."""
    _atomic_write_text(NODES_FILE, json.dumps(rows, indent=2))


_instance_id_cache: str | None = None


def get_instance_id() -> str:
    """This install's stable identity: a 32-hex-char id, persisted on first use.

    Cached after the first read; a wiped cache dir simply mints a new identity
    (a fresh install IS a new node as far as peers are concerned).
    """
    global _instance_id_cache
    if _instance_id_cache is not None:
        return _instance_id_cache
    try:
        with open(INSTANCE_ID_FILE) as f:
            stored = f.read().strip()
    except OSError:
        stored = ""
    if not re.fullmatch(r"[0-9a-f]{32}", stored):
        stored = uuid.uuid4().hex
        _atomic_write_text(INSTANCE_ID_FILE, stored + "\n")
    _instance_id_cache = stored
    return stored


def _port_file_for(robot_type: RobotSide) -> str:
    if robot_type == "leader":
        return LEADER_PORT_FILE
    if robot_type == "follower":
        return FOLLOWER_PORT_FILE
    raise ValueError(f"robot_type must be 'leader' or 'follower', got {robot_type!r}")


def _require_assigned_config(config: str, side: str) -> None:
    """Fail with a legible message when an arm has no calibration assigned.

    An empty name would otherwise resolve to the calibration *directory* and
    crash shutil.copy2 with an opaque IsADirectoryError. This happens when a
    robot record's config field was cleared (e.g. its calibration config was
    deleted) and a start request is issued anyway.
    """
    if not (config or "").strip():
        raise FileNotFoundError(
            f"The {side} arm has no calibration assigned. Calibrate it "
            "(or assign a saved calibration config) before starting."
        )


def setup_calibration_files(leader_config: str, follower_config: str, arm_type: object = DEFAULT_ARM_TYPE):
    """Setup calibration files in the correct locations for teleoperation and recording.

    ``arm_type`` selects which library pair to read/write — an SO-101 session
    stages from so_leader/so_follower, a Maker session from
    rebot_102_leader/maker_follower. Those ARE lerobot's expected locations for
    each device class, so this stays a validating no-op copy within one dir.
    """
    _require_assigned_config(leader_config, "leader")
    _require_assigned_config(follower_config, "follower")
    # Extract config names from file paths (remove .json extension)
    leader_config_name = os.path.splitext(leader_config)[0]
    follower_config_name = os.path.splitext(follower_config)[0]

    leader_library = leader_config_path_for(arm_type)
    follower_library = follower_config_path_for(arm_type)

    # Log the full paths to check if files exist
    leader_config_full_path = os.path.join(leader_library, leader_config)
    follower_config_full_path = os.path.join(follower_library, follower_config)

    logger.info("Checking calibration files:")
    logger.info(f"Leader config path: {leader_config_full_path}")
    logger.info(f"Follower config path: {follower_config_full_path}")
    logger.info(f"Leader config exists: {os.path.exists(leader_config_full_path)}")
    logger.info(f"Follower config exists: {os.path.exists(follower_config_full_path)}")

    # Create calibration directories if they don't exist
    leader_calibration_dir = leader_library
    follower_calibration_dir = follower_library
    os.makedirs(leader_calibration_dir, exist_ok=True)
    os.makedirs(follower_calibration_dir, exist_ok=True)

    # Copy calibration files to the correct locations if they're not already there
    leader_target_path = os.path.join(leader_calibration_dir, f"{leader_config_name}.json")
    follower_target_path = os.path.join(follower_calibration_dir, f"{follower_config_name}.json")

    if not os.path.exists(leader_target_path):
        if os.path.exists(leader_config_full_path):
            shutil.copy2(leader_config_full_path, leader_target_path)
            logger.info(f"Copied leader calibration to {leader_target_path}")
        else:
            raise FileNotFoundError(f"Leader calibration file not found: {leader_config_full_path}")
    else:
        logger.info(f"Leader calibration already exists at {leader_target_path}")

    if not os.path.exists(follower_target_path):
        if os.path.exists(follower_config_full_path):
            shutil.copy2(follower_config_full_path, follower_target_path)
            logger.info(f"Copied follower calibration to {follower_target_path}")
        else:
            raise FileNotFoundError(f"Follower calibration file not found: {follower_config_full_path}")
    else:
        logger.info(f"Follower calibration already exists at {follower_target_path}")

    return leader_config_name, follower_config_name


def setup_follower_calibration_file(follower_config: str, arm_type: object = DEFAULT_ARM_TYPE):
    """Setup follower calibration file in the correct location for replay functionality"""
    _require_assigned_config(follower_config, "follower")
    # Extract config name from file path (remove .json extension)
    follower_config_name = os.path.splitext(follower_config)[0]

    follower_library = follower_config_path_for(arm_type)

    # Log the full path to check if file exists
    follower_config_full_path = os.path.join(follower_library, follower_config)

    logger.info("Checking follower calibration file:")
    logger.info(f"Follower config path: {follower_config_full_path}")
    logger.info(f"Follower config exists: {os.path.exists(follower_config_full_path)}")

    # Create calibration directory if it doesn't exist
    follower_calibration_dir = follower_library
    os.makedirs(follower_calibration_dir, exist_ok=True)

    # Copy calibration file to the correct location if it's not already there
    follower_target_path = os.path.join(follower_calibration_dir, f"{follower_config_name}.json")

    if not os.path.exists(follower_target_path):
        if os.path.exists(follower_config_full_path):
            shutil.copy2(follower_config_full_path, follower_target_path)
            logger.info(f"Copied follower calibration to {follower_target_path}")
        else:
            raise FileNotFoundError(f"Follower calibration file not found: {follower_config_full_path}")
    else:
        logger.info(f"Follower calibration already exists at {follower_target_path}")

    return follower_config_name


def find_available_ports():
    """Find all available serial ports on the system"""
    try:
        from serial.tools import list_ports  # Part of pyserial library
    except ImportError as exc:
        raise ImportError("pyserial library is required. Install it with: pip install pyserial") from exc

    if platform.system() == "Windows":
        # List COM ports using pyserial
        ports = [port.device for port in list_ports.comports()]
    else:
        # Linux/macOS: globbing all of /dev/tty* returns dozens of pseudo-ttys
        # and Bluetooth/debug devices. Restrict to USB-serial adapters — the only
        # thing an SO-101 arm shows up as — and keep the tty.* naming the rest of
        # the code (and saved robot records) use.
        #   macOS:  /dev/tty.usbmodem*  /dev/tty.usbserial*
        #   Linux:  /dev/ttyUSB*        /dev/ttyACM*
        patterns = ("tty.usbmodem*", "tty.usbserial*", "ttyUSB*", "ttyACM*")
        ports = [str(path) for pattern in patterns for path in Path("/dev").glob(pattern)]
    return sorted(ports)


def get_saved_robot_port(robot_type: RobotSide) -> str | None:
    """Return the saved port for `robot_type`, or None if no file exists."""
    port_file = _port_file_for(robot_type)
    if not os.path.exists(port_file):
        logger.info(f"No saved port found for {robot_type}")
        return None
    with open(port_file) as f:
        port = f.read().strip()
    logger.info(f"Retrieved saved {robot_type} port: {port}")
    return port


def get_default_robot_port(robot_type: RobotSide) -> str:
    """Saved port if present, else a platform-typical default."""
    saved_port = get_saved_robot_port(robot_type)
    if saved_port:
        return saved_port
    if platform.system() == "Windows":
        return "COM3"
    return "/dev/ttyUSB0"


# ---------------------------------------------------------------------------
# Robot record helpers
# ---------------------------------------------------------------------------

# Characters disallowed in a robot name (filesystem safety)
_INVALID_NAME_CHARS = ("/", "\\", "..")

# The primary leader/follower pair. In bimanual mode this is the LEFT arm pair;
# in single mode it's the only pair. Reusing these keeps existing records valid.
_SINGLE_CONFIG_FIELDS = ("leader_port", "follower_port", "leader_config", "follower_config")
# The RIGHT arm pair — populated only when mode == "bimanual".
_BIMANUAL_CONFIG_FIELDS = (
    "right_leader_port",
    "right_follower_port",
    "right_leader_config",
    "right_follower_config",
)
_ROBOT_STRING_FIELDS = _SINGLE_CONFIG_FIELDS + _BIMANUAL_CONFIG_FIELDS
_ROBOT_LIST_FIELDS = ("cameras",)

# Auto-calibration drive torque, as a percentage of full torque. Threaded into
# the vendored autocal subprocess as --torque-limit (percent × 10; see
# makermodslab/auto_calibrate.py). Regular sessions (teleop/record/policy runs) run
# at stock LeRobot torque and ignore this value (makermodslab/motor_power.py
# reset_torque_limit). Bounded below because under ~10% the arm can't reliably
# move its own weight; default = the vendored script's own DEFAULT_TORQUE_LIMIT
# (380 ÷ _TORQUE_LIMIT_PER_PERCENT 10 = 38%).
MOTOR_POWER_MIN = 10
MOTOR_POWER_MAX = 100
DEFAULT_MOTOR_POWER = 38


def clamp_motor_power(value: object) -> int:
    """Coerce a motor_power value to a safe integer percent in [10, 100].

    Anything non-numeric (including bool, a subclass of int) falls back to
    DEFAULT_MOTOR_POWER rather than raising, so a corrupted record can never
    block a session start.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DEFAULT_MOTOR_POWER
    return max(MOTOR_POWER_MIN, min(MOTOR_POWER_MAX, int(value)))


# Config-name fields whose stored value may carry a ".json" extension to strip.
_CONFIG_NAME_FIELDS = ("leader_config", "follower_config", "right_leader_config", "right_follower_config")
_VALID_MODES = ("single", "bimanual")
_DEFAULT_MODE = "single"


def _robot_record_path(name: str) -> str:
    return os.path.join(ROBOTS_PATH, f"{name}.json")


def is_valid_robot_name(name: str) -> bool:
    """Check that a robot name is safe to use as a filename."""
    if not name or not isinstance(name, str):
        return False
    if name.strip() != name:
        return False
    return not any(bad in name for bad in _INVALID_NAME_CHARS)


# Display names for training runs (JobRecord.name / display_name). Long enough
# for any sentence a card can render, short enough that a pasted document can't
# become a "name" that bloats every listing response carrying it.
JOB_NAME_MAX_LENGTH = 200


def validate_job_name(name: str) -> str:
    """Validate a training-run display name; returns the trimmed name.

    THE shared validator for both paths that accept one — submit
    (`TrainingRequest.job_name`, via JobRegistry.start) and
    `JobRegistry.rename` — so what one path refuses the other can't store.
    Raises ValueError with a user-facing message (both callers surface it as
    HTTP 400). Deliberately a boundary check, not a pydantic model constraint:
    legacy records persisted before validation existed must keep loading."""
    trimmed = name.strip()
    if not trimmed:
        raise ValueError("Display name cannot be empty.")
    if len(trimmed) > JOB_NAME_MAX_LENGTH:
        raise ValueError(f"Display name is too long — keep it under {JOB_NAME_MAX_LENGTH} characters.")
    if not is_valid_robot_name(trimmed):
        raise ValueError("Invalid display name.")
    return trimmed


def _empty_record(name: str) -> dict:
    record: dict = {
        "name": name,
        "mode": _DEFAULT_MODE,
        "arm_type": DEFAULT_ARM_TYPE,
        "motor_power": DEFAULT_MOTOR_POWER,
    }
    for field in _ROBOT_STRING_FIELDS:
        record[field] = ""
    for field in _ROBOT_LIST_FIELDS:
        record[field] = []
    return record


def get_robot_record(name: str) -> dict | None:
    """Return the robot record by name, or None if missing."""
    path = _robot_record_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read robot record {name}: {e}")
        return None
    # Ensure all expected fields exist (forward/back compat)
    record = _empty_record(name)
    record.update({k: v for k, v in data.items() if k in record})
    record["name"] = name
    # Canonical config names are STEMS (no .json). Older records stored the
    # filename with the extension — normalize on read so every consumer sees the
    # same form. The on-disk file keeps its .json.
    for field in _CONFIG_NAME_FIELDS:
        value = record.get(field, "")
        if isinstance(value, str) and value.endswith(".json"):
            record[field] = value[: -len(".json")]
    # Guard against an unknown mode on disk.
    if record.get("mode") not in _VALID_MODES:
        record["mode"] = _DEFAULT_MODE
    # Records written before the Maker arm existed carry no arm_type; they are
    # SO-101s by definition, which is exactly what normalize_arm_type returns.
    record["arm_type"] = normalize_arm_type(record.get("arm_type"))
    # Older records have no motor_power (→ full power via _empty_record); an
    # out-of-range or corrupted value on disk is clamped so every consumer
    # sees a safe 10-100 integer.
    record["motor_power"] = clamp_motor_power(record.get("motor_power"))
    return record


def list_robot_records() -> list[dict]:
    """Return all robot records on disk."""
    if not os.path.exists(ROBOTS_PATH):
        return []
    records = []
    for filename in sorted(os.listdir(ROBOTS_PATH)):
        if not filename.endswith(".json"):
            continue
        name = os.path.splitext(filename)[0]
        record = get_robot_record(name)
        if record is not None:
            records.append(record)
    return records


def save_robot_record(name: str, data: dict, allow_create: bool = True) -> bool:
    """
    Upsert a robot record. Merges `data` into the existing record, preserving
    fields not provided. Returns True if a write occurred, False if no-oped.

    - If the record exists: merge and write.
    - If the record does not exist and `allow_create` is True: create with empty
      fields then merge.
    - If the record does not exist and `allow_create` is False: log and no-op.
    """
    if not is_valid_robot_name(name):
        logger.error(f"Invalid robot name: {name!r}")
        return False

    os.makedirs(ROBOTS_PATH, exist_ok=True)
    existing = get_robot_record(name)
    if existing is None and not allow_create:
        logger.info(f"save_robot_record no-op: {name} does not exist (allow_create=False)")
        return False

    record = existing if existing is not None else _empty_record(name)
    # Decided BEFORE the merge below, because the switch blanks hardware-bound
    # fields and must not blank ones this same payload is setting.
    switching_arm_type = data.get("arm_type") in ARM_TYPES and data["arm_type"] != record.get("arm_type")
    for field in _ROBOT_STRING_FIELDS:
        if field in data and isinstance(data[field], str):
            record[field] = data[field]
    for field in _ROBOT_LIST_FIELDS:
        if field in data and isinstance(data[field], list):
            record[field] = data[field]
    # Same known-typed-fields-only merge as above: a numeric motor_power is
    # clamped to the safe range, anything else is ignored (keeps existing).
    value = data.get("motor_power")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        record["motor_power"] = clamp_motor_power(value)
    if data.get("mode") in _VALID_MODES:
        record["mode"] = data["mode"]
    record.setdefault("mode", _DEFAULT_MODE)
    # Switching arm type invalidates every hardware-bound field on the record.
    # The ports name physically different adapters (a Feetech USB-serial bridge
    # vs a CANable + a FashionStar UART bridge) and the calibration names point
    # into the OTHER arm type's library, where they do not exist — a stale
    # reference would fail deep inside lerobot's connect() as a missing-file
    # error instead of here as "this arm needs setting up". Blank them so the
    # robot lands back in the normal needs-calibration state. Fields set by
    # THIS payload survive: a caller that switches type and assigns new ports
    # in one request means both.
    if switching_arm_type:
        record["arm_type"] = data["arm_type"]
        for stale in _ROBOT_STRING_FIELDS:
            if stale not in data:
                record[stale] = ""
    record.setdefault("arm_type", DEFAULT_ARM_TYPE)
    record["name"] = name

    path = _robot_record_path(name)
    _atomic_write_text(path, json.dumps(record, indent=2))
    logger.info(f"Saved robot record {name}: {record}")
    return True


def delete_robot_record(name: str) -> bool:
    """Delete a robot record. Returns True if a file was removed."""
    if not is_valid_robot_name(name):
        return False
    path = _robot_record_path(name)
    if not os.path.exists(path):
        return False
    os.remove(path)
    logger.info(f"Deleted robot record {name}")
    return True


def rename_robot_record(old_name: str, new_name: str) -> tuple[bool, str]:
    """
    Rename a robot record file. Returns (ok, reason).

    `reason` is a machine-readable code on failure: "invalid_name" (either name
    fails validation), "not_found" (no record under old_name), or "name_taken"
    (a record already exists under new_name). On success reason is "".

    Renaming the *robot* record never touches calibration files: those live under
    config-name paths (leader_config / follower_config), independent of the robot
    record's name. A no-op rename (old == new) succeeds.
    """
    if not is_valid_robot_name(old_name) or not is_valid_robot_name(new_name):
        return False, "invalid_name"

    record = get_robot_record(old_name)
    if record is None:
        return False, "not_found"

    if old_name == new_name:
        return True, ""

    if os.path.exists(_robot_record_path(new_name)):
        return False, "name_taken"

    record["name"] = new_name
    _atomic_write_text(_robot_record_path(new_name), json.dumps(record, indent=2))
    os.remove(_robot_record_path(old_name))
    logger.info(f"Renamed robot record {old_name} -> {new_name}")
    return True, ""


# ---------------------------------------------------------------------------
# Session cameras — resolved from the robot record, never from the request
# ---------------------------------------------------------------------------
#
# The robot record is the single source of truth for which cameras a session
# opens. Recording and inference used to carry their own camera dicts in the
# start request, so a session could run against a camera set that no longer
# matched (or never matched) the saved robot — and edits made on the panel were
# silently discarded. Both flows now name a robot and let the server resolve
# the cameras; cameras are edited in one place only (the robot settings dialog).


class CameraResolutionError(ValueError):
    """A session's cameras could not be resolved from its robot record.

    Carries a user-facing message: the route layer turns it into a 400 with
    this text, so it must say what to fix and where (robot settings), not name
    internals.
    """


# The subset of a stored camera entry a SESSION needs. `id`/`device_id`/`name`
# are record-keeping and UI concerns: `name` becomes the dict key, and the other
# two must not leak into the camera config (rollout's `--robot.cameras=` CLI
# serializer forwards every key it is handed straight to lerobot's
# OpenCVCameraConfig, which would reject them).
_SESSION_CAMERA_KEYS = (
    "type",
    "camera_index",
    "unique_id",
    "width",
    "height",
    "fps",
    "fourcc",
    "backend",
)


def session_camera_config(entry: dict) -> dict:
    """One stored camera entry → the per-camera dict a session consumes.

    That shape is what `record._build_camera_configs` and
    `rollout._format_cameras_arg` already take (both of which keep their own
    hardening: platform backend pin and the MJPG fourcc default).

    Keys the entry doesn't carry are omitted rather than sent as None, so the
    consumers' own defaults still apply. `type` defaults to "opencv" — records
    written before the field existed have no other kind of camera.
    """
    config = {key: entry[key] for key in _SESSION_CAMERA_KEYS if entry.get(key) is not None}
    config.setdefault("type", "opencv")
    return config


def record_cameras_by_name(cameras: list) -> dict[str, dict]:
    """Every camera of a robot record, keyed by its name.

    Names are the record's own user-facing labels and become the dataset's
    camera keys (recording) or the binding targets (inference), so a duplicate
    is ambiguous rather than merely untidy: keying a dict by name would
    silently drop one of the two. Raises CameraResolutionError instead.
    Non-dict entries in a hand-edited record are skipped.
    """
    by_name: dict[str, dict] = {}
    for entry in cameras or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            raise CameraResolutionError(
                "A camera in this robot's configuration has no name. "
                "Open Robot settings and name it before starting."
            )
        if name in by_name:
            raise CameraResolutionError(
                f"This robot has two cameras named '{name}'. "
                "Rename one in Robot settings so each camera is unambiguous."
            )
        by_name[name] = session_camera_config(entry)
    return by_name


def load_robot_cameras(robot_name: str) -> dict[str, dict]:
    """The named robot record's cameras, keyed by camera name.

    A blank name means a camera-less session ({}) — recording without cameras
    is legitimate. A NON-blank name that doesn't resolve to a record on disk
    raises: recording camera-less because the robot name was wrong is a silent
    data loss (a whole session with no video), so it must fail loudly.
    """
    name = (robot_name or "").strip()
    if not name:
        return {}
    record = get_robot_record(name) if is_valid_robot_name(name) else None
    if record is None:
        raise CameraResolutionError(
            f"No saved configuration found for robot '{name}'. "
            "Select an existing robot (or save this one in Robot settings) before starting."
        )
    return record_cameras_by_name(record.get("cameras") or [])


def _positive_int(value: object) -> int | None:
    """`value` as a usable pixel dimension, or None. bool is excluded (it is an
    int subclass, and `True` as a width is always a bug, never a 1-pixel frame)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def bind_robot_cameras(
    robot_name: str,
    bindings: dict[str, str],
    dims: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Resolve {policy-expected camera name: robot-record camera name} bindings
    into the session camera dict, keyed by the POLICY-expected names.

    Inference needs the checkpoint's own camera names, which rarely match the
    labels on the robot — hence the indirection. The request carries only the
    name pairing; the camera's IDENTITY and transport settings (index,
    unique_id, fps, fourcc, backend) come from the record.

    CAPTURE RESOLUTION is the one exception, supplied by `dims`
    ({policy camera name: {"width": w, "height": h}}, from the checkpoint's
    image_features). lerobot's standard rollout pipeline does NOT resize frames
    to the policy's input shape — only the HIL path has a resize processor — so
    the frames a policy sees at deployment must be captured at the resolution it
    was trained on. Taking width/height from the robot record instead would
    silently change policy behaviour whenever the record's configured capture
    size differs from the checkpoint's. Overlaid per camera and only for sane
    positive values, so a checkpoint that omits image dims (or an older client
    that sends none) gracefully falls back to the record's own width/height.

    Empty bindings mean a camera-less policy ({}). Raises
    CameraResolutionError when no robot is named, when the record is missing,
    or when a binding points at a camera the record doesn't have (the message
    lists what it does have).
    """
    if not bindings:
        return {}
    if not (robot_name or "").strip():
        raise CameraResolutionError(
            "No robot selected, so the bound cameras can't be resolved. Select a robot before starting."
        )
    available = load_robot_cameras(robot_name)
    resolved: dict[str, dict] = {}
    for policy_name, camera_name in bindings.items():
        key = str(camera_name or "").strip()
        entry = available.get(key)
        if entry is None:
            listing = ", ".join(sorted(available)) if available else "none"
            raise CameraResolutionError(
                f"This robot has no camera named '{key}' (bound to '{policy_name}'). "
                f"Cameras on this robot: {listing}. Fix the binding, or add the camera in Robot settings."
            )
        config = dict(entry)
        override = (dims or {}).get(policy_name) or {}
        width = _positive_int(override.get("width"))
        height = _positive_int(override.get("height"))
        # Both or neither: a half-applied override would capture at a mixed
        # policy/record aspect, which is worse than either source alone.
        if width is not None and height is not None:
            config["width"] = width
            config["height"] = height
        resolved[policy_name] = config
    return resolved


def is_robot_record_clean(record: dict, arms: str = "all") -> bool:
    """
    A record is 'clean' when every operational field for its mode is populated AND
    every referenced calibration file exists on disk. Cameras are optional.

    - single   : the leader/follower pair (4 fields, 2 calibration files).
    - bimanual : that pair (= left arm) plus the right pair (8 fields, 4 files).

    `arms` scopes the check to what an activity actually drives:
    - "all"      — leader + follower (teleoperation, recording).
    - "follower" — follower side only (inference, replay never open the leader
      bus, so an unassigned leader port / missing leader calibration must not
      block them; bimanual = both followers, still no leaders).
    """
    if not record:
        return False
    follower_only = arms == "follower"

    # Config fields are stems; the file on disk is "<stem>.json". Tolerate a
    # stored value that still carries the extension (defensive).
    def _file_for(base: str, name: str) -> str:
        stem = name[: -len(".json")] if name.endswith(".json") else name
        return os.path.join(base, f"{stem}.json")

    bimanual = record.get("mode") == "bimanual"
    required_fields = _SINGLE_CONFIG_FIELDS + (_BIMANUAL_CONFIG_FIELDS if bimanual else ())
    if follower_only:
        required_fields = tuple(f for f in required_fields if "follower" in f)
    for field in required_fields:
        value = record.get(field, "")
        if not isinstance(value, str) or not value.strip():
            return False

    # Resolve the libraries by THIS record's arm type: the SO-101 and Maker
    # pairs keep separate directories, so checking the SO-101 ones for a Maker
    # robot looks for a file that was never going to be there and the robot can
    # never read as ready.
    follower_library = follower_config_path_for(record.get("arm_type"))
    leader_library = leader_config_path_for(record.get("arm_type"))

    config_files = [
        _file_for(follower_library, record["follower_config"]),
    ]
    if not follower_only:
        config_files.append(_file_for(leader_library, record["leader_config"]))
    if bimanual:
        config_files.append(_file_for(follower_library, record["right_follower_config"]))
        if not follower_only:
            config_files.append(_file_for(leader_library, record["right_leader_config"]))
    return all(os.path.exists(p) for p in config_files)


def config_slot_conflict(record: dict) -> str | None:
    """
    Detect when a bimanual record points two same-side arms at the SAME config.

    The two leader slots share the so_leader dir and the two follower slots share
    so_follower, so an identical config name on both = one physical arm's
    calibration on two arms (at least one is wrong). Returns "leader"/"follower"
    for the offending side, or None. Single mode (one slot per side) never
    conflicts. A leader and follower sharing a name is fine — different dirs.
    """
    if record.get("mode") != "bimanual":
        return None
    leader = record.get("leader_config", "")
    if leader and leader == record.get("right_leader_config", ""):
        return "leader"
    follower = record.get("follower_config", "")
    if follower and follower == record.get("right_follower_config", ""):
        return "follower"
    return None


# Port fields per mode. Unlike configs (which may legitimately share a name
# across leader/follower dirs), a serial PORT is one physical USB device, so
# every arm's port must be distinct — across BOTH sides.
_SINGLE_PORT_FIELDS = ("leader_port", "follower_port")
_BIMANUAL_PORT_FIELDS = ("right_leader_port", "right_follower_port")


def _stage_one_side(
    library_dir: str, staging_dir: str, base: str, left_stem: str, right_stem: str, side: str
):
    """Copy one device's two library calibrations into its staging dir as
    "<base>_left.json"/"<base>_right.json". Overwrites unconditionally so a
    recalibrated library file always refreshes the staging alias. Raises a
    clear, user-facing FileNotFoundError naming the slot and file when a
    referenced library file is missing (before lerobot's connect() can fall into
    interactive recalibration, which hangs the headless thread)."""
    os.makedirs(staging_dir, exist_ok=True)
    for arm, stem in (("left", left_stem), ("right", right_stem)):
        slot = f"{arm} {side}"
        _require_assigned_config(stem, slot)
        stem = stem[: -len(".json")] if stem.endswith(".json") else stem
        src = os.path.join(library_dir, f"{stem}.json")
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"The {slot} arm's calibration file '{stem}.json' was not found in "
                f"{library_dir}. Calibrate that arm (or assign a saved calibration) "
                "before starting."
            )
        dst = os.path.join(staging_dir, f"{base}_{arm}.json")
        shutil.copy2(src, dst)
        logger.info(f"Staged {slot} calibration {src} -> {dst}")


def bimanual_base_id(robot_name: str | None) -> str:
    """Filesystem-safe, stable BiSO staging base id from a robot record name.

    The robot name (already validated by is_valid_robot_name when set via the
    record API) is ideal — one staging dir per robot. Blank or unsafe names fall
    back to DEFAULT_BIMANUAL_BASE so a start request can never produce an unsafe
    path or hang; the copy is unconditional so reuse of the fallback dir is safe.
    """
    name = (robot_name or "").strip()
    if name and is_valid_robot_name(name):
        return name
    return DEFAULT_BIMANUAL_BASE


# BiSO staging dir layout, shared so leader/follower stagers agree on paths.
# lerobot's BiSO devices load each sub-arm's calibration as "<base>_left.json"
# /"<base>_right.json" from a single calibration_dir, so MakerMods Lab's arbitrary
# library names can't be pointed at left/right directly — hence per-device
# staging dirs under MAKERMODSLAB_BISO_STAGING_PATH/<base>/{leader,follower}/.
def _bimanual_leader_staging_dir(base: str) -> str:
    return os.path.join(MAKERMODSLAB_BISO_STAGING_PATH, base, "leader")


def _bimanual_follower_staging_dir(base: str) -> str:
    return os.path.join(MAKERMODSLAB_BISO_STAGING_PATH, base, "follower")


def stage_bimanual_calibrations(
    base: str,
    leader_left: str,
    leader_right: str,
    follower_left: str,
    follower_right: str,
    arm_type: object = DEFAULT_ARM_TYPE,
) -> tuple[str, str, str]:
    """Stage the four arbitrarily-named library calibrations for a BiSO session.

    Copies the four selected library files into per-device staging dirs named to
    match lerobot's "<base>_left/right.json" convention, and returns the leader
    staging dir, the follower staging dir, and the base id for building
    BiSO*Config(id=base, calibration_dir=<staging dir>). The copy OVERWRITES
    unconditionally every call so a recalibrated library file refreshes its stale
    staging alias. Any missing library file fails fast with a clear per-slot
    error BEFORE lerobot's connect() (an absent calibration makes lerobot fall
    into interactive recalibration, which hangs the headless thread).

    Both sides are staged; use stage_bimanual_follower_calibrations for flows
    (inference) that drive followers only.
    """
    leader_staging = _bimanual_leader_staging_dir(base)
    follower_staging = _bimanual_follower_staging_dir(base)
    _stage_one_side(
        leader_config_path_for(arm_type), leader_staging, base, leader_left, leader_right, "leader"
    )
    _stage_one_side(
        follower_config_path_for(arm_type),
        follower_staging,
        base,
        follower_left,
        follower_right,
        "follower",
    )
    return leader_staging, follower_staging, base


def stage_bimanual_follower_calibrations(
    base: str,
    follower_left: str,
    follower_right: str,
    arm_type: object = DEFAULT_ARM_TYPE,
) -> tuple[str, str]:
    """Stage only the two follower calibrations for a follower-only BiSO session.

    Inference has no leader arms, so staging (and thus requiring) leader library
    files would fail spuriously — the leader library dir need not even contain a
    file matching the follower's name. Produces the identical follower staging
    layout as stage_bimanual_calibrations (same dir, same "<base>_left/right.json"
    names, same overwrite/fail-fast semantics) and returns (follower_staging_dir,
    base).
    """
    follower_staging = _bimanual_follower_staging_dir(base)
    _stage_one_side(
        follower_config_path_for(arm_type),
        follower_staging,
        base,
        follower_left,
        follower_right,
        "follower",
    )
    return follower_staging, base


def port_slot_conflict(record: dict) -> str | None:
    """
    Return a serial port assigned to more than one arm of this robot, or None.

    Two physical arms can't share a port, so all of a robot's ports must differ —
    leader vs follower in single mode, and all four in bimanual mode. Empty ports
    are ignored (not yet set).
    """
    fields = _SINGLE_PORT_FIELDS + (_BIMANUAL_PORT_FIELDS if record.get("mode") == "bimanual" else ())
    seen: set[str] = set()
    for field in fields:
        port = record.get(field, "")
        if not isinstance(port, str) or not port.strip():
            continue
        if port in seen:
            return port
        seen.add(port)
    return None


# ---------------------------------------------------------------------------
# Dismissed hub jobs
# ---------------------------------------------------------------------------


def get_dismissed_hub_jobs() -> set[str]:
    """Return the set of hub-job ids the user dismissed from the jobs UI.

    A missing or corrupted file yields the empty set — dismissal is cosmetic,
    so it must never block the hub listing.
    """
    if not os.path.exists(DISMISSED_HUB_JOBS_FILE):
        return set()
    try:
        with open(DISMISSED_HUB_JOBS_FILE) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read dismissed hub jobs: {e}")
        return set()
    if not isinstance(data, list):
        return set()
    return {j for j in data if isinstance(j, str) and j.strip()}


def add_dismissed_hub_job(job_id: str) -> bool:
    """Persist a hub-job id as dismissed. Returns False for a blank id.

    Idempotent — re-dismissing an already-dismissed id is a no-op success.
    """
    job_id = (job_id or "").strip()
    if not job_id:
        return False
    dismissed = get_dismissed_hub_jobs()
    if job_id not in dismissed:
        dismissed.add(job_id)
        _atomic_write_text(DISMISSED_HUB_JOBS_FILE, json.dumps(sorted(dismissed), indent=2))
        logger.info(f"Dismissed hub job {job_id}")
    return True


def prune_dismissed_hub_jobs(live_job_ids: set[str]) -> None:
    """Drop dismissed ids that no longer appear in the Hub listing, so the file
    only tracks ids that still need hiding.

    Call only with a listing that actually succeeded — pruning against a failed
    (empty) fetch would forget every dismissal.
    """
    dismissed = get_dismissed_hub_jobs()
    kept = dismissed & live_job_ids
    if kept != dismissed:
        _atomic_write_text(DISMISSED_HUB_JOBS_FILE, json.dumps(sorted(kept), indent=2))


# ---------------------------------------------------------------------------
# Pinned / hidden repo-id collections (datasets + models)
# ---------------------------------------------------------------------------
#
# Four persisted JSON list-of-strings files share one shape: read-tolerant
# (missing / corrupt / non-list -> empty), atomic writes, and get/add/remove.
# They differ on only two axes, both captured by _JsonRepoCollection:
#   * ordered pins ("saved custom" datasets/models) keep most-recently-added
#     first — re-adding an id moves it to the front — and get() returns a list.
#   * hidden sets ("hidden" datasets/models) are order-free: add() is idempotent
#     (writes only on a genuinely new id), persists sorted, and get() returns a
#     set.
# The public get_/add_/remove_ functions below stay individually named, thin
# delegators — their names and signatures are imported across
# datasets.py / models.py / server.py, so they must not change.


class _JsonRepoCollection:
    """One persisted JSON list-of-repo-id file, exposing get/add/remove.

    ``path_of`` is invoked on every access rather than captured once, so a caller
    (or a test) monkeypatching the module-level ``*_FILE`` constant is honored.
    ``ordered`` selects list-vs-set semantics (see the block comment above). A
    missing / corrupt / non-list file degrades to empty — this persistence is
    cosmetic and must never raise — and every write goes through
    ``_atomic_write_text``.
    """

    def __init__(self, path_of, *, ordered: bool, add_log: str, remove_log: str, read_error: str):
        self._path_of = path_of
        self._ordered = ordered
        self._add_log = add_log
        self._remove_log = remove_log
        self._read_error = read_error

    def _clean(self, data) -> list[str]:
        """Deduplicate to non-blank strings, preserving first-seen order."""
        seen: set[str] = set()
        out: list[str] = []
        for r in data:
            if isinstance(r, str) and r.strip() and r not in seen:
                seen.add(r)
                out.append(r)
        return out

    def get(self):
        """The stored ids as a list (ordered) or set (hidden). Empty on a
        missing / corrupt / non-list file."""
        path = self._path_of()
        if not os.path.exists(path):
            data: object = []
        else:
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Failed to read {self._read_error}: {e}")
                data = []
            if not isinstance(data, list):
                data = []
        cleaned = self._clean(data)
        return cleaned if self._ordered else set(cleaned)

    def _write(self, collection) -> None:
        payload = collection if self._ordered else sorted(collection)
        _atomic_write_text(self._path_of(), json.dumps(payload, indent=2))

    def add(self, repo_id: str) -> bool:
        """Persist ``repo_id``. False for a blank id, True otherwise. Ordered:
        moves an existing id to the front and always rewrites. Hidden: idempotent
        — a genuinely new id is added and written, a re-add is a no-op success."""
        repo_id = (repo_id or "").strip()
        if not repo_id:
            return False
        if self._ordered:
            saved = [r for r in self.get() if r != repo_id]
            saved.insert(0, repo_id)
            self._write(saved)
            logger.info(f"{self._add_log} {repo_id}")
        else:
            current = self.get()
            if repo_id not in current:
                current.add(repo_id)
                self._write(current)
                logger.info(f"{self._add_log} {repo_id}")
        return True

    def remove(self, repo_id: str) -> bool:
        """Drop ``repo_id``. True if it was present, False if it wasn't."""
        repo_id = (repo_id or "").strip()
        current = self.get()
        if repo_id not in current:
            return False
        if self._ordered:
            self._write([r for r in current if r != repo_id])
        else:
            self._write(current - {repo_id})
        logger.info(f"{self._remove_log} {repo_id}")
        return True


# The *_FILE constants are read through a lambda (not captured) so monkeypatching
# them in tests reaches the collection — see _JsonRepoCollection.path_of.
_SAVED_CUSTOM_DATASETS = _JsonRepoCollection(
    lambda: SAVED_CUSTOM_DATASETS_FILE,
    ordered=True,
    add_log="Saved custom dataset",
    remove_log="Removed saved custom dataset",
    read_error="saved custom datasets",
)
_SAVED_CUSTOM_MODELS = _JsonRepoCollection(
    lambda: SAVED_CUSTOM_MODELS_FILE,
    ordered=True,
    add_log="Saved custom model",
    remove_log="Removed saved custom model",
    read_error="saved custom models",
)
_HIDDEN_DATASETS = _JsonRepoCollection(
    lambda: SAVED_HIDDEN_DATASETS_FILE,
    ordered=False,
    add_log="Hid dataset",
    remove_log="Unhid dataset",
    read_error="hidden datasets",
)
_HIDDEN_MODELS = _JsonRepoCollection(
    lambda: SAVED_HIDDEN_MODELS_FILE,
    ordered=False,
    add_log="Hid model",
    remove_log="Unhid model",
    read_error="hidden models",
)


def get_saved_custom_datasets() -> list[str]:
    """Return the Hub dataset repo ids the user pinned by typing them into the
    picker, most-recently-used first.

    A missing or corrupted file yields the empty list — pinning is cosmetic, so
    it must never block the dataset listing. Order is preserved (unlike the
    dismissed-jobs set) so the picker can show the freshest picks first.
    """
    return _SAVED_CUSTOM_DATASETS.get()


def add_saved_custom_dataset(repo_id: str) -> bool:
    """Pin a typed Hub dataset repo id so it persists in the picker. Returns
    False for a blank id.

    Idempotent; re-saving an already-pinned id moves it to the front (so the
    listing shows most-recently-used first).
    """
    return _SAVED_CUSTOM_DATASETS.add(repo_id)


def remove_saved_custom_dataset(repo_id: str) -> bool:
    """Unpin a saved custom dataset. Returns True if it was present, False if it
    wasn't pinned in the first place."""
    return _SAVED_CUSTOM_DATASETS.remove(repo_id)


def get_saved_custom_models() -> list[str]:
    """The Hub MODEL repo ids the user pinned via the "Add model" chooser,
    most-recently-used first. Mirrors get_saved_custom_datasets: a missing or
    corrupted file yields the empty list — pinning is cosmetic, so it must never
    block the /models listing."""
    return _SAVED_CUSTOM_MODELS.get()


def add_saved_custom_model(repo_id: str) -> bool:
    """Pin a Hub model repo id so it persists in the /models listing. Returns
    False for a blank id. Idempotent; re-saving an already-pinned id moves it to
    the front (most-recently-used first). Mirrors add_saved_custom_dataset."""
    return _SAVED_CUSTOM_MODELS.add(repo_id)


def remove_saved_custom_model(repo_id: str) -> bool:
    """Unpin a saved custom model. Returns True if it was present, False if it
    wasn't pinned in the first place. Mirrors remove_saved_custom_dataset."""
    return _SAVED_CUSTOM_MODELS.remove(repo_id)


def get_hidden_datasets() -> set[str]:
    """The Hub dataset repo ids the user removed from their picker ("hidden").

    A missing or corrupted file yields the empty set — hiding is cosmetic, so it
    must never block the dataset listing. Mirrors get_dismissed_hub_jobs."""
    return _HIDDEN_DATASETS.get()


def add_hidden_dataset(repo_id: str) -> bool:
    """Hide a Hub dataset repo id from the picker listing. Returns False for a
    blank id. Idempotent — re-hiding an already-hidden id is a no-op success.
    NEVER touches the Hub repo or any local copy."""
    return _HIDDEN_DATASETS.add(repo_id)


def remove_hidden_dataset(repo_id: str) -> bool:
    """Unhide a dataset. Returns True if it was hidden, False if it wasn't.
    Also called by the pin route so re-adding a hidden repo makes it visible
    again (the auto-unhide)."""
    return _HIDDEN_DATASETS.remove(repo_id)


def get_hidden_models() -> set[str]:
    """The Hub model repo ids the user removed from their picker ("hidden").
    Mirrors get_hidden_datasets — a missing/corrupted file yields the empty set."""
    return _HIDDEN_MODELS.get()


def add_hidden_model(repo_id: str) -> bool:
    """Hide a Hub model repo id from the picker listing. Idempotent; returns
    False for a blank id. NEVER touches the Hub repo or any local copy."""
    return _HIDDEN_MODELS.add(repo_id)


def remove_hidden_model(repo_id: str) -> bool:
    """Unhide a model. Returns True if it was hidden, False if it wasn't. Also
    called by the pin route so re-adding a hidden repo makes it visible again."""
    return _HIDDEN_MODELS.remove(repo_id)


def _read_excluded_episodes_file() -> dict[str, list[int]]:
    """The whole excluded-episodes map. A missing/corrupt/non-object file
    degrades to empty — this is cosmetic curation state, so it must never
    raise or block training."""
    path = EXCLUDED_EPISODES_FILE
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read excluded episodes: {e}")
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[int]] = {}
    for repo_id, indices in data.items():
        if isinstance(repo_id, str) and isinstance(indices, list):
            out[repo_id] = sorted({i for i in indices if isinstance(i, int)})
    return out


def get_excluded_episodes(repo_id: str) -> list[int]:
    """Episode indices excluded from training for this dataset. Empty for a
    dataset with no exclusions, or when the file is missing/corrupt."""
    return _read_excluded_episodes_file().get(repo_id, [])


def set_excluded_episodes(repo_id: str, episode_indices: list[int]) -> None:
    """Replace the excluded-episode set for one dataset. An empty list clears
    the dataset's entry entirely rather than persisting a blank one. NEVER
    touches the dataset's files or Hub copy — this is a training-time filter
    applied client-side when building the run request, not a deletion."""
    data = _read_excluded_episodes_file()
    cleaned = sorted({i for i in episode_indices if isinstance(i, int)})
    if cleaned:
        data[repo_id] = cleaned
    else:
        data.pop(repo_id, None)
    _atomic_write_text(EXCLUDED_EPISODES_FILE, json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Calibration config import
# ---------------------------------------------------------------------------

# A lerobot motor calibration entry has exactly these integer fields.
_CALIBRATION_MOTOR_FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")


def calibration_dir_for_device(device_type: str, arm_type: object = DEFAULT_ARM_TYPE) -> str | None:
    """Map an API device_type ("teleop"/"robot") to its calibration dir, or None.

    ``arm_type`` picks the library: the SO-101 pair and the Maker pair keep
    entirely separate directories (see _CALIBRATION_DIRS), so a caller that
    forgets to thread it through reads the SO-101 library by default.
    """
    if device_type == "robot":
        return follower_config_path_for(arm_type)
    if device_type == "teleop":
        return leader_config_path_for(arm_type)
    return None


# A dataset id is either a bare "name" or "namespace/name" (exactly one slash).
# Each segment is an HF-style path component: 1-96 chars of [A-Za-z0-9._-] that
# starts and ends with an alphanumeric. We REJECT bad names (rather than silently
# sanitize) so e.g. "whoo/" fails loudly at the source instead of smuggling in a
# namespace and landing the dataset in a surprising path like "user/whoo/".
_DATASET_SEGMENT_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$")


def validate_dataset_name(name: object) -> tuple[bool, str]:
    """Validate ONE dataset repo-id segment (the user-typed name, or a namespace).

    Returns (ok, human_readable_reason).
    """
    if not isinstance(name, str) or not name.strip():
        return False, "Dataset name can't be empty."
    if name != name.strip():
        return False, "Dataset name can't have leading or trailing spaces."
    if "/" in name or "\\" in name:
        return False, "Dataset name can't contain slashes."
    if name in (".", ".."):
        return False, "Dataset name can't be '.' or '..'."
    if len(name) > 96:
        return False, "Dataset name is too long (max 96 characters)."
    if not _DATASET_SEGMENT_RE.match(name):
        return False, (
            "Dataset name may only use letters, digits, '.', '_' and '-', and must "
            "start and end with a letter or digit."
        )
    return True, ""


def validate_dataset_repo_id(repo_id: object) -> tuple[bool, str]:
    """Validate a full dataset id: a bare name, or 'namespace/name' (one slash).

    Returns (ok, human_readable_reason). Used by both recording and merge so a bad
    name is refused at the point of creation, not silently rewritten.
    """
    if not isinstance(repo_id, str) or not repo_id.strip():
        return False, "Dataset name can't be empty."
    parts = repo_id.split("/")
    if len(parts) > 2:
        return False, "Dataset name may contain at most one '/' (namespace/name)."
    if len(parts) == 2:
        ns_ok, ns_reason = validate_dataset_name(parts[0])
        if not ns_ok:
            return False, ns_reason.replace("Dataset name", "Namespace")
        return validate_dataset_name(parts[1])
    return validate_dataset_name(parts[0])


def validate_calibration_data(data: object) -> tuple[bool, str]:
    """
    Check that `data` looks like a lerobot motor calibration: a non-empty dict of
    motor_name -> {id, drive_mode, homing_offset, range_min, range_max} with
    integer values. Returns (ok, human_readable_reason). Validating here means a
    bad import fails loudly at upload instead of later inside teleop/calibration.
    """
    if not isinstance(data, dict) or not data:
        return False, "Calibration must be a non-empty object of motors."
    for motor, fields in data.items():
        if not isinstance(fields, dict):
            return False, f"Motor '{motor}' must be an object."
        for key in _CALIBRATION_MOTOR_FIELDS:
            if key not in fields:
                return False, f"Motor '{motor}' is missing '{key}'."
            value = fields[key]
            # bool is a subclass of int; a JSON true/false here is not valid.
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"Motor '{motor}' field '{key}' must be an integer."
    return True, ""


def save_imported_calibration(
    device_type: str, name: str, data: object, arm_type: object = DEFAULT_ARM_TYPE
) -> tuple[bool, str, str]:
    """
    Validate and persist an uploaded calibration as <name>.json under the side's
    config dir for this arm type. Never overwrites an existing file. Returns
    (ok, reason, name) where `name` is the normalized config name (extension
    stripped). Reason codes: "invalid_device", "invalid_name",
    "invalid_data:<msg>", "name_taken", "".
    """
    config_path = calibration_dir_for_device(device_type, arm_type)
    if config_path is None:
        return False, "invalid_device", ""

    name = name.strip()
    # Accept either a stem or a "<name>.json" filename (records carry the ext).
    if name.endswith(".json"):
        name = name[: -len(".json")]
    if not is_valid_robot_name(name):
        return False, "invalid_name", name

    ok, msg = validate_calibration_data(data)
    if not ok:
        return False, f"invalid_data:{msg}", name

    os.makedirs(config_path, exist_ok=True)
    file_path = os.path.join(config_path, f"{name}.json")
    if os.path.exists(file_path):
        return False, "name_taken", name

    _atomic_write_text(file_path, json.dumps(data, indent=2))
    logger.info(f"Imported calibration {normalize_arm_type(arm_type)}/{device_type}/{name}")
    return True, "", name


def rename_calibration_config(
    device_type: str, old_name: str, new_name: str, arm_type: object = DEFAULT_ARM_TYPE
) -> tuple[bool, str]:
    """
    Rename a calibration config file within a side's dir. Never overwrites an
    existing target. Robot records that referenced the old name (on this side,
    AND of this arm type) are repointed to the new name so they stay valid.
    Returns (ok, reason): "invalid_device", "invalid_name", "not_found",
    "name_taken", "".
    """
    arm_type = normalize_arm_type(arm_type)
    config_path = calibration_dir_for_device(device_type, arm_type)
    if config_path is None:
        return False, "invalid_device"

    old_stem = old_name[: -len(".json")] if old_name.endswith(".json") else old_name
    new_stem = new_name.strip()
    if new_stem.endswith(".json"):
        new_stem = new_stem[: -len(".json")]
    if not is_valid_robot_name(old_stem) or not is_valid_robot_name(new_stem):
        return False, "invalid_name"

    old_path = os.path.join(config_path, f"{old_stem}.json")
    if not os.path.exists(old_path):
        return False, "not_found"
    if old_stem == new_stem:
        return True, ""  # no-op

    new_path = os.path.join(config_path, f"{new_stem}.json")
    if os.path.exists(new_path):
        return False, "name_taken"

    os.rename(old_path, new_path)

    # Repoint any robot records that used the old config on this side — both the
    # primary/left slot and the bimanual right slot live in the same dir. Only
    # records of the SAME arm type: the two libraries are separate namespaces,
    # so an SO-101 record naming "arm_a" is a different file from a Maker record
    # naming "arm_a" and must not be dragged along by this rename.
    fields = (
        ("leader_config", "right_leader_config")
        if device_type == "teleop"
        else ("follower_config", "right_follower_config")
    )
    for rec in list_robot_records():
        if rec.get("arm_type") != arm_type:
            continue
        patch = {f: new_stem for f in fields if rec.get(f) == old_stem}
        if patch:
            save_robot_record(rec["name"], patch, allow_create=False)

    logger.info(f"Renamed calibration {arm_type}/{device_type}/{old_stem} -> {new_stem}")
    return True, ""


def clear_config_references(
    device_type: str, config_name: str, arm_type: object = DEFAULT_ARM_TYPE
) -> list[dict]:
    """Blank every robot-record field (on this side) that references this
    calibration config, across all robot records OF THIS ARM TYPE — both the
    primary/left slot and the bimanual right slot, regardless of mode. A stale
    right_* reference in a single-mode record is cleared too: it points at a
    file that no longer exists, so leaving it would resurface a dangling name
    on a mode switch. Records of the other arm type are skipped: their config
    names live in a separate library, so an identical name there is a different
    file that this delete did not touch.

    Called when a calibration config is deleted: instead of refusing the
    delete, the referencing arms are unassigned and return to the "needs
    calibration" state (is_robot_record_clean → False, and teleop/record refuse
    to start with a clear message until the arm is recalibrated or reassigned).

    Returns [{"robot": <name>, "fields": [<cleared fields>]}] for each record
    modified, so callers can tell the user which arms now need calibration.
    """
    fields = (
        ("leader_config", "right_leader_config")
        if device_type == "teleop"
        else ("follower_config", "right_follower_config")
    )
    arm_type = normalize_arm_type(arm_type)
    stem = config_name.removesuffix(".json")
    cleared: list[dict] = []
    for rec in list_robot_records():
        if rec.get("arm_type") != arm_type:
            continue
        hit = [f for f in fields if rec.get(f) == stem]
        if hit:
            save_robot_record(rec["name"], dict.fromkeys(hit, ""), allow_create=False)
            cleared.append({"robot": rec["name"], "fields": hit})
    return cleared
