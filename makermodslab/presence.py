"""Cross-device visibility for LOCAL training runs.

A cloud run is visible from any machine signed into the account, because HF Jobs
lists it. A LOCAL run is visible only on the machine it runs on — so a user with
a training desktop and a laptop has no way to see, from the laptop, that the
desktop is three hours into a run.

This module closes that gap with a presence board: a private Hub repo holding
one small JSON file per device, which every device writes to and reads from.

Two halves, deliberately separated:

  * The PURE half (payload building, staleness classification, device identity)
    is plain functions with no I/O, so it is testable and so the request path
    can call it freely.
  * The WRITER half is one daemon thread that does every Hub call. Nothing on a
    request path or a registry callback ever touches the network here.

WHY A MODEL REPO AND NOT A DATASET REPO. `datasets.list_hub_datasets` lists the
account's datasets with NO tag filter, deliberately — datasets pushed outside
lerobot's push_to_hub carry no tag and filtering made them invisible. A presence
*dataset* repo would therefore show up in the dataset picker as something the
user could train on. The models listing is the opposite: an allowlist admitting
only the `lerobot` tag or the run-repo naming pattern (`_list_author_models` in
server.py), and this repo matches neither, so it stays out of every picker for
free.

WHAT THIS IS NOT. Presence is observation, never control. There is no channel to
another machine, so a remote run cannot be stopped, resumed, or downloaded from
here, and nothing in this module pretends otherwise.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
import uuid
from typing import Any

from .utils.config import DEVICE_ID_FILE, PRESENCE_SETTINGS_FILE, _atomic_write_text
from .utils.hf_auth import cached_whoami, hf_hub_offline, shared_hf_api

logger = logging.getLogger(__name__)

#: Repo name under the user's own namespace. Never an org: org members would
#: otherwise read each other's device presence.
PRESENCE_REPO_NAME = "makermodslab-presence"

#: Payload version. Bumped only on a breaking shape change; a reader that meets
#: a version it does not know skips the file rather than guessing.
PRESENCE_SCHEMA = 1

#: How often a device re-publishes while it has an active run. Event writes
#: (start / finish) happen immediately and are NOT on this clock — this is
#: liveness only, so it is deliberately coarse.
#:
#: 10 minutes puts a training device at <=6 commits/hour and an idle one at
#: zero. A 60s heartbeat would be ~60/hour/device, which is where Hub throttling
#: and huggingface_hub retry noise start.
KEEPALIVE_INTERVAL_S = 600.0

#: A device unheard from for longer than this is no longer believed. 2.5x the
#: keepalive: one missed beat is tolerated, two are not.
STALE_AFTER_S = 1500.0

#: Past this, we stop saying "unknown" and call it stopped. Still never
#: "failed" — we did not observe a failure, we observed a silence.
PRESUMED_STOPPED_AFTER_S = 7200.0

#: A run that finished stays on the board this long, so a finish is OBSERVED
#: from the other machine rather than the row silently vanishing.
TERMINAL_GRACE_S = 3600.0

#: How many of the newest registry records a publish considers. Presence only
#: ever emits active runs plus recently-finished ones, so this is a ceiling on
#: work, not a window the user can notice.
_PUBLISHED_RUN_LIMIT = 50

#: Ceiling on devices read from the board in one pass — one download each.
_MAX_DEVICES = 25

#: How long shutdown waits for an in-flight publish to finish before giving up
#: on ordering, and then for the goodbye write itself. Both are bounded because
#: neither may hold the process open: the Hub client has no timeout of its own.
_WORKER_JOIN_TIMEOUT_S = 5.0
_GOODBYE_TIMEOUT_S = 10.0

#: Deadline for the whole board read. `/jobs/hub` budgets its fan-out for the
#: same reason: a blackholed connection must cost a stale page, not a hung one.
READ_BOARD_TIMEOUT_S = 20.0

#: NOTE on hung Hub calls. huggingface_hub's client has timeout=None, and
#: upload_file exposes no per-call deadline, so a dead socket can park the
#: writer thread indefinitely. That is tolerated rather than worked around: the
#: writer is a daemon thread that holds no lock while it calls out, so a wedged
#: write costs this device its own presence updates and nothing else — no
#: request path, no shutdown, no other feature waits on it.

_TERMINAL_STATES = frozenset({"done", "failed", "interrupted"})

#: Process-wide guards. `_DEVICE_ID` is cached because an unwritable cache file
#: would otherwise mint a FRESH uuid on every call — the payload and the file it
#: is written to would disagree, this device would stop recognizing its own
#: file, and its own runs would come back as a phantom other device.
_ID_LOCK = threading.Lock()
_DEVICE_ID: str | None = None
#: `load_settings` + `save_settings` is a read-modify-write; without this the
#: writer and a settings POST can clobber each other.
_SETTINGS_LOCK = threading.RLock()


# --------------------------------------------------------------------------
# Device identity (pure-ish: one file read, no network)
# --------------------------------------------------------------------------


def device_id() -> str:
    """This machine's stable presence id, minting one on first use.

    A uuid4 rather than the hostname: two laptops both called "MacBook-Pro"
    must not collide into one presence file, and a hostname change must not
    orphan the old file. The hostname travels separately as a LABEL.

    A corrupt or empty file is re-minted rather than raising — presence is a
    convenience, and refusing to start over an unreadable cache file would be a
    worse failure than a new id.

    Cached for the life of the process: if the file cannot be written, every
    call would otherwise return a DIFFERENT uuid, and the two calls behind one
    publish (the payload's `device_id` and the path it is uploaded to) would
    disagree. This device would then fail to recognize its own file on the
    board and would list its own local runs back to itself as a remote device.
    """
    global _DEVICE_ID
    with _ID_LOCK:
        if _DEVICE_ID is not None:
            return _DEVICE_ID
        _DEVICE_ID = _mint_device_id()
        return _DEVICE_ID


def _mint_device_id() -> str:
    try:
        with open(DEVICE_ID_FILE) as f:
            existing = f.read().strip()
        if existing:
            return existing
    except OSError:
        pass
    minted = str(uuid.uuid4())
    try:
        _atomic_write_text(DEVICE_ID_FILE, minted + "\n")
    except OSError as exc:
        # Un-persisted id: presence still works this session, and the device
        # simply gets a new row after a restart.
        logger.warning("Could not persist the device id: %s", exc)
    return minted


def reset_device_id_cache() -> None:
    """Drop the cached device id. For tests, which redirect DEVICE_ID_FILE per
    case and would otherwise all share whichever id the first one minted."""
    global _DEVICE_ID
    with _ID_LOCK:
        _DEVICE_ID = None


def _default_label() -> str:
    try:
        return socket.gethostname() or "this device"
    except OSError:
        return "this device"


def load_settings() -> dict:
    """Per-device presence settings. Publishing is ON by default."""
    try:
        with open(PRESENCE_SETTINGS_FILE) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return {
                "enabled": bool(raw.get("enabled", True)),
                "label": str(raw.get("label") or _default_label()),
                # Set once the first successful publish has been announced, so
                # the UI raises that notice exactly once per device.
                "announced": bool(raw.get("announced", False)),
            }
    except (OSError, ValueError):
        pass
    return {"enabled": True, "label": _default_label(), "announced": False}


def save_settings(**changes: Any) -> dict:
    """Merge `changes` into the stored settings and return the result.

    Locked: this is a read-modify-write, and the writer thread and a settings
    POST can otherwise interleave and drop one of the two updates.
    """
    with _SETTINGS_LOCK:
        settings = load_settings()
        settings.update(changes)
        try:
            _atomic_write_text(PRESENCE_SETTINGS_FILE, json.dumps(settings, indent=2))
        except OSError as exc:
            logger.warning("Could not persist presence settings: %s", exc)
        return settings


# --------------------------------------------------------------------------
# Payload (pure)
# --------------------------------------------------------------------------


def build_payload(records: list, *, now: float, label: str, dev_id: str) -> dict:
    """The presence file this device should publish, from the job registry.

    Only LOCAL runs: a cloud run is already visible from every machine through
    HF Jobs, and republishing it here would give it two rows in one library.
    Imports are not runs at all.

    Carries active runs plus anything that reached a terminal state within
    TERMINAL_GRACE_S, so a run finishing is something the other machine SEES
    rather than a row that quietly disappears.

    Deliberately minimal: run identity, progress and what it trains. No paths,
    no log lines, no metrics history, no config. This file is published to the
    Hub, so every field here is a field the user is publishing.
    """
    runs = []
    for rec in records:
        if getattr(rec, "runner", None) != "local":
            continue
        state = getattr(rec, "state", None)
        if state in _TERMINAL_STATES:
            ended = getattr(rec, "ended_at", None) or 0
            if now - ended > TERMINAL_GRACE_S:
                continue
        config = getattr(rec, "config", None)
        metrics = getattr(rec, "metrics", None)
        runs.append(
            {
                "job_id": rec.id,
                "job_number": getattr(rec, "job_number", 0),
                "name": getattr(rec, "name", None),
                "display_name": getattr(rec, "display_name", None),
                "state": state,
                "current_step": getattr(metrics, "current_step", 0) if metrics else 0,
                "total_steps": getattr(config, "steps", 0) if config else 0,
                "policy_type": getattr(config, "policy_type", None) if config else None,
                "dataset_repo_id": getattr(config, "dataset_repo_id", None) if config else None,
                "started_at": getattr(rec, "started_at", None),
                "ended_at": getattr(rec, "ended_at", None),
            }
        )
    runs.sort(key=lambda r: r.get("started_at") or 0, reverse=True)
    return {
        "schema": PRESENCE_SCHEMA,
        "device_id": dev_id,
        "device_label": label,
        "updated_at": now,
        "runs": runs,
    }


def has_active_runs(payload: dict) -> bool:
    """Whether the keepalive clock should be running at all."""
    return any(r.get("state") == "running" for r in payload.get("runs", []))


def classify_liveness(last_seen: float | None, *, now: float) -> str:
    """How much of a device's payload we still believe.

    'live' | 'unknown' | 'presumed_stopped'. A device that was unplugged mid-run
    never got to write a goodbye, so its last payload still says "running"
    forever — which is exactly the lie this exists to stop the UI telling.
    """
    if last_seen is None:
        return "presumed_stopped"
    age = now - last_seen
    if age <= STALE_AFTER_S:
        return "live"
    if age <= PRESUMED_STOPPED_AFTER_S:
        return "unknown"
    return "presumed_stopped"


def project_device(payload: dict, *, last_seen: float | None, now: float) -> dict:
    """One device's board entry, as the frontend should render it.

    Applies the one rule that matters most here: a device we no longer believe
    NEVER reports a running state (R6). Its runs keep their other facts, so the
    card can still say what it was doing when it was last heard from.
    """
    liveness = classify_liveness(last_seen, now=now)
    runs = []
    for r in payload.get("runs", []) or []:
        run = dict(r)
        if liveness != "live" and run.get("state") == "running":
            # Not "failed": we observed a silence, not a failure.
            run["state"] = "unknown"
        runs.append(run)
    return {
        "device_id": payload.get("device_id"),
        "device_label": payload.get("device_label"),
        "last_seen": last_seen,
        "liveness": liveness,
        "runs": runs,
    }


def presence_repo_id(username: str) -> str:
    return f"{username}/{PRESENCE_REPO_NAME}"


def device_file_path(dev_id: str) -> str:
    return f"devices/{dev_id}.json"


# --------------------------------------------------------------------------
# Writer (the only half that touches the network)
# --------------------------------------------------------------------------


class PresencePublisher:
    """Publishes this device's local runs to the presence repo.

    One daemon thread, woken either by a registry change or by the keepalive
    timer. Every Hub call happens here and nowhere else, so no request handler
    and no registry callback can ever block on the network.

    Failure policy, in two kinds:

      * TRANSIENT (network, 5xx, timeout): exponential backoff, and only the
        first failure in a streak is logged. On a hostile network this module
        must be quiet, not a log firehose — it writes on a timer forever, so a
        per-attempt log line would drown everything else in the file.
      * PERMANENT (401/403): the token cannot write. Retrying cannot fix that,
        and a silent forever-retry while the UI claims sharing is on is the
        exact failure this design has to avoid. Publishing disables itself for
        the session and records why, so the UI can say so once.
    """

    def __init__(self, registry, *, api_factory=shared_hf_api) -> None:
        self._registry = registry
        self._api_factory = api_factory
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        #: Serializes publishes so the shutdown goodbye can never race a
        #: keepalive already in flight and be overwritten by it.
        self._publish_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_write = 0.0
        self._failures = 0
        #: "Something changed, publish regardless of the keepalive clock."
        #: A real flag rather than a reading of `_wake`: the loop must CLEAR the
        #: event before publishing (so a change arriving mid-publish is not
        #: swallowed), which means `_wake.is_set()` is always False by the time
        #: the decision is made. Reading the event there silently killed the
        #: whole event-write path.
        self._dirty = False
        #: Set when publishing has given up for this session, with the reason.
        self._disabled_reason: str | None = None
        self._last_error: str | None = None

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        """Start the writer, unless there is nothing it could ever do.

        Offline mode is a hard no rather than a failing loop: every Hub write is
        disabled there by definition.
        """
        if hf_hub_offline():
            self._disabled_reason = "offline"
            logger.info("Presence publishing disabled: HF_HUB_OFFLINE is set.")
            return
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="presence-writer", daemon=True)
        self._thread.start()

    def stop(self, *, goodbye: bool = True) -> None:
        """Stop the writer, publishing a final state first.

        The goodbye is what turns "this machine is training" into "this machine
        finished" on every other device promptly, instead of leaving the last
        payload to age out through `unknown` over 25 minutes.

        Three things have to be true here, and each was wrong in the first cut:

          * The goodbye must actually be WRITTEN, not merely requested. It goes
            through `_publish_once(dirty=True)`, which bypasses the keepalive
            clock — an idle machine (the common shutdown case: the run just
            finished) would otherwise decline to write at all.
          * It must not race the worker. The worker is asked to stop and joined
            first, and every publish takes `_publish_lock`, so a keepalive
            already uploading cannot land AFTER the goodbye and restore a stale
            "running" payload.
          * It must not be able to hang shutdown. `upload_file` has no timeout
            and the Hub client's is None, so the write runs on a throwaway
            daemon thread that is joined with a deadline. A wedged socket costs
            us the goodbye, never the ability to exit.
        """
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            # Let an in-flight publish finish so it cannot overwrite the goodbye.
            thread.join(timeout=_WORKER_JOIN_TIMEOUT_S)
        if not goodbye or self._disabled_reason is not None:
            return
        if not load_settings()["enabled"]:
            return

        def _goodbye() -> None:
            try:
                # dirty=True: shutting down IS the event worth publishing.
                self._publish_once(dirty=True)
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                logger.debug("Final presence write failed: %s", exc)

        farewell = threading.Thread(target=_goodbye, name="presence-goodbye", daemon=True)
        farewell.start()
        farewell.join(timeout=_GOODBYE_TIMEOUT_S)
        if farewell.is_alive():
            logger.info("Presence goodbye did not complete in time; exiting anyway.")

    def mark_dirty(self) -> None:
        """Ask for a publish soon. Called from the registry change callback.

        Sets a flag and returns; it must never do I/O, because it runs on
        registry mutation paths (and the progress tick fires ~1Hz elsewhere).
        """
        with self._lock:
            self._dirty = True
        self._wake.set()

    def _take_dirty(self) -> bool:
        """Consume the dirty flag. A change arriving after this read re-sets
        both the flag and the event, so the next pass picks it up rather than
        losing it inside the publish that is already running."""
        with self._lock:
            was = self._dirty
            self._dirty = False
            return was

    # -- status ---------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            return {
                "device_id": device_id(),
                "disabled_reason": self._disabled_reason,
                "last_error": self._last_error,
                "last_write": self._last_write,
            }

    # -- internals ------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            # Wait for a change, but never longer than the keepalive.
            self._wake.wait(timeout=KEEPALIVE_INTERVAL_S)
            # Clear BEFORE reading the flag, so a change that arrives during the
            # publish below re-arms the event and is served next pass instead of
            # being cleared away unseen.
            self._wake.clear()
            dirty = self._take_dirty()
            if self._stop.is_set():
                return
            if self._disabled_reason is not None:
                return
            if not load_settings()["enabled"]:
                continue
            try:
                self._publish_once(dirty=dirty)
            except Exception as exc:  # noqa: BLE001 - the writer never dies
                # Preserve the dirty state: an event that failed to publish is
                # still an event, and must not be dropped by the retry.
                if dirty:
                    with self._lock:
                        self._dirty = True
                self._note_failure(exc)
            else:
                with self._lock:
                    self._failures = 0
                    self._last_error = None

    def _should_write(self, payload: dict, *, now: float, dirty: bool) -> bool:
        """Whether this wake-up is worth a commit.

        An idle device writes NOTHING: no active run means no liveness question
        to answer, and a commit per 10 minutes forever on every idle machine is
        exactly the Hub traffic this design promised not to generate. A change
        (mark_dirty) always writes, because that is the event worth seeing.
        """
        if dirty:
            return True
        if not has_active_runs(payload):
            return False
        return (now - self._last_write) >= KEEPALIVE_INTERVAL_S

    def _publish_once(self, *, dirty: bool = False) -> None:
        info = cached_whoami()
        username = (info or {}).get("name")
        if not username:
            return  # Signed out: nothing to publish to.
        now = time.time()
        settings = load_settings()
        # with_checkpoints=False is not an optimization here, it is required:
        # counting checkpoints goes to the Hub once per cloud/imported record,
        # serially and without a deadline. Presence needs none of those counts.
        dev_id = device_id()
        payload = build_payload(
            self._registry.list(limit=_PUBLISHED_RUN_LIMIT, with_checkpoints=False),
            now=now,
            label=settings["label"],
            # Resolved ONCE: the payload's id and the file it is written to must
            # agree, or this device cannot recognize and skip its own file.
            dev_id=dev_id,
        )
        if not self._should_write(payload, now=now, dirty=dirty):
            return

        api = self._api_factory()
        repo_id = presence_repo_id(username)
        # One publish at a time: a keepalive already uploading must not land
        # after the shutdown goodbye and restore a stale "running" payload.
        with self._publish_lock:
            api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
            api.upload_file(
                path_or_fileobj=json.dumps(payload, indent=2).encode(),
                path_in_repo=device_file_path(dev_id),
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"presence: {settings['label']}",
            )
        with self._lock:
            self._last_write = now

    def _note_failure(self, exc: Exception) -> None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        with self._lock:
            self._last_error = str(exc)
            if status in (401, 403):
                # The token cannot write. Backing off forever here would be a
                # silent lie: the toggle would still read "on".
                self._disabled_reason = "forbidden"
                logger.warning(
                    "Presence publishing disabled: the Hugging Face token cannot write "
                    "to the presence repo (%s). Sharing this device's runs is off until "
                    "a token with write access is configured.",
                    status,
                )
                return
            self._failures += 1
            first = self._failures == 1
        if first:
            logger.warning("Presence write failed (will retry quietly): %s", exc)
        # Exponential backoff, capped at an hour. Bounded sleep so stop() is
        # still responsive.
        delay = min(KEEPALIVE_INTERVAL_S * (2 ** min(self._failures - 1, 3)), 3600.0)
        self._stop.wait(timeout=delay)


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------


def _sanitize_payload(payload: object) -> dict | None:
    """A presence file reduced to a shape the UI can trust, or None to skip it.

    Everything here crossed the network from ANOTHER machine, so nothing about
    it is guaranteed — a half-written commit, an older schema, or a file some
    other tool put in the repo. Checking only the schema number (as the first
    cut did) still let a `runs` that is not a list, or run entries that are not
    objects, through to React.
    """
    if not isinstance(payload, dict) or payload.get("schema") != PRESENCE_SCHEMA:
        return None
    raw_runs = payload.get("runs")
    runs = [r for r in raw_runs if isinstance(r, dict)] if isinstance(raw_runs, list) else []
    label = payload.get("device_label")
    dev = payload.get("device_id")
    return {
        "schema": PRESENCE_SCHEMA,
        "device_id": dev if isinstance(dev, str) else None,
        "device_label": label if isinstance(label, str) and label.strip() else "unknown device",
        "updated_at": payload.get("updated_at"),
        "runs": runs,
    }


def read_board(*, now: float | None = None, api=None) -> list[dict]:
    """Every OTHER device's presence, newest activity first.

    This device is filtered out by id: its own runs are already on the board as
    local records, and listing them a second time as "somebody else's" is the
    single most confusing thing this feature could do.

    Costs one tree listing plus one download per device. That is fine at the
    handful of machines one person owns, and is bounded by _MAX_DEVICES so a
    repo that somehow accumulated many files cannot turn one page load into
    hundreds of round-trips.

    Never raises: a device whose file is missing, corrupt, or written by a
    schema we do not know is skipped, and a total failure returns an empty
    board. Presence is a convenience — it must not be able to break the jobs
    library.
    """
    now = time.time() if now is None else now
    info = cached_whoami()
    username = (info or {}).get("name")
    if not username:
        return []
    api = api or shared_hf_api()
    repo_id = presence_repo_id(username)
    mine = device_id()

    try:
        entries = list(api.list_repo_tree(repo_id, repo_type="model", path_in_repo="devices", expand=True))
    except Exception as exc:  # noqa: BLE001 - no board yet is the normal case
        logger.debug("Presence board unavailable: %s", exc)
        return []

    out: list[dict] = []
    for entry in entries[:_MAX_DEVICES]:
        path = getattr(entry, "path", "") or ""
        if not path.endswith(".json"):
            continue
        if path.rsplit("/", 1)[-1][: -len(".json")] == mine:
            continue
        # Commit time, not the payload's own clock: devices with skewed clocks
        # exist, and liveness is the one thing that must not be self-reported.
        last_commit = getattr(entry, "last_commit", None)
        last_seen = None
        stamp = getattr(last_commit, "date", None)
        if stamp is not None:
            try:
                last_seen = stamp.timestamp()
            except (AttributeError, OSError, ValueError):
                last_seen = None
        try:
            local = api.hf_hub_download(repo_id=repo_id, filename=path, repo_type="model")
            with open(local) as f:
                payload = json.load(f)
        except Exception as exc:  # noqa: BLE001 - one bad file is not a failure
            logger.debug("Skipping presence file %s: %s", path, exc)
            continue
        payload = _sanitize_payload(payload)
        if payload is None:
            continue
        if last_seen is None:
            raw = payload.get("updated_at")
            last_seen = raw if isinstance(raw, (int, float)) else None
        out.append(project_device(payload, last_seen=last_seen, now=now))

    out.sort(key=lambda d: d.get("last_seen") or 0, reverse=True)
    return out
