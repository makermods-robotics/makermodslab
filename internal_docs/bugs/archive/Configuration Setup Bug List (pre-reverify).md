# Configuration Setup Bug List

Audit date: July 14, 2026  
Audited commit: `518ca56` (`min_stable`)

Audit scope: robot records and selection, ports, calibration library, manual and automatic calibration, cameras, motor power, authentication/offline behavior, deployment entry points, and downstream configuration handoffs.

This audit is implementation-backed. “Confirmed” means the broken behavior follows directly from the current frontend/backend path; it does not mean it was exercised against physical hardware.

## Confirmed bugs

### P0 — none found

No configuration/setup defect was found that currently warrants a P0 classification.

### P1 — Failed or stopped follower auto-recalibration can delete the previous valid calibration — In progress

**Status update:** Live-confirmed on the Jetson bench 2026-07-14 (Extended Checks "cancel auto-calibration mid-run" line destroyed the pre-existing follower profile). Fix implemented 2026-07-14 — the vendored subprocess now runs with `HF_LEROBOT_CALIBRATION` redirected to a private staging dir, so its `--save` output never lands in the real library; only a fully-successful run atomically `os.replace`s the staged file into place, while failure/stop delete the staged file only (`makermodslab/auto_calibrate.py`). Unit-tested (pre-existing profile survives failure/stop/post-processing-failure; staged output still cleaned; success promotes into the library). Live revalidation pending.

**Verdict:** CONFIRMED — frontend always sends `overwrite: true` (`frontend/src/pages/Calibration.tsx:856`, `:927`); every non-success terminal path unconditionally `os.remove`s the real follower library path (`makermodslab/auto_calibrate.py:117-130` via `:279-280`, `:294-295`, `:426-428`; `FOLLOWER_CONFIG_PATH` = that dir, `makermodslab/utils/config.py:32`).
**Priority:** P1 — silent loss of the arm's last working follower profile on an ordinary cancel/failure; recoverable only by recalibrating (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. The configuration page reuses the arm's assigned calibration name and always sends `overwrite: true` for batch auto-calibration (`frontend/src/pages/Calibration.tsx:836-856`).
2. For a follower, the vendored subprocess output path is the real MakerMods Lab follower calibration-library path (`makermodslab/auto_calibrate.py:104-114`).
3. On a nonzero exit, post-processing failure, or stop, cleanup removes any file at that path without distinguishing a newly written stray file from the profile that existed before the run (`makermodslab/auto_calibrate.py:117-130`, `makermodslab/auto_calibrate.py:264-298`, `makermodslab/auto_calibrate.py:414-428`).

**Impact**

Canceling or failing an ordinary recalibration can permanently remove the arm's last working follower profile. The robot record can continue referencing the deleted name, immediately making the robot incomplete and blocking later teleoperation/recording.

**Safe reproduction/reasoning**

The existing temp-backed tests plant a follower file before failure/stop and explicitly assert that it is deleted (`tests/test_auto_calibrate.py:244-277`, `tests/test_auto_calibrate.py:280-327`). Those tests currently codify the destructive behavior; they do not distinguish “old valid profile” from “new stray output.” No hardware is required to reproduce it.

### P1 — Canceling manual calibration does not restore persistent servo calibration state — In progress

**Status update:** Live-confirmed on the Jetson bench 2026-07-14 (Extended Checks "cancel manual calibration mid-run" line left the servos' `Homing_Offset` diverged from the untouched calibration file, tripping the next session's arm-identity check). Fix implemented 2026-07-14 — snapshot-and-rollback: the homing step now snapshots each servo's `Homing_Offset`/`Min_Position_Limit`/`Max_Position_Limit` before any mutation; a cancel/error restores them best-effort before disconnect (torque left disabled), surfacing a `warning` field if any restore write fails; a successful save discards the snapshot (`makermodslab/calibrate.py`). Unit-tested (cancel restores every register; restore failure surfaces the warning; success discards the snapshot so no restore runs). Live revalidation pending.

**Verdict:** CONFIRMED — homing offsets are written to every servo before range recording (`makermodslab/calibrate.py:415-427`) and stop/error cleanup has no rollback (`makermodslab/calibrate.py:653-667`); impact is stronger than stated — the arm-identity guard fingerprints `Homing_Offset` (`makermodslab/arm_identity.py:160-172`, `:206-208`), so a cancel trips the next session's identity check.
**Priority:** P1 — a canceled calibration (common) can block the next hardware session's identity verification with no workaround but recalibration (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. Manual calibration's homing step disables torque, calls `reset_calibration()`, calculates new offsets, and writes `Homing_Offset` to every servo before range recording is complete (`makermodslab/calibrate.py:412-429`).
2. Stop/error cleanup only disconnects the device and marks the session inactive; it does not restore the calibration values that were present before the session (`makermodslab/calibrate.py:655-669`).
3. The page-leave confirmation says that leaving aborts calibration and “nothing will be saved” (`frontend/src/pages/Calibration.tsx:502-518`).

**Impact**

The servo EEPROM can already differ from the still-assigned calibration file after a cancel or error. The arm can then fail identity checks, require the old profile to be reapplied, or behave differently from what the UI says is saved.

**Safe reproduction/reasoning**

The write occurs before the range step and the cleanup path contains no rollback. This is established from control flow; physical reproduction was intentionally not attempted.

### P1 — Manual calibration permits saving grossly incomplete ranges

**Verdict:** CONFIRMED — `allComplete` only recolors the Save button, which is disabled solely on `!calibration_active` (`frontend/src/pages/Calibration.tsx:2165-2192`, disable at `:2179`); backend rejects only exact `min == max` and merely warns under 100 steps before saving (`makermodslab/calibrate.py:552-558`, `:560-573`).
**Priority:** P1 — an invalid/potentially unsafe calibration is acceptable in the primary calibration flow; downstream normalized commands interpret a tiny interval as full travel (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. The frontend defines expected SO-101 motion ranges of roughly 1,150–3,700 raw steps and computes whether every joint reached its target (`frontend/src/lib/calibrationTargets.ts:1-45`).
2. During recording, `allComplete` changes only the Save button's color/icon. The button remains enabled whenever calibration is active (`frontend/src/pages/Calibration.tsx:2176-2207`).
3. The backend rejects only an exact `min == max`. A non-full-turn range below 100 steps produces only a server warning and is still saved to the servo and file (`makermodslab/calibrate.py:551-575`, `makermodslab/calibrate.py:623-626`).

**Impact**

A few encoder ticks can be accepted as a joint's entire normalized range. That produces an invalid and potentially unsafe calibration because later normalized commands interpret the tiny interval as full travel.

**Safe reproduction/reasoning**

Provide centered min/max values that differ by a small nonzero amount. They bypass the equality check and can pass the separate centering check, after which `_complete_calibration()` writes them. No hardware is needed to prove the acceptance path.

### P1 — Manual calibration lifecycle can overlap worker threads on one singleton

**Verdict:** CONFIRMED — active check runs outside `_status_lock` with shared-field reset (`makermodslab/calibrate.py:237`, `:255-278`); stop joins only 5s then runs cleanup and reports idle with the worker possibly still alive (`makermodslab/calibrate.py:322-330`), so a new start can overlap the stale worker.
**Priority:** P1 — stale worker can use or disconnect the new session's device (competing serial access, corrupted session state); requires restart to clear (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. `CalibrationManager` stores the device, request, stop flag, event, and recorded ranges as shared singleton fields (`makermodslab/calibrate.py:147-162`).
2. `start_calibration()` checks `calibration_active` outside the status lock, resets those shared fields, and starts a worker (`makermodslab/calibrate.py:225-276`).
3. `stop_calibration_process()` waits only five seconds. If the worker is still alive, it logs a warning but runs cleanup and reports the manager idle anyway (`makermodslab/calibrate.py:306-331`).
4. A new start can then overwrite the singleton's request/device and reset `stop_calibration` while the old worker is still running. Two simultaneous FastAPI requests can also race the unlocked start check.

**Impact**

The stale worker can use or disconnect the new session's device, save against the wrong request, or corrupt the session state. On real hardware this can mean competing serial access and unpredictable cleanup.

**Safe reproduction/reasoning**

A mocked worker that remains alive longer than the stop join timeout is sufficient: stop marks the singleton inactive, and a second start is accepted while the first thread still exists. Existing double-start coverage exercises only a sequential start while the active flag remains true.

### P2 — Concurrent robot-record updates can lose independent fields or collide on the temporary file

**Verdict:** CONFIRMED — `save_robot_record` is an unlocked read-modify-write (`makermodslab/utils/config.py:351-391`); `_atomic_write_text` always uses the single `<path>.tmp` (`makermodslab/utils/config.py:104-112`); background write-backs race the config page (`makermodslab/calibrate.py:648-649`, `makermodslab/auto_calibrate.py:329-334`).
**Priority:** P2 — real but narrow timing window; a lost field is re-settable and normal single-user sequential use is unaffected (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. `save_robot_record()` performs an unlocked read-modify-write of the whole JSON record (`makermodslab/utils/config.py:386-425`).
2. `_atomic_write_text()` always uses the same `<record>.tmp` path for that record (`makermodslab/utils/config.py:109-117`).
3. Calibration completion writes port/config assignments from background workers (`makermodslab/calibrate.py:630-653`, `makermodslab/auto_calibrate.py:319-336`) while the configuration page can save ports, cameras, and motor power.

**Impact**

Two valid updates close together can each read the same old record, then the later full-file replacement loses fields written by the other update. Concurrent use of the same `.tmp` path can also make one writer's `os.replace()` fail or replace another writer's content.

**Safe reproduction/reasoning**

Two temp-backed threads can be paused after reading the same record and then allowed to write different patches. The final file necessarily contains only one thread's view of fields not included in its patch. No user cache is required.

### P2 — Auto-calibration can report success and assign a missing calibration file

**Verdict:** CONFIRMED — zero exit reports `completed` unconditionally (`makermodslab/auto_calibrate.py:266-271`); leader copy is existence-guarded, follower output never checked, record write-back proceeds regardless (`makermodslab/auto_calibrate.py:311-336`); codified by `tests/test_auto_calibrate.py:123-152` (zero-exit fake proc, no file, still `completed`).
**Priority:** P2 — misleading success and a silently incomplete robot; downstream flows refuse to start with a clear error rather than corrupting anything (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. A zero subprocess exit calls `_finalize_success()` and then unconditionally reports `completed` (`makermodslab/auto_calibrate.py:254-271`).
2. Leader copying is conditional on the source file existing, follower output is never checked, and robot-record write-back proceeds regardless (`makermodslab/auto_calibrate.py:301-336`).

**Impact**

The UI can show a successful auto-calibration while the assigned profile does not exist. The robot immediately remains incomplete and downstream hardware flows refuse to start.

**Safe reproduction/reasoning**

The current mocked zero-exit test produces no calibration file and still expects `completed` (`tests/test_auto_calibrate.py:123-152`).

### P2 — Imported calibration validation accepts unusable SO-101 profiles

**Verdict:** CONFIRMED — `validate_calibration_data` accepts any non-empty dict whose present motors carry the 5 integer fields (`makermodslab/utils/config.py:963-982`); no six-motor/ID/range-order/bounds checks; readiness (`is_robot_record_clean`) checks file existence only (`makermodslab/utils/config.py:438-471`).
**Priority:** P2 — requires a user-imported malformed file; fails at hardware open rather than producing silently wrong data (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. Import validation accepts any nonempty motor dictionary as long as every supplied entry has five integer fields (`makermodslab/utils/config.py:998-1017`).
2. It does not require the six SO-101 motors/IDs defined by the repository, unique/correct IDs, `range_min < range_max`, or valid register bounds (`makermodslab/vendor/feetech_autocal/calibration_defaults.py:40-61`).
3. Readiness validates only that the referenced file exists, not that it is readable or usable (`makermodslab/utils/config.py:473-506`).
4. The test suite's accepted “good” calibration contains only `shoulder_pan`, and its readiness test treats `{}` files as sufficient (`tests/test_utils_config.py:259-274`, `tests/test_utils_config.py:562-579`).

**Impact**

An imported partial or nonsensical profile can be assigned and make a robot appear ready, then fail or be only partially applied when LeRobot opens the arm.

**Safe reproduction/reasoning**

Upload the one-motor object used by the existing unit test, assign its name to an otherwise complete record, and create the counterpart file. The validator accepts it and readiness depends only on file existence.

### P2 — Browser navigation silently discards unsaved configuration drafts

**Verdict:** CONFIRMED — `isDirty` (`frontend/src/pages/Calibration.tsx:1194`) is guarded only by the page's Quit handler (`frontend/src/pages/Calibration.tsx:1261-1267`); `useSessionExitGuard` is armed only while manual calibration is live (`frontend/src/pages/Calibration.tsx:508-509`; listeners gated on `active` in `frontend/src/hooks/useSessionExitGuard.ts:94-152`).
**Priority:** P2 — drafts are re-enterable and only browser-level navigation (Back/reload/close) bypasses the confirmation (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. `isDirty` tracks unsaved ports, cameras, and motor power (`frontend/src/pages/Calibration.tsx:1187-1209`).
2. The discard dialog is invoked only by the page's custom Quit handler (`frontend/src/pages/Calibration.tsx:1273-1287`, `frontend/src/pages/Calibration.tsx:2359-2415`).
3. The browser-level `useSessionExitGuard` is active only while manual calibration is live (`frontend/src/pages/Calibration.tsx:502-519`, `frontend/src/hooks/useSessionExitGuard.ts:91-128`).

**Impact**

Browser Back, reload, tab close, or direct navigation outside a live manual session discards configuration drafts without the confirmation presented by the page's Quit button.

**Safe reproduction/reasoning**

Change a port, camera, or motor-power draft without pressing Save, then use browser Back or reload. No route blocker or `beforeunload` handler is active for `isDirty` alone.

### P2 — Duplicate camera names cause a configured camera to disappear from recording

**Verdict:** CONFIRMED — camera addition blocks duplicate cv2 index/deviceId but not duplicate semantic names (`frontend/src/components/recording/CameraConfiguration.tsx:139-151`); recording reduces the list into a dict keyed by `cam.name`, so a later duplicate overwrites the earlier entry (`frontend/src/pages/Landing.tsx:185-210`).
**Priority:** P2 — requires the operator to give two cameras identical names; the result is a wrong-but-not-corrupt dataset (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. Camera addition rejects duplicate physical indices/device IDs but does not reject duplicate semantic names (`frontend/src/components/recording/CameraConfiguration.tsx:112-164`).
2. Recording converts the saved camera list to a dictionary keyed by `cam.name`; a later duplicate overwrites the earlier entry (`frontend/src/pages/Landing.tsx:185-210`).

**Impact**

The configuration page can display two configured cameras while the recording request contains only one. The recorded dataset's camera features then differ from what the operator configured.

**Safe reproduction/reasoning**

Add two different physical cameras with the same name and inspect the reduced request object. JavaScript assignment to the same object key deterministically replaces the first camera.

### P3 — Offline station mode labels cached authentication as “HF not configured”

**Verdict:** CONFIRMED — `makermodslab-station` forces `HF_HUB_OFFLINE=1` (`makermodslab/scripts/makermodslab.py:481-485`, `:499-506`); auth status collapses missing-token/Hub-HTTP/OSError (incl. offline-mode errors) into one unauthenticated response with no offline field (`makermodslab/utils/hf_auth.py:118-126`); frontend has only loading/authenticated/unauthenticated (`frontend/src/contexts/HfAuthContext.tsx:12-22`).
**Priority:** P2 — lowest-priority P2, cosmetic/misleading only; the verifier's proposed P3 is folded into P2 because the ratified scale is P0/P1/P2 (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. `makermodslab-station` always sets `HF_HUB_OFFLINE=1` (`makermodslab/scripts/makermodslab.py:481-505`).
2. Auth status maps missing-token, Hub HTTP, and OS/offline failures to the same unauthenticated response and exposes no offline state (`makermodslab/utils/hf_auth.py:103-126`).
3. The frontend state model has only loading/authenticated/unauthenticated (`frontend/src/contexts/HfAuthContext.tsx:12-22`, `frontend/src/contexts/HfAuthContext.tsx:37-65`).
4. The UI displays “HF not configured” and tells the user to log in and recheck (`frontend/src/components/landing/HfAuthChip.tsx:34-49`, `frontend/src/components/landing/HfAuthDialog.tsx:46-84`).

**Impact**

A station with a valid cached token appears unconfigured simply because live verification is intentionally disabled. Rechecking cannot succeed until the process is restarted online. Local hardware flows are unaffected, but the setup state and recovery advice are misleading.

**Safe reproduction/reasoning**

With Hub access disabled, any `OSError`/Hub error follows the unauthenticated branch. The response provides no field by which the frontend could distinguish offline from missing credentials.

## Design gaps and lower-confidence risks

These items deserve design decisions or targeted tests, but the current audit did not classify them as confirmed user-facing defects.

### No global hardware-session exclusion across calibration managers

**Verdict:** CONFIRMED — three independent singletons behind independent endpoints with no cross-guard: manual (`makermodslab/server.py:1804-1826`), single-arm auto (`makermodslab/server.py:1832-1847`), batch auto (`makermodslab/server.py:1850-1869`); the legacy single-arm path is still routed and reachable.
**Priority:** P2 — overlap requires a direct API caller; the normal UI drives one path at a time (verified 2026-07-14 against the current working tree)

Manual calibration, legacy single-arm auto-calibration, and batch auto-calibration use independent managers and endpoints (`makermodslab/server.py:1803-1871`). A direct API caller can attempt overlapping sessions even though all can own serial ports and mutate calibration. The normal configuration UI chiefly uses the batch path, so the reachability and intended compatibility of the legacy single path should be confirmed before assigning severity.

### Calibration-library mutation is not guarded during an active calibration

**Verdict:** CONFIRMED-WITH-CORRECTIONS — delete/rename/upload still perform no calibration-activity check (`makermodslab/server.py:1910-1959`), but the stale-assignment fallout is partly mitigated in the current tree: delete unassigns referencing records via `clear_config_references` (`makermodslab/utils/config.py:1063-1090`) and rename repoints them (`makermodslab/utils/config.py:1017-1060`).
**Priority:** P2 — the remaining gap is the missing "does an active calibration own this profile?" guard on library mutations (verified 2026-07-14 against the current working tree)

Delete, upload, and rename routes do not check manual/auto calibration activity (`makermodslab/server.py:1911-1964`, `makermodslab/server.py:2016-2073`). Library controls also appear available on the page during a session. Renaming/deleting the target name mid-run can produce stale assignments, duplicate/resurrected names, or confusing completion results. This needs an explicit policy: disable mutations while any calibration owns the profile, or snapshot and reconcile names safely.

### macOS camera enumeration has no generic fallback after AVFoundation-helper failure

**Verdict:** CONFIRMED (static) / CANNOT-VERIFY-STATICALLY (manifestation) — the Darwin path returns the AVFoundation list directly with no OpenCV retry (`makermodslab/server.py:2329-2333`); a helper failure falls to the outer handler and yields an empty camera list (`makermodslab/server.py:2345-2347`); whether the helper actually fails on a packaged install needs a live run.
**Priority:** P2 — static gap confirmed; live macOS packaged-install run needed to see it manifest (verified 2026-07-14 against the current working tree)

The Darwin path returns the AVFoundation-derived list directly. If its helper is unavailable or fails, enumeration can be empty even when OpenCV could use a camera. Current dependency packaging may make this uncommon; validate a minimal production install before treating it as a confirmed platform bug.

## Validation performed

Focused mock/temp-backed test run:

```text
.venv/bin/python -m pytest \
  tests/test_utils_config.py \
  tests/test_server.py \
  tests/test_calibrate.py \
  tests/test_auto_calibrate.py \
  tests/test_utils_hf_auth.py \
  tests/test_devices.py \
  tests/test_open_calibration_folder.py \
  tests/test_wiggle.py -q

235 passed, 5 warnings in 4.10s
```

Passing tests do not invalidate the findings above. Several current tests explicitly accept the problematic contracts, including deleting a planted follower profile on failure, treating a zero-exit subprocess with no output file as successful, accepting a one-motor import, and treating empty calibration files as sufficient for readiness.

Repository state at the end of validation: branch `min_stable`, commit `518ca56`; tracked status clean.

## Missing regression coverage

- Preserve and restore an existing follower profile when overwrite auto-calibration fails or stops.
- Restore pre-session servo calibration/EEPROM values when manual calibration is canceled or errors.
- Prevent saving until all required joint-range targets pass, and enforce the same rule in the backend.
- Reject a new manual calibration while an earlier worker is alive, including stop-timeout and truly concurrent starts.
- Serialize/merge concurrent robot-record writes without lost updates or shared-temp collisions.
- Require a real output file before auto-calibration reports success or assigns a profile.
- Validate the complete six-motor SO-101 calibration schema, IDs, ranges, and bounds.
- Guard dirty configuration drafts on browser navigation and unload, not only the custom Quit button.
- Reject duplicate camera names before persistence and downstream dictionary conversion.
- Represent offline Hub state separately from missing/invalid authentication.
- Block or reconcile calibration-library rename/delete/upload while a relevant calibration session is active.

## Audit safety and repository changes

- No production or test files were edited.
- No files were staged and no commit was created.
- This bug-list document is the only intentional filesystem edit from the audit handoff and is gitignored under `internal_docs/`.
- No user Hugging Face cache or calibration data was modified.
- No serial port or camera stream/device was opened, and no motor commands were sent.
- No external service was called.
