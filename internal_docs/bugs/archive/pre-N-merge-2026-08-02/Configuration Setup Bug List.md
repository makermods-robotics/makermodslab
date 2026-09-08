# Configuration Setup Bug List

*Citations synced 2026-07-26 against `main` @ `ce42158`. **Only `file:line` coordinates were changed** — no prose, verdicts, severities, entry IDs or numbering were altered. References whose construct no longer exists at HEAD are marked inline `⚠ UNRESOLVED @ HEAD ce42158`.*

> **Re-verified against `main` @ `c7d9f27` on 2026-07-23.** This list folds the 23-07-26 re-audit into the
> original. It **supersedes** the pre-reverification version, archived at
> [`archive/Configuration Setup Bug List (pre-reverify).md`](archive/Configuration%20Setup%20Bug%20List%20%28pre-reverify%29.md);
> the standalone re-audit is archived at
> [`archive/Configuration Setup REVERIFIED 23-07-26.md`](archive/Configuration%20Setup%20REVERIFIED%2023-07-26.md).
> Cross-module current-state overview: [`CURRENT 23-07-26.md`](CURRENT%2023-07-26.md).
>
> **Context you need for every verdict below:**
> - The original entries were audited **2026-07-14 at `518ca56`** (on `min_stable`), then annotated with
>   fixes made on the **`andrew`** branch. **Those `andrew`-branch fixes are NOT on `main`** — the two
>   "Fix implemented 2026-07-14" status notes (auto-calibration staging dir; manual-calibration
>   snapshot-and-rollback) describe code that does not exist on `main`. Both defects are re-confirmed
>   against `main`'s current source.
> - **The redesign moved the whole configuration surface.** `frontend/src/pages/Calibration.tsx` and
>   `frontend/src/pages/Landing.tsx` no longer exist; configuration now lives in
>   `frontend/src/components/dialogs/RobotConfigDialog.tsx` (a modal window),
>   `frontend/src/components/calibration/CalibrationLibrary.tsx`, and
>   `frontend/src/components/studio/CollectPanel.tsx`. Every frontend citation in the original was stale;
>   `main`-current cites are layered on below.
> - **PRs #5–#11 are CLOSED with `merged=no`** (`mergedAt: null` on all seven). Two target entries here —
>   **#7 (T4, cross-feature mutex incl. calibration/auto-cal/wiggle)** overlaps entry 11 / NEW-2, and
>   **#11 (I4, atomic check-and-claim in `start_calibration` + exit guard)** overlaps entry 4. Neither is on
>   `main`, so `main` still contains both bugs.
> - **Numbering:** the original list used descriptive P1/P2/P3 headings with no numbers. The re-verification
>   assigned **entry numbers 1–13**; `RANKING.md` numbers the same bugs **C1–C10 + D1–D3**. Both schemes are
>   carried on each heading (they map in order — entries 1–4 = C1–C4, entries 5–13 = C5–C10 + D1–D3). New
>   findings use the **NEW-n** prefix. Nothing is renumbered.
> - Analysis is **purely static** — no hardware, no serial/camera open, no server start, no Hub call. No code
>   was modified.

Audit date: July 14, 2026 · Audited commit: `518ca56` (`min_stable`) · Re-verified: July 23, 2026 against `main` @ `c7d9f27`.

Audit scope: robot records and selection, ports, calibration library, manual and automatic calibration, cameras, motor power, authentication/offline behavior, deployment entry points, and downstream configuration handoffs.

This audit is implementation-backed. "Confirmed" means the broken behavior follows directly from the current frontend/backend path; it does not mean it was exercised against physical hardware.

**Excluded by brief:** the arm-identity guard severity (`makermodslab/arm_identity.py` warn-and-proceed). Referenced only where it is the *detector* for another defect.

## Confirmed bugs

### P0 — none in the original audit

The original audit found no P0. **Re-verification adds one: NEW-1 (P0)** — auto-calibration wipes the servos' persistent calibration registers before measuring, with no rollback on Stop/exception. This is **CURRENT P0-1**. See New findings below.

### P1 — Failed or stopped follower auto-recalibration can delete the previous valid calibration — In progress   [Re-verify entry 1 · RANKING C1]

**First identified:** 2026-07-14

**Status update:** Live-confirmed on the Jetson bench 2026-07-14 (Extended Checks "cancel auto-calibration mid-run" line destroyed the pre-existing follower profile). Fix implemented 2026-07-14 — the vendored subprocess now runs with `HF_LEROBOT_CALIBRATION` redirected to a private staging dir, so its `--save` output never lands in the real library; only a fully-successful run atomically `os.replace`s the staged file into place, while failure/stop delete the staged file only (`makermodslab/auto_calibrate.py`). Unit-tested (pre-existing profile survives failure/stop/post-processing-failure; staged output still cleaned; success promotes into the library). Live revalidation pending.

> **Verdict on `main` (2026-07-23): STILL REAL — the andrew staging-dir fix is absent.** Feeds **CURRENT
> P0-1**. `_subprocess_output_path("robot", stem)` still resolves to `CALIBRATE_BASE_PATH_ROBOTS/so_follower/<stem>.json`
> (`makermodslab/auto_calibrate.py:112-122`), byte-identical to `FOLLOWER_CONFIG_PATH` (`makermodslab/utils/config.py:32`) —
> the real library. `_remove_stray_calibration_file` (`:125-138`) `os.remove`s that path with no way to tell a
> newly written stray from a pre-existing profile, called on **all three** non-success terminal paths:
> post-processing failure (`:291-293`), nonzero exit / dropped connection (`:306-307`), stop (`:438-440`).
> The frontend still reuses the arm's assigned config name and always sends `overwrite: true`
> (`RobotConfigDialog.tsx:931-938` picks the name; `:951-956` sends `overwrite: true`).
> **Refinement over the original:** this is **follower-only** — for a leader (`device_type == "teleop"`) the
> subprocess writes to scratch `robots/so_leader/`, and the cleanup deletes only that scratch file; the real
> leader library under `teleoperators/so_leader` is untouched (`:141-149`, `:323-329`). Tests still codify the
> destructive contract (`tests/test_auto_calibrate.py:253-277`, `:280-311`, `:313-328`).

**Verdict:** CONFIRMED — frontend always sends `overwrite: true` (`frontend/src/pages/Calibration.tsx:856`, `:927`); every non-success terminal path unconditionally `os.remove`s the real follower library path (`makermodslab/auto_calibrate.py:117-130` via `:279-280`, `:294-295`, `:426-428`; `FOLLOWER_CONFIG_PATH` = that dir, `makermodslab/utils/config.py:32`). ⚠ UNRESOLVED @ HEAD ce42158 — `Calibration.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/calibration/ + dialogs/, construct not re-located.
**Priority:** P1 — silent loss of the arm's last working follower profile on an ordinary cancel/failure; recoverable only by recalibrating (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. The configuration page reuses the arm's assigned calibration name and always sends `overwrite: true` for batch auto-calibration (`frontend/src/pages/Calibration.tsx:836-856`). ⚠ UNRESOLVED @ HEAD ce42158 — `Calibration.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/calibration/ + dialogs/, construct not re-located.
2. For a follower, the vendored subprocess output path is the real MakerMods Lab follower calibration-library path (`makermodslab/auto_calibrate.py:104-114`).
3. On a nonzero exit, post-processing failure, or stop, cleanup removes any file at that path without distinguishing a newly written stray file from the profile that existed before the run (`makermodslab/auto_calibrate.py:117-130`, `makermodslab/auto_calibrate.py:264-298`, `makermodslab/auto_calibrate.py:414-428`).

**Impact**

Canceling or failing an ordinary recalibration can permanently remove the arm's last working follower profile. The robot record can continue referencing the deleted name, immediately making the robot incomplete and blocking later teleoperation/recording.

**Safe reproduction/reasoning**

The existing temp-backed tests plant a follower file before failure/stop and explicitly assert that it is deleted (`tests/test_auto_calibrate.py:244-277`, `tests/test_auto_calibrate.py:280-327`). Those tests currently codify the destructive behavior; they do not distinguish "old valid profile" from "new stray output." No hardware is required to reproduce it.

### P1 — Canceling manual calibration does not restore persistent servo calibration state — In progress   [Re-verify entry 2 · RANKING C2]

**First identified:** 2026-07-14

**Status update:** Live-confirmed on the Jetson bench 2026-07-14 (Extended Checks "cancel manual calibration mid-run" line left the servos' `Homing_Offset` diverged from the untouched calibration file, tripping the next session's arm-identity check). Fix implemented 2026-07-14 — snapshot-and-rollback: the homing step now snapshots each servo's `Homing_Offset`/`Min_Position_Limit`/`Max_Position_Limit` before any mutation; a cancel/error restores them best-effort before disconnect (torque left disabled), surfacing a `warning` field if any restore write fails; a successful save discards the snapshot (`makermodslab/calibrate.py`). Unit-tested (cancel restores every register; restore failure surfaces the warning; success discards the snapshot so no restore runs). Live revalidation pending.

> **Verdict on `main` (2026-07-23): STILL REAL — the andrew snapshot/rollback fix is absent.** `_step_homing`
> disables torque, calls `reset_calibration()` (`makermodslab/calibrate.py:421`) and writes a new `Homing_Offset`
> to every servo (`:428-429`) before range recording. Every cancel/error path (`:367-370`, `:375-378`,
> `:383-386`, `:394-402`) goes to `_cleanup_and_finish` (`:655-659`), which only disconnects and flips status
> — no register restore, no `warning` field. Compounds NEW-1 (auto-cal has the same no-rollback hole on the
> primary path).

**Verdict:** CONFIRMED — homing offsets are written to every servo before range recording (`makermodslab/calibrate.py:415-427`) and stop/error cleanup has no rollback (`makermodslab/calibrate.py:653-667`); impact is stronger than stated — the arm-identity guard fingerprints `Homing_Offset` (`makermodslab/arm_identity.py:160-172`, `:206-208`), so a cancel trips the next session's identity check.
**Priority:** P1 — a canceled calibration (common) can block the next hardware session's identity verification with no workaround but recalibration (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. Manual calibration's homing step disables torque, calls `reset_calibration()`, calculates new offsets, and writes `Homing_Offset` to every servo before range recording is complete (`makermodslab/calibrate.py:412-429`).
2. Stop/error cleanup only disconnects the device and marks the session inactive; it does not restore the calibration values that were present before the session (`makermodslab/calibrate.py:655-669`).
3. The page-leave confirmation says that leaving aborts calibration and "nothing will be saved" (`frontend/src/pages/Calibration.tsx:502-518`). ⚠ UNRESOLVED @ HEAD ce42158 — `Calibration.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/calibration/ + dialogs/, construct not re-located.

**Impact**

The servo EEPROM can already differ from the still-assigned calibration file after a cancel or error. The arm can then fail identity checks, require the old profile to be reapplied, or behave differently from what the UI says is saved.

**Safe reproduction/reasoning**

The write occurs before the range step and the cleanup path contains no rollback. This is established from control flow; physical reproduction was intentionally not attempted.

### P1 — Manual calibration permits saving grossly incomplete ranges   [Re-verify entry 3 · RANKING C3]

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL — restructured frontend, same behavior.** Backend unchanged:
> only exact `min == max` is rejected (`makermodslab/calibrate.py:554-560`); a range under 100 steps produces two
> `logger.warning` lines and is saved anyway (`:562-578`). Frontend moved but behaves identically: `allComplete`
> (from `isMotorRangeComplete`, `frontend/src/lib/calibrationTargets.ts:34-45`) only changes the "Save
> calibration" button's colour/icon; the button is disabled solely on `!calibrationStatus.calibration_active`
> (`RobotConfigDialog.tsx:1701-1744`, disable at `:1718`).

**Verdict:** CONFIRMED — `allComplete` only recolors the Save button, which is disabled solely on `!calibration_active` (`frontend/src/pages/Calibration.tsx:2165-2192`, disable at `:2179`); backend rejects only exact `min == max` and merely warns under 100 steps before saving (`makermodslab/calibrate.py:552-558`, `:560-573`). ⚠ UNRESOLVED @ HEAD ce42158 — `Calibration.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/calibration/ + dialogs/, construct not re-located.
**Priority:** P1 — an invalid/potentially unsafe calibration is acceptable in the primary calibration flow; downstream normalized commands interpret a tiny interval as full travel (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. The frontend defines expected SO-101 motion ranges of roughly 1,150–3,700 raw steps and computes whether every joint reached its target (`frontend/src/lib/calibrationTargets.ts:1-45`).
2. During recording, `allComplete` changes only the Save button's color/icon. The button remains enabled whenever calibration is active (`frontend/src/pages/Calibration.tsx:2176-2207`). ⚠ UNRESOLVED @ HEAD ce42158 — `Calibration.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/calibration/ + dialogs/, construct not re-located.
3. The backend rejects only an exact `min == max`. A non-full-turn range below 100 steps produces only a server warning and is still saved to the servo and file (`makermodslab/calibrate.py:551-575`, `makermodslab/calibrate.py:623-626`).

**Impact**

A few encoder ticks can be accepted as a joint's entire normalized range. That produces an invalid and potentially unsafe calibration because later normalized commands interpret the tiny interval as full travel.

**Safe reproduction/reasoning**

Provide centered min/max values that differ by a small nonzero amount. They bypass the equality check and can pass the separate centering check, after which `_complete_calibration()` writes them. No hardware is needed to prove the acceptance path.

### P1 — Manual calibration lifecycle can overlap worker threads on one singleton   [Re-verify entry 4 · RANKING C4]

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL — overlaps PR #11 (I4), CLOSED-unmerged.** `start_calibration`
> reads `self.status.calibration_active` **outside** `_status_lock` (`makermodslab/calibrate.py:235`), then resets
> the shared singleton fields and starts a worker (`:252-276`). `stop_calibration_process` joins for 5 s, logs
> a warning if the thread is still alive, then runs `_cleanup_and_finish(..., status="idle")` anyway (`:306-331`)
> — reporting idle while the stale worker may still hold the device. PR #11
> (`fix/i4-calibration-atomic-exclusion-and-exit-guard`) makes the check-and-claim atomic; it is closed
> unmerged, so `main` retains the race.

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

### P2 — Concurrent robot-record updates can lose independent fields or collide on the temporary file   [Re-verify entry 5 · RANKING C5]

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL.** `save_robot_record` is still an unlocked read-modify-write of
> the whole record (`makermodslab/utils/config.py:387-427`, read at `:402`, write at `:425`) and `_atomic_write_text`
> still uses the single fixed `<path>.tmp` (`:109-117`). Background writers that race the config dialog's Save:
> `calibrate.py:649-651`, `auto_calibrate.py:341-348`, plus the per-record loops in `rename_calibration_config`
> (`config.py:1099-1102`) and `clear_config_references` (`:1130-1133`).

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

### P2 — Auto-calibration can report success and assign a missing calibration file   [Re-verify entry 6 · RANKING C6]

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL.** A zero exit calls `_finalize_success()` and unconditionally
> sets `status="completed"` (`makermodslab/auto_calibrate.py:277-283`). `_finalize_success` guards the *leader copy*
> on `os.path.exists(src)` (`:326`) but never checks that a follower file was produced, and the robot-record
> write-back runs regardless (`:333-348`). `tests/test_auto_calibrate.py:123-152` is a zero-exit fake process
> that writes no file and still asserts `completed`.

**Verdict:** CONFIRMED — zero exit reports `completed` unconditionally (`makermodslab/auto_calibrate.py:266-271`); leader copy is existence-guarded, follower output never checked, record write-back proceeds regardless (`makermodslab/auto_calibrate.py:311-336`); codified by `tests/test_auto_calibrate.py:123-152` (zero-exit fake proc, no file, still `completed`).
**Priority:** P2 — misleading success and a silently incomplete robot; downstream flows refuse to start with a clear error rather than corrupting anything (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. A zero subprocess exit calls `_finalize_success()` and then unconditionally reports `completed` (`makermodslab/auto_calibrate.py:254-271`).
2. Leader copying is conditional on the source file existing, follower output is never checked, and robot-record write-back proceeds regardless (`makermodslab/auto_calibrate.py:301-336`).

**Impact**

The UI can show a successful auto-calibration while the assigned profile does not exist. The robot immediately remains incomplete and downstream hardware flows refuse to start.

**Safe reproduction/reasoning**

The current mocked zero-exit test produces no calibration file and still expects `completed` (`tests/test_auto_calibrate.py:123-152`).

### P2 — Imported calibration validation accepts unusable SO-101 profiles   [Re-verify entry 7 · RANKING C7]

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL.** `validate_calibration_data` still accepts any non-empty dict
> whose *present* motors each carry the five integer fields (`makermodslab/utils/config.py:1008-1027`) — no
> six-motor requirement, no ID uniqueness, no `range_min < range_max`, no register bounds. `is_robot_record_clean`
> still checks file *existence* only (`:474-516`). Tests still bless a one-motor profile
> (`tests/test_utils_config.py:259-274`) and `{}` calibration files as sufficient for readiness (`:493-501`).

**Verdict:** CONFIRMED — `validate_calibration_data` accepts any non-empty dict whose present motors carry the 5 integer fields (`makermodslab/utils/config.py:963-982`); no six-motor/ID/range-order/bounds checks; readiness (`is_robot_record_clean`) checks file existence only (`makermodslab/utils/config.py:438-471`).
**Priority:** P2 — requires a user-imported malformed file; fails at hardware open rather than producing silently wrong data (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. Import validation accepts any nonempty motor dictionary as long as every supplied entry has five integer fields (`makermodslab/utils/config.py:998-1017`).
2. It does not require the six SO-101 motors/IDs defined by the repository, unique/correct IDs, `range_min < range_max`, or valid register bounds (`makermodslab/vendor/feetech_autocal/calibration_defaults.py:40-61`).
3. Readiness validates only that the referenced file exists, not that it is readable or usable (`makermodslab/utils/config.py:473-506`).
4. The test suite's accepted "good" calibration contains only `shoulder_pan`, and its readiness test treats `{}` files as sufficient (`tests/test_utils_config.py:259-274`, `tests/test_utils_config.py:562-579`).

**Impact**

An imported partial or nonsensical profile can be assigned and make a robot appear ready, then fail or be only partially applied when LeRobot opens the arm.

**Safe reproduction/reasoning**

Upload the one-motor object used by the existing unit test, assign its name to an otherwise complete record, and create the counterpart file. The validator accepts it and readiness depends only on file existence.

### P2 — Browser navigation silently discards unsaved configuration drafts   [Re-verify entry 8 · RANKING C8]

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL, NARROWED.** The window now funnels **every in-window close
> vector** (Quit button, X, Esc, overlay click) through `requestClose` → discard prompt
> (`RobotConfigDialog.tsx:1338-1348`, `:2422-2442`), which is an improvement. What remains unguarded is
> browser-level and route-level leave: `useSessionExitGuard` is armed only on `manualCalibLive` (`:595-606`),
> and its `beforeunload`/`popstate`/unmount handlers are all gated on `active` (`useSessionExitGuard.ts:94-128`,
> `:134-152`, `:159-166`). A reload, tab close, or in-app route change while `isDirty` (`:1270`) silently drops
> port/camera/torque drafts.

**Verdict:** CONFIRMED — `isDirty` (`frontend/src/pages/Calibration.tsx:1194`) is guarded only by the page's Quit handler (`frontend/src/pages/Calibration.tsx:1261-1267`); `useSessionExitGuard` is armed only while manual calibration is live (`frontend/src/pages/Calibration.tsx:508-509`; listeners gated on `active` in `frontend/src/hooks/useSessionExitGuard.ts:94-152`). ⚠ UNRESOLVED @ HEAD ce42158 — `Calibration.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/calibration/ + dialogs/, construct not re-located.
**Priority:** P2 — drafts are re-enterable and only browser-level navigation (Back/reload/close) bypasses the confirmation (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. `isDirty` tracks unsaved ports, cameras, and motor power (`frontend/src/pages/Calibration.tsx:1187-1209`). ⚠ UNRESOLVED @ HEAD ce42158 — `Calibration.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/calibration/ + dialogs/, construct not re-located.
2. The discard dialog is invoked only by the page's custom Quit handler (`frontend/src/pages/Calibration.tsx:1273-1287`, `frontend/src/pages/Calibration.tsx:2359-2415`). ⚠ UNRESOLVED @ HEAD ce42158 — `Calibration.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/calibration/ + dialogs/, construct not re-located.
3. The browser-level `useSessionExitGuard` is active only while manual calibration is live (`frontend/src/pages/Calibration.tsx:502-519`, `frontend/src/hooks/useSessionExitGuard.ts:91-128`). ⚠ UNRESOLVED @ HEAD ce42158 — `Calibration.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/calibration/ + dialogs/, construct not re-located.

**Impact**

Browser Back, reload, tab close, or direct navigation outside a live manual session discards configuration drafts without the confirmation presented by the page's Quit button.

**Safe reproduction/reasoning**

Change a port, camera, or motor-power draft without pressing Save, then use browser Back or reload. No route blocker or `beforeunload` handler is active for `isDirty` alone.

### P2 — Duplicate camera names cause a configured camera to disappear from recording   [Re-verify entry 9 · RANKING C9]

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL (moved file).** `addCamera` blocks duplicates by cv2 index
> **or** browser `deviceId` only (`CameraConfiguration.tsx:141-155`); `updateCamera` (`:185-191`) applies a
> renamed camera with no uniqueness check at all. `CollectPanel.tsx:198-210` reduces the list into a dict keyed
> by `cam.name`, so a later duplicate silently replaces the earlier entry, and `:249` sends that dict as the
> recording's `cameras`.

**Verdict:** CONFIRMED — camera addition blocks duplicate cv2 index/deviceId but not duplicate semantic names (`frontend/src/components/recording/CameraConfiguration.tsx:139-151`); recording reduces the list into a dict keyed by `cam.name`, so a later duplicate overwrites the earlier entry (`frontend/src/pages/Landing.tsx:185-210`). ⚠ UNRESOLVED @ HEAD ce42158 — `Landing.tsx` deleted in the redesign (9e18b27); searched frontend/src/pages/Launchpad.tsx + components/launchpad/, construct not re-located.
**Priority:** P2 — requires the operator to give two cameras identical names; the result is a wrong-but-not-corrupt dataset (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. Camera addition rejects duplicate physical indices/device IDs but does not reject duplicate semantic names (`frontend/src/components/recording/CameraConfiguration.tsx:112-164`).
2. Recording converts the saved camera list to a dictionary keyed by `cam.name`; a later duplicate overwrites the earlier entry (`frontend/src/pages/Landing.tsx:185-210`). ⚠ UNRESOLVED @ HEAD ce42158 — `Landing.tsx` deleted in the redesign (9e18b27); searched frontend/src/pages/Launchpad.tsx + components/launchpad/, construct not re-located.

**Impact**

The configuration page can display two configured cameras while the recording request contains only one. The recorded dataset's camera features then differ from what the operator configured.

**Safe reproduction/reasoning**

Add two different physical cameras with the same name and inspect the reduced request object. JavaScript assignment to the same object key deterministically replaces the first camera.

### P3 — Offline station mode labels cached authentication as "HF not configured"   [Re-verify entry 10 · RANKING C10]

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL.** `makermodslab-station` still injects `--offline`
> (`makermodslab/scripts/makermodslab.py:499-506`), which sets `HF_HUB_OFFLINE=1` before the server imports (`:481-489`).
> `handle_hf_auth_status` still collapses `LocalTokenNotFoundError`, `HfHubHTTPError` and `OSError` into one
> unauthenticated response with no offline field (`makermodslab/utils/hf_auth.py:116-124`), and the frontend state
> model is still three-valued (`HfAuthContext.tsx:12-22`, `:37-64`) with the "Hugging Face CLI not configured"
> copy at `HfAuthDialog.tsx:51`.

**Verdict:** CONFIRMED — `makermodslab-station` forces `HF_HUB_OFFLINE=1` (`makermodslab/scripts/makermodslab.py:481-485`, `:499-506`); auth status collapses missing-token/Hub-HTTP/OSError (incl. offline-mode errors) into one unauthenticated response with no offline field (`makermodslab/utils/hf_auth.py:118-126`); frontend has only loading/authenticated/unauthenticated (`frontend/src/contexts/HfAuthContext.tsx:12-22`).
**Priority:** P2 — lowest-priority P2, cosmetic/misleading only; the verifier's proposed P3 is folded into P2 because the ratified scale is P0/P1/P2 (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. `makermodslab-station` always sets `HF_HUB_OFFLINE=1` (`makermodslab/scripts/makermodslab.py:481-505`).
2. Auth status maps missing-token, Hub HTTP, and OS/offline failures to the same unauthenticated response and exposes no offline state (`makermodslab/utils/hf_auth.py:103-126`).
3. The frontend state model has only loading/authenticated/unauthenticated (`frontend/src/contexts/HfAuthContext.tsx:12-22`, `frontend/src/contexts/HfAuthContext.tsx:37-65`).
4. The UI displays "HF not configured" and tells the user to log in and recheck (`frontend/src/components/landing/HfAuthChip.tsx:34-49`, `frontend/src/components/landing/HfAuthDialog.tsx:46-84`).

**Impact**

A station with a valid cached token appears unconfigured simply because live verification is intentionally disabled. Rechecking cannot succeed until the process is restarted online. Local hardware flows are unaffected, but the setup state and recovery advice are misleading.

**Safe reproduction/reasoning**

With Hub access disabled, any `OSError`/Hub error follows the unauthenticated branch. The response provides no field by which the frontend could distinguish offline from missing credentials.

## Design gaps and lower-confidence risks

These items deserve design decisions or targeted tests, but the current audit did not classify them as confirmed user-facing defects.

### No global hardware-session exclusion across calibration managers   [Re-verify entry 11 · RANKING D1]

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL, WORSE THAN STATED — overlaps PR #7 (T4), CLOSED-unmerged.**
> Three independent managers behind independent routes with no cross-guard (`makermodslab/server.py:1804-1871`),
> and the legacy single-arm auto path is still routed (`:1834-1849`). The original judged this P2 because "the
> normal UI drives one path at a time" — but on `main` the batch auto-calibration subprocess **deliberately
> survives closing the settings window** (`RobotConfigDialog.tsx:571-576`). The reciprocal-flag mutex covers
> only teleop ↔ record ↔ inference (`teleoperate.py:565-579` checks record+inference; `record.py:461-472`
> checks teleop+inference; `rollout.py:918-930` checks teleop+record) — none consults any calibration manager.
> See **NEW-2** for the reachable consequence. PR #7 (`fix/t4-cross-feature-mutex`) adds calibration/auto-cal/
> wiggle to the mutex; closed unmerged.

**Verdict:** CONFIRMED — three independent singletons behind independent endpoints with no cross-guard: manual (`makermodslab/server.py:1804-1826`), single-arm auto (`makermodslab/server.py:1832-1847`), batch auto (`makermodslab/server.py:1850-1869`); the legacy single-arm path is still routed and reachable.
**Priority:** P2 — overlap requires a direct API caller; the normal UI drives one path at a time (verified 2026-07-14 against the current working tree)

Manual calibration, legacy single-arm auto-calibration, and batch auto-calibration use independent managers and endpoints (`makermodslab/server.py:1803-1871`). A direct API caller can attempt overlapping sessions even though all can own serial ports and mutate calibration. The normal configuration UI chiefly uses the batch path, so the reachability and intended compatibility of the legacy single path should be confirmed before assigning severity.

### Calibration-library mutation is not guarded during an active calibration   [Re-verify entry 12 · RANKING D2]

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL (same mitigations).** Delete (`server.py:1911-1964`), upload
> (`:2016-2048`) and rename (`:2051-2073`) still perform no calibration-activity check. The mitigations noted
> originally are present: delete unassigns referencing records via `clear_config_references` (`config.py:1108-1135`)
> and rename repoints them (`:1062-1105`). Library controls are also still enabled in the UI during a live
> session — `CalibrationLibrary.tsx` has no `disabled` wiring for calibration state, and
> `RobotConfigDialog.tsx:2189-2201` renders it unconditionally.

**Verdict:** CONFIRMED-WITH-CORRECTIONS — delete/rename/upload still perform no calibration-activity check (`makermodslab/server.py:1910-1959`), but the stale-assignment fallout is partly mitigated in the current tree: delete unassigns referencing records via `clear_config_references` (`makermodslab/utils/config.py:1063-1090`) and rename repoints them (`makermodslab/utils/config.py:1017-1060`).
**Priority:** P2 — the remaining gap is the missing "does an active calibration own this profile?" guard on library mutations (verified 2026-07-14 against the current working tree)

Delete, upload, and rename routes do not check manual/auto calibration activity (`makermodslab/server.py:1911-1964`, `makermodslab/server.py:2016-2073`). Library controls also appear available on the page during a session. Renaming/deleting the target name mid-run can produce stale assignments, duplicate/resurrected names, or confusing completion results. This needs an explicit policy: disable mutations while any calibration owns the profile, or snapshot and reconcile names safely.

### macOS camera enumeration has no generic fallback after AVFoundation-helper failure   [Re-verify entry 13 · RANKING D3]

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL (static) / CANNOT VERIFY STATICALLY (manifestation).** The
> Darwin branch returns the AVFoundation list directly with no OpenCV retry (`server.py:2332-2336`); a helper
> failure is swallowed inside `_avfoundation_cameras_in_cv2_order` and returns `[]` (`:2206-2213`), so the
> endpoint answers `status: "success", cameras: []`. **What would settle it:** a packaged (non-editable,
> non-repo-venv) install on macOS, `curl localhost:8000/available-cameras` with a camera plugged in, watching
> for the "AVFoundation enumeration subprocess failed" log line. Not attempted — needs live USB cameras.

**Verdict:** CONFIRMED (static) / CANNOT-VERIFY-STATICALLY (manifestation) — the Darwin path returns the AVFoundation list directly with no OpenCV retry (`makermodslab/server.py:2329-2333`); a helper failure falls to the outer handler and yields an empty camera list (`makermodslab/server.py:2345-2347`); whether the helper actually fails on a packaged install needs a live run.
**Priority:** P2 — static gap confirmed; live macOS packaged-install run needed to see it manifest (verified 2026-07-14 against the current working tree)

The Darwin path returns the AVFoundation-derived list directly. If its helper is unavailable or fails, enumeration can be empty even when OpenCV could use a camera. Current dependency packaging may make this uncommon; validate a minimal production install before treating it as a confirmed platform bug.

---

## New findings (from the 2026-07-23 re-verification against `main`)

Ranked by the rubric (P0 = hardware safety, data loss/corruption, silent private-data publication, or main path completely broken). IDs (NEW-1..NEW-7) preserved as assigned. `NEW-n` is per-module — "Config NEW-1" is distinct from "Training NEW-1".

### NEW-1 · **P0** — A stopped or failed auto-calibration leaves the servos' persistent calibration registers wiped, with no rollback

**First identified:** 2026-07-23

> This is **CURRENT P0-1** (the top-ranked P0), compounding entry 1 / C1.

**Files:** `makermodslab/vendor/feetech_autocal/auto_calibrate_script.py`

Stage 0 of the vendored script destroys the arm's existing persistent calibration **before any measurement begins**:

- `_init_checks()` includes `("Homing_Offset", 0)` (`auto_calibrate_script.py:726`, table at `:715-727`), written to every motor at `:744`.
- `bus.write_position_limits(m, 0, 4095)` at `:751` clears the servo's soft travel limits.
- Both writes happen with `Lock=1` (`:717`) — i.e. into EEPROM — for all six motors, in `_run_init` (`:728-760`).

The calibration is only written back at the *natural end* of a fully successful run (`:906-925`). Neither terminal failure path restores anything:

- `KeyboardInterrupt` (what a Stop sends): `_graceful_stop` + `safe_disable_all`, then `return 130` (`:684-690`) — freeze/return-to-pose and torque release only.
- `Exception`: `safe_disable_all`, `return 1` (`:691-699`).

There is no snapshot of the pre-run `Homing_Offset` / `Min_Position_Limit` / `Max_Position_Limit`, and nothing in MakerMods Lab restores them afterwards.

**Why this is P0 rather than a duplicate of entry 2:**

1. Auto-calibration is now the **primary** calibration action on `main` (`RobotConfigDialog.tsx:1431-1438` — "Auto-calibrate" is the default button; "Calibrate manually" is the outline secondary), so this is the main path, not the fallback.
2. Nothing re-writes the file's calibration into the servos on a later session. MakerMods Lab connects with `calibrate=False` (`makermodslab/record.py:1283`, `:1342`), and pinned lerobot's `MotorsBus.connect()` does **not** write calibration (`.venv/.../lerobot/motors/motors_bus.py:513-527`). `_normalize` applies only the file's `range_min`/`range_max` to the raw `Present_Position` that the servo has already offset by its own `Homing_Offset` register (`motors_bus.py:850-877`). So after a cancelled auto-cal, every normalized reading and every `_unnormalize`d goal position is shifted by the destroyed offset — silently, for the whole session.
3. `Min/Max_Position_Limit` left at `(0, 4095)` removes the firmware clamp that stops the follower being commanded outside its safe travel. That is a hardware-safety regression produced by an ordinary "Stop".
4. Combined with entry 1 (which deletes the follower's calibration *file* on the very same stop), a single cancelled recalibration can destroy **both** the file and the servo state.

The excluded arm-identity guard is the only thing that notices — and it warns and proceeds.

**Cannot verify statically:** the exact post-cancel register values on hardware. The control flow above is unambiguous; a bench check (read `Homing_Offset` and position limits after a Stop mid-autocal) would confirm magnitude.

### NEW-2 · **P1** — Nothing prevents starting teleoperation, recording, or inference while arms are auto-calibrating under torque

**First identified:** 2026-07-23

**Files:** `makermodslab/server.py`, `makermodslab/teleoperate.py`, `makermodslab/record.py`, `makermodslab/rollout.py`, `frontend/src/components/dialogs/RobotConfigDialog.tsx`

This is the reachable consequence of entry 11, reachable **through the normal UI**, not only by a direct API caller:

1. The batch auto-calibration is designed to survive closing the settings window (`RobotConfigDialog.tsx:566-576`: "The batch auto-calibration subprocess is deliberately designed to SURVIVE close … it is intentionally NOT aborted on close").
2. Once the window is closed, the Launchpad/studio offers Teleoperate / Collect / Deploy as usual — no calibration-activity check anywhere in their start paths (`teleoperate.py:565-579`, `record.py:461-472`, `rollout.py:918-930`).
3. Serial-port exclusivity only saves the case where the *same* ports are involved. On a **bimanual** rig the user can auto-calibrate the right pair while teleoperating the left pair: different ports, both start, and two features drive parts of the same machine simultaneously — one moving on its own under torque with the operator's hands on the leader.

PR #7 (`fix/t4-cross-feature-mutex`, T4) adds calibration, auto-calibration and wiggle to the mutex. It is **closed unmerged**, so `main` has no such guard.

### NEW-3 · **P1** — Calibration write-back bypasses the duplicate-port guard, and the resulting record still reports `is_clean: true`

**First identified:** 2026-07-23

**Files:** `makermodslab/calibrate.py`, `makermodslab/auto_calibrate.py`, `makermodslab/utils/config.py`, `frontend/src/components/dialogs/RobotConfigDialog.tsx`

`port_slot_conflict` and `config_slot_conflict` are enforced **only** inside the `/robots/{name}` HTTP handler (`makermodslab/server.py:2450-2486`, definitions at `config.py:519-537` and `:647-664`). Both calibration write-backs call `save_robot_record` directly (`calibrate.py:649-651`, `auto_calibrate.py:341-348`), and `save_robot_record` performs no conflict check (`config.py:387-427`).

The port swap/take flow makes this reachable in the ordinary first-run sequence "Detect the port, then Calibrate":

1. `handleDetect` / `handleSelectPort` stage a swap into the **local draft only** — `persistPorts({[releasedField]: swapPort ?? "", [portField]: detected})` (`RobotConfigDialog.tsx:745-781`, `:1201-1213`). Nothing is written to the record yet.
2. `handleStartCalibration` sends `port` (the drafted value, synced at `:1169-1173`) to `/start-calibration` (`:1018-1029`). The batch path does the same via `slotPort` → `draftPort` (`:813-829`, `:931-938`).
3. On success the backend writes back **only the calibrated slot's** port (`calibrate.py:642-651`). The other half of the swap — clearing or reassigning the released slot — never reaches disk.

Result: the record now names the **same serial port on two arms**, a state the API would have rejected with 409. It is silent, because `is_robot_record_clean` checks only field-non-emptiness and calibration-file existence (`config.py:474-516`) — it never calls `port_slot_conflict`. The robot therefore shows `is_clean: true` / "All changes saved", and the failure only surfaces later as an opaque busy-port error when teleop tries to open one bus twice.

Secondary defect in the same path: the write-back violates the window's own stated contract. The Quit dialog says "Closing now discards them — **nothing was written to the robot**" (`RobotConfigDialog.tsx:2426-2430`), but a calibration run has already persisted the drafted port, and Discard cannot undo it.

### NEW-4 · **P1** — Camera identity binds to the browser `deviceId` via fuzzy name matching; the backend's stable `unique_id` is discarded

**First identified:** 2026-07-23

**Files:** `makermodslab/server.py`, `frontend/src/hooks/useAvailableCameras.ts`, `frontend/src/components/recording/CameraConfiguration.tsx`

The backend already computes the stable macOS camera identity and returns it: `/available-cameras` emits `{"index", "name", "unique_id"}` per device (`server.py:2179-2182`, `:2332-2336`), where `unique_id` is AVFoundation's `uniqueID` — the only field that distinguishes two identical camera models.

The frontend **never reads it**. `useAvailableCameras` types the backend payload as `{index, name?, available}` (`useAvailableCameras.ts:80-85`) and binds each backend index to a browser `deviceId` by **fuzzy label matching** — exact, then prefix, then either-contains — with a first-come-first-served `used` set (`:87-105`). `grep -rn "unique_id" frontend/src/` returns no hits. Two cameras of the same model have the *same* `localizedName`, so all three tiers tie and the pairing is decided by array order: backend cameras are `uniqueID`-sorted, browser devices come in `enumerateDevices()` order, and the two orders are unrelated.

Downstream, the (possibly mis-bound) `device_id` is persisted on the camera record and used as the authority to **rewrite `camera_index`** on every enumeration change (`CameraConfiguration.tsx:93-109`).

Consequences: the operator names a camera from a preview driven by `deviceId` while the recorder opens `camera_index`, so "wrist" and "front" can be swapped silently, and the binding can flip between sessions. The recorded dataset's image keys then describe the wrong physical camera — a data-correctness failure that survives into training. This is precisely the rig documented as three same-model USB cameras. Windows/Linux return no `unique_id` at all (`server.py:2230-2248`, `:2278-2310`), so there is no stable identity to fall back on — and their names are equally non-unique across identical models.

**Cannot verify statically:** the actual mis-binding, which needs two same-model cameras on a live machine. The absence of any `unique_id` consumer in `frontend/src/` is verifiable and confirmed.

### NEW-5 · **P2** — `clamp_motor_power` raises on NaN/Infinity, contradicting its documented "never raises" contract, and can hide every robot from the UI

**First identified:** 2026-07-23

**File:** `makermodslab/utils/config.py:300-309`

The docstring promises "Anything non-numeric … falls back to `DEFAULT_MOTOR_POWER` rather than raising, so a corrupted record can never block a session start." But the guard is `isinstance(value, (int, float))`, which NaN and Infinity satisfy, and `int(value)` is evaluated **first** inside the clamp expression: `int(float("nan"))` raises `ValueError`, `int(float("inf"))` raises `OverflowError`.

Python's `json.load` accepts the bare literals `NaN`, `Infinity`, `-Infinity`, so a record on disk carrying one makes `get_robot_record` raise past its `except (json.JSONDecodeError, OSError)` handler (`:340-369`), which propagates through `list_robot_records` (`:372-384`) to `get_robots`, whose blanket `except Exception` returns `{"status": "error", "robots": []}` (`server.py:2382-2390`). **One malformed record hides every robot in the UI**, not just its own. Reachability is low (MakerMods Lab's own writers can't produce it) — hence P2. Fix is a one-line reorder plus a `math.isfinite` check.

### NEW-6 · **P2** — Auto-calibrated leader runs leave a permanent scratch copy in `robots/so_leader/`

**First identified:** 2026-07-23

**File:** `makermodslab/auto_calibrate.py:313-329`

On a successful leader auto-calibration, `_finalize_success` copies `robots/so_leader/<stem>.json` → `teleoperators/so_leader/<stem>.json` (`:323-329`) but never removes the source. The scratch dir accumulates stale leader calibrations that are invisible to the library UI (`get_calibration_configs` reads only `LEADER_CONFIG_PATH`/`FOLLOWER_CONFIG_PATH`, `server.py:1874-1908`) but *are* readable by anything that walks `calibration/robots/`. Worse, they go stale: a later library rename/delete of the leader config (`config.py:1062-1135`) never touches the scratch copy, so an out-of-date profile persists under the old name indefinitely.

### NEW-7 · **P2** — `ports/{leader,follower}_port.txt` is dead persistence, and `/robot-port/{robot_type}` has no caller

**First identified:** 2026-07-23

**Files:** `makermodslab/utils/config.py:34-37`, `:246-265`; `makermodslab/server.py:2356-2361`

Nothing in `makermodslab/` writes `LEADER_PORT_FILE` or `FOLLOWER_PORT_FILE` — the only references are the constants and the reader `get_saved_robot_port`. `get_default_robot_port` therefore always falls through to the hardcoded `"COM3"` / `"/dev/ttyUSB0"` (`:258-265`), and `grep -rn "robot-port" frontend/src/` has no hits, so the endpoint is unreachable from the UI. Ports live on the robot record now. Stale surface area rather than a live defect, but `MakerLab/CLAUDE.md` still documents these files as "last-used serial ports", which will mislead the next reader.

---

## New findings (from the 2026-07-24 working session)

Entries from a 2026-07-24 working session against `main` @ `ce42158`. IDs continue this module's `NEW`-series;
nothing above is renumbered.

### NEW-8 · **P2** — The `lelab_biso` → `makermodslab_biso` staging rename has no migration, unlike its two siblings

**First identified:** 2026-07-24

**Files:** `makermodslab/utils/config.py:42-51`; `makermodslab/models.py:295-308`; `makermodslab/jobs.py:986-1035`

The package rename (`5821334` "rename lelab package/command to makerlab", finished in `257580f`) moved the
bimanual calibration staging root: `MAKERMODSLAB_BISO_STAGING_PATH` is now
`~/.cache/huggingface/lerobot/makermodslab_biso` (`config.py:51`). `grep -rn "lelab_biso" makermodslab/ tests/
frontend/src/` returns **nothing** — the old name survives only in git history — and no code migrates or
cleans a pre-rename `lelab_biso/`.

That is the odd one out. The rename's two sibling directories both got one:

- `_local_models_root()` (`models.py:295-308`) documents *"Existing installs used `lelab_models`. Move that
  directory on first use"* and does exactly that; `tests/test_models.py` has
  `test_local_models_root_migrates_pre_rebrand_cache` covering it.
- `JobRegistry._migrate_legacy_cwd_jobs()` (`jobs.py:986-1035`) is a documented, idempotent one-shot move for
  the legacy `<cwd>/outputs/train/`.

Impact is bounded: staging is a *copy* of library files, rewritten unconditionally every session
(`config.py:44-51`), so a stale `lelab_biso/` is orphaned disk rather than lost data. It is invisible to the
calibration-library UI (which reads only `LEADER_CONFIG_PATH` / `FOLLOWER_CONFIG_PATH`) yet readable by
anything walking the cache — the same shape as **NEW-6**'s scratch-copy accumulation.

**Discrepancy with the session note that prompted this entry:** the note reported ~8 orphaned bimanual setups
still sitting in `lelab_biso/` on this machine. A read-only listing of
`~/.cache/huggingface/lerobot/` on 2026-07-24 shows **`makermodslab_biso/` present and no `lelab_biso/` at all**,
so the orphan is not reproducible on this install (which may simply post-date the rename). The code fact — a
rename with no migration and no cleanup, where both siblings have one — is what is filed here; any
pre-rename install remains the open question. *Cannot verify statically:* whether any live install still
carries the directory.

---

## Validation performed (original audit)

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

Repository state at the end of the original validation: branch `min_stable`, commit `518ca56`; tracked status clean.

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

## Context: contradictions to the original brief (resolved during re-verification)

1. **PR state.** The re-verification brief described #5–#11 as "OPEN". They are **CLOSED without merge** (all seven verified; `mergedAt: null` for every one). Operative conclusion unchanged and stronger: not merged, no longer on track.
2. **Audited branch.** The original document records "branch `min_stable`, commit `518ca56`", with the two fix write-ups apparently describing `andrew` work. Either way the fixes are absent from `main`.
3. **File paths.** `frontend/src/components/calibration/` contains only `CalibrationLibrary.tsx` and `ImportCalibrationButton.tsx` — the calibration *flow* lives in `RobotConfigDialog.tsx`. `makermodslab/wiggle.py` is documented in `CLAUDE.md` as legacy, but it is live: `/wiggle` (`server.py:2124-2127`) is wired to the "Wiggle" button (`RobotConfigDialog.tsx:2039-2056`) and drives the gripper.

## Audit safety and repository changes

- No production or test files were edited.
- No files were staged and no commit was created.
### NEW-9 · **P2** — The Quality workflow has failed on `main` for days and gives no signal, because it lints the whole repo including vendored kernel source

**First identified:** 2026-07-25

**Files:** `.github/workflows/quality.yml`; `.pre-commit-config.yaml`; `pyproject.toml` (bandit config)

`gh run list --workflow Quality` shows **failure on every run since at least 2026-07-23**, on `main` and on
every branch — `baafef1`, `7af3359`, `0d4a96e`, `5711668`, `b32f80d`, and each of the `fix/i*` / `fix/t*`
branches. The check is a standing red light, so it carries no information about any individual PR and will
hide a real failure when one arrives.

The cause is the action's `extra_args: --all-files`: it lints the **entire repository** rather than the diff,
so every PR inherits the whole repo's debt. A frontend-only PR fails on `bandit`, `ruff-format` and a Markdown
prettier hook it cannot possibly have provoked.

**None of the failures are defects.** Verified by running `.venv/bin/pre-commit run --all-files` on
2026-07-25:

| hook | finding | assessment |
|---|---|---|
| `end-of-file-fixer`, `trailing-whitespace`, `ruff-format`, `prettier` | rewrote 20 files (12 reformatted by ruff) | auto-fixable; pure accumulated drift |
| `typos` | ~100 hits | **false positives** — see below |
| `zizmor` | 3 findings (2 info, 1 medium), all in `build_frontend.yml` | deliberate design |
| `bandit` | 2 findings | both already annotated as intentional |
| `ruff`, `pyupgrade`, `detect-secrets`, `mypy` | — | pass |

**`typos`** — nearly every hit is in `jetson/uvcvideo-mjpg/src/`, the **vendored Linux kernel uvcvideo
driver**: `parm`, `baSourceID`, `iterm`, `MODULE_PARM_DESC`, `bPreferedVersion`, `OT`, `ba`. These are kernel
API identifiers; renaming them breaks the build. `metalness` (×3 in `reference/` and `reference-design/`) is a
three.js material property. Only three hits are in first-party code and all are benign: a regex
`match="[Ff]ine-tun"` (flags `ine`) in `tests/test_runners_hf_cloud.py:248`, a lambda parameter `fo` in
`tests/test_arm_identity.py:572`, and the word "mis-bind" in an `InferenceModal.tsx` comment.

**`zizmor`** — `artipacked` asks for `persist-credentials: false` on the checkout in `build_frontend.yml`, but
that workflow's own inline comment states it needs the credential so the final `git push` can commit the
rebuilt `dist` back. The two `template-injection` findings are Low-confidence expansions of
`steps.app-token.outputs.app-slug` from a GitHub App token step.

**`bandit`** — `B104` (bind to `0.0.0.0`) at `makermodslab/scripts/makermodslab.py:312` is the documented `--lan`
headless-station mode, and the line **already carries `# noqa: S104`**; bandit doesn't read ruff's `noqa` and
wants `# nosec`. `B606` is `os.startfile` on the Windows branch of `utils/system.py:48`.

**Fix is configuration, not code.** Suggested order:

1. **Scope the workflow to changed files** (drop `--all-files`) so the check becomes meaningful immediately and
   PRs stop inheriting a vendored kernel driver's spelling. Smallest change, biggest signal gain.
2. Add `extend-exclude` for `jetson/uvcvideo-mjpg/` plus `extend-words` for `metalness`/`fo`/`ine` to the typos
   config; swap the two `# noqa` → `# nosec` (or add a bandit skip); suppress the three zizmor findings on
   `build_frontend.yml` with a reason.
3. Run the auto-fixers once and commit the result as a formatting-only change, so the drift stops accumulating.

Until at least (1) or (3), a green Quality check is unreachable and reviewers will keep learning to ignore it.

- This bug-list document is the only intentional filesystem edit; it is gitignored under `internal_docs/`.
- No user Hugging Face cache or calibration data was modified.
- No serial port or camera stream/device was opened, and no motor commands were sent.
- No external service was called.
