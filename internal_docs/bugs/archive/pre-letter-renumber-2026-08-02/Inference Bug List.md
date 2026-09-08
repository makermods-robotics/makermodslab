# Inference Bug List

*Citations synced 2026-07-26 against `main` @ `ce42158`. **Only `file:line` coordinates were changed** — no prose, verdicts, severities, entry IDs or numbering were altered. References whose construct no longer exists at HEAD are marked inline `⚠ UNRESOLVED @ HEAD ce42158`.*

> **Re-verified against `main` @ `c7d9f27` on 2026-07-23.** This list folds the 23-07-26 re-audit into the
> original. It **supersedes** the pre-reverification version, archived at
> [`archive/Inference Bug List (pre-reverify).md`](archive/Inference%20Bug%20List%20%28pre-reverify%29.md);
> the standalone re-audit is archived at
> [`archive/Inference REVERIFIED 23-07-26.md`](archive/Inference%20REVERIFIED%2023-07-26.md).
> Cross-module current-state overview: [`CURRENT 23-07-26.md`](CURRENT%2023-07-26.md).
>
> **Context you need for every verdict below:**
> - The original entries were audited **2026-07-14 against `518ca56`** (the **`andrew`** branch). Re-verification
>   target is **`main` @ `c7d9f27`** (post-redesign). **`andrew`-branch fixes are NOT on `main`.**
> - **The redesign moved the inference surface.** `frontend/src/pages/Inference.tsx` no longer exists — the page
>   became a dialog (`InferenceSessionContext.tsx` + `InferenceSessionDialog.tsx`), and there is now a **second**
>   launch+stop surface, `components/studio/DeployPanel.tsx` ("ported VERBATIM"). Every frontend finding must be
>   checked on **two** files; several reproduce on both. All original frontend `file:line` cites are stale;
>   `main`-current cites are layered on.
> - **PRs #5–#11 are CLOSED with `merged=no`.** Three target entries here — **#9 = I1, #10 = I5, #11 = I4** —
>   none merged, so `main` carries all three.
> - **Numbering:** the original used descriptive P1/P2/P3 headings; `RANKING.md` numbers them **I1–I12** (existing
>   findings, in order) **+ G1–G4** (design gaps). Those IDs are carried on each heading; new findings use
>   **N1–N10**. Nothing is renumbered.
> - **Excluded per brief (context only):** the arm-identity warn-and-proceed severity, and the lerobot 0.6.0-
>   checkpoint-on-0.5.2-runtime `draccus`/`pretrained_revision` constraint.
> - Read-only static audit — no cameras, serial ports, robots, Hub tokens, or network. No code modified.

**Audit date:** July 14, 2026 · **Audited commit:** `518ca56` · **Re-verified:** July 23, 2026 against `main` @ `c7d9f27`.
**Scope:** Inference launch, model/checkpoint resolution, robot and camera preparation, subprocess lifecycle, stop/leave safety, status/log reporting, and model deletion interactions.

## Status and severity key

- **P0:** hardware-safety — an arm can be driven, or left energized, with no working stop path.
- **P1:** High-impact hardware-safety, lifecycle, or mutual-exclusion failure.
- **P2:** Major feature/pathway failure without an unavoidable catastrophic outcome.
- **P3:** Lower-impact correctness, reporting, or compatibility defect.

## Re-verification verdict key (2026-07-23)

- **STILL REAL** — reproduces in `main`'s code as written.
- **FIXED** — `main` no longer has the defect.
- **CANNOT VERIFY STATICALLY** — needs a runtime/hardware observation.

## Confirmed findings

### I1 · P1 — A stopped startup worker can keep touching hardware and corrupt a newer session

**First identified:** 2026-07-14

**Status:** Confirmed

> **Verdict on `main` (2026-07-23): STILL REAL (one sub-case FIXED; a false invariant now written into a
> comment).** State is still global and un-keyed (`rollout.py:92-114`); `_set_phase` writes into whatever meta
> is present (`:158-166`); `_pump_stdout` writes module-global `_inference_rollout_started_at` with no session
> check (`:189-196`); `_report_download_progress` mutates whichever meta exists (`:283-290`); `_fail_startup`
> carries **no session token** and clears whichever session is active — it only checks `if not inference_active:
> return` (`:733-741`). Cancellation is still checked only **after** `_prepare_robot` returns (`:786-788`,
> `:803-805`); nothing inside `_prepare_robot` (`:651-718`) consults the cancel event, so a stop landing there
> lets the worker keep opening follower buses and rewriting `Torque_Limit`/`Goal_Velocity` (`:695-696`,
> `:712-716`). Stop with no process still clears state and returns success without joining the worker
> (`:1020-1033`). **Fixed sub-case:** the subprocess commit is now under `_state_lock` with a re-check of the
> cancel event and `inference_active` (`:852-888`) — a stale worker can no longer overwrite a *newer* session's
> `_inference_proc`; the P0-adjacent untracked-child variant is closed on that path (replaced by **N2**). **New,
> sharper evidence:** `:1021-1027` now *asserts as an invariant* "no robot touched here" via a download-first
> ordering, which is false whenever the stop lands during phase 2 (`_prepare_robot`). *Overlaps PR #9
> (CLOSED-unmerged).* This is part of **CURRENT P0-3**.

**Verdict:** CONFIRMED-WITH-CORRECTIONS — session-token-less `_fail_startup` clears whichever session is currently active (makermodslab/rollout.py:715-741), cancel is checked only after `_prepare_robot` returns (makermodslab/rollout.py:787), stop-with-no-proc clears state without joining the worker (makermodslab/rollout.py:998-1011); correction: the preflight never enables torque — `_open_follower` releases with `disconnect(disable_torque=False)` (makermodslab/rollout.py:420-435), so the "motor-power writes" are RAM `Torque_Limit`/`Goal_Velocity` priming, not motion, and the clobber fires only on the stale worker's raise path.
**Priority:** P1 — state/mutex-integrity break, P0-adjacent in the untracked-child subcase; no torque is enabled by the stale worker — RAM register writes only (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. Inference A starts and enters model resolution or follower preflight.
2. The user stops A before `_inference_proc` is installed.
3. Stop sets A's cancel event, immediately clears the global inference state, and reports success without waiting for A's startup thread.
4. Inference B can now claim the same global slot.
5. A can continue an already-entered `_prepare_robot` call, including serial-bus access and motor-power writes. Cancellation is checked only after the complete preflight returns.
6. If stale A then fails, `_fail_startup` sees only that some inference is active and clears B's global process/state. If B already spawned, its child can continue without a tracked process handle.

**Impact**

- Hardware can still be touched after Stop reports that inference stopped.
- A fresh session can be marked failed by the previous session.
- A live policy subprocess can become untracked, reopening the mutex while the arm may still be driven.

**Evidence** — `makermodslab/rollout.py:95-109`, `:153-200`, `:271-285`, `:717-743`, `:765-790`, `:645-714`, `:984-1015`.

### I2 · P1 — Stop reports success and loses the process handle even when termination fails

**First identified:** 2026-07-14

**Status:** Confirmed

> **Verdict on `main` (2026-07-23): STILL REAL.** `rollout.py:1035-1052`: `terminate()` / `wait(timeout=5)` /
> `kill()` / `wait()` are wrapped in one `try/except Exception` that only logs (`:1043-1044`); state is then
> cleared unconditionally (`:1046-1051`) and `{"success": True, …}` returned (`:1052`) without verifying the
> child exited. The only handle is discarded. Aggravates N1 (SIGKILL case still returns 200).

**Verdict:** CONFIRMED — `terminate()`/`wait()`/`kill()` exceptions are caught and only logged, then state is cleared unconditionally and `success: true` returned without verifying the child exited (makermodslab/rollout.py:1013-1030).
**Priority:** P1 — stop-path weakening: the only process handle is discarded without confirming the child is dead, so a still-running rollout becomes untrackable and the feature mutex reopens (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. A rollout subprocess is active.
2. `terminate()`, timed `wait()`, `kill()`, or final `wait()` raises.
3. The exception is logged but not returned.
4. The backend clears `_inference_proc`, marks inference inactive, and returns `success: true` without verifying the child exited.

**Impact** — An autonomous rollout may keep driving while the UI and backend report idle. The only process handle is discarded, and another robot-driving feature can be started. **Evidence:** `makermodslab/rollout.py:1017-1034`.

### I3 · P1 — A failed explicit Stop permanently disarms the page-leave safety guard

**First identified:** 2026-07-14

**Status:** Confirmed

> **Verdict on `main` (2026-07-23): STILL REAL, relocated.** The logic moved to `InferenceSessionDialog.tsx`:
> `markHandled()` still called **before** the awaited stop request (`:235`, request at `:237`); the catch
> (`:239-246`) re-arms nothing; `useSessionExitGuard`'s latch re-arms only on an `active` false→true transition
> (`useSessionExitGuard.ts:87-89`), which never happens while the run stays active; every leave vector
> short-circuits on the latch (`:98`, `:104-105`, `:139`, `:161`). The hung-run watchdog's one-shot
> `stopRequestedRef` is set before the request (`:200`, call at `:208`) and never reset when `stopIfHung`
> swallows the failure (`:118-124`).

**Verdict:** CONFIRMED — `markHandled()` runs before the awaited stop request with no re-arm in the catch path (frontend/src/pages/Inference.tsx:217-227); the latch re-arms only on an `active` false→true transition (frontend/src/hooks/useSessionExitGuard.ts:87-89); the hung-run watchdog's `stopRequestedRef` is likewise never reset on failure (frontend/src/pages/Inference.tsx:185). ⚠ UNRESOLVED @ HEAD ce42158 — `Inference.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/inference/ + studio/DeployPanel.tsx, construct not re-located.
**Priority:** P1 — the automatic leave-stop (back/unmount/reload/page-hide) is disarmed after one failed Stop while the policy keeps running; the manual Stop button still works on retry (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. The user confirms Stop.
2. The page calls `markHandled()` before sending the stop request.
3. The request fails while inference remains active.
4. The guard's one-way latch remains handled because `active` never changes from true to false and back to true.
5. Back navigation, component unmount, reload, and page hide skip their stop attempt.

**Impact** — After a transient backend or network failure, the operator can leave the page without another stop attempt while the policy continues running. **Evidence:** `frontend/src/pages/Inference.tsx:213-227`, `:114-122`, `:173-194`; `frontend/src/hooks/useSessionExitGuard.ts:65-89`, `:91-166`. ⚠ UNRESOLVED @ HEAD ce42158 — `Inference.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/inference/ + studio/DeployPanel.tsx, construct not re-located.

### I4 · P1 — Robot-driving feature exclusion is not atomic and calibration is omitted

**First identified:** 2026-07-14

**Status:** Confirmed

> **Verdict on `main` (2026-07-23): STILL REAL, and broader than audited.** Inference reads the other features'
> flags while holding only its own `_state_lock` (`rollout.py:917-935`); teleop and recording do the same under
> their own locks. **Inference never mentions calibration at all** (no `calibration_active` reference in
> `rollout.py`). **Correction/extension:** the audit scoped the calibration case to manual `calibrate.py:235`;
> `main` shows **auto-calibration has the same hole** (`auto_calibrate.py:210`, `:518-519` guard only against
> another auto-cal). Auto-cal is the sharper case — it drives the arm under torque and writes servo EEPROM —
> and can be admitted on the same follower port as a live inference run. *Overlaps PR #11 (CLOSED-unmerged).*
> Same hardware-arbiter family as Teleop T4 / Recording R9 / Config entry 11.

**Verdict:** CONFIRMED-WITH-CORRECTIONS — each feature guards with its own separate lock (makermodslab/rollout.py:111, makermodslab/teleoperate.py:102, makermodslab/record.py:221) and checks the other flags under only its own lock (makermodslab/rollout.py:898-916, makermodslab/teleoperate.py:568-582, makermodslab/record.py:461-474); calibration checks only `calibration_active` (makermodslab/calibrate.py:237) and no feature checks calibration; correction: OS-level serial exclusivity makes the second `bus.connect()` fail, so true concurrent driving of one port does not occur — the realistic outcome is unguarded contention and a confusing failed start.
**Priority:** P1 — driven by the calibration-vs-inference same-port case (calibration moves the arm and can stamp EEPROM while inference is admitted on the same follower port) (verified 2026-07-14 against the current working tree)

**Broken pathway**

Inference, teleoperation, and recording each protect their own active flag with a different lock. Two concurrent starts can interleave: inference holds its lock and reads teleoperation as inactive; teleoperation holds its lock and reads inference as inactive; each sets its own active flag. There is no single lock/coordinator covering the check-and-claim across features. Calibration is not checked at all by inference start, and calibration start checks only for another calibration.

**Impact** — Two workers can contend for the same follower serial port or attempt incompatible hardware operations concurrently. **Evidence:** `makermodslab/rollout.py:899-930`, `makermodslab/teleoperate.py:101-102`, `:572-596`, `makermodslab/record.py:214-216`, `:463-476`, `makermodslab/calibrate.py:232-275`, `makermodslab/server.py:1803-1813`.

### I5 · P1 — FastAPI shutdown does not stop an active inference subprocess

**First identified:** 2026-07-14

**Status:** Confirmed

> **Verdict on `main` (2026-07-23): STILL REAL.** `server.py:2549-2558`: the shutdown handler stops only the
> broadcast thread; no inference cleanup, no reference to `handle_stop_inference`. The child is still spawned
> without `start_new_session` (`rollout.py:829-835`), so a terminal Ctrl-C reaches it via the shared process
> group, but a `--reload` worker restart, a SIGTERM to the uvicorn pid, or a programmatic shutdown orphans it.
> *Overlaps PR #10 (CLOSED-unmerged).*

**Verdict:** CONFIRMED — the shutdown handler stops only the broadcast thread, never the inference lifecycle (makermodslab/server.py:2526-2535); the child is spawned without `start_new_session` (makermodslab/rollout.py:813-819), so it shares the process group and a terminal Ctrl-C does reach it — the orphan arises on `--reload` worker restart, a SIGTERM to the uvicorn pid, or programmatic shutdown.
**Priority:** P1 — orphaned autonomous policy with no backend or UI stop path, bounded only by `--duration` and `return_to_initial_position`; the trigger is common in dev (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. MakerMods Lab has spawned `lerobot.scripts.lerobot_rollout` as a child process.
2. FastAPI begins normal shutdown or reload.
3. The shutdown handler stops only the broadcast manager and never calls the inference stop lifecycle or signals the child.

**Impact** — The server can disappear while the policy subprocess remains alive, leaving no backend or UI path to track and stop it. **Evidence:** `makermodslab/rollout.py:793-883`, `makermodslab/server.py:2540-2549`.

### I6 · P2 — A model downloaded for offline use is still launched through the Hub

**First identified:** 2026-07-14

**Status:** Confirmed

> **Verdict on `main` (2026-07-23): STILL REAL, now on 2 surfaces.** `importSourceForModel` still returns
> `model.hf_repo_id ?? model.path ?? model.id` (`inferenceLaunch.ts:47-49`), preferring the Hub id over the
> downloaded local checkpoint. A non-directory source registers as Hub-backed and lists checkpoints through the
> Hub API (`jobs.py:1317`, `:774`). New on `main`: the studio Deploy panel takes the same path
> (`DeployPanel.tsx:321`) — two entry points.

**Verdict:** CONFIRMED — `importSourceForModel` returns `hf_repo_id ?? path ?? id`, preferring the Hub id over the local checkpoint path (frontend/src/lib/inferenceLaunch.ts:48); a non-directory source is registered Hub-backed (makermodslab/jobs.py:1305-1313) whose checkpoint listing calls `api.list_repo_files` (makermodslab/jobs.py:779-788); only bites when `findJobForModel` finds no covering registry record (frontend/src/lib/inferenceLaunch.ts:21-38).
**Priority:** P2 — the offline promise is broken but the workaround is being online, and a downloaded model already covered by a registry job launches locally (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. The user adds a Hub model and selects "Download to this machine now."
2. The model listing correctly carries both a local checkpoint `path` and `hf_repo_id`.
3. When no existing job record matches, launch chooses `hf_repo_id` before `path`.
4. Imported-job registration treats the source as a Hub repository, calls Hub listing, and stores a Hub-backed pseudo-job.
5. Checkpoint selection and inference resolution remain network-dependent instead of using the downloaded directory.

**Impact** — The explicit promise that the downloaded model works offline is false. **Evidence:** `frontend/src/components/landing/AddModelFromHubDialog.tsx:98-110`, `frontend/src/lib/inferenceLaunch.ts:40-48`, `frontend/src/hooks/useInferenceLaunch.tsx:44-60`, `makermodslab/models.py:600-652`, `makermodslab/jobs.py:1288-1327`, `:774-789`.

### I7 · P2 — A selected local model can be deleted during active startup

**First identified:** 2026-07-14

**Status:** Confirmed

> **Verdict on `main` (2026-07-23): STILL REAL.** The meta seeded at claim time carries only `phase` +
> `policy_ref` (`rollout.py:945`); `policy_path` is written only inside the subprocess-commit block (`:862-876`,
> key at `:867`). `inference_in_use_path()` returns that field (`:996-999`), so it is `None` for the whole
> download + preflight window, and `models._model_in_use` treats a missing path as permission to delete
> (`models.py:977-979`). See **N3** for a second, unguarded deletion route (`DELETE /jobs/{id}`) that makes this
> materially worse (bypasses the guard for the *whole* run, not just startup).

**Verdict:** CONFIRMED — `policy_path` is written to the meta only at subprocess commit (makermodslab/rollout.py:846-851); during download/preflight the meta holds only `phase`/`policy_ref` (makermodslab/rollout.py:926), so `inference_in_use_path()` returns None (makermodslab/rollout.py:965-977) and `_model_in_use` permits deletion (makermodslab/models.py:974-976).
**Priority:** P2 — narrow concurrent window (another tab/API caller during startup); concrete harm is limited to local-only checkpoints (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. Inference claims the active slot and resolves a local checkpoint.
2. It proceeds through follower staging and hardware preflight.
3. Before subprocess commit, `policy_path` exists only in the startup worker's local variable.
4. `inference_in_use_path()` returns `None` because metadata does not receive `policy_path` until after preflight and spawn.
5. Another tab or API caller can delete the selected model during this window.

**Impact** — Startup can touch and configure the follower, then fail to load or run because its checkpoint was removed. **Evidence:** `makermodslab/rollout.py:776-800`, `:836-862`, `:969-981`, `makermodslab/models.py:962-987`.

### I8 · P2 — Overlapping status polls can consume and erase a terminal failure

**First identified:** 2026-07-14

**Status:** Confirmed → **FIXED (backend) on `main`**

> **Verdict on `main` (2026-07-23): FIXED (backend); two residuals.** The one-shot terminal payload is gone.
> `_last_result` is now documented as idempotent and *kept* until the next start claims the slot
> (`rollout.py:97-104`); the idle branch returns a copy without clearing (`:1112-1113`), finalisation stores and
> returns a copy (`:1138-1154`), and only `handle_start_inference` clears it (`:948`). Residuals, now cosmetic:
> the frontend still has no in-flight guard on its 1 Hz interval (`InferenceSessionDialog.tsx:220-221`) and sets
> `doneRef` only after two awaits — worst outcome is a duplicated failure toast, not a swallowed error; and
> finalisation still mines the log (`:1136`) without joining the stdout pump thread (`:892-897`), narrowed by
> the pump's per-line `flush()` (`:181-183`). **This is the only FIXED verdict in the module.**

**Verdict:** CONFIRMED — `setInterval(tick, POLL_MS)` has no in-flight guard (frontend/src/pages/Inference.tsx:206) and `doneRef` is set only after the awaited log fetch (frontend/src/pages/Inference.tsx:143 then :155/:163), while the terminal payload is one-shot server-side — both finalisation paths clear `_inference_meta` on read (makermodslab/rollout.py:1089-1091, :1119-1123); the secondary claim (log mined without joining the pump thread, makermodslab/rollout.py:1130) also holds but is largely mitigated by the pump's per-line `flush()` (makermodslab/rollout.py:176). ⚠ UNRESOLVED @ HEAD ce42158 — `Inference.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/inference/ + studio/DeployPanel.tsx, construct not re-located.
**Priority:** P2 — lost error reporting only, no hardware effect (verified 2026-07-14 against the current working tree). **← Backend half FIXED on `main`.**

**Broken pathway**

1. The frontend starts an async `tick` every second without preventing overlap.
2. Tick A receives the one-shot terminal failure payload; the backend clears terminal metadata while returning it.
3. Tick A then waits for the log request.
4. Tick B can run meanwhile, receive ordinary idle status, overwrite page state, mark exit handled, and navigate home.
5. Tick A's later attempt to freeze the failure display is too late.

**Impact** — A real download, preflight, or runtime failure can disappear before the operator sees its error and hint. **Evidence:** `makermodslab/rollout.py:1093-1111`, `frontend/src/pages/Inference.tsx:123-170`, `:205-211`, `makermodslab/rollout.py:164-200`, `:1112-1151`. ⚠ UNRESOLVED @ HEAD ce42158 — `Inference.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/inference/ + studio/DeployPanel.tsx, construct not re-located.

### I9 · P2 — External bimanual checkpoints with right-arm cameras are mapped to the wrong feature keys

**First identified:** 2026-07-14

**Status:** Confirmed

> **Verdict on `main` (2026-07-23): STILL REAL, now duplicated.** `cameraMappings` still strips a unique
> `left_`/`right_` prefix to the bare name (`InferenceModal.tsx:143-149`), and the backend still attaches
> **every** camera to the BiSO **left** arm (`rollout.py:647`, rationale `:636-638`) — so `right_wrist` becomes
> `left_wrist`; the collision branch still yields `left_left_x`/`left_right_x`. The whole function is duplicated
> verbatim in the studio panel (`DeployPanel.tsx:141-158`), so any fix must land twice.

**Verdict:** CONFIRMED — the modal strips a unique `right_` prefix to the bare name (frontend/src/components/landing/InferenceModal.tsx:143-149) and the backend attaches every camera to the BiSO left arm (makermodslab/rollout.py:638-639), so the checkpoint's `right_wrist` becomes `left_wrist`; the `left_x`+`right_x` collision case yields `left_left_x`/`left_right_x` as described.
**Priority:** P2 — affects externally recorded bimanual checkpoints only; makermodslab cannot produce `right_*` camera checkpoints itself (verified 2026-07-14 against the current working tree)

**Broken pathway** — For an external bimanual checkpoint expecting `right_wrist`, the modal strips the unique `right_` prefix and sends `wrist`. The backend puts every camera on the BiSO left-arm config, so LeRobot exposes it as `left_wrist`. If a checkpoint contains both `left_x` and `right_x`, collision handling keeps both full names, but placing them on the left arm produces `left_left_x` and `left_right_x`.

**Impact** — Externally recorded bimanual policies with right-arm camera features cannot receive the image observations they were trained against. **Evidence:** `frontend/src/components/landing/InferenceModal.tsx:120-149`, `makermodslab/rollout.py:622-642`.

### I10 · P2 — A real mid-run motor overload can be mislabeled as a cleanup-only warning

**First identified:** 2026-07-14

**Status:** Confirmed

> **Verdict on `main` (2026-07-23): STILL REAL (acknowledged tradeoff).** `_classify_outcome` still downgrades
> any post-marker non-zero exit whose mined text matches the `CLEANUP_MARKERS` (`rollout.py:584-586`; markers
> `("overload", "torque_enable")` at `utils/errors.py:33`, matcher `:36-46`) with no teardown-vs-active
> evidence. `utils/errors.py:62-77` still owns this as rollout's log-tail-only fallback — a documented
> limitation.

**Verdict:** CONFIRMED (deliberate, documented limitation) — `_classify_outcome` downgrades any post-marker nonzero exit whose mined text matches `overload`/`torque_enable` (makermodslab/rollout.py:565-578, makermodslab/utils/errors.py:33-45) with no teardown-vs-active evidence; the in-process `classify_outcome` docstring explicitly owns this as rollout's log-tail-only fallback (makermodslab/utils/errors.py:72-77).
**Priority:** P2 — reporting only, mildly safety-adjacent: a real mid-run motor fault is understated as an amber warning, but the run has already ended either way (verified 2026-07-14 against the current working tree)

**Broken pathway** — Any nonzero exit after the rollout-start marker is classified `ran_with_warning` when the mined error includes `overload` or `torque_enable`. The classifier has no evidence that the failure occurred during teardown rather than while actively driving.

**Impact** — A real runtime motor failure can be downgraded from a failed inference to an amber cleanup warning. **Evidence:** `makermodslab/rollout.py:567-580`, `makermodslab/utils/errors.py:29-45`, `tests/test_rollout.py:875-890`.

### I11 · P3 — Active startup can display the previous inference run's log

**First identified:** 2026-07-14

**Status:** Confirmed

> **Verdict on `main` (2026-07-23): STILL REAL.** The log file is created only immediately before spawn
> (`rollout.py:811-814`), i.e. after download and preflight. `_resolve_inference_log_path` still falls back to
> the newest historical `*.log` whenever the active meta has no usable path (`:1066-1077`), and the dialog
> fetches the log on every 1 s tick (`InferenceSessionDialog.tsx:144-149`).

**Verdict:** CONFIRMED — the log file is created only at spawn, after download and preflight (makermodslab/rollout.py:795-798); `_resolve_inference_log_path` falls back to the newest historical `*.log` whenever the active meta lacks a path (makermodslab/rollout.py:1044-1055); the page fetches the log on every poll tick (frontend/src/pages/Inference.tsx:142-147). ⚠ UNRESOLVED @ HEAD ce42158 — `Inference.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/inference/ + studio/DeployPanel.tsx, construct not re-located.
**Priority:** P2 — stale prior-run log visible during startup; reporting-only, brief window (verified 2026-07-14 against the current working tree, coordinator-ratified)

**Broken pathway** — The active session has no `log_path` during model resolution and robot preflight because the log is created just before spawn. The log endpoint falls back to the newest historical log whenever active metadata lacks a usable path, and the page fetches logs on every startup tick.

**Impact** — The current startup can show unrelated errors and output from a previous rollout. **Evidence:** `makermodslab/rollout.py:793-800`, `:1042-1059`, `frontend/src/pages/Inference.tsx:140-147`. ⚠ UNRESOLVED @ HEAD ce42158 — `Inference.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/inference/ + studio/DeployPanel.tsx, construct not re-located.

### I12 · P3 — Hub checkpoint cache conflates imported-model and cloud-job listing rules

**First identified:** 2026-07-14

**Status:** Confirmed

> **Verdict on `main` (2026-07-23): STILL REAL.** Imported records fetch with `_list_imported_hub` and cloud
> records with the tree-only default (`jobs.py:1465-1472`), but `_list_cloud_cached` keys its 30 s TTL cache on
> `repo_id` alone with no scanner mode in the key (`jobs.py:1493-1499`).

**Verdict:** CONFIRMED — imported records fetch via `_list_imported_hub` (root-or-tree) while cloud records use the tree-only default (makermodslab/jobs.py:1460-1467), but `_list_cloud_cached` keys its 30s TTL cache on `repo_id` alone with no scanner mode (makermodslab/jobs.py:1490-1494), so the first lookup's shape is served to the other record type.
**Priority:** P2 — lowest-priority P2; transient (30s TTL) and requires the same repo registered as both an imported and a cloud record (verified 2026-07-14 against the current working tree)

**Broken pathway** — Cloud training jobs use a checkpoints-tree-only scanner. Imported Hub models use a root-or-tree scanner. Both results are cached solely by `repo_id`, without the scanner/listing mode in the key.

**Impact** — Checkpoint lists can temporarily disappear or take the wrong shape when the same repository is represented by both record types. **Evidence:** `makermodslab/jobs.py:774-800`, `:1465-1472`, `:1486-1500`.

## Design gaps and lower-confidence risks

### G1 · P2 — Required language task is not enforced

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL, now on both surfaces.** `canStart` has no task check on either
> surface (`InferenceModal.tsx:353-360`, `DeployPanel.tsx:491-499`); the task input renders for `requires_task`
> policies but empty is submittable (`InferenceModal.tsx:543-559`, `DeployPanel.tsx:754-769`); the backend
> defaults `task: str = ""` (`rollout.py:69`) and forwards verbatim (`:603`). Whether a blank task degrades or
> fails is policy-dependent — **cannot verify statically**.

**Verdict:** CONFIRMED (gap) / CANNOT-VERIFY-STATICALLY (user-facing failure — policy-dependent) — `canStart` has no task check (frontend/src/components/landing/InferenceModal.tsx:343-350), the task input renders for `requires_task` policies but empty is submittable (frontend/src/components/landing/InferenceModal.tsx:534-550), and the backend defaults `task=""` (makermodslab/rollout.py:71).
**Priority:** P2 — preflight gap on a common flow for language-conditioned policies; failure mode is policy-dependent, not a safety issue (verified 2026-07-14 against the current working tree)

Language-conditioned policies expose `requires_task`, but Start remains enabled for a blank task and the backend defaults to `task=""`. Evidence: `frontend/src/components/landing/InferenceModal.tsx:343-350`, `:534-549`, `makermodslab/rollout.py:65-71`.

### G2 · P2 — Backend duration has no positive lower bound

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL — severity UNDERSTATED (raise).** `duration_s: int = 60` is
> unvalidated (`rollout.py:71`) and passed verbatim as `--duration=` (`:604`); the installed lerobot prints
> "infinite" for a non-positive duration (`lerobot/scripts/lerobot_rollout.py:226`). **Correction:** the audit
> priced this as "reachable only by a direct API caller" because the UI sets `min={1}`. That is wrong on `main`
> — `NumberInput` never clamps to `min`, it only rejects non-finite input
> (`ui/number-input.tsx:39-47`), and both call sites pass the value straight through
> (`InferenceModal.tsx:568-570`, `DeployPanel.tsx:778-780`). A user typing `0` into "Max duration (seconds)"
> starts an **unbounded autonomous run from the normal UI**.

**Verdict:** CONFIRMED — `duration_s: int = 60` is unvalidated (makermodslab/rollout.py:73) and passed verbatim to the subprocess (makermodslab/rollout.py:595); verified against installed lerobot 0.6.0: `duration: float = 0.0  # 0 = infinite (24/7 mode)` and the script prints "infinite" for nonpositive duration, so a direct API caller gets an unbounded autonomous run; the UI's `min={1}` is cosmetic (frontend/src/components/landing/InferenceModal.tsx:557).
**Priority:** P2 → **raise: reachable from the normal UI** (NumberInput does not clamp). Safety-adjacent (unbounded run) (verified 2026-07-14 against the current working tree)

The UI uses `min={1}`, but `InferenceRequest.duration_s` accepts any integer and passes it directly to rollout. The pinned rollout runtime treats nonpositive duration as unbounded. Evidence: `makermodslab/rollout.py:65-71`, `:583-605`, `frontend/src/components/landing/InferenceModal.tsx:551-559`.

### G3 · P2 — Multiple roles may bind the same physical camera

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL, now on both surfaces.** `allCamerasBound` enforces only that
> every role resolves to a live camera, never device uniqueness (`InferenceModal.tsx:347-349`,
> `DeployPanel.tsx:482-484`). Whether the second open fails or duplicates frames is device-dependent —
> **cannot verify statically**.

**Verdict:** CONFIRMED (gap) — `allCamerasBound` enforces only that every role resolves to a live camera, not device uniqueness (frontend/src/components/landing/InferenceModal.tsx:339-341); whether the second open fails or duplicates frames is backend/device-dependent.
**Priority:** P2 — misconfiguration slips past preflight (verified 2026-07-14 against the current working tree)

The modal verifies that every role resolves to a live camera, but does not enforce unique devices. Two expected observation keys can therefore be configured with the same OpenCV index. Evidence: `frontend/src/components/landing/InferenceModal.tsx:339-350`.

### G4 · P2 — Server arm-count guard loses the frontend's action-dimension fallback

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL, now on both surfaces.** Both surfaces compute arm count from
> `state_dim ?? action_dim` (`InferenceModal.tsx:330`, `DeployPanel.tsx:469-470`) but forward only `state_dim`
> (`InferenceModal.tsx:425`, `DeployPanel.tsx:554`), and the server's `_arm_count_mismatch` defers to the
> subprocess when `checkpoint_state_dim` is `None` (`rollout.py:378-379`). An action-only checkpoint keeps the
> UI guard and loses the authoritative backend one.

**Verdict:** CONFIRMED — the modal computes arm count from `state_dim ?? action_dim` (frontend/src/components/landing/InferenceModal.tsx:322) but forwards only `state_dim` (frontend/src/components/landing/InferenceModal.tsx:417), and the server's `_arm_count_mismatch` defers to the subprocess when `checkpoint_state_dim` is None (makermodslab/rollout.py:371-372).
**Priority:** P2 — lowest-priority P2; action-only checkpoints remain UI-guarded, and the subprocess's own shape check still catches the mismatch, just less legibly (verified 2026-07-14 against the current working tree)

The modal determines checkpoint arm count from `state_dim ?? action_dim`, but sends only `state_dim` as `checkpoint_state_dim`. Evidence: `frontend/src/components/landing/InferenceModal.tsx:315-337`, `:399-410`, `makermodslab/rollout.py:82-86`.

---

## Appended findings (N-series)

One consolidated, append-only list: every finding since the original audit gets the next `N` here, newest last. IDs are permanent and per-module — "Recording N1" is distinct from "Dataset N1" / "Inference N1". N1–N10 date from the 2026-07-23 re-verification; N11+ were appended in later sessions (dated inline where recorded). *(Consolidated 2026-08-02 from the former "New findings" sections and later in-place appends; no prose, verdicts or severities were changed.)*

### N1 · P0 — A Stop that overruns the 5 s budget SIGKILLs the rollout mid-teardown, leaving follower torque enabled; MakerMods Lab has no fallback release on any inference exit path

**First identified:** 2026-07-23

> This is part of **CURRENT P0-3**.

**Status:** Confirmed structurally; the timing threshold needs one hardware measurement.

`handle_stop_inference` gives the child a **hard 5-second budget**, then SIGKILLs it (`rollout.py:1036-1042` — `terminate()` / `wait(timeout=5)` / on timeout `kill()`/`wait()`). What that 5 s must cover, in the pinned lerobot (`lerobot/rollout/strategies/core.py:119-146`):

1. `self._engine.stop()` — for the RTC engine joins a background thread.
2. `_return_to_initial_position(hw)` — `duration_s: float = 3.0, fps: int = 50` (`core.py:141`), i.e. **150 iterations** of `send_action` + `precise_sleep(1/50)`; `precise_sleep` is a fixed wait (`robot_utils.py:19-32`), so the **3.0 s of sleeps is a floor** and the 150 serial round-trips are additive.
3. Only *after* that does `robot.disconnect()` run (`core.py:133-134`) — which is what **disables torque** and releases cameras.

So the teardown that ends with torque release cannot finish in under 3.0 s, and on a real rig (150 serial writes + bimanual second bus + camera release) plausibly exceeds 5 s. When it does, MakerMods Lab SIGKILLs the process **before** `robot.disconnect()`. **MakerMods Lab has no fallback torque release for inference** — `rollout.py`'s import block (`:42-53`) does not import `.torque`; the only consumer of `force_disable_bus_torque` (`torque.py:30`) in the package is `auto_calibrate.py:39,179`. No fallback on the stop path, abandon path, `_fail_startup`, or status-finalisation. This undercuts pinned contracts (`rollout.py:605-611` sets `--return_to_initial_position=true` "so the contract is ours"; `InferenceSessionDialog.tsx:228-230` "eases back … and releases torque"). Aggravating: `DeployPanel.tsx:584-585` toasts "winding down" on any HTTP 200, including the SIGKILL case (I2).

**Impact.** After a Stop, the follower can be left energized, holding a mid-motion pose indefinitely, with the backend and both UIs reporting idle. The only recovery is a power cycle. *Not verifiable statically:* whether teardown actually crosses 5 s on this hardware; the structural facts do not depend on that.

### N2 · P0/P1 — The abandoned-after-spawn path can leak a live, completely untracked rollout subprocess

**First identified:** 2026-07-23

> This is part of **CURRENT P0-3** (replaces the I1 untracked-child sub-case that `main` fixed).

**Status:** Confirmed.

When a stop races the spawn, the worker takes the abandon branch (`rollout.py:878-888`): `terminate()` + `wait(timeout=5)` with **no `kill()` fallback** (a `TimeoutExpired` is swallowed by `contextlib.suppress`), the stdout pump is never started (`:890-897` runs only after commit), and `proc` is a local variable never assigned to `_inference_proc` (`:853-854`) — **no handle survives anywhere.** The SIGTERM is only observed through lerobot's `shutdown_event`, which the *run loop* polls; during `strategy.setup(ctx)` (policy load + `robot.connect()`) nothing checks it (`lerobot/scripts/lerobot_rollout.py:232-238`), so a multi-GB policy load easily outlasts the 5 s window.

**Broken pathway.** Stop pressed as the subprocess spawns → worker terminates and waits 5 s → child still loading the policy → worker gives up silently and returns → child finishes loading, **connects and energizes the follower**, enters the loop, and only then observes the shutdown event. During that window the backend reports idle, the mutex is open, and no handle exists to stop it. Distinct from I2 (terminate-fails *with* a retained handle) and I1.

### N3 · P1 — `DELETE /jobs/{job_id}` bypasses the inference in-use guard and `rmtree`s a checkpoint a live rollout is reading

**First identified:** 2026-07-23

**Status:** Confirmed.

`server.py:1664-1677` deletes a job with **no** `_model_in_use` / `inference_in_use_path()` check. `JobRegistry.delete` removes the directory outright — `shutil.rmtree(_job_dir(...))` (`jobs.py:1555-1556`). Its only guard is `record.state == "running"` (`:1550-1551`), about the **training** job, not an inference run reading its checkpoints. The in-use guard exists only on the models route (`delete_local_model` → `_model_in_use` → `rollout.inference_in_use_path()`, `models.py:975-987`, `:1019-1021`, `:1057-1059`). Two delete routes, one guard. Reachable from the UI (`deleteJob` in `jobsApi.ts:247`, called from `JobsDataContext.tsx` and `TrainingJobDialog.tsx`).

**Impact.** A local training run's output dir can be deleted while a rollout is running that checkpoint — the harm I7 describes but with the guard bypassed entirely, for the *whole* run. *(Same family as Training MT7.)*

### N4 · P1 — During a status-endpoint outage the dialog shows no Stop control and the exit guard is disarmed

**First identified:** 2026-07-23

**Status:** Confirmed.

`InferenceSessionDialog.tsx:338-342`: while `status` is `null` the entire body — including the Stop button (`:437-447`) — is replaced by a "Connecting to inference…" spinner. The tick's catch (`:210-218`) only toasts "Lost connection to backend" and **never sets `status`**, so a status-endpoint outage keeps `status` null for its whole duration. In that state: the exit guard's `active` is `status?.inference_active === true` (`:107`) → **false** → no leave protection fires (`useSessionExitGuard.ts:94-95`, `:134-135`); but `live` is `status == null` → **true** (`:252`), so ESC, outside-click and the X are all blocked (`:315-329`) — the operator is pinned on a dialog with no stop control while the run continues. Contradicts the file's own contract (`:100-105`). Partial mitigation: `DeployPanel`'s independent Stop (`:874-882`) still works, but only if the studio panel is open and its own poll succeeds.

### N5 · P2 — The subprocess is reaped only by a `/inference-status` poll; with nobody polling, the slot stays latched active forever

**First identified:** 2026-07-23

**Status:** Confirmed.

`handle_inference_status` is the **only** place `proc.poll()` is consulted and the only writer of the terminal `_last_result` (`rollout.py:1114-1154`). There is no background reaper (the startup handler runs only `warn_if_cuda_mismatch()`, `server.py:2543-2546`; shutdown touches only the broadcast thread). If the last polling surface goes away — tab closed and the `pagehide` beacon fails (`useSessionExitGuard.ts:114-119`), studio panel closed (`DeployPanel.tsx:442-443`) — a run that ends on its own `--duration` is never reaped: `inference_active` stays `True` and `_inference_proc` stays set. **Impact.** The feature mutex stays claimed (inference/teleop/recording all 409 indefinitely), and `inference_in_use_path()` keeps blocking model deletion against a process that no longer exists. Recoverable by Stop or restart, but the error messages give no clue the run already ended.

### N6 · P2 — Bimanual right-arm fields are unvalidated, so a bad request opens and primes the LEFT arm before failing

**First identified:** 2026-07-23

**Status:** Confirmed.

`handle_start_inference`'s synchronous pre-spawn checks are only the mutex (`rollout.py:917-935`), the arm-count guard (`:962-965`) and the policy-ref shape check (`:969-975`). It never validates `follower_port`, `follower_config`, or — when `mode == "bimanual"` — `right_follower_port` / `right_follower_config`. `mode` is taken verbatim from the client (`:75`) and never cross-checked against the named robot record. `_prepare_robot` works left-then-right (`:672-696`), so a blank or wrong right-arm port surfaces only **after** the left follower's bus has been opened and its `Torque_Limit`/`Goal_Velocity` rewritten (`:688-695`). The comment at `:958-961` (guards exist so a mismatch is rejected "BEFORE spawning the worker") holds only for the arm-count case.

### N7 · P2 — Camera dict keys are interpolated into a draccus dict literal with no escaping

**First identified:** 2026-07-23

**Status:** Confirmed (robustness/legibility, not a shell-injection).

`_format_cameras_arg` (`rollout.py:500-520`) builds lerobot's CLI dict by f-string (`:518-520`). `name` is the checkpoint's `observation.images.*` feature key, forwarded verbatim by both frontends as `m.requestKey` (`InferenceModal.tsx:387-400`, `DeployPanel.tsx:528-541`). A feature name containing `,`, `:`, `{` or `}` — possible in an externally produced checkpoint, which the bimanual-collision branch already treats as a real case — corrupts the argument and fails deep inside the subprocess with an unrelated parse error rather than a legible pre-spawn rejection. No shell is involved (`subprocess.Popen` with an argv list, `:829-835`), so this is not an injection vector.

### N8 · P2 — Camera bindings are frozen cv2 indices; a replug that reuses an index silently rebinds a role to a different physical camera

**First identified:** 2026-07-23

**Status:** Confirmed statically; the physical outcome needs a hardware check.

Both surfaces key bindings on the **cv2 index string** — `cameraKey = (cam) => String(cam.index)` (`InferenceModal.tsx:50`, `DeployPanel.tsx:80`) — and send `camera_index: live.index` (`InferenceModal.tsx:395`, `DeployPanel.tsx:536`), forwarded verbatim (`rollout.py:513-515`) with no re-resolution. The "drop a binding whose camera vanished" effect tests `liveCameraByKey(key)` i.e. "does index N still exist?" (`InferenceModal.tsx:301-314`, `DeployPanel.tsx:426-439`). Per this repo's documented macOS gotcha, removing one device **renumbers the rest**, so index `1` usually still exists pointing at a *different* physical camera — the stale-binding effect does not fire and the role is silently rebound. The stable `unique_id` is available (`server.py:2180`) and discarded (`useAvailableCameras.ts:4-9` carries only index/name/deviceId/available). **Impact.** A policy can be handed the wrong camera on each observation key — a silently wrong run rather than an error. Inference-side analogue of the recording camera-identity re-anchoring (`c4e5198`); no equivalent exists in `rollout.py`.

### N9 · P3 — The final `proc.wait()` on the stop path has no timeout

**First identified:** 2026-07-23

`rollout.py:1042`: after `proc.kill()`, `proc.wait()` is called with no timeout, so an unkillable child (blocked in an uninterruptible serial ioctl) hangs the `/stop-inference` request thread indefinitely. The UI has no timeout of its own — `handleStop` awaits the request (`InferenceSessionDialog.tsx:237`) and leaves the button pinned at "Stopping…" (`:445`).

### N10 · P3 — Two independent Stop surfaces can double-signal the child, turning a clean user stop into a reported failure

**First identified:** 2026-07-23

The terminate/wait sequence runs **outside** `_state_lock` while `inference_active` is still `True` (`rollout.py:1035-1044` — lock released at `:1034`, re-taken at `:1046`), so a second `/stop-inference` in that window passes the `if not inference_active` check (`:1007`) and issues a second `proc.terminate()`. Two Stop surfaces are on screen simultaneously — the dialog (`InferenceSessionDialog.tsx:231-247`) and the studio panel underneath (`DeployPanel.tsx:581-595`) — plus the hung-run watchdog (`:192-209`), each with its own in-flight guard. lerobot escalates on the second signal: `if self._counter > 1: sys.exit(1)` (`lerobot/utils/process.py:57-71`). **Steelmanned:** `sys.exit` raises `SystemExit` in the main thread and `finally: strategy.teardown(ctx)` still runs — **not** a torque-loss path. The consequence is reporting: the process exits `rc=1`, and `_classify_outcome` (`:573-586`) reports a clean user stop as `failed` (or `ran_with_warning`).

### N11 · P2 — `rollout.py` hardcodes the lerobot cache root twice, so inference logs ignore `HF_LEROBOT_HOME`

**First identified:** 2026-07-26

`rollout.py:811` (spawn) and `rollout.py:1072` (post-run log lookup) both build the log directory as `Path.home() / ".cache" / "huggingface" / "lerobot" / "inference_logs"` — an inline literal, resolved from `Path.home()` and **ignoring the `HF_LEROBOT_HOME` environment variable entirely**. CLAUDE.md's convention is explicit: *"Import shared constants from [utils/config.py], do not hardcode paths in feature modules."*

**Steelmanned — this is not simply a missed import.** There is no inference-logs constant in `utils/config.py` to import: that module itself spells the root as an `os.path.expanduser("~/.cache/huggingface/lerobot/…")` literal at every one of its ~10 definitions (`:29-80`), also env-var-blind. The codebase resolves this one root **four different ways, split env-aware vs env-blind**:

| | Site |
|---|---|
| **Honors `HF_LEROBOT_HOME`** | `lerobot.utils.constants.HF_LEROBOT_HOME` (`constants.py:69`, imported in-body at `record.py:863`), `merge.py:52-55`, `datasets.py:330`, `runners/hf_cloud.py:655` |
| **Ignores it** | `utils/config.py`'s ~10 literals (`:29-80`), `rollout.py:811,1072` |

*(Correction, 2026-07-26: an earlier revision of this entry claimed `merge.py` was the **only** env-aware site. That was wrong — four sites honor the variable. The defect is the env-blind half, not a lone correct outlier.)* Note `merge.py`'s docstring calls `merge_logs/` a *"sibling of lerobot's `inference_logs/` (see rollout.py)"* — i.e. it was written as a deliberate copy of a path it then resolves differently from its stated sibling.

**Consequence:** on a station that sets `HF_LEROBOT_HOME` (the supported way to relocate the cache) the root **splits in half**: datasets and merge logs follow the variable, while **calibrations, robot records, ports, `saved_*.json` and inference logs stay at `~/.cache/...` regardless**. A run's log silently lands outside the configured root, and `:1072` looks for it in the same wrong place, so the "just-finished run's log" feature keeps working only by symmetry of the bug. It also makes the directory untestable by fixture: patching `makermodslab.utils.config` cannot redirect it.

*Scope note: `rollout.py` is the smaller half of this. The systemic half is `utils/config.py` being env-blind — which is worse, because that module is exactly the one CLAUDE.md designates as the single source for shared paths, and it owns calibrations, robot records and ports. That half is not filed here (wrong module) and is pending a Configuration Setup entry.*

**Fix:** adopt `lerobot.utils.constants.HF_LEROBOT_HOME` as the single root (it is the canonical definition and already resolves the variable at `constants.py:69`), derive `inference_logs`/`merge_logs`/calibration/ports/state-file paths from it in `utils/config.py`, and have `rollout.py` import from there — retiring the three ad-hoc spellings. Related: the same env-var blindness in `utils/config.py`'s own constants is the reason `tests/conftest.py::tmp_lerobot_home` cannot redirect every path — see the coverage-gap note.

*Discovered while building `tests/repro/` for the P0 set; not caused by that work.*

### N12 · P1 — The policy-extra preflight is wired into training only, so launching inference on a policy that needs an optional extra dies with a buried ImportError

**First identified:** 2026-07-28

MakerMods Lab already has the whole mechanism for this, and it is only connected at one end.

`utils/system.py:245` defines `POLICY_EXTRAS`, mapping each policy type that imports an optional dependency at construction time to `(probe_module, install_target)`: `smolvla` → `transformers` / `lerobot[smolvla]`, `pi0` and `pi0_fast` → `transformers` / `lerobot[pi]`, `diffusion` → `diffusers` / `lerobot[diffusion]`. Its own comment states the intent: *"Some LeRobot policies import an optional extra at construction time; **training (or inference)** otherwise dies with a buried ImportError once the subprocess is already running."* The parenthetical is the bug — inference was anticipated and never wired.

The backend half is complete: `GET /system/policy-extra/{policy_type}` (`server.py:1768`) and `POST /system/policy-extra/{policy_type}/install` (`:1775`), with a per-target `InstallManager` so `pi0`/`pi0_fast` share one `lerobot[pi]` install.

**The only frontend consumer is `TrainingConfigurator.tsx`** (`:403` probes the endpoint, `:616-622` renders the install prompt). Grep finds no other caller. And `rollout.py` performs no dependency check of any kind — no probe, no `ImportError` handling, nothing that mentions an extra. So the inference launch path has neither a preflight nor a graceful failure.

**Failure:** starting inference with a `smolvla` / `pi0` / `pi0_fast` / `diffusion` policy on a machine where that extra is not installed spawns `lerobot-rollout`, which dies on the import once the subprocess is already running. The user sees a generic startup failure; the actual cause is a `ModuleNotFoundError` buried in the subprocess log, with no prompt to install and no indication which package is missing — even though MakerMods Lab knows the exact install target.

**The path that makes this routine, not an edge case:** a policy trained on HF Jobs installs its extra *in-container* (`runners/hf_cloud.py:137` builds the same install command for the remote image), so the local machine never acquires it. "Train in the cloud, deploy locally" therefore hits this every time for the four affected policy types — and the local machine may never have run a local training of that type to trigger the training-side prompt.

**Fix:** reuse what exists — probe `handle_get_policy_extra(policy_type)` on the inference launch path and surface the same install prompt the training configurator already renders, before spawning the subprocess. Note the resolved policy type must come from the checkpoint being deployed, not from a form field. Failing that, at minimum `rollout.py` should catch the import failure and map it to the known install target rather than reporting a generic startup error.

*Related:* Model Training's "host-only-dep gate too broad" concerns the same install machinery from the training side.

---

### N13 · P2 — Deploy uses the checkpoint's image-feature dims as the camera CAPTURE resolution, so base/generic checkpoints (256×256) crash at robot.connect

**First identified:** 2026-07-30

**Status:** Confirmed live (smolvla_base rollout).

`DeployPanel.tsx` (~line 533, "Resolution comes from the checkpoint feature's dims") builds the rollout camera dict with `width/height = policyConfig.image_features[feature]` — i.e. the policy's input TENSOR shape. That holds only for checkpoints fine-tuned on this rig's own recordings (dims = recording resolution, 640×480). `lerobot/smolvla_base` declares generic 256×256 placeholder slots (`observation.images.camera1/2/3`); the rig cameras' nearest real mode is 352×288, so lerobot's strict `_validate_width_and_height` raises `RuntimeError: failed to set capture_width=256 (actual_width=352)` inside `robot.connect()` and the run dies. Conceptually capture resolution and model input resolution are different things — SmolVLA resizes with padding internally (`resize_imgs_with_padding`), so the camera should capture a mode it actually supports.

**Fix direction:** capture at the robot record's configured camera resolution (or default 640×480) regardless of checkpoint dims; alternatively validate checkpoint dims against the camera's supported modes at deploy time and surface a friendly error instead of the subprocess traceback. Zero-shot base deploys will still act near-randomly — arguably Deploy should warn when the selected ref is a known base model (`camera1/2/3` feature names are a cheap tell).

## Validation (original audit)

```text
.venv/bin/python -m pytest tests/test_rollout.py tests/test_log_endpoints.py -q
67 passed
```

The passing suite validates existing isolated helpers and nominal lifecycle branches. It does not invalidate the concurrency and cross-request defects above because those interleavings are not covered. No cameras, serial ports, robots, user cache files, Hub tokens, or external network services were accessed.

## Coverage gaps (original + re-verification additions)

- Stop during follower preflight, not only during model download.
- Stop termination failure with an assertion that the child is still tracked until confirmed dead.
- Stop inference A, immediately start B, then complete or fail A's stale worker.
- Concurrent inference/teleoperation/recording starts under a shared barrier.
- Calibration (and auto-calibration) start against the same follower port as a live inference.
- Explicit Stop request failure followed by back navigation or unmount.
- Overlapping status ticks around the (now idempotent) terminal payload.
- Downloaded Hub model launch with Hub access unavailable.
- Local model deletion between policy resolution and subprocess commit.
- Bimanual `right_*` and colliding left/right camera feature mappings.
- Server shutdown while an inference child is active.
- Mid-run overload versus teardown-only overload outcome classification.
- **Stop where the child needs >5 s to tear down: assert torque is released (or a fallback fires) rather than SIGKILL-and-forget (N1).**
- **Stop racing the spawn: assert no untracked process survives the abandon branch (N2).**
- **`DELETE /jobs/{id}` while an inference is running that job's checkpoint: assert 409 (N3).**
- **Status endpoint 5xx for N ticks while a run is live: assert a Stop control is still reachable and the guard armed (N4).**
- **Run ends by `--duration` with no status poller: assert the slot is released (N5).**
- **`duration_s = 0` submitted from the UI's number field: assert rejection (G2).**
- **Camera replug that reuses a cv2 index between picker-open and Start: assert the binding is invalidated (N8).**

## Context: contradictions to the original brief (resolved during re-verification)

1. **`frontend/src/pages/Inference.tsx` does not exist on `main`.** Verdicts for I3, I8, I11 are re-anchored to `InferenceSessionDialog.tsx`.
2. **One original finding is FIXED, not merely re-ranked** — I8's backend half (the consume-once terminal payload) is genuinely closed by the `_last_result` idempotence rework.
3. **G2's stated severity is wrong on `main`** — `NumberInput` does not clamp, so an unbounded run is reachable by typing `0` in the normal UI.
4. **"arm-count validation before hardware is touched" holds; the general request-shape claim does not** — the bimanual right-arm fields are unvalidated (N6).
