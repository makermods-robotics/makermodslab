# Inference Bug List

**Audit date:** July 14, 2026  
**Audited commit:** `518ca56`  
**Scope:** Inference launch, model/checkpoint resolution, robot and camera preparation, subprocess lifecycle, stop/leave safety, status/log reporting, and model deletion interactions.

This audit uses the implementation-backed pathways documented in `internal_docs/complete functionalities/Inference Interface Pathways.md` as its baseline. Findings are limited to high-confidence pathway breaks visible in the current source. No fixes were made.

## Status and severity key

- **Confirmed:** The implementation itself establishes a reproducible broken pathway or safety failure; hardware execution is not required to prove the defect.
- **Design gap:** A missing invariant or validation leaves behavior unsafe or ambiguous, but the exact user-visible failure depends on a policy, device, direct API caller, or external runtime behavior.
- **P0:** Immediate, unavoidable catastrophic impact. None found.
- **P1:** High-impact hardware-safety, lifecycle, or mutual-exclusion failure.
- **P2:** Major feature/pathway failure without an unavoidable catastrophic outcome.
- **P3:** Lower-impact correctness, reporting, or compatibility defect.

## Confirmed findings

### P1 — A stopped startup worker can keep touching hardware and corrupt a newer session

**Status:** Confirmed

**Verdict:** CONFIRMED-WITH-CORRECTIONS — session-token-less `_fail_startup` clears whichever session is currently active (makermodslab/rollout.py:715-741), cancel is checked only after `_prepare_robot` returns (makermodslab/rollout.py:787), stop-with-no-proc clears state without joining the worker (makermodslab/rollout.py:998-1011); correction: the preflight never enables torque — `_open_follower` releases with `disconnect(disable_torque=False)` (makermodslab/rollout.py:420-435), so the "motor-power writes" are RAM `Torque_Limit`/`Goal_Velocity` priming, not motion, and the clobber fires only on the stale worker's raise path.
**Priority:** P1 — state/mutex-integrity break, P0-adjacent in the untracked-child subcase (a stale worker's raise can null the new session's process handle, leaving a live policy subprocess with no stop handle); no torque is enabled by the stale worker — RAM register writes only (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. Inference A starts and enters model resolution or follower preflight.
2. The user stops A before `_inference_proc` is installed.
3. Stop sets A's cancel event, immediately clears the global inference state, and reports success without waiting for A's startup thread.
4. Inference B can now claim the same global slot.
5. A can continue an already-entered `_prepare_robot` call, including serial-bus access and motor-power writes. Cancellation is checked only after the complete preflight returns.
6. If stale A then fails, `_fail_startup` sees only that some inference is active and clears B's global process/state. If B already spawned, its child can continue without a tracked process handle.

Download progress, phase changes, and stdout markers are also global rather than session-keyed, so stale A callbacks can overwrite B's progress or phase.

**Impact**

- Hardware can still be touched after Stop reports that inference stopped.
- A fresh session can be marked failed by the previous session.
- A live policy subprocess can become untracked, reopening the mutex while the arm may still be driven.

**Evidence**

- Global, non-session-keyed state: `makermodslab/rollout.py:95-109`
- Phase and stdout callbacks mutate shared state without checking their process/session: `makermodslab/rollout.py:153-200`
- Download progress mutates whichever metadata is currently present: `makermodslab/rollout.py:271-285`
- Startup failure clears whichever session is currently active: `makermodslab/rollout.py:717-743`
- Cancellation occurs only after model resolution and after the complete robot preflight: `makermodslab/rollout.py:765-790`
- Robot preflight opens buses and writes motor-power state: `makermodslab/rollout.py:645-714`
- Stop with no process clears state immediately instead of waiting for startup: `makermodslab/rollout.py:984-1015`

### P1 — Stop reports success and loses the process handle even when termination fails

**Status:** Confirmed

**Verdict:** CONFIRMED — `terminate()`/`wait()`/`kill()` exceptions are caught and only logged, then state is cleared unconditionally and `success: true` returned without verifying the child exited (makermodslab/rollout.py:1013-1030).
**Priority:** P1 — stop-path weakening: the only process handle is discarded without confirming the child is dead, so a still-running rollout becomes untrackable and the feature mutex reopens (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. A rollout subprocess is active.
2. `terminate()`, timed `wait()`, `kill()`, or final `wait()` raises.
3. The exception is logged but not returned.
4. The backend clears `_inference_proc`, marks inference inactive, and returns `success: true` without verifying the child exited.

**Impact**

An autonomous rollout may keep driving while the UI and backend report idle. The only process handle is discarded, and another robot-driving feature can be started.

**Evidence**

- Termination exceptions are swallowed and state is unconditionally cleared: `makermodslab/rollout.py:1017-1034`

### P1 — A failed explicit Stop permanently disarms the page-leave safety guard

**Status:** Confirmed

**Verdict:** CONFIRMED — `markHandled()` runs before the awaited stop request with no re-arm in the catch path (frontend/src/pages/Inference.tsx:217-227); the latch re-arms only on an `active` false→true transition, which never occurs while the run stays active (frontend/src/hooks/useSessionExitGuard.ts:87-89); the hung-run watchdog's `stopRequestedRef` is likewise never reset on failure (frontend/src/pages/Inference.tsx:185).
**Priority:** P1 — the automatic leave-stop (back/unmount/reload/page-hide) is disarmed after one failed Stop while the policy keeps running; the manual Stop button still works on retry, so the gap is the automatic leave-stop, not all stopping (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. The user confirms Stop.
2. The page calls `markHandled()` before sending the stop request.
3. The request fails while inference remains active.
4. The guard's one-way latch remains handled because `active` never changes from true to false and back to true.
5. Back navigation, component unmount, reload, and page hide skip their stop attempt.

The hung-run watchdog has the same one-attempt behavior: it sets `stopRequestedRef` before stopping and never resets it if the request fails.

**Impact**

After a transient backend or network failure, the operator can leave the page without another stop attempt while the policy continues running.

**Evidence**

- Stop is marked handled before the request, with no re-arm in the catch path: `frontend/src/pages/Inference.tsx:213-227`
- Hung-run stop failure is ignored and its one-shot latch is never reset: `frontend/src/pages/Inference.tsx:114-122`, `frontend/src/pages/Inference.tsx:173-194`
- `markHandled` is a one-way latch re-armed only when `active` changes: `frontend/src/hooks/useSessionExitGuard.ts:65-89`
- Back, page-hide, and unmount all skip work when that latch is set: `frontend/src/hooks/useSessionExitGuard.ts:91-166`

### P1 — Robot-driving feature exclusion is not atomic and calibration is omitted

**Status:** Confirmed

**Verdict:** CONFIRMED-WITH-CORRECTIONS — each feature guards with its own separate lock (makermodslab/rollout.py:111, makermodslab/teleoperate.py:102, makermodslab/record.py:220) and checks the other flags under only its own lock (makermodslab/rollout.py:898-916, makermodslab/teleoperate.py:570-584, makermodslab/record.py:461-474); calibration checks only `calibration_active` (makermodslab/calibrate.py:237) and no feature checks calibration; correction: OS-level serial exclusivity makes the second `bus.connect()` fail, so true concurrent driving of one port does not occur — the realistic outcome is unguarded contention and a confusing failed start.
**Priority:** P1 — driven by the calibration-vs-inference same-port case (calibration moves the arm and can stamp EEPROM while inference is admitted on the same follower port); the parked 2026-07-10 camera-lock / 409-gate branch (not in this tree) may cover part of this (verified 2026-07-14 against the current working tree)

**Broken pathway**

Inference, teleoperation, and recording each protect their own active flag with a different lock. Two concurrent starts can therefore interleave as follows:

1. Inference holds its lock and reads teleoperation as inactive.
2. Teleoperation holds its lock and reads inference as inactive.
3. Each sets its own active flag.

There is no single lock or coordinator covering the check-and-claim operation across features.

Calibration is not checked at all by inference start, and calibration start checks only for another calibration. A follower calibration and inference can therefore both be accepted for the same port even without a request race.

**Impact**

Two workers can contend for the same follower serial port or attempt incompatible hardware operations concurrently. The active flags can also claim that multiple mutually exclusive features are running.

**Evidence**

- Inference checks other feature flags while holding only its own lock: `makermodslab/rollout.py:899-930`
- Teleoperation has a separate lock and check/claim section: `makermodslab/teleoperate.py:101-102`, `makermodslab/teleoperate.py:572-596`
- Recording has a separate lock and check/claim section: `makermodslab/record.py:213-215`, `makermodslab/record.py:463-476`
- Calibration start checks only `calibration_active`: `makermodslab/calibrate.py:232-275`
- Calibration route delegates directly without a cross-feature guard: `makermodslab/server.py:1803-1813`

### P1 — FastAPI shutdown does not stop an active inference subprocess

**Status:** Confirmed

**Verdict:** CONFIRMED — the shutdown handler stops only the broadcast thread, never the inference lifecycle (makermodslab/server.py:2526-2535); the child is spawned without `start_new_session` (makermodslab/rollout.py:813-819), so it shares the process group and a terminal Ctrl-C does reach it — the orphan arises on `--reload` worker restart, a SIGTERM to the uvicorn pid, or programmatic shutdown.
**Priority:** P1 — orphaned autonomous policy with no backend or UI stop path, bounded only by `--duration` and `return_to_initial_position`; the trigger (`--reload`/pid-kill/programmatic shutdown) is common in dev (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. MakerMods Lab has spawned `lerobot.scripts.lerobot_rollout` as a child process.
2. FastAPI begins normal shutdown or reload.
3. The shutdown handler stops only the broadcast manager and never calls the inference stop lifecycle or signals the child.

**Impact**

The server can disappear while the policy subprocess remains alive, leaving no backend or UI path to track and stop it.

**Evidence**

- Rollout is spawned as a separate subprocess: `makermodslab/rollout.py:793-883`
- Shutdown performs no inference cleanup: `makermodslab/server.py:2540-2549`

### P2 — A model downloaded for offline use is still launched through the Hub

**Status:** Confirmed

**Verdict:** CONFIRMED — `importSourceForModel` returns `hf_repo_id ?? path ?? id`, preferring the Hub id over the local checkpoint path (frontend/src/lib/inferenceLaunch.ts:48); a non-directory source is registered Hub-backed (makermodslab/jobs.py:1305-1313) whose checkpoint listing calls `api.list_repo_files` (makermodslab/jobs.py:779-788); only bites when `findJobForModel` finds no covering registry record (frontend/src/lib/inferenceLaunch.ts:21-38).
**Priority:** P2 — the offline promise is broken but the workaround is being online, and a downloaded model already covered by a registry job launches locally (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. The user adds a Hub model and selects “Download to this machine now.”
2. The model listing correctly carries both a local checkpoint `path` and `hf_repo_id`.
3. When no existing job record matches, launch chooses `hf_repo_id` before `path`.
4. Imported-job registration treats the source as a Hub repository, calls Hub listing, and stores a Hub-backed pseudo-job.
5. Checkpoint selection and inference resolution remain network-dependent instead of using the downloaded directory.

**Impact**

The explicit promise that the downloaded model works offline is false. Inference launch can fail offline even though the full local checkpoint is present.

**Evidence**

- UI promises that downloading enables offline inference: `frontend/src/components/landing/AddModelFromHubDialog.tsx:98-110`
- Launch fallback prefers `hf_repo_id` over local `path`: `frontend/src/lib/inferenceLaunch.ts:40-48`
- Launch registers that selected source before opening the modal: `frontend/src/hooks/useInferenceLaunch.tsx:44-60`
- Downloaded models retain an authoritative local path and are collapsed to `source="both"`: `makermodslab/models.py:600-652`
- A non-directory import is immediately queried and stored as a Hub source: `makermodslab/jobs.py:1288-1327`
- Hub-imported checkpoint listing calls the Hub API: `makermodslab/jobs.py:774-789`

### P2 — A selected local model can be deleted during active startup

**Status:** Confirmed

**Verdict:** CONFIRMED — `policy_path` is written to the meta only at subprocess commit (makermodslab/rollout.py:846-851); during download/preflight the meta holds only `phase`/`policy_ref` (makermodslab/rollout.py:926), so `inference_in_use_path()` returns None (makermodslab/rollout.py:965-977) and `_model_in_use` permits deletion (makermodslab/models.py:974-976).
**Priority:** P2 — narrow concurrent window (another tab/API caller during startup); concrete harm is limited to local-only checkpoints — a Hub-ref run just re-downloads (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. Inference claims the active slot and resolves a local checkpoint.
2. It proceeds through follower staging and hardware preflight.
3. Before subprocess commit, `policy_path` exists only in the startup worker's local variable.
4. `inference_in_use_path()` returns `None` because metadata does not receive `policy_path` until after preflight and spawn.
5. Another tab or API caller can delete the selected model during this window.

**Impact**

Startup can touch and configure the follower, then fail to load or run because its checkpoint was removed. The deletion guard provides no protection during most of the active startup pathway.

**Evidence**

- Robot preflight occurs before subprocess commit: `makermodslab/rollout.py:776-800`
- `policy_path` is stored only during subprocess commit: `makermodslab/rollout.py:836-862`
- The in-use API returns only that metadata field: `makermodslab/rollout.py:969-981`
- Model deletion treats a missing in-use path as permission to continue: `makermodslab/models.py:962-987`

### P2 — Overlapping status polls can consume and erase a terminal failure

**Status:** Confirmed

**Verdict:** CONFIRMED — `setInterval(tick, POLL_MS)` has no in-flight guard (frontend/src/pages/Inference.tsx:206) and `doneRef` is set only after the awaited log fetch (frontend/src/pages/Inference.tsx:143 then :155/:163), while the terminal payload is one-shot server-side — both finalisation paths clear `_inference_meta` on read (makermodslab/rollout.py:1089-1091, :1119-1123); the secondary claim (log mined without joining the pump thread, makermodslab/rollout.py:1130) also holds but is largely mitigated by the pump's per-line `flush()` (makermodslab/rollout.py:176).
**Priority:** P2 — lost error reporting only, no hardware effect; the operator loses the failure hint, not control of the arm (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. The frontend starts an async `tick` every second without preventing overlap.
2. Tick A receives the one-shot terminal failure payload; the backend clears terminal metadata while returning it.
3. Tick A then waits for the log request.
4. Tick B can run meanwhile, receive ordinary idle status, overwrite page state, mark exit handled, and navigate home.
5. Tick A's later attempt to freeze the failure display is too late.

**Impact**

A real download, preflight, or runtime failure can disappear before the operator sees its error and hint.

The same status finalization can read an incomplete error because it does not join the stdout-pump thread before mining the log.

**Evidence**

- Terminal startup failure is reported once and immediately cleared: `makermodslab/rollout.py:1093-1111`
- Frontend sets status, then awaits a log fetch before terminal handling: `frontend/src/pages/Inference.tsx:123-170`
- `setInterval` launches ticks without an in-flight guard: `frontend/src/pages/Inference.tsx:205-211`
- Subprocess finalization mines the log while stdout pumping is independently threaded: `makermodslab/rollout.py:164-200`, `makermodslab/rollout.py:1112-1151`

### P2 — External bimanual checkpoints with right-arm cameras are mapped to the wrong feature keys

**Status:** Confirmed

**Verdict:** CONFIRMED — the modal strips a unique `right_` prefix to the bare name (frontend/src/components/landing/InferenceModal.tsx:143-149) and the backend attaches every camera to the BiSO left arm (makermodslab/rollout.py:638-639), so the checkpoint's `right_wrist` becomes `left_wrist`; the `left_x`+`right_x` collision case yields `left_left_x`/`left_right_x` as described.
**Priority:** P2 — affects externally recorded bimanual checkpoints only; makermodslab cannot produce `right_*` camera checkpoints itself (per the code's own defensive-branch comment) (verified 2026-07-14 against the current working tree)

**Broken pathway**

For an external bimanual checkpoint expecting `right_wrist`, the modal strips the unique `right_` prefix and sends `wrist`. The backend puts every camera on the BiSO left-arm config, so LeRobot exposes it as `left_wrist`, not the expected `right_wrist`.

If a checkpoint contains both `left_x` and `right_x`, collision handling keeps both full names, but placing them on the left arm produces `left_left_x` and `left_right_x`. Neither key matches the checkpoint.

**Impact**

Externally recorded bimanual policies with right-arm camera features cannot receive the image observations they were trained against.

**Evidence**

- Modal strips arm prefixes and retains full names only on collisions: `frontend/src/components/landing/InferenceModal.tsx:120-149`
- Backend always attaches bimanual cameras to the left arm: `makermodslab/rollout.py:622-642`

### P2 — A real mid-run motor overload can be mislabeled as a cleanup-only warning

**Status:** Confirmed

**Verdict:** CONFIRMED (deliberate, documented limitation) — `_classify_outcome` downgrades any post-marker nonzero exit whose mined text matches `overload`/`torque_enable` (makermodslab/rollout.py:565-578, makermodslab/utils/errors.py:33-45) with no teardown-vs-active evidence; the in-process `classify_outcome` docstring explicitly owns this as rollout's log-tail-only fallback (makermodslab/utils/errors.py:72-77), so it is an acknowledged tradeoff rather than an oversight.
**Priority:** P2 — reporting only, mildly safety-adjacent: a real mid-run motor fault is understated as an amber warning, but the run has already ended either way (verified 2026-07-14 against the current working tree)

**Broken pathway**

Any nonzero exit after the rollout-start marker is classified `ran_with_warning` when the mined error includes `overload` or `torque_enable`. The classifier has no evidence that the failure occurred during teardown rather than while the policy was actively driving.

**Impact**

A real runtime motor failure can be downgraded from a failed inference to an amber cleanup warning, understating the run's outcome and the need for operator action.

**Evidence**

- Rollout outcome uses only “started” plus text markers: `makermodslab/rollout.py:567-580`
- Broad cleanup markers are `overload` and `torque_enable`: `makermodslab/utils/errors.py:29-45`
- Existing test codifies only the text-based assumption, not teardown timing: `tests/test_rollout.py:875-890`

### P3 — Active startup can display the previous inference run's log

**Status:** Confirmed

**Verdict:** CONFIRMED — the log file is created only at spawn, after download and preflight (makermodslab/rollout.py:795-798); `_resolve_inference_log_path` falls back to the newest historical `*.log` whenever the active meta lacks a path (makermodslab/rollout.py:1044-1055); the page fetches the log on every poll tick (frontend/src/pages/Inference.tsx:142-147).
**Priority:** P2 — stale prior-run log visible during startup; reporting-only, brief window (verified 2026-07-14 against the current working tree, coordinator-ratified)

**Broken pathway**

The active session has no `log_path` during model resolution and robot preflight because the log is created just before spawn. The log endpoint falls back to the newest historical log whenever active metadata lacks a usable path, and the page fetches logs on every startup tick.

**Impact**

The current startup can show unrelated errors and output from a previous rollout, misleading diagnosis and progress interpretation.

**Evidence**

- Log path is created only after preflight: `makermodslab/rollout.py:793-800`
- Missing active path falls back to the newest historical log: `makermodslab/rollout.py:1042-1059`
- Page fetches log output on every poll: `frontend/src/pages/Inference.tsx:140-147`

### P3 — Hub checkpoint cache conflates imported-model and cloud-job listing rules

**Status:** Confirmed

**Verdict:** CONFIRMED — imported records fetch via `_list_imported_hub` (root-or-tree) while cloud records use the tree-only default (makermodslab/jobs.py:1460-1467), but `_list_cloud_cached` keys its 30s TTL cache on `repo_id` alone with no scanner mode (makermodslab/jobs.py:1490-1494), so the first lookup's shape is served to the other record type.
**Priority:** P2 — lowest-priority P2 (proposed P3 folded into P2 on the three-tier scale); transient (30s TTL) and requires the same repo registered as both an imported and a cloud record (verified 2026-07-14 against the current working tree)

**Broken pathway**

Cloud training jobs use a checkpoints-tree-only scanner. Imported Hub models use a root-or-tree scanner. Both results are cached solely by `repo_id`, without the scanner/listing mode in the key.

A tree-only lookup can cache an empty result for a valid root model and cause its imported record to show no checkpoints for the cache lifetime. The reverse order can also give a cloud-job record root-model semantics it did not request.

**Impact**

Checkpoint lists can temporarily disappear or take the wrong shape when the same repository is represented by both record types.

**Evidence**

- Imported and cloud scanners have different root handling: `makermodslab/jobs.py:774-800`
- Record type selects a different scanner: `makermodslab/jobs.py:1465-1472`
- Cache key contains only `repo_id`: `makermodslab/jobs.py:1486-1500`

## Design gaps and lower-confidence risks

### Required language task is not enforced

**Verdict:** CONFIRMED (gap) / CANNOT-VERIFY-STATICALLY (user-facing failure — policy-dependent) — `canStart` has no task check (frontend/src/components/landing/InferenceModal.tsx:343-350), the task input renders for `requires_task` policies but empty is submittable (frontend/src/components/landing/InferenceModal.tsx:534-550), and the backend defaults `task=""` (makermodslab/rollout.py:71); whether a blank task degrades or fails needs a live run of a language-conditioned policy.
**Priority:** P2 — preflight gap on a common flow for language-conditioned policies; failure mode is policy-dependent, not a safety issue (verified 2026-07-14 against the current working tree)

Language-conditioned policies expose `requires_task`, but Start remains enabled for a blank task and the backend defaults to `task=""`. The outcome depends on the policy implementation, so this is a preflight gap rather than a universally reproducible failure.

Evidence: `frontend/src/components/landing/InferenceModal.tsx:343-350`, `frontend/src/components/landing/InferenceModal.tsx:534-549`, `makermodslab/rollout.py:65-71`.

### Backend duration has no positive lower bound

**Verdict:** CONFIRMED — `duration_s: int = 60` is unvalidated (makermodslab/rollout.py:73) and passed verbatim to the subprocess (makermodslab/rollout.py:595); verified against the installed lerobot 0.6.0: `duration: float = 0.0  # 0 = infinite (24/7 mode)` (.venv lerobot/rollout/configs.py:224) and the script prints "infinite" for nonpositive duration, so a direct API caller gets an unbounded autonomous run; the UI's `min={1}` is cosmetic (frontend/src/components/landing/InferenceModal.tsx:557).
**Priority:** P2 — safety-adjacent (unbounded run) but reachable only by a direct API caller; verified against installed lerobot 0.6.0 that duration<=0 is infinite (verified 2026-07-14 against the current working tree)

The UI uses `min={1}`, but `InferenceRequest.duration_s` accepts any integer and passes it directly to rollout. A direct API caller can therefore bypass the UI's “Max duration” constraint. The pinned rollout runtime treats nonpositive duration as unbounded.

Evidence: `makermodslab/rollout.py:65-71`, `makermodslab/rollout.py:583-605`, `frontend/src/components/landing/InferenceModal.tsx:551-559`.

### Multiple roles may bind the same physical camera

**Verdict:** CONFIRMED (gap) — `allCamerasBound` enforces only that every role resolves to a live camera, not device uniqueness (frontend/src/components/landing/InferenceModal.tsx:339-341); whether the second open fails or duplicates frames is backend/device-dependent, as the entry says.
**Priority:** P2 — misconfiguration slips past preflight; likely addressed by the parked 2026-07-10 camera-lock / 409-gate branch, which is not in this tree (verified 2026-07-14 against the current working tree)

The modal verifies that every role resolves to a live camera, but does not enforce unique devices. Two expected observation keys can therefore be configured with the same OpenCV index. Whether the second open fails or duplicates frames is backend/device-dependent.

Evidence: `frontend/src/components/landing/InferenceModal.tsx:339-350`.

### Server arm-count guard loses the frontend's action-dimension fallback

**Verdict:** CONFIRMED — the modal computes arm count from `state_dim ?? action_dim` (frontend/src/components/landing/InferenceModal.tsx:322) but forwards only `state_dim` (frontend/src/components/landing/InferenceModal.tsx:417), and the server's `_arm_count_mismatch` defers to the subprocess when `checkpoint_state_dim` is None (makermodslab/rollout.py:371-372), so an action-only checkpoint lacks the authoritative backend guard.
**Priority:** P2 — lowest-priority P2 (proposed P3 folded into P2 on the three-tier scale); action-only checkpoints remain UI-guarded, and the subprocess's own shape check still catches the mismatch, just less legibly (verified 2026-07-14 against the current working tree)

The modal determines checkpoint arm count from `state_dim ?? action_dim`, but sends only `state_dim` as `checkpoint_state_dim`. A checkpoint with action width but no state width is protected by the current UI yet lacks the claimed authoritative backend guard.

Evidence: `frontend/src/components/landing/InferenceModal.tsx:315-337`, `frontend/src/components/landing/InferenceModal.tsx:399-410`, `makermodslab/rollout.py:82-86`.

## Validation

Focused mocked and temporary-directory tests were run without hardware or network access:

```text
.venv/bin/python -m pytest tests/test_rollout.py tests/test_log_endpoints.py -q
67 passed
```

The passing suite validates existing isolated helpers and nominal lifecycle branches. It does not invalidate the concurrency and cross-request defects above because those interleavings are not covered.

No cameras, serial ports, robots, user cache files, Hub tokens, or external network services were accessed. No production or test files were changed, staged, or committed.

## Coverage gaps

High-value missing regression scenarios are:

- Stop during follower preflight, rather than only during model download.
- Stop termination failure with an assertion that the child is still tracked until confirmed dead.
- Stop inference A, immediately start B, then complete or fail A's stale worker.
- Concurrent inference/teleoperation/recording starts under a shared barrier.
- Calibration and inference start against the same follower port.
- Explicit Stop request failure followed by back navigation or unmount.
- Overlapping status ticks around the one-shot terminal payload.
- Downloaded Hub model launch with Hub access unavailable.
- Local model deletion between policy resolution and subprocess commit.
- Bimanual `right_*` and colliding left/right camera feature mappings.
- Server shutdown while an inference child is active.
- Mid-run overload versus teardown-only overload outcome classification.
