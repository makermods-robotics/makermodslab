# Recording Bug List

Audit date: July 14, 2026  
Current branch and commit: `andrew` at `257580f`  
Working-tree note: the audit also includes the uncommitted LeRobot v0.6 migration currently present in `makermodslab/record.py`, `pyproject.toml`, `uv.lock`, and related files.

Pathway baseline: `internal_docs/complete functionalities/Recording Interface Pathways.md`, originally audited against `518ca56`.

Scope: fresh and resumed dataset recording, single-arm and bimanual setup, request construction, dataset creation, hardware connection, arm identity, calibration application, follower power, episode/reset control, status and log polling, Done/Quit/page-leave behavior, dataset handoff, rest-pose return, torque release, cross-feature ownership, failure recovery, and the current LeRobot v0.6 migration.

No hardware, user dataset, Hub repository, or external service was touched during this audit. Runtime validation used mocked tests and process-local status inspection only.

## Severity key

- **P0:** immediate, broadly reachable destructive or physical-safety failure with no meaningful mitigation.
- **P1:** major feature blocker, data-loss path, orphaned hardware/session path, or hardware-safety boundary failure.
- **P2:** significant correctness, recovery, or operator-trust failure with narrower reach or an available workaround.
- **Design gap:** intended product behavior is missing, but the current implementation does not establish one unambiguous defective outcome.

## Confirmed bugs

### P0 — none found

No recording defect was classified as P0. The current v0.6 migration blocks recording and several P1 paths can lose data or leave hardware ownership uncertain, but none was shown to cause an immediate destructive or unsafe outcome on every invocation.

### P1 — [FIXED IN TREE] The current LeRobot v0.6 migration makes every fresh and resumed recording fail before capture

**Verdict:** ALREADY-FIXED-IN-TREE (fixed in uncommitted tree, 2026-07-14) — the claim is factually inverted. Installed lerobot `0.6.0`'s `DatasetRecordConfig` exposes `rgb_encoder`/`depth_encoder`, NOT `vcodec` (`.venv/.../lerobot/configs/dataset.py:62-64`); `LeRobotDataset.create()`/`.resume()` accept `rgb_encoder`/`depth_encoder` (`lerobot_dataset.py:676-677,785-786`). The uncommitted `record.py:1198-1199,1229-1230` correctly reads `cfg.dataset.rgb_encoder`/`depth_encoder`; every field record.py passes/reads was verified present in the installed v0.6.0. No `AttributeError` occurs. The audit's "7 test failures" almost certainly came from running pytest against a venv still on the OLD pinned commit (`82dffde`, which had `vcodec`) before `pip install -e .` synced it to v0.6.0. Committed HEAD `257580f` (pre-diff, `vcodec=cfg.dataset.vcodec`) WOULD fail on v0.6.0; the uncommitted migration is the fix.
**Priority:** resolved — would have been P1 (blocks all recording) if the tree still used `vcodec`; the uncommitted v0.6 migration already corrects it, no action needed (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. The current working tree changes the dependency to LeRobot `v0.6.0` (`pyproject.toml:15-20`; `uv.lock:1055-1059`).
2. LeRobot v0.6's `DatasetRecordConfig` exposes `vcodec`; it does not expose `rgb_encoder` or `depth_encoder`.
3. MakerMods Lab now reads `cfg.dataset.rgb_encoder` and `cfg.dataset.depth_encoder` while constructing both a resumed and a fresh `LeRobotDataset` (`makermodslab/record.py:1201-1216`, `makermodslab/record.py:1228-1243`).
4. Argument evaluation raises `AttributeError` before `LeRobotDataset.create()` or `.resume()` can run.
5. The asynchronous recording start has already returned success, so the user reaches the recording screen only to receive a terminal failure without recording an episode (`makermodslab/record.py:537-657`).

**Impact**

Recording is functionally unavailable in the current v0.6 worktree for both new and append sessions. The failure occurs after the user completes setup and after the backend reports a successful start.

**Validation**

`tests/test_record.py` produced 7 failures and 53 passes. All seven lifecycle failures stop at `AttributeError: 'DatasetRecordConfig' object has no attribute 'rgb_encoder'`. Direct inspection of the installed v0.6 signatures confirms that `LeRobotDataset.create()` and `.resume()` accept `vcodec`, not `rgb_encoder` or `depth_encoder`.

**Fix boundary**

Use the v0.6 configuration and dataset-construction contract consistently, then add fresh and resume regression tests that reach `save_episode()` under the pinned dependency.

### P1 — A logically rejected start is HTTP 200, and the frontend treats it as a successful recording

**Verdict:** CONFIRMED — `/start-recording` returns `handle_start_recording()`'s dict verbatim (`server.py:838-841`), so `{success:false}` (from active-session/invalid-name/setup-fail at `record.py:463-485,657-660`) serializes as HTTP 200; the client checks only `response.ok` (`Recording.tsx:344`) and sets `recordingSessionStarted=true` + "Recording Started". The intended non-2xx rejection branch (`Recording.tsx:350-361`, whose comment even names a "409 already-active" that no code returns) is unreachable.
**Priority:** P1 — with no session id and process-global state, a second client hitting an already-active session begins polling/controlling that session (Done/Quit act on it); an invalid-name rejection strands the page on a dead "Connecting…" screen. Wrong behavior in a reachable flow, no in-page workaround (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. `handle_start_recording()` returns dictionaries such as `{success: false, message: ...}` for an active session, invalid repository ID, or synchronous setup failure (`makermodslab/record.py:463-487`, `makermodslab/record.py:659-662`).
2. `/start-recording` returns that dictionary directly and never converts the rejection to a non-2xx response (`makermodslab/server.py:838-842`).
3. The recording page checks only `response.ok`, not `data.success` (`frontend/src/pages/Recording.tsx:335-360`).
4. Because HTTP 200 is `ok`, it sets `recordingSessionStarted=true` and displays “Recording Started” even when the backend explicitly rejected the request.

**Impact**

- If another recording is active, the new page begins polling and controlling the process-global existing session. Done, Quit, re-record, or page-leave discard can therefore act on a session the page did not start.
- If the request was rejected before any session became active, the page can remain stuck polling an idle or non-terminal state while claiming recording started.
- The intended “Error Starting Recording” branch is unreachable for normal backend logical rejections.

**Fix boundary**

Make logical start rejection non-2xx and also require `data.success === true` in the client. A durable fix should return an immutable session identifier and require it on later control requests.

### P1 — Setup failures after the follower connects can leave hardware and cameras open

**Verdict:** CONFIRMED-WITH-CORRECTIONS — real leak: `teleop.connect()` failing (`record.py:1324-1326`) re-raises without disconnecting the already-connected follower, and the only unconditional rest/torque-disable/disconnect `try/finally` starts later at the episode loop (`record.py:1410-1632`), so a pre-loop raise never cleans up; the worker (`record.py:590-647`) clears `recording_active` and retains no device handle. CORRECTION to the "partially configured motors" framing: the follower is left with torque OFF, not energized — lerobot's `so_follower.configure()` runs inside `torque_disabled()` and never re-enables (`.venv/.../so_follower.py:159-173`), so this is a serial-port + camera resource leak, NOT a torque-left-enabled hazard.
**Priority:** P1 — leaked follower port/cameras after a failed start; a retry hits busy ports/cameras and there is no recording control left to release the leaked device (session already inactive). No physical-safety hazard (torque off), so not P0 (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. MakerMods Lab constructs the dataset, connects the follower and its cameras, then connects the leader (`makermodslab/record.py:1201-1243`, `makermodslab/record.py:1245-1335`).
2. A final follower-connect failure can raise after a partial connection. More importantly, if the follower succeeds and the leader connection fails, the exception is re-raised without disconnecting the already-connected follower (`makermodslab/record.py:1318-1335`).
3. The only unconditional rest-return, torque-disable, and disconnect block begins later, around the episode loop (`makermodslab/record.py:1417-1418`, `makermodslab/record.py:1619-1639`). Exceptions before that `try/finally` never enter its cleanup.
4. The worker catches the exception, clears `recording_active`, and reports a failed session, but it does not retain a device reference that a later stop can clean up (`makermodslab/record.py:592-647`).

**Impact**

The follower serial bus, cameras, or partially configured motors can remain open after a failed start. A retry may report busy ports/cameras, and the operator has no recording control capable of releasing the leaked device because the session is already marked inactive.

**Fix boundary**

Wrap the complete construction/connect/setup lifecycle in one ownership-aware cleanup block. Cleanup must attempt every connected device independently and surface an explicit torque/port warning when release is uncertain.

### P1 — Reaching the episode-duration limit discards the full take and can retry forever

**Verdict:** CONFIRMED — when `record_loop` returns naturally at `episode_time_s` without `_exit_early_triggered`, the else-branch sets `rerecord_episode=True`, `clear_episode_buffer()`, resets, and retries the same episode without incrementing `saved_episodes` (`record.py:1474-1490`). Only an operator End Episode / SPACE / → (exit-early) marks the take for saving (`record.py:1466-1473`; `Recording.tsx:670-675,762-774`). The setup modal labels the value "Episode duration (seconds)" and the page renders it as a countdown/progress limit (`RecordingModal.tsx:312`; `Recording.tsx:623-631,738-754`), implying reaching it completes the take. This is a deliberate MakerMods Lab addition (upstream lerobot saves on timeout), not an upstream behavior.
**Priority:** P1 — an operator who lets the labeled "duration" elapse silently loses a completed demonstration (expensive physical demo time, which this project weighs heavily) and the episode re-records instead of advancing. Clear workaround (press End Episode before the timer), but the label/countdown actively mislead toward the lossy path (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. The setup modal labels the value “Episode duration (seconds)” and the recording page renders a countdown/progress limit (`frontend/src/components/landing/RecordingModal.tsx:306-339`, `frontend/src/pages/Recording.tsx:623-631`, `frontend/src/pages/Recording.tsx:738-751`).
2. When `record_loop()` returns naturally at that limit, `_exit_early_triggered` is false.
3. MakerMods Lab interprets that natural completion as a re-record request, sets `rerecord_episode=true`, clears the episode buffer, performs reset, and retries the same episode (`makermodslab/record.py:1472-1545`).
4. Only the operator's **End Episode** action marks the take for saving. A hands-off timed session never advances `saved_episodes` and can repeat indefinitely.

**Impact**

A user can perform a complete, apparently valid timed demonstration and lose it at exactly the configured duration. Unattended or hands-busy recording never reaches the requested episode count, continually discards full takes, and holds the robot/cameras until manually stopped.

**Fix boundary**

Either save a naturally completed timed take, matching the normal meaning of an episode duration, or rename and explain the setting as a mandatory retake deadline. Tests must distinguish timeout-save, explicit End Episode, explicit re-record, and mid-episode stop.

### P1 — Terminal status erases the number of saved episodes and hides recoverable results

**Status:** In progress — 2026-07-14: the worker now snapshots `saved_episodes` into a new `last_session_saved_episodes` global before its finally zeroes the live counter, and `handle_recording_status()` echoes it in every ended payload (`makermodslab/record.py`); the frontend already consumes `saved_episodes` to gate the recovery actions and now shows the kept count in the warning banner (`Recording.tsx`). Validated by unit test `test_terminal_status_retains_saved_episodes`; live hardware validation pending.

**Verdict:** CONFIRMED — the worker resets `saved_episodes=0` in its finally (`record.py:644`), and `handle_recording_status()` adds `saved_episodes` to the payload only while `recording_active` is true (`record.py:823-826`), so every terminal payload omits it. Clean completion hands off `status.saved_episodes || 0` (→0) to `/upload` (`Recording.tsx:309`); the failed/warning end-state gates "Continue to upload" and "Discard & exit" behind `keptSomething = savedEpisodes > 0` (`Recording.tsx:799-800,840,851`), so a warning/failed session that DID keep episodes shows only "Back to home".
**Priority:** P1 — a `ran_with_warning` session tells the operator "your episodes are safe" yet offers no button to reach them; the intended recovery flow is blocked. The dataset is still on disk and reachable via the normal dataset browser (workaround), and clean uploads still work by repo_id, so not P0 (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. The worker resets the process-global `saved_episodes` to zero before marking the session fully inactive (`makermodslab/record.py:642-647`).
2. `handle_recording_status()` includes `saved_episodes` only while `recording_active` is true (`makermodslab/record.py:823-845`). Terminal status therefore omits it entirely.
3. Clean completion constructs the post-recording handoff with `status.saved_episodes || 0` (`frontend/src/pages/Recording.tsx:288-317`).
4. Failed/warning completion shows recovery actions only when the missing terminal value is greater than zero (`frontend/src/pages/Recording.tsx:797-860`).

**Impact**

- Every completed recording is handed off as though zero episodes were saved, regardless of on-disk reality.
- A failed or warning session that retained episodes hides **Continue to upload** and **Discard & exit**, leaving only **Back to home**.
- The UI cannot distinguish “nothing saved” from “saved count was forgotten.”

**Validation**

A process-local probe set `saved_episodes=4` with an ended session and called `handle_recording_status()`. The returned terminal payload contained no `saved_episodes` field.

**Fix boundary**

Persist a terminal saved count separately from live counters, include it in every ended payload, and test clean, warning, failure, fresh-discard, resume, and zero-episode outcomes.

### P1 — “Discard & exit” after a resumed session deletes the entire pre-existing dataset

**Status:** In progress — 2026-07-14: the ended-session `discardAndExit()` is now resume-aware (`Recording.tsx`) — for a RESUME session it deletes NOTHING (keeps every episode, shows a "kept N episodes, dataset was resumed" toast, and the button reads "Exit without uploading" with non-destructive styling); whole-directory `/delete-dataset` deletion is kept only for FRESH sessions, where it is a correct full rollback. Path (b) from the settled design was chosen: lerobot 0.6.0's `dataset_tools.delete_episodes` exists but builds a whole NEW dataset (re-encoding video) and needs a pre-session episode count that nothing tracks at the stateless `/delete-dataset` path — neither cheap nor reliable against expensive demo data, so per-episode surgical removal was rejected. Backend `handle_delete_dataset` is intentionally left non-resume-aware (it is the general dataset-browser delete). Validated by unit test `test_worker_quit_keeps_resumed_dataset` (worker-level resume-safety) plus the existing `_discard_session_dataset` resume guard; live hardware validation pending.

**Verdict:** CONFIRMED (latent; currently masked by the terminal-count bug above) — `discardAndExit()` sends the repo_id to `/delete-dataset` ignoring `recordingConfig.resume` (`Recording.tsx:520-535`), and `handle_delete_dataset()` `shutil.rmtree`s the whole resolved directory with no resume-awareness (`record.py:849-885`). The backend's own worker discard path IS resume-safe (`_discard_session_dataset` no-ops on resume, `record.py:977-979`), but the frontend button bypasses it. The button is only rendered when `keptSomething > 0`, which is always false today because terminal `saved_episodes` is omitted (entry above), so it is currently unreachable.
**Priority:** P0 — once the terminal-count bug is fixed, a resume-session "Discard & exit" deletes the entire pre-existing dataset (all episodes recorded before this session): unrecoverable loss of expensive demo data. DEPENDENCY: must be fixed together with / before the terminal-`saved_episodes` fix, which is what unmasks it (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. A resume session appends into an existing repository and deliberately does not delete it when active-session Quit is requested (`makermodslab/record.py:624-640`, `makermodslab/record.py:947-985`).
2. The ended failure/warning UI offers **Discard & exit** whenever it believes something was kept.
3. That callback ignores `recordingConfig.resume` and sends the repository ID to the general `/delete-dataset` endpoint (`frontend/src/pages/Recording.tsx:517-535`).
4. The backend deletes the entire resolved repository directory, not merely episodes appended by the last session (`makermodslab/record.py:852-886`).

**Impact**

Once the missing terminal count is fixed, using **Discard & exit** after an errored or warning resume session can erase the original dataset and all episodes that existed before recording began. This is a latent but fully implemented data-loss path currently masked by the terminal-count bug above.

**Fix boundary**

Never present whole-repository deletion as session rollback. For resume, either retain all committed episodes, implement a transactional append with a session boundary that can be rolled back safely, or require an explicit whole-dataset deletion confirmation naming the pre-existing episode count.

### P1 — Done and Quit disarm page-leave safety before the backend acknowledges a stop

**Verdict:** CONFIRMED — `doStopRecording()` calls `markHandled()` before the fetch and never checks `response.ok`/`{success}` (`Recording.tsx:448-471`); `confirmQuit` navigates home unconditionally (`Recording.tsx:489-495`). `/stop-recording` returns HTTP 200 even when no session is active (`server.py:844-853`; `record.py:691-692`). Once latched, the exit guard's unmount fallback is suppressed (`useSessionExitGuard.ts:68-70,159-166`).
**Priority:** P1 — a transient stop failure leaves recording active after the UI has navigated away with its only fallback disarmed; hardware keeps running with no visible control, and a fresh session may report a discard that never happened. No in-page recovery (backend restart / reload) (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. `doStopRecording()` calls `markHandled()` before sending the stop request (`frontend/src/pages/Recording.tsx:445-456`).
2. It does not check `response.ok` or the returned `{success}` value; any HTTP response produces a “Finishing” or “Quitting” toast (`frontend/src/pages/Recording.tsx:455-468`).
3. The stop endpoint also returns logical failure as HTTP 200 when no session is active (`makermodslab/server.py:844-853`, `makermodslab/record.py:684-702`).
4. **Quit** navigates home unconditionally after the attempted request, including network failure (`frontend/src/pages/Recording.tsx:489-495`).
5. Because the exit guard is already latched handled, unmount cannot issue its best-effort fallback discard (`frontend/src/hooks/useSessionExitGuard.ts:58-76`, `frontend/src/hooks/useSessionExitGuard.ts:157-168`).

**Impact**

A transient failure can leave recording active after the interface has navigated away and after its only fallback has been disabled. The follower, leader, and cameras may continue running without visible controls; for a fresh session, the user may also believe a requested discard occurred when it did not.

**Fix boundary**

Latch the exit guard only after an acknowledged stop for the correct session. Quit must stay on the page and retain/re-arm safety controls when stop fails.

### P1 — Resume recording can mutate a dataset while upload, merge, or local training uses it

**Verdict:** CONFIRMED — `handle_start_recording()` checks only recording/teleop/inference flags + name syntax (`record.py:461-485`) and never calls `_dataset_in_use()`, which exists and guards the reverse direction (delete/rename/upload) for recording/upload/merge/local-training (`datasets.py:697-745`). A resume session writes directly into the un-timestamped existing repo_id (`record.py:520-521`), so an upload/merge/train already reading it is not refused.
**Priority:** P2 — reachable only if the user starts an upload/merge/local-training on a dataset and THEN starts a resume recording into that same dataset; narrow reach in a single-user local app. Consequence when hit is real (partial publish / inconsistent training metadata / merge race), so P1 if that ordering proves common (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. MakerMods Lab already has `_dataset_in_use(repo_id)`, which tracks active recording, upload, merge output, and local training (`makermodslab/datasets.py:696-748`). Delete, rename, and upload paths use this guard.
2. `handle_start_recording()` does not call it. It checks only recording, teleoperation, inference, and name syntax (`makermodslab/record.py:445-511`).
3. A resumed session writes directly into the existing repository ID without timestamping (`makermodslab/record.py:517-535`).
4. Therefore an upload, merge output, or local training read can begin first, after which resume recording can still append and rewrite metadata under that active operation.

**Impact**

The uploader can publish a changing directory, training can observe inconsistent episode/data metadata, and merge output can race an append. The operation that starts second is not refused even though the reverse ordering is guarded.

**Fix boundary**

Introduce an atomic per-dataset read/write lease shared by recording, upload, merge, delete/rename, and training. A one-time boolean check is not sufficient because two starts can race between checking and claiming ownership.

### P1 — Robot hardware ownership is incomplete and non-atomic across features

**Verdict:** CONFIRMED — each feature holds its OWN module-local `_state_lock` (`record.py:220`, `teleoperate.py:102`, `rollout.py:111`) while reading the OTHER modules' booleans (e.g. `record.py:461-473` reads `teleoperate.teleoperation_active`/`rollout.inference_active` under record's lock), so two concurrent starts can each observe the other inactive before claiming their own flag (cross-module TOCTOU). Recording also checks none of calibration/auto-calibration/wiggle (`record.py:461-485`); wiggle opens the bus with no lease (`wiggle.py`).
**Priority:** P2 — requires genuinely concurrent starts in a single-user local app (SingleTabGuard limits tabs), and OS exclusive serial-open usually makes the loser fail with a port-busy error rather than issue competing motor commands. Real ownership gap, low practical reach; the shared-coordinator fix is worth doing but the window is narrow (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. Recording, teleoperation, and inference each use their own module-local `_state_lock` while checking the other modules' booleans (`makermodslab/record.py:213-215`, `makermodslab/record.py:445-494`; `makermodslab/teleoperate.py:101-103`; `makermodslab/rollout.py:109`). Two simultaneous starts can both observe the other feature as inactive before either claims its own flag.
2. Recording checks teleoperation and inference but not manual calibration, auto-calibration, batch auto-calibration, or gripper wiggle (`makermodslab/record.py:463-476`).
3. Manual and automatic calibration similarly claim only their own manager state (`makermodslab/calibrate.py:232-278`, `makermodslab/auto_calibrate.py:199-252`). Wiggle opens and commands the requested serial bus without a feature lease (`makermodslab/wiggle.py:55-85`).
4. The browser's single-tab takeover changes UI election only; it does not stop or transfer ownership of the old tab's live hardware session (`frontend/src/components/SingleTabGuard.tsx:64-89`, `frontend/src/components/SingleTabGuard.tsx:109-139`).

**Impact**

Two features can open the same serial ports or issue competing motor commands. This is especially reachable when a second browser tab takes over or two API starts arrive concurrently.

**Fix boundary**

Use one server-owned hardware-session coordinator with atomic acquire/release, exact port ownership, release-in-progress state, and a session token required by every control endpoint.

### P1 — Failure to apply the saved motor-power limit is hidden while recording proceeds

**Verdict:** CONFIRMED — `apply_motor_power()` and `clear_goal_velocity()` catch per-motor failures and RETURN warning lists (`motor_power.py:120-133,136-164,167-198`), but recording calls both and discards the returned lists (`record.py:1382,1386`); the warning never reaches a status payload (unlike teleoperation, which surfaces it).
**Priority:** P2 — triggers only on a per-motor register write failure (rare); the motor then keeps its previous limit / servo default (full power on a fresh power-up). Full power is not uncommanded motion, but it silently defeats an operator-selected safety cap with zero visibility — safety-adjacent. Bump to P1 if `motor_power` is treated as a hard safety guarantee for delicate setups (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. The selected robot's saved `motor_power` is sent in the recording request (`frontend/src/pages/Landing.tsx:212-239`).
2. `apply_motor_power()` catches per-motor failures and returns explicit warnings that affected motors retain their previous limit, or full power after power-up (`makermodslab/motor_power.py:136-164`).
3. Recording discards that returned warning and continues into the control loop (`makermodslab/record.py:1385-1393`). It also discards warnings from clearing stale goal-velocity limits.
4. Unlike teleoperation, the warning is never added to an operator-facing status payload.

**Impact**

An operator who selected a gentler limit can begin recording with one or more follower motors at their previous setting or 100%. A stale speed cap can also remain, causing unexpectedly slow motion. Neither deviation is visible in the recording screen.

**Fix boundary**

Treat the selected safety limit as an enforced precondition or require explicit operator confirmation of degraded operation before motion begins. Surface per-arm and per-motor failures before starting the episode loop.

### P1 — Rest-pose capture and return failures are logged but reported as a clean recording

**Verdict:** CONFIRMED — `capture_rest_pose()` returns `{}` on comm failure (`rest_pose.py:60-73`); `return_to_rest_pose()` returns `(arrived, reason)` with failure reasons no-pose/comm-error/settled/stalled/ceiling/cut-short (`rest_pose.py:76-129`), but `_return_followers_to_rest()` and its per-arm helper only LOG the reason and return `None` (`teleoperate.py:237-291`). The recording finally ignores the outcome, disables torque, and the worker sets `last_session_outcome="ok"` when no exception was raised (`record.py:1612-1632,589`). Done/Quit copy promises the arm returns to its start pose then goes limp (`recordingExit.ts:21,30-32`).
**Priority:** P2 — torque IS released either way (safe end state), so no torque-left-enabled hazard; the defect is that the arm may release where it stalled while the UI reports a clean return. Operator-trust plus a minor physical consideration (arm parked off its intended rest position could settle under gravity). P1 if the un-parked release is judged a real flop/collision risk on this rig (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. Rest-pose capture returns an empty pose on communication failure and does not fail or warn the session status (`makermodslab/rest_pose.py:60-73`, consumed by `makermodslab/record.py:1395-1405`).
2. Return can report `no-pose`, `comm-error`, `settled`, `stalled`, `ceiling`, or `cut-short` as unsuccessful outcomes (`makermodslab/rest_pose.py:76-129`, `makermodslab/rest_pose.py:149-192`).
3. `_return_followers_to_rest()` logs those outcomes but returns no aggregate result (`makermodslab/teleoperate.py:237-291`).
4. Recording ignores the return outcome, disables torque, and can finish with `outcome="ok"` (`makermodslab/record.py:1619-1639`, `makermodslab/record.py:583-614`).
5. Done/Quit copy promises that the arm returns to its starting position and then goes limp (`frontend/src/lib/recordingExit.ts:17-38`).

**Impact**

One or both followers can fail to reach the captured safe pose and then release torque where they stalled, while the operator is told the recording completed cleanly. A missing start pose silently converts the advertised return into an immediate release.

**Fix boundary**

Aggregate per-arm capture/return results into terminal status. A failed return should be a visible cleanup warning with the final reason and a clear physical-safety instruction.

### P1 — An arm-identity read failure silently disables the safety check before calibration writes and motion

**Verdict:** CONFIRMED-WITH-CORRECTIONS — on a `Present_Position`/`Homing_Offset` read failure `verify_arm()` returns `(None, None)` (no refusal, no warning) (`arm_identity.py:279-288`), so `verify_devices()` treats the arm as verified (`arm_identity.py:391-413`) and recording proceeds to write calibration to EEPROM and start motion (`record.py:1333-1376`). CORRECTION: this is a DELIBERATE, documented fail-open (`arm_identity.py:270-272`: "a read failure skips the check … a genuinely dead bus will fail loudly on the first real action"), and the practical exposure is narrow — it needs the identity read to fail yet the subsequent EEPROM write to succeed AND the arm to actually be swapped.
**Priority:** P2 — edge case with a coherent design rationale; the fail-open removes the swap guard only in a narrow intermittent-comm window. Safety-adjacent (could stamp a wrong calibration into a swapped arm), so worth a tri-state (verified/mismatch/unverifiable) hardening, but not a broadly reachable defect (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. If reading `Present_Position` or `Homing_Offset` fails, `verify_arm()` logs the error and returns no refusal and no warning (`makermodslab/arm_identity.py:252-288`).
2. `verify_devices()` therefore lets setup continue as though no identity problem was found (`makermodslab/arm_identity.py:391-413`).
3. Recording then writes the selected calibration to the connected devices, applies follower settings, and starts leader-to-follower control (`makermodslab/record.py:1337-1405`, `makermodslab/record.py:1417-1446`).

**Impact**

A transient read failure removes the guard intended to catch swapped or misassigned arms. MakerMods Lab can write a selected calibration into an unverified physical arm and begin motion without any operator-facing warning.

**Fix boundary**

Identity must be tri-state: verified, mismatch, or unverifiable. Unverifiable hardware must be flagged before calibration writes and motion, with any override explicit, narrow, and recorded in session status.

## P2 correctness and recovery bugs

### P2 — Ended-session deletion ignores the server result and always tells navigation to continue

**Verdict:** CONFIRMED — `discardAndExit()` awaits the fetch inside try/catch, never inspects `response.ok` or `{success}`, and calls `navigate("/")` regardless (`Recording.tsx:520-535`); `/delete-dataset` legitimately refuses on busy/missing (`record.py:868-872`). (Currently also gated behind `keptSomething`, same masking as the resume-delete entry.)
**Priority:** P2 — the user cannot tell whether the discard succeeded; compounds the resume whole-dataset-deletion entry (too destructive when accepted, falsely reassuring when refused). Narrow reach today (verified 2026-07-14 against the current working tree).

`discardAndExit()` catches transport errors, ignores non-2xx and `{success: false}` responses, shows no failure, and navigates home regardless (`frontend/src/pages/Recording.tsx:517-535`). The server can legitimately refuse deletion because the dataset is busy or missing (`makermodslab/record.py:867-879`). The user therefore cannot tell whether discard succeeded. This compounds the resumed-dataset deletion bug: the same control is simultaneously too destructive when accepted and falsely reassuring when refused.

### P2 — Re-record is accepted by the backend outside the recording phase

**Verdict:** CONFIRMED — `handle_rerecord_episode()` validates only `recording_active and recording_events is not None` (`record.py:738-749`), while the status payload advertises the control only during `current_phase == "recording"` (`record.py:779-780`). A stale poll or direct POST during reset/stopping sets shared `rerecord_episode`+`exit_early`, which can carry into the next loop.
**Priority:** P2 — requires a direct API call or a poll racing a phase change; affects flag state, not data on disk. Backend phase validation should be authoritative (verified 2026-07-14 against the current working tree).

The status payload advertises re-record only during `current_phase == "recording"`, but `handle_rerecord_episode()` validates only that the overall session is active (`makermodslab/record.py:738-753`, `makermodslab/record.py:778-784`). A stale poll or direct request during reset/stopping sets shared `rerecord_episode` and `exit_early` flags that can carry into the next loop. Backend phase validation should be authoritative, and episode-control events should be scoped to an episode/phase generation.

### P2 — The backend accepts recording parameters outside the interface's validated contract

**Verdict:** CONFIRMED — `RecordingRequest` uses unconstrained `int` for `num_episodes`/`episode_time_s`/`reset_time_s`/`fps` and a free-form `mode: str = "single"` with no validators (`record.py:268-302`); `create_record_config` routes any non-`"bimanual"` mode into the single-arm path (`record.py:376-382`). The frontend bounds these, but a direct API call can send zero/negative/arbitrary values.
**Priority:** P2 — reachable only via direct API (not the shipped UI); defense-in-depth. Add schema constraints + 422s if the endpoint is treated as product-facing (verified 2026-07-14 against the current working tree).

`RecordingRequest` uses unconstrained integers and a free-form `mode` string (`makermodslab/record.py:263-299`). The frontend bounds episodes and positive durations, but direct API calls can send zero/negative episodes or timing, unsupported FPS, arbitrary mode strings that silently fall into the single-arm path, and out-of-range values later clamped or passed downstream. If the endpoint is product-facing rather than an expert low-level API, these should be schema constraints with clear 422 responses.

## Design gaps and lower-confidence risks

**Verdict (all four gaps below):** CONFIRMED as accurate design-gap observations, not defects — the cited code exists as described (process-global state with no session id: `record.py:180-239`; router-location config + always-start-on-mount: `Recording.tsx:88-101,199-207,335-372`; module-global counters/logs; no in-page frame-health surface). These are missing-feature / robustness observations, so no P0–P2 severity is assigned; they map to the "Design gap" key. The "Cross-feature bug duplication" note correctly restates entries above (hardware mutex, arm-identity fail-open, motor-power, rest-return) and should share their fixes rather than be re-flagged (verified 2026-07-14 against the current working tree).

### No durable session identity, ownership, or reattachment contract

The page receives configuration only through router location state and always attempts a new start on mount (`frontend/src/pages/Recording.tsx:88-101`, `frontend/src/pages/Recording.tsx:201-211`, `frontend/src/pages/Recording.tsx:335-372`). Backend state and controls are process-global and carry no session ID (`makermodslab/record.py:180-239`, `makermodslab/record.py:665-850`). Reload, browser restart, a delayed beacon, or a second client cannot prove which session it owns. This makes safe reattachment impossible and allows a stale control request to act on a newer session.

### No episode review or selective commit workflow

The operator can save or re-record only the current in-memory take. Once `save_episode()` commits an episode, the recording interface provides no thumbnail/video review, quality flag, metadata edit, selective delete, or reorder step. Resume Quit also cannot roll back already appended episodes. A transactional session manifest would support review and safe session-level discard.

### Recording observability does not prove frame health

The live page shows phase, time, episode count, and an in-memory log, but no camera frame preview, per-camera freshness, dropped-frame count, effective resolution/FPS, disk-space estimate, encoder backlog, or serial retry rate. The audit did not classify this as a defect without a defined health contract, but camera/encoding failures can otherwise be discovered only after an exception or later dataset inspection.

### Recovery state is process-local

Active flags, phase, counters, event controls, logs, and terminal outcome live in module globals or memory-only buffers. A backend restart loses ownership and recovery context even if a partial dataset remains on disk. Empty-dataset cleanup and terminal handoff are therefore not crash-consistent.

### Cross-feature bug duplication needs one canonical owner

Hardware mutex, arm-identity fail-open, motor-power enforcement, and rest-return reporting also affect teleoperation, inference, and calibration pathways. They are included here because they directly break recording, but fixes should live in shared ownership/safety services and be referenced from each feature ledger rather than implemented independently.

## Validation performed

- Read the complete recording pathway baseline and traced the current frontend, API routes, recording worker, dataset persistence, shared exit guard, hardware helpers, and tests.
- Ran `.venv/bin/python -m pytest tests/test_record.py -q`: **53 passed, 7 failed**. All failures were the current v0.6 `rgb_encoder` compatibility regression.
- Inspected the installed v0.6 signatures for `DatasetRecordConfig`, `LeRobotDataset.create()`, and `LeRobotDataset.resume()`.
- Used a process-local, no-hardware status probe to confirm that terminal status omits a nonzero saved-episode count.
- Did not run a real server, connect hardware, mutate user data, start uploads/training/jobs, or contact Hugging Face.

## Test coverage gaps exposed by the audit

- No regression test currently reaches fresh/resume episode saving under the dependency version in the current worktree.
- No frontend test asserts both HTTP status and `{success}` for start/stop/delete actions.
- No test preserves and asserts terminal `saved_episodes` across clean, warning, and failed outcomes.
- No test covers ended-session discard after resume.
- No test simulates leader connection failure after follower/camera connection and verifies full cleanup.
- No test specifies whether natural episode timeout saves or discards.
- No concurrency test races recording against teleoperation, inference, calibration, upload, merge, or training ownership.
- No integration test covers reload, stale page-leave beacon, second tab, or session reattachment.
- Existing rest-return tests spy that the helper was called, but do not require a failed return to alter terminal outcome.

