# Recording Bug List

*Citations synced 2026-07-26 against `main` @ `ce42158`. **Only `file:line` coordinates were changed** — no prose, verdicts, severities, entry IDs or numbering were altered. References whose construct no longer exists at HEAD are marked inline `⚠ UNRESOLVED @ HEAD ce42158`.*

*Renumbered 2026-08-02 into one continuous per-module sequence: the former appended `N`-series became **R16–R30** (N1→R16 … N15→R30); originals **R1–R15** and the design gaps **D1–D5** keep their IDs. Every bug entry now carries a **First identified** date (and **Resolved**, where the entry attests a fix or a descope). Entries may also carry a **Fix:** line, directly under **First identified**, linking the pull request(s) that fix or attempt to fix them.*

> **Re-verified against `main` @ `c7d9f27` on 2026-07-23.** This list folds the 23-07-26 re-audit into the
> original. It **supersedes** the pre-reverification version, archived at
> [`archive/Recording Bug List (pre-reverify).md`](archive/Recording%20Bug%20List%20%28pre-reverify%29.md);
> the standalone re-audit is archived at
> [`archive/Recording REVERIFIED 23-07-26.md`](archive/Recording%20REVERIFIED%2023-07-26.md).
> Cross-module current-state overview: [`CURRENT 23-07-26.md`](CURRENT%2023-07-26.md).
>
> **Context you need for every verdict below:**
> - The original entries were forensically verified **2026-07-14 against the `andrew` branch at `257580f`**,
>   including that branch's uncommitted LeRobot v0.6 migration. **None of that state is on `main`.** Two entries
>   carried "Status: In progress" fix notes (R5, R6); **neither fix exists on `main`.**
> - **The redesign moved/deleted the whole recording surface.** `frontend/src/pages/Recording.tsx` and
>   `frontend/src/pages/Landing.tsx` are gone; the live session now lives in
>   `frontend/src/components/studio/CollectPanel.tsx`, `frontend/src/components/recording/RecordingSessionDialog.tsx`,
>   `frontend/src/components/studio/CollectHandoff.tsx`. Every frontend `file:line` in the original was dead;
>   `main`-current cites are layered on. Two behaviours changed materially: **resume recording is no longer
>   reachable from the shipped UI** (`CollectPanel.tsx:247` hardcodes `resume: false`), and **a default-on
>   automatic public Hub push was added** (new finding R17, ranked P0).
> - **PRs #5–#11 are CLOSED with `merged=no`.** Nothing here is in flight.
> - **Numbering:** the original used descriptive P1/P2 headings; `RANKING.md` numbers them **R1–R15** (existing
>   entries in order) **+ D1–D5** (design gaps). Those IDs are carried on each heading; new findings use
>   **N1–N9** — *renumbered 2026-08-02: the whole appended series became **R16–R30**; see the
>   note at the top of this file.* (Note: Recording's `D1–D5` design-gap prefix collides with
>   Config/RANKING's `D1–D3` and, since the 2026-08-02 renumber, with Dataset's `D1–D26`; different
>   modules, module prefix disambiguates.)
> - Read-only static audit — no server, no serial/camera open, no motor energized, no dataset mutated, no Hub
>   call. No code modified.

Audit date: July 14, 2026 · Original branch/commit: `andrew` at `257580f` · Re-verified: July 23, 2026 against `main` @ `c7d9f27`.

Pathway baseline: `internal_docs/complete functionalities/Recording Interface Pathways.md`, originally audited against `518ca56`.

Scope: fresh and resumed dataset recording, single-arm and bimanual setup, request construction, dataset creation, hardware connection, arm identity, calibration application, follower power, episode/reset control, status and log polling, Done/Quit/page-leave behavior, dataset handoff, rest-pose return, torque release, cross-feature ownership, and failure recovery.

## Severity key

- **P0:** immediate, broadly reachable destructive or physical-safety failure with no meaningful mitigation.
- **P1:** major feature blocker, data-loss path, orphaned hardware/session path, or hardware-safety boundary failure.
- **P2:** significant correctness, recovery, or operator-trust failure with narrower reach or an available workaround.
- **Design gap:** intended product behavior is missing, but the current implementation does not establish one unambiguous defective outcome.

## Structural change that dominates the re-verification (from 2026-07-23)

The redesign merge (`9e18b27`, on `main` via `c7d9f27`) **deleted `frontend/src/pages/Recording.tsx`**. The live session is now: `CollectPanel.tsx` (builds the request, owns the session dialog), `RecordingSessionDialog.tsx:101-919` (ported session UI, modal over the Launchpad), `CollectHandoff.tsx` (post-session banner, replaces the old `/upload` page), plus unchanged `lib/recordingExit.ts` and `hooks/useSessionExitGuard.ts`. The port was near-verbatim, so **most frontend defects survived the rewrite at new coordinates.**

## Confirmed bugs

### P0 — none in the original audit

No recording defect was classified as P0 in the original audit. **Re-verification escalates two coupled entries (R5+R6) to a P0-on-fix data-loss path (CURRENT P0-4).** It also raised a P0 auto-publish finding (R17), **descoped 2026-07-26** — public-by-default upload is intended product behavior, so recording has exactly one P0. See below.

### R1 · P1 — [was FIXED IN TREE] The LeRobot v0.6 migration makes every fresh and resumed recording fail before capture

**First identified:** 2026-07-14 · **Resolved:** 2026-07-14 — already fixed in the audited working tree; `main` never took the v0.6 migration (2026-07-23 verdict: N/A-superseded)
**Fix:** [PR #12](https://github.com/makermods-robotics/makermodslab/pull/12) (merged) · branch `port/lerobot-0.6.0`

> **Verdict on `main` (2026-07-23): NOT APPLICABLE — SUPERSEDED (but see R19).** `main` never took the v0.6
> migration: `pyproject.toml:22` still pins `lerobot@82dffde…` and `record.py:1202,1232` still pass ⚠ UNRESOLVED @ HEAD ce42158 — `vcodec=cfg.dataset.vcodec` no longer exists in `makermodslab/record.py`; the lerobot 0.6.0 port (`c6d97e7`) replaced it with `rgb_encoder=RGBEncoderConfig(...)` (`record.py:418`) and `rgb_encoder=cfg.dataset.rgb_encoder` (`:1216-1217`, `:1249-1250`). Line numbers left as written (they described the pre-port tree); see report.
> `vcodec=cfg.dataset.vcodec`, self-consistent with that pin. The original entry described the `andrew`
> working tree only. **However, the installed `.venv` contradicts `main`'s pin (lerobot 0.6.0 is installed) —
> so recording is dead-on-arrival in this checkout as installed; see R19.** Per `CURRENT 23-07-26`, `main`'s
> HEAD has since advanced past `c7d9f27` with the lerobot 0.6.0 port (`c6d97e7`) which resolves the pin
> mismatch on `main` proper; this verdict was computed at `c7d9f27` before that port.

**Verdict:** ALREADY-FIXED-IN-TREE (fixed in uncommitted tree, 2026-07-14) — the claim is factually inverted. Installed lerobot `0.6.0`'s `DatasetRecordConfig` exposes `rgb_encoder`/`depth_encoder`, NOT `vcodec` (`.venv/.../lerobot/configs/dataset.py:62-64`); `LeRobotDataset.create()`/`.resume()` accept `rgb_encoder`/`depth_encoder` (`lerobot_dataset.py:676-677,785-786`). The uncommitted `record.py:1210-1211,1244-1245` correctly reads `cfg.dataset.rgb_encoder`/`depth_encoder`. Committed HEAD `257580f` (pre-diff, `vcodec=cfg.dataset.vcodec`) WOULD fail on v0.6.0; the uncommitted migration is the fix.
**Priority:** resolved — would have been P1 (blocks all recording) if the tree still used `vcodec` (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. The current working tree changes the dependency to LeRobot `v0.6.0` (`pyproject.toml:15-20`; `uv.lock:1055-1059`).
2. LeRobot v0.6's `DatasetRecordConfig` exposes `vcodec`; it does not expose `rgb_encoder` or `depth_encoder`.
3. MakerMods Lab now reads `cfg.dataset.rgb_encoder` and `cfg.dataset.depth_encoder` while constructing both a resumed and a fresh `LeRobotDataset` (`makermodslab/record.py:1201-1216`, `makermodslab/record.py:1243-1261`).
4. Argument evaluation raises `AttributeError` before `LeRobotDataset.create()` or `.resume()` can run.
5. The asynchronous recording start has already returned success, so the user reaches the recording screen only to receive a terminal failure without recording an episode (`makermodslab/record.py:549-669`).

**Fix boundary**

Use the v0.6 configuration and dataset-construction contract consistently, then add fresh and resume regression tests that reach `save_episode()` under the pinned dependency.

### R2 · P1 — A logically rejected start is HTTP 200, and the frontend treats it as a successful recording

**First identified:** 2026-07-14
**Fix:** [PR #35](https://github.com/makermods-robotics/makermodslab/pull/35) (draft) · branch `fix/r2-recording-start-rejection-status-code`

> **Verdict on `main` (2026-07-23): STILL REAL — and the consequence on `main` is worse (escalates to dataset
> loss; feeds CURRENT P0-4).** `/start-recording` returns the handler dict verbatim (`server.py:838-841`), so
> `{success: false}` from `record.py:474-496` (already-active / teleop / inference / invalid name) or `:668-670`
> (synchronous setup failure) serializes as **HTTP 200**. `startRecordingSession()` branches on `response.ok`
> alone (`RecordingSessionDialog.tsx:343`), so a rejection sets `recordingSessionStarted = true` and toasts
> "Recording Started"; the comment at `:200-202` even claims "the second call returns 409" — no code path
> returns 409. Because `guardActive = recordingSessionStarted && !sessionEnded` (`:191`), the phantom dialog
> then **arms the page-leave guard with `beaconUrl = /stop-recording?discard=true`** (`:195`), so any exit from
> that phantom dialog issues a discard against **the real, unrelated live session**, whose fresh stamped
> dataset is then deleted wholesale (`record.py:632-640`). Reach: `SingleTabGuard.tsx:120-140` renders a
> "Use this tab" overlay and does *not* unmount the app, so a second tab lands on a working Launchpad while
> tab 1's session keeps running. Fix is two lines (non-2xx on logical rejection; require `data.success === true`).

**Verdict:** CONFIRMED — `/start-recording` returns `handle_start_recording()`'s dict verbatim (`server.py:838-841`), so `{success:false}` (from active-session/invalid-name/setup-fail at `record.py:475-497,669-672`) serializes as HTTP 200; the client checks only `response.ok` (`Recording.tsx:344`) and sets `recordingSessionStarted=true` + "Recording Started". The intended non-2xx rejection branch (`Recording.tsx:350-361`, whose comment even names a "409 already-active" that no code returns) is unreachable. ⚠ UNRESOLVED @ HEAD ce42158 — `Recording.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/studio/ + recording/, construct not re-located.
**Priority:** P1 — with no session id and process-global state, a second client hitting an already-active session begins polling/controlling that session (Done/Quit act on it); an invalid-name rejection strands the page on a dead "Connecting…" screen (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. `handle_start_recording()` returns dictionaries such as `{success: false, message: ...}` for an active session, invalid repository ID, or synchronous setup failure (`makermodslab/record.py:475-499`, `makermodslab/record.py:659-662`).
2. `/start-recording` returns that dictionary directly and never converts the rejection to a non-2xx response (`makermodslab/server.py:838-842`).
3. The recording page checks only `response.ok`, not `data.success` (`frontend/src/pages/Recording.tsx:335-360`). ⚠ UNRESOLVED @ HEAD ce42158 — `Recording.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/studio/ + recording/, construct not re-located.
4. Because HTTP 200 is `ok`, it sets `recordingSessionStarted=true` and displays "Recording Started" even when the backend explicitly rejected the request.

**Impact**

- If another recording is active, the new page begins polling and controlling the process-global existing session. Done, Quit, re-record, or page-leave discard can therefore act on a session the page did not start.
- If the request was rejected before any session became active, the page can remain stuck polling an idle or non-terminal state while claiming recording started.
- The intended "Error Starting Recording" branch is unreachable for normal backend logical rejections.

**Fix boundary**

Make logical start rejection non-2xx and also require `data.success === true` in the client. A durable fix should return an immutable session identifier and require it on later control requests.

### R3 · P1 — Setup failures after the follower connects can leave hardware and cameras open

**First identified:** 2026-07-14
**Fix:** [PR #37](https://github.com/makermods-robotics/makermodslab/pull/37) (draft) · branch `fix/record-connect-partial-teardown`

> **Verdict on `main` (2026-07-23): STILL REAL, verbatim.** `teleop.connect(calibrate=False)` failure re-raises
> at `record.py:1346` with the follower and its cameras already connected (`:1265`); the only unconditional
> rest/torque-disable/disconnect block is `record.py:1430` … `1614-1634`, entered only after the identity
> guard, calibration write, torque reset and rest-pose capture. A pre-loop raise skips it entirely, and the
> worker (`record.py:600-655`) keeps no device handle. The `ArmIdentityError` path does disconnect both
> (`record.py:1359-1363`) — that one case is covered; the others are not. (Corrected framing carried from
> original: the follower is left with torque OFF, so this is a serial-port + camera resource leak, not a
> torque-left-enabled hazard.)

**Verdict:** CONFIRMED-WITH-CORRECTIONS — real leak: `teleop.connect()` failing (`record.py:1342-1344`) re-raises without disconnecting the already-connected follower, and the only unconditional rest/torque-disable/disconnect `try/finally` starts later at the episode loop (`record.py:1428-1650`), so a pre-loop raise never cleans up; the worker (`record.py:602-659`) clears `recording_active` and retains no device handle. CORRECTION to the "partially configured motors" framing: the follower is left with torque OFF, not energized — lerobot's `so_follower.configure()` runs inside `torque_disabled()` and never re-enables (`.venv/.../so_follower.py:159-173`), so this is a serial-port + camera resource leak, NOT a torque-left-enabled hazard.
**Priority:** P1 — leaked follower port/cameras after a failed start; a retry hits busy ports/cameras and there is no recording control left to release the leaked device. No physical-safety hazard (torque off), so not P0 (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. MakerMods Lab constructs the dataset, connects the follower and its cameras, then connects the leader (`makermodslab/record.py:1213-1261`, `makermodslab/record.py:1245-1335`).
2. A final follower-connect failure can raise after a partial connection. More importantly, if the follower succeeds and the leader connection fails, the exception is re-raised without disconnecting the already-connected follower (`makermodslab/record.py:1336-1353`).
3. The only unconditional rest-return, torque-disable, and disconnect block begins later, around the episode loop (`makermodslab/record.py:1435-1436`, `makermodslab/record.py:1637-1657`). Exceptions before that `try/finally` never enter its cleanup.
4. The worker catches the exception, clears `recording_active`, and reports a failed session, but it does not retain a device reference that a later stop can clean up (`makermodslab/record.py:604-659`).

**Fix boundary**

Wrap the complete construction/connect/setup lifecycle in one ownership-aware cleanup block. Cleanup must attempt every connected device independently and surface an explicit torque/port warning when release is uncertain.

### R4 · P1 — Reaching the episode-duration limit discards the full take and can retry forever

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL — worse: the UI now names it "auto-advance".** `record.py:1486-1500`:
> a natural `record_loop` return without `_exit_early_triggered` logs "TIMEOUT — triggering re-record" and sets
> `web_events["rerecord_episode"] = True`; the take is cleared at `:1510` and the same episode retried without
> incrementing `saved_episodes` (`:1557-1558`). Only an operator action sets `_exit_early_triggered`
> (`record.py:737`). New: `RecordingSessionDialog.tsx:274-289` plays `playAutoAdvanceWarning()` three seconds
> before the limit and the primary button reads "End Episode" (`:640`) — the UI explicitly primes the operator
> that the timer will **auto-advance**, when reaching it actually destroys the take.

**Verdict:** CONFIRMED — when `record_loop` returns naturally at `episode_time_s` without `_exit_early_triggered`, the else-branch sets `rerecord_episode=True`, `clear_episode_buffer()`, resets, and retries the same episode without incrementing `saved_episodes` (`record.py:1492-1508`). Only an operator End Episode / SPACE / → (exit-early) marks the take for saving (`record.py:1466-1473`; `Recording.tsx:670-675,762-774`). The setup modal labels the value "Episode duration (seconds)" and the page renders it as a countdown/progress limit (`RecordingModal.tsx:312`; `Recording.tsx:623-631,738-754`), implying reaching it completes the take. This is a deliberate MakerMods Lab addition (upstream lerobot saves on timeout), not an upstream behavior. ⚠ UNRESOLVED @ HEAD ce42158 — `Recording.tsx`, `RecordingModal.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/studio/ + recording/; searched frontend/src/components/recording/ + studio/, construct not re-located.
**Priority:** P1 — an operator who lets the labeled "duration" elapse silently loses a completed demonstration and the episode re-records instead of advancing. Clear workaround (press End Episode before the timer), but the label/countdown actively mislead toward the lossy path (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. The setup modal labels the value "Episode duration (seconds)" and the recording page renders a countdown/progress limit (`frontend/src/components/landing/RecordingModal.tsx:306-339`, `frontend/src/pages/Recording.tsx:623-631`, `frontend/src/pages/Recording.tsx:738-751`). ⚠ UNRESOLVED @ HEAD ce42158 — `Recording.tsx`, `RecordingModal.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/studio/ + recording/; searched frontend/src/components/recording/ + studio/, construct not re-located.
2. When `record_loop()` returns naturally at that limit, `_exit_early_triggered` is false.
3. MakerMods Lab interprets that natural completion as a re-record request, sets `rerecord_episode=true`, clears the episode buffer, performs reset, and retries the same episode (`makermodslab/record.py:1490-1563`).
4. Only the operator's **End Episode** action marks the take for saving. A hands-off timed session never advances `saved_episodes` and can repeat indefinitely.

**Fix boundary**

Either save a naturally completed timed take, matching the normal meaning of an episode duration, or rename and explain the setting as a mandatory retake deadline. Tests must distinguish timeout-save, explicit End Episode, explicit re-record, and mid-episode stop.

### R5 · P1 — Terminal status erases the number of saved episodes and hides recoverable results

**First identified:** 2026-07-14
**Fix:** [PR #32](https://github.com/makermods-robotics/makermodslab/pull/32) (draft) · branch `fix/r5-terminal-status-saved-episodes`

> **Verdict on `main` (2026-07-23): STILL REAL — the andrew fix is ABSENT from `main`.** This is half of
> **CURRENT P0-4** (the R5/R6 coupling). (1) The worker's `finally` zeroes the live counter `saved_episodes = 0`
> (`record.py:654`), the only place the count lives. (2) `handle_recording_status()` adds `saved_episodes` only
> inside `if recording_active and recording_config:` (`record.py:833-836`); the terminal block above (`:802-813`)
> adds `discarded_empty`/`outcome`/`error`/`hint` and no count. (3) There is **no** `last_session_saved_episodes`
> global on `main` (grep returns nothing; `test_terminal_status_retains_saved_episodes` is absent). (4) The
> worker sets `recording_active = False` at `:650` *before* zeroing at `:654`, so no terminal poll can ever
> observe the true count. Consumers: clean finish hands off `saved_episodes || 0` → always 0, so
> `CollectHandoff.tsx:90-96` renders "0 episodes" for every successful session; failed/warning finish computes
> `keptSomething = savedEpisodes > 0 && !discarded_empty` (`:787-788`) which is **always false**, so
> "Keep episodes & continue" and "Discard & exit" never render — only "Back to home". Keep P1: dataset is on
> disk and reachable from the library, so this is blocked recovery + a false count, not loss. **Fixing R5
> without R6 unmasks whole-dataset deletion — the two MUST ship together.**

**Status (original):** In progress — 2026-07-14: the worker now snapshots `saved_episodes` into a new `last_session_saved_episodes` global before its finally zeroes the live counter, and `handle_recording_status()` echoes it in every ended payload (`makermodslab/record.py`); the frontend shows the kept count in the warning banner (`Recording.tsx`). Validated by unit test `test_terminal_status_retains_saved_episodes`; live hardware validation pending. **← This fix is NOT on `main`.**

**Verdict:** CONFIRMED — the worker resets `saved_episodes=0` in its finally (`record.py:654`), and `handle_recording_status()` adds `saved_episodes` to the payload only while `recording_active` is true (`record.py:835-838`), so every terminal payload omits it. Clean completion hands off `status.saved_episodes || 0` (→0) to `/upload` (`Recording.tsx:309`); the failed/warning end-state gates "Continue to upload" and "Discard & exit" behind `keptSomething = savedEpisodes > 0` (`Recording.tsx:799-800,840,851`), so a warning/failed session that DID keep episodes shows only "Back to home". ⚠ UNRESOLVED @ HEAD ce42158 — `Recording.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/studio/ + recording/, construct not re-located.
**Priority:** P1 — a `ran_with_warning` session tells the operator "your episodes are safe" yet offers no button to reach them; the intended recovery flow is blocked. The dataset is still reachable via the normal dataset browser (workaround), so not P0 (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. The worker resets the process-global `saved_episodes` to zero before marking the session fully inactive (`makermodslab/record.py:654-659`).
2. `handle_recording_status()` includes `saved_episodes` only while `recording_active` is true (`makermodslab/record.py:823-845`). Terminal status therefore omits it entirely.
3. Clean completion constructs the post-recording handoff with `status.saved_episodes || 0` (`frontend/src/pages/Recording.tsx:288-317`). ⚠ UNRESOLVED @ HEAD ce42158 — `Recording.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/studio/ + recording/, construct not re-located.
4. Failed/warning completion shows recovery actions only when the missing terminal value is greater than zero (`frontend/src/pages/Recording.tsx:797-860`). ⚠ UNRESOLVED @ HEAD ce42158 — `Recording.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/studio/ + recording/, construct not re-located.

**Fix boundary**

Persist a terminal saved count separately from live counters, include it in every ended payload, and test clean, warning, failure, fresh-discard, resume, and zero-episode outcomes.

### R6 · P1 (→ P0 on fix) — "Discard & exit" after a resumed session deletes the entire pre-existing dataset

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL, STILL LATENT — the andrew fix is ABSENT; R5/R6 coupling STILL
> LIVE.** This is the other half of **CURRENT P0-4**. (1) `discardAndExit()` (`RecordingSessionDialog.tsx:512-527`)
> posts `{dataset_repo_id: repoId}` to `/delete-dataset` with **no reference to `recordingConfig.resume`** —
> the component *has* `resume` in scope (`:160`, used for confirm copy at `:194,896-898`) but not here. (2)
> `handle_delete_dataset()` (`record.py:859-897`) `shutil.rmtree`s the whole directory (`:887`), resume-unaware
> by design. (3) The backend's own worker-side discard **is** resume-safe and tested (`_discard_session_dataset`
> early-returns on resume, `:989-991`; `_discard_empty_dataset` `:923-925`) — the frontend button bypasses both.
> (4) The button renders only when `keptSomething` is true (`:855`), which R5 makes permanently false —
> **fixing R5 unmasks R6.** (5) **New gate on `main`:** `CollectPanel.tsx:247` hardcodes `resume: false`, so a
> resume session can today only be created by a direct `POST /start-recording` (the UI resume entry point was
> dropped — `CONFIRMED 22-07-26` #5). **P0 if R5 is fixed without R6; the two must be fixed in the same change.**
> A resume-safe `discardAndExit` is ~3 lines and should land before/with the `last_session_saved_episodes`
> snapshot. Additional phantom-dialog aggravator: R2 (HTTP-200 rejection) can arm the leave guard against the
> real session and delete its fresh dataset wholesale.

**Status (original):** In progress — 2026-07-14: the ended-session `discardAndExit()` is now resume-aware for a RESUME session (deletes nothing, keeps every episode; button reads "Exit without uploading"); whole-directory `/delete-dataset` deletion kept only for FRESH sessions. Backend `handle_delete_dataset` intentionally left non-resume-aware. Validated by `test_worker_quit_keeps_resumed_dataset` plus the existing `_discard_session_dataset` resume guard; live hardware validation pending. **← This fix is NOT on `main`.**

**Verdict:** CONFIRMED (latent; currently masked by the terminal-count bug R5) — `discardAndExit()` sends the repo_id to `/delete-dataset` ignoring `recordingConfig.resume` (`Recording.tsx:520-535`), and `handle_delete_dataset()` `shutil.rmtree`s the whole resolved directory with no resume-awareness (`record.py:861-897`). The backend's own worker discard path IS resume-safe (`_discard_session_dataset` no-ops on resume, `record.py:977-979`), but the frontend button bypasses it. The button is only rendered when `keptSomething > 0`, which is always false today because terminal `saved_episodes` is omitted (R5), so it is currently unreachable. ⚠ UNRESOLVED @ HEAD ce42158 — `Recording.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/studio/ + recording/, construct not re-located.
**Priority:** P0 — once R5 is fixed, a resume-session "Discard & exit" deletes the entire pre-existing dataset: unrecoverable loss of expensive demo data. DEPENDENCY: must be fixed together with / before the terminal-`saved_episodes` fix, which is what unmasks it (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. A resume session appends into an existing repository and deliberately does not delete it when active-session Quit is requested (`makermodslab/record.py:636-652`, `makermodslab/record.py:959-997`).
2. The ended failure/warning UI offers **Discard & exit** whenever it believes something was kept.
3. That callback ignores `recordingConfig.resume` and sends the repository ID to the general `/delete-dataset` endpoint (`frontend/src/pages/Recording.tsx:517-535`). ⚠ UNRESOLVED @ HEAD ce42158 — `Recording.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/studio/ + recording/, construct not re-located.
4. The backend deletes the entire resolved repository directory, not merely episodes appended by the last session (`makermodslab/record.py:864-898`).

**Fix boundary**

Never present whole-repository deletion as session rollback. For resume, either retain all committed episodes, implement a transactional append with a session boundary that can be rolled back safely, or require an explicit whole-dataset deletion confirmation naming the pre-existing episode count.

### R7 · P1 — Done and Quit disarm page-leave safety before the backend acknowledges a stop

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL.** `doStopRecording()` calls `markHandled()` before the fetch
> and never checks `response.ok`/`{success}` (`RecordingSessionDialog.tsx:447-470`); `confirmQuit` navigates
> home unconditionally (`:488-494`). `/stop-recording` returns HTTP 200 even when no session is active
> (`server.py:844-853`; `record.py:701-702`). Once latched, the exit guard's unmount fallback is suppressed
> (`useSessionExitGuard.ts:65-70,159-166`). One change (map logical rejections to 409 + require `data.success`)
> fixes R2, R7, and R20 together.

**Verdict:** CONFIRMED — `doStopRecording()` calls `markHandled()` before the fetch and never checks `response.ok`/`{success}` (`Recording.tsx:448-471`); `confirmQuit` navigates home unconditionally (`Recording.tsx:489-495`). `/stop-recording` returns HTTP 200 even when no session is active (`server.py:844-853`; `record.py:703-704`). Once latched, the exit guard's unmount fallback is suppressed (`useSessionExitGuard.ts:68-70,159-166`). ⚠ UNRESOLVED @ HEAD ce42158 — `Recording.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/studio/ + recording/, construct not re-located.
**Priority:** P1 — a transient stop failure leaves recording active after the UI has navigated away with its only fallback disarmed; hardware keeps running with no visible control, and a fresh session may report a discard that never happened (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. `doStopRecording()` calls `markHandled()` before sending the stop request (`frontend/src/pages/Recording.tsx:445-456`). ⚠ UNRESOLVED @ HEAD ce42158 — `Recording.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/studio/ + recording/, construct not re-located.
2. It does not check `response.ok` or the returned `{success}` value; any HTTP response produces a "Finishing" or "Quitting" toast (`frontend/src/pages/Recording.tsx:455-468`). ⚠ UNRESOLVED @ HEAD ce42158 — `Recording.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/studio/ + recording/, construct not re-located.
3. The stop endpoint also returns logical failure as HTTP 200 when no session is active (`makermodslab/server.py:844-853`, `makermodslab/record.py:684-702`).
4. **Quit** navigates home unconditionally after the attempted request, including network failure (`frontend/src/pages/Recording.tsx:489-495`). ⚠ UNRESOLVED @ HEAD ce42158 — `Recording.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/studio/ + recording/, construct not re-located.
5. Because the exit guard is already latched handled, unmount cannot issue its best-effort fallback discard (`frontend/src/hooks/useSessionExitGuard.ts:58-76`, `frontend/src/hooks/useSessionExitGuard.ts:157-168`).

**Fix boundary**

Latch the exit guard only after an acknowledged stop for the correct session. Quit must stay on the page and retain/re-arm safety controls when stop fails.

### R8 · P1 — Resume recording can mutate a dataset while upload, merge, or local training uses it

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL, reach reduced to API-only.** `_dataset_in_use()` exists and
> covers recording/upload/merge/local training (`datasets.py:696-748`), called by delete (`record.py:877-881`)
> and upload (`record.py:1076-1080`), but never by `handle_start_recording` (`record.py:472-496`). Because the
> UI can no longer request a resume (`CollectPanel.tsx:247`), a fresh session always writes a new timestamped
> directory (`record.py:531-532`), so the reverse-ordering race is API-only today. Keep P2.

**Verdict:** CONFIRMED — `handle_start_recording()` checks only recording/teleop/inference flags + name syntax (`record.py:461-485`) and never calls `_dataset_in_use()`, which exists and guards the reverse direction (delete/rename/upload) for recording/upload/merge/local-training (`datasets.py:697-745`). A resume session writes directly into the un-timestamped existing repo_id (`record.py:532-533`), so an upload/merge/train already reading it is not refused.
**Priority:** P2 — reachable only if the user starts an upload/merge/local-training on a dataset and THEN starts a resume recording into that same dataset; narrow reach in a single-user local app. Consequence when hit is real, so P1 if that ordering proves common (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. MakerMods Lab already has `_dataset_in_use(repo_id)`, which tracks active recording, upload, merge output, and local training (`makermodslab/datasets.py:696-748`). Delete, rename, and upload paths use this guard.
2. `handle_start_recording()` does not call it. It checks only recording, teleoperation, inference, and name syntax (`makermodslab/record.py:457-523`).
3. A resumed session writes directly into the existing repository ID without timestamping (`makermodslab/record.py:517-535`).
4. Therefore an upload, merge output, or local training read can begin first, after which resume recording can still append and rewrite metadata under that active operation.

**Fix boundary**

Introduce an atomic per-dataset read/write lease shared by recording, upload, merge, delete/rename, and training. A one-time boolean check is not sufficient because two starts can race between checking and claiming ownership.

### R9 · P1 — Robot hardware ownership is incomplete and non-atomic across features

**First identified:** 2026-07-14
**Fix:** [PR #7](https://github.com/makermods-robotics/makermodslab/pull/7) (closed) · branch `fix/t4-cross-feature-mutex` · [PR #22](https://github.com/makermods-robotics/makermodslab/pull/22) (open) · branch `fix/t4-cross-feature-mutex`

> **Verdict on `main` (2026-07-23): STILL REAL.** Each feature holds its OWN module-local `_state_lock`
> (`record.py:216`, `teleoperate.py:102`, `rollout.py:114`) while reading the OTHER modules' booleans
> (`record.py:473-485`), so two concurrent starts can each observe the other inactive before claiming their own
> flag (cross-module TOCTOU). Recording checks none of calibration/auto-calibration/wiggle (`record.py:473-497`);
> wiggle opens the bus with no lease. Same family as Teleop T4 / Config entry 11 / Inference I4 — one
> server-owned session arbiter closes them together.

**Verdict:** CONFIRMED — each feature holds its OWN module-local `_state_lock` (`record.py:221`, `teleoperate.py:102`, `rollout.py:111`) while reading the OTHER modules' booleans (e.g. `record.py:473-485` reads `teleoperate.teleoperation_active`/`rollout.inference_active` under record's lock), so two concurrent starts can each observe the other inactive before claiming their own flag (cross-module TOCTOU). Recording also checks none of calibration/auto-calibration/wiggle (`record.py:461-485`); wiggle opens the bus with no lease (`wiggle.py`).
**Priority:** P2 — requires genuinely concurrent starts in a single-user local app (SingleTabGuard limits tabs), and OS exclusive serial-open usually makes the loser fail with a port-busy error rather than issue competing motor commands (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. Recording, teleoperation, and inference each use their own module-local `_state_lock` while checking the other modules' booleans (`makermodslab/record.py:214-216`, `makermodslab/record.py:445-494`; `makermodslab/teleoperate.py:101-103`; `makermodslab/rollout.py:109`). Two simultaneous starts can both observe the other feature as inactive before either claims its own flag.
2. Recording checks teleoperation and inference but not manual calibration, auto-calibration, batch auto-calibration, or gripper wiggle (`makermodslab/record.py:475-488`).
3. Manual and automatic calibration similarly claim only their own manager state (`makermodslab/calibrate.py:232-278`, `makermodslab/auto_calibrate.py:199-252`). Wiggle opens and commands the requested serial bus without a feature lease (`makermodslab/wiggle.py:55-85`).
4. The browser's single-tab takeover changes UI election only; it does not stop or transfer ownership of the old tab's live hardware session (`frontend/src/components/SingleTabGuard.tsx:64-89`, `frontend/src/components/SingleTabGuard.tsx:109-139`).

**Fix boundary**

Use one server-owned hardware-session coordinator with atomic acquire/release, exact port ownership, release-in-progress state, and a session token required by every control endpoint.

### R10 · P1 — Failure to apply the saved motor-power limit is hidden while recording proceeds

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): NOT APPLICABLE — SUPERSEDED as framed; a narrower defect REMAINS (P2).**
> Commit `714d29e` ("replace motor_power percent with raw max_torque_limit") changed the product semantics: per
> `utils/config.py:288-294`, the robot's torque slider now sets **auto-calibration drive torque only**, and
> "regular sessions (teleop/record/skill runs) run at stock LeRobot torque". So there is no longer an
> operator-selected recording cap that can be silently unenforced. What survives is narrower: `reset_torque_limit()`
> and `clear_goal_velocity()` return warning lists (`motor_power.py:138,181`), and **recording alone discards
> them** (`record.py:1402,1406`) while teleop folds them into its surfaced warnings (`teleoperate.py:533,537,803-804`)
> and rollout collects them (`rollout.py:488`). Recording already has an `identity_warnings` global that reaches
> the status payload (`record.py:212,829-830`), so this is a one-line inconsistency. **P2.**

**Verdict:** CONFIRMED — `apply_motor_power()` and `clear_goal_velocity()` catch per-motor failures and RETURN warning lists (`motor_power.py:120-133,136-164,167-198`), but recording calls both and discards the returned lists (`record.py:1400,1404`); the warning never reaches a status payload (unlike teleoperation, which surfaces it).
**Priority:** P2 — triggers only on a per-motor register write failure (rare); the motor then keeps its previous limit / servo default. Bump to P1 if `motor_power` is treated as a hard safety guarantee for delicate setups (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. The selected robot's saved `motor_power` is sent in the recording request (`frontend/src/pages/Landing.tsx:212-239`). ⚠ UNRESOLVED @ HEAD ce42158 — `Landing.tsx` deleted in the redesign (9e18b27); searched frontend/src/pages/Launchpad.tsx + components/launchpad/, construct not re-located.
2. `apply_motor_power()` catches per-motor failures and returns explicit warnings that affected motors retain their previous limit, or full power after power-up (`makermodslab/motor_power.py:136-164`).
3. Recording discards that returned warning and continues into the control loop (`makermodslab/record.py:1403-1411`). It also discards warnings from clearing stale goal-velocity limits.
4. Unlike teleoperation, the warning is never added to an operator-facing status payload.

**Fix boundary**

Treat the selected safety limit as an enforced precondition or require explicit operator confirmation of degraded operation before motion begins. Surface per-arm and per-motor failures before starting the episode loop. *(On `main` the semantics changed — see verdict; the surviving one-line inconsistency is folding the `reset_torque_limit`/`clear_goal_velocity` warnings into the status payload as teleop does.)*

### R11 · P1 — Rest-pose capture and return failures are logged but reported as a clean recording

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL.** `verify`/return path unchanged: `teleoperate.py:234-291`
> returns `None`; `record.py:1642` ignores it; `record.py:599` sets `outcome="ok"` when no exception was raised.
> `capture_rest_pose()` returns `{}` on comm failure (`rest_pose.py:60-73`). Done/Quit copy still promises the
> arm returns to its start pose then goes limp (`recordingExit.ts:21,30-32`).

**Verdict:** CONFIRMED — `capture_rest_pose()` returns `{}` on comm failure (`rest_pose.py:60-73`); `return_to_rest_pose()` returns `(arrived, reason)` with failure reasons no-pose/comm-error/settled/stalled/ceiling/cut-short (`rest_pose.py:76-129`), but `_return_followers_to_rest()` and its per-arm helper only LOG the reason and return `None` (`teleoperate.py:237-291`). The recording finally ignores the outcome, disables torque, and the worker sets `last_session_outcome="ok"` when no exception was raised (`record.py:1630-1650,601`). Done/Quit copy promises the arm returns to its start pose then goes limp (`recordingExit.ts:21,30-32`).
**Priority:** P2 — torque IS released either way (safe end state), so no torque-left-enabled hazard; the defect is that the arm may release where it stalled while the UI reports a clean return. P1 if the un-parked release is judged a real flop/collision risk on this rig (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. Rest-pose capture returns an empty pose on communication failure and does not fail or warn the session status (`makermodslab/rest_pose.py:60-73`, consumed by `makermodslab/record.py:1395-1405`).
2. Return can report `no-pose`, `comm-error`, `settled`, `stalled`, `ceiling`, or `cut-short` as unsuccessful outcomes (`makermodslab/rest_pose.py:76-129`, `makermodslab/rest_pose.py:149-192`).
3. `_return_followers_to_rest()` logs those outcomes but returns no aggregate result (`makermodslab/teleoperate.py:237-291`).
4. Recording ignores the return outcome, disables torque, and can finish with `outcome="ok"` (`makermodslab/record.py:1637-1657`, `makermodslab/record.py:595-626`).
5. Done/Quit copy promises that the arm returns to its starting position and then goes limp (`frontend/src/lib/recordingExit.ts:17-38`).

**Fix boundary**

Aggregate per-arm capture/return results into terminal status. A failed return should be a visible cleanup warning with the final reason and a clear physical-safety instruction.

### R12 · P1 — An arm-identity read failure silently disables the safety check before calibration writes and motion

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL.** `verify_arm()` returns `(None, None)` on a
> `Present_Position`/`Homing_Offset` read failure (`arm_identity.py:283-288`), `verify_devices()` reports no
> refusal (`:391-413`), and recording proceeds to `_write_calibration` (`record.py:1395-1396`) and motion.
> Documented fail-open (`arm_identity.py:270-272`). Per the re-verification brief, the separate arm-identity
> warn-and-proceed *severity* question (rows 3/4) is EXCLUDED — a fix is committed on an unmerged branch. This
> entry is the *read-failure* branch, a different code path; verdict recorded without re-litigating severity.

**Verdict:** CONFIRMED-WITH-CORRECTIONS — on a `Present_Position`/`Homing_Offset` read failure `verify_arm()` returns `(None, None)` (no refusal, no warning) (`arm_identity.py:279-288`), so `verify_devices()` treats the arm as verified (`arm_identity.py:391-413`) and recording proceeds to write calibration to EEPROM and start motion (`record.py:1333-1376`). CORRECTION: this is a DELIBERATE, documented fail-open (`arm_identity.py:270-272`), and the practical exposure is narrow — it needs the identity read to fail yet the subsequent EEPROM write to succeed AND the arm to actually be swapped.
**Priority:** P2 — edge case with a coherent design rationale; the fail-open removes the swap guard only in a narrow intermittent-comm window. Worth a tri-state (verified/mismatch/unverifiable) hardening, but not broadly reachable (verified 2026-07-14 against the current working tree).

**Broken pathway**

1. If reading `Present_Position` or `Homing_Offset` fails, `verify_arm()` logs the error and returns no refusal and no warning (`makermodslab/arm_identity.py:252-288`).
2. `verify_devices()` therefore lets setup continue as though no identity problem was found (`makermodslab/arm_identity.py:391-413`).
3. Recording then writes the selected calibration to the connected devices, applies follower settings, and starts leader-to-follower control (`makermodslab/record.py:1337-1405`, `makermodslab/record.py:1435-1464`).

**Fix boundary**

Identity must be tri-state: verified, mismatch, or unverifiable. Unverifiable hardware must be flagged before calibration writes and motion, with any override explicit, narrow, and recorded in session status.

## P2 correctness and recovery bugs (R13–R15) · appended findings, mixed severity (R16–R30)

### R13 · P2 — Ended-session deletion ignores the server result and always tells navigation to continue

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL.** `discardAndExit()` (`RecordingSessionDialog.tsx:512-527`)
> never inspects `response.ok`/`{success}` and calls `navigate("/")` regardless; `/delete-dataset` legitimately
> refuses on busy/missing (`record.py:879-884`). Currently also gated behind `keptSomething` (same R5 masking).

**Verdict:** CONFIRMED — `discardAndExit()` awaits the fetch inside try/catch, never inspects `response.ok` or `{success}`, and calls `navigate("/")` regardless (`Recording.tsx:520-535`); `/delete-dataset` legitimately refuses on busy/missing (`record.py:868-872`). (Currently also gated behind `keptSomething`, same masking as R6.) ⚠ UNRESOLVED @ HEAD ce42158 — `Recording.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/studio/ + recording/, construct not re-located.
**Priority:** P2 — the user cannot tell whether the discard succeeded; compounds the resume whole-dataset-deletion entry (too destructive when accepted, falsely reassuring when refused). Narrow reach today (verified 2026-07-14 against the current working tree).

`discardAndExit()` catches transport errors, ignores non-2xx and `{success: false}` responses, shows no failure, and navigates home regardless (`frontend/src/pages/Recording.tsx:517-535`). The server can legitimately refuse deletion because the dataset is busy or missing (`makermodslab/record.py:867-879`). The user therefore cannot tell whether discard succeeded. This compounds the resumed-dataset deletion bug: the same control is simultaneously too destructive when accepted and falsely reassuring when refused. ⚠ UNRESOLVED @ HEAD ce42158 — `Recording.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/studio/ + recording/, construct not re-located.

### R14 · P2 — Re-record is accepted by the backend outside the recording phase

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL — and materially under-rated: it DOES destroy data (see R16).**
> `handle_rerecord_episode()` validates only `recording_active and recording_events is not None`
> (`record.py:748-759`), while the status payload advertises the control only during
> `current_phase == "recording"` (`record.py:789-790`). A stale poll or direct POST during reset/stopping sets
> shared `rerecord_episode`+`exit_early`, which can carry into the next loop and silently discard the NEXT full
> episode — see new finding R16.

**Verdict:** CONFIRMED — `handle_rerecord_episode()` validates only `recording_active and recording_events is not None` (`record.py:750-761`), while the status payload advertises the control only during `current_phase == "recording"` (`record.py:779-780`). A stale poll or direct POST during reset/stopping sets shared `rerecord_episode`+`exit_early`, which can carry into the next loop.
**Priority:** P2 — requires a direct API call or a poll racing a phase change; affects flag state, not data on disk. Backend phase validation should be authoritative (verified 2026-07-14 against the current working tree). **← Re-verification raises the real impact: see R16 (data loss).**

The status payload advertises re-record only during `current_phase == "recording"`, but `handle_rerecord_episode()` validates only that the overall session is active (`makermodslab/record.py:738-753`, `makermodslab/record.py:778-784`). A stale poll or direct request during reset/stopping sets shared `rerecord_episode` and `exit_early` flags that can carry into the next loop. Backend phase validation should be authoritative, and episode-control events should be scoped to an episode/phase generation.

### R15 · P2 — The backend accepts recording parameters outside the interface's validated contract

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL.** `RecordingRequest` uses unconstrained `int` for
> `num_episodes`/`episode_time_s`/`reset_time_s`/`fps` and a free-form `mode: str = "single"` with no
> validators (`record.py:264-295` — no `Field(...)`/validator anywhere in the module); `create_record_config`
> routes any non-`"bimanual"` mode into the single-arm path (`record.py:376-382`).

**Verdict:** CONFIRMED — `RecordingRequest` uses unconstrained `int` for `num_episodes`/`episode_time_s`/`reset_time_s`/`fps` and a free-form `mode: str = "single"` with no validators (`record.py:269-303`); `create_record_config` routes any non-`"bimanual"` mode into the single-arm path (`record.py:376-382`). The frontend bounds these, but a direct API call can send zero/negative/arbitrary values.
**Priority:** P2 — reachable only via direct API (not the shipped UI); defense-in-depth. Add schema constraints + 422s if the endpoint is treated as product-facing (verified 2026-07-14 against the current working tree).

`RecordingRequest` uses unconstrained integers and a free-form `mode` string (`makermodslab/record.py:264-300`). The frontend bounds episodes and positive durations, but direct API calls can send zero/negative episodes or timing, unsupported FPS, arbitrary mode strings that silently fall into the single-arm path, and out-of-range values later clamped or passed downstream. If the endpoint is product-facing rather than an expert low-level API, these should be schema constraints with clear 422 responses.

### R16 · P1 — A re-record accepted during the reset phase silently discards the NEXT fully recorded episode *(formerly N1)*

**First identified:** 2026-07-23

`handle_rerecord_episode()` sets `recording_events["rerecord_episode"] = True` and `exit_early = True` whenever the session is active (`record.py:748-759`) — no phase check (R14's mechanism). The loop **only ever clears `rerecord_episode` inside its own re-record branch** (`record.py:1508`), and lerobot's `record_loop` clears `exit_early` but never `rerecord_episode` (`lerobot_record.py:285-287`; the CLI's outer loop clears it at `:508-511`, which MakerMods Lab's reimplementation does not mirror in the reset paths).

Trace on `main`:

1. Re-record arrives during a **reset** phase → `exit_early` set → the reset `record_loop` (`record.py:1596-1609`) returns early. The post-reset checks at `:1614-1623` inspect `exit_early` and `stop_recording` — **never `rerecord_episode`**.
2. The while-loop continues: `current_phase = "recording"`, `_exit_early_triggered` reset (`:1444`), and a **full episode is recorded**.
3. After that episode, `if web_events["rerecord_episode"]:` at `:1503` is still True → `dataset.clear_episode_buffer()` at `:1510` → **the complete take is thrown away** before reaching `save_episode()` at `:1563`.

Reachability without a direct API call: the dialog polls at 1 Hz (`RecordingSessionDialog.tsx:322`) and the Re-record button's `disabled` reads only the stale `available_controls.rerecord_episode` (`:766`) — **not** gated by `optimisticPhase`, which the same component sets to `"resetting"` the instant the user presses End Episode (`:384`). So for up to ~1 s after ending an episode — precisely when an operator realises the take was bad — the button and the Backspace shortcut (`:567-571`) are live and land in the reset phase.

Consequences: one completed demonstration destroyed; the toast said "Episode N will be re-recorded" (`:426`) while episode N is already saved and kept; the counter does not advance and no error shows. Fix: clear `rerecord_episode` at the top of each recording phase (or after every reset), reject the request server-side outside `current_phase == "recording"`, and disable the button on the optimistic phase. *(Caveat: R16's proof that `record_loop` never clears `rerecord_episode` was read from the installed lerobot 0.6.0, not `main`'s pin `82dffde`; MakerMods Lab's own side is unambiguous, but confirm against the pinned source. Per CURRENT, `main`'s HEAD has since pulled the 0.6.0 port, so the pinned source now matches — this resolves in the direction that keeps R16 live.)*

### R17 · ~~P0~~ **DESCOPED — not a bug** (public auto-push is intended) *(formerly N2)*

**First identified:** 2026-07-23 · **Resolved:** 2026-07-26 — descoped (product decision)

> Was **CURRENT P0-2**, found independently as **Dataset D3**. **Both IDs are retired, not reused.**
> **Do not re-file this chain from a future audit.**

The entry claimed that every recorded dataset is auto-published to a **public** Hub repo by a default-on toggle
hidden in a collapsed "Advanced parameters" section — `StudioContext.tsx:42` (`pushToHub: true`) →
`RecordingForm.tsx:204-259` (collapsed, copy never says *public*) → `CollectHandoff.tsx:102-110,191-202`
(`start([], false)` on mount, no confirmation) → `useDatasetUpload.ts:111-119` → `replayApi.ts:279-292`
`{private: false}` → `record.py:1116` → public repo.

**The chain is accurate and still present on `main`. The verdict was wrong.** Public-by-default dataset upload
is the product's intent: recorded datasets are meant to be discoverable, published without a per-session
confirmation. The rubric line this was ranked under — *silent publication of private data* — does not apply to
a publication the product means to perform. Nothing here is to be "fixed".

Residue, deliberately **not** filed: the "Push to Hugging Face Hub" helper copy still never states that the
repo will be public. Under the settled policy that is a disclosure nicety, not a defect. See CURRENT →
"Descoped by product decision" for the knock-on effects on Dataset D1 and Training MT27 (formerly NEW-3).

### R18 · P1 — The post-session release grace is invisible, and its controls report actions that never happen *(formerly N3)*

**First identified:** 2026-07-23

The backend deliberately publishes `releasing` "so the UI isn't lying about the arm's state" (`record.py:218-225,785`) and a matching message "Returning the arm to its rest position…" (`:795`). **`RecordingSessionDialog.tsx` reads neither.** The `BackendStatus` interface (`:65-92`) has no `releasing`/`message` field, and `getStatusText()` (`:614-629`) has no release branch. Teleop does consume it (`TeleopDialog.tsx:67,106`), so this is recording-specific.

During the grace the follower is **energized and driving** back to its session-start pose (`record.py:1634-1642`). Meanwhile: `current_phase` is already `"completed"` (set at `:1626`, before the `finally`) and `recording_active` is still true, so `sessionEnded` (`:157-160`) is false → the phase pill reads **"SESSION COMPLETE"** (`:628`); `available_controls.stop_recording` is `recording_active` (`:787`) → **Done and Quit remain enabled** (`:672-689`); pressing **Quit** calls `handle_stop_recording(discard=True)`, which hits the `if releasing:` early return at `:680-687` and **never sets `discard_requested`** — yet the UI toasts "Quitting — Discarding the recording…" (`:470`) and closes (`:505`). Nothing is discarded; every episode is kept. The inverse is equally wrong: an operator who genuinely wants to discard cannot.

Impact: an operator is told the session is complete while an arm is still moving under power (workspace-reach / cable-tug hazard), and told a discard happened when it did not. Fix: add `releasing` to `BackendStatus`, render the backend's message, hide/relabel Done+Quit to a single "Release now" during the grace.

### R19 · P1 (environment) — The installed environment contradicts `main`'s lerobot pin; recording fails immediately in this checkout as installed *(formerly N4)*

**First identified:** 2026-07-23
**Fix:** [PR #12](https://github.com/makermods-robotics/makermodslab/pull/12) (merged) · branch `port/lerobot-0.6.0`

- `main` uses the pre-v0.6 dataset API: `vcodec=cfg.dataset.vcodec` (`record.py:1202,1232`), consistent with `pyproject.toml:22` (`lerobot@82dffde…`) and `uv.lock:1041,1057`. ⚠ UNRESOLVED @ HEAD ce42158 — `vcodec=cfg.dataset.vcodec` no longer exists in `makermodslab/record.py`; the lerobot 0.6.0 port (`c6d97e7`) replaced it with `rgb_encoder=RGBEncoderConfig(...)` (`record.py:418`) and `rgb_encoder=cfg.dataset.rgb_encoder` (`:1216-1217`, `:1249-1250`). Line numbers left as written (they described the pre-port tree); see report.
- The repo `.venv` holds a **different** build: `.venv/.../lerobot-0.6.0.dist-info/direct_url.json` → `{"requested_revision": "v0.6.0", "commit_id": "30da8e68…"}`.
- Verified by introspection: `hasattr(cfg, "vcodec")` is **False**, `hasattr(cfg, "rgb_encoder")` is **True**; `LeRobotDataset.create/.resume` accept `rgb_encoder`/`depth_encoder`, not `vcodec`.

Consequence: running `makermodslab` from this checkout against the venv as it stands raises `AttributeError: 'DatasetRecordConfig' object has no attribute 'vcodec'` inside `record_with_web_events` — after the start response already reported success (`record.py:660-665`) — for both fresh and resumed sessions. This is environment drift (the `andrew` migration was installed and never reverted), not a `main` source defect: a `uv sync` / editable reinstall against `main`'s pin restores a working pair. Flagged P1 because it is the state a developer or the on-bench laptop is in right now. **Per CURRENT, `main`'s HEAD has since pulled the lerobot 0.6.0 port (`c6d97e7`), which resolves the mismatch on `main` proper — this finding was computed at `c7d9f27` before that port.**

### R20 · P2 — Every recording control endpoint returns HTTP 200 on a logical rejection, so three client error branches are unreachable *(formerly N5)*

**First identified:** 2026-07-23

R2 covers `/start-recording`. The same shape holds for the rest: `server.py:844-853` (`/stop-recording`), `:870-873` (`/recording-exit-early`), `:876-879` (`/recording-rerecord-episode`) all return the handler dict verbatim, while `record.py:701-702,732-733,750-751` return `{"success": False, …}` when no session is active. Consequences: `handleExitEarly` branches on `!response.ok` only (`RecordingSessionDialog.tsx:391`), so a rejected skip never clears `optimisticPhase`; `handleRerecordEpisode` toasts success for a rejected request (`:422-434`); `doStopRecording` inspects nothing (`:455` — that is R7). Fix: map logical rejections to 409 in `server.py`, and have the client require `data.success` (one change fixes R2, R7 and this).

### R21 · P2 — Quitting re-opens the browser camera previews while the backend still owns the cameras *(formerly N6)*

**First identified:** 2026-07-23

`confirmQuit` deliberately exits before teardown (`RecordingSessionDialog.tsx:488-494`). Because `recorded` is undefined, `handleRecordingExit` skips `closeStudio()`/`navigate` (`CollectPanel.tsx:263-274`) but still runs `setSessionCount((n) => n + 1)` (`:262`), and `RecordingForm` is keyed on it (`key={sessionCount}`, `:300`). The remount rebuilds `CameraConfiguration` with `streamsPaused` back at its initial `false` (`CameraConfiguration.tsx:70`), which re-enables `useAvailableCameras` (`:76`) and every `CameraPreview`'s `getUserMedia` (`:369`). So the browser starts grabbing the USB cameras again while the recording worker is still inside its rest-pose return and has not yet reached `robot.disconnect()` (`record.py:1634-1650`). On macOS usually tolerated; on Linux/V4L2 exclusive-open a real conflict, and it re-triggers the out-of-process AVFoundation enumeration.

### R22 · P2 — A page reload or tab close destroys every saved episode of a fresh session behind a generic browser prompt *(formerly N7)*

**First identified:** 2026-07-23

`useSessionExitGuard`'s `beforeunload` sets only `e.returnValue = ""` (`useSessionExitGuard.ts:97-102`), so browsers show their generic "Leave site?" — the carefully written `leaveDiscardMessage(resume)` (`recordingExit.ts:40-44`) is used only for the in-app popstate confirm (`:140`). On `pagehide` the beacon POSTs `/stop-recording?discard=true` (`:114-119`), which sets `discard_requested` (`record.py:692-694`) and makes the worker `rmtree` the **whole** stamped directory including every already-saved episode (`record.py:620-628`, `_discard_session_dataset` at `:955-1007`). Net: an accidental ⌘R twenty episodes in destroys all twenty, warned only by a browser-generic dialog. The semantic is deliberate and documented (`recordingExit.ts:4-9`) — filed P2 as a design/copy risk, but the coordinator may want it higher.

### R23 · P2 — `RecordingRequest.test_mode` is declared, documented as "skip robot connection for testing", and never honoured *(formerly N8)*

**First identified:** 2026-07-23

`record.py:292` declares `test_mode: bool = False  # Skip robot connection for testing`. `grep -rn "test_mode" makermodslab/` returns that single line — nothing reads it. An API user who sets it expecting a dry run gets a full hardware session: ports opened, calibration written to EEPROM (`record.py:1395-1396`), motion started. On a live-hardware repo a dead safety-shaped flag is worse than no flag.

### R24 · P2 (minor) — Terminal status logs and `print()`s on every poll after the session ends *(formerly N9)*

**First identified:** 2026-07-23

`handle_recording_status()` emits a log line **and** a bare `print()` on every request once `session_ended` is true (`record.py:756-764`). The dialog freezes polling on failed/warning outcomes (`:306`) and exits on clean ones, so the loop is bounded in practice — but any client left polling (a second tab's phantom dialog, per R2) produces one log + one stdout line per second indefinitely. `/recording-status` is in the quiet-poll access filter (`server.py:177`) precisely because this path is noisy; these two statements bypass it.

---

### R25 · P2 — A logged-out Collect session records into a BARE dataset name, at the same level as MakerMods Lab's state dirs *(formerly N10)*

**First identified:** 2026-07-24

The Collect flow namespaces the dataset id **only when the user is authenticated**:

```tsx
const datasetRepoId =
  auth.status === "authenticated" ? `${auth.username}/${datasetName}` : datasetName;
```

(`CollectPanel.tsx:180-183`; the form's preview *"Will be saved as `{auth.username}/{datasetName}`"* at
`RecordingForm.tsx:133-138` is likewise gated on `auth.status === "authenticated"`, and shows *"Log in to
Hugging Face to set the repository owner"* otherwise — so the UI is honest, it just proceeds anyway.)

An offline / logged-out station — the documented deployment mode, cf. Config C10 — therefore records into
`<HF_LEROBOT_HOME>/<name>`, the same directory level as `calibration/`, `robots/`, `ports/`, `outputs/`,
`makermodslab_models/` and `makermodslab_biso/`. Two consequences, neither of which the recorder checks for:

1. If the chosen name collides with a state dir, the recorder writes its dataset files **inside** it — unlike
   the import path, which refuses an existing target (`datasets.py:1106-1107`).
2. The resulting bare-named dataset is the precondition for the destructive delete in **Dataset D22 (P0)** —
   read that entry for the delete-side trace. It also cannot be pushed without a rename
   (`CollectHandoff.tsx:109` gates auto-push on `repoId.includes("/")`).

Filed P2 **here** because the recording-side defect is only the missing namespace fallback (one branch); the
severity lives on the delete side in Dataset D22. Cross-filed as **Dataset D23**.

---

### R26 · P1 — Follower calibration writes are silently discarded: `connect()` leaves the EEPROM write-protected (`Lock=1`), then `_write_calibration` writes into it and logs success *(formerly N11)*

**First identified:** 2026-07-25

The record flow energizes the follower and **write-protects its EEPROM** before writing calibration into that
EEPROM. The writes are discarded; the code logs *"Robot calibration applied successfully"* anyway.

The mechanism is the Feetech **`Lock`** register, not torque — upstream's framing of this as a torque-ordering
bug is a proxy for the real cause. `enable_torque()` bundles the two:

```python
# .venv/…/lerobot/motors/feetech/feetech.py:302-305
def enable_torque(self, motors=None, num_retry=0):
    for motor in self._get_motors_list(motors):
        self.write("Torque_Enable", motor, TorqueMode.ENABLED.value, num_retry=num_retry)
        self.write("Lock", motor, 1, num_retry=num_retry)     # ← EEPROM write-protect
```

The chain, verified against the pinned lerobot v0.6.0:

1. `record.py:1283` — `robot.connect(calibrate=False)`
2. `SOFollower.connect()` calls `self.configure()` **unconditionally**; `calibrate=False` skips only the
   `calibrate()` branch
3. `configure()` runs under `with self.bus.torque_disabled():`, whose `finally` calls `enable_torque()`
   (`motors_bus.py:686-690`) → **`Lock=1` on return**
4. `record.py:1395` — `_write_calibration(robot, "robot")` → feetech `write_calibration()` writes
   `Homing_Offset` (addr 31), `Min_Position_Limit` (9), `Max_Position_Limit` (11) — all EEPROM-region, all
   protected. No `disable_torque` occurs anywhere between steps 1 and 4.

**First-party corroboration:** the repo's own vendored autocal documents the precondition —
`vendor/feetech_autocal/auto_calibrate_script.py:895-897` explicitly writes `Lock=0` *"Before persistence:
unlock EEPROM"* ahead of its own `write_calibration`.

**Nothing is corrupted.** The previous EEPROM values survive intact; there is no path to *wrong* limits
persisted across a power-cycle. Both plausible servo behaviours land badly, though, because of
`record.py:1382-1387`: a silent ACK-and-discard leaves `wrote = True` and logs success, while an error bit is
swallowed by the `except` into one ERROR line and **recording continues anyway**. Upstream's `3d77190` deletes
that `try/except` so the call raises — arguably as important as the reordering.

Residual mismatch: `write_calibration(cal, cache=True)` sets `self.calibration` in Python regardless of whether
the servo accepted it, so Python normalizes against the **new** calibration while the servo reports positions
and clamps motion against the **old** one.

**Blast radius — record's follower only:**

| flow | affected | why |
|---|---|---|
| `record.py` follower | **yes** | chain above; **doubles on bimanual** — `utils/bimanual.py:48-50` delegates to `left_arm.connect()` + `right_arm.connect()`, both ending at `Lock=1` |
| `record.py` leader | no | `SOLeader.configure()` ends with a bare `disable_torque()` and never re-enables → exits at `Lock=0` |
| `calibrate.py` | no | `_step_homing` calls `disable_torque()` (`:417`) for the manual sweep, clearing `Lock` long before the write at `:625` — safe by accident of the UX, not by design |
| `teleoperate.py` | no | correct order: `bus.connect()` → `write_calibration()` → `configure()` (`:615-651`) |
| `rollout.py`, `auto_calibrate.py` | no | read-only preflight / subprocess; autocal unlocks explicitly |

**`arm_identity` already guards the dangerous case.** `verify_devices` compares EEPROM `homing_offset` against
the assigned calibration (±2 ticks, `arm_identity.py:172,206-208`) and raises `ArmIdentityError`, which
`record.py:1355-1359` catches to abort the session — so recording under a *stale* calibration is caught before
`_write_calibration` runs. Two residual gaps: `skip_identity_check` bypasses it entirely, and changing
`range_min`/`range_max` while `homing_offset` stays put (re-sweeping limits around the same centre) passes the
fingerprint and then never persists.

**Related, and undefended anywhere:** `Lock` is itself an EEPROM register, so `Lock=1` **survives a
power-cycle**. Clean teardown restores it via `bus.disconnect(disable_torque=True)`, but a crash, unplug, or
hard kill can latch it — after which *even the correctly-ordered teleop path* silently fails its next
`write_calibration`. No flow currently checks or clears `Lock` defensively on connect.

**Why P1 and not P0:** nothing is destroyed, old values survive, `arm_identity` catches the primary dangerous
case, and three of five robot flows plus both leader arms are structurally immune. Contrast Dataset D22, where
a bare name causes `rmtree` of a real state directory — irreversible, unguarded, no recovery.

**Verification before fixing — read-only, ~1 minute, no writes and no motion.** With one follower connected and
nothing else running, after `robot.connect(calibrate=False)` read `Lock` and `Torque_Enable` back on every
motor. `Lock == 1` confirms the whole chain without writing a byte. `Lock == 0` falsifies the analysis at its
root — run this first. **Escalate to P0 only if** a follow-up shows the writes actually land: writing
`Homing_Offset` to a *torqued* servo in POSITION mode shifts `Present_Position` instantly, and the control loop
would drive hard against an unchanged goal — a motion-safety issue rather than a silent no-op.

**Do not silently drop the `sync_read` retry monkeypatch** (`record.py:76-87`, global `num_retry=2` default)
when fixing this. It may be masking this bug's loud profile — intermittent
`sync_read 'Present_Position' … no status packet` seconds into a session, which presents exactly like a flaky
USB cable. Treat it as load-bearing until tested.

### R27 · P1 — One camera hiccup poisons every later recording in the process: the connect retry can't release a partially-connected robot, so the bus and camera read threads leak *(formerly N12)*

**First identified:** 2026-07-29 · **Resolved:** 2026-07-29 — fixed same day (`record.py`, `teleoperate.force_disconnect_partial`)
**Fix:** [PR #37](https://github.com/makermods-robotics/makermodslab/pull/37) (draft) · branch `fix/record-connect-partial-teardown`

A transient camera failure during `robot.connect()` left the follower bus open and the already-opened cameras'
background read threads running, for the **rest of the process**. Every subsequent recording attempt then failed
— first with the misleading `FeetechMotorsBus is already connected`, and thereafter by timing out against the
leaked read threads still holding the OS camera devices. Only restarting `makermodslab` cleared it.

Observed sequence (`front`/`wrist` rig): `attempt 1/3` → bus connects → `front` opens but comes up frame-dead →
`Timed out waiting for frame from camera OpenCVCamera(1)` → retry teardown → `attempt 2/3` →
`FeetechMotorsBus is already connected` → session failed, empty dataset removed.

**Mechanism.** The retry path did the right thing in spirit — `with contextlib.suppress(Exception): robot.disconnect()`
— but `disconnect()` is structurally incapable of releasing a partially-connected robot:

```python
# .venv/…/lerobot/robots/so_follower/so_follower.py
def is_connected(self):                      # all-or-nothing
    return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

@check_if_not_connected                      # raises DeviceNotConnectedError if not is_connected
def disconnect(self): ...
```

`connect()` opens the bus, then the cameras **in dict order**. When it dies on a camera, every camera *after* the
failing one was never opened → `is_connected` is `False` → the guard raises before a single component is
released → `suppress(Exception)` swallows it silently. So the teardown was a no-op, and the retry was guaranteed
to hit the already-connected bus. The bug bites whenever ≥1 camera follows the failing one in iteration order —
i.e. always on a 2-camera rig unless the *last* camera is the one that fails.

The leak was also self-perpetuating and cross-session: the read-failure spam for `OpenCVCamera(1)` appears in the
log a second *before* `attempt 1/3` connects anything — that thread belonged to the **previous** failed session,
and it was what made this session's `front` come up frame-dead. The `time.sleep(2.0)` "wait for camera resources
to be released" can never help, because the holder is in-process.

**Fix.** `teleoperate.force_disconnect_partial(device, label)` — component-wise teardown, each component
independently guarded, safe on fully/partially/never-connected devices: release every camera in `.cameras`
(tolerating `DeviceNotConnectedError` for the never-opened ones), then every bus from `_device_buses()` (covers
bimanual sub-arms). Called on **every** failed connect attempt in `record.py`, not just the retried ones, so a
terminal failure also leaves the process clean; and on the teleop-connect failure path, so a leader failure
can't strand the follower.

**Scope — recording only.** Teleoperation and calibration build their follower configs with `cameras == {}`, so
their `is_connected` is bus-only and `disconnect()` works; rollout runs as a subprocess, so its exit releases
everything. Recording is the only in-process flow that opens cameras.

**Regression tests** (`tests/test_teleoperate.py`): the mechanism is pinned against lerobot's *real*
`check_if_not_connected` decorator and the real all-or-nothing `is_connected` shape, so the test fails (and the
helper can collapse back to `robot.disconnect()`) if upstream ever makes `disconnect()` tolerant of partial
state.

**2026-07-30 follow-up — vanished-device hole closed.** Bench test (unplug the camera at recording start)
showed one path still poisoned the process until restart: a camera whose device disappears between the cv2
open and `_start_read_thread` reports `is_connected == False` with `thread is None`, so lerobot's
`disconnect()` raises `DeviceNotConnectedError` — which the helper treated as "never opened, nothing to
release" — while `videocapture` still held the OS capture session. The stale in-process session then degraded
every later open of the replugged camera (wrong fps profile / no frames). The helper now releases the raw
`videocapture` handle on exactly that path; regression test
`test_force_disconnect_partial_releases_vanished_cameras_capture_handle` pins lerobot's guard semantics
verbatim. Related polish: `friendly_hint` translates "Failed to open …Camera" as unplugged (and that branch
must precede the generic network hint — lerobot raises it as a `ConnectionError`, which the download hint's
`"connectionerror"` token would otherwise swallow).

### R28 · P1 — Chrome's camera hold outlives the preview UI **and makermodslab restarts**: record's connect races the browser's asynchronous device release, and after a camera replug it loses every time *(formerly N13)*

**First identified:** 2026-07-30 — root-caused live on the bench; **fix direction 1 implemented same day**
(verify-release gate `_wait_for_camera_release` + `_probe_camera_state` in `record.py`, replacing the blind
`sleep(2.0)`; polls each camera until it negotiates its configured mode, 30s cap, 0.5s settle after the gate's
own probe sessions release; unit tests in `tests/test_record.py`). Bench feedback folded in same evening:
the probe is tri-state — **free / absent (won't open → "likely unplugged, replug it") / held (degraded mode →
"another app has it")** — so an unplugged camera is no longer misreported as browser-held; a user **Stop
aborts the wait** immediately; and `friendly_hint` now translates the whole turbulence family ("failed to set
fps", "timed out waiting for frame", "do not match configured") — lerobot's raw message on a vanished device
is a nonsense fps read-back (e.g. `actual_fps=30.000003`), which the hint replaces with held-or-unplugged
guidance. On timeout the gate proceeds into the normal connect retries with a per-camera state warning.
Uncommitted on the `fix/training` working tree pending hardware verification.
Interim workaround if a camera stays *held* past the gate: quit Chrome (⌘Q); reopening it afterwards is fine.
**Fix:** [PR #38](https://github.com/makermods-robotics/makermodslab/pull/38) (draft) · branch `fix/record-connect-retry-misses-capture-size-mismatch` · [PR #39](https://github.com/makermods-robotics/makermodslab/pull/39) (draft) · branch `fix/camera-wedge-misleading-resolution-hint` · [PR #40](https://github.com/makermods-robotics/makermodslab/pull/40) (draft) · branch `fix/camera-release-settle-delay`

After a camera unplug/replug (done to exercise the R27 fix), **every** recording start failed at
`robot.connect()` with roving camera symptoms: first the wrist camera frame-dead (`Timed out waiting for frame …
Read thread alive: True`), later a stable `failed to set fps=30 (actual_fps=25.0)` — the "neighboring native
format" negotiation from the transient-turbulence taxonomy in `record.py` (comment at the connect retry loop).
Restarting makermodslab did **not** clear it, which falsely implicated hardware/process state.

**Evidence chain (all on the same bench, same evening):**
- Solo cv2 probe (AVFoundation, MJPG 640x480@30): both cameras deliver ~30fps. Concurrent probe (wrist opened
  while front streams, record's exact sequence): both fine — USB bandwidth/power exonerated.
- lerobot-native probe (`OpenCVCamera.connect()` with record's exact config, worker thread, wrist→front):
  **passes** in a fresh process.
- Same in-app record start driven by `curl` with Chrome **fully quit**: connects and records first try.

**Mechanism.** The recording modal's browser previews (getUserMedia) hold live AVCaptureSessions on the devices
until Start flips `streamsPaused`; `record.py` then sleeps a fixed 2.0s ("Waiting for camera resources to be
released") and connects, with 3 attempts × 2s backoff. Chrome's device release is asynchronous, and after USB
device churn its camera service can hold (or keep re-touching) the device far longer than the ~8s the retry
window covers — possibly indefinitely. A cv2 session opened against a device Chrome still holds comes up
degraded: frame-dead, or negotiated onto a lower-rate profile (actual_fps=5.0/25.0), or the neighboring
resolution. Because the hold lives in Chrome, it survives any number of makermodslab restarts — the discriminating
observation that separates this from R27-style in-process leaks.

**Fix directions** (not yet chosen):
1. Replace the blind `sleep(2.0)` with a verify-release gate: pre-open each configured camera with bare cv2 and
   require the 640x480@30 negotiation to succeed before entering `robot.connect()`; on bounded timeout, fail
   with an error that names the browser as the likely holder instead of a lerobot fps assertion.
2. Eliminate the class: serve previews from a backend cv2 feed so the browser never owns the devices — the
   approach already prototyped on the unmerged `fix/camera-preview-recording-conflict` branch (`d3de059`).
3. UI hint: map the three turbulence signatures to a plain-language "another app (usually the browser) is
   holding the camera — close other tabs/apps or quit Chrome" message.

**The holder, found (same evening):** recording runs in a **dialog**, so the studio's three panels stay
mounted underneath — and DeployPanel's per-binding `CameraThumbnail`s paused only on
`submitting || inferenceActive`. A camera bound in the Deploy form (wrist, on this bench) kept a live
getUserMedia stream through the entire release-gate + connect, invisible behind the recording dialog: the
gate honestly reported `wrist: held` for its full 30s because wrist *was* held — by our own UI. Fixed by an
app-wide `recordingCamerasBusy` flag in `StudioContext`: CollectPanel sets it at recording start (cleared in
`handleRecordingExit`, the dialog's single exit path, plus a panel-unmount guard), and DeployPanel's
thumbnails now pause on it. Any future browser-preview surface must pause on this flag too.

**SHELVED 2026-07-30 (late evening) — one holder still unidentified.** After all fixes (gate, tri-state,
app-wide pause, vanished-device teardown), `wrist: held` persisted in `--dev`. Decoded: `actual_fps=30.00003`
means the read-back PASSED — lerobot raised on `set()` returning False, which on AVFoundation means
`lockForConfiguration` failed: another process holds the device's config. OS-level confirmation: Chrome's
`video_capture.mojom.VideoCaptureService` helper had been running with continuously-accumulating CPU since
20:19 — i.e. some Chrome tab captured non-stop from the moment Chrome was relaunched (tab restore), through
every failed session. Prime suspect: a restored stale MakerMods Lab tab (`localhost:8000`, old bundle, no fixes)
— unconfirmed; debugging stopped here. **Resume with:** look for Chrome's tab-strip camera indicator / close
all MakerMods Lab tabs except one; verify the hold is gone with
`ps aux | grep video_capture.mojom | grep "Google Chrome"` (service exits seconds after the last stream
stops). Everything implemented tonight stands regardless and is uncommitted on the `fix/training` tree.

**2026-07-31 update — holder confirmed still live, new symptom signature.** Chrome's
`video_capture.mojom.VideoCaptureService` (PID 90602) observed running with 6m42s accumulated CPU and a start
time of **yesterday 20:19** — the exact service/start-time from the shelved note, so a Chrome tab has captured
continuously for ~15h through today's sessions. Today it manifested MID-STREAM instead of at connect: a
recording session (`eraser_place_20260731_111646`) ran 10 clean episodes, then died with lerobot's frame-age
watchdog (`latest frame is too old`) — the concurrent Chrome session intermittently starves/renegotiates the
device rather than blocking connect. Same holder, different phase. Saved episodes persisted (per-episode
save_episode + discard-only-empty), so only the in-flight episode was lost. User pointed at the tab-strip
indicator / close-stale-MakerModsLab-tabs / ⌘Q-Chrome step with the `ps` verification.

**User attestation (2026-07-31): there was only ONE MakerMods Lab tab yesterday too** — the "restored stale second
tab" suspect is dead. Additional measurement: the capture service is idle-resident (zero CPU growth over 30s
with no preview surface showing) — the "captured non-stop since 20:19" reading conflated process lifetime with
active capture; the hold is situational, active only while camera surfaces are mounted. Surviving hypothesis
for the --dev "wrist: held" result: `makermodslab --dev` serves the fixed React app on **:8080** while **:8000**
keeps serving the old committed dist (server.py: "In dev mode the React app runs on :8080") — a single tab on
:8000 during the --dev test would run the fix-less bundle (wrist bound in the Deploy form → its thumbnail
streams straight through the gate). CONFIRMED BY REPRODUCTION 2026-07-31 (late): the user ran `makermodslab
--dev` and reported a source-verified UI change (the training-toggle removal, present on the checked-out
branch) as missing — because the tab was on :8000, which serves the on-disk dist bundle even in dev mode,
while the live app runs on :8080. Same user habit, same mechanism, benign payload this time. Treat the
yesterday "fix didn't work in --dev" result as explained. Standing rule: in --dev, the tab must be on :8080;
anything observed on :8000 reflects the committed/locally-built dist, not the source.

**Related:** R21 (quitting re-opens browser previews while the backend owns the cameras — same ownership
conflict, opposite direction); R3/R27 (the failure-path leak that made earlier incarnations of this
unrecoverable in-process; with R27 fixed, tonight's failures stayed clean and retryable, which is what made the
Chrome hold observable at all). The landing `InferenceModal` has an equivalent thumbnail; it can't be open
concurrently with the recording dialog today, but wire it to the flag if that ever changes.

### R29 · P1 — Identical-name cameras get paired to cv2 indices by coin flip: name-based browser matching swaps front/wrist silently, previews look correct while the dataset records crossed streams *(formerly N14)*

**First identified:** 2026-07-31
**Fix:** [PR #41](https://github.com/makermods-robotics/makermodslab/pull/41) (merged) · branch `fix/camera-identity-twin-cameras`

Both rig cameras report AVFoundation `localizedName` "USB Camera". `useAvailableCameras.ts` pairs backend cv2 indices with browser `MediaDeviceInfo.deviceId`s by label (exact → prefix → contains); with two identical labels, the pairing follows Chrome's `enumerateDevices()` order, which is not guaranteed to match cv2's uniqueID-sorted order. The preview tile shows the browser stream (deviceId) while recording opens the stored integer `camera_index` (record.py `_build_camera_configs`, `index_or_path=`), so a mis-pair is invisible in the UI: the user names cameras by looking at correct previews and the config binds each name to the other physical device.

Live hit: `home.json` configured 2026-07-31 10:27, dataset `eraser_place_20260731_103436` recorded 10:34 — `observation.images.front` contains the wrist-cam stream and vice versa (verified by frame inspection; front view is gripper-POV and moves with the arm). Ground truth at time of hit: physical front = index 0 (`0x1113…`), wrist = index 1 (`0x1114…`); record had them swapped.

History: fixed twice, both fixes absent from main. (1) The leLab-era camera arc made unique_id the canonical camera address with per-open `resolve_index` (07-08); reverted wholesale to pre-fork index-based behavior in `dc427fb` (07-09). (2) `camera_identity.py` + `cameraResolve.ts` re-anchoring landed in `c4e5198` (07-15), reached the main lineage, and was deleted two days later by the redesign snapshot `9e18b27` (07-17, "camera_identity/camera_preview removed" — Layout D frontend on the min_stable backend). Every redesign-UI dataset was recorded unprotected; the fix is portable (123+35 lines).

Fix direction: store `unique_id` in the robot record per camera; at record/rollout start re-resolve cv2 index from unique_id and fail loudly on mismatch (port `camera_identity.py` from `c4e5198`). Frontend: when two backend cameras share a name, refuse silent pairing — surface an explicit "identify by covering a camera" disambiguation or show unique_id.

**Fix implemented 2026-07-31** on branch `fix/camera-unique-id-rebind`: `makermodslab/camera_identity.py` (in-process resolver for record, fresh-subprocess resolver for rollout), re-anchor + loud-fail wiring in `record._build_camera_configs` and `rollout._format_cameras_arg`, `unique_id` stored from `/available-cameras` through CameraConfiguration → robot record → record/inference requests, re-sync effect never rewrites a set `unique_id` (backfills only unambiguous names), twin-camera warning + per-port uniqueID shown in the picker. `home.json` migrated with correct bindings (front=0x1113…@0, wrist=0x1114…@1). Residual risk: with twin cameras the *browser preview tiles* can still pair swapped at naming time (banner warns; cover-lens verification is the manual check).

### R30 · P2 — The preview-handover pattern wedges its feature flag (and now all previews) if `stop_all()` raises: shared shape across record, rollout, and focus-tune starts *(formerly N15)*

**First identified:** 2026-07-31 (flagged during the PR #41 merge integration)

The backend-preview handover pattern (set the feature's active flag under its lock → `camera_preview_manager.stop_all()` → proceed) has a shared latent failure: if `stop_all()` raises, the active flag stays set with no worker running to reset it. Since `/camera-preview/{index}` 409s on these flags, previews are then blocked process-wide until restart. Present identically in `record.handle_start_recording`, `rollout.handle_start_inference`, and (on `feat/focus-tune-button` post-merge) `focus_tune.handle_start_focus_tune` — the focus-tune port deliberately mirrored the existing shape rather than diverging. Fix direction: wrap the pre-worker section in try/except that clears the flag on failure, in all three modules at once (single pattern, single fix). Low likelihood (stop_all is defensive), but the blast radius is every camera surface.

## Design gaps and lower-confidence risks

> **Verdict on `main` (2026-07-23): all five STILL REAL** as accurate design-gap observations (not defects) —
> the cited code exists as described (process-global state with no session id `record.py:183-240`;
> always-start-on-mount `RecordingSessionDialog.tsx:203-211`; module-global counters/logs; no in-page
> frame-health surface). No P0–P2 severity assigned; they map to the "Design gap" key.

### D1 — No durable session identity, ownership, or reattachment contract

**First identified:** 2026-07-14

The page receives configuration only through router location state and always attempts a new start on mount (`frontend/src/pages/Recording.tsx:88-101`, `frontend/src/pages/Recording.tsx:201-211`, `frontend/src/pages/Recording.tsx:335-372`). Backend state and controls are process-global and carry no session ID (`makermodslab/record.py:181-240`, `makermodslab/record.py:677-862`). Reload, browser restart, a delayed beacon, or a second client cannot prove which session it owns. ⚠ UNRESOLVED @ HEAD ce42158 — `Recording.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/studio/ + recording/, construct not re-located.

### D2 — No episode review or selective commit workflow

**First identified:** 2026-07-14

The operator can save or re-record only the current in-memory take. Once `save_episode()` commits an episode, the recording interface provides no thumbnail/video review, quality flag, metadata edit, selective delete, or reorder step. Resume Quit also cannot roll back already appended episodes.

### D3 — Recording observability does not prove frame health

**First identified:** 2026-07-14

The live page shows phase, time, episode count, and an in-memory log, but no camera frame preview, per-camera freshness, dropped-frame count, effective resolution/FPS, disk-space estimate, encoder backlog, or serial retry rate.

### D4 — Recovery state is process-local

**First identified:** 2026-07-14

Active flags, phase, counters, event controls, logs, and terminal outcome live in module globals or memory-only buffers. A backend restart loses ownership and recovery context even if a partial dataset remains on disk.

### D5 — Cross-feature bug duplication needs one canonical owner

**First identified:** 2026-07-14

Hardware mutex, arm-identity fail-open, motor-power enforcement, and rest-return reporting also affect teleoperation, inference, and calibration pathways. Fixes should live in shared ownership/safety services and be referenced from each feature ledger.

## Validation performed (original audit)

- Traced the frontend, API routes, recording worker, dataset persistence, shared exit guard, hardware helpers, and tests.
- Ran `.venv/bin/python -m pytest tests/test_record.py -q`: **53 passed, 7 failed** (all the v0.6 `rgb_encoder` compatibility regression — see R1/R19).
- Inspected installed v0.6 signatures for `DatasetRecordConfig`, `LeRobotDataset.create()`, `.resume()`.
- Used a process-local, no-hardware status probe to confirm terminal status omits a nonzero saved-episode count.
- Did not run a real server, connect hardware, mutate user data, start uploads/training/jobs, or contact Hugging Face.

## Test coverage gaps exposed by the audit

- No regression test reaches fresh/resume episode saving under the pinned dependency.
- No frontend test asserts both HTTP status and `{success}` for start/stop/delete actions.
- No test preserves and asserts terminal `saved_episodes` across clean, warning, and failed outcomes.
- No test covers ended-session discard after resume.
- No test simulates leader connection failure after follower/camera connection and verifies full cleanup.
- No test specifies whether natural episode timeout saves or discards.
- No concurrency test races recording against teleoperation, inference, calibration, upload, merge, or training ownership.
- No integration test covers reload, stale page-leave beacon, second tab, or session reattachment.
- Existing rest-return tests spy that the helper was called, but do not require a failed return to alter terminal outcome.

## Context: contradictions to the original brief (resolved during re-verification)

1. **"recording frontend (`frontend/src/components/recording/`)"** contains only `RecordingSessionDialog.tsx` and `CameraConfiguration.tsx`. Request construction, exit handling and post-session handoff live in `frontend/src/components/studio/` (`CollectPanel.tsx`, `CollectHandoff.tsx`) and `StudioContext.tsx`. R17, R21, and R2's reach are only visible there.
2. **The brief framed R5/R6 as "the two former P0s" whose fixes may have landed.** On `main` neither fix exists in any form — the `andrew` work simply never merged.
3. **The excluded item.** A verdict is recorded for R12 (arm-identity **read-failure** fail-open) because it is a distinct code path from the excluded rows-3/4 warn-and-proceed severity question; nothing is said about the latter's severity.
