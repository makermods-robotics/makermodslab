# Recording Bug List — RE-VERIFIED against `main`

Re-verification date: July 23, 2026
Branch and commit re-verified: `main` at `c7d9f27` (working tree carries only untracked `.agents/`, `debug/`, `internal_docs/` plus the `frontend/dist` bundle swap; no source modifications).

Source re-verified: `internal_docs/bugs/Recording Bug List.md`, forensically verified **2026-07-14 against the `andrew` branch at `257580f`**, including that branch's uncommitted LeRobot v0.6 migration. **None of that state is on `main`.** Two of its entries carried "Status: In progress" fix notes (R5, R6); **neither fix exists on `main`.**

Read-only audit. No server was started, no serial port opened, no camera opened, no motor energized, no dataset mutated, no Hub call made. Evidence is static code reading plus three no-hardware Python introspection calls against the installed `lerobot` package.

## Structural change that dominates this re-verification

The redesign merge (`9e18b27`, `feat(redesign): Launchpad + Skill studio snapshot; sessions become window dialogs`, on `main` via `c7d9f27`) **deleted `frontend/src/pages/Recording.tsx`**. Every frontend `file:line` in the original list is dead. The live session is now:

- `frontend/src/components/studio/CollectPanel.tsx` — builds the recording request, owns the session dialog.
- `frontend/src/components/recording/RecordingSessionDialog.tsx:101-919` — the ported session UI (modal over the Launchpad).
- `frontend/src/components/studio/CollectHandoff.tsx` — the post-session banner (replaces the old `/upload` page).
- `frontend/src/lib/recordingExit.ts`, `frontend/src/hooks/useSessionExitGuard.ts` — unchanged in substance.

The port was near-verbatim, so **most frontend defects survived the rewrite at new coordinates**. Two behaviours changed materially: **resume recording is no longer reachable from the shipped UI** (`CollectPanel.tsx:140-141,247` — `resume: false` hardcoded, comment: "resume path removed — sessions always create a new dataset"), and **a default-on automatic Hub push was added** (new finding N3, ranked P0).

## Severity key

Unchanged from the source list (P0 / P1 / P2 / Design gap). New findings are ranked by the brief's rubric: P0 = hardware safety, dataset loss/corruption, silent private-data publication, or main path completely broken.

## Part 1 — verdict table

| # | Original entry (short) | Original sev. | Verdict on `main` | Key evidence |
|---|---|---|---|---|
| R1 | v0.6 migration breaks all recording | resolved | **NOT APPLICABLE-SUPERSEDED** (but see N5) | `pyproject.toml:22` still pins `82dffde`; `record.py:1202,1232` still pass `vcodec=` |
| R2 | Logically rejected start is HTTP 200; client trusts `response.ok` | P1 | **STILL REAL** (escalates to dataset loss) | `server.py:838-841`; `record.py:462-484,656-658`; `RecordingSessionDialog.tsx:341-360` |
| R3 | Setup failure after follower connect leaks port/cameras | P1 | **STILL REAL** | `record.py:1326-1328` re-raise; only `try/finally` starts at `record.py:1412`/`1614` |
| R4 | Episode-duration timeout discards the take, can retry forever | P1 | **STILL REAL** (worse: UI now names it "auto-advance") | `record.py:1476-1482,1485-1540`; `RecordingSessionDialog.tsx:274-289,640` |
| R5 | Terminal status erases `saved_episodes` | P1 (was flagged "in progress") | **STILL REAL — fix absent from `main`** | `record.py:642` zeroes it; `record.py:821-824` gates it on `recording_active`; no `last_session_saved_episodes` anywhere in `makermodslab/` |
| R6 | "Discard & exit" after a resumed session deletes the whole dataset | P0 (latent) | **STILL REAL, STILL LATENT — fix absent; R5/R6 coupling STILL LIVE** | `RecordingSessionDialog.tsx:512-527` (no `resume` check); `record.py:847-885` (whole-dir `rmtree`); gate `keptSomething` at `:788` |
| R7 | Done/Quit disarm the page-leave guard before the backend acknowledges | P1 | **STILL REAL** | `RecordingSessionDialog.tsx:447-470,488-494`; `useSessionExitGuard.ts:65-70,159-166`; `record.py:689-690` |
| R8 | Resume recording can mutate a dataset in use by upload/merge/training | P2 (P1 if common) | **STILL REAL, reach reduced to API-only** | `record.py:460-484` never calls `_dataset_in_use` (`datasets.py:696-748`); `CollectPanel.tsx:247` |
| R9 | Hardware ownership incomplete / non-atomic across features | P2 | **STILL REAL** | `record.py:215`, `teleoperate.py:102`, `rollout.py:114` — three module-local locks; `record.py:461-473` reads the others' flags |
| R10 | Motor-power limit failure hidden while recording proceeds | P2 | **NOT APPLICABLE-SUPERSEDED as framed; a narrower defect REMAINS** | `utils/config.py:288-294` (sessions run at stock torque now); warnings still dropped at `record.py:1384,1388` vs surfaced at `teleoperate.py:535,539,805-806` |
| R11 | Rest-pose capture/return failures reported as a clean recording | P2 | **STILL REAL** | `teleoperate.py:234-291` returns `None`; `record.py:1624` ignores it; `record.py:587` sets `outcome="ok"` |
| R12 | Arm-identity read failure silently disables the swap guard | P2 | **STILL REAL** (excluded topic acknowledged below) | `arm_identity.py:285-288` returns `(None, None)`; `verify_devices` `arm_identity.py:391-413` |
| R13 | Ended-session deletion ignores the server result | P2 | **STILL REAL** | `RecordingSessionDialog.tsx:512-527`; `record.py:867-872` legitimately refuses |
| R14 | Re-record accepted outside the recording phase | P2 | **STILL REAL — and it DOES destroy data (see N1)** | `record.py:736-747` vs advertised control at `record.py:777-778` |
| R15 | Backend accepts recording parameters outside the UI contract | P2 | **STILL REAL** | `record.py:263-294` — no `Field(...)`/validator anywhere in the module |
| D1 | No durable session identity / reattachment | design gap | **STILL REAL** | `record.py:182-239` process globals; `RecordingSessionDialog.tsx:203-211` starts on mount |
| D2 | No episode review or selective commit | design gap | **STILL REAL** | no per-episode delete/review path in `record.py` or `recording/` |
| D3 | Observability does not prove frame health | design gap | **STILL REAL** | dialog renders phase/timer/log only (`RecordingSessionDialog.tsx:692-867`) |
| D4 | Recovery state is process-local | design gap | **STILL REAL** | `record.py:182-239`; ring buffer `record.py:102-133` |
| D5 | Cross-feature duplication needs one owner | design gap | **STILL REAL** | unchanged |

### Highest-value item: the R5 / R6 coupling — determination

**Both fixes described as "In progress" in the source list are absent from `main`. The coupling is still live, in exactly the shape the list warned about.**

**R5 — STILL REAL.**

1. The worker's `finally` zeroes the live counter: `saved_episodes = 0` (`makermodslab/record.py:642`), and it is the only place the count lives.
2. `handle_recording_status()` adds `saved_episodes` to the payload only inside `if recording_active and recording_config:` (`makermodslab/record.py:821-824`). The terminal block above it (`record.py:802-813`) adds `discarded_empty`, `outcome`, `error`, `hint` — and no count.
3. There is **no** `last_session_saved_episodes` global on `main`. `grep -rn "last_session_saved_episodes" makermodslab/ tests/` returns nothing, and `tests/test_record.py` contains no `test_terminal_status_retains_saved_episodes`.
4. Ordering makes the omission total: the worker sets `recording_active = False` at `record.py:638` **before** zeroing at `:642`, but since the field is only emitted while `recording_active` is true, no poll can ever observe the true count in a terminal payload.

Consumers on `main`:

- Clean finish: `RecordingSessionDialog.tsx:308-314` hands off `saved_episodes: status.saved_episodes || 0` → **always 0**. `CollectHandoff.tsx:90-96` then renders "Dataset `<id>` saved · **0 episodes**" for every successful session.
- Failed / warning finish: the dialog freezes on the terminal payload (`doneRef` at `:305`) and computes `const savedEpisodes = backendStatus.saved_episodes ?? 0; const keptSomething = savedEpisodes > 0 && !discarded_empty` (`:787-788`). `keptSomething` is therefore **always false**, so **"Keep episodes & continue"** (`:832-839`) and **"Discard & exit"** (`:843-851`) never render — only "Back to home" (`:852-858`), which exits with no handoff payload, so no CollectHandoff banner and no dataset preselection.

Severity: **P1** (unchanged). The dataset is on disk and reachable from the dataset library, so this is a blocked recovery flow plus a false episode count, not loss.

**R6 — STILL REAL, still latent, still masked by R5, and now additionally gated by the UI's loss of resume.**

1. `discardAndExit()` (`RecordingSessionDialog.tsx:512-527`) posts `{dataset_repo_id: repoId}` to `/delete-dataset` with **no reference to `recordingConfig.resume`** — the component *has* `resume` in scope (`:160`) and uses it for the confirm copy (`:194,896-898`), but not here.
2. `handle_delete_dataset()` (`record.py:847-885`) resolves `HF_LEROBOT_HOME / repo_id` and `shutil.rmtree`s the whole directory (`:875`). It is resume-unaware by design (general dataset-browser delete).
3. The backend's own worker-side discard **is** resume-safe and tested: `_discard_session_dataset` returns early on `resume` (`record.py:977-979`), `_discard_empty_dataset` likewise (`record.py:911-913`), covered by `tests/test_record.py:359,403`. The frontend button bypasses both.
4. The button is rendered only when `keptSomething` is true (`:843`), which R5 makes permanently false. **Fixing R5 unmasks R6 — the dependency the source list called out is unchanged.**
5. **New gate on `main`:** `CollectPanel.tsx:247` hardcodes `resume: false` and no other frontend file sets it (`grep -rn "resume" frontend/src` → only training-side and copy-string hits). So a resume session can today only be created by a direct `POST /start-recording`. R6 therefore needs *both* R5 fixed *and* an API-created resume session.

Severity: **P0 if R5 is fixed without fixing R6** (unrecoverable loss of a pre-existing dataset), otherwise unreachable today. **The two must be fixed in the same change.** A resume-safe `discardAndExit` is ~3 lines (skip the delete when `resume`, keep the episodes, retitle the button) and should land **before or with** the `last_session_saved_episodes` snapshot.

**What I could not settle statically:** whether a fix landed and was reverted, or never landed — I did not walk branch history beyond `main`'s log for `makermodslab/record.py`. The `andrew` branch's tests (`test_terminal_status_retains_saved_episodes`, `test_worker_quit_keeps_resumed_dataset`) are simply absent here.

### Per-entry notes where the verdict needs justification

**R1 — NOT APPLICABLE-SUPERSEDED.** `main` never took the v0.6 migration. `pyproject.toml:22` pins `lerobot@82dffde…` and `record.py:1202,1232` pass `vcodec=cfg.dataset.vcodec`, which is self-consistent with that pin. The entry described the `andrew` working tree only. **However the installed environment contradicts the pin — see N5.**

**R2 — STILL REAL, and the consequence on `main` is worse than "confusion".** `/start-recording` returns the handler dict verbatim (`server.py:838-841`), so `{success: false}` from `record.py:462-484` (already-active / teleop / inference / invalid name) or `:656-658` (synchronous setup failure) serializes as **HTTP 200**. `startRecordingSession()` branches on `response.ok` alone (`RecordingSessionDialog.tsx:343`), so a rejection sets `recordingSessionStarted = true` and toasts "Recording Started". The comment at `:200-202` even claims "the second call returns 409" — no code path returns 409. Because `guardActive = recordingSessionStarted && !sessionEnded` (`:191`), the phantom dialog then **arms the page-leave guard with `beaconUrl = /stop-recording?discard=true`** (`:195`) and `onLeave = stopRecordingForLeave` (`:166-182`). Any exit from that phantom dialog — Quit button, back button, unmount, tab close — issues a discard against **the real, unrelated live session**, whose fresh stamped dataset is then deleted wholesale by `_discard_session_dataset` (`record.py:620-628`). Reach: `SingleTabGuard.tsx:120-140` renders an overlay with a **"Use this tab"** button and does *not* unmount the app, so a second tab that takes over lands on a working Launchpad while tab 1's session keeps running — pressing Start there produces exactly this phantom. Fix is two lines (non-2xx on logical rejection; require `data.success === true`).

**R3 — STILL REAL, verbatim.** `teleop.connect(calibrate=False)` failure re-raises at `record.py:1328` with the follower and its cameras already connected (`:1265`); the only unconditional rest/torque-disable/disconnect block is `record.py:1412` … `1614-1634`, entered only after the identity guard, calibration write, torque reset and rest-pose capture. A pre-loop raise skips it entirely, and the worker (`record.py:588-643`) keeps no device handle. The `ArmIdentityError` path does disconnect both (`record.py:1341-1345`) — that one case is covered; the others are not.

**R4 — STILL REAL, and the new UI reinforces the misleading reading.** `record.py:1468-1482`: a natural `record_loop` return without `_exit_early_triggered` logs "TIMEOUT — triggering re-record" and sets `web_events["rerecord_episode"] = True`; the take is then cleared at `:1492` and the same episode retried without incrementing `saved_episodes` (`:1539-1540`). Only an operator action sets `_exit_early_triggered` (`record.py:725`). New: `RecordingSessionDialog.tsx:274-289` plays `playAutoAdvanceWarning()` three seconds before the limit, and the primary button reads "End Episode" (`:640`) — the UI now explicitly primes the operator that the timer will **auto-advance**, when reaching it actually destroys the take.

**R8 — STILL REAL, reach reduced.** `_dataset_in_use()` exists and covers recording / upload / merge / local training (`datasets.py:696-748`) and is called by delete (`record.py:865-869`) and upload (`record.py:1064-1068`), but never by `handle_start_recording` (`record.py:460-484`). Because the UI can no longer request a resume (`CollectPanel.tsx:247`), a fresh session always writes a new timestamped directory (`record.py:519-520`), so the reverse-ordering race is API-only today. Keep P2.

**R10 — SUPERSEDED as framed.** Commit `714d29e` ("replace motor_power percent with raw max_torque_limit") changed the product semantics: per `utils/config.py:288-294`, the robot's torque slider now sets **auto-calibration drive torque only**, and "regular sessions (teleop/record/skill runs) run at stock LeRobot torque". So there is no longer an operator-selected recording cap that can be silently unenforced. What survives is narrower and still real: `reset_torque_limit()` and `clear_goal_velocity()` return warning lists (`motor_power.py:138,181`), and **recording alone discards them** (`record.py:1384,1388`) while teleoperation folds them into its surfaced warnings (`teleoperate.py:535,539,658,663,805-806`) and rollout collects them (`rollout.py:488`). Recording already has an `identity_warnings` global that reaches the status payload (`record.py:212,817-818`), so this is a one-line inconsistency. **P2.**

**R12 — STILL REAL.** `verify_arm()` returns `(None, None)` on a `Present_Position`/`Homing_Offset` read failure (`arm_identity.py:283-288`), `verify_devices()` therefore reports no refusal (`arm_identity.py:391-413`), and recording proceeds to `_write_calibration` (`record.py:1377-1378`) and motion. Documented fail-open (`arm_identity.py:270-272`). Per the brief, the separate **arm-identity warn-and-proceed severity question (rows 3/4) is excluded from this report** — a fix is committed on an unmerged branch. This entry is the *read-failure* branch, which is a different code path; I record the verdict without re-litigating severity.

**R14 — STILL REAL and materially under-rated.** See new finding **N1**: on `main` this is a data-loss path, not just flag state.

## Part 2 — new findings

### N1 — P1 — A re-record accepted during the reset phase silently discards the NEXT fully recorded episode

`handle_rerecord_episode()` sets `recording_events["rerecord_episode"] = True` and `exit_early = True` whenever the session is active (`record.py:736-747`) — no phase check (this is R14's mechanism). The loop, however, **only ever clears `rerecord_episode` inside its own re-record branch** (`record.py:1490`), and lerobot's `record_loop` clears `exit_early` but never `rerecord_episode`:

```
.venv/.../lerobot/scripts/lerobot_record.py:285-287
        if events["exit_early"]:
            events["exit_early"] = False
            break
```

(The CLI's own outer loop clears it at `lerobot_record.py:508-511`; MakerMods Lab's reimplementation does not mirror that in the reset paths.)

Trace, on `main`:

1. Re-record arrives during a **reset** phase → `exit_early` set → the reset `record_loop` (`record.py:1578-1591`) returns early. The post-reset checks at `record.py:1596-1605` inspect `exit_early` and `stop_recording` — **never `rerecord_episode`**.
2. The while-loop continues to the next iteration: `current_phase = "recording"`, `_exit_early_triggered` reset (`record.py:1426`), and a **full episode is recorded**.
3. After that episode, `if web_events["rerecord_episode"]:` at `record.py:1485` is still True → `dataset.clear_episode_buffer()` at `:1492` → **the complete take is thrown away** before ever reaching `dataset.save_episode()` at `:1545`.

Reachability without a direct API call: the frontend gates the control on `available_controls.rerecord_episode`, which the backend sets only during `current_phase == "recording"` (`record.py:777-778`), but the dialog polls at 1 Hz (`RecordingSessionDialog.tsx:322`) and the Re-record button's `disabled` prop reads only that stale flag (`:766`) — it is **not** gated by `optimisticPhase`, which the same component already sets to `"resetting"` the instant the user presses End Episode (`:384`). So for up to ~1 s after ending an episode — precisely when an operator realises the take was bad — the button and the Backspace shortcut (`:567-571`) are live and land in the reset phase.

Consequences: one completed demonstration destroyed; the toast said "Episode N will be re-recorded" (`:426`) while episode N is in fact already saved and kept; the episode counter does not advance and no error is shown.

Fix boundary: clear `rerecord_episode` at the top of each recording phase (or after every reset phase), reject the request server-side outside `current_phase == "recording"`, and disable the button on the optimistic phase as well.

### N2 — P0 — Every recorded dataset is auto-published to a PUBLIC Hub repo by a default-on toggle hidden in a collapsed section

Chain, all on `main`:

1. `StudioContext.tsx:42` — `DEFAULT_COLLECT_FORM.pushToHub = true`.
2. `RecordingForm.tsx:204-254` — the "Push to Hugging Face Hub" checkbox lives inside `<Collapsible className="group space-y-3">` with **no `defaultOpen`**, i.e. the "Advanced parameters" section is **collapsed by default**. Its helper copy (`:245-251`) reads "Uploads the dataset to your Hugging Face account in the background once the session ends" — it never says the repo will be **public**.
3. `CollectHandoff.tsx:102-110` — the post-session banner renders `<UploadToHubAction autoStart={collectForm.pushToHub && repoId.includes("/")} />`.
4. `CollectHandoff.tsx:191-202` — on mount, `start([], false)` fires with **no confirmation dialog**.
5. `useDatasetUpload.ts:111-119` → `replayApi.ts:279-292` → body `{tags: [], private: false}` → `record.py:1104` `dataset.push_to_hub(tags=tags, private=request.private)` → **public repository**.

So: an authenticated user who records a session gets the full dataset — raw camera video of their lab/home plus the free-text task description — pushed to a world-readable Hugging Face repo automatically, with the only opt-out being a checkbox they must first expand "Advanced parameters" to see. The manual path at least offers a visibility control (`UploadDatasetDialog.tsx` + `VisibilityToggle.tsx`); the automatic path offers none.

Per the brief's rubric this is **silent private-data publication → P0**. Mitigations that keep it from being catastrophic-on-every-run: it requires HF authentication (the `repoId.includes("/")` guard at `CollectHandoff.tsx:109`), and a success toast with a link appears afterwards (`:147-163`).

Fix boundary: default `pushToHub` to false, or make the auto-push private (`start([], true)`), or route the auto-push through the same visibility confirmation as the manual dialog. Also worth noting: `UploadDatasetDialog.tsx:14-18` documents itself as "Private-by-default toggle" while `:38` is `useState(false)` — i.e. **public** by default. Doc and behaviour disagree; the manual path may belong to the Dataset ledger rather than this one.

### N3 — P1 — The post-session release grace is invisible, and its controls report actions that never happen

The backend deliberately publishes `releasing` "so the UI isn't lying about the arm's state" (`record.py:217-224,773`) and a matching message "Returning the arm to its rest position…" (`record.py:783`). **`RecordingSessionDialog.tsx` reads neither.** The `BackendStatus` interface (`:65-92`) has no `releasing` field and no `message` field, and `getStatusText()` (`:614-629`) has no release branch. Teleoperation does consume it (`TeleopDialog.tsx:67,106`; `Teleoperation.tsx:49,87`), so this is recording-specific.

During the grace the follower is **energized and driving** back to its session-start pose (`record.py:1616-1624`). Meanwhile:

- `current_phase` is already `"completed"` (set at `record.py:1608`, before the `finally`), and `recording_active` is still true, so `sessionEnded` (`:156-159`) is false → the live chrome stays up and the phase pill reads **"SESSION COMPLETE"** (`:628`).
- `available_controls.stop_recording` is `recording_active` (`record.py:775`) → **Done and Quit remain enabled** (`:672-689`).
- Pressing **Quit** calls `handle_stop_recording(discard=True)`, which hits the `if releasing:` early return at `record.py:680-687` and **never sets `discard_requested`** — yet the UI toasts "Quitting — Discarding the recording…" (`:458`) and closes (`:493`). Nothing is discarded; every episode is kept. The inverse is equally wrong: an operator who genuinely wants to discard at that moment cannot.

Impact: an operator is told the session is complete while an arm is still moving under power (workspace-reach / cable-tug hazard), and is told a discard happened when it did not.

Fix boundary: add `releasing` to `BackendStatus`, render the backend's own message, hide/relabel Done+Quit to a single "Release now" during the grace.

### N4 — P1 — The installed environment contradicts `main`'s lerobot pin; recording fails immediately in this checkout as installed

- `main` uses the pre-v0.6 dataset API: `vcodec=cfg.dataset.vcodec` for both resume and create (`record.py:1202,1232`), consistent with `pyproject.toml:22` (`lerobot@82dffde…`) and `uv.lock:1041,1057`.
- The repo `.venv` holds a **different** build: `.venv/lib/python3.12/site-packages/lerobot-0.6.0.dist-info/direct_url.json` → `{"requested_revision": "v0.6.0", "commit_id": "30da8e68…"}`.
- Verified by introspection (no hardware): `DatasetRecordConfig` fields are `[… video_encoding_batch_size, rgb_encoder, depth_encoder, streaming_encoding, encoder_queue_maxsize, encoder_threads]` — `hasattr(cfg, "vcodec")` is **False**, `hasattr(cfg, "rgb_encoder")` is **True**; and `LeRobotDataset.create/.resume` accept `rgb_encoder`/`depth_encoder`, not `vcodec`.

Consequence: running `makermodslab` from this checkout **against the venv as it stands today** raises `AttributeError: 'DatasetRecordConfig' object has no attribute 'vcodec'` inside `record_with_web_events` — after the start response has already reported success (`record.py:648-653`) — for both fresh and resumed sessions. Every recording is dead on arrival on this machine.

This is environment drift (the `andrew` branch's migration was installed and never reverted), not a `main` source defect: a `uv sync` / editable reinstall against `main`'s pin restores a working pair. Flagged P1 because it is the state a developer or the on-bench laptop is in **right now** and it presents as "recording is completely broken". The durable lesson also matters: `main` and the `andrew` migration are API-incompatible in both directions, so the venv must be resynced on every branch switch.

I did not run the test suite (brief: no full-suite runs), so I did not measure how many `tests/test_record.py` cases fail in this state; the source list's "7 failures" figure is consistent with it.

### N5 — P2 — Every recording control endpoint returns HTTP 200 on a logical rejection, so three client error branches are unreachable

R2 covers `/start-recording`. The same shape holds for the rest: `server.py:844-853` (`/stop-recording`), `:870-873` (`/recording-exit-early`), `:876-879` (`/recording-rerecord-episode`) all return the handler dict verbatim, while `record.py:689-690,720-721,738-739` return `{"success": False, …}` when no session is active. Consequences in the dialog:

- `handleExitEarly` branches on `!response.ok` only (`RecordingSessionDialog.tsx:391`), so a rejected skip never clears `optimisticPhase` — the Advance button stays disabled and the phase timer frozen at 0 for the remainder of that render, with no error toast.
- `handleRerecordEpisode` likewise (`:422-434`) toasts success for a rejected request.
- `doStopRecording` inspects nothing at all (`:455`) — that is R7.

Fix boundary: map logical rejections to 409 in `server.py`, and have the client require `data.success` (one change fixes R2, R7 and this).

### N6 — P2 — Quitting re-opens the browser camera previews while the backend still owns the cameras

`confirmQuit` deliberately exits before teardown: "don't wait for the backend's rest-return/cleanup" (`RecordingSessionDialog.tsx:488-494`). Because `recorded` is undefined, `handleRecordingExit` skips `closeStudio()`/`navigate` (`CollectPanel.tsx:263-274`) but still runs `setSessionCount((n) => n + 1)` (`:262`), and `RecordingForm` is keyed on it (`key={sessionCount}`, `:300`). The remount rebuilds `CameraConfiguration` with `streamsPaused` back at its initial `false` (`CameraConfiguration.tsx:70` — `releaseAllCameraStreams` only ever sets it true, `:197-199`), which simultaneously re-enables `useAvailableCameras` (`:76`) and every `CameraPreview`'s `getUserMedia` (`:369`).

So the browser starts grabbing the USB cameras again while the recording worker is still inside its rest-pose return and has not yet reached `robot.disconnect()` (`record.py:1616-1632`). On macOS this is usually tolerated; on Linux/V4L2 exclusive-open it is a real conflict, and it also re-triggers the out-of-process AVFoundation enumeration the codebase is careful about. Symmetric risk: the previews are live again just as the user may press Start for the next session (the 2 s grace at `record.py:560-565` assumes the previews were released, and `handleStartRecording` only releases them 500 ms before the request, `CollectPanel.tsx:184-196`).

### N7 — P2 — A page reload or tab close destroys every saved episode of a fresh session behind a generic browser prompt

`useSessionExitGuard`'s `beforeunload` sets only `e.returnValue = ""` (`useSessionExitGuard.ts:97-102`), so browsers show their own generic "Leave site? Changes you made may not be saved" — the carefully written `leaveDiscardMessage(resume)` (`recordingExit.ts:40-44`) is used only for the in-app popstate confirm (`:140`). On `pagehide` the beacon POSTs `/stop-recording?discard=true` (`:114-119`), which sets `discard_requested` (`record.py:692-694`) and makes the worker `rmtree` the **whole** stamped directory including every already-saved episode (`record.py:620-628`, `_discard_session_dataset` at `:955-1007`).

Net: an accidental ⌘R twenty episodes into a session destroys all twenty, and the only warning the operator sees is a browser-generic dialog that says nothing about data. The semantic is deliberate and documented (`recordingExit.ts:4-9`), so this is filed P2 as a design/copy risk rather than a code bug — but of everything in this report it is the cheapest real-world way to lose a day of demonstrations, and the coordinator may want it higher.

### N8 — P2 — `RecordingRequest.test_mode` is declared, documented as "skip robot connection for testing", and never honoured

`record.py:291` declares `test_mode: bool = False  # Skip robot connection for testing`. `grep -rn "test_mode" makermodslab/` returns that single line — nothing reads it. An API user who sets it expecting a dry run gets a full hardware session: ports opened, calibration written to EEPROM (`record.py:1377-1378`), motion started. On a live-hardware repo a dead safety-shaped flag is worse than no flag.

### N9 — P2 (minor) — Terminal status logs and `print()`s on every poll after the session ends

`handle_recording_status()` emits a log line **and** a bare `print()` on every request once `session_ended` is true (`record.py:756-764`). The dialog freezes polling on failed/warning outcomes (`:305`) and exits on clean ones, so the loop is bounded in practice — but any client left polling (a second tab's phantom dialog, per R2) produces one log + one stdout line per second indefinitely. `/recording-status` is in the quiet-poll access filter (`server.py:177`) precisely because this path is noisy; these two statements bypass it.

## Part 3 — contradictions to the brief

1. **"recording frontend (`frontend/src/components/recording/`)"** — that directory contains only `RecordingSessionDialog.tsx` and `CameraConfiguration.tsx`. The request construction, the exit handling and the post-session handoff live in `frontend/src/components/studio/` (`CollectPanel.tsx`, `CollectHandoff.tsx`) and `frontend/src/contexts/StudioContext.tsx`. Three of my findings (N2, N6, and R2's reach) are only visible there, so I read them; flagging in case those files are also fenced to another agent this round.
2. **The brief frames R5/R6 as "the two former P0s" whose fixes may have landed.** On `main` neither fix exists in any form — this is not a regression from a landed fix, the `andrew` work simply never merged. I state this rather than assume.
3. **The excluded item.** I record a verdict for R12 (arm-identity **read-failure** fail-open) because it is a distinct code path from the excluded rows-3/4 warn-and-proceed severity question, and say nothing about the latter's severity.

## Part 4 — what I could not verify, and why

- **Whether R5/R6 fixes ever landed on `main` and were reverted.** I read `main`'s current code and `git log --oneline -- makermodslab/record.py` only; I did not walk other branches or full history (bounded-commands rule, and branch operations are forbidden by the brief). Verdict "absent from `main`" is certain; "never landed" is not asserted.
- **Runtime behaviour of anything past `robot.connect()`.** No server, no serial port, no camera, no motor — so R3 (leaked port/cameras on a leader-connect failure), R11 (rest-return outcomes), R12 (identity read failure) and N6 (camera hand-off timing) are verified as **code paths**, not as observed failures. N6 in particular depends on OS camera-exclusivity behaviour that differs between macOS and the Jetson.
- **Whether `82dffde`'s `record_loop` matches the installed v0.6.0 one.** N1's proof that `rerecord_episode` is never cleared by `record_loop` was read from the **installed** `lerobot 0.6.0` (`lerobot_record.py:285-287,508-511`), which is not `main`'s pinned commit (see N4). MakerMods Lab's own side of the trace (`record.py:1485-1540,1564-1605`) is unambiguous either way — the flag is cleared in exactly one place — but if `82dffde`'s `record_loop` cleared `rerecord_episode` on exit, N1 would not fire. I could not fetch that commit (no network). **Confirm against the pinned source before acting on N1.**
- **The exact `saved_episodes` value a real failed session leaves on disk.** R5's consequence chain is static; I did not run a session to observe a real terminal payload (the source list's process-local probe did, on `andrew`).
- **Frontend type/lint baseline.** Not run — no code was changed, and the brief set checks to none.
