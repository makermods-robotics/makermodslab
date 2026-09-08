# Teleoperation — RE-VERIFIED against `main` (23-07-26)

Re-verification date: July 23, 2026
Verified against: branch `main`, HEAD `c7d9f27` (`Merge pull request #4 from makermods-robotics/redesign`),
working tree as checked out (only `frontend/dist/**` + untracked `internal_docs/`, `debug/`, `.agents/` differ).

Source list: [`Teleoperation Bug List.md`](Teleoperation%20Bug%20List.md), forensically verified **2026-07-14 against
the `andrew` branch** (HEAD `257580f`). T-numbers per [`RANKING.md`](RANKING.md) §"Teleoperation":
T1 Done/page-leave latch · T2 rest-return discarded · T3 second-stop orphan · T4 cross-feature mutex ·
T5 identity fail-open · T6 motor power · T7 startup cleanup warning · T8–T11 the four P2s.

Rubric (unchanged): **P0** = hardware safety, data loss, silent private publication, or main path completely
broken. **P1** = wrong behavior in a common flow, real consequence, no workaround. **P2** = edge/cosmetic/polish.

**Open PRs against `main`, NOT merged** (so `main` still carries all three; each verified below independently
of whether the PR is correct or complete): **#5 = T7**, **#6 = T3**, **#7 = T4**.

**Excluded by brief:** the arm-identity guard *severity* question (`makermodslab/arm_identity.py`, decision-table
rows 3/4 warn-and-proceed). Known P0; a fix is committed on `fix/arm-identity-block-unverified`, not merged, so
`main` shows the old behavior. Context only — not re-reported here.

---

## Structural change since the audit — read this before the table

The redesign merge moved the live teleoperation surface. On `main`:

- The session is started from **`frontend/src/components/launchpad/RobotCorner.tsx:158-214`**
  (`handleTeleop` → `POST /move-arm`) and displayed by
  **`frontend/src/components/dialogs/TeleopDialog.tsx`** — a floating, **non-modal** window.
- **`frontend/src/pages/Teleoperation.tsx` is orphaned.** The `/teleoperation` route still exists
  (`App.tsx:41`) but **nothing in the app navigates to it** (repo-wide grep for `/teleoperation` finds only
  the route registration and the two status polls). Its two children —
  `components/control/VisualizerPanel.tsx` and `components/control/TeleopCameraPanel.tsx` — are reachable
  from nowhere else. This is the code cause of [`CONFIRMED 22-07-26.md`](CONFIRMED%2022-07-26.md) #2
  ("Teleop camera visualization gone"): the backend feed was never the issue, the panel simply isn't mounted
  by the live surface.
- `TeleopDialog.tsx:13-19` says the session state machine is "ported **verbatim** from
  pages/Teleoperation.tsx". It is — including every defect. **T1, T8, T9, T11 now exist twice.**
  Any fix must land in both files, or the dead page must be deleted.

Consequently every frontend `file:line` in the original list points at the dead copy. Verdicts below cite the
**live** copy first and note the dead twin.

---

## Part 1 — verdict table

| ID | Entry (original priority) | Verdict on `main` | Priority now |
|----|---------------------------|-------------------|--------------|
| T1 | Done/page-leave treat a stop *attempt* as an acknowledged stop (P1) | **STILL REAL** — and now duplicated into the live dialog | P1 |
| T2 | Rest-pose capture/return failures discarded, classified `outcome="ok"` (P1) | **STILL REAL** — code unchanged | P1 |
| T3 | Timed-out second stop orphans a live worker (P1) | **STILL REAL** (PR #6 open) — partially mitigated, reachability narrowed | P1 |
| T4 | Cross-feature hardware ownership incomplete/non-atomic (P1) | **STILL REAL** (PR #7 open) — **strengthened** by the non-modal window | P1 |
| T5 | Identity read failure fails open (P2, deliberate design) | **STILL REAL** as code fact; severity **EXCLUDED** by brief | — |
| T6 | Saved low-power setting can fail to apply (P2) | **NOT APPLICABLE — SUPERSEDED** — `apply_motor_power` removed; teleop deliberately runs stock torque | closed |
| T7 | Startup cleanup discards "torque may still be enabled" (P1, top-ranked) | **STILL REAL** (PR #5 open) — code unchanged | P1 |
| T8 | The interface removes the advertised second-stop "release now" (P2) | **STILL REAL — worse**: Done now closes the whole window | P1 (raised) |
| T9 | Route can display "Live Robot Data" with no session (P2) | **STILL REAL** — new manifestation inside the dialog | P2 |
| T10 | Saved-camera preview throws when `navigator.mediaDevices` absent (P2) | **STILL REAL in code**, **NOT REACHABLE from teleop** (panel orphaned); still reachable from three other surfaces | P2 |
| T11 | No immutable session identity for the rendered robot/cameras (P2) | **STILL REAL — strengthened**: the robot dropdown is clickable mid-session | P2 |

### T1 — Done and page-leave latch "stopped" on a stop *attempt* — **STILL REAL**

Live copy, `frontend/src/components/dialogs/TeleopDialog.tsx`:

1. `stoppedRef.current = true` is set **before** the POST (`TeleopDialog.tsx:91-93`).
2. A network/JSON failure is swallowed with no warning and the guard is never re-armed
   (`TeleopDialog.tsx:145-147`).
3. `handleDone` awaits that function and closes the window regardless (`TeleopDialog.tsx:195-198`);
   the ESC handler does the same without even awaiting (`TeleopDialog.tsx:185-190`).
4. The effect cleanup can no longer retry — the guard is latched (`TeleopDialog.tsx:174-177`).
5. `pagehide` writes the sessionStorage marker **before** firing the unacknowledged keepalive and never
   checks a response (`TeleopDialog.tsx:161-171`); `TeleopStopNotice.tsx:26-31` renders that
   attempted-stop marker as the affirmative "Teleoperation stopped … the arm returns to its starting
   position, then goes limp."

Dead twin, unchanged: `pages/Teleoperation.tsx:72-73`, `:125-127`, `:140-149`, `:159-162`.

**Impact unchanged**, plus: the guard is now latched by a *second* code path — the status poll sets
`stoppedRef.current = true` on a detected mid-loop death (`TeleopDialog.tsx:72`). That one is correct (the
session really is gone), but it means two very different states share one latch.

### T2 — Rest-return failures discarded, then classified as a clean stop — **STILL REAL**

Code is unchanged from the audited tree, only renumbered:

1. Capture failure returns `{}` silently (`makermodslab/rest_pose.py:68-73`), consumed without a check at
   `makermodslab/teleoperate.py:675-678`.
2. `return_to_rest_pose` still reports `no-pose` / `comm-error` / `settled` / `stalled` / `ceiling` /
   `cut-short` as unsuccessful (`rest_pose.py:82-97`, `:105-119`, `:165-192`).
3. `_return_one_follower_to_rest` only **logs** the reason (`teleoperate.py:246-252`);
   `_return_followers_to_rest` returns `None` (`teleoperate.py:255-288`).
4. The terminal outcome is built solely from `loop_error` / `last_cleanup_error`
   (`teleoperate.py:784-785`), so a stalled or comm-errored return classifies `outcome="ok"` and the arm's
   torque is released wherever it stopped (`teleoperate.py:772-777`).
5. Both UIs still promise the graceful landing (`TeleopDialog.tsx:109-114`, `TeleopStopNotice.tsx:26-31`).

Bimanual asymmetry is intact: one arm can fail while the other succeeds and the session still reports clean.

### T3 — Timed-out second stop orphans a live worker — **STILL REAL** (PR #6 open, not merged)

The defect is verbatim present:

- `handle_stop_teleoperation` sets `_release_now`, joins 5 s, and on `worker.is_alive()` **nulls the only
  reference to a still-running worker** (`teleoperate.py:871-883`).
- A later start rejects only an active session or a non-`None`, alive worker
  (`teleoperate.py:567-577`) — the orphan is invisible — and then **clears `_release_now`**
  (`teleoperate.py:591`), which is the very Event the orphan's rest-return is waiting on as its abort
  (`teleoperate.py:767` → `rest_pose.py:166-167`).
- The orphan's `finally` later overwrites the process globals of the *new* session:
  `teleoperation_active = False`, `releasing = False`, `current_robot/current_teleop = None`
  (`teleoperate.py:778-790`). Because the new worker's loop condition is `while teleoperation_active`
  (`teleoperate.py:705`), the orphan doesn't merely corrupt state — **it terminates the new session's
  control loop**, and the new worker then treats that as `stopped_normally = True` (`:747`), i.e. a
  user-requested stop.

Two changes since the audit *narrow reachability without fixing the defect*:

- `finish_pending_release` (`teleoperate.py:207-231`) is the correct shape — it refuses to null a worker that
  is still alive after the join (`:228-230`).
- The start path now explicitly rejects while a previous worker is alive (`teleoperate.py:570-577`).

So the orphan can now only be created through the 5 s-timeout branch at `:873-883`, which **no UI control can
reach** (see T8: there is no second-stop control, and `stoppedRef` prevents a second POST from the dialog).
On `main` this is an API-only path. Keep P1: the blast radius is unchanged, and the hardening in PR #6 should
also cover the `_release_now.clear()` at `:591` and the orphan's global writes at `:778-790`, not just the
null-ing.

### T4 — Cross-feature hardware ownership incomplete and not atomic — **STILL REAL, strengthened** (PR #7 open)

1. Separate per-module locks make check-then-set non-atomic: `teleoperate.py:102`, `record.py:215`,
   `rollout.py:114` — three unrelated `_state_lock`s. Each claims its own flag under only its own lock
   (`teleoperate.py:567-591`, `record.py:460-491`, `rollout.py:917-948`), so two simultaneous starts can both
   observe the other feature as inactive before either writes its flag.
2. **Inference still blocks only on `teleoperation_active`** (`rollout.py:918`) — not on `releasing`, not on
   `teleoperation_thread.is_alive()` — and, unlike recording, **never calls
   `_teleoperate.finish_pending_release()`** (repo-wide grep: the only callers are `record.py:457-458` and
   `teleoperate.py:564-565`). A normal stop clears `teleoperation_active` at `teleoperate.py:841`, *before*
   the arm is driven home and released, so inference is admitted during the entire releasing window and
   races the same serial ports. See also new finding **N3**, the same family on the recording side.
3. Manual calibration (`calibrate.py:232-236`), single auto-calibration (`auto_calibrate.py:207-215`), batch
   auto-calibration (`auto_calibrate.py:516-521`), wiggle (`wiggle.py:56-85`) and hand-motion port detection
   (`identify.py` — it only *reports* a busy port at `:117`) still check **only their own manager**. Repo-wide
   grep for `teleoperation_active` outside `teleoperate.py` returns hits in `record.py` and `rollout.py` only;
   no `calibration_in_progress` / `hardware_session_*` guard exists anywhere. The `c737a30` mutex documented
   in `logs/PROJECT.md` is **still not on this branch** — the 2026-07-14 verification note carries forward
   unchanged.
4. `SingleTabGuard.tsx:120-140` still only overlays the losing tab; a teleop session running in that tab keeps
   running and its Done button is now *underneath* a `z-[9999]` backdrop, i.e. unreachable.

**New strengthening (see N4):** the teleop window is a plain `fixed` div with **no backdrop and no focus
trap** (`TeleopDialog.tsx:204-211`). Every other control in the app stays clickable during a live session,
including "Robot settings" → calibration, which is the unguarded feature in point 3. What was a theoretical
race in the audited tree is now a two-click path.

### T5 — Identity read failure fails open — **STILL REAL** (code fact); severity excluded

`verify_arm` still returns `(None, None)` — no refusal, no warning — when `Present_Position` or
`Homing_Offset` cannot be read (`makermodslab/arm_identity.py:285-288`, docstring `:268-270`), and
`verify_devices` therefore treats the arm as verified (`arm_identity.py:391-411`). Teleoperation then writes
calibration and configures motors in both paths (`teleoperate.py:527-530`, `:645-653`).

Per the brief, the *severity* of the arm-identity guard is out of scope for this pass. No new assessment;
noted only so the row isn't silently dropped.

### T6 — Saved low-power setting can fail to apply — **NOT APPLICABLE — SUPERSEDED**

The mechanism no longer exists. `apply_motor_power` is gone from the tree (repo-wide grep: zero hits).
`makermodslab/motor_power.py:16-21` now states the policy explicitly: the per-robot percentage is the
**auto-calibration** drive torque only, and "Regular sessions — teleoperation, recording, skill runs —
deliberately run at stock LeRobot torque". Teleop start correspondingly calls `reset_torque_limit` +
`clear_goal_velocity` (`teleoperate.py:658,663` single; `:535,539` bimanual) instead of applying a cap, and
`TeleoperateRequest` (`teleoperate.py:291-308`) no longer carries a `motor_power` field — `RobotCorner`
doesn't send one (`RobotCorner.tsx:161-179`).

The residual "per-motor write failure warns but does not abort" shape survives in `_for_each_motor`
(`motor_power.py:120-135`), but its **direction is inverted**: a failed `reset_torque_limit` leaves the motor
at whatever *lower* cap a previous auto-calibration stamped (sluggish, safe), not at full power. The audit's
hazard — "immediate follower motion at 100%" — cannot occur on this path. Close the entry.

⚠️ **Contradiction with [`CONFIRMED 22-07-26.md`](CONFIRMED%2022-07-26.md) #3** ("Torque / motor-power slider
gone … UI door missing"): on `main` the slider is present and wired —
`RobotConfigDialog.tsx:1215-1245` (percent↔raw conversion, `DEFAULT_MOTOR_POWER = 38`),
`:1269` (dirty tracking), `:1276` (`patch.motor_power`). Either #3 was fixed by the tier-1 UX commits
(`fc85b40` / `e7f7d18`) or it referred to a different surface. Coordinator call.

### T7 — Startup cleanup discards "torque may still be enabled" — **STILL REAL** (PR #5 open, not merged)

1. `_safe_disconnect` still builds the safety text including "TORQUE MAY STILL BE ENABLED — the arm can stay
   rigid; unplug its power to release it" (`teleoperate.py:444-469`).
2. Setup has already written calibration and run `configure()` before several later operations can fail
   (single: `teleoperate.py:643-663`, then `capture_rest_pose` at `:675-678`; bimanual: `:527-539`).
3. **Both cleanup call sites discard the return value**: `_connect_bimanual`'s except path
   (`teleoperate.py:542-545`) and the outer start except path (`teleoperate.py:811-813`).
4. The response is `{"success": False, "message": str(e)}` (`teleoperate.py:820`) — the original setup error
   only. `last_cleanup_error` is left at the `None` the start path set (`:587`), so
   `/teleoperation-status` shows nothing either (`:916`).
5. The startup-failure path **never runs the per-motor `force_disable_torque` hardening** — that exists only
   in the worker's `finally` (`teleoperate.py:772-773`), which a failed start never reaches.

Unchanged from the audit, and still the top-ranked teleop P1.

### T8 — The interface removes the advertised second-stop "release now" — **STILL REAL, and worse (raise to P1)**

The backend still tells the operator to do something the UI cannot do: `"…then goes limp. Press Stop again to
release it now."` (`teleoperate.py:861-864`), and the second-stop handler still exists (`:867-894`).

On `main` the live window has exactly one control — Done (`TeleopDialog.tsx:217-223`) — and it **closes the
window** (`:197`). ESC does the same (`:185-190`). There is no second-stop control anywhere, and after Done
there is no teleop surface at all. See **N2**: this is now the same defect as "the UI disappears while the arm
is still energized and moving", which is why I raise it from P2 to P1.

### T9 — "Live Robot Data" while no session exists — **STILL REAL**

- The status poll transitions only on an inactive `failed` / `ran_with_warning` outcome
  (`TeleopDialog.tsx:65-69`; dead twin `pages/Teleoperation.tsx:47-51`); inactive `ok` / `None` is ignored.
- The badge is driven purely by socket connectivity (`UrdfViewer.tsx:352-378`; compact mode renders a green
  dot titled "Live robot data", `:355-360`).
- `/ws/joint-data` still accepts any connection regardless of teleoperation state
  (`server.py:806-836`), and `useRealTimeJoints` reconnects with backoff independently of session state
  (`useRealTimeJoints.ts:53-118`).

New manifestation: because the dialog stays open after a mid-loop death, the header can show a **green "live"
dot next to a red "Teleoperation failed" banner** — the dot never goes red, since the WebSocket is still up.
The original bookmark/direct-route scenario now only applies to the orphaned `/teleoperation` page.

### T10 — Saved-camera preview can throw on an absent camera API — **STILL REAL in code, NOT REACHABLE from teleop**

`useCameraStream` still dereferences `navigator.mediaDevices.addEventListener` unguarded in its first effect
(`hooks/useCameraStream.ts:43`) and `getUserMedia` at `:60`. With `mediaDevices` absent (plain-HTTP LAN
origin, unsupported browser) the effect throws before `hasError` can be set, and no error boundary wraps the
callers.

**Teleop reachability is gone**: `TeleopCameraPanel` is mounted only by the orphaned page
(`pages/Teleoperation.tsx:211`), and `TeleopDialog` renders no camera panel at all. The hook itself is still
live on three other surfaces — `components/recording/CameraConfiguration.tsx:369`,
`components/studio/DeployPanel.tsx:89`, `components/landing/InferenceModal.tsx:59` — so the defect should be
re-filed against those lists rather than closed.

### T11 — No immutable session identity — **STILL REAL, strengthened**

- Start snapshots one record into the request (`RobotCorner.tsx:161-179`).
- The window derives its title, arm count and layout from the live, mutable `selectedRecord`
  (`TeleopDialog.tsx:25-26`, `:207`, `:215`, `:266-289`).
- `useRobots` records are shared, refreshable module state.
- `/teleoperation-status` still returns **no session descriptor** — no id, robot name, mode, or camera
  snapshot (`teleoperate.py:899-927`).

Strengthened: because the window is non-modal, `RobotCorner`'s robot dropdown
(`RobotCorner.tsx:270-351`) stays clickable **behind and around** the live session window. Selecting another
robot re-renders the open session window with the other robot's name and arm count while the hardware keeps
running the original request. In the audited tree this needed a second client or a direct API call.

---

## Part 2 — new findings

### P1 — N1. Teleoperation is the only live-hardware surface with no page-leave guard

Recording, inference and calibration all route their leave protection through
`hooks/useSessionExitGuard.ts` — native `beforeunload` prompt, back-button `confirm`, keepalive beacon, and a
single `handledRef` latch (`useSessionExitGuard.ts:90-160`). Consumers:
`RecordingSessionDialog.tsx:192`, `InferenceSessionDialog.tsx:106`, `RobotConfigDialog.tsx:595`.

**Teleop does not use it.** `TeleopDialog.tsx:159-178` hand-rolls only the `pagehide` half:

- **No `beforeunload`** → a reload, typed URL, or tab close during a live session proceeds with **no
  confirmation**, on the one surface where the arm is being driven by a human at that instant.
- **No `popstate` guard** → the browser Back button leaves without a prompt.
- The hand-rolled handler also lacks the guard's `handledRef` discipline, which is the mechanism that would
  have prevented T1's latch-before-ack (the guard latches *inside* the leave vector, and the caller marks
  deliberate exits via `markHandled()`).

**Impact:** the operator loses the one confirmation the other three hardware surfaces give them, and the only
stop that fires is the unacknowledged keepalive of T1. **Fix note:** adopting `useSessionExitGuard` here
closes N1 and the page-leave half of T1 in one change.

### P1 — N2. Done closes the only session window while the arm is still energized and moving

`handle_stop_teleoperation`'s first stop returns immediately with `releasing: true` while the worker drives
the follower home for up to `RETURN_CEILING_S = 10 s` with torque **on** (`teleoperate.py:856-865`,
`rest_pose.py:54`, `teleoperate.py:759-767`). The UI:

1. `handleDone` awaits the stop and then calls `onOpenChange(false)` (`TeleopDialog.tsx:195-198`) — the
   window unmounts (`:202`) while the arm is still moving.
2. Nothing on the Launchpad polls `/teleoperation-status`; the only two pollers are the teleop window itself
   and the dead page (repo-wide grep). `releasing` is never rendered anywhere — the flag exists in the
   payload (`teleoperate.py:913`) and is read only to pick a toast (`TeleopDialog.tsx:106`).
3. The Teleop button re-enables the instant the request resolves — `teleopDisabledReason` depends only on
   `selectedRecord` / `is_clean` and `teleopStarting` (`RobotCorner.tsx:216-220`, `:360`).
4. Pressing it during the release calls `finish_pending_release()` (`teleoperate.py:564`), which
   **sets `_release_now`** — cutting the rest return short (`rest_pose.py:166-167`) so the worker proceeds
   straight to `force_disable_torque` (`teleoperate.py:772`). **The arm is de-energized mid-motion, part-way
   home, with no warning.**

So the advertised "press Stop again to release it now" (T8) is not merely missing — it has been replaced by
an *unlabelled* release-now control that looks like "start a new session". Ranked P1 (arm state vs. UI state
diverge on the ordinary Done path; the recovery affordance is a hardware action in disguise).

### P1 — N3. `finish_pending_release()`'s failure return is ignored by both start paths

`finish_pending_release` returns `False` when the previous worker did not exit within the 10 s join
(`teleoperate.py:228-231`; recording twin `record.py:251-260`). **Neither caller checks it:**

- `makermodslab/teleoperate.py:564-565` calls both, discards both, then checks only
  `_record.recording_active` (`:578`).
- `makermodslab/record.py:457-458` calls both, discards both, then checks only
  `_teleoperate.teleoperation_active` (`:470`).

The teleop→record direction happens to be safe: `recording_active` stays `True` until the very end of the
recording worker (`record.py:638`), after its hardware release. **The record→teleop direction is not.**
`teleoperation_active` is cleared by the *stop handler* (`teleoperate.py:841`), long before the worker
finishes the rest return and the torque release (`:759-790`). Therefore:

> teleop stopped → worker still returning/releasing → operator starts a recording →
> `_teleoperate.finish_pending_release()` returns `False` (worker wedged past 10 s) → **ignored** →
> `_teleoperate.teleoperation_active` is `False` → recording proceeds to open the same leader/follower
> ports the live teleop worker still holds.

Teleop's own start does check the live-worker case (`:570-577`); recording and inference do not. Same family
as T4 point 2, but a distinct, cheaply-fixable defect: **use the return value** (or have recording/inference
consult `teleoperate.teleoperation_thread`/`releasing`, as teleop consults its own).

### P1 — N4. The live teleop window is non-modal: the rest of the app stays interactive during a session

`TeleopDialog` renders a bare `fixed … z-50` div with `role="dialog"` and **no overlay/backdrop element and
no focus trap** (`TeleopDialog.tsx:204-211`) — deliberately "not a window dialog" per its own docstring
(`:13-16`). Unlike the Radix dialogs used elsewhere in the app, nothing blocks interaction with what's behind
it.

While an arm is being teleoperated a click away, the operator can:

- open **Robot settings** → run a manual or automatic calibration on the same ports —
  **the unguarded feature of T4 point 3** (`RobotCorner.tsx:255`, `calibrate.py:232-236`,
  `auto_calibrate.py:207-215`);
- **switch, rename, or delete** the active robot record (`RobotCorner.tsx:302-349`) — the deletion path is
  not gated on any session state, and switching triggers T11;
- open the studio and use the second `RobotCorner` instance (both are mounted simultaneously —
  `Launchpad.tsx:48` and `StudioOverlay.tsx:74`), whose Teleop button is also enabled.

Recording and inference sessions get real modal dialogs. This is the UI half of T4 and turns a race into a
routine mis-click.

### P2 — N5. The teleop state machine is duplicated, and `/teleoperation` is a live footgun

`TeleopDialog.tsx:13-19` documents the state machine as "ported verbatim" from `pages/Teleoperation.tsx`.
Both files now carry the same stop latch, the same swallowed catch, the same pagehide handler and the same
status poll (compare `TeleopDialog.tsx:55-178` with `pages/Teleoperation.tsx:38-157`). Consequences:

- Every fix to T1/T8/T9/T11 must be applied twice or the dead page must go.
- The dead page is still routable. Navigating to `/teleoperation` **unmounts the Launchpad**, hence
  `RobotCorner`, hence `TeleopDialog` — whose cleanup fires a real stop (`TeleopDialog.tsx:174-177`). So a
  URL typo silently ends a live session, and the page you land on then shows the T9 "Live Robot Data" badge
  for the session it just killed.
- `TeleopCameraPanel` / `VisualizerPanel` are dead code carrying their own findings (T10).

### P2 — N6. The 13 s post-stop cleanup check reads process-global state a newer session clears

After a `releasing: true` response the UI schedules a single status re-check 13 s later
(`TeleopDialog.tsx:119-138`) and toasts `last_cleanup_error` if present. But `last_cleanup_error` is
process-global and **cleared by the next start** (`teleoperate.py:587`), and the timer keeps running after the
window closes (by design — "the toast store is global"). Two failure modes:

- Start a new session within 13 s → the old session's genuine "TORQUE MAY STILL BE ENABLED" warning is
  erased before the check reads it, and the operator never sees it.
- The check can equally read a *newer* session's cleanup error and attribute it to the closed one.

A fixed 13 s also has no relationship to the actual release time (which is bounded by the 10 s return
ceiling *plus* the per-motor torque disable with `num_retry=5`, `teleoperate.py:430`).

### P2 — N7. The rest-return join is unbounded, and part of the return sits outside both the ceiling and the abort

`_return_followers_to_rest` starts one thread per follower and joins each **with no timeout**
(`teleoperate.py:285-288`). The 10 s `RETURN_CEILING_S` bounds only `_run_return_loop`
(`rest_pose.py:165`); everything else on the path is unbounded and un-abortable:

- the pre-loop `bus.write("Goal_Velocity", …)` per motor plus `sync_write("Goal_Position", …)`
  (`rest_pose.py:111-113`) — before any `abort_event` check;
- `time.sleep(RETURN_SETTLE_S)` at `rest_pose.py:159`, again before the first abort check at `:166`;
- `_restore_goal_velocity`'s `sync_write` in the `finally` (`rest_pose.py:129`, `:144`).

If a follower bus wedges there, the worker never reaches `force_disable_torque` (`teleoperate.py:772`) and
**the arm stays energized indefinitely** — while the second stop's 5 s join expires and takes the T3 orphan
branch (`:872-883`). Bounded in practice by pyserial/SDK packet timeouts, so P2 rather than P1, but it is the
mechanism that makes T3's "worker still alive after 5 s" reachable at all.

### P2 — N8. The stop path mutates shared state without `_state_lock`

`handle_stop_teleoperation` reads and writes `teleoperation_active` and `teleoperation_thread` with no lock
at all (`teleoperate.py:836-896`), while `handle_start_teleoperation` claims exactly those globals **under**
`_state_lock` (`teleoperate.py:567-591`). Both endpoints are plain `def` handlers
(`server.py:503-512`), so FastAPI runs them on the threadpool — genuinely concurrent.

Concrete window: start assigns `teleoperation_thread` at `:792` but does not call `.start()` until `:795`. A
stop landing in between sees `teleoperation_active is True` and `worker.is_alive() is False`, takes the
"no worker" branch, sets `teleoperation_thread = None` and returns **"Teleoperation stopped
successfully"** (`:843-854`) — while `handle_start_teleoperation` goes on to launch a worker that no longer
has a tracked reference and whose loop exits on its first condition check (`:705`). The start still returns
`{"success": True}` (`:797-807`) and `RobotCorner` opens the session window for a session that is already
tearing itself down (`RobotCorner.tsx:181-197`).

A related, more likely variant: a stop arriving during the multi-second synchronous connect (between `:586`
and `:795`) makes the start report success for a session that never runs. Neither is user-reachable through
the current UI, but both are trivially reachable through the API and through a double-click on Teleop plus a
stale keepalive.

### P2 — N9. The WebSocket broadcast queue is unbounded and its overflow branch is dead

`ConnectionManager.broadcast_queue = queue.Queue()` has **no `maxsize`** (`server.py:254`), so
`put_nowait` can never raise and the "Broadcast queue is full, dropping data" branch
(`server.py:381-382`) is unreachable. Meanwhile the drain is bounded per client:
`_send_to_all_connections` waits `future.result(timeout=1.0)` for each pending send
(`server.py:368-374`).

Teleop produces at 20 Hz (`teleoperate.py:703`, `:710`) plus the ~1 Hz current sample. One wedged or
suspended client (a backgrounded tab whose event loop is throttled) can make the drain slower than the
producer; the queue then grows without bound for the length of the session, and every client receives
progressively **staler** joint data — the 3-D arm lags the physical arm with no indication, and the "Live
Robot Data" badge stays green (T9). A bounded queue with drop-oldest is the right shape for a
latest-value telemetry stream.

### P2 — N10. `strict=False` hides a telemetry/bus count mismatch

`telemetry_targets = list(zip(_device_buses(robot), ["left_", "right_"] if is_bimanual else [""], strict=False))`
(`teleoperate.py:689-691`). If `_device_buses` ever returns a count other than 1 (single) or 2 (bimanual) —
e.g. a BiSO device where one sub-arm's bus failed to attach — `zip` silently truncates and that follower's
current is simply never sampled, producing an all-zero peak that reads exactly like the
"firmware leaves the register at 0" case the module docstring warns about (`teleoperate.py:133-135`).
Cheap fix: `strict=True`, or build the pairs explicitly.

### P2 — N11. A failed start leaves no trace in the status payload

The start except path sets `teleoperation_active = False` and clears the device globals
(`teleoperate.py:814-816`) but never populates `last_session_outcome` / `last_session_error` — which the
start path had just cleared to `None` (`:588-589`). `/teleoperation-status` therefore reports
`outcome: None, error: None, last_cleanup_error: None` after a start that failed *after* connecting and
configuring the follower (`teleoperate.py:920-925`). The only record is the HTTP response body, which the
frontend renders as a transient toast (`RobotCorner.tsx:198-204`). Combined with **T7**, a startup failure
whose cleanup also failed leaves an energized arm and **zero** persistent server-side evidence.

---

## Contradictions with this brief / with the existing lists

1. **The brief's frontend paths are stale.** It names `frontend/src/pages/Teleoperation.tsx`,
   `frontend/src/components/control/` and `frontend/src/components/dialogs/TeleopDialog.tsx` as
   "teleop frontend". On `main` only `TeleopDialog.tsx` (plus `RobotCorner.tsx`, which the brief does not
   name and which owns the start request) is live; the page and the whole `control/` directory are
   unreachable. I verified all of them and labelled each accordingly.
2. **T6's premise no longer exists.** The brief asks for a verdict on every entry; T6 is verdicted
   NOT APPLICABLE-SUPERSEDED rather than STILL REAL/FIXED, because the code it describes
   (`apply_motor_power`, `motor_power` in `TeleoperateRequest`) was deleted, not fixed.
3. **`CONFIRMED 22-07-26.md` #3 ("Torque / motor-power slider gone") does not reproduce on `main`** — the
   slider is present and wired (`RobotConfigDialog.tsx:1215-1276`). Flagged for the coordinator; I did not
   change that file.
4. **`CONFIRMED 22-07-26.md` #2 ("Teleop camera visualization gone") is confirmed and now has a root
   cause**: the camera panel lives only on the orphaned `/teleoperation` page
   (`pages/Teleoperation.tsx:211`); the live window renders no cameras.
5. **`CONFIRMED 22-07-26.md` #15 ("Power telemetry not implemented")**: the backend telemetry exists and is
   broadcast (`teleoperate.py:141-178`, `:734-737` sends `follower_currents_ma`), and the session summary is
   logged at INFO (`:756-758`). Repo-wide grep finds **no frontend consumer** of
   `follower_currents_ma` — so "not implemented" is accurate for the UI, not for the backend.
6. **PR overlap:** #5/#6/#7 target T7/T3/T4. I verified each against `main`'s code independently and did not
   read the PRs (no network per the brief). Note that N3 and N4 sit inside T4's family and N7 inside T3's —
   a PR that closes only the entry as originally written will leave those open.

## Not verified / cannot verify statically

- **Whether PRs #5/#6/#7 actually fix their entries.** No network access was used; the verdicts above
  describe `main`'s code only.
- **Real-world timing of N7** (whether a wedged Feetech bus write can block past the SDK packet timeout) and
  of N9 (whether a throttled tab actually drives the queue backlog). Both need a bench session with a
  deliberately stalled client / bus — not statically decidable.
- **N8's races and T3's orphan** are shape-verified from the code but need threaded tests (mock worker whose
  `is_alive()` stays true past the join; barriers around two start handlers) to demonstrate. No such test
  exists today — `tests/test_teleoperate.py` covers `finish_pending_release`'s three branches
  (`:447-493`) and the stale-`_release_now` regression (`:947`), but nothing asserts single-ownership across
  features.
- **T2's operator-visible impact** (how often a return actually stalls) is a hardware question; the code path
  is unambiguous.
- **Whether the `/teleoperation` route is intentionally kept** (e.g. for an in-flight feature) or is merge
  debris — a product/coordinator call, not a code fact.
- **`useRobots` refresh semantics under T11** were read only to the extent of confirming `selectedRecord` is
  shared mutable module state; I did not trace every refresh trigger.

## Audit safety

- No production, test, or frontend file was modified. This report is the only file written.
- No git state was changed: no commit, add, push, branch switch, stash, reset, or clean. Only
  `git status --short`, `git branch --show-current`, and a bounded `git log --oneline -15` were run.
- No server was started, no serial port opened, no camera opened, no motor energized, no test suite run.
- No network, Hub, or credential access.
