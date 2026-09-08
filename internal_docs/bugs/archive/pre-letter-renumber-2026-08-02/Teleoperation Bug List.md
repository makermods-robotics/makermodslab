# Teleoperation Bug List

*Citations synced 2026-07-26 against `main` @ `ce42158`. **Only `file:line` coordinates were changed** — no prose, verdicts, severities, entry IDs or numbering were altered. References whose construct no longer exists at HEAD are marked inline `⚠ UNRESOLVED @ HEAD ce42158`.*

> **Re-verified against `main` @ `c7d9f27` on 2026-07-23.** This list folds the 23-07-26 re-audit into the
> original. It **supersedes** the pre-reverification version, archived at
> [`archive/Teleoperation Bug List (pre-reverify).md`](archive/Teleoperation%20Bug%20List%20%28pre-reverify%29.md);
> the standalone re-audit is archived at
> [`archive/Teleoperation REVERIFIED 23-07-26.md`](archive/Teleoperation%20REVERIFIED%2023-07-26.md).
> Cross-module current-state overview: [`CURRENT 23-07-26.md`](CURRENT%2023-07-26.md).
>
> **Context you need for every verdict below:**
> - The original entries were forensically verified **2026-07-14 against the `andrew` branch** (HEAD `257580f`).
>   Re-verification target is **`main` @ `c7d9f27`**. **`andrew`-branch fixes are NOT on `main`.**
> - **The redesign moved the live teleoperation surface** (see "Structural change" below). The session is now
>   started from `RobotCorner.tsx` (`POST /move-arm`) and displayed by the non-modal `TeleopDialog.tsx`;
>   `pages/Teleoperation.tsx` and the whole `components/control/` directory are **orphaned** — every frontend
>   `file:line` in the original list points at that dead copy. Verdicts cite the **live** copy first and note
>   the dead twin.
> - **PRs #5–#11 are CLOSED with `merged=no`.** Three target entries here — **#5 = T7**, **#6 = T3**,
>   **#7 = T4** — none merged, so `main` carries all three.
> - **Numbering:** the original used descriptive P1/P2 headings; `RANKING.md` §Teleoperation numbers them
>   **T1–T11** (T1 Done/page-leave latch · T2 rest-return discarded · T3 second-stop orphan · T4 cross-feature
>   mutex · T5 identity fail-open · T6 motor power · T7 startup cleanup warning · T8–T11 the four P2s). Those
>   IDs are carried on each heading; new findings use **N1–N11**. Nothing is renumbered.
> - Analysis is **purely static** — no server, no serial/camera open, no motor energized, no Hub call. No code
>   was modified.

Audit date: July 14, 2026 · Audited commit: `518ca56` (`min_stable`) · Re-verified: July 23, 2026 against `main` @ `c7d9f27`.

## Verification note (2026-07-14)

Every entry below was re-verified on 2026-07-14 against the current working tree of branch `andrew` (HEAD `257580f`, uncommitted changes included). One structural finding: the calibration↔session bidirectional mutex documented as deliberate design in `logs/PROJECT.md` (commit `c737a30`, "calibration and hardware sessions are mutually exclusive") is **NOT present in this branch** — `c737a30` is not an ancestor of HEAD, and no `calibration_in_progress` / `hardware_session_in_progress` guard exists under `makermodslab/`. Calibration (manual, single, batch) and wiggle are genuinely unguarded against live teleop/record/inference sessions, which strengthens the cross-feature-ownership entry (T4). *(Re-verification 2026-07-23: still absent on `main` — see T4.)*

Pathway baseline: `internal_docs/complete functionalities/Teleoperation Interface Pathways.md`.

Scope: saved robot selection and request construction, single/bimanual startup, arm identity, calibration application, follower motor power, the live control loop, WebSocket/status state, saved-camera previews, normal and emergency stop, rest-pose return, torque release, page leave, cross-feature ownership, and recovery.

Camera design rule used by this audit: teleoperation must use the saved robot camera configuration. A missing or invalid saved device ID must be surfaced to the operator; it must not be silently replaced with a discovered camera.

## Structural change since the audit — read this before the entries (from 2026-07-23 re-verification)

The redesign merge moved the live teleoperation surface. On `main`:

- The session is started from **`frontend/src/components/launchpad/RobotCorner.tsx:158-214`** (`handleTeleop` → `POST /move-arm`) and displayed by **`frontend/src/components/dialogs/TeleopDialog.tsx`** — a floating, **non-modal** window.
- **`frontend/src/pages/Teleoperation.tsx` is orphaned.** The `/teleoperation` route still exists (`App.tsx:41`) but **nothing in the app navigates to it**. Its two children — `components/control/VisualizerPanel.tsx` and `components/control/TeleopCameraPanel.tsx` — are reachable from nowhere else. This is the code cause of `CONFIRMED 22-07-26.md` #2 ("Teleop camera visualization gone"): the backend feed was never the issue, the panel simply isn't mounted by the live surface.
- `TeleopDialog.tsx:13-19` says the session state machine is "ported **verbatim** from pages/Teleoperation.tsx". It is — including every defect. **T1, T8, T9, T11 now exist twice.** Any fix must land in both files, or the dead page must be deleted (see N5).

## Confirmed bugs

### P0 — none in the original audit

No defect was classified as P0. Several P1 findings can leave a robot-driving worker active or allow conflicting hardware ownership, but each requires a failure or race rather than occurring on every ordinary session. *(Re-verification note: the arm-identity guard severity is a known P0 but is EXCLUDED by the re-verification brief; a fix is committed on `fix/arm-identity-block-unverified`, not merged, so `main` shows the old behavior — see T5.)*

### T1 · P1 — Done and page-leave flows treat a stop attempt as if it were an acknowledged stop

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL — and now duplicated into the live dialog.** Live copy,
> `TeleopDialog.tsx`: `stoppedRef.current = true` set **before** the POST (`:91-93`); a network/JSON failure
> swallowed with no warning, guard never re-armed (`:145-147`); `handleDone` awaits and closes the window
> regardless (`:195-198`), ESC does the same without awaiting (`:185-190`); effect cleanup can't retry — guard
> latched (`:174-177`); `pagehide` writes the sessionStorage marker **before** the unacknowledged keepalive
> and never checks a response (`:161-171`), and `TeleopStopNotice.tsx:26-31` renders that attempted-stop marker
> as the affirmative "Teleoperation stopped …". Dead twin unchanged: `pages/Teleoperation.tsx:72-73`,
> `:125-127`, `:140-149`, `:159-162`. New wrinkle: the guard is now also latched by the status poll on a
> detected mid-loop death (`TeleopDialog.tsx:72`) — correct there, but two very different states share one
> latch. **Adopting `useSessionExitGuard` (see N1) closes the page-leave half of this in one change.**

**Verdict:** CONFIRMED — `stoppedRef` latched before the POST and never reset on failure (`frontend/src/pages/Teleoperation.tsx:72-73,125-127`); `pagehide` writes the stopped marker before the unacknowledged keepalive (`Teleoperation.tsx:141-149`) and `TeleopStopNotice` renders it as an affirmative stop (`frontend/src/components/TeleopStopNotice.tsx:26-30`).
**Priority:** P1 — the UI affirmatively reports the arm stopped/released while the session may still be live after a transient stop failure (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. `stopTeleoperation()` sets `stoppedRef.current = true` before sending `POST /stop-teleoperation` (`frontend/src/pages/Teleoperation.tsx:71-78`).
2. A network/JSON failure is swallowed without a warning or resetting the guard (`frontend/src/pages/Teleoperation.tsx:119-128`).
3. The Done handler navigates home after awaiting that function even when the function swallowed a failure (`frontend/src/pages/Teleoperation.tsx:159-162`).
4. Unmount cleanup then cannot retry because the guard is already latched (`frontend/src/pages/Teleoperation.tsx:139-157`).
5. Browser-level leave writes the "stopped" session-storage marker before firing a best-effort keepalive request and never checks a response (`frontend/src/pages/Teleoperation.tsx:139-150`). `TeleopStopNotice` later turns that attempted-stop marker into the affirmative message "Teleoperation stopped" (`frontend/src/components/TeleopStopNotice.tsx:17-30`).

**Impact**

A transient backend/network failure during Done can leave the control loop running after the operator has been navigated away from the only teleoperation controls. A dropped unload request can likewise leave teleoperation active while the next page explicitly tells the operator it stopped and the arm is returning/releasing.

**Safe reproduction/reasoning**

Mock `fetchWithHeaders` to reject. `stoppedRef` remains true, `stopTeleoperation()` resolves through its catch, and `handleGoBack()` navigates. For the unload branch, the marker is committed synchronously before the unacknowledged request. No hardware is required.

### T2 · P1 — Rest-pose capture and return failures are discarded, then classified as a clean stop

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL — code unchanged, only renumbered.** Capture failure returns
> `{}` silently (`rest_pose.py:68-73`), consumed without a check (`teleoperate.py:673-676`);
> `return_to_rest_pose` still reports `no-pose`/`comm-error`/`settled`/`stalled`/`ceiling`/`cut-short` as
> unsuccessful (`rest_pose.py:82-97`, `:105-119`, `:165-192`); `_return_one_follower_to_rest` only **logs** the
> reason (`teleoperate.py:246-252`), `_return_followers_to_rest` returns `None` (`:255-288`); the terminal
> outcome is built solely from `loop_error`/`last_cleanup_error` (`:782-783`), so a stalled/comm-errored return
> classifies `outcome="ok"` and torque is released wherever the arm stopped (`:770-775`). Bimanual asymmetry
> intact. Both UIs still promise the graceful landing (`TeleopDialog.tsx:109-114`, `TeleopStopNotice.tsx:26-31`).

**Verdict:** CONFIRMED — the per-arm return outcome is only logged (`makermodslab/teleoperate.py:250,255`), `_return_followers_to_rest` returns nothing (`teleoperate.py:288-291`), and the terminal outcome is built solely from `loop_error`/`last_cleanup_error` (`teleoperate.py:784-785`), so a `settled`/`stalled`/`comm-error` return classifies as `outcome="ok"`.
**Priority:** P1 — the graceful-landing promise can fail silently and torque is released wherever the arm stalled, with no warning in the stop response or terminal status (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. Failure to capture a starting pose returns `{}` and does not fail or warn the start response (`makermodslab/rest_pose.py:60-73`; capture is consumed at `makermodslab/teleoperate.py:671-680`).
2. `return_to_rest_pose()` explicitly reports `no-pose`, `comm-error`, `settled`, `stalled`, `ceiling`, and `cut-short` as unsuccessful outcomes (`makermodslab/rest_pose.py:76-129`, `makermodslab/rest_pose.py:149-192`).
3. `_return_one_follower_to_rest()` only logs the returned reason; `_return_followers_to_rest()` returns no result to the worker (`makermodslab/teleoperate.py:237-291`).
4. The worker's terminal error is built only from the control-loop exception or torque/disconnect cleanup. A failed rest return is absent, so a normal stop with successful torque release becomes `outcome="ok"` (`makermodslab/teleoperate.py:747-787`; `makermodslab/utils/errors.py:61-77`).
5. The UI tells the operator the arm returns to its starting position and then goes limp (`frontend/src/pages/Teleoperation.tsx:87-123`).

**Impact**

An arm can fail to reach its safe starting pose, then have torque released wherever it stalled or lost communication, with no warning in the stop response or terminal status. In bimanual mode one arm can fail while the other succeeds and the whole session still reports clean.

**Safe reproduction/reasoning**

Existing tests prove all failure reasons and explicitly prove that one arm's raised return error is swallowed (`tests/test_teleoperate.py:612-729`, `tests/test_teleoperate.py:874-898`). There is no worker-level assertion that any of those results affect `last_session_error` or `outcome`.

### T3 · P1 — A timed-out second stop orphans a live worker and permits a conflicting new session

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL (PR #6 CLOSED-unmerged) — partially mitigated, reachability
> narrowed.** The defect is verbatim present: `handle_stop_teleoperation` sets `_release_now`, joins 5 s, and
> on `worker.is_alive()` **nulls the only reference to a still-running worker** (`teleoperate.py:869-881`); a
> later start rejects only an active session or a non-`None` alive worker (`:565-575`) — the orphan is
> invisible — then **clears `_release_now`** (`:589`), the very Event the orphan's rest-return waits on as its
> abort (`:765` → `rest_pose.py:166-167`); the orphan's `finally` overwrites the *new* session's globals
> (`:778-790`), and because the new loop condition is `while teleoperation_active` (`:705`) it **terminates the
> new session's control loop**, which the new worker treats as a user stop (`:747`). Two changes narrow
> reachability without fixing it: `finish_pending_release` (`:207-231`) refuses to null a still-alive worker
> (`:228-230`), and the start path explicitly rejects while a previous worker is alive (`:570-577`). So the
> orphan is now only creatable via the 5 s-timeout branch (`:873-883`), which **no UI control reaches** (T8:
> no second-stop control; `stoppedRef` blocks a second POST) — an API-only path on `main`. Keep P1: blast
> radius unchanged. Any fix (PR #6) must also cover the `_release_now.clear()` at `:591` and the orphan's
> global writes at `:778-790`, not just the null-ing. N7 is the mechanism that makes the "still alive after
> 5 s" state reachable.

**Verdict:** CONFIRMED — a still-alive worker is nulled after the 5 s join (`makermodslab/teleoperate.py:871-875`); a later start passes both guards (`teleoperate.py:569-571`) and clears `_release_now` (`teleoperate.py:592`) while the orphan's finally later overwrites the process globals (`teleoperate.py:789-792`).
**Priority:** P1 — P0-arguable: low-probability (hung cleanup plus a start inside the window) but high-blast-radius hardware-ownership and state-corruption race (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. A second stop sets `_release_now` and waits five seconds (`makermodslab/teleoperate.py:869-874`).
2. If the worker is still alive, the handler sets `teleoperation_thread = None` even though that thread has not exited (`makermodslab/teleoperate.py:875-885`).
3. A later start rejects only an active session or a non-`None`, alive worker. The orphan is therefore invisible, and start clears `_release_now` for the new session (`makermodslab/teleoperate.py:567-594`).
4. The old worker later writes process-global `teleoperation_active = false`, `releasing = false`, and clears the current-device globals during its cleanup (`makermodslab/teleoperate.py:787-792`).

**Impact**

The new start can race the old worker for the same serial ports. If the new start succeeds on other ports, the old worker can still terminate the new session's global active state and erase its current-device state. This is both a hardware-ownership and process-state corruption path.

**Safe reproduction/reasoning**

A mocked worker whose `is_alive()` remains true past `join(timeout=5)` is sufficient. The second stop nulls the only worker reference; a subsequent start passes the worker check and clears the abort event while the mock remains alive.

### T4 · P1 — Cross-feature hardware ownership is incomplete and not atomic

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL, STRENGTHENED (PR #7 CLOSED-unmerged).** (1) Separate per-module
> locks make check-then-set non-atomic (`teleoperate.py:102`, `record.py:215`, `rollout.py:114`); each claims
> its own flag under only its own lock (`teleoperate.py:565-589`, `record.py:472-503`, `rollout.py:917-948`).
> (2) **Inference still blocks only on `teleoperation_active`** (`rollout.py:918`) — not on `releasing` or
> `teleoperation_thread.is_alive()` — and never calls `finish_pending_release()` (only callers are
> `record.py:469-470` and `teleoperate.py:562-563`); a normal stop clears `teleoperation_active` at `:839`
> *before* the arm is driven home, so inference is admitted during the entire releasing window (see N3 for the
> record-side twin). (3) Manual calib (`calibrate.py:232-236`), single/batch auto-cal
> (`auto_calibrate.py:207-215`, `:516-521`), wiggle (`wiggle.py:56-85`) and hand-motion port detection
> (`identify.py:117` only *reports* busy) check **only their own manager** — no `calibration_in_progress`
> guard exists; the `c737a30` mutex is still not on this branch. (4) `SingleTabGuard.tsx:120-140` only overlays
> the losing tab; a teleop session there keeps running with Done now *underneath* a `z-[9999]` backdrop.
> **New strengthening (N4):** the teleop window is a plain `fixed` div with **no backdrop, no focus trap**
> (`TeleopDialog.tsx:204-211`) — every other control (incl. Robot settings → calibration) stays clickable
> during a live session, turning the point-3 race into a two-click path.

**Verdict:** CONFIRMED — separate per-module locks make check-then-set non-atomic (`makermodslab/teleoperate.py:102`, `makermodslab/record.py:220`, `makermodslab/rollout.py` `_state_lock`); inference checks only `teleoperation_active`, not releasing/worker-alive (`rollout.py:899`); calibration, auto-calibration, and wiggle have no cross-feature guard at all (`calibrate.py:237`, `auto_calibrate.py:201,506`, `wiggle.py:62-67`).
**Priority:** P1 — strengthened by the missing mutex (see the Verification note): calibration and wiggle are genuinely unguarded against live sessions, and the existing record/inference↔teleop checks are racy and miss the releasing window (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. Teleoperation, recording, and inference each use a different module-local `_state_lock` (`makermodslab/teleoperate.py:101-103`, `makermodslab/record.py:214-216`, `makermodslab/rollout.py:109`). Each checks the other modules' booleans and then claims its own flag under only its own lock (`makermodslab/teleoperate.py:570-594`, `makermodslab/record.py:463-494`, `makermodslab/rollout.py:899-930`). Two simultaneous starts can therefore both observe the other feature as inactive before either writes its flag.
2. Inference blocks only on `teleoperation_active`, not on the teleoperation worker/releasing state (`makermodslab/rollout.py:899-930`). A normal stop clears `teleoperation_active` before the worker returns the arm and releases the port (`makermodslab/teleoperate.py:841-867`).
3. Manual calibration, single auto-calibration, and batch auto-calibration check only their own managers, not teleoperation (`makermodslab/calibrate.py:232-276`, `makermodslab/auto_calibrate.py:199-252`, `makermodslab/auto_calibrate.py:504-586`). Wiggle likewise opens and commands the requested serial bus without a feature-ownership guard (`makermodslab/wiggle.py:55-85`).
4. `SingleTabGuard` takeover only overlays the old tab and changes tab election; it does not stop a running teleoperation session (`frontend/src/components/SingleTabGuard.tsx:64-89`, `frontend/src/components/SingleTabGuard.tsx:109-140`). The new primary tab can reach another hardware feature while the old teleoperation page remains mounted underneath its overlay.

**Impact**

Two robot-driving features can claim the same follower/leader ports or issue competing motor commands. The sequential happy-path checks do not provide a global mutex, and the normal post-Done window exposes `teleoperation_active=false` while the arm is still energized and moving home.

**Safe reproduction/reasoning**

The race can be demonstrated with barriers around two mocked start handlers because the check-and-set operations are protected by different locks. The releasing gap is direct: inference accepts once the active boolean is false even if `teleoperation_thread.is_alive()` and `releasing` are true. No real process or serial device is needed.

### T5 · P1 — Arm-identity read failure silently fails open before MakerMods Lab writes calibration and drives the arm

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL as a code fact; severity EXCLUDED by the re-verification brief.**
> `verify_arm` still returns `(None, None)` — no refusal, no warning — when `Present_Position` or
> `Homing_Offset` cannot be read (`arm_identity.py:285-288`, docstring `:268-270`); `verify_devices` treats the
> arm as verified (`:391-411`); teleop then writes calibration and configures motors in both paths
> (`teleoperate.py:525-528`, `:643-651`). The severity of the arm-identity guard is out of scope for the
> re-verification pass (a fix is committed on `fix/arm-identity-block-unverified`, not merged). Noted so the
> row isn't silently dropped.

**Verdict:** CONFIRMED-WITH-CORRECTIONS — the behavior exists as described (`verify_arm` returns `(None, None)` on a read failure, `makermodslab/arm_identity.py:286-288`), but it is a deliberate, documented, tested design tradeoff (docstring at `arm_identity.py:271-272`; contract pinned by `tests/test_arm_identity.py`), not an oversight.
**Priority:** P2 — deliberate, documented, tested fail-open design — flagged as a fail-open-vs-fail-closed product decision to revisit (given the F.1 swapped-port incident), not a code bug (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. If either `Present_Position` or `Homing_Offset` reading fails, `verify_arm()` logs and returns `(None, None)`: no refusal and no warning (`makermodslab/arm_identity.py:252-288`).
2. `verify_devices()` therefore treats the arm as verified enough to continue (`makermodslab/arm_identity.py:391-413`).
3. Teleoperation immediately writes the assigned calibration, configures motors, and starts control in both single and bimanual paths (`makermodslab/teleoperate.py:511-543`, `makermodslab/teleoperate.py:636-665`, `makermodslab/teleoperate.py:695-709`).

**Impact**

The one guard intended to catch swapped or misassigned arms can disappear silently on a transient read failure. MakerMods Lab can then stamp the selected calibration into an unverified physical arm and transmit leader actions with no operator-facing warning.

**Safe reproduction/reasoning**

The existing test explicitly requires this fail-open contract: a bus raising `ConnectionError("no response")` produces neither refusal nor warning (`tests/test_arm_identity.py:505-521`). That contradicts a hardware-safety boundary where an unverified device must be flagged before calibration writes and motion.

### T6 · P1 — A saved low-power setting can fail to apply while teleoperation starts at previous or full power

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): NOT APPLICABLE — SUPERSEDED.** The mechanism no longer exists.
> `apply_motor_power` is gone from the tree (repo-wide grep: zero hits). `motor_power.py:16-21` now states the
> policy: the per-robot percentage is the **auto-calibration** drive torque only, and "Regular sessions —
> teleoperation, recording, skill runs — deliberately run at stock LeRobot torque". Teleop start calls
> `reset_torque_limit` + `clear_goal_velocity` (`teleoperate.py:656,661` single; `:535,539` bimanual) instead
> of applying a cap, and `TeleoperateRequest` (`:291-308`) no longer carries a `motor_power` field —
> `RobotCorner` doesn't send one (`RobotCorner.tsx:161-179`). The residual "per-motor write failure warns but
> doesn't abort" shape survives in `_for_each_motor` (`motor_power.py:120-135`) but its direction is inverted:
> a failed `reset_torque_limit` leaves the motor at a *lower* prior cap (sluggish, safe), not full power. The
> audit's hazard cannot occur on this path. **Entry closed.**
>
> ⚠️ **Contradiction with `CONFIRMED 22-07-26.md` #3** ("Torque / motor-power slider gone … UI door missing"):
> on `main` the slider is present and wired — `RobotConfigDialog.tsx:1215-1245` (percent↔raw conversion,
> `DEFAULT_MOTOR_POWER = 38`), `:1269` (dirty tracking), `:1276` (`patch.motor_power`). Either #3 was fixed by
> the tier-1 UX commits (`fc85b40` / `e7f7d18`) or it referred to a different surface. Coordinator call.

**Verdict:** CONFIRMED-WITH-CORRECTIONS — the mechanism is real (per-motor write failures leave the previous limit, `makermodslab/motor_power.py:149-164`; start proceeds with `success: true`, `makermodslab/teleoperate.py:658,807-808`), but the warning is surfaced as a start toast (`frontend/src/components/landing/RobotConfigManager.tsx:135-140`), not silently dropped — the residual gap is timing (motion can begin before the toast is read). ⚠ UNRESOLVED @ HEAD ce42158 — `RobotConfigManager.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/dialogs/RobotConfigDialog.tsx + launchpad/, construct not re-located.
**Priority:** P2 — warned via toast, degraded-but-safe by design, and needs a real per-motor write failure (with `num_retry=2`) to trigger (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. The selected robot's saved `motor_power` is included in the start request (`frontend/src/components/landing/RobotConfigManager.tsx:104-125`). ⚠ UNRESOLVED @ HEAD ce42158 — `RobotConfigManager.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/dialogs/RobotConfigDialog.tsx + launchpad/, construct not re-located.
2. `apply_motor_power()` catches per-motor write failures and returns only warnings; failed motors retain their previous limit, explicitly full power after power-up (`makermodslab/motor_power.py:136-164`).
3. Teleoperation appends those warnings but proceeds to capture the rest pose, start the worker, and return `success: true` (`makermodslab/teleoperate.py:535-544`, `makermodslab/teleoperate.py:657-679`, `makermodslab/teleoperate.py:794-809`).
4. The frontend shows a warning toast only after the backend has already started the worker and navigates into teleoperation (`frontend/src/components/landing/RobotConfigManager.tsx:127-147`). ⚠ UNRESOLVED @ HEAD ce42158 — `RobotConfigManager.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/dialogs/RobotConfigDialog.tsx + launchpad/, construct not re-located.

**Impact**

An operator who selected a gentler 10–40% profile can receive immediate follower motion from one or more motors at their previous setting or 100%. The saved hardware-safety setting is therefore advisory rather than enforced.

**Safe reproduction/reasoning**

The unit suite explicitly asserts "failure warns but does not abort" and leaves the failing motor unwritten (`tests/test_motor_power.py:95-108`). No hardware is needed.

### T7 · P1 — Startup cleanup can report an ordinary setup error while discarding "torque may still be enabled" failures

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL — code unchanged, still the top-ranked teleop P1 (PR #5
> CLOSED-unmerged).** `_safe_disconnect` still builds "TORQUE MAY STILL BE ENABLED — the arm can stay rigid;
> unplug its power to release it" (`teleoperate.py:442-467`); setup has already written calibration and run
> `configure()` before later operations can fail (single `:641-661` then `capture_rest_pose` `:673-676`;
> bimanual `:527-539`); **both cleanup call sites discard the return value** (`:540-543`, `:809-811`); the
> response is `{"success": False, "message": str(e)}` (`:818`) — the original setup error only, and
> `last_cleanup_error` stays `None` (`:585`) so `/teleoperation-status` shows nothing (`:914`); the
> startup-failure path **never runs the per-motor `force_disable_torque` hardening** — that exists only in the
> worker's `finally` (`:770-771`), which a failed start never reaches. See N11: this leaves zero persistent
> server-side evidence.

**Verdict:** CONFIRMED — both cleanup call sites discard `_safe_disconnect`'s returned torque warning (`makermodslab/teleoperate.py:544-545,812-813`; warning text at `teleoperate.py:460-463`), and the startup-failure path never runs the per-motor `force_disable_torque` hardening (that exists only in the worker's finally, `teleoperate.py:772-773`).
**Priority:** P1 — top P1: failed startup can strand an energized follower and drops the unplug warning; F.1 incident precedent (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. `_safe_disconnect()` returns a safety warning if disconnect fails, including "TORQUE MAY STILL BE ENABLED" (`makermodslab/teleoperate.py:448-473`).
2. Setup has already written calibration and configured the follower before several later operations can fail (`makermodslab/teleoperate.py:648-681`; bimanual equivalent `makermodslab/teleoperate.py:527-543`).
3. Both the bimanual connector's exception path and the outer start exception path call `_safe_disconnect()` but discard its returned warning (`makermodslab/teleoperate.py:545-548`, `makermodslab/teleoperate.py:811-822`).
4. The API response contains only the original setup exception and does not populate `last_cleanup_error` or a terminal warning.

**Impact**

If start fails after follower configuration and cleanup also fails, the follower may remain energized or its port may remain in an uncertain state. The operator sees only the original startup error, not the explicit unplug-power warning generated by cleanup.

**Safe reproduction/reasoning**

Mock a post-configuration operation to raise and the follower's `disconnect()` to raise. `_safe_disconnect()` constructs the safety text, but the handler discards the return value and returns `str(original_exception)`.

### T8 · P2 — The ordinary interface removes the advertised second-stop "release now" control

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL, WORSE — raise P2 → P1.** The backend still tells the operator
> to do something the UI cannot do: "…then goes limp. Press Stop again to release it now." (`teleoperate.py:859-862`),
> and the second-stop handler still exists (`:865-892`). On `main` the live window has exactly one control —
> Done (`TeleopDialog.tsx:217-223`) — which **closes the window** (`:197`); ESC does the same (`:185-190`).
> There is no second-stop control anywhere, and after Done there is no teleop surface at all. This is now the
> same defect as N2 ("the UI disappears while the arm is still energized and moving") — hence raised to P1.

**Verdict:** CONFIRMED — the backend advertises "Press Stop again to release it now" (`makermodslab/teleoperate.py:860-864`), but the page's single Done control latches the one-shot guard and navigates home (`frontend/src/components/control/VisualizerPanel.tsx:34-39`, `frontend/src/pages/Teleoperation.tsx:73,159-162`); no second-stop control exists anywhere in the UI.
**Priority:** P2 — the operator cannot abort the gentle progress-based return from the UI and must wait out the 10 s ceiling or cut power (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. The first backend stop responds `releasing: true` and instructs "Press Stop again to release it now" (`makermodslab/teleoperate.py:858-867`).
2. The page has one Done control (`frontend/src/components/control/VisualizerPanel.tsx:29-39`).
3. Done sets the one-shot stop guard and immediately navigates home after the first response (`frontend/src/pages/Teleoperation.tsx:71-78`, `frontend/src/pages/Teleoperation.tsx:159-162`).
4. The landing page exposes no second teleoperation-stop control. The second-stop behavior remains API-only.

**Impact**

If the automatic return is moving toward an obstacle, pinching something, or otherwise should be aborted, the operator cannot perform the action the backend message tells them to perform. They must wait for the return/stall ceiling or physically remove power.

**Safe reproduction/reasoning**

The frontend control flow deterministically leaves the route after the first response and latches `stoppedRef`; no browser/hardware probe is required.

### T9 · P2 — The teleoperation route can display "Live Robot Data" while no session exists

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL — new manifestation inside the dialog.** The status poll
> transitions only on an inactive `failed`/`ran_with_warning` outcome (`TeleopDialog.tsx:65-69`; dead twin
> `pages/Teleoperation.tsx:47-51`); inactive `ok`/`None` is ignored. The badge is driven purely by socket
> connectivity (`UrdfViewer.tsx:352-378`). `/ws/joint-data` still accepts any connection regardless of teleop
> state (`server.py:806-836`). New manifestation: because the dialog stays open after a mid-loop death, the
> header can show a **green "live" dot next to a red "Teleoperation failed" banner** — the dot never goes red
> since the WebSocket is still up. The bookmark/direct-route scenario now only applies to the orphaned
> `/teleoperation` page.

**Verdict:** CONFIRMED — the status poll only transitions on an inactive `failed`/`ran_with_warning` outcome (`frontend/src/pages/Teleoperation.tsx:47-51`), the badge is driven purely by socket connectivity (`frontend/src/components/UrdfViewer.tsx:337-347`), and `/ws/joint-data` accepts connections regardless of teleoperation state (`makermodslab/server.py`, `websocket_endpoint`).
**Priority:** P2 — misleading idle-state display; no hardware effect (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. `/teleoperation` is directly routable and does not start or attach to a named session (`frontend/src/App.tsx:40-49`).
2. Status polling only transitions the page when it sees an inactive terminal `failed` or `ran_with_warning` outcome; inactive `None` or `ok` remains rendered indefinitely (`frontend/src/pages/Teleoperation.tsx:38-70`).
3. The URDF badge labels an open shared WebSocket "Live Robot Data" solely from socket connectivity (`frontend/src/components/UrdfViewer.tsx:332-348`). The WebSocket accepts connections regardless of teleoperation state (`makermodslab/server.py:806-835`).

**Impact**

A bookmark, reload after a clean session, or direct navigation can show a normal teleoperation screen and green "Live Robot Data" badge even though no arm session is active and no joint updates are being produced. Done then receives "No teleoperation session is active," which the page does not surface.

**Safe reproduction/reasoning**

Open the route with backend state `teleoperation_active=false`, `releasing=false`, `outcome=None`. The poll has no matching branch, while the shared WebSocket can still connect.

### T10 · P2 — Saved-camera preview can throw instead of flagging an unavailable browser camera API

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL in code, NOT REACHABLE from teleop (panel orphaned).**
> `useCameraStream` still dereferences `navigator.mediaDevices.addEventListener` unguarded in its first effect
> (`hooks/useCameraStream.ts:43`) and `getUserMedia` at `:60`. **Teleop reachability is gone**:
> `TeleopCameraPanel` is mounted only by the orphaned page (`pages/Teleoperation.tsx:211`), and `TeleopDialog`
> renders no camera panel. The hook is still live on three other surfaces —
> `components/recording/CameraConfiguration.tsx:369`, `components/studio/DeployPanel.tsx:89`,
> `components/landing/InferenceModal.tsx:59` — so the defect should be **re-filed against those lists**
> (Recording / Deploy / Inference) rather than closed.

**Verdict:** CONFIRMED — the hook dereferences `navigator.mediaDevices.addEventListener` unguarded in its first effect (`frontend/src/hooks/useCameraStream.ts:43`) and `getUserMedia` likewise (`useCameraStream.ts:60`); with `mediaDevices` absent (plain-HTTP LAN origin, unsupported browser) the effect throws a `TypeError` before `hasError` can be set, and no error boundary wraps the panel.
**Priority:** P2 — requires a non-secure origin or unsupported browser; HTTPS is the documented camera path (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. The panel correctly maps only the selected record's saved camera IDs; it does not rebind them (`frontend/src/components/control/TeleopCameraPanel.tsx:25-36`, `frontend/src/components/control/TeleopCameraPanel.tsx:67-83`).
2. When a feed mounts, `useCameraStream` unconditionally calls `navigator.mediaDevices.addEventListener(...)` before checking whether `mediaDevices` or `getUserMedia` exists (`frontend/src/hooks/useCameraStream.ts:35-46`).
3. The later `getUserMedia` call is also unguarded for an unavailable API (`frontend/src/hooks/useCameraStream.ts:48-99`).
4. Plain HTTP on a non-local LAN origin and unsupported browsers can expose no `navigator.mediaDevices` at all.

**Impact**

Turning Cameras on can raise a `TypeError` in the React effect instead of leaving the configured role visible with "Preview failed." The motor loop remains backend-independent, but the teleoperation interface can crash or lose its controls precisely when the user asks to view a saved camera.

**Safe reproduction/reasoning**

Render a configured `CameraFeed` with `navigator.mediaDevices` undefined. The first effect dereferences `addEventListener` before the hook can set `hasError`.

### T11 · P2 — The active hardware session and rendered robot/cameras do not share an immutable session identity

**First identified:** 2026-07-14

> **Verdict on `main` (2026-07-23): STILL REAL, STRENGTHENED — the robot dropdown is clickable mid-session.**
> Start snapshots one record into the request (`RobotCorner.tsx:161-179`); the window derives its title, arm
> count and layout from the live, mutable `selectedRecord` (`TeleopDialog.tsx:25-26`, `:207`, `:215`,
> `:266-289`); `useRobots` records are shared refreshable module state; `/teleoperation-status` still returns
> **no session descriptor** (`teleoperate.py:897-925`). Strengthened: because the window is non-modal,
> `RobotCorner`'s robot dropdown (`:270-351`) stays clickable behind/around the live session window — selecting
> another robot re-renders the open window with the other robot's name/arm count while the hardware keeps
> running the original request. In the audited tree this needed a second client or a direct API call.

**Verdict:** CONFIRMED — start snapshots one record into the request (`frontend/src/components/landing/RobotConfigManager.tsx:104-125`) while the page renders layout and cameras from the live, mutable `selectedRecord` (`frontend/src/pages/Teleoperation.tsx:15-16`, `frontend/src/components/control/TeleopCameraPanel.tsx:31-36`), and status returns no session descriptor (`makermodslab/teleoperate.py:897-925`). ⚠ UNRESOLVED @ HEAD ce42158 — `RobotConfigManager.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/dialogs/RobotConfigDialog.tsx + launchpad/, construct not re-located.
**Priority:** P2 — edge case (record renamed/deleted/changed mid-session); UI/telemetry mismatch only, no hardware misbehavior (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. Start snapshots one robot record into the request (`frontend/src/components/landing/RobotConfigManager.tsx:104-125`). ⚠ UNRESOLVED @ HEAD ce42158 — `RobotConfigManager.tsx` deleted in the redesign (9e18b27); searched frontend/src/components/dialogs/RobotConfigDialog.tsx + launchpad/, construct not re-located.
2. The teleoperation page independently derives single/bimanual layout and camera feeds from the current module-level `selectedRecord` (`frontend/src/pages/Teleoperation.tsx:13-16`, `frontend/src/components/control/TeleopCameraPanel.tsx:25-36`).
3. Robot records are refreshed on location changes, and the selected name is mutable browser state (`frontend/src/hooks/useRobots.ts:29-84`, `frontend/src/hooks/useRobots.ts:96-110`, `frontend/src/hooks/useRobots.ts:273-294`).
4. Teleoperation status returns no session ID, robot name, mode, or camera snapshot (`makermodslab/teleoperate.py:897-925`).

**Impact**

If the selected record is renamed/deleted/changed from another client, or a direct API start is followed by route navigation with a different selected record, the hardware continues using the original request while the page can render the wrong arm count and wrong saved-camera roles/IDs. It does not silently rebind a saved camera within a record, but it can bind the interface to the wrong record altogether.

**Safe reproduction/reasoning**

Start a mocked bimanual session from record A, then make the shared selected record resolve to single-arm record B. The page chooses one viewer and B's cameras; the backend worker still owns A's four-arm request.

## Design gaps and lower-confidence risks

### No server-owned session token or immutable session descriptor

**First identified:** 2026-07-14

Start, stop, status, and WebSocket updates are process-global. There is no session token to prevent an old page, delayed stop, delayed 13-second cleanup check, or another client from acting on a newer session. Adding an active session descriptor would also resolve the confirmed layout/camera mismatch instead of asking the page to infer ownership from mutable selection.

### Direct API requests bypass saved-record validation

**First identified:** 2026-07-14

`TeleoperateRequest` accepts arbitrary ports/configs, defaults omitted power to 100, accepts any mode string (anything other than exact `bimanual` follows the single path), and exposes `skip_identity_check` (`makermodslab/teleoperate.py:294-314`, `makermodslab/teleoperate.py:603-607`). The ordinary frontend sends the saved record correctly, so this is an API policy gap rather than a normal-interface defect. Decide whether `/move-arm` is a low-level expert API or should reload/validate the named saved record.

### Camera MJPEG behavior remains browser-negotiated and unverified on target hardware

**First identified:** 2026-07-14

The current teleoperation camera path is browser-only and correctly uses exact saved `device_id` values. It requests ideal 1280×720 at 30 FPS specifically to encourage Chromium to negotiate MJPEG (`frontend/src/hooks/useCameraStream.ts:58-76`). Whether that choice behaves correctly on every target UVC camera is hardware/browser-dependent; the audit did not classify it as a defect without fresh runtime evidence. It must not be "fixed" by selecting another camera or changing the saved ID.

### WebSocket connectivity is not robot-data freshness

**First identified:** 2026-07-14

The UI has no last-joint-message age. Even during a real session, a connected shared socket can remain green when broadcasts stop or when only non-joint events are flowing. The direct-route issue above is confirmed; broader data-staleness semantics need a product decision and browser tests. *(Re-verification: N9 shows the broadcast queue is unbounded and can make joint data progressively staler while the badge stays green.)*

---

## Appended findings (N-series)

One consolidated, append-only list: every finding since the original audit gets the next `N` here, newest last. IDs are permanent and per-module — "Recording N1" is distinct from "Dataset N1" / "Inference N1". N1–N11 all date from the 2026-07-23 re-verification. *(Consolidated 2026-08-02 from the former "New findings" sections and later in-place appends; no prose, verdicts or severities were changed.)*

### N1 · P1 — Teleoperation is the only live-hardware surface with no page-leave guard

**First identified:** 2026-07-23

Recording, inference and calibration all route their leave protection through `hooks/useSessionExitGuard.ts` — native `beforeunload` prompt, back-button `confirm`, keepalive beacon, and a single `handledRef` latch (`useSessionExitGuard.ts:90-160`). Consumers: `RecordingSessionDialog.tsx:192`, `InferenceSessionDialog.tsx:106`, `RobotConfigDialog.tsx:595`.

**Teleop does not use it.** `TeleopDialog.tsx:159-178` hand-rolls only the `pagehide` half:

- **No `beforeunload`** → a reload, typed URL, or tab close during a live session proceeds with **no confirmation**, on the one surface where the arm is being driven by a human at that instant.
- **No `popstate` guard** → the browser Back button leaves without a prompt.
- The hand-rolled handler also lacks the guard's `handledRef` discipline — the mechanism that would have prevented T1's latch-before-ack.

**Impact:** the operator loses the one confirmation the other three hardware surfaces give them, and the only stop that fires is the unacknowledged keepalive of T1. **Fix note:** adopting `useSessionExitGuard` here closes N1 and the page-leave half of T1 in one change.

### N2 · P1 — Done closes the only session window while the arm is still energized and moving

**First identified:** 2026-07-23

`handle_stop_teleoperation`'s first stop returns immediately with `releasing: true` while the worker drives the follower home for up to `RETURN_CEILING_S = 10 s` with torque **on** (`teleoperate.py:856-865`, `rest_pose.py:54`, `teleoperate.py:759-767`). The UI:

1. `handleDone` awaits the stop and then calls `onOpenChange(false)` (`TeleopDialog.tsx:195-198`) — the window unmounts (`:202`) while the arm is still moving.
2. Nothing on the Launchpad polls `/teleoperation-status`; `releasing` is never rendered anywhere — the flag exists in the payload (`teleoperate.py:911`) and is read only to pick a toast (`TeleopDialog.tsx:106`).
3. The Teleop button re-enables the instant the request resolves (`RobotCorner.tsx:216-220`, `:360`).
4. Pressing it during the release calls `finish_pending_release()` (`teleoperate.py:562`), which **sets `_release_now`** — cutting the rest return short (`rest_pose.py:166-167`) so the worker proceeds straight to `force_disable_torque` (`teleoperate.py:770`). **The arm is de-energized mid-motion, part-way home, with no warning.**

So the advertised "press Stop again to release it now" (T8) is not merely missing — it has been replaced by an *unlabelled* release-now control that looks like "start a new session". Ranked P1.

### N3 · P1 — `finish_pending_release()`'s failure return is ignored by both start paths

**First identified:** 2026-07-23

`finish_pending_release` returns `False` when the previous worker did not exit within the 10 s join (`teleoperate.py:228-231`; recording twin `record.py:252-261`). **Neither caller checks it:**

- `makermodslab/teleoperate.py:562-563` calls both, discards both, then checks only `_record.recording_active` (`:576`).
- `makermodslab/record.py:469-470` calls both, discards both, then checks only `_teleoperate.teleoperation_active` (`:482`).

The teleop→record direction happens to be safe: `recording_active` stays `True` until the very end of the recording worker (`record.py:650`), after its hardware release. **The record→teleop direction is not.** `teleoperation_active` is cleared by the *stop handler* (`teleoperate.py:839`), long before the worker finishes the rest return and the torque release (`:757-788`). Therefore:

> teleop stopped → worker still returning/releasing → operator starts a recording → `_teleoperate.finish_pending_release()` returns `False` (worker wedged past 10 s) → **ignored** → `_teleoperate.teleoperation_active` is `False` → recording proceeds to open the same leader/follower ports the live teleop worker still holds.

Teleop's own start does check the live-worker case (`:568-575`); recording and inference do not. Same family as T4 point 2, but a distinct, cheaply-fixable defect: **use the return value** (or have recording/inference consult `teleoperate.teleoperation_thread`/`releasing`).

### N4 · P1 — The live teleop window is non-modal: the rest of the app stays interactive during a session

**First identified:** 2026-07-23

`TeleopDialog` renders a bare `fixed … z-50` div with `role="dialog"` and **no overlay/backdrop element and no focus trap** (`TeleopDialog.tsx:204-211`) — deliberately "not a window dialog" per its own docstring (`:13-16`). Unlike the Radix dialogs used elsewhere, nothing blocks interaction with what's behind it.

While an arm is being teleoperated a click away, the operator can:

- open **Robot settings** → run a manual or automatic calibration on the same ports — **the unguarded feature of T4 point 3** (`RobotCorner.tsx:255`, `calibrate.py:232-236`, `auto_calibrate.py:207-215`);
- **switch, rename, or delete** the active robot record (`RobotCorner.tsx:302-349`) — the deletion path is not gated on any session state, and switching triggers T11;
- open the studio and use the second `RobotCorner` instance (both are mounted simultaneously — `Launchpad.tsx:48` and `StudioOverlay.tsx:74`), whose Teleop button is also enabled.

Recording and inference sessions get real modal dialogs. This is the UI half of T4 and turns a race into a routine mis-click.

### N5 · P2 — The teleop state machine is duplicated, and `/teleoperation` is a live footgun

**First identified:** 2026-07-23

`TeleopDialog.tsx:13-19` documents the state machine as "ported verbatim" from `pages/Teleoperation.tsx`. Both files now carry the same stop latch, swallowed catch, pagehide handler and status poll (compare `TeleopDialog.tsx:55-178` with `pages/Teleoperation.tsx:38-157`). Consequences:

- Every fix to T1/T8/T9/T11 must be applied twice or the dead page must go.
- The dead page is still routable. Navigating to `/teleoperation` **unmounts the Launchpad**, hence `RobotCorner`, hence `TeleopDialog` — whose cleanup fires a real stop (`TeleopDialog.tsx:174-177`). So a URL typo silently ends a live session, and the page you land on then shows the T9 "Live Robot Data" badge for the session it just killed.
- `TeleopCameraPanel` / `VisualizerPanel` are dead code carrying their own findings (T10).

### N6 · P2 — The 13 s post-stop cleanup check reads process-global state a newer session clears

**First identified:** 2026-07-23

After a `releasing: true` response the UI schedules a single status re-check 13 s later (`TeleopDialog.tsx:119-138`) and toasts `last_cleanup_error` if present. But `last_cleanup_error` is process-global and **cleared by the next start** (`teleoperate.py:585`), and the timer keeps running after the window closes. Two failure modes:

- Start a new session within 13 s → the old session's genuine "TORQUE MAY STILL BE ENABLED" warning is erased before the check reads it.
- The check can equally read a *newer* session's cleanup error and attribute it to the closed one.

A fixed 13 s also has no relationship to the actual release time (bounded by the 10 s return ceiling *plus* the per-motor torque disable with `num_retry=5`, `teleoperate.py:428`).

### N7 · P2 — The rest-return join is unbounded, and part of the return sits outside both the ceiling and the abort

**First identified:** 2026-07-23

`_return_followers_to_rest` starts one thread per follower and joins each **with no timeout** (`teleoperate.py:285-288`). The 10 s `RETURN_CEILING_S` bounds only `_run_return_loop` (`rest_pose.py:165`); everything else is unbounded and un-abortable:

- the pre-loop `bus.write("Goal_Velocity", …)` per motor plus `sync_write("Goal_Position", …)` (`rest_pose.py:111-113`) — before any `abort_event` check;
- `time.sleep(RETURN_SETTLE_S)` at `rest_pose.py:159`, again before the first abort check at `:166`;
- `_restore_goal_velocity`'s `sync_write` in the `finally` (`rest_pose.py:129`, `:144`).

If a follower bus wedges there, the worker never reaches `force_disable_torque` (`teleoperate.py:770`) and **the arm stays energized indefinitely** — while the second stop's 5 s join expires and takes the T3 orphan branch (`:870-881`). Bounded in practice by pyserial/SDK packet timeouts (hence P2), but it is the mechanism that makes T3's "worker still alive after 5 s" reachable at all.

### N8 · P2 — The stop path mutates shared state without `_state_lock`

**First identified:** 2026-07-23

`handle_stop_teleoperation` reads and writes `teleoperation_active` and `teleoperation_thread` with no lock (`teleoperate.py:834-894`), while `handle_start_teleoperation` claims those globals **under** `_state_lock` (`:565-589`). Both endpoints are plain `def` handlers (`server.py:503-512`), so FastAPI runs them on the threadpool — genuinely concurrent.

Concrete window: start assigns `teleoperation_thread` at `:792` but doesn't call `.start()` until `:795`. A stop landing in between sees `teleoperation_active is True` and `worker.is_alive() is False`, takes the "no worker" branch, sets `teleoperation_thread = None` and returns "Teleoperation stopped successfully" (`:843-854`) — while the start goes on to launch a worker with no tracked reference whose loop exits on its first condition check (`:705`). A related variant: a stop arriving during the multi-second synchronous connect (between `:586` and `:795`) makes the start report success for a session that never runs. Reachable through the API and through a double-click on Teleop plus a stale keepalive.

### N9 · P2 — The WebSocket broadcast queue is unbounded and its overflow branch is dead

**First identified:** 2026-07-23

`ConnectionManager.broadcast_queue = queue.Queue()` has **no `maxsize`** (`server.py:254`), so `put_nowait` can never raise and the "Broadcast queue is full, dropping data" branch (`server.py:381-382`) is unreachable. The drain is bounded per client: `future.result(timeout=1.0)` (`server.py:368-374`). Teleop produces at 20 Hz (`teleoperate.py:701`, `:710`) plus ~1 Hz current sample. One wedged/suspended client (a throttled backgrounded tab) can make the drain slower than the producer; the queue then grows without bound for the session, every client gets progressively **staler** joint data, and the "Live Robot Data" badge stays green (T9). A bounded queue with drop-oldest is the right shape for a latest-value telemetry stream.

### N10 · P2 — `strict=False` hides a telemetry/bus count mismatch

**First identified:** 2026-07-23

`telemetry_targets = list(zip(_device_buses(robot), ["left_", "right_"] if is_bimanual else [""], strict=False))` (`teleoperate.py:687-689`). If `_device_buses` ever returns a count other than 1 (single) or 2 (bimanual) — e.g. a BiSO device where one sub-arm's bus failed to attach — `zip` silently truncates and that follower's current is never sampled, producing an all-zero peak that reads exactly like the "firmware leaves the register at 0" case the docstring warns about (`teleoperate.py:133-135`). Cheap fix: `strict=True`, or build the pairs explicitly.

### N11 · P2 — A failed start leaves no trace in the status payload

**First identified:** 2026-07-23

The start except path sets `teleoperation_active = False` and clears the device globals (`teleoperate.py:814-816`) but never populates `last_session_outcome`/`last_session_error` — which the start path had just cleared to `None` (`:586-587`). `/teleoperation-status` therefore reports `outcome: None, error: None, last_cleanup_error: None` after a start that failed *after* connecting and configuring the follower (`:918-923`). The only record is the HTTP response body, rendered as a transient toast (`RobotCorner.tsx:198-204`). Combined with **T7**, a startup failure whose cleanup also failed leaves an energized arm and **zero** persistent server-side evidence.

---

## Behaviors checked and not found defective

- Teleoperation backend configurations contain no LeRobot/OpenCV cameras; browser previews cannot block the motor loop (`makermodslab/utils/robot_factory.py:49-79`, `makermodslab/utils/robot_factory.py:82-119`).
- The teleoperation camera panel uses the saved robot camera list and exact saved `device_id` values. It does not enumerate or silently substitute another camera (`frontend/src/components/control/TeleopCameraPanel.tsx:25-36`; `frontend/src/hooks/useCameraStream.ts:58-76`). *(On `main` this panel is orphaned — see T10.)*
- Single and bimanual paths both run the identity guard before calibration writes and apply motor power only to follower arms when those operations succeed (`makermodslab/teleoperate.py:476-544`, `makermodslab/teleoperate.py:638-667`).
- Mid-loop action/read failures reach a terminal `failed` status and skip the rest return, then attempt torque disable/disconnect (`makermodslab/teleoperate.py:707-792`).
- Bimanual rest returns run concurrently and wait for both return threads before torque release (`makermodslab/teleoperate.py:258-291`).

## Focused validation (original audit)

```text
.venv/bin/python -m pytest -p no:cacheprovider \
  tests/test_teleoperate.py \
  tests/test_arm_identity.py \
  tests/test_motor_power.py \
  tests/test_devices.py -q
```

Result: **96 passed, 5 warnings in 2.17 seconds**. The suites are mock/temp-backed and did not open hardware. Passing results do not invalidate the findings: current tests explicitly require identity-read failure to fail open, motor-power write failure not to abort, unsuccessful rest returns not to raise, and a failing bimanual return not to propagate from the wrapper.

## Missing regression coverage

- Done/network failure must keep the page in a recoverable state, warn the operator, and retry/confirm stop before navigation.
- Browser unload notice must distinguish "stop requested" from server-confirmed "stopped."
- Rest-pose capture/return failure must reach the stop response and terminal outcome, including one-arm bimanual failure.
- A worker still alive after second-stop timeout must remain owned and block every new robot-driving start.
- One global hardware-session arbiter must make teleoperation/recording/inference/calibration/auto-calibration/wiggle ownership atomic and include releasing workers.
- Identity read failure must produce an operator-visible warning or refusal before any calibration write.
- Saved motor-power application failure needs an explicit safety policy and a test that prevents unacknowledged full-power motion. *(Superseded on `main` — see T6.)*
- Startup cleanup failure must preserve and surface `_safe_disconnect()` torque warnings.
- The ordinary page must retain an accessible release-now control while `releasing=true`.
- Direct-route/idle status must redirect or render an explicit "no active session" state.
- Camera hooks must handle absent `navigator.mediaDevices` as a saved-device preview error without crashing.
- Status should include immutable active-session robot/mode/camera metadata, with tests for record changes during a session.
- Frontend tests are missing for teleoperation Done, unload, camera API absence, selected-record drift, and direct-route state.

## Context: contradictions with the original brief (resolved during re-verification)

1. **The brief's frontend paths are stale.** On `main` only `TeleopDialog.tsx` (plus `RobotCorner.tsx`, which owns the start request) is live; the page and the whole `control/` directory are unreachable.
2. **T6's premise no longer exists** — verdicted NOT APPLICABLE-SUPERSEDED, because `apply_motor_power` / the `motor_power` field in `TeleoperateRequest` were deleted, not fixed.
3. **`CONFIRMED 22-07-26.md` #3 ("Torque / motor-power slider gone") does not reproduce on `main`** — the slider is present and wired (`RobotConfigDialog.tsx:1215-1276`).
4. **`CONFIRMED 22-07-26.md` #2 ("Teleop camera visualization gone") is confirmed and now has a root cause**: the camera panel lives only on the orphaned `/teleoperation` page; the live window renders no cameras.
5. **`CONFIRMED 22-07-26.md` #15 ("Power telemetry not implemented")**: the backend telemetry exists and is broadcast (`teleoperate.py:141-178`, `:732-735` sends `follower_currents_ma`), logged at INFO (`:754-756`). Repo-wide grep finds **no frontend consumer** — "not implemented" is accurate for the UI, not the backend.
6. **PR overlap:** #5/#6/#7 target T7/T3/T4. Each was verified against `main`'s code independently. N3 and N4 sit inside T4's family and N7 inside T3's — a PR that closes only the entry as originally written will leave those open.

## Audit safety

- No production, test, or frontend file was modified.
- No git state was changed: no commit, add, push, branch switch, stash, reset, or clean.
- No server was started, no serial port opened, no camera opened, no motor energized, no test suite run during re-verification.
- No network, Hub, or credential access.
