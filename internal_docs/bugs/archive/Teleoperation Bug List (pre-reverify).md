# Teleoperation Bug List

Audit date: July 14, 2026  
Audited commit: `518ca56` (`min_stable`)

## Verification note (2026-07-14)

Every entry below was re-verified on 2026-07-14 against the current working tree of branch `andrew` (HEAD `257580f`, uncommitted changes included). One structural finding: the calibration↔session bidirectional mutex documented as deliberate design in `logs/PROJECT.md` (commit `c737a30`, "calibration and hardware sessions are mutually exclusive") is **NOT present in this branch** — `c737a30` is not an ancestor of HEAD, and no `calibration_in_progress` / `hardware_session_in_progress` guard exists under `makermodslab/`. Calibration (manual, single, batch) and wiggle are genuinely unguarded against live teleop/record/inference sessions, which strengthens the cross-feature-ownership entry (fourth P1 below). Flag: coordinator to determine whether the mutex was lost in the rebrand and needs porting.

Pathway baseline: `internal_docs/complete functionalities/Teleoperation Interface Pathways.md`.

Scope: saved robot selection and request construction, single/bimanual startup, arm identity, calibration application, follower motor power, the live control loop, WebSocket/status state, saved-camera previews, normal and emergency stop, rest-pose return, torque release, page leave, cross-feature ownership, and recovery.

Camera design rule used by this audit: teleoperation must use the saved robot camera configuration. A missing or invalid saved device ID must be surfaced to the operator; it must not be silently replaced with a discovered camera.

## Confirmed bugs

### P0 — none found

No defect was classified as P0. Several P1 findings can leave a robot-driving worker active or allow conflicting hardware ownership, but each requires a failure or race rather than occurring on every ordinary session.

### P1 — Done and page-leave flows treat a stop attempt as if it were an acknowledged stop

**Verdict:** CONFIRMED — `stoppedRef` latched before the POST and never reset on failure (`frontend/src/pages/Teleoperation.tsx:72-73,125-127`); `pagehide` writes the stopped marker before the unacknowledged keepalive (`Teleoperation.tsx:141-149`) and `TeleopStopNotice` renders it as an affirmative stop (`frontend/src/components/TeleopStopNotice.tsx:26-30`).
**Priority:** P1 — the UI affirmatively reports the arm stopped/released while the session may still be live after a transient stop failure (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. `stopTeleoperation()` sets `stoppedRef.current = true` before sending `POST /stop-teleoperation` (`frontend/src/pages/Teleoperation.tsx:71-78`).
2. A network/JSON failure is swallowed without a warning or resetting the guard (`frontend/src/pages/Teleoperation.tsx:119-128`).
3. The Done handler navigates home after awaiting that function even when the function swallowed a failure (`frontend/src/pages/Teleoperation.tsx:159-162`).
4. Unmount cleanup then cannot retry because the guard is already latched (`frontend/src/pages/Teleoperation.tsx:139-157`).
5. Browser-level leave writes the “stopped” session-storage marker before firing a best-effort keepalive request and never checks a response (`frontend/src/pages/Teleoperation.tsx:139-150`). `TeleopStopNotice` later turns that attempted-stop marker into the affirmative message “Teleoperation stopped” (`frontend/src/components/TeleopStopNotice.tsx:17-30`).

**Impact**

A transient backend/network failure during Done can leave the control loop running after the operator has been navigated away from the only teleoperation controls. A dropped unload request can likewise leave teleoperation active while the next page explicitly tells the operator it stopped and the arm is returning/releasing.

**Safe reproduction/reasoning**

Mock `fetchWithHeaders` to reject. `stoppedRef` remains true, `stopTeleoperation()` resolves through its catch, and `handleGoBack()` navigates. For the unload branch, the marker is committed synchronously before the unacknowledged request. No hardware is required.

### P1 — Rest-pose capture and return failures are discarded, then classified as a clean stop

**Verdict:** CONFIRMED — the per-arm return outcome is only logged (`makermodslab/teleoperate.py:250,255`), `_return_followers_to_rest` returns nothing (`teleoperate.py:288-291`), and the terminal outcome is built solely from `loop_error`/`last_cleanup_error` (`teleoperate.py:786-787`), so a `settled`/`stalled`/`comm-error` return classifies as `outcome="ok"`.
**Priority:** P1 — the graceful-landing promise can fail silently and torque is released wherever the arm stalled, with no warning in the stop response or terminal status (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. Failure to capture a starting pose returns `{}` and does not fail or warn the start response (`makermodslab/rest_pose.py:60-73`; capture is consumed at `makermodslab/teleoperate.py:673-682`).
2. `return_to_rest_pose()` explicitly reports `no-pose`, `comm-error`, `settled`, `stalled`, `ceiling`, and `cut-short` as unsuccessful outcomes (`makermodslab/rest_pose.py:76-129`, `makermodslab/rest_pose.py:149-192`).
3. `_return_one_follower_to_rest()` only logs the returned reason; `_return_followers_to_rest()` returns no result to the worker (`makermodslab/teleoperate.py:237-291`).
4. The worker's terminal error is built only from the control-loop exception or torque/disconnect cleanup. A failed rest return is absent, so a normal stop with successful torque release becomes `outcome="ok"` (`makermodslab/teleoperate.py:749-789`; `makermodslab/utils/errors.py:61-77`).
5. The UI tells the operator the arm returns to its starting position and then goes limp (`frontend/src/pages/Teleoperation.tsx:87-123`).

**Impact**

An arm can fail to reach its safe starting pose, then have torque released wherever it stalled or lost communication, with no warning in the stop response or terminal status. In bimanual mode one arm can fail while the other succeeds and the whole session still reports clean.

**Safe reproduction/reasoning**

Existing tests prove all failure reasons and explicitly prove that one arm's raised return error is swallowed (`tests/test_teleoperate.py:612-729`, `tests/test_teleoperate.py:874-898`). There is no worker-level assertion that any of those results affect `last_session_error` or `outcome`.

### P1 — A timed-out second stop orphans a live worker and permits a conflicting new session

**Verdict:** CONFIRMED — a still-alive worker is nulled after the 5 s join (`makermodslab/teleoperate.py:873-877`); a later start passes both guards (`teleoperate.py:571-573`) and clears `_release_now` (`teleoperate.py:594`) while the orphan's finally later overwrites the process globals (`teleoperate.py:789-792`).
**Priority:** P1 — P0-arguable: low-probability (hung cleanup plus a start inside the window) but high-blast-radius hardware-ownership and state-corruption race (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. A second stop sets `_release_now` and waits five seconds (`makermodslab/teleoperate.py:871-876`).
2. If the worker is still alive, the handler sets `teleoperation_thread = None` even though that thread has not exited (`makermodslab/teleoperate.py:877-887`).
3. A later start rejects only an active session or a non-`None`, alive worker. The orphan is therefore invisible, and start clears `_release_now` for the new session (`makermodslab/teleoperate.py:569-596`).
4. The old worker later writes process-global `teleoperation_active = false`, `releasing = false`, and clears the current-device globals during its cleanup (`makermodslab/teleoperate.py:789-794`).

**Impact**

The new start can race the old worker for the same serial ports. If the new start succeeds on other ports, the old worker can still terminate the new session's global active state and erase its current-device state. This is both a hardware-ownership and process-state corruption path.

**Safe reproduction/reasoning**

A mocked worker whose `is_alive()` remains true past `join(timeout=5)` is sufficient. The second stop nulls the only worker reference; a subsequent start passes the worker check and clears the abort event while the mock remains alive.

### P1 — Cross-feature hardware ownership is incomplete and not atomic

**Verdict:** CONFIRMED — separate per-module locks make check-then-set non-atomic (`makermodslab/teleoperate.py:102`, `makermodslab/record.py:220`, `makermodslab/rollout.py` `_state_lock`); inference checks only `teleoperation_active`, not releasing/worker-alive (`rollout.py:899`); calibration, auto-calibration, and wiggle have no cross-feature guard at all (`calibrate.py:237`, `auto_calibrate.py:201,506`, `wiggle.py:62-67`).
**Priority:** P1 — strengthened by the missing mutex (see the Verification note): calibration and wiggle are genuinely unguarded against live sessions, and the existing record/inference↔teleop checks are racy and miss the releasing window (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. Teleoperation, recording, and inference each use a different module-local `_state_lock` (`makermodslab/teleoperate.py:101-103`, `makermodslab/record.py:213-215`, `makermodslab/rollout.py:109`). Each checks the other modules' booleans and then claims its own flag under only its own lock (`makermodslab/teleoperate.py:572-596`, `makermodslab/record.py:463-494`, `makermodslab/rollout.py:899-930`). Two simultaneous starts can therefore both observe the other feature as inactive before either writes its flag.
2. Inference blocks only on `teleoperation_active`, not on the teleoperation worker/releasing state (`makermodslab/rollout.py:899-930`). A normal stop clears `teleoperation_active` before the worker returns the arm and releases the port (`makermodslab/teleoperate.py:843-869`).
3. Manual calibration, single auto-calibration, and batch auto-calibration check only their own managers, not teleoperation (`makermodslab/calibrate.py:232-276`, `makermodslab/auto_calibrate.py:199-252`, `makermodslab/auto_calibrate.py:504-586`). Wiggle likewise opens and commands the requested serial bus without a feature-ownership guard (`makermodslab/wiggle.py:55-85`).
4. `SingleTabGuard` takeover only overlays the old tab and changes tab election; it does not stop a running teleoperation session (`frontend/src/components/SingleTabGuard.tsx:64-89`, `frontend/src/components/SingleTabGuard.tsx:109-140`). The new primary tab can reach another hardware feature while the old teleoperation page remains mounted underneath its overlay.

**Impact**

Two robot-driving features can claim the same follower/leader ports or issue competing motor commands. The sequential happy-path checks do not provide a global mutex, and the normal post-Done window exposes `teleoperation_active=false` while the arm is still energized and moving home.

**Safe reproduction/reasoning**

The race can be demonstrated with barriers around two mocked start handlers because the check-and-set operations are protected by different locks. The releasing gap is direct: inference accepts once the active boolean is false even if `teleoperation_thread.is_alive()` and `releasing` are true. No real process or serial device is needed.

### P1 — Arm-identity read failure silently fails open before MakerMods Lab writes calibration and drives the arm

**Verdict:** CONFIRMED-WITH-CORRECTIONS — the behavior exists as described (`verify_arm` returns `(None, None)` on a read failure, `makermodslab/arm_identity.py:286-288`), but it is a deliberate, documented, tested design tradeoff (docstring at `arm_identity.py:271-272`; contract pinned by `tests/test_arm_identity.py`), not an oversight.
**Priority:** P2 — deliberate, documented, tested fail-open design — flagged as a fail-open-vs-fail-closed product decision to revisit (given the F.1 swapped-port incident), not a code bug (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. If either `Present_Position` or `Homing_Offset` reading fails, `verify_arm()` logs and returns `(None, None)`: no refusal and no warning (`makermodslab/arm_identity.py:252-288`).
2. `verify_devices()` therefore treats the arm as verified enough to continue (`makermodslab/arm_identity.py:391-413`).
3. Teleoperation immediately writes the assigned calibration, configures motors, and starts control in both single and bimanual paths (`makermodslab/teleoperate.py:513-545`, `makermodslab/teleoperate.py:638-667`, `makermodslab/teleoperate.py:697-711`).

**Impact**

The one guard intended to catch swapped or misassigned arms can disappear silently on a transient read failure. MakerMods Lab can then stamp the selected calibration into an unverified physical arm and transmit leader actions with no operator-facing warning.

**Safe reproduction/reasoning**

The existing test explicitly requires this fail-open contract: a bus raising `ConnectionError("no response")` produces neither refusal nor warning (`tests/test_arm_identity.py:505-521`). That contradicts a hardware-safety boundary where an unverified device must be flagged before calibration writes and motion.

### P1 — A saved low-power setting can fail to apply while teleoperation starts at previous or full power

**Verdict:** CONFIRMED-WITH-CORRECTIONS — the mechanism is real (per-motor write failures leave the previous limit, `makermodslab/motor_power.py:149-164`; start proceeds with `success: true`, `makermodslab/teleoperate.py:660,807-808`), but the warning is surfaced as a start toast (`frontend/src/components/landing/RobotConfigManager.tsx:135-140`), not silently dropped — the residual gap is timing (motion can begin before the toast is read).
**Priority:** P2 — warned via toast, degraded-but-safe by design, and needs a real per-motor write failure (with `num_retry=2`) to trigger (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. The selected robot's saved `motor_power` is included in the start request (`frontend/src/components/landing/RobotConfigManager.tsx:104-125`).
2. `apply_motor_power()` catches per-motor write failures and returns only warnings; failed motors retain their previous limit, explicitly full power after power-up (`makermodslab/motor_power.py:136-164`).
3. Teleoperation appends those warnings but proceeds to capture the rest pose, start the worker, and return `success: true` (`makermodslab/teleoperate.py:537-546`, `makermodslab/teleoperate.py:659-681`, `makermodslab/teleoperate.py:796-811`).
4. The frontend shows a warning toast only after the backend has already started the worker and navigates into teleoperation (`frontend/src/components/landing/RobotConfigManager.tsx:127-147`).

**Impact**

An operator who selected a gentler 10–40% profile can receive immediate follower motion from one or more motors at their previous setting or 100%. The saved hardware-safety setting is therefore advisory rather than enforced.

**Safe reproduction/reasoning**

The unit suite explicitly asserts “failure warns but does not abort” and leaves the failing motor unwritten (`tests/test_motor_power.py:95-108`). No hardware is needed.

### P1 — Startup cleanup can report an ordinary setup error while discarding “torque may still be enabled” failures

**Verdict:** CONFIRMED — both cleanup call sites discard `_safe_disconnect`'s returned torque warning (`makermodslab/teleoperate.py:546-547,814-815`; warning text at `teleoperate.py:462-465`), and the startup-failure path never runs the per-motor `force_disable_torque` hardening (that exists only in the worker's finally, `teleoperate.py:774-775`).
**Priority:** P1 — top P1: failed startup can strand an energized follower and drops the unplug warning; F.1 incident precedent (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. `_safe_disconnect()` returns a safety warning if disconnect fails, including “TORQUE MAY STILL BE ENABLED” (`makermodslab/teleoperate.py:450-475`).
2. Setup has already written calibration and configured the follower before several later operations can fail (`makermodslab/teleoperate.py:648-681`; bimanual equivalent `makermodslab/teleoperate.py:529-545`).
3. Both the bimanual connector's exception path and the outer start exception path call `_safe_disconnect()` but discard its returned warning (`makermodslab/teleoperate.py:547-550`, `makermodslab/teleoperate.py:813-824`).
4. The API response contains only the original setup exception and does not populate `last_cleanup_error` or a terminal warning.

**Impact**

If start fails after follower configuration and cleanup also fails, the follower may remain energized or its port may remain in an uncertain state. The operator sees only the original startup error, not the explicit unplug-power warning generated by cleanup.

**Safe reproduction/reasoning**

Mock a post-configuration operation to raise and the follower's `disconnect()` to raise. `_safe_disconnect()` constructs the safety text, but the handler discards the return value and returns `str(original_exception)`.

### P2 — The ordinary interface removes the advertised second-stop “release now” control

**Verdict:** CONFIRMED — the backend advertises “Press Stop again to release it now” (`makermodslab/teleoperate.py:862-866`), but the page's single Done control latches the one-shot guard and navigates home (`frontend/src/components/control/VisualizerPanel.tsx:34-39`, `frontend/src/pages/Teleoperation.tsx:73,159-162`); no second-stop control exists anywhere in the UI.
**Priority:** P2 — the operator cannot abort the gentle progress-based return from the UI and must wait out the 10 s ceiling or cut power (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. The first backend stop responds `releasing: true` and instructs “Press Stop again to release it now” (`makermodslab/teleoperate.py:860-869`).
2. The page has one Done control (`frontend/src/components/control/VisualizerPanel.tsx:29-39`).
3. Done sets the one-shot stop guard and immediately navigates home after the first response (`frontend/src/pages/Teleoperation.tsx:71-78`, `frontend/src/pages/Teleoperation.tsx:159-162`).
4. The landing page exposes no second teleoperation-stop control. The second-stop behavior remains API-only.

**Impact**

If the automatic return is moving toward an obstacle, pinching something, or otherwise should be aborted, the operator cannot perform the action the backend message tells them to perform. They must wait for the return/stall ceiling or physically remove power.

**Safe reproduction/reasoning**

The frontend control flow deterministically leaves the route after the first response and latches `stoppedRef`; no browser/hardware probe is required.

### P2 — The teleoperation route can display “Live Robot Data” while no session exists

**Verdict:** CONFIRMED — the status poll only transitions on an inactive `failed`/`ran_with_warning` outcome (`frontend/src/pages/Teleoperation.tsx:47-51`), the badge is driven purely by socket connectivity (`frontend/src/components/UrdfViewer.tsx:337-347`), and `/ws/joint-data` accepts connections regardless of teleoperation state (`makermodslab/server.py`, `websocket_endpoint`).
**Priority:** P2 — misleading idle-state display; no hardware effect (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. `/teleoperation` is directly routable and does not start or attach to a named session (`frontend/src/App.tsx:40-49`).
2. Status polling only transitions the page when it sees an inactive terminal `failed` or `ran_with_warning` outcome; inactive `None` or `ok` remains rendered indefinitely (`frontend/src/pages/Teleoperation.tsx:38-70`).
3. The URDF badge labels an open shared WebSocket “Live Robot Data” solely from socket connectivity (`frontend/src/components/UrdfViewer.tsx:332-348`). The WebSocket accepts connections regardless of teleoperation state (`makermodslab/server.py:806-835`).

**Impact**

A bookmark, reload after a clean session, or direct navigation can show a normal teleoperation screen and green “Live Robot Data” badge even though no arm session is active and no joint updates are being produced. Done then receives “No teleoperation session is active,” which the page does not surface.

**Safe reproduction/reasoning**

Open the route with backend state `teleoperation_active=false`, `releasing=false`, `outcome=None`. The poll has no matching branch, while the shared WebSocket can still connect.

### P2 — Saved-camera preview can throw instead of flagging an unavailable browser camera API

**Verdict:** CONFIRMED — the hook dereferences `navigator.mediaDevices.addEventListener` unguarded in its first effect (`frontend/src/hooks/useCameraStream.ts:43`) and `getUserMedia` likewise (`useCameraStream.ts:60`); with `mediaDevices` absent (plain-HTTP LAN origin, unsupported browser) the effect throws a `TypeError` before `hasError` can be set, and no error boundary wraps the panel.
**Priority:** P2 — requires a non-secure origin or unsupported browser; HTTPS is the documented camera path (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. The panel correctly maps only the selected record's saved camera IDs; it does not rebind them (`frontend/src/components/control/TeleopCameraPanel.tsx:25-36`, `frontend/src/components/control/TeleopCameraPanel.tsx:67-83`).
2. When a feed mounts, `useCameraStream` unconditionally calls `navigator.mediaDevices.addEventListener(...)` before checking whether `mediaDevices` or `getUserMedia` exists (`frontend/src/hooks/useCameraStream.ts:35-46`).
3. The later `getUserMedia` call is also unguarded for an unavailable API (`frontend/src/hooks/useCameraStream.ts:48-99`).
4. Plain HTTP on a non-local LAN origin and unsupported browsers can expose no `navigator.mediaDevices` at all.

**Impact**

Turning Cameras on can raise a `TypeError` in the React effect instead of leaving the configured role visible with “Preview failed.” The motor loop remains backend-independent, but the teleoperation interface can crash or lose its controls precisely when the user asks to view a saved camera.

**Safe reproduction/reasoning**

Render a configured `CameraFeed` with `navigator.mediaDevices` undefined. The first effect dereferences `addEventListener` before the hook can set `hasError`.

### P2 — The active hardware session and rendered robot/cameras do not share an immutable session identity

**Verdict:** CONFIRMED — start snapshots one record into the request (`frontend/src/components/landing/RobotConfigManager.tsx:104-125`) while the page renders layout and cameras from the live, mutable `selectedRecord` (`frontend/src/pages/Teleoperation.tsx:15-16`, `frontend/src/components/control/TeleopCameraPanel.tsx:31-36`), and status returns no session descriptor (`makermodslab/teleoperate.py:908-929`).
**Priority:** P2 — edge case (record renamed/deleted/changed mid-session); UI/telemetry mismatch only, no hardware misbehavior (verified 2026-07-14 against the current working tree)

**Broken pathway**

1. Start snapshots one robot record into the request (`frontend/src/components/landing/RobotConfigManager.tsx:104-125`).
2. The teleoperation page independently derives single/bimanual layout and camera feeds from the current module-level `selectedRecord` (`frontend/src/pages/Teleoperation.tsx:13-16`, `frontend/src/components/control/TeleopCameraPanel.tsx:25-36`).
3. Robot records are refreshed on location changes, and the selected name is mutable browser state (`frontend/src/hooks/useRobots.ts:29-84`, `frontend/src/hooks/useRobots.ts:96-110`, `frontend/src/hooks/useRobots.ts:273-294`).
4. Teleoperation status returns no session ID, robot name, mode, or camera snapshot (`makermodslab/teleoperate.py:903-930`).

**Impact**

If the selected record is renamed/deleted/changed from another client, or a direct API start is followed by route navigation with a different selected record, the hardware continues using the original request while the page can render the wrong arm count and wrong saved-camera roles/IDs. It does not silently rebind a saved camera within a record, but it can bind the interface to the wrong record altogether.

**Safe reproduction/reasoning**

Start a mocked bimanual session from record A, then make the shared selected record resolve to single-arm record B. The page chooses one viewer and B's cameras; the backend worker still owns A's four-arm request.

## Design gaps and lower-confidence risks

### No server-owned session token or immutable session descriptor

Start, stop, status, and WebSocket updates are process-global. There is no session token to prevent an old page, delayed stop, delayed 13-second cleanup check, or another client from acting on a newer session. Adding an active session descriptor would also resolve the confirmed layout/camera mismatch instead of asking the page to infer ownership from mutable selection.

### Direct API requests bypass saved-record validation

`TeleoperateRequest` accepts arbitrary ports/configs, defaults omitted power to 100, accepts any mode string (anything other than exact `bimanual` follows the single path), and exposes `skip_identity_check` (`makermodslab/teleoperate.py:294-314`, `makermodslab/teleoperate.py:605-609`). The ordinary frontend sends the saved record correctly, so this is an API policy gap rather than a normal-interface defect. Decide whether `/move-arm` is a low-level expert API or should reload/validate the named saved record.

### Camera MJPEG behavior remains browser-negotiated and unverified on target hardware

The current teleoperation camera path is browser-only and correctly uses exact saved `device_id` values. It requests ideal 1280×720 at 30 FPS specifically to encourage Chromium to negotiate MJPEG (`frontend/src/hooks/useCameraStream.ts:58-76`). Whether that choice behaves correctly on every target UVC camera is hardware/browser-dependent; the audit did not classify it as a defect without fresh runtime evidence. It must not be “fixed” by selecting another camera or changing the saved ID.

### WebSocket connectivity is not robot-data freshness

The UI has no last-joint-message age. Even during a real session, a connected shared socket can remain green when broadcasts stop or when only non-joint events are flowing. The direct-route issue above is confirmed; broader data-staleness semantics need a product decision and browser tests.

## Behaviors checked and not found defective

- Teleoperation backend configurations contain no LeRobot/OpenCV cameras; browser previews cannot block the motor loop (`makermodslab/utils/robot_factory.py:49-79`, `makermodslab/utils/robot_factory.py:82-119`).
- The teleoperation camera panel uses the saved robot camera list and exact saved `device_id` values. It does not enumerate or silently substitute another camera (`frontend/src/components/control/TeleopCameraPanel.tsx:25-36`; `frontend/src/hooks/useCameraStream.ts:58-76`).
- Single and bimanual paths both run the identity guard before calibration writes and apply motor power only to follower arms when those operations succeed (`makermodslab/teleoperate.py:478-546`, `makermodslab/teleoperate.py:638-667`).
- Mid-loop action/read failures reach a terminal `failed` status and skip the rest return, then attempt torque disable/disconnect (`makermodslab/teleoperate.py:709-794`).
- Bimanual rest returns run concurrently and wait for both return threads before torque release (`makermodslab/teleoperate.py:258-291`).

## Focused validation

Command:

```text
.venv/bin/python -m pytest -p no:cacheprovider \
  tests/test_teleoperate.py \
  tests/test_arm_identity.py \
  tests/test_motor_power.py \
  tests/test_devices.py -q
```

Result: **96 passed, 5 warnings in 2.17 seconds**.

The suites are mock/temp-backed and did not open hardware. Passing results do not invalidate the findings: current tests explicitly require identity-read failure to fail open, motor-power write failure not to abort, unsuccessful rest returns not to raise, and a failing bimanual return not to propagate from the wrapper.

## Missing regression coverage

- Done/network failure must keep the page in a recoverable state, warn the operator, and retry/confirm stop before navigation.
- Browser unload notice must distinguish “stop requested” from server-confirmed “stopped.”
- Rest-pose capture/return failure must reach the stop response and terminal outcome, including one-arm bimanual failure.
- A worker still alive after second-stop timeout must remain owned and block every new robot-driving start.
- One global hardware-session arbiter must make teleoperation/recording/inference/calibration/auto-calibration/wiggle ownership atomic and include releasing workers.
- Identity read failure must produce an operator-visible warning or refusal before any calibration write.
- Saved motor-power application failure needs an explicit safety policy and a test that prevents unacknowledged full-power motion.
- Startup cleanup failure must preserve and surface `_safe_disconnect()` torque warnings.
- The ordinary page must retain an accessible release-now control while `releasing=true`.
- Direct-route/idle status must redirect or render an explicit “no active session” state.
- Camera hooks must handle absent `navigator.mediaDevices` as a saved-device preview error without crashing.
- Status should include immutable active-session robot/mode/camera metadata, with tests for record changes during a session.
- Frontend tests are missing for teleoperation Done, unload, camera API absence, selected-record drift, and direct-route state.

## Audit safety and repository changes

- No production or test files were edited.
- No files were staged and no commit, push, branch change, stash, worktree, or reset was performed.
- This report is the only intentional filesystem edit for this audit and is gitignored under `internal_docs/`.
- No serial port or camera device/stream was opened, no arm was driven, and no motor command was sent.
- No calibration, Hugging Face cache, credentials, datasets, models, or other user data was modified.
- No external service was called.
