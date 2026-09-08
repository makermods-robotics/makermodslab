# Teleoperation Interface Pathways

This document inventories how an operator selects a robot, starts teleoperation, views live state and cameras, stops or leaves a session, and recovers from errors through MakerMods Lab. It describes the current implementation without proposing alternative behavior.

This inventory was audited on July 14, 2026, against MakerMods Lab commit `518ca56`.

## Core teleoperation model

MakerMods Lab teleoperation reads actions from one SO-101 leader and sends them to one SO-101 follower, or controls two leader/follower pairs as one bimanual BiSO session. The backend owns arm connections, calibration application, the control loop, rest-pose return, torque release, and terminal outcomes. Cameras are separate browser-side previews and are never opened by the teleoperation backend.

The major backend states are:

| State | Meaning | Operator-facing behavior |
| --- | --- | --- |
| **Idle** | `teleoperation_active=false` and `releasing=false`. | A clean saved robot can start teleoperation. |
| **Starting** | The start request has claimed `teleoperation_active=true` and is synchronously connecting and configuring hardware. | The landing page waits for `POST /move-arm`; there is no separate start-progress page or phase payload. |
| **Running** | The worker repeatedly reads leader actions and sends them to the follower. | The teleoperation page shows live URDF state and optional browser camera feeds. |
| **Returning and releasing** | `teleoperation_active=false` and `releasing=true` while a normal stop returns the followers to their captured starting pose. | The control loop is over, but follower torque and serial ownership remain until return and cleanup finish. |
| **Terminal** | Both flags are false and the most recent outcome is `ok`, `ran_with_warning`, or `failed`. | A warning or failure can be surfaced by status polling; a normal clean stop returns the user to the landing page. |

Source: `makermodslab/teleoperate.py:81-112`, `makermodslab/teleoperate.py:553-596`, `makermodslab/teleoperate.py:903-931`.

## Entry and saved-configuration pathway

The normal entry point is the **Teleoperation** button on the selected robot profile on the landing page.

1. Robot records are loaded from `GET /robots`.
2. The selected robot name is persisted in browser local storage under `makermodslab.selectedRobot`.
3. The module-level `useRobots` store shares the selected record among the landing page, teleoperation page, inference controls, and other mounted consumers.
4. The Teleoperation button is enabled only when the selected record has `is_clean=true`.
5. A robot is clean when every port and calibration field required by its single-arm or bimanual mode is populated and each referenced calibration file exists. Cameras are optional.
6. Pressing Teleoperation snapshots the selected record into a `POST /move-arm` request.
7. The frontend navigates to `/teleoperation` only when the response is HTTP-successful and also contains `success: true`.

The normal request carries:

- Primary leader and follower serial ports.
- Primary leader and follower calibration names.
- The saved mode, `single` or `bimanual`.
- Right leader and follower ports and calibrations for a bimanual robot.
- The robot record name, used as the bimanual calibration-staging base ID.
- The saved follower motor-power percentage.

The backend does not reload the robot record by name for a single-arm start. It consumes the request snapshot. For bimanual starts, `robot_name` chooses the staging directory and alias names; it does not decide which calibration belongs to each arm. Direct API callers can construct a `TeleoperateRequest`, but the ordinary interface is driven by the selected saved robot record.

Source: `frontend/src/hooks/useRobots.ts:29-84`, `frontend/src/hooks/useRobots.ts:273-294`, `frontend/src/components/landing/RobotTile.tsx:76-91`, `frontend/src/components/landing/RobotTile.tsx:192-221`, `frontend/src/components/landing/RobotConfigManager.tsx:104-162`, `makermodslab/utils/config.py:473-506`, `makermodslab/server.py:2368-2392`.

## Request shape and defaults

`TeleoperateRequest` requires the primary leader/follower ports and calibration names. Its additional defaults are:

- `mode="single"`.
- Empty right-arm ports and calibration names.
- Empty `robot_name`.
- `skip_identity_check=false`.
- `motor_power=100` for a direct API request that omits it.

The normal frontend supplies the selected robot's saved motor power and does not expose the identity-skip flag.

Source: `makermodslab/teleoperate.py:294-314`, `frontend/src/components/landing/RobotConfigManager.tsx:109-125`.

## Start ownership and mutual exclusion

Before opening hardware, teleoperation asks both teleoperation and recording to finish any pending post-stop release. A pending release is cut short through the shared release-now event so the previous worker can free its serial ports.

Under the teleoperation state lock, a start is refused when:

- Teleoperation is already active.
- A previous teleoperation worker is still releasing the arms.
- Recording is active.
- Inference is active.

The start then clears the prior cleanup error, terminal outcome, terminal error, releasing flag, and stale release-now event before connecting hardware. The active flag is claimed before synchronous hardware setup, so other recording, inference, or teleoperation requests see the slot as occupied while connection and configuration are in progress.

Source: `makermodslab/teleoperate.py:210-234`, `makermodslab/teleoperate.py:553-596`.

## Single-arm start pathway

A single-arm start follows this sequence:

1. Resolve the selected leader and follower calibration names.
2. Build one `SO101FollowerConfig` and one `SO101LeaderConfig` without backend cameras.
3. Connect the follower motor bus.
4. Connect the leader motor bus.
5. Run the read-only arm-identity guard.
6. Write the selected calibration into each bus.
7. Configure the follower and leader devices.
8. Apply the saved RAM `Torque_Limit` to the follower motors only.
9. Clear any follower `Goal_Velocity` cap left by a previous arm-driving feature.
10. Capture the follower's raw starting pose, excluding the gripper.
11. Start the background control-loop worker.

Follower connection occurs first. If it fails, the response names the follower and its requested port. If the leader connection fails afterward, MakerMods Lab disconnects the already-opened follower before returning an error.

Source: `makermodslab/utils/robot_factory.py:49-79`, `makermodslab/teleoperate.py:608-681`.

## Bimanual start pathway

For a bimanual robot, the primary fields represent the left pair and the `right_*` fields represent the right pair.

MakerMods Lab first stages the four arbitrary library calibration names into the naming convention required by LeRobot's BiSO devices:

```text
~/.cache/huggingface/lerobot/makermodslab_biso/<robot>/leader/<robot>_left.json
~/.cache/huggingface/lerobot/makermodslab_biso/<robot>/leader/<robot>_right.json
~/.cache/huggingface/lerobot/makermodslab_biso/<robot>/follower/<robot>_left.json
~/.cache/huggingface/lerobot/makermodslab_biso/<robot>/follower/<robot>_right.json
```

These staging aliases are overwritten on every session so recently recalibrated library files are refreshed.

The hardware sequence is:

1. Build one `BiSOFollowerConfig` and one `BiSOLeaderConfig`.
2. Connect left follower, right follower, left leader, then right leader.
3. Verify all four arms using their real saved calibration stems rather than their BiSO staging aliases.
4. Write calibration to all four buses.
5. Configure the two BiSO devices.
6. Apply motor power and clear speed caps on both follower arms.
7. Capture both follower starting poses.
8. Start one worker controlling the paired devices.

Source: `makermodslab/utils/config.py:537-615`, `makermodslab/utils/robot_factory.py:82-119`, `makermodslab/teleoperate.py:478-550`, `makermodslab/teleoperate.py:605-606`, `makermodslab/teleoperate.py:673-695`.

## Arm-identity pathway

Identity verification runs after serial connection but before calibration writes, torque-enabling configuration, or action transmission. It reads present positions and EEPROM homing offsets without writing to the servos.

The decision pathway for each arm is:

1. **Fingerprint matches the assigned calibration:** start without an identity warning. Position fallback is skipped.
2. **Fingerprint matches another arm assigned to this same session:** refuse the start and identify the likely swapped ports and counterpart calibration.
3. **Fingerprint matches a different saved library calibration:** start with a named warning explaining the difference.
4. **Fingerprint matches no saved calibration:** compare present positions to the assigned ranges.
   - If at least three position-informative joints are materially outside their ranges, refuse the start.
   - Otherwise start with an unable-to-verify warning.
5. **Identity reads fail:** log that verification could not be completed and proceed; later device operations determine whether the bus is usable.

For bimanual sessions, all four assigned slots are counterpart candidates. The backend request has a `skip_identity_check` escape hatch, but the normal frontend does not expose it.

A warn-but-allow start returns `success: true` plus a `warning`. The landing page displays that warning for ten seconds before navigating to the teleoperation page.

Source: `makermodslab/arm_identity.py:15-43`, `makermodslab/arm_identity.py:252-316`, `makermodslab/arm_identity.py:335-413`, `makermodslab/teleoperate.py:513-527`, `makermodslab/teleoperate.py:638-667`, `frontend/src/components/landing/RobotConfigManager.tsx:131-147`.

## Follower power and inherited motor state

The saved motor-power percentage is applied only to follower motors. It writes the volatile STS3215 `Torque_Limit` register after device configuration. The value lasts for the current power cycle and is written on every session, including 100 percent, so a previous gentler session does not remain in effect accidentally.

Teleoperation also writes `Goal_Velocity=0` to follower motors at startup. This removes a RAM speed cap that may have been left by auto-calibration or a prior return-to-rest sequence. Motor-power or speed-cap write failures are collected as warnings rather than aborting the session.

The human-held leader is not given follower motor-power or speed-cap writes.

Source: `makermodslab/motor_power.py:14-36`, `makermodslab/motor_power.py:53-70`, `makermodslab/motor_power.py:136-198`, `makermodslab/teleoperate.py:537-544`, `makermodslab/teleoperate.py:659-667`.

## Camera pathway

Teleoperation opens no OpenCV or LeRobot cameras. Camera viewing is optional, browser-side, and independent of the motor control loop.

1. The camera panel starts **Off** and does not request camera access merely by opening the teleoperation page.
2. Turning it on creates one feed for every camera in the selected robot's saved `cameras` list.
3. Each feed uses that camera's stored browser `device_id` with an exact `getUserMedia` constraint.
4. The teleoperation page does not enumerate cameras or select a replacement device.
5. A configured camera with a missing or invalid ID remains visible by its saved role name and shows a no-camera or failed-preview placeholder.
6. `NotReadableError` and `AbortError` are retried with exponential backoff.
7. Permission or device-exposure changes can trigger another attempt through the browser's `devicechange` event.
8. The explicit retry button remounts all feeds and starts fresh requests.
9. Browser camera tracks stop when their components unmount.

Each stream requests an ideal 1280 by 720 image at 30 FPS. These are browser preferences, not recording settings. Camera frames are never sent through the teleoperation backend or saved into a dataset.

This is a strict saved-configuration pathway: the teleoperation page uses the configured IDs and surfaces failed previews instead of silently rebinding cameras.

Source: `makermodslab/utils/robot_factory.py:23-28`, `frontend/src/components/control/TeleopCameraPanel.tsx:9-36`, `frontend/src/components/control/TeleopCameraPanel.tsx:38-89`, `frontend/src/components/control/CameraFeed.tsx:12-44`, `frontend/src/hooks/useCameraStream.ts:19-108`.

## Live control-loop pathway

Once setup succeeds, the worker owns the devices and their cleanup.

The main loop:

1. Calls `teleop_device.get_action()` to read the leader or BiSO leaders.
2. Passes that action to `robot.send_action()` on the follower or BiSO followers.
3. Sleeps approximately one millisecond before the next iteration.
4. Approximately every 50 milliseconds, reads follower observations for visualization.
5. Approximately every second, samples follower `Present_Current` for power telemetry.
6. Queues a `joint_update` message when WebSocket clients are connected.

The control loop does not depend on a WebSocket or camera feed being present. Browser disconnection affects visualization, not leader-to-follower action transmission.

Source: `makermodslab/teleoperate.py:697-758`.

## Joint visualization and WebSocket pathway

Follower observations are mapped into the SO-101 URDF's joint angles using each arm's calibration range. A single-arm message carries:

- `type="joint_update"`.
- `joints` for the follower.
- A timestamp.

A bimanual message additionally carries `joints_right`. Current telemetry can be included as `follower_currents_ma`, with `left_` and `right_` prefixes for bimanual sessions. The current frontend does not render this current telemetry.

The server's shared `ConnectionManager` queues messages from the teleoperation thread and sends them on the asyncio event loop that accepted each `/ws/joint-data` connection. Each URDF viewer selects either `joints` or `joints_right` and reconnects with exponential backoff up to 30 seconds.

The viewer's green or red badge represents WebSocket connectivity. Teleoperation activity and terminal outcome come from the separate HTTP status pathway.

Source: `makermodslab/teleoperate.py:317-379`, `makermodslab/teleoperate.py:723-746`, `makermodslab/server.py:249-383`, `makermodslab/server.py:806-835`, `frontend/src/hooks/useRealTimeJoints.ts:20-135`, `frontend/src/components/control/VisualizerPanel.tsx:45-64`, `frontend/src/components/UrdfViewer.tsx:332-349`.

## Live interface and controls

The teleoperation page contains:

- A **Done** button.
- One URDF viewer for a single-arm profile.
- Left and right URDF viewers for a bimanual profile.
- The optional saved-camera panel.
- A terminal warning or failure banner when a session ends underneath the page.

There are no pause, resume, manual-jog, camera-recording, or live log controls on this page.

The page decides whether to render one or two viewers from the currently selected frontend robot record. The backend status does not return the active session's robot name, mode, or camera list.

Source: `frontend/src/pages/Teleoperation.tsx:9-31`, `frontend/src/pages/Teleoperation.tsx:164-213`, `frontend/src/components/control/VisualizerPanel.tsx:7-69`.

## Status polling and terminal outcomes

The page polls `GET /teleoperation-status` every two seconds while it still considers the session live. The status payload includes:

- `teleoperation_active`.
- `available_controls.stop_teleoperation`.
- `releasing`.
- `last_cleanup_error`.
- `outcome`.
- `error`.
- A plain-language `hint` for recognized error patterns.
- A status message.

The page looks for an inactive, non-releasing session whose outcome is `failed` or `ran_with_warning`. It then stops polling, marks the leave safety net handled, and renders a persistent terminal banner. A clean `ok` outcome does not create this banner.

Source: `makermodslab/teleoperate.py:903-931`, `frontend/src/pages/Teleoperation.tsx:23-70`, `makermodslab/utils/errors.py:48-127`.

## Normal stop and return-to-start pathway

The page's **Done** action calls `POST /stop-teleoperation`, awaits the first response, and then navigates home.

On the first stop request:

1. The backend sets `teleoperation_active=false`.
2. If the worker is still alive, the response returns immediately with `releasing=true`.
3. The worker recognizes that the control loop ended through a user-requested stop.
4. Each follower is driven toward the raw pose captured before teleoperation began.
5. The gripper is excluded so a held object is not automatically dropped during the return.
6. Bimanual follower returns run concurrently, one thread per arm.
7. Every return is progress-based, detects stalls, and has a ten-second absolute ceiling.
8. The temporary gentle return speed is reset to uncapped speed on every exit path.
9. Torque is disabled motor by motor on the followers and leaders.
10. The devices are disconnected and terminal state is recorded.

The first response explains that the arm is returning and will then go limp. The page shows that response as a toast. Because cleanup continues after the response, the page schedules a status check after 13 seconds and produces a destructive toast if `last_cleanup_error` reports that an arm may still be energized.

Source: `frontend/src/pages/Teleoperation.tsx:71-128`, `makermodslab/teleoperate.py:749-794`, `makermodslab/teleoperate.py:827-869`, `makermodslab/rest_pose.py:43-57`, `makermodslab/rest_pose.py:76-146`, `makermodslab/rest_pose.py:149-192`.

## Second-stop release-now pathway

A second backend stop while the return or release worker is still alive means **release now**:

1. Set the shared abort event.
2. Cut short any in-progress rest-pose return.
3. Wait up to five seconds for torque release and disconnect.
4. Return any cleanup warning after the worker exits.
5. If the worker remains alive after five seconds, warn that the arms may not have been released and advise unplugging power if an arm remains rigid.

This pathway exists at API level. The ordinary page marks its stop callback handled after the first request and navigates away, so it does not present a persistent second-stop control after **Done**.

Source: `makermodslab/teleoperate.py:871-900`, `frontend/src/pages/Teleoperation.tsx:71-73`, `frontend/src/pages/Teleoperation.tsx:159-162`.

## In-app navigation pathway

The page guards its stop callback with `stoppedRef` so the Done button, route unmount, and other in-app navigation do not intentionally send duplicate ordinary stops.

- **Done:** await stop, then navigate to `/`.
- **Any other in-app navigation:** unmount cleanup calls the guarded stop function.
- **Session already ended with warning or failure:** the status poll marks the stop guard handled so unmount does not send a stop to an already terminal session.

Source: `frontend/src/pages/Teleoperation.tsx:18-21`, `frontend/src/pages/Teleoperation.tsx:52-59`, `frontend/src/pages/Teleoperation.tsx:130-162`.

## Reload, tab-close, and browser-leave pathway

Browser-level departures use `pagehide` because React cleanup may not finish during unload.

1. Store `makermodslab:teleop-stopped=1` in session storage.
2. Send a bare `POST /stop-teleoperation` with `keepalive=true`.
3. Avoid a JSON content type so the unload request remains a CORS-simple request and does not require a preflight.
4. On the next page load, `TeleopStopNotice` consumes the marker and displays a toast explaining that the arm returns to its starting position and then goes limp.

A reload therefore follows the stop pathway rather than reattaching the refreshed page to the existing live session.

Source: `frontend/src/pages/Teleoperation.tsx:130-157`, `frontend/src/components/TeleopStopNotice.tsx:4-35`, `frontend/src/App.tsx:37-54`.

## Start-error and retry pathway

Hardware and setup work stays synchronous until devices are configured and the worker starts. Consequently, connection and setup failures return to the landing-page request rather than failing later on an empty teleoperation page.

On a start failure MakerMods Lab:

1. Attempts to disconnect any follower device that was created or connected.
2. Attempts to disconnect any leader device that was created or connected.
3. Clears the active flag and current device globals.
4. Returns `success: false` with the error message.
5. Leaves the user on the landing page with an error toast.

There is no automatic arm-start retry. After correcting the port, power, calibration, or ownership condition, the operator presses **Teleoperation** again.

Source: `makermodslab/teleoperate.py:611-646`, `makermodslab/teleoperate.py:813-824`, `frontend/src/components/landing/RobotConfigManager.tsx:127-160`.

## Mid-loop failure pathway

An exception from reading the leader action or sending the follower action ends the worker independently of the frontend.

The worker then:

1. Formats the caught exception into the terminal error field.
2. Classifies the outcome as `failed`.
3. Skips return-to-start motion because communication or hardware state may be uncertain.
4. Immediately attempts torque disable and device disconnect.
5. Clears active and current-device state.

The status poll detects this terminal outcome and displays a red failure banner with the raw error and any recognized friendly hint.

Source: `makermodslab/teleoperate.py:749-789`, `makermodslab/utils/errors.py:48-77`, `frontend/src/pages/Teleoperation.tsx:38-70`, `frontend/src/pages/Teleoperation.tsx:164-204`.

## Cleanup-warning pathway

After any worker exit, MakerMods Lab disables torque on every motor individually before calling the device's normal disconnect. If disconnect raises, it also force-closes underlying serial resources as a last resort.

When normal teleoperation ran and the user requested stop, but torque disable or disconnect reported a problem, the terminal outcome is `ran_with_warning`. The warning text explicitly says torque may remain enabled and advises unplugging the arm's power if it remains rigid.

The interface can surface this through:

- The immediate stop response when cleanup has already completed.
- The delayed 13-second post-stop status check.
- The amber terminal banner when a cleanup warning appears while the page remains mounted.

Source: `makermodslab/teleoperate.py:405-475`, `makermodslab/teleoperate.py:772-794`, `makermodslab/teleoperate.py:852-898`, `makermodslab/utils/devices.py:16-51`, `frontend/src/pages/Teleoperation.tsx:79-123`, `frontend/src/pages/Teleoperation.tsx:164-204`.

## Cross-feature transitions

### Recording

Teleoperation and recording mutually refuse one another while active. Either start path also asks both features to finish pending post-stop releases before attempting to claim the serial ports.

Source: `makermodslab/teleoperate.py:569-587`, `makermodslab/record.py:457-476`.

### Inference

Teleoperation refuses to start while inference is active. Inference likewise refuses while `teleoperation_active` is true. These are feature-local active flags around the common follower serial hardware.

Source: `makermodslab/teleoperate.py:585-587`, `makermodslab/rollout.py:887-930`.

### Manual and automatic calibration

Manual calibration, single-arm auto-calibration, and batch auto-calibration have their own managers and activity state. The ordinary interface reaches calibration from the landing page, while leaving the teleoperation route runs the teleoperation stop safety net first.

Source: `makermodslab/calibrate.py:147-167`, `makermodslab/calibrate.py:232-250`, `makermodslab/auto_calibrate.py:180-252`, `makermodslab/auto_calibrate.py:480-588`, `frontend/src/components/landing/RobotConfigManager.tsx:100-102`.

### Training and dataset operations

Local or cloud training jobs and dataset download, upload, merge, inspection, and other storage operations do not use the teleoperation active flag. They may proceed independently because they do not participate in the leader-to-follower control loop.

Job progress, job-change events, and joint updates share `/ws/joint-data`. Each frontend consumer filters the message type or joint field relevant to it.

Source: `makermodslab/server.py:376-405`, `frontend/src/hooks/useRealTimeJoints.ts:80-89`, `frontend/src/hooks/useJobsChangedSignal.ts:6-36`.

### Robot-record edits during a session

The backend hardware session uses the request snapshot already used to construct its devices. Later robot-record edits do not rebuild an active teleoperation session. The page's layout and camera panel independently derive their display from the selected frontend robot record because teleoperation status does not carry session robot metadata.

Source: `frontend/src/pages/Teleoperation.tsx:13-16`, `frontend/src/components/control/TeleopCameraPanel.tsx:25-36`, `makermodslab/teleoperate.py:903-931`.

### Multiple browser tabs

`SingleTabGuard` uses `BroadcastChannel` heartbeats to elect one primary browser tab. Secondary tabs remain covered by an overlay until they take over, at which point the previous tab becomes secondary. This is a frontend control-coordination layer; backend teleoperation ownership remains the feature's process-global active state.

Source: `frontend/src/components/SingleTabGuard.tsx:18-43`, `frontend/src/components/SingleTabGuard.tsx:45-117`, `frontend/src/components/SingleTabGuard.tsx:119-140`.

## Direct routes and API access

The normal journey starts on landing and navigates to `/teleoperation` only after a successful start. The route itself is directly addressable and renders from the selected frontend robot record; it does not perform a separate start request.

The teleoperation API surface is:

| Endpoint | Purpose |
| --- | --- |
| `POST /move-arm` | Validate the request, connect and configure arms synchronously, then start the worker. |
| `POST /stop-teleoperation` | Signal an ordinary stop or, if already returning, request immediate release. |
| `GET /teleoperation-status` | Return active/releasing state, terminal outcome, error, hint, and cleanup warning. |
| `WS /ws/joint-data` | Broadcast live joint updates alongside other application events. |

Source: `frontend/src/App.tsx:40-52`, `makermodslab/server.py:503-518`, `makermodslab/server.py:806-835`.

## Expected operating environment

The implemented pathway assumes:

- One supervised operator at the robot bench.
- One primary MakerMods Lab browser tab.
- One saved SO-101 single-arm or bimanual robot profile as the source of ports, calibrations, camera IDs, mode, and motor power.
- Browser-side camera viewing on the same workstation that owns the saved camera identities.
- Backend process-global feature state rather than per-user teleoperation sessions.
- A normal stop that returns followers to the position where the session began, then releases torque.

Teleoperation itself does not require Hugging Face authentication, dataset selection, model selection, or internet access.

## Validation boundary

The following focused mocked suites were used to validate the mapped backend behavior:

- `tests/test_teleoperate.py`
- `tests/test_arm_identity.py`
- `tests/test_motor_power.py`
- `tests/test_devices.py`

They passed 96 tests covering request defaults, connection cleanup, camera-free backend configurations, identity decisions, single and bimanual torque handling, rest-pose return, second-stop behavior, terminal status, power telemetry, and serial force-close fallback.

The audit did not operate real serial arms or cameras and did not exercise real browser `getUserMedia`, page-unload delivery, or a browser-to-hardware end-to-end session.
