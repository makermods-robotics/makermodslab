# Recording Interface Pathways

This document inventories how an operator can configure, start, control, finish, recover, and reuse a recording session through MakerMods Lab. It separates behavior implemented in the current interface from important missing or incomplete pathways.

This inventory was audited on July 14, 2026, against MakerMods Lab commit `518ca56`.

## Core recording model

MakerMods Lab records episodic LeRobot datasets from one SO-101 leader/follower pair or a bimanual pair. A session alternates between an active recording phase and an unrecorded reset phase.

The frontend is the operator console, but the backend is the source of truth. The recording page polls backend status and logs once per second, so the displayed phase and episode count follow the worker even when a frontend action is optimistic or briefly delayed.

The major session states are:

| State | Meaning | Operator-facing behavior |
| --- | --- | --- |
| **Preparing** | MakerMods Lab is validating the request and preparing paths and camera ownership. | Start controls remain unavailable. |
| **Connecting arm and cameras** | The follower arm and configured cameras are being opened. | Progress text identifies the hardware stage. |
| **Connecting leader** | The leader arm is being connected after the follower identity check. | Progress text identifies the hardware stage. |
| **Recording** | Robot observations, actions, and camera frames are being added to the current episode buffer. | The operator can end or re-record the episode. |
| **Resetting** | The operator resets the scene between saved episodes. No dataset frames are collected. | The operator can begin the next episode early. |
| **Stopping** | The worker is closing the session, returning the robot when appropriate, and releasing hardware. | The interface waits for the final outcome. |
| **Complete** | The requested episodes were saved and teardown succeeded. | The local dataset is handed back to the dataset interface. |
| **Completed with warning** | Recording work finished, but teardown or another non-recording operation failed. | Saved episodes are retained and the operator chooses how to leave. |
| **Failed** | Recording ended before normal completion. | The error, friendly hint, logs, and any recoverable saved result are shown. |

## Recording entry pathways

### 1. Record a new dataset

Entry point: **Add dataset -> Record a dataset**.

The user must first select a robot profile that is marked clean. The modal then collects:

- Dataset name.
- Task description.
- Number of episodes, from 1 through 100.
- Recording duration for each episode.
- Reset duration between saved episodes.
- Camera selection and camera roles.
- Streaming-encoding preference.

When the user is authenticated with Hugging Face, MakerMods Lab prefixes a bare dataset name with the user's namespace. When no Hub account is active, a bare local name is allowed so recording can work offline.

Before starting, the frontend releases its browser camera previews and waits briefly. The backend then creates a unique local repository ID by appending a timestamp such as `_20260714_153000`. This avoids overwriting another fresh recording with the same requested name.

The normal interface always records locally first. It sends `push_to_hub=false`; publishing is a later dataset-card action.

### 2. Record more episodes into an existing dataset

Entry point: select a local dataset and click **Record more episodes**.

MakerMods Lab resumes the existing repository ID without adding a timestamp. The modal fixes the destination ID, changes the episode field to **Episodes to add**, and seeds the task from the dataset's first known task.

The interface compares the selected setup with available dataset metadata and warns about likely differences in:

- Robot type.
- Configured camera names.
- Dataset FPS versus MakerMods Lab's 30 FPS recording request.

These frontend comparisons are advisory. LeRobot's dataset compatibility validation is the final gate. A Hub-only dataset must be downloaded before more episodes can be recorded because resume recording mutates a local dataset directory.

### 3. Record with one arm

For a standard SO-101 profile, MakerMods Lab connects one follower and one leader. It applies the selected calibration files, motor-power limit, and configured cameras to the follower-side recording configuration.

### 4. Record with a bimanual pair

For a bimanual profile, the request carries left and right leader/follower ports and calibration configurations. Both followers are controlled as one BiSO robot.

Cameras are attached to the left follower configuration. Their dataset feature names receive a `left_` prefix even though the camera-role names shown in the setup modal remain unprefixed. Resume checks account for that naming rule.

## Robot and session configuration

### Robot profile

The selected saved robot supplies the normal recording hardware context:

- Leader and follower serial ports.
- Calibration configuration names.
- Single-arm or bimanual mode.
- Saved camera configuration.
- Follower motor-power limit.

The modal refuses to start from a robot profile that is not marked clean. The recording backend also checks for active recording, teleoperation, or inference before claiming hardware.

### Task and timing

Each recording session carries one task string. All episodes added in that session use that task. The interface does not currently offer a per-episode task chooser or a multi-task recording queue.

The normal frontend request uses:

- 30 FPS.
- Video enabled.
- Local-only recording.
- Frontend audio cues instead of LeRobot sounds.
- No LeRobot display window.

### Encoding modes

The advanced **Streaming encoding** option determines when video is encoded:

- **On:** encode while recording, reducing the post-session encoding step but adding real-time load.
- **Off:** collect frames first and encode afterward, avoiding some real-time encoding pressure at the cost of temporary storage and later work.

### Camera configuration

Saved robot cameras prepopulate the recording modal. The operator can also enumerate available cameras, preview one before assigning a role, add or remove roles, and configure each camera.

Per-camera controls include:

- Width and height.
- FPS.
- FOURCC, including Auto, MJPG, YUYV, I420, NV12, H264, and MP4V.
- Capture backend, including the platform default and supported OpenCV overrides.

MakerMods Lab prevents obvious duplicate camera selections by camera index or browser device ID. If browser enumeration changes an OpenCV index, it attempts to reconcile the new index using the stable browser device ID.

The default backend follows the platform: AVFoundation on macOS, V4L2 on Linux, and DirectShow on Windows. The recording path defaults an unspecified FOURCC to MJPG. Backend overrides can change enumeration order, particularly on macOS, so camera roles should be visually verified after recabling or changing backends.

Immediately before recording, browser previews and enumeration are stopped so OpenCV can own the devices exclusively. The live recording page therefore does not show the modal's camera previews.

## Start and hardware-preflight pathway

Starting a session follows this sequence:

1. Validate the request and destination repository ID.
2. Finish any pending hardware release from a prior recording or teleoperation session.
3. Refuse the start if recording, teleoperation, or inference already owns the relevant runtime.
4. Release browser camera streams and allow device ownership to settle.
5. Create the LeRobot recording configuration and local dataset path.
6. Connect the follower arm and cameras without invoking interactive calibration.
7. Verify arm identity before applying calibration or sending motion commands.
8. Apply the saved follower calibration.
9. Apply the requested motor-power limit to the follower or followers.
10. Clear stale follower goal-velocity limits from previous operations.
11. Capture the follower's starting pose, excluding the gripper, for the normal return-home sequence.
12. Connect the leader and apply its saved calibration.
13. Begin the first recording phase.

Camera connection includes bounded retries for transient cold-open and resource-conflict failures. Failed attempts are cleaned up before retrying.

The arm identity check is a hard safety boundary. A confirmed mismatch disconnects hardware and stops the session before calibration is written or actions are sent. Findings that are uncertain but not hard mismatches are surfaced as a one-time warning in the recording interface.

The backend request contains a `skip_identity_check` escape hatch, but the normal MakerMods Lab interface does not expose it.

## Live operator controls

The recording page shows the current episode, total requested episodes, total session time, phase time and limit, phase progress, status color, identity warnings, and a collapsible log.

Audio cues announce recording, reset, and the last three seconds of a timed phase. A persistent mute toggle controls these cues.

| Current state | Main action | Shortcut | Effect |
| --- | --- | --- | --- |
| Recording | **End Episode** | Space or Right Arrow | Saves the current episode and either enters reset or finishes the session. |
| Resetting | **Start Next Episode** | Space or Right Arrow | Ends reset early and starts the next recording phase. |
| Recording | **Re-record** | Backspace | Discards the current in-memory take, enters reset, and retries the same episode number. |
| Recording or resetting | **Done** | Escape, followed by confirmation | Stops the session while retaining episodes already committed. |
| Recording or resetting | **Quit** | Explicit button and confirmation | Requests discard behavior; exact effects differ for fresh and resumed datasets. |

The frontend makes an optimistic phase change after **End Episode** or **Start Next Episode**, then reconciles with the next backend status response.

There is intentionally no Left Arrow shortcut for re-recording, reducing accidental destructive input.

## Episode lifecycle and save semantics

### Ending an episode early

Pressing **End Episode** during recording commits the current episode. If more episodes remain, MakerMods Lab enters the reset phase. After the final saved episode, it skips reset and begins teardown.

### Re-recording an episode

Pressing **Re-record** clears the current episode buffer, leaves the saved-episode count unchanged, and enters reset. The next recording phase retries the same episode number.

### Resetting the scene

Reset is deliberately outside dataset collection. The operator can wait for its timer or press **Start Next Episode** to end it early.

### Reaching the recording timer

In the current implementation, reaching the recording-duration limit does not commit the take. It marks the episode for re-recording, clears the episode buffer, and enters reset so the same episode is attempted again.

This differs from the likely expectation that a timed episode is automatically saved. Until the behavior or copy changes, operators should use **End Episode** to commit each wanted take.

### Stopping in the middle of an episode

An incomplete in-memory episode is dropped. Episodes that were already finalized remain available unless a fresh-session discard deletes the whole session directory.

## Session completion and exit pathways

The meaning of an exit depends on whether the session created a new timestamped dataset or resumed an existing one.

| Exit pathway | Fresh recording | Resume recording |
| --- | --- | --- |
| Requested episode count completed | All committed episodes are retained. | New committed episodes are appended to the existing dataset. |
| **Done** | Retains committed episodes; drops the in-progress take. | Retains the original dataset and newly committed episodes; drops the in-progress take. |
| **Quit** | Deletes the entire new timestamped session dataset, including episodes already saved in that session. | Preserves the pre-existing dataset and any appended episodes already committed; only the in-progress take is inherently discardable. |
| Browser back, tab close, or route abandonment | Best-effort discard behavior deletes the fresh session dataset. | Best-effort discard cannot roll back episodes already appended to the existing dataset. |
| Zero committed episodes | Removes the empty fresh dataset automatically. | Leaves the pre-existing dataset in place. |

Browser navigation is guarded in several ways: an in-app confirmation for back navigation, the browser's native unload confirmation, a keepalive stop beacon, and a best-effort discard request during component cleanup. These mechanisms reduce accidental orphan sessions but cannot make page-unload networking perfectly reliable.

The current **Quit** wording can overpromise rollback during resume. Once an appended episode has been committed to an existing dataset, quitting does not restore the dataset to its pre-session state.

## Teardown and hardware safety

After a normal completion or operator stop, MakerMods Lab attempts to return each follower to the pose captured at session start. The gripper joint is excluded so returning home does not automatically release a held object.

For a bimanual robot, the two follower returns run concurrently. MakerMods Lab then disables torque and disconnects the leader, follower, and cameras. A second layer of torque-disable cleanup is used as a safety backstop.

On a recording failure, the backend favors immediate release instead of graceful return motion. This avoids sending additional motion after an uncertain failure.

The backend supports a second stop request during graceful return to cut the return short and release immediately. The current recording page does not clearly expose that second-stop pathway once the session has entered its ended state.

## Outcome and error-recovery pathways

MakerMods Lab distinguishes three final outcome classes:

- **Success:** recording and teardown completed normally.
- **Completed with warning:** recording work and saved episodes succeeded, but release or another teardown step reported a problem.
- **Failure:** recording was cut short by a setup, hardware, data, or runtime error.

For a warning or failure, the page freezes the final status, displays a friendly error hint when available, and keeps the bounded recording log visible. If episodes were saved, the user may continue to the saved-dataset handoff rather than treating a teardown error as data loss.

The recording log is an in-memory bounded tail for the current or most recent in-process session. It is not a durable log file and should not be described as surviving a MakerMods Lab restart.

### Potentially destructive failure action

The ended-with-issue interface offers **Discard & exit**. Its frontend path directly calls the dataset-deletion endpoint. In a resumed session, that appears capable of deleting the entire existing local dataset rather than merely undoing episodes from the failed attempt.

This should be treated as a high-priority UX and safety gap: resume-session failure recovery must distinguish **drop the in-progress take**, **keep appended episodes**, and **delete the entire existing dataset** with separate, accurately labeled confirmations.

## Dataset handoff after recording

After a successful fresh session, MakerMods Lab routes through the page currently named `/upload`. Despite that route name, the page does not upload automatically. It confirms that the dataset was saved locally and returns the user home with the new dataset preselected.

From the dataset information card, the user can then:

- Inspect dataset metadata.
- Record more episodes.
- Train locally.
- Upload to the Hugging Face Hub with visibility and tag choices.
- Start Hugging Face Jobs training, uploading first when necessary.
- Merge, rename, or delete the local dataset when no active operation protects it.

The post-record route and component name are therefore stale and potentially misleading. A name such as `recording-result` would reflect the actual local-save handoff.

## Expected operating environment

The current interface is designed around these working assumptions:

- One supervised operator at a robot bench.
- One active browser tab controlling one MakerMods Lab process.
- One SO-101 single-arm or bimanual rig.
- Roughly 30 FPS episodic data collection.
- Saved robot configurations as the source of truth for ports, calibration, and cameras.
- Local-first recording that can operate without Hugging Face authentication.
- Visual confirmation of camera roles after hardware changes.

The interface is not a multi-user session manager. It has feature-local state and partial runtime exclusion rather than a single global ownership model.

## Current missing or incomplete pathways

### Episode review before acceptance

There is no first-class review step during or immediately after recording. The operator cannot:

- Replay synchronized camera streams, state, and actions before accepting a take.
- Mark an episode good or bad.
- Add notes, quality labels, or failure reasons.
- Compare the current take with previous takes.
- Delete, trim, reorder, or relabel individual episodes.

The current decision is therefore binary and immediate: commit with **End Episode** or discard with **Re-record**.

### Live recording observability

Once browser previews are released, the recording page does not show live camera frames. It also lacks first-class indicators for:

- Actual camera FPS and resolution.
- Dropped or delayed frames.
- Encoder backlog.
- Disk throughput and remaining capacity.
- Robot power or voltage telemetry.
- Per-device connection health.

These are especially important because recording can expose camera instability that a lightweight preview does not.

### Preflight and compatibility

Resume warnings are advisory and the definitive compatibility check occurs as part of the recording start. A better preflight would validate dataset metadata, features, FPS, cameras, robot type, writable storage, and expected encoding space before opening or energizing hardware.

The recording start gate checks recording, teleoperation, and inference ownership, but it does not appear to include calibration in the same exclusion check. Hardware features should share one explicit ownership mechanism.

There is also no final camera-role snapshot or checklist immediately before device ownership transfers from browser preview to OpenCV.

### Session control

The interface has no true pause-and-resume control. It supports phase advance, re-record, Done, and Quit, but not pausing an active episode while keeping its buffer and timers intact.

There are no saved recording presets for common tasks, timing, cameras, and episode counts. There is also no batch task list for collecting multiple task labels in one guided run.

The interface permits zero configured cameras but does not ask the operator to explicitly confirm an intentional proprioception-only dataset.

### Timeout clarity

The timer currently causes re-recording rather than automatic commit. The UI does not explain this clearly enough. The product should either save a complete timed episode automatically or label the timer as a take deadline that discards and retries.

### Resume and rollback

Resume modifies the existing dataset incrementally. There is no transaction, temporary branch, or session snapshot that can atomically roll back all newly appended episodes.

A safer pathway would record additions into a temporary dataset, validate them, and merge them into the destination only after explicit acceptance. At minimum, confirmation copy must distinguish fresh-session deletion from resume-session behavior.

### Recovery after page or process loss

The recording page depends on navigation state containing the request. Refreshing or reopening the route does not provide a deliberate reattach-to-active-session workflow. The unload beacon is best effort, so the interface should be able to discover an active backend recording and offer to reattach, stop safely, or inspect its status.

Logs are not persisted across MakerMods Lab restarts, and there is no **Export diagnostic log** action. There is also no guided recovery for a partial or corrupt dataset after a process or power failure.

### Post-session summary

The final handoff does not provide a detailed session report with:

- Saved and discarded take counts.
- Per-episode duration.
- Camera and robot configuration actually used.
- Warnings and retry counts.
- Disk usage and encoding result.
- Dataset format and revision compatibility.

The stale `/upload` route also adds conceptual friction because upload is now a separate later action.

## Recommended normal workflow

For the current implementation, the clearest operator pathway is:

1. Select and verify a clean saved robot profile.
2. Open **Record a dataset** or **Record more episodes**.
3. Preview every camera and confirm its semantic role.
4. Configure timing, task, episode count, and encoding mode.
5. Start and wait for the follower, cameras, and leader to connect.
6. Perform the task, then press **End Episode** to commit the take before its timer expires.
7. During reset, restore the scene and press **Start Next Episode** when ready.
8. Use **Re-record** immediately when a take is known to be bad.
9. Use **Done** to retain committed work when ending early.
10. Return home, inspect the local dataset summary, and upload or train only after a separate quality check.

For resumed datasets, operators should assume that every completed episode is appended permanently as the session proceeds. **Quit** is not a full rollback mechanism.

## Product model

The intended recording lifecycle can be summarized as:

> select hardware -> preflight -> preview -> record -> judge -> commit or retry -> reset -> finish -> verify -> publish or train

MakerMods Lab already covers hardware selection, camera setup, episodic capture, rapid re-recording, local save, and dataset handoff. The largest remaining gaps are explicit take review, live health visibility, accurate resume/discard semantics, atomic recovery, and a durable post-session audit trail.
