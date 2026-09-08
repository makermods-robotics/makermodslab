# Configuration Setup Interface Pathways

This document inventories how an operator launches MakerMods Lab, creates and selects a robot profile, assigns hardware, manages calibration, configures cameras and motor power, and hands the saved profile to the rest of the interface. It describes the pathways implemented in the current application without evaluating them as defects or proposing replacements.

This inventory was audited on July 14, 2026, against MakerMods Lab commit `518ca56`.

## Core configuration model

MakerMods Lab's reusable hardware configuration is a named **robot record**. The record is separate from the physical arms and from the calibration JSON files stored in the calibration library.

A record contains:

- A name and an immutable mode: `single` or `bimanual`.
- One leader port and one follower port for a single-arm profile.
- The same primary fields as the left pair, plus `right_leader_port` and `right_follower_port`, for a bimanual profile.
- One assigned calibration name for each arm slot.
- A list of saved cameras.
- A follower motor-power percentage.
- A computed `is_clean` readiness value returned by the API.

The principal state layers are:

| Layer | Contents | Lifetime |
| --- | --- | --- |
| Robot record | Mode, arm ports, calibration assignments, cameras, and motor power. | Persisted on the MakerMods Lab host. |
| Calibration library | Raw LeRobot calibration JSON, separated into leader and follower libraries. | Persisted on the MakerMods Lab host independently of robot records. |
| Configuration-page draft | Unsaved port, camera, and motor-power edits. | Browser memory for the current page visit. |
| Selected robot | The name of the profile used by landing-page and downstream flows. | Browser `localStorage`, shared by mounted frontend consumers. |
| Active operation | Manual calibration or auto-calibration process state and logs. | Backend process memory; status endpoints let the page resume its view. |

The full setup spine is:

1. Start MakerMods Lab in the desired local, development, LAN, or offline posture.
2. Create or select a robot profile on the landing page.
3. Open **Configure** for that profile.
4. Assign each arm's serial port.
5. Calibrate each arm or assign an existing calibration from the relevant library.
6. Optionally configure cameras and adjust follower motor power.
7. Save the drafted ports, cameras, and power settings.
8. Return to the landing page when the profile is ready.
9. Use the selected profile for teleoperation, recording, or inference.

## Launch and access pathways

### Standard local application

Running `makermodslab` starts the FastAPI backend on port 8000, serves the committed production frontend from the same process, binds to localhost, and opens the local browser when the server is ready.

This is the normal workstation pathway. The browser and hardware backend run on the same machine, so browser camera discovery and backend OpenCV enumeration describe the same host.

### Development application

Running `makermodslab --dev` starts the Vite development server on port 8080 and an auto-reloading backend on port 8000. The browser opens the Vite URL. This pathway serves frontend source directly and is intended for interface development.

### LAN station

Running `makermodslab --lan` binds the production server to `0.0.0.0` and does not open a local browser. Other machines on the LAN can open the station's port 8000.

The installed `makermodslab-station` entry point applies `--lan --offline`. It therefore exposes the interface to the LAN while disabling Hub-dependent behavior.

In a LAN session, robot records, calibration files, serial ports, and backend camera indices belong to the station host. Browser media devices and previews belong to the client device that opened the page. This distinction is part of the configuration context whenever the browser is not running on the station itself.

### Offline posture

Running with `--offline` sets the Hugging Face offline posture before the backend imports Hub libraries. Local profile creation, port assignment, calibration, camera configuration, and local hardware operation remain host-local pathways. Hub authentication, discovery, upload, download, and cloud-job functionality are outside that posture.

## Robot profile lifecycle

### Create a profile

Entry point: the robot card on the landing page.

1. The user chooses the single-arm or bimanual layout.
2. The user supplies a filesystem-safe name.
3. The frontend sends `POST /robots/{name}?create=true` with the selected mode.
4. The backend creates an otherwise empty record with a 38% motor-power default.
5. The new profile becomes the selected robot and the layout filter follows its mode.

Names may not be empty, contain `/` or `\`, or contain `..`. A duplicate name is rejected rather than overwriting the existing profile.

The selected mode defines the shape of the record for its lifetime. The landing-page single/bimanual control filters and changes selection; it does not convert an existing record between modes.

### Select a profile

Robot selection is stored under the browser key `makermodslab.selectedRobot`. All mounted `useRobots` consumers subscribe to one module-level frontend store, so the landing card, recording controls, training-area consumers, and inference entry points see the same selected record.

The frontend refreshes records from `GET /robots` on navigation. If a selected record has been removed elsewhere, the selection is cleared.

Switching the layout filter selects the first alphabetically sorted profile in that mode. If the selected mode has no profiles, the selection is cleared.

### Configure a profile

The landing-page gear/configure action navigates to `/calibration` with the robot name in route state. The page fetches the current record and treats it as the baseline for drafts and calibration assignments.

For bimanual profiles, the primary leader/follower fields represent the left pair and the `right_*` fields represent the right pair. The page lets the user switch among the applicable arm slots while keeping one robot-level camera and motor-power configuration.

### Rename a profile

Renaming changes the robot-record filename, its stored `name`, and the browser selection when the renamed profile was selected. It does not rename calibration files because those are keyed independently by calibration name.

### Delete a profile

Deleting removes only the robot record. The frontend clears the selection if it referred to that profile. Calibration files remain in their leader or follower libraries and can be assigned to another profile later.

## Readiness and the clean-profile gate

The backend computes `is_clean` whenever it returns a robot record.

A single-arm profile is clean when:

- Leader and follower ports are non-empty.
- Leader and follower calibration names are non-empty.
- Both referenced calibration JSON files exist in the correct libraries.

A bimanual profile must satisfy the same conditions for all four arm slots.

Cameras are optional and do not participate in the clean calculation. Motor power is normalized separately and also does not determine cleanliness.

The landing card and recording flow use this readiness result before starting an operation. A profile can therefore be saved incrementally while incomplete, then become ready as its missing ports and calibrations are filled.

## Serial-port configuration pathways

### Enumerate ports

The configuration page calls `GET /available-ports`. MakerMods Lab uses `pyserial` on Windows and enumerates USB serial naming patterns on macOS and Linux:

- macOS: `/dev/tty.usbmodem*` and `/dev/tty.usbserial*`.
- Linux and Jetson: `/dev/ttyUSB*` and `/dev/ttyACM*`.
- Windows: available COM ports reported by `pyserial`.

The user can refresh the list after plugging in, unplugging, or reconnecting an arm.

### Assign a port manually

Each arm slot has a port selector. Choosing a free port stages that value for the current slot. Clearing the selector stages an empty assignment.

Port changes remain in the configuration-page draft until **Save**. Save submits every dirty port field together, allowing a coordinated swap or move to be validated against the final prospective record.

Every non-empty port in a profile must be unique. In a bimanual profile this uniqueness applies across all four leader and follower slots.

### Detect an arm by moving it

The **Detect** pathway calls `POST /identify-arm`. The backend watches candidate ports while the user moves the shoulder-pan joint by hand, then reports the port on which motion was observed.

This pathway is read-only with respect to the motors. The detected port is staged for the current arm slot. If another slot already holds that port, the interface asks the user to confirm either:

- A swap, when the current slot already had a different port.
- A move, when the current slot was empty and the other slot will be cleared.

### Identify an assigned port by wiggling the gripper

The **Wiggle** pathway calls `POST /wiggle` for the currently selected port. It actively moves the gripper so the user can visually identify the physical arm connected to that port.

Detect and Wiggle are complementary: Detect maps a hand-moved arm to a port, while Wiggle maps a known port to a visible arm movement.

### Legacy port defaults

The `GET /robot-port/{robot_type}` route still reads the legacy `leader_port.txt` and `follower_port.txt` files. The calibration page consults this fallback only when it was not opened for a named robot profile. Named profiles own the normal persistent port assignments; the legacy files are not the write path for current profile edits.

## Calibration-library pathways

MakerMods Lab presents separate libraries for leader (`device_type="teleop"`) and follower (`device_type="robot"`) calibrations. The API vocabulary is intentionally different from the port endpoint vocabulary, which uses `leader` and `follower`.

### Browse and assign a saved calibration

The page fetches the relevant side from `GET /calibration-configs/{device_type}`. Assigning a library entry writes its stem into the selected robot slot. The robot record refers to the calibration by name; the calibration remains an independent raw JSON file.

For a bimanual profile, the left and right leaders cannot share one leader calibration, and the left and right followers cannot share one follower calibration. A leader and follower may have the same stem because their files live in different directories.

### Import a calibration

The import control reads a raw LeRobot calibration JSON in the browser and posts a requested name plus the parsed data to `POST /calibration-configs/{device_type}/upload`.

The backend validates the expected motor-calibration shape, saves into the selected side's library, and never overwrites an existing name. The imported calibration can then be assigned to a robot slot.

### Download a calibration

`GET /calibration-configs/{device_type}/{config_name}/download` returns the original raw LeRobot JSON with no MakerMods Lab wrapper. The file can be shared, copied, or re-imported.

### Rename a calibration

The rename endpoint moves the library file without overwriting another name. Robot records that referenced the old name are updated to reference the new stem, preserving their assignments.

### Delete a calibration

Deleting removes the library file. Every robot slot that referenced it is cleared, so those profiles return to an incomplete calibration state. Deleting a robot profile and deleting a calibration are therefore separate operations with different scope.

### Open the calibration folder

The page can ask the backend to open the selected leader or follower calibration directory in Finder, Explorer, or the Linux file browser. This action opens a GUI on the MakerMods Lab server host and creates the directory first when necessary.

## Manual calibration pathway

Manual calibration is a guided one-arm session tied to the currently selected robot slot.

1. The user selects a leader or follower slot with an assigned port.
2. The page derives the target calibration name from the slot's current assignment or the robot-and-arm default name.
3. `POST /start-calibration` starts the backend calibration manager for that port, side, robot name, and arm slot.
4. The backend connects the SO-101 device and guides the operator to the middle/start position.
5. The operator confirms the guided step from the page.
6. The backend records the movement range while the operator moves every joint through its intended travel.
7. The backend validates the captured range, constructs the motor calibration, writes it to the servo bus, and saves the raw JSON file.
8. The completed calibration name is written back to the originating robot slot.
9. The page refreshes the profile and calibration library, then advances its focus to another incomplete side when applicable.
10. The device is disconnected and the serial port is released.

The page polls `/calibration-status` during the session and sends `/complete-calibration-step` for the guided confirmation. The default slot name is overwritten intentionally for a repeat calibration; a user who wants to preserve a prior file can rename it through the library.

### Cancel and leave handling

The explicit Cancel action calls `/stop-calibration`. Navigation away from an active manual session is guarded with a confirmation and a stop request so the calibration is aborted, no result is saved, and the arm is released.

The leave guard applies to the manual calibration flow. Auto-calibration has its own persistent batch status model.

## Auto-calibration pathway

Auto-calibration is the powered, scripted calibration path. The interface marks that the selected arms move on their own and requires the operator to keep the workspace clear.

### Select arms

The page builds the available arm slots from the current profile:

- Two slots for a single-arm profile.
- Four slots for a bimanual profile.

Only slots with an assigned port that is currently detected can be selected. The user may select one arm or any subset up to all applicable arms. Duplicate ports are rejected before launch.

### Run a batch

`POST /start-auto-calibration-batch` sends one request per selected arm inside a batch description. Each arm receives its side, port, target config name, and left/right arm label.

The backend launches the selected arms concurrently as independent subprocess-backed sessions. Per-arm status and logs are combined under `/auto-calibration-batch-status`, including total, completed, and failed counts. One arm's outcome does not determine the state of the other arms.

The page can be revisited while a batch is still active; polling reconstructs the visible progress from backend state. **Stop all** calls `/stop-auto-calibration-batch`, terminates the running children, performs fallback torque release, and marks the affected sessions stopped.

### Successful completion

For each successful arm, MakerMods Lab places the resulting JSON in the correct leader or follower library and writes the calibration name back to that robot slot. Unsuccessful or interrupted arms do not receive the same success write-back.

The vendored script naturally writes under robot calibration paths. MakerMods Lab post-processes a leader result into the leader library so the library boundary remains consistent.

## Bimanual calibration staging

The user-facing calibration library allows arbitrary names, but LeRobot's BiSO configuration expects a single base ID and files named `<base>_left.json` and `<base>_right.json` in one calibration directory.

At the start of a bimanual hardware session, MakerMods Lab resolves this difference by copying the four selected library files into:

```text
~/.cache/huggingface/lerobot/makermodslab_biso/<robot>/leader/
~/.cache/huggingface/lerobot/makermodslab_biso/<robot>/follower/
```

The staged aliases use the expected left/right names and are refreshed on every session. Teleoperation and recording stage both leader and follower sides. Follower-only inference stages only the two follower calibrations.

The staging directory is an execution adapter, not the user-facing calibration library or the source of robot assignments.

## Camera-configuration pathways

### Enable camera setup

Camera scanning is off when the configuration page opens. The user turns it on explicitly, which begins browser and backend enumeration. Turning it off releases preview streams.

This opt-in boundary keeps opening the robot configuration page from immediately claiming camera resources.

### Enumerate and match cameras

The backend `GET /available-cameras` route enumerates indices in the order used by OpenCV for recording:

- AVFoundation device names on macOS.
- DirectShow friendly names on Windows.
- V4L2 card names on Linux and Jetson.

The browser separately enumerates `videoinput` devices. After camera permission makes labels available, the frontend matches backend names to browser labels and stores the browser `deviceId` for preview. USB hotplug events trigger a debounced refresh.

The backend index is the hardware value used by recording and inference. The browser `deviceId` is the preview binding. A camera can remain selectable for backend use without a browser-label match, although the browser preview then has no stream binding.

### Add and edit cameras

Each saved camera has a semantic name and a backend camera index. A newly added camera defaults to 640 by 480 at 30 FPS.

The advanced controls can set:

- Width and height.
- FPS.
- FOURCC pixel format, or automatic selection.
- OpenCV backend, or the platform default.

Camera names become the semantic keys used for dataset image features and model-camera matching. Camera additions, removals, automatic binding updates, and advanced edits remain drafts until the page's shared **Save** action.

## Motor-power configuration pathway

Motor power is stored per robot as an integer percentage from 10 through 100. A new or normalized profile defaults to 38%, which corresponds to a servo `Torque_Limit` value of 380 on the 0–1000 register scale.

The configuration UI presents the raw `Torque_Limit` scale while converting to and from the persisted percentage at the boundary:

- 10% becomes `Torque_Limit` 100.
- 38% becomes `Torque_Limit` 380.
- 100% becomes `Torque_Limit` 1000.

The slider is a draft. It is persisted with cameras and dirty ports through the same Save request. The saved limit is applied to follower motors at the start of teleoperation, recording, and inference rather than being treated as a calibration-file property.

## Save and persistence behavior

The configuration page has one shared Save transaction for three robot-record areas:

- Dirty port slots.
- The camera list.
- Motor power.

The page compares these drafts with the last fetched robot record and sends only changed fields to `POST /robots/{name}`. On success, the returned record becomes the new baseline and the drafts are cleared.

Calibration assignment and calibration completion are separate server writes because they act immediately on a named library entry or completed hardware session. They do not wait for the port/camera/power Save button.

The backend merges known typed fields into the record, atomically writes its JSON, normalizes config names to stems when reading, clamps motor power, rejects duplicate ports, and rejects same-side bimanual calibration reuse.

## Persistent storage map

All normal hardware configuration lives under the MakerMods Lab host's LeRobot cache:

| Data | Path | Notes |
| --- | --- | --- |
| Robot records | `~/.cache/huggingface/lerobot/robots/<name>.json` | One record per named profile. |
| Leader calibrations | `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/*.json` | API `device_type="teleop"`. |
| Follower calibrations | `~/.cache/huggingface/lerobot/calibration/robots/so_follower/*.json` | API `device_type="robot"`. |
| Bimanual staging aliases | `~/.cache/huggingface/lerobot/makermodslab_biso/<robot>/{leader,follower}/` | Refreshed execution copies for BiSO naming. |
| Legacy leader port | `~/.cache/huggingface/lerobot/ports/leader_port.txt` | Read-only fallback for the legacy port endpoint. |
| Legacy follower port | `~/.cache/huggingface/lerobot/ports/follower_port.txt` | Read-only fallback for the legacy port endpoint. |
| Hugging Face token | Hugging Face's standard local token store | Shared with the `hf` CLI and Hub libraries. |

Robot records and calibration files are deliberately independent. A record points to library names; it does not embed calibration content.

## Hugging Face authentication context

Authentication is app-wide context rather than part of an individual robot record.

The frontend fetches `/hf-auth-status` and exposes authenticated, unauthenticated, and loading states. The landing-page account chip can show the `hf auth login` command, and the token login endpoint validates and stores a supplied token through the Hugging Face library.

Authentication influences dataset, model, and cloud-job pathways. It does not change the shape or readiness of a local robot profile.

## Downstream profile consumers

The saved robot profile is reused differently by each hardware workflow.

| Workflow | Profile fields consumed | Camera behavior |
| --- | --- | --- |
| Teleoperation | Name, mode, leader/follower ports, all applicable calibration names, and motor power. | LeRobot opens no cameras; any visual panel is browser-side. |
| Recording | Name, mode, all applicable ports and calibrations, motor power, and a camera configuration seeded from the saved profile. | The recording modal receives a clone of saved cameras; session edits in that modal configure that recording session. |
| Inference | Name, mode, follower ports and calibrations, motor power, and cameras. | Saved cameras are matched to the checkpoint's expected semantic image features. |
| Training | Dataset, model, and trainer settings. | Robot hardware profiles are not part of model training configuration. |

Single-arm teleoperation and recording construct an SO-101 follower and leader from the selected library names. Bimanual flows stage the arbitrary library names into BiSO-compatible aliases first.

The profile is the reusable default, while operation-specific interfaces may derive a session request from it. Saving an operation's temporary request is not the same as editing the robot record unless that interface explicitly calls the robot-record endpoint.

## Recovery and continuity pathways

- Returning to the landing page refreshes robot records and recomputes readiness from current calibration files.
- Reopening Configure fetches the current robot record rather than reusing an old page draft.
- Manual calibration status is polled rapidly while active, and navigation asks to abort before leaving.
- Auto-calibration batch state and per-arm logs are read from backend status, allowing visible progress to be reconstructed after navigation.
- Port and camera lists can be refreshed after hardware changes; camera hotplug also triggers an automatic debounced refresh.
- Deleting or renaming a calibration updates robot references, so later readiness reflects the library operation.
- A missing or unplugged saved port remains part of the record but is distinguished from a currently detected port when selecting arms for auto-calibration.

## Platform and deployment context

The same profile model is used on macOS, Windows, Linux, and Jetson. Platform differences are concentrated at the hardware-discovery and launch boundaries:

- Serial enumeration follows each platform's device naming.
- Camera enumeration uses AVFoundation, DirectShow, or V4L2 ordering and names.
- A browser camera preview requires `navigator.mediaDevices`, which is available on localhost or a secure HTTPS context.
- Opening a calibration folder launches the file browser on the server host.
- A LAN browser controls serial devices and calibration files on the station, while browser media permissions and device IDs belong to the client browser.
- Jetson driver installation, CUDA compatibility, serial permissions, and service provisioning are machine preparation outside the robot-record editor; once available to the process, the devices enter the same port and camera pathways described above.

## Implementation evidence

The pathway map above is grounded in the following current implementation areas:

- Robot record type, shared selection store, CRUD calls, and persistence key: `frontend/src/hooks/useRobots.ts:7-29`, `frontend/src/hooks/useRobots.ts:31-84`, `frontend/src/hooks/useRobots.ts:106-145`, `frontend/src/hooks/useRobots.ts:147-294`.
- Landing-page filtering, create/select behavior, Configure navigation, and teleoperation request derivation: `frontend/src/components/landing/RobotConfigManager.tsx:37-102`, `frontend/src/components/landing/RobotConfigManager.tsx:104-178`.
- Configuration-page record fetch, drafts, port identification, batch auto-calibration, manual calibration, and Save: `frontend/src/pages/Calibration.tsx:120-291`, `frontend/src/pages/Calibration.tsx:371-466`, `frontend/src/pages/Calibration.tsx:547-749`, `frontend/src/pages/Calibration.tsx:759-897`, `frontend/src/pages/Calibration.tsx:899-1121`, `frontend/src/pages/Calibration.tsx:1123-1258`.
- Configuration-page camera and motor-power controls: `frontend/src/pages/Calibration.tsx:1150-1185`, `frontend/src/pages/Calibration.tsx:1788-1860`, `frontend/src/pages/Calibration.tsx:2290-2335`.
- Camera defaults and advanced controls: `frontend/src/components/recording/CameraConfiguration.tsx:40-53`, `frontend/src/components/recording/CameraConfiguration.tsx:145-166`, `frontend/src/components/recording/CameraConfiguration.tsx:425-532`.
- Browser/backend camera matching and hotplug refresh: `frontend/src/hooks/useAvailableCameras.ts:18-41`, `frontend/src/hooks/useAvailableCameras.ts:43-121`, `frontend/src/hooks/useAvailableCameras.ts:123-143`.
- Persistent paths, record normalization, CRUD, readiness, uniqueness, and BiSO staging: `makermodslab/utils/config.py:28-56`, `makermodslab/utils/config.py:270-336`, `makermodslab/utils/config.py:339-506`, `makermodslab/utils/config.py:509-654`, `makermodslab/utils/config.py:939-945`.
- Robot and configuration API routes: `makermodslab/server.py:1804-1871`, `makermodslab/server.py:1874-2101`, `makermodslab/server.py:2109-2139`, `makermodslab/server.py:2313-2350`, `makermodslab/server.py:2367-2531`.
- Manual calibration session, validation, save, robot write-back, and cleanup: `makermodslab/calibrate.py:232-339`, `makermodslab/calibrate.py:339-421`, `makermodslab/calibrate.py:484-656`.
- Concurrent auto-calibration lifecycle and post-processing: `makermodslab/auto_calibrate.py:180-438`, `makermodslab/auto_calibrate.py:469-626`.
- Shared SO-101 and BiSO runtime configuration assembly: `makermodslab/utils/robot_factory.py:14-32`, `makermodslab/utils/robot_factory.py:49-119`.
- Authentication context and host token storage: `frontend/src/contexts/HfAuthContext.tsx:12-82`, `makermodslab/utils/hf_auth.py:109-151`.
- Local, development, LAN, offline, and station launcher behavior: `makermodslab/scripts/makermodslab.py:15-23`, `makermodslab/scripts/makermodslab.py:300-330`, `makermodslab/scripts/makermodslab.py:362-505`.
