# Inference Interface Pathways

**Audit date:** July 14, 2026  
**Audited commit:** `518ca56`

This document maps the inference interface as implemented at the audited commit. It follows user-visible pathways through the React frontend, FastAPI endpoints, job/model registries, robot configuration, rollout subprocess, status handling, and exit behavior. It describes current behavior neutrally; it is not a bug list.

## 1. State model

Inference is a process-local singleton owned by `makermodslab/rollout.py`:

- `inference_active` claims the feature.
- `_inference_proc` owns the rollout subprocess once it has spawned.
- `_inference_started_at` times setup plus rollout execution.
- `_inference_rollout_started_at` starts only when LeRobot emits `Rollout setup complete`.
- `_inference_meta` carries phase, model ref/path, duration, log path, warnings, and download progress.
- `_inference_cancel` cancels startup before a subprocess has been committed.
- `_state_lock` protects short mutations of those values.

Source: `makermodslab/rollout.py:95-116`, `makermodslab/rollout.py:153-195`.

The state sequence is:

1. **Idle:** `inference_active=False`, no process, no active metadata.
2. **Claimed/startup:** `inference_active=True`, total-start timestamp set, per-session cancel event created, phase seeded as `starting`.
3. **Optional model download:** phase becomes `downloading_model` for Hub refs.
4. **Robot preparation:** calibrations are staged and follower preflights run.
5. **Subprocess committed:** phase remains at least `starting`, then stdout markers advance it through `loading_policy`, `connecting`, and `running`.
6. **Termination:** natural exit, explicit `stopping`, or startup/runtime `error`.
7. **Finalization:** the terminal status payload is exposed and the singleton returns to idle.

Source: `makermodslab/rollout.py:118-150`, `makermodslab/rollout.py:887-966`, `makermodslab/rollout.py:1081-1176`.

The inference API surface is:

- `POST /start-inference`
- `POST /stop-inference`
- `GET /inference-status`
- `GET /inference-log`

Source: `makermodslab/server.py:521-554`.

## 2. Entry and selection pathways

### 2.1 Landing Models panel

The landing Models panel obtains the merged model listing from `GET /models`, owns the home-page model selection, and presents the shared inference launch footer. The chosen model id is persisted in `localStorage` under `makermodslab.selectedModel`, so it survives navigation/reload and synchronizes across tabs.

Source: `frontend/src/components/landing/ModelsPanel.tsx:47-88`, `frontend/src/components/landing/ModelsPanel.tsx:281-303`, `frontend/src/hooks/useSelectedModel.ts:3-35`.

The selected `ModelItem` is mapped to a launchable job through these branches:

1. A local training model or local/Hub-collapsed model matches an existing job by `ModelItem.id`.
2. An already imported or cloud-tracked Hub model matches an existing job by `hf_repo_id`, case-insensitively.
3. A model with no matching job is lazily registered as an imported pseudo-job only when Run is clicked.
4. The source used for that lazy import is selected in this order: Hub repo id, local checkpoint path, then model id.

Source: `frontend/src/lib/inferenceLaunch.ts:5-48`, `frontend/src/components/landing/ModelLaunchFooter.tsx:62-121`.

Browsing or selecting a model does not create a job. Once the model maps to a job, the footer loads `/jobs/{id}/checkpoints`, defaults to the last checkpoint returned, and opens `InferenceModal`. An untracked model leaves the dropdown at “Latest checkpoint”; clicking Run registers the source, then opens the modal with `initialStep=null` so the modal selects the latest available checkpoint.

Source: `frontend/src/components/landing/ModelLaunchFooter.tsx:26-45`, `frontend/src/components/landing/ModelLaunchFooter.tsx:68-121`, `frontend/src/hooks/useInferenceLaunch.tsx:29-76`.

### 2.2 Jobs dashboard

Tracked local, cloud, and imported jobs with runnable checkpoints display a checkpoint dropdown and green inference button. A resumed run can show checkpoints inherited from its lineage; when the user selects one, the launch is routed to the job that actually owns that checkpoint.

Source: `frontend/src/components/jobs/JobCard.tsx:287-299`, `frontend/src/components/jobs/JobCard.tsx:418-558`.

An untracked Hub model card first calls `POST /jobs/import`, refreshes the job listing, then opens the same inference modal using the imported record and its latest checkpoint. This makes the model follow the same modal, policy-config, robot, camera, and start pathways as an already tracked job.

Source: `frontend/src/components/jobs/HubModelCard.tsx:243-258`, `frontend/src/components/jobs/JobsSection.tsx:196-218`.

### 2.3 Training monitoring page

`/training/:jobId` exposes a “Run on robot” row once the monitored job has checkpoints. It passes the monitored job id, chosen step, and globally selected robot into the same `InferenceModal` used by Landing and Jobs.

Source: `frontend/src/pages/Training.tsx:933-965`.

### 2.4 Inference route

The live inference screen is routed at `/inference`. It is a monitor/stop destination, not a configuration entry point; configuration happens in `InferenceModal` before navigation.

Source: `frontend/src/App.tsx:40-49`, `frontend/src/components/landing/InferenceModal.tsx:391-420`.

## 3. Model discovery, import, and acquisition

### 3.1 Merged model listing

`GET /models` merges:

- completed local training runs that have a valid checkpoint;
- model checkpoints copied or downloaded into the local model cache;
- relevant Hub repos belonging to the authenticated user and organizations;
- persistently pinned custom Hub ids;
- equivalent local and Hub entries into `source="both"` rows;
- the persistent hidden-model filter.

Source: `makermodslab/models.py:240-261`, `makermodslab/models.py:413-458`, `makermodslab/models.py:494-518`, `makermodslab/models.py:526-667`, `makermodslab/server.py:915-932`.

The associated persistent locations are:

- training jobs and checkpoints: `~/.cache/huggingface/lerobot/outputs/train/<job>/`;
- copied/downloaded models: `<lerobot_home>/makermodslab_models/<repo_id>/`;
- pinned Hub ids: `~/.cache/huggingface/lerobot/saved_custom_models.json`;
- hidden Hub ids: `~/.cache/huggingface/lerobot/hidden_models.json`.

Source: `makermodslab/jobs.py:1792-1799`, `makermodslab/models.py:283-315`, `makermodslab/utils/config.py:69-80`.

### 3.2 Acquisition branches

#### Existing training job

A local or cloud training job is already represented in the job registry. Inference reads the job’s checkpoint listing directly; it does not create another job record.

Source: `makermodslab/jobs.py:1465-1500`.

#### Add Hub model without Download

The Landing “Add from Hugging Face” flow pins the typed repo id through `POST /models/custom` and selects it. Without Download, no weights are copied at this stage. Running it later lazily registers the Hub source as an imported pseudo-job.

Source: `frontend/src/components/landing/ModelsPanel.tsx:147-172`, `makermodslab/server.py:967-986`, `frontend/src/hooks/useInferenceLaunch.tsx:44-64`.

#### Add Hub model with Download

The same flow can also call `POST /models/download`. A background download snapshots the repo into the local models directory. The model info card can reattach after navigation by polling `/models/download-status`, and the merged listing changes from Hub-only to `both` when the local checkpoint becomes usable.

Source: `frontend/src/components/landing/ModelsPanel.tsx:147-172`, `makermodslab/server.py:1023-1049`, `makermodslab/models.py:1090-1123`.

The launch mapper still uses `hf_repo_id` before `path` when both are present. A downloaded model whose merged row retains Hub identity therefore follows the Hub pseudo-job branch; a disk/local-only row with no Hub identity follows its local path branch.

Source: `frontend/src/lib/inferenceLaunch.ts:40-48`.

#### Import from disk through Models

`POST /models/import` synchronously copies a root checkpoint or checkpoints tree into the local model directory. The original folder remains untouched. The returned local id is selected and the merged model list refreshes.

Source: `frontend/src/components/landing/ModelsPanel.tsx:174-180`, `makermodslab/server.py:1052-1069`, `makermodslab/models.py:1131-1193`.

#### Import through Jobs

`POST /jobs/import` creates a pointer-only imported pseudo-job. An existing local directory is stored in `output_dir`; otherwise the normalized source is stored as `hf_repo_id`. No model files are copied. Registration is idempotent by filesystem identity for local paths and case-insensitive repo id for Hub sources.

Source: `makermodslab/server.py:1161-1184`, `makermodslab/jobs.py:1258-1351`.

### 3.3 Checkpoint reference shapes

The checkpoint API returns opaque refs that the frontend passes back unchanged:

- local training tree: absolute `.../checkpoints/<step>/pretrained_model`;
- flat imported local model: absolute directory at sentinel step `0`;
- Hub training tree: `repo@checkpoints/<zero-padded-step>`;
- flat Hub model: `repo@root` at sentinel step `0`.

Source: `makermodslab/jobs.py:546-574`, `makermodslab/jobs.py:739-800`, `frontend/src/components/jobs/CheckpointDropdown.tsx:26-51`.

### 3.4 Policy metadata versus policy weights

Selecting a Hub checkpoint may download only its `config.json` so the modal can inspect policy type, feature names, dimensions, and language conditioning. Starting inference later resolves the actual weights:

- a local ref is returned unchanged;
- a Hub checkpoint ref downloads only `checkpoints/<step>/pretrained_model/*`;
- a Hub root ref downloads the root model while excluding `checkpoints/**` and `training_state/**`.

Source: `makermodslab/jobs.py:810-836`, `makermodslab/rollout.py:288-356`.

## 4. Robot and calibration selection

The modal receives the globally selected robot record. Robot selection is shared across all `useRobots()` consumers and persisted as `localStorage["makermodslab.selectedRobot"]`.

Source: `frontend/src/hooks/useRobots.ts:29-84`, `frontend/src/hooks/useRobots.ts:96-137`.

The inference modal does not edit or switch robots. It follows one of three branches:

- no robot: direct the user to select and configure one on Landing;
- robot not clean: explain that calibration/configuration is incomplete and disable Start;
- clean robot: show its name and whether both bimanual followers will be driven.

Source: `frontend/src/components/landing/InferenceModal.tsx:456-483`.

A robot record is clean when every operational leader/follower port and calibration field required by its mode is populated and every referenced calibration file exists. Cameras are optional at this level.

Source: `makermodslab/utils/config.py:473-506`.

Robot and calibration persistence is:

- robot record: `~/.cache/huggingface/lerobot/robots/<name>.json`;
- follower calibrations: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<config>.json`;
- bimanual follower staging: `~/.cache/huggingface/lerobot/makermodslab_biso/<robot>/follower/`.

Source: `makermodslab/utils/config.py:28-56`, `makermodslab/utils/config.py:317-425`, `makermodslab/utils/config.py:577-634`.

The frontend passes the selected record’s follower ports/configs, mode, robot name, motor power, and selected cameras by value in `StartInferenceRequest`. The backend uses those submitted values; it does not re-fetch the named robot record to construct the session.

Source: `frontend/src/components/landing/InferenceModal.tsx:397-418`, `makermodslab/rollout.py:65-92`.

## 5. Compatibility and preflight pathways

### 5.1 Policy-config inspection

Changing checkpoint step calls `/jobs/{job}/checkpoints/{step}/policy-config`. The response contains:

- policy type;
- expected visual feature names and their height/width;
- whether a task string is relevant;
- flat state dimension;
- flat action dimension.

Source: `frontend/src/components/landing/InferenceModal.tsx:225-261`, `makermodslab/server.py:1560-1573`, `makermodslab/jobs.py:1505-1543`.

### 5.2 Single-arm versus bimanual compatibility

For SO-101 checkpoints, a recognizable 6-dimensional state is treated as single arm and 12 dimensions as two arms. The frontend blocks both mismatch directions and forwards `checkpoint_state_dim`; the backend repeats the same guard before spawning its worker. Missing or nonstandard state dimensions are not guessed and are left to LeRobot’s runtime validation.

Source: `frontend/src/components/landing/InferenceModal.tsx:313-350`, `makermodslab/rollout.py:359-396`, `makermodslab/rollout.py:940-947`.

### 5.3 Arm identity

Before LeRobot opens the follower for rollout, MakerMods Lab opens a bare follower bus, verifies the arm against its assigned calibration, and closes the port without disabling torque. Since inference has no leader in its runtime session, the verifier looks through saved robot records for leader configurations paired with the selected follower configuration; matching the connected follower against that counterpart identifies a swapped leader/follower port as a hard refusal.

Source: `makermodslab/rollout.py:399-462`.

Hard identity findings abort startup. Warn-but-allow findings continue and are stored in the inference status. The request schema supports `skip_identity_check`, but the current frontend does not expose or submit it, so normal UI starts use the identity check.

Source: `makermodslab/rollout.py:65-92`, `makermodslab/rollout.py:676-712`, `makermodslab/rollout.py:776-788`, `makermodslab/rollout.py:858-862`.

### 5.4 Motor power preflight

The same preparation stage writes the selected follower torque percentage to volatile RAM and clears any stale `Goal_Velocity` left by a previous arm-driving feature. Failures are converted into warnings and the session continues with the previous register values. Bimanual mode performs this sequentially on both followers.

Source: `makermodslab/rollout.py:465-491`, `makermodslab/rollout.py:688-712`, `makermodslab/motor_power.py:49-70`, `makermodslab/motor_power.py:136-147`.

## 6. Camera setup pathways

### 6.1 Camera enumeration and browser matching

Opening the modal enables `useAvailableCameras`. It enumerates OpenCV indices from `GET /available-cameras`, enumerates browser video devices, and matches backend device names to browser labels so each OpenCV index can also carry a browser `deviceId` for preview. Enumeration refreshes on USB device changes.

Source: `frontend/src/hooks/useAvailableCameras.ts:18-143`, `makermodslab/server.py:2313-2350`.

If browser media APIs are unavailable, backend camera indices can still be listed without browser preview matching. If labels are hidden, the hook performs a one-time permission probe, stops that stream, and enumerates again.

Source: `frontend/src/hooks/useAvailableCameras.ts:43-72`.

### 6.2 Checkpoint camera roles

Visual checkpoint features become camera roles. Each role uses the feature suffix after `observation.images.`, while its resolution comes directly from the checkpoint’s declared feature shape.

Source: `makermodslab/jobs.py:1517-1531`.

For bimanual checkpoints, unique leading `left_` or `right_` prefixes are stripped for display and request naming so BiSO can add its runtime arm prefix. When multiple features would collide after stripping, their full names are retained.

Source: `frontend/src/components/landing/InferenceModal.tsx:81-149`.

### 6.3 Saved-camera auto-binding

The modal compares every expected policy role against the selected robot’s saved cameras, case-insensitively. It prefers the saved browser `device_id`, then falls back to saved `camera_index`. A binding is cleared if that physical index disappears from the current enumeration.

Source: `frontend/src/components/landing/InferenceModal.tsx:263-306`.

Every visual feature must have a binding to a currently enumerated camera before Start is enabled. A checkpoint with no visual features needs no camera binding.

Source: `frontend/src/components/landing/InferenceModal.tsx:339-350`, `frontend/src/components/landing/InferenceModal.tsx:567-641`.

### 6.4 Preview and release before launch

Each bound role shows a `getUserMedia` thumbnail. Transient `NotReadableError` and `AbortError` failures retry with exponential backoff; permission/device exposure changes can retrigger a failed stream.

Source: `frontend/src/components/landing/InferenceModal.tsx:50-79`, `frontend/src/hooks/useCameraStream.ts:19-106`.

When Start is pressed, setting `submitting=true` causes all preview streams to stop. The handler waits 300 ms before posting so OpenCV can claim the devices without competing with the browser.

Source: `frontend/src/components/landing/InferenceModal.tsx:352-365`.

The request sends:

- OpenCV camera index;
- checkpoint width and height;
- 30 FPS;
- no explicit FOURCC from the current modal.

The backend converts `camera_index` to LeRobot’s `index_or_path` and supplies the recording default MJPG FOURCC when an OpenCV camera has no explicit value.

Source: `frontend/src/components/landing/InferenceModal.tsx:369-390`, `makermodslab/rollout.py:494-514`.

Single-arm rollout attaches cameras to `--robot.cameras`. Bimanual rollout attaches the camera dictionary to `--robot.left_arm_config.cameras`; the right follower is camera-free in the constructed command.

Source: `makermodslab/rollout.py:610-642`.

## 7. Run parameters and Start request

The modal exposes:

- a checkpoint dropdown;
- an optional task description for language-conditioned policy types;
- a maximum duration, default 60 seconds and UI minimum 1;
- camera bindings for every expected visual role.

Source: `frontend/src/components/landing/InferenceModal.tsx:486-565`, `frontend/src/components/landing/InferenceModal.tsx:567-641`.

Start is enabled only when:

- a robot is selected;
- its record is clean;
- checkpoint and robot arm counts are compatible when recognizable;
- the selected step resolves to a checkpoint ref;
- policy config loaded successfully;
- every required camera is bound;
- no submission is already underway.

Source: `frontend/src/components/landing/InferenceModal.tsx:308-350`.

The posted request contains the selected follower configuration, policy ref, task, camera dictionary, duration, motor power, mode, optional right follower values, robot name for BiSO staging, and checkpoint state dimension.

Source: `frontend/src/lib/inferenceApi.ts:3-30`, `frontend/src/components/landing/InferenceModal.tsx:397-418`.

## 8. Backend start and worker lifecycle

### 8.1 Synchronous gate

`POST /start-inference` performs only cheap synchronous checks:

1. teleoperation is not active;
2. recording is not active;
3. another inference is not active;
4. submitted mode and recognizable state dimension do not conflict;
5. policy ref has a supported local/Hub shape.

It then claims the singleton, creates a fresh cancel event, seeds phase `starting`, launches a daemon startup thread, and returns `Inference starting`.

Source: `makermodslab/rollout.py:887-966`, `makermodslab/server.py:521-529`.

The frontend closes the modal and navigates immediately to `/inference`. Lengthy model acquisition and hardware setup are therefore observed on the inference page through status polling rather than blocking the modal request.

Source: `frontend/src/components/landing/InferenceModal.tsx:391-420`.

### 8.2 Background ordering

The background worker deliberately orders work as:

1. resolve or download policy;
2. check cancellation;
3. stage calibrations and preflight robot;
4. check cancellation;
5. construct rollout command;
6. open the persistent log;
7. spawn subprocess;
8. re-check cancellation while committing the subprocess;
9. start the stdout pump.

Source: `makermodslab/rollout.py:746-884`.

This ordering means a stop during a long model download does not proceed into serial-bus access. The in-flight `snapshot_download` itself can finish into the Hugging Face cache; the worker exits at its next cancellation check.

Source: `makermodslab/rollout.py:746-791`, `makermodslab/rollout.py:1002-1015`.

### 8.3 Single-arm construction

Single-arm preparation ensures the selected follower calibration exists and uses its stem as `--robot.id`. The rollout robot type is `so101_follower`; the selected port and optional camera dictionary are added directly.

Source: `makermodslab/rollout.py:610-619`, `makermodslab/rollout.py:696-714`.

### 8.4 Bimanual construction

Bimanual preparation derives a safe base id from the robot name, copies the selected follower calibrations to `<base>_left.json` and `<base>_right.json`, and preflights the two followers sequentially. The rollout robot type is `bi_so_follower`, with left/right ports and the shared follower calibration directory.

Source: `makermodslab/utils/config.py:563-634`, `makermodslab/rollout.py:622-694`.

### 8.5 Rollout command

The final command runs:

- the current Python interpreter;
- `-m lerobot.scripts.lerobot_rollout`;
- base rollout strategy;
- the resolved local policy path;
- server-selected CUDA, then MPS, then CPU fallback;
- mode-specific robot arguments;
- task and duration;
- explicit `--return_to_initial_position=true`.

No dataset argument is passed.

Source: `makermodslab/rollout.py:203-214`, `makermodslab/rollout.py:583-607`, `makermodslab/rollout.py:129-132`.

The process receives one newline for a single follower or two for bimanual, allowing LeRobot to accept existing calibration files without waiting on an interactive terminal prompt. A subsequent recalibration prompt receives EOF rather than entering an interactive web-incompatible path.

Source: `makermodslab/rollout.py:802-834`.

## 9. Runtime status, logs, cameras, and visualization

### 9.1 Status polling

The inference page polls `/inference-status` once per second. On the same tick it best-effort fetches `/inference-log`; a log-fetch failure does not interrupt status handling.

Source: `frontend/src/pages/Inference.tsx:29`, `frontend/src/pages/Inference.tsx:114-211`.

The page distinguishes:

- setup time before `rollout_started_at`;
- rollout time after LeRobot’s main-loop marker;
- configured duration and rollout percentage;
- structured phase;
- optional model download byte progress;
- warning, error, and final outcome.

Source: `frontend/src/pages/Inference.tsx:238-292`.

### 9.2 Phase detection

The stdout pump tees combined stdout/stderr into the log and recognizes:

- `Loading policy from` → `loading_policy`;
- `Connecting robot` → `connecting`;
- `Rollout setup complete` → `running` and rollout-start timestamp.

If upstream log wording changes, the rollout continues; the UI simply remains on a coarser earlier phase.

Source: `makermodslab/rollout.py:142-200`.

### 9.3 Download progress

Hub snapshot downloads use a custom `tqdm_class` that records bytes completed, the currently known total, and percentage into inference metadata. Until a total is known the UI displays an indeterminate bar; the total may grow as file metadata is discovered.

Source: `makermodslab/rollout.py:217-285`, `frontend/src/pages/Inference.tsx:284-292`, `frontend/src/pages/Inference.tsx:440-460`.

### 9.4 Persistent logs

Each spawned rollout writes to `~/.cache/huggingface/lerobot/inference_logs/<timestamp>.log`. `/inference-log` prefers the active session’s file, otherwise the newest prior log, and returns at most 500 trailing lines from a bounded tail read.

Source: `makermodslab/rollout.py:797-803`, `makermodslab/rollout.py:1037-1078`, `makermodslab/server.py:548-554`.

The frontend renders the tail in a collapsible, monospace panel that auto-scrolls only while the user remains pinned near the bottom.

Source: `frontend/src/components/LogPanel.tsx:14-80`, `frontend/src/pages/Inference.tsx:462-464`.

### 9.5 Live camera and joint visualization

Camera thumbnails exist only in the pre-launch modal and are intentionally released before the Start POST. The `/inference` runtime page renders status, timers, progress, Stop, outcomes, and logs; it does not render `CameraFeed`, `VisualizerPanel`, `UrdfViewer`, or subscribe to `/ws/joint-data`.

Source: `frontend/src/components/landing/InferenceModal.tsx:50-79`, `frontend/src/pages/Inference.tsx:1-27`, `frontend/src/pages/Inference.tsx:294-464`.

The rollout runs in a subprocess and does not receive the server `ConnectionManager`, so it has no in-process joint-data broadcast path.

Source: `makermodslab/rollout.py:887-966`, `makermodslab/server.py:806-835`.

## 10. Natural completion, Stop, leave, return pose, and release

### 10.1 Natural completion

LeRobot governs the configured duration. On clean subprocess exit, the next status poll returns a terminal `outcome="ok"`. The frontend marks the exit handled, toasts “Inference finished,” and navigates to Landing.

Source: `makermodslab/rollout.py:1112-1151`, `frontend/src/pages/Inference.tsx:148-171`.

The frontend also monitors the rollout-only timer. If the main loop exceeds configured duration by more than ten seconds, it reports that inference seems hung and requests Stop. Policy load, model download, bus connection, and camera connection time are excluded because this check begins only after `rollout_started_at` exists.

Source: `frontend/src/pages/Inference.tsx:173-194`.

### 10.2 Explicit Stop

Stop opens a confirmation explaining that the follower returns toward the pose where rollout began, releases torque, and goes limp. The inference pathway implements that contract through LeRobot’s explicitly pinned `return_to_initial_position` option; it does not call MakerMods Lab’s separate `rest_pose.py` flow.

Source: `frontend/src/pages/Inference.tsx:468-489`, `makermodslab/rollout.py:599-605`.

Backend Stop branches are:

- **No active inference:** return 409.
- **Startup without a committed subprocess:** set the cancel event, clear state, and return success. The startup worker exits at its next cancellation point.
- **Committed subprocess:** stamp `stopping`, call `terminate()`, wait up to five seconds, call `kill()` if necessary, then clear state.

Source: `makermodslab/rollout.py:984-1034`, `makermodslab/server.py:532-540`.

The page marks explicit Stop as handled before posting, so its unmount guard does not send a duplicate request. The following status transition to inactive returns the user to Landing.

Source: `frontend/src/pages/Inference.tsx:213-228`.

### 10.3 Page-leave safety

The session exit guard is active for every live phase, including model download:

- reload/tab close/typed URL: native unload prompt, then best-effort keepalive POST on actual page hide;
- browser Back: sentinel history entry plus blocking confirmation, then Stop;
- other in-app route change: Stop during component cleanup;
- natural completion or explicit Stop: caller marks the exit handled to suppress duplicate requests.

Source: `frontend/src/pages/Inference.tsx:98-112`, `frontend/src/hooks/useSessionExitGuard.ts:91-166`.

The page’s own top-left navigation button performs a confirmation because `navigate("/")` is a push and does not trigger the popstate branch. Confirmed navigation is then caught by the guard’s unmount cleanup, which posts Stop.

Source: `frontend/src/pages/Inference.tsx:294-316`.

## 11. Error, retry, and recovery pathways

### 11.1 Modal-time errors

Synchronous mutex, arm-count, and policy-ref errors return from the Start POST while the modal is still mounted. The modal shows a destructive toast, clears `submitting`, and re-enables camera previews so the user can adjust the selection and try again.

Source: `frontend/src/components/landing/InferenceModal.tsx:391-429`, `makermodslab/rollout.py:899-957`.

Checkpoint-list failure becomes an empty-checkpoint state. Policy-config failure is rendered inline and prevents Start. Camera enumeration failure becomes an empty list; browser camera streams independently retry transient device-busy failures and can recover on a media-device change.

Source: `frontend/src/components/landing/InferenceModal.tsx:202-261`, `frontend/src/components/landing/InferenceModal.tsx:490-503`, `frontend/src/components/landing/InferenceModal.tsx:571-582`, `frontend/src/hooks/useCameraStream.ts:19-106`.

### 11.2 Background startup errors

Model download, calibration staging, arm identity, bus preparation, or process-spawn failures use a common terminal state:

- `phase="error"`;
- `outcome="failed"`;
- short error text;
- optional friendly hint;
- original policy ref.

Source: `makermodslab/rollout.py:717-743`, `makermodslab/rollout.py:765-788`, `makermodslab/rollout.py:815-827`.

This terminal startup payload is returned once by `/inference-status`, then its metadata is cleared, matching subprocess-exit finalization.

Source: `makermodslab/rollout.py:1085-1111`.

### 11.3 Runtime and cleanup outcomes

For a nonzero subprocess exit, the backend reads a bounded log tail and extracts the last Python exception line plus following message text, or falls back to the final non-empty lines. The snippet is capped at 500 characters.

Source: `makermodslab/rollout.py:517-564`.

Outcome classification is:

- zero/empty return code: `ok`;
- nonzero return after the rollout started, with a recognized overload/torque cleanup marker: `ran_with_warning`;
- every other nonzero return: `failed`.

Source: `makermodslab/rollout.py:567-580`, `makermodslab/utils/errors.py:29-45`.

Friendly hints cover common motor overload, missing motor, model download/disk/Hub access, arm connection, camera frame/resolution, and serial-port permission messages.

Source: `makermodslab/utils/errors.py:80-127`.

### 11.4 Frontend terminal handling

For `failed` or `ran_with_warning`, the inference page stays open and freezes further polling so the one-shot terminal payload is not replaced by a later idle response. Cleanup warnings render amber, failures red, with the hint, extracted error, and persistent log available together.

Source: `frontend/src/pages/Inference.tsx:88-96`, `frontend/src/pages/Inference.tsx:148-170`, `frontend/src/pages/Inference.tsx:244-251`, `frontend/src/pages/Inference.tsx:370-415`.

There is no inline Retry button. Recovery returns through “Back to jobs” to Landing, where the user can adjust robot, calibration, checkpoint, task, camera bindings, or model source and launch a new singleton session.

Source: `frontend/src/pages/Inference.tsx:409-415`.

### 11.5 Backend connection loss

If status polling fails, the page reports “Lost connection to backend.” Log-fetch failures are quieter and simply retry on the next status tick.

Source: `frontend/src/pages/Inference.tsx:140-147`, `frontend/src/pages/Inference.tsx:195-203`.

## 12. Cross-feature ownership and handoffs

### 12.1 Hardware feature mutex

Inference refuses to start while teleoperation or recording is active, and refuses a second inference. Recording and teleoperation likewise check `rollout.inference_active` before claiming their sessions.

Source: `makermodslab/rollout.py:899-920`, `makermodslab/record.py:445-476`, `makermodslab/teleoperate.py:564-586`.

Training jobs, model downloads, dataset operations, and calibration managers own separate state. The inference start gate itself consults teleoperation, recording, and inference flags; it does not claim those other state machines.

Source: `makermodslab/rollout.py:887-966`.

### 12.2 Model deletion protection

Once startup has resolved and committed the local policy path, `inference_in_use_path()` exposes it while inference remains active. Local model deletion resolves both paths and refuses deletion when the active checkpoint equals or sits under the target run/model directory.

Source: `makermodslab/rollout.py:848-862`, `makermodslab/rollout.py:969-981`, `makermodslab/models.py:962-1082`.

### 12.3 Job handoff

Training and inference share checkpoint contracts through:

- `/jobs/{id}/checkpoints`;
- `/jobs/{id}/checkpoints/{step}/policy-config`;
- the opaque local/Hub `ref` returned for each checkpoint.

Inference reads those contracts but does not mutate the owning training job. A selected inherited checkpoint is routed to its owning job before the modal opens.

Source: `makermodslab/server.py:1551-1573`, `frontend/src/components/jobs/JobCard.tsx:287-299`.

### 12.4 Dataset handoff

The normal project sequence is dataset recording → model training → checkpoint inference, but the inference runtime consumes no dataset and creates no evaluation dataset. Its base-strategy rollout command passes policy, robot, task, duration, and teardown behavior only.

Source: `makermodslab/rollout.py:129-132`, `makermodslab/rollout.py:583-607`.

### 12.5 Exit destinations

- Clean natural finish: completion toast, then Landing.
- Explicit Stop: status becomes inactive, then Landing.
- Confirmed page leave: destination chosen by navigation, with Stop sent in parallel/before leaving.
- Failure or cleanup warning: remain on `/inference` until “Back to jobs,” which navigates to Landing.
- Start rejected synchronously: remain in the configuration modal.

Source: `frontend/src/pages/Inference.tsx:148-171`, `frontend/src/pages/Inference.tsx:213-228`, `frontend/src/pages/Inference.tsx:294-316`, `frontend/src/pages/Inference.tsx:409-415`, `frontend/src/components/landing/InferenceModal.tsx:421-429`.

## 13. Tests and coverage boundaries

Backend tests cover these inference contracts using mocks and temporary paths:

- request defaults and required fields;
- single/bimanual command construction;
- local and Hub policy ref resolution;
- camera CLI formatting and default MJPG;
- arm-count and hardware-feature mutex gates;
- startup phase transitions;
- background Hub download progress and failure;
- cancellation during download;
- stop/status/outcome/error classification;
- explicit return-to-initial-position flag;
- active-model deletion containment.

Source: `tests/test_rollout.py:73-1152`, `tests/test_models.py:1289-1348`.

Related coverage includes:

- local/Hub/imported checkpoint discovery and config reading: `tests/test_jobs.py:346-476`;
- imported-job registration and deduplication: `tests/test_jobs.py:476-702`;
- follower-only identity refusal and counterpart-slot lookup: `tests/test_arm_identity.py:686-780`;
- active/newest inference-log selection and bounded tails: `tests/test_log_endpoints.py:29-101`;
- motor-power request defaults and RAM writes: `tests/test_motor_power.py:54-207`.

No frontend component tests are present for `InferenceModal`, `Inference.tsx`, the model-to-job launch mapping, or the session exit guard. Hardware, real-camera, real-Hub, and full browser end-to-end execution lie outside the pure/mocked coverage described above.
