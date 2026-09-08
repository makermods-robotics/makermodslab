# MakerMods Lab Features Over LeLab

MakerMods Lab is a fork of Hugging Face's [LeLab](https://github.com/huggingface/leLab). It retains LeLab's browser-based workflow for calibration, teleoperation, recording, training, inference, replay, and Hugging Face Hub uploads, while extending it with hardware safety, bimanual operation, richer data and model management, and headless robot-station support.

This inventory was audited on July 14, 2026, using the current MakerMods Lab development checkout at commit `8af5b09` and LeLab `main` at commit [`def3e9e`](https://github.com/huggingface/leLab/commit/def3e9e51e99c03e01b214dc8a0d9b7c2dd5f0da). It describes the current development checkout, including work not yet present on MakerMods Lab's public `main` branch.

## Bimanual robot support

- First-class single-arm and bimanual robot layouts.
- Four-arm configuration with left/right leaders and left/right followers.
- Four-arm calibration workflow.
- Concurrent calibration of a selected subset of arms.
- Bimanual teleoperation.
- Dual-arm 3D visualization.
- Bimanual dataset recording.
- Bimanual policy inference.
- Checkpoint-versus-robot arm-count validation before inference.
- Simultaneous return-to-rest for both followers.
- Correct bimanual camera-name and calibration staging behavior.

## Robot and port setup

- Guided robot-creation dialog.
- Immutable arm layout after robot creation, preventing stale single/bimanual state.
- Robot renaming.
- Shared robot selection throughout the application.
- Saved ports, calibrations, cameras, and motor-power limits per robot.
- Hand-motion port detection: swing an arm by hand to identify its serial port without energizing it.
- Gripper-wiggle identification as an alternative.
- Confirmation before assigning a detected port.
- Duplicate-port protection.
- Atomic reassignment when a detected port belongs to another slot.
- Ability to release an existing port assignment.
- Saved but disconnected ports treated as unavailable.
- Per-platform camera enumeration with real device names.

## Hardware safety

- Arm-identity guard before motors are energized.
- Calibration and EEPROM fingerprint matching to detect swapped leader and follower arms.
- Identity checks for every arm participating in single or bimanual sessions.
- Detection when an arm appears to match another calibration in the library.
- Position-range fallback checks for unidentified or factory-reset arms.
- Risk-appropriate warnings and hard blocks.
- Per-robot motor-power limiting.
- Live supply-voltage measurement.
- Teleoperation power telemetry, including peak and average values.
- Clearing of stale motor speed caps before sessions.
- Bounded motor-bus retries for noisy USB environments.
- Reliable torque release across normal, interrupted, and error paths.

## Graceful stopping and session safety

- Teleoperation freezes followers, returns them toward their starting pose, and then releases torque.
- Recording returns followers to their starting pose when the session ends.
- Bimanual followers return concurrently.
- Automatic calibration uses the same controlled stop behavior.
- Pressing Stop a second time skips the graceful return and releases immediately.
- Failed or stalled returns fall back to holding and releasing safely.
- Page-leave protection for recording and teleoperation.
- Explicit **Done** and **Quit** semantics.
- Unintentional navigation is treated as quitting instead of silently saving.
- Consistent session outcomes: success, warning, failure, stopped, and quit.
- User-facing error summaries and suggested recovery hints.
- Teardown failures distinguished from failures during the active session.

## Calibration management

- Fully automatic calibration in addition to LeLab's manual flow.
- Concurrent batch auto-calibration of up to four selected arms.
- Named calibration files instead of repeatedly overwriting one slot.
- Automatic sensible calibration names.
- Calibration library UI.
- Calibration rename and delete actions.
- Calibration JSON import and download/export.
- Direct opening of leader and follower calibration folders.
- Safe unassignment when an in-use calibration is deleted.
- Explicit **Save** and **Quit**, with edits accumulated as local drafts.
- Starting-position validation for invalid calibration sweeps.
- Full-turn handling for `wrist_roll`.
- Cleanup of incomplete calibration files after failure or cancellation.
- Per-arm progress, logs, failure status, and graceful cancellation.

## Cameras

- Live camera previews while configuring a robot or recording.
- Camera previews during teleoperation.
- Camera previews and explicit camera-to-feature binding before inference.
- Saved robot cameras automatically prefill recording and inference.
- Missing saved cameras are flagged instead of silently substituted.
- Camera previews before naming a recording dataset.
- Correct physical camera names and OpenCV indexes on macOS, Windows, and Linux.
- Linux enumeration through V4L2 capability queries, allowing busy cameras to remain visible.
- MJPG capture defaults for recording and inference to reduce USB bandwidth.
- 720p/30 preview requests that encourage MJPEG negotiation.
- Explicit preview release before recording or inference takes camera ownership.
- Retry handling for transient camera startup failures.
- Bimanual checkpoint camera-prefix mapping.
- WebGL-unavailable fallback instead of crashing teleoperation.

## Recording

- Bimanual recording.
- Local recording without a Hugging Face login by using bare dataset names.
- Automatic population of robot configuration and cameras.
- First-class red **Stop** action.
- First-class **Re-record episode** action with a Backspace shortcut.
- Stopping during an episode discards only that incomplete episode.
- Pressing Done preserves completed episodes.
- Resume recording into an existing dataset.
- Automatic cleanup of fresh sessions that save no episodes.
- Detailed preparation phases, including arm and camera connection.
- Persistent recording logs.
- Background dataset upload with progress and error status.
- Protection against dataset mutation while recording, uploading, merging, or local training is using it.

## Dataset library

- Combined listing of local, downloaded, and Hub datasets.
- Dataset selection from the landing page.
- Dataset information cards showing:
  - Episodes and frames.
  - Duration and FPS.
  - Cameras.
  - Robot type.
  - Tasks and per-task episode counts.
  - Disk size.
  - Local and Hub availability.
- Warnings for empty or vision-unusable datasets.
- Full repository names instead of ambiguous short names.
- Sorting with the user's namespace first.
- Dataset creation from the landing page.
- Add or pin an arbitrary Hub dataset.
- Background Hub downloads with progress.
- Dataset import from disk.
- Safe local dataset rename and deletion.
- Hide and unhide unwanted Hub entries.
- Background upload with status polling.
- Public/private visibility controls.
- Hub tag editing after upload.
- Default MakerMods Lab and OpenBooth tags.
- Upload permission checks.
- Cached-dataset management and cleanup.
- Offline-aware Hub status.
- Parallel and cached Hub listings with bounded timeouts.
- More resilient behavior on unreliable or restricted internet connections.

## Dataset merging

- Merge two or more datasets from the browser.
- Integration with LeRobot's `aggregate_datasets` command.
- Preflight checks for mismatched cameras, FPS, and feature shapes.
- Validation of required local Parquet metadata.
- Output-name validation.
- Protection against overwriting an existing dataset.
- Namespace inheritance for bare output names.
- Background merge status and persistent logs.
- Partial-output cleanup after failure.
- Clearer compatibility and command errors.

## Model library

- Models represented separately from training jobs.
- Combined listing of locally trained, downloaded, pinned, and Hub models.
- Model information cards with policy type, dataset, size, update time, local path, and Hub status.
- Add or pin arbitrary Hub policy repositories.
- Model downloads with progress.
- Checkpoint import from disk.
- Upload of locally trained models.
- Safe deletion of local and downloaded models.
- Hide and unhide Hub models.
- Protection against deleting a model while inference is using it.
- Policy-type detection from metadata and tags.
- Policy availability and stability labels.
- Automatic import of discoverable Hub models.
- Deduplication of repeated or case-variant imports.
- Display-name aliases for models and jobs.
- Model and checkpoint selection from the landing page.
- Persistent **Deploy selected model** action.

## Training

- Policy type and dataset selected before entering training.
- Dedicated create-model flow.
- Policy and dataset choices frozen for the run.
- Policy availability checks.
- Installation prompts for policy-specific LeRobot extras.
- Clear local and Hugging Face Jobs compute targets.
- Device auto-detection across CUDA, MPS, and CPU.
- More practical default optimizer and configuration values.
- Validated run and repository names.
- Offline guard against training a Hub-only dataset that is not cached locally.
- Upload-before-training flow when cloud training needs a local-only dataset.
- User-configurable Hugging Face Jobs timeout.
- Continue local training from a saved checkpoint.
- Continue cloud training from a Hub checkpoint.
- Checkpoint picker and checkpoint configuration inspection.
- Nested visualization of resumed-run lineage.
- Loss history stitched across resumed runs.
- Correct global-step display after resuming.
- Denser loss and learning-rate charts.
- Job aliases, imports, deduplication, and cleanup.
- Removal of finished, failed, untracked, or orphaned cloud entries.
- Missing policy-extra installation directly from a failed job card.

## Inference

- Deploy the selected model and checkpoint from the landing page.
- Run local, downloaded, imported, or Hub-hosted policies.
- Bimanual inference.
- Arm-count compatibility checks before hardware is touched.
- Discovery of required camera features from checkpoint metadata.
- Physical camera binding for every expected feature, with live previews.
- Automatic binding from the selected robot when names match.
- Asynchronous model downloads with byte progress.
- Navigation to the inference page before lengthy startup or download work.
- Detailed phases for model download, arm connection, camera connection, execution, and stopping.
- Persistent inference logs.
- Extraction of the meaningful exception tail instead of a generic failure.
- Friendly hints for common camera, model, and hardware failures.
- Clean cancellation during model download.
- Protection against deleting a model during rollout.

## Headless, offline, and Jetson operation

- `makermodslab --lan` for LAN and headless serving without opening a browser.
- `makermodslab --offline` so Hub calls fail fast while hardware workflows remain available.
- `makermodslab --stop` to stop a previous process tree and reclaim ports.
- Port preflight checks with actionable errors.
- `makermodslab-station` command combining LAN and offline behavior.
- systemd service for boot-to-robot stations.
- Process-tree shutdown instead of terminating only the parent process.
- Self-installation of launcher commands onto `PATH`.
- UI operation from another machine on the LAN.
- Detailed macOS, Ubuntu, and Jetson installation guidance.
- Proxy, VPN, and restricted-network guidance.
- LAN cache-seeding workflow.
- JetPack, CUDA, and cuBLAS guidance.
- Vendored Jetson `uvcvideo` DKMS patch for rigs with more than two USB cameras.

## Features inherited from LeLab

The following core capabilities are important MakerMods Lab features, but they are not differentiators because the original implementations came from LeLab:

- Browser-based manual calibration.
- Single-arm teleoperation.
- Basic dataset recording.
- Local policy training.
- Basic policy inference.
- Episode replay.
- Basic dataset upload.
- Hugging Face authentication.
- The original FastAPI, React, and training-job framework.

## Comparison caveat

MakerMods Lab is a divergent fork rather than a strict superset of LeLab. At the time of this audit, the current MakerMods Lab development branch contained 207 commits not in LeLab `main`, while LeLab contained nine commits not in the MakerMods Lab branch. Upstream changes should therefore be reviewed and reconciled independently when updating the fork.
