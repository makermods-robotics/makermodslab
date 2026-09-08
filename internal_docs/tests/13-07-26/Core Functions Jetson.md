### 0. Serve

[x] Run `makermodslab`. Built on frontend. One process on :8000

### 1. Robot setup

For single arm and bimanual
[x] Create/select a robot record.
[x] Enumerate serial ports
[x] **Detect** (identify-arm)
[x] **Wiggle** to tell arms apart physically.
[x] Assign ports with explicit confirmation, duplicate-port guard, swaps land atomically.
[x] Calibrate all arms
[x] Enumerate and support cameras

### 2. Teleoperation

[x] Start teleop, confirm models are accurate
[! webgl not supported] Camera previews are accurate
[x] Stop in an extended postiion, arms return gracefully

### 3. Recording

[x] Start a session, config and cameras autofilled
[x] Record 2 episodes, click "done" prematurely, they save.
[x] Resume recording
[x] Start recording an episode and re-record. Pressing done saves all completed episodes and returns arms to rest.

### 4. Training

[x] Train ACT locally

### 5. Inference (deploy)

[x] Start a session, config and cameras autofilled
[] Deploy local model successfully
[] Deploy trained model successfully