# Core Functions

The minimal environment where a user can record, train, and run a policy while doing everything correctly — pure happy path, inside the [Usage Premises](Usage%20Premises.md) envelope.
### 0. Serve

[x] Run `makermodslab`. Built on frontend. One process on :8000

### 1. Robot Config

For single arm and bimanual
[x] Create/select a robot record.
[x] Enumerate serial ports
[x] **Detect** (identify-arm)
[] **Wiggle** to tell arms apart physically.
[x] Assign ports with explicit confirmation, duplicate-port guard, swaps land atomically.
[x] Calibrate all arms
[x] Enumerate and support cameras

### 2. Teleoperation

[x] Start teleop, confirm models are accurate
[x] Camera previews are accurate
[x] Stop in an extended position, arms return gracefully

### 3. Recording

[x] Start a session, config and cameras autofilled
[x] Record 2 episodes, click "done" prematurely, they save.
[] Resume recording
[x] Start recording an episode and re-record. Pressing done saves all completed episodes and returns arms to rest.

### 4. Dataset

[] Replay/inspect an episode of the just-recorded dataset — **GAP, not implemented**: current "replay" is the Hub-Space iframe (Hub-only, unusable offline/local); local episode replay is planned

### 5. Training

[x] Train ACT locally
[] Train SmolVLA on HF Jobs
[x] Completed run's checkpoint list shows the final policy

### 6. Inference (deploy)

[] Start a session, config and cameras autofilled
[] Deploy successfully
