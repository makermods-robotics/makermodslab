# Core Functions

The minimal environment where a user can record, train, and run a policy while doing everything correctly — pure happy path, inside the [Usage Premises](Usage%20Premises.md) envelope. A checklist, NOT an inventory: the full feature surface lives in [Features Over LeLab.md](Features%20Over%20LeLab.md); guard/negative-path and feature checks in [Extended Checks.md](Extended%20Checks.md); the verified bug backlog in [bogs/RANKING.md](bogs/RANKING.md).
Reorganized 2026-07-14 against the current tree (`andrew`, includes uncommitted v0.6.0 work).

### 0. Serve

[x] Run `makermodslab`. Built on frontend. One process on :8000

### 1. Robot Config

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
[x] Camera previews are accurate
[x] Stop in an extended position, arms return gracefully

### 3. Recording

[x] Start a session, config and cameras autofilled
[x] Record 2 episodes, click "done" prematurely, they save.
[x] Resume recording
[x] Start recording an episode and re-record. Pressing done saves all completed episodes and returns arms to rest.

### 4. Dataset

[] Replay/inspect an episode of the just-recorded dataset — **GAP, not implemented**: current "replay" is the Hub-Space iframe (Hub-only, unusable offline/local); local episode replay is planned

### 5. Training

[x] Train ACT locally
[] Train SmolVLA on HF Jobs, watch the job reach a terminal state in the UI (also the definitive close-out for the ruled-out terminal-stage bug)
[x] Completed run's checkpoint list shows the final policy

### 6. Inference (deploy)

[] Start a session, config and cameras autofilled
[] Deploy successfully
