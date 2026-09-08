# Core Functions

The minimal environment where a user can record, train, and run a policy while doing everything correctly — pure happy path, inside the [Usage Premises](Usage%20Premises.md) envelope.

### 0. Serve
[x] Run `makermodslab`. Built on frontend. One process on :8000

### 1. Robot Config

For single arm and bimanual
[x] Create/select a robot record.
[x] Enumerate serial ports
[x] **Detect** (identify-arm)
[x] **Wiggle** to tell arms apart physically.
[x] Assign ports with explicit confirmation.
[x] Calibrate all arms
[x] Enumerate and support cameras

### 2. Teleoperation

[x] Start teleop, confirm models are accurate
[-] Camera previews are accurate
[x] Stop in an extended position, arms return gracefully

### 3. Recording

[x] Start a session, config and cameras autofilled
[x] Record 5 episodes, click "done" after finishing recording 2nd, they save.
[-] Resume recording, record 5 episodes, re-record episode 1. Click done in the middle of recording 2nd episode. 1 episode should save.
[x] Record 2 more episodes in a separate dataset. 
[x] Merge with first.

### 4. Dataset

[] Replay/inspect an episode of the just-recorded dataset — **GAP, not implemented**: current "replay" is the Hub-Space iframe (Hub-only, unusable offline/local); local episode replay is planned

### 5. Training

[x] Train ACT locally
[x] Train SmolVLA on HF Jobs
[x] Completed run's checkpoint list shows the final policy

### 6. Inference (deploy)

Use a verified working policy
[x] Start a session, config and cameras autofilled
[-] Deploy successfully
