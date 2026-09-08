# Pathway Test Checklist — Works As Intended

Comprehensive functional acceptance sheet: **every intended feature, tested for "works as intended"** — not merely that it's reachable. A line passes only when the feature actually does what it's supposed to, end to end (operation completes, correct result), on real hardware where hardware is involved.

Organized by the **6 workflow phases** (Robot Config → Teleoperation → Recording → Dataset → Training → Inference), matching [Core Functions.md](Core%20Functions.md) and the dated runs, with **Serve** as phase 0. This sheet is a **superset of Core Functions** — the ⭐ lines ARE the Core Functions minimal happy-path (the end-to-end backbone); everything else is extended coverage that hangs off each phase. **Placement rule:** each line sits in the phase where you actually *verify* it (e.g. camera-key prefixing is applied at record time but *checked* by inspecting the dataset, so it's under Dataset). Feature surface from [Features Over LeLab.md](Features%20Over%20LeLab.md). Run single-arm **and** bimanual wherever a robot is driven.

Legend — `[]` not tested · `[x]` **works as intended** · `[~]` partially / flaky (note how) · `[✗]` **does not work** — always add `— what's wrong, date` (+ link a bug entry) · `[-]` n/a / deferred · **⭐ = Core Functions** (minimal happy-path) · **GAP** = intended but not implemented. Targets the redesign UI on `integrate/andrew-into-redesign`.

## Test order

The 6 phases are a **dependency chain** — each needs the output of the previous, so test **in phase order (0 → 6)**: you can't teleop without a calibrated robot, can't record without a working robot, can't train without a dataset, can't deploy without a trained model. Within Phase 1, order is **create robot → enumerate/detect/assign ports → calibrate → cameras** (each step needs the one before).

Run it in **two passes:**
1. **⭐ Core smoke first** — walk only the ⭐ lines, 0 → 6, as one continuous thread (create + calibrate a robot → teleop → record a small dataset → train ACT on it → deploy that model). This proves the whole pipeline and its handoffs end-to-end with the fewest steps. If a ⭐ line fails, the pipeline is broken there — fix before going further.
2. **Extended pass** — return phase-by-phase and fill in the non-⭐ lines (guards, edge cases, management ops, bimanual variants).

**Platform splits:** where the Mac and Linux/Jetson pipelines are *different code* (camera enumeration, MJPG/USB-bandwidth, local-training device), the line is split into `(Mac)` and `(Linux/Jetson)` variants — passing one does NOT validate the other. On a given bench run (`Mac.md` / `Jetson.md`), tick your platform's line and mark the other `[-]`.

> **Master template — keep this copy unticked.** Copy into `tests/<DD-MM-YY host>/Pathway Test Checklist.md` per bench run and mark there. Confirmed fails feed [bugs/](bugs/).

---

## 0. Serve
[] ⭐ `makermodslab` starts, one process on `:8000`, browser opens the Launchpad
[] Second tab → SingleTabGuard blocks it (only one live UI)

**Headless / Jetson**
[] `makermodslab --stop` stops a running MakerMods Lab + reclaims ports; port preflight gives actionable errors
[] Jetson `uvcvideo` DKMS patch enables >2 USB cameras
(`--lan` / `--offline` / `makermodslab-station` removed — flagged for deletion; deleting the launcher code is a separate main change, not yet done)

## 1. Robot Config
*(only what you verify while configuring a robot — record/inference-time behavior is checked in those phases)*
[] ⭐ Create a single-arm robot (one leader + one follower); layout fixed after creation
[] ⭐ Create a bimanual (4-arm) robot with left/right leaders + followers; layout fixed after creation

**Ports & detection**
[] Guided robot-creation dialog produces a usable robot; rename works; selection shared app-wide
[] Ports / calibrations / cameras / motor-power persist per robot (re-select the robot → settings survive)
[] ⭐ Enumerate serial ports (available-ports lists the connected buses)
[] ⭐ Detect an arm (hand-motion identify without energizing); gripper-wiggle works as the alternative
[] ⭐ Assign ports: confirmation shown, duplicate-port blocked, atomic reassign when the port belonged to another slot, release works
[] Saved-but-disconnected ports show as unavailable

**Calibration**
[] ⭐ Calibrate all arms — fully-automatic completes; manual flow still works
[] Batch autocal of up to 4 selected arms, concurrent, with per-arm progress/logs/failure/cancel
[] Named calibration files (auto-named sensibly); library UI lists them; rename/delete work
[] Calibration JSON import + download/export; open leader/follower folder; deleting an in-use calibration safely unassigns
[] Save/Quit drafts accumulate then commit; cancel/failure cleans up incomplete files (existing profile survives)
[] (Manual calibration, fallback for when auto can't run) Starting-position validation + `wrist_roll` full-turn handling correct

**Cameras (enumerate & configure)**
[] ⭐ (Mac) Enumerate + assign cameras — AVFoundation lists real device names; identity by uniqueID (port-path); correct cv2 indexes
[] ⭐ (Linux/Jetson) Enumerate + assign cameras — V4L2 `VIDIOC_QUERYCAP` lists devices (busy cameras stay visible); index-based identity; correct cv2 indexes
[] Live previews render while configuring a robot (both platforms)
[] (Mac) Cameras default to MJPG FOURCC
[] (Linux/Jetson) 720p30 preview request negotiates MJPEG (not raw YUV) + MJPG default avoids USB-bandwidth exhaustion
[] (Mac) 3+ cameras run concurrently (powered hub covers the power ceiling)
[] (Linux/Jetson) 3+ cameras run concurrently (uvcvideo DKMS patch / split across USB buses for the bandwidth ceiling)

**Hardware safety (identity & power)**
[] ⭐ Arm-identity guard fires **before motors energize** when starting a driven session; catches swapped leader/follower (calibration + EEPROM fingerprint), single and bimanual *(verify by swapping arms → start teleop/record → it blocks)*
[] "Matches another calibration" detection + position-range fallback for factory-reset arms; warnings vs hard blocks at the right severity
[] Per-robot motor-power limit applied; live supply-voltage reads correctly; stale motor speed caps cleared before a session; bounded motor-bus retries on noisy USB

## 2. Teleoperation
[] ⭐ Start teleop — drives follower(s) from leader(s) accurately, single-arm and bimanual
[] Dual-arm 3D visualization tracks both arms (single-arm tracks the one)
[] (Linux/Jetson) Teleop rate stays usable despite CH341 serial latency (~16ms/read, no `latency_timer` knob) — not visibly laggy
[] ⭐ Camera previews render accurately during teleop; WebGL-unavailable → falls back instead of crashing

**Stop & safety**
[] ⭐ Stop in an extended pose: freeze → return toward start pose → release torque (normal/interrupted/error paths); bimanual followers return concurrently
[] Second Stop skips the graceful return and releases immediately; stalled/failed return falls back to hold-then-release
[] Power telemetry (peak/avg) reports; page-leave protection (unintended nav = Quit, not silent save)

## 3. Recording
[] ⭐ Start a session — config + cameras autofill (missing saved cameras flagged, not silently substituted); prep phases shown; logs persist; preview released before record takes camera ownership
[] Local recording with a bare dataset name (no HF login) works
[] ⭐ Record 2 episodes — single-arm AND bimanual; click **Done** prematurely → completed episodes save
[] ⭐ **Resume** recording into an existing dataset appends correctly; 0-episode fresh session auto-cleaned
[] ⭐ **Re-record episode** (+ Backspace); pressing Done saves all completed episodes and returns arms to rest
[] Background upload runs with progress/error status
[] Dataset mutation blocked while recording/uploading/merging/local-training uses it

**Stop & safety**
[] Followers return to start pose on session end (bimanual concurrent); torque releases on normal/interrupted/error paths
[] Done vs Quit semantics correct; outcomes (success/warning/failure/stopped/quit) reported with recovery hints; teardown failure distinguished from in-session failure; page-leave protection

## 4. Dataset

**Recorded-dataset keys** *(verify the recording produced correct feature keys — inspect the dataset's `meta/info.json`, or the deferred `/debug/dataset/{id}`)*
[] Single-arm recording → **UNprefixed** camera + calibration keys (`observation.images.front` / `wrist`; no `left_`/`right_`)
[] Bimanual recording → `left_`/`right_` prefixed keys, **each exactly once** (no double-prefix like `left_left_front`)

**Library**
[] ⭐ Inspect the just-recorded dataset (episodes/frames/cameras) — *(⭐ GAP: local episode **replay** not implemented; current "replay" is the Hub-Space iframe, Hub-only/offline-unusable → mark `[✗ GAP]`)*
[] Combined local + downloaded + Hub listing; select from landing
[] Info card correct: episodes, frames, duration, FPS, cameras, robot type, tasks + per-task counts, disk size, local/Hub status
[] Empty/vision-unusable datasets warned; full repo names; namespace-first sort
[] Create / add-Hub / background-download / import-from-disk / rename / delete all work
[] Hide/unhide Hub entries; visibility (public/private) + tag editing after upload; default makermods/openbooth tags; upload-permission check
[] Cached-dataset management + cleanup; offline-aware Hub status; parallel/cached listings degrade gracefully offline

**Merging**
[] Merge 2+ datasets from the browser (via `aggregate_datasets`)
[] Preflight rejects mismatched cameras / FPS / feature shapes with a clear error *(the camera-mismatch guard)*
[] Missing local Parquet metadata caught; output-name validated; existing-name overwrite blocked; bare name inherits namespace
[] Background status + persistent logs; partial output cleaned up on failure

## 5. Training

**Training**
[] ⭐ Train ACT locally — job starts, logs stream, metrics/checkpoints appear
[] ⭐ Train SmolVLA on HF Jobs — job reaches a terminal state visible in the UI
[] ⭐ Completed run's checkpoint list shows the final policy
[] Policy + dataset chosen and frozen before the run; create-model flow works; policy-availability check + extras-install prompt
[] Sensible optimizer defaults; run/repo names validated
[] (Mac) Local training runs on **MPS** (device auto-detected)
[] (Linux/Jetson) Local training runs on **CUDA** (device auto-detected); cuBLAS initializes — no `CUBLAS_STATUS_NOT_INITIALIZED` (cu13 wheel / system-cublas mismatch)
[] Offline guard blocks training a Hub-only uncached dataset; upload-before-cloud-training flow works; cloud upload visibility default (⚠ confirm intended: private vs public — see bugs)
[] Configurable HF-Jobs timeout honored
[] Continue local from checkpoint; continue cloud from Hub checkpoint; checkpoint picker + config inspection
[] Resumed-run lineage nested; loss stitched across resumes; global-step correct after resume; denser loss/LR charts
[] Job aliases/imports/dedup/cleanup; remove finished/failed/orphaned cloud entries; install missing policy-extra from a failed job card

**Model library**
[] Models listed separately from jobs (local/downloaded/pinned/Hub combined)
[] Info card: policy type, dataset, size, update time, local path, Hub status
[] Add-Hub / download / import-checkpoint / upload / delete work; hide/unhide Hub models
[] Deleting a model in use by inference is blocked
[] Policy-type detection + availability/stability labels; auto-import of discoverable Hub models; dedup of case-variant imports; display-name aliases
[] **Deploy selected model** action works from landing

## 6. Inference
[] ⭐ Start a session — config + cameras autofill; every feature bound to a physical camera w/ live preview; auto-bind from robot when names match; bimanual camera-prefix mapping correct
[] ⭐ Deploy successfully — runs the selected model+checkpoint on the robot (local/downloaded/imported/Hub; single-arm and bimanual)
[] Arm-count compatibility checked before hardware is touched (6-dim single, 12-dim bimanual)
[] Async model download with byte progress; page shown before long startup; phases (download/arm/camera/run/stop) reported; logs persist
[] Meaningful exception tail + friendly hints on common failures; clean cancel during download
[] Deleting the deployed model mid-rollout is blocked

**Stop & safety**
[] Graceful stop returns follower(s) and releases torque (normal/interrupted/error paths); second stop releases immediately

---

**How to run:** do the **⭐ core smoke** (0→6) first, then the extended pass per phase. Mark `[x]` only when the feature *does the right thing*, `[✗ — reason, date]` when it doesn't. Each line is placed where you can verify it at that point in the walk. Copy this file into `tests/<DD-MM-YY host>/` per bench run; keep this master unticked. When a `[✗]` is a UI-wiring loss rather than a backend bug, cross-check the "Bench-run findings" section of [`bugs/CURRENT 23-07-26.md`](bugs/CURRENT%2023-07-26.md) — that is where the known dropped-UI-entry-point regressions are recorded. Hardware-dependent lines need the arms/cameras connected and the branch's matching lerobot env.
