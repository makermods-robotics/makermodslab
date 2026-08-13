# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## I10: real SIGTERM does not appear to gracefully stop a live, real-hardware teleoperation session

**Status: RESOLVED for the scope actually tested — false alarm caused by a test-setup bug, not a real defect. A plain `kill -TERM` against a prod-mode, single-arm, actively-driving teleoperation session correctly runs I8's graceful stop. `uvicorn --reload`, bimanual, and a signal landing mid-release are all still untested — see "What was NOT tested" below before assuming those are covered too.**

PR #29 (I8) added `stop_and_wait()` to `teleoperate.py`/`record.py` and wired both into `shutdown_event()` so that a `SIGTERM` (plain `kill <pid>`, or a `uvicorn --reload` restart) gracefully stops an active teleoperation/recording session — return-to-rest, torque release, disconnect — before the process exits, instead of orphaning the in-process control-loop thread. All of I8's automated tests pass, but they only ever invoke `shutdown_event()` directly via `asyncio.run(...)`; none of them go through a real OS signal delivered to a live process that's actually driving hardware.

### Original finding (real hardware, twice in a row) — since invalidated, see below

Tested that gap directly, against a real, connected SO-101 (single arm, leader `/dev/tty.usbmodem5B140311451` / follower `/dev/tty.usbmodem5B3E0901701`, calibration profile "autocalli robot"): started a real teleoperation session and never called stop, then sent a real `kill -TERM` to the running `makerlab` server process (prod mode, no `--reload`). Twice in a row the process exited in ~0.2–0.3s with zero log output from `handle_stop_teleoperation()`/`stop_and_wait()`, versus ~4s and full logging for an uninterrupted stop. Several hardware-free repros against "the real code" (`stop_and_wait()` directly, the real `server:app`, real pyserial reads at loop cadence) all worked correctly under SIGTERM, which pointed suspicion at the real `teleoperation_worker()` loop itself — possibly a Feetech bus call not responding the same way to a genuine OS signal as to an in-process call.

**Root cause of the original finding: it wasn't a code bug, it was a branch bug.** `fix/i10-real-sigterm-skips-teleoperation-stop` was cut from `89eef7b`, the same commit I8 branched from — but I10 was never rebased or merged with I8's commits. Every real-hardware test run described above (and the hardware-free repros) ran against a `shutdown_event()` that was still the *original two-line stub* (log line, `stop_broadcast_thread()`, log line) — it never called `handle_stop_teleoperation()` or `stop_and_wait()` at all, because I8's wiring simply wasn't present on this branch. So of course nothing happened: there was nothing to run. This was caught by inspecting `git log fix/i10...origin/fix/i8` and finding I8's four commits absent, then confirming `shutdown_event()`'s body directly (no `asyncio.gather`, no `stop_teleoperation_and_wait` reference — just the pre-I8 original).

### Corrected re-test (after merging I8 into this branch)

Merged `origin/fix/i8-shutdown-orphans-teleop-recording-threads` into this branch (clean merge, no conflicts; `git merge-base` confirmed both branches shared `89eef7b`). Re-verified `shutdown_event()` now contains I8's `asyncio.gather(asyncio.to_thread(stop_teleoperation_and_wait), asyncio.to_thread(stop_recording_and_wait), ...)`. Full suite: 867 passed. Also added temporary entry/exit logging (`[I10]` tag) around every real Feetech bus call reachable from `teleoperation_worker()` — `teleop_device.get_action()`, `robot.send_action()`, the Present_Current `sync_read` in `PowerTelemetry.sample()`, and `robot.get_observation()` in `get_joint_positions_from_robot()` — to catch a call stuck mid-flight if the failure was real.

Repeated the identical real-hardware procedure (start teleoperation, never call stop, real `kill -TERM` to the live prod-mode process) twice:

- **First run** (pre-merge, unfixed code, instrumentation only): every `[I10] enter`/`exit` pair matched — no call was ever stuck. `shutdown_event()` ran (`Cleanup completed` logged) but with no trace of `stop_teleoperation_and_wait()`, exactly reproducing (and confirming the cause of) the original finding.
- **Second run** (post-merge, I8's fix actually present): the log shows the full graceful path firing under the real signal — `Stop teleoperation triggered from web interface` → power-telemetry summary → `Rest-pose return starting`/`finished` (~4s, matching normal-stop timing) → both devices disconnected → `Teleoperation stopped` → `Broadcast thread stop requested` → `Cleanup completed`. One `[I10] enter teleop_device.get_action` line was in flight when the signal landed (interleaved with the shutdown log lines from the concurrent async handler) and still exited normally at 1.0ms — no hang, no interrupted call. `/teleoperation-status` confirmed `teleoperation_active: false` afterward with no cleanup error.

Within the scope actually tested (below), I8 protects the arm during a real `kill <pid>` exactly as designed, once its own commits are actually part of the running code. There is no signal-vs-Feetech-call mechanism to chase.

### What was NOT tested — do not assume these are fine

- **`uvicorn --reload` restarts.** Both this investigation's two runs used a real `kill -TERM` against a **prod-mode** process (`_run_prod()`, no `--reload`). `--reload` goes through uvicorn's separate file-watcher/reloader supervisor process, which is a different signal-delivery path (parent forwards the signal to a child worker, on its own timing) — not exercised here at all, despite being named as one of the two triggers this whole investigation (and I8 itself) is meant to cover.
- **Bimanual mode.** Both runs used a single-arm rig. `teleoperation_worker()` has a separate `is_bimanual` path (`BiSOFollower`/`BiSOLeader`, two buses, two rest poses, `asyncio.gather`-free sequential-looking bimanual bus access) that was never driven under a real signal.
- **A SIGTERM landing mid-release** (i.e. while a previous stop's rest-pose return/torque-release is already in flight — the `releasing` state and its "second stop forces immediate release" path). Only a SIGTERM against an actively-driving, not-yet-stopped session was tested.
- Recording's side of the same I8 mechanism (`stop_recording_and_wait`) — see below, blocked separately.

Treat the resolution above as "plain `kill -TERM` against a prod-mode, single-arm, actively-driving teleoperation session is fine" — not as "the whole real-world SIGTERM/`--reload` risk is closed."

### Still open (unrelated to the above, not blocking)

- Recording's side of the same I8 mechanism (`stop_recording_and_wait`) was not exercised end-to-end against real hardware — blocked by a pre-existing, unrelated bug (`AttributeError: 'DatasetRecordConfig' object has no attribute 'vcodec'`) that stops any real recording session from starting in this environment. Worth a real-SIGTERM pass once that's fixed, but there's no reason from this investigation to expect it behaves differently from teleoperation — same `stop_and_wait` pattern, same `asyncio.gather` wiring, same in-process thread shape.
- Whether I5 (inference) / I7 (auto-calibration) share any gap: not applicable here — this whole investigation turned out to be about a missing merge, not a subprocess-vs-thread distinction, so it says nothing new about those.

### Recommendation

Close this specific finding — the original SIGTERM-does-nothing report was a test-setup artifact, not a defect, for the scope actually tested. Don't broaden that to "I8 is fully validated on real hardware": `--reload`, bimanual, and mid-release SIGTERM are all still open and worth a real-hardware pass before treating those specifically as safe. PR #29's own diff was not found to have any defect by this investigation.

## Repository purpose

MakerMods Lab is a FastAPI + React web interface wrapping the [LeRobot](https://github.com/huggingface/lerobot) framework for the SO-101 leader/follower arm (single or bimanual). It exposes teleoperation, dataset recording, calibration, training, inference, and replay as HTTP/WebSocket endpoints, replacing LeRobot's CLI + keyboard-driven flows. It is a fork of Hugging Face's [leLab](https://github.com/huggingface/leLab), heavily extended by [makermods-robotics](https://github.com/makermods-robotics).

The frontend (React + Vite) lives in [`frontend/`](frontend/). The built bundle in `frontend/dist/` is committed and shipped inside the Python wheel as package data (`frontend.__init__.py` makes setuptools treat it as a package); [`makermodslab/server.py`](makermodslab/server.py) mounts it as `StaticFiles` at `/` so a single `makermodslab` process serves both API and UI on `:8000`.

## Upstream (leLab)

We forked leLab and have diverged past the point where git-level integration works.

**Never `git merge`, `git rebase`, or `git cherry-pick` from `upstream`.** Anything we take from leLab is a deliberate re-implementation against our code. Upstream `lelab/<x>.py` maps to our `makermodslab/<x>.py`, but the contents usually differ heavily — read our file before assuming an upstream change applies. Upstream commits often arrive with tests; the `tests/` policy below still governs which of those we keep.

The `upstream` remote is not in a fresh clone. Add it when you need it:

```bash
git remote add upstream https://github.com/huggingface/leLab.git
```

Run `/review-upstream` ([`.claude/commands/review-upstream.md`](.claude/commands/review-upstream.md)) weekly to triage what they have shipped. It writes a report under `docs/upstream/reviews/` and updates [`docs/upstream/ledger.md`](docs/upstream/ledger.md), the record of every upstream commit we have judged and why. **Read the ledger before porting anything** — some rows are `rejected` because our implementation is already ahead of theirs, and re-porting those is a regression.

Both projects are Apache-2.0. Credit ports with the upstream SHA, and send fixes back when they apply to code we still share.

## Common commands

Install and run: see [README.md](README.md) Quick Start (uv editable install; `makermodslab` / `makermodslab --dev`). Requires Python ≥3.12. Use the repo `.venv` — pytest fails to collect under other interpreters because of the pinned lerobot.

`lerobot` is pinned to the `v0.6.0` **release tag** on `huggingface/lerobot` (see [pyproject.toml](pyproject.toml)); that release exposes the `lerobot-rollout` script [makermodslab/rollout.py](makermodslab/rollout.py) shells out to. Track lerobot's releases, not its `main` branch — a moving pin makes every unrelated failure a bisect. lerobot is the one dependency the app cannot run without, so treat a bump as a real change: expect import-path drift, adjust call sites, and exercise a rollout end-to-end before landing it.

When `frontend/**` (excluding `frontend/dist/**`) changes on `main`, [`.github/workflows/build_frontend.yml`](.github/workflows/build_frontend.yml) auto-rebuilds `frontend/dist/` and commits it back — don't rebuild it by hand for a PR. `makermodslab --dev` serves from Vite, no rebuild needed.

**`frontend/package-lock.json` is the most fragile file in the repo** — it has broken CI three times (`e345e61`, `944cb05`, `c18d42c`). npm records platform-specific optional dependencies (esbuild, rollup, the `@emnapi/*` wasm shims) in it, so a lockfile written by `npm install` on macOS can fail `npm ci` on the Linux runner. Never hand-edit it; regenerate it on Linux or in a container when it genuinely needs regenerating; and treat an unexplained `package-lock.json` diff in an otherwise-unrelated PR as stray local churn to drop, not a change to keep. [`.github/workflows/frontend.yml`](.github/workflows/frontend.yml) runs `npm ci` + `npx tsc --noEmit` + `npm run build` on every PR so this is caught before merge rather than after.

Test with `pytest` (install `.[test]`), lint with `ruff check` / `ruff format` (config for both in [pyproject.toml](pyproject.toml)). Tests in [tests/](tests/) cover request schemas, pure helpers, and idle/mutex branches; subprocess/thread happy paths and HF Jobs integration are **deliberately** not unit-tested — don't add coverage there. There is no Python build step; for end-to-end validation, run `makermodslab` and exercise endpoints.

Frontend checks (run from `frontend/`): `npm run lint`, `npx tsc --noEmit`, `npm run build`. Some type/lint errors pre-date any given change — record the baseline before you start and compare against it; don't fix pre-existing errors unrelated to your change.

## Architecture

### Backend module layout (`makermodslab/`)

[server.py](makermodslab/server.py) is the FastAPI router (~2600 lines; many request models are defined inline there). Each feature lives in its own module that owns its global state (module-level flags + per-feature locks) and exposes handler functions the router calls.

**Robot flows:**

- [record.py](makermodslab/record.py) — dataset recording. The record loop is reimplemented as `record_with_web_events`, driven by a `web_events` dict (`exit_early` / `stop_recording` / `rerecord_episode`) that frontend buttons toggle — there is no keyboard listener.
- [teleoperate.py](makermodslab/teleoperate.py) — leader→follower teleoperation.
- [calibrate.py](makermodslab/calibrate.py) — step-by-step manual web calibration: `CalibrationManager` singleton with a `_step_complete` threading.Event.
- [auto_calibrate.py](makermodslab/auto_calibrate.py) — automatic calibration: runs the vendored Feetech autocal ([vendor/feetech_autocal/](makermodslab/vendor/feetech_autocal/)) as a subprocess that drives the arm under torque and **writes servo EEPROM**.
- [rollout.py](makermodslab/rollout.py) — inference: runs a trained policy on the follower via a `lerobot-rollout` subprocess; single global session.

**Hardware safety (modules that guard or touch servos):**

- [arm_identity.py](makermodslab/arm_identity.py) — fingerprints each arm via servo EEPROM before energizing; read-only, runs after bus connect, strictly before torque.
- [identify.py](makermodslab/identify.py) — hand-motion port detection (watches raw positions while the user swings the arm); read-only, no torque. [wiggle.py](makermodslab/wiggle.py) is the legacy variant that drives the gripper.
- [motor_power.py](makermodslab/motor_power.py) — per-robot motor-power cap (used as autocal drive torque); `reset_torque_limit` un-throttles `Torque_Limit` so an earlier autocal doesn't silently limit the next session. **Writes servo registers.**
- [rest_pose.py](makermodslab/rest_pose.py) — captures the start pose and gently returns the arm before torque release. Hand-mirrored twin of logic in the vendored autocal script — change one, check the other.
- [torque.py](makermodslab/torque.py) — shared `force_disable_bus_torque` fallback (motor-by-motor release, loud on failure).

**Data & training:**

- [datasets.py](makermodslab/datasets.py) / [models.py](makermodslab/models.py) — local + Hub browsers (fan-out Hub listing with offline resilience; caches behind locks).
- [merge.py](makermodslab/merge.py) — wraps lerobot's `aggregate_datasets` as a subprocess.
- [train.py](makermodslab/train.py) / [jobs.py](makermodslab/jobs.py) — local training subprocess lifecycle; `JobRunner`/`JobRegistry` persist run history to `outputs/train/`.
- [runners/hf_cloud.py](makermodslab/runners/hf_cloud.py) — training on HF Jobs GPUs (replaces the image's bundled lerobot with MakerMods Lab's pin in-container).

**utils/:**

- [utils/config.py](makermodslab/utils/config.py) — shared paths and persistence. **Import shared constants from here, do not hardcode paths in feature modules.**
- [utils/robot_factory.py](makermodslab/utils/robot_factory.py) — the single place `SO101LeaderConfig`/`SO101FollowerConfig`/`BiSO*Config` objects are assembled (`build_single_configs` / `build_bimanual_configs`); rollout.py builds CLI args separately.
- [utils/hf_auth.py](makermodslab/utils/hf_auth.py) (cached `whoami`, offline detection), [utils/devices.py](makermodslab/utils/devices.py) (force-close serial ports/cameras), [utils/errors.py](makermodslab/utils/errors.py) (error-text → plain-language hints), [utils/system.py](makermodslab/utils/system.py) (optional-extra pip installs as subprocess).

### State model & mutual exclusion

Each feature module owns module-level globals (`recording_active`, `teleoperation_active`, `inference_active`, `replay.replay_active`, plus `calibrate.calibration_is_active()`, `auto_calibrate.auto_calibration_is_active()`, and `wiggle.wiggle_active`) protected by per-feature locks. Teleoperation, recording, inference, replay, manual calibration, auto-calibration, and wiggle **are all mutually exclusive, enforced in code** — not by a shared lock, but by reciprocal active-flag checks at each feature's start (e.g. `handle_start_teleoperation` refuses while recording, inference, replay, calibration, auto-calibration, or a wiggle is active). New features that drive the robot must add the same reciprocal checks against every existing one.

### WebSocket broadcast

server.py defines a single `ConnectionManager` with a background `_broadcast_worker` thread that drains a `queue.Queue` and forwards joint data to all `/ws/joint-data` clients via a thread-local asyncio loop. Feature modules get the manager passed in and call `manager.broadcast_joint_data_sync(data)` from their worker threads. Don't `await` from these threads — use the sync queue method.

### Persistent state on disk

All under `~/.cache/huggingface/lerobot/` (managed in [utils/config.py](makermodslab/utils/config.py); writes are atomic):

- `calibration/teleoperators/so_leader/*.json`, `calibration/robots/so_follower/*.json` — named calibrations (leader = "teleop", follower = "robot")
- `robots/*.json` — per-robot records: arm layout (`mode: single|bimanual` with right-arm fields), ports, cameras, calibration names, `motor_power`
- `makermodslab_biso/` — bimanual calibration staging
- `ports/{leader,follower}_port.txt` — last-used serial ports
- `dismissed_hub_jobs.json`, `saved_custom_{datasets,models}.json`, `hidden_{datasets,models}.json` — UI-level bookkeeping

`device_type` in API requests is `"teleop"` or `"robot"` (mapped to leader/follower paths). `robot_type` in port endpoints is `"leader"` or `"follower"`. Don't conflate the two vocabularies.

### Calibration files: dual-location pattern

`setup_calibration_files` ([utils/config.py](makermodslab/utils/config.py)) copies user-selected configs into LeRobot's expected locations under `calibration/`. Recording and teleoperation call it before starting (replay uses `setup_follower_calibration_file`). New features that drive a robot must do the same.

## Frontend layout (`frontend/src/`)

React + Vite + TypeScript with shadcn/radix primitives. Four pages (`Launchpad`, `Teleoperation`, `Training`, `NotFound`); ~100 components grouped by feature area (`calibration/`, `control/`, `dialogs/`, `studio/`, `library/`, `recording/`, `jobs/`, `launchpad/`, … plus shared `ui/`); state via React contexts (`ApiContext`, `StudioContext`, `InferenceSessionContext`, `UrdfContext`, …) and ~19 data/session hooks (`useRobots`, `useDatasets`, `useRealTimeJoints`, …). No Redux/Zustand.

## Hardware target

SO-101 leader/follower arms, single or **bimanual** (two leader/follower pairs via `BiSOLeaderConfig`/`BiSOFollowerConfig`). Robot config construction is centralized in [utils/robot_factory.py](makermodslab/utils/robot_factory.py) — adding a robot type means extending the factory, plus calibrate.py and rollout.py which build their configs/args themselves.
