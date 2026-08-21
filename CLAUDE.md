# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

When `frontend/**` (excluding `frontend/dist/**`) changes on `main` or `staging`, [`.github/workflows/build_frontend.yml`](.github/workflows/build_frontend.yml) auto-rebuilds `frontend/dist/` and commits it back — don't rebuild it by hand for a PR. `makermodslab --dev` serves from Vite, no rebuild needed.

**`frontend/package-lock.json` is the most fragile file in the repo** — it has broken CI three times (`e345e61`, `944cb05`, `c18d42c`). npm records platform-specific optional dependencies (esbuild, rollup, the `@emnapi/*` wasm shims) in it, so a lockfile written by `npm install` on macOS can fail `npm ci` on the Linux runner. Never hand-edit it; regenerate it on Linux or in a container when it genuinely needs regenerating; and treat an unexplained `package-lock.json` diff in an otherwise-unrelated PR as stray local churn to drop, not a change to keep. [`.github/workflows/frontend.yml`](.github/workflows/frontend.yml) runs `npm ci` + both typecheck projects + `npm run build` on every PR so this is caught before merge rather than after.

Test with `pytest` (install `.[test]`), lint with `ruff check` / `ruff format` (config for both in [pyproject.toml](pyproject.toml)). Tests in [tests/](tests/) cover request schemas, pure helpers, and idle/mutex branches; subprocess/thread happy paths and HF Jobs integration are **deliberately** not unit-tested — don't add coverage there. There is no Python build step; for end-to-end validation, run `makermodslab` and exercise endpoints.

Frontend checks (run from `frontend/`): `npm run lint`, `npx tsc --noEmit -p tsconfig.app.json`, `npx tsc --noEmit -p tsconfig.node.json`, `npm run build`. **The `-p` is not optional, and there are two projects** — the root `tsconfig.json` is a solution file (`"files": []` plus project references), so a bare `npx tsc --noEmit` checks nothing and always exits 0; `tsconfig.app.json` covers `src`, `tsconfig.node.json` covers the build configs (`vite.config.ts`, `vitest.config.ts`, `tailwind.config.ts`). Some type/lint errors pre-date any given change — record the baseline before you start and compare against it; don't fix pre-existing errors unrelated to your change.

### Branches and CI

`main` is the release branch; `staging` is a permanent integration branch in front of it — feature branches PR into `staging`, and `staging` is promoted to `main` by PR. Both run the same workflows, so what passes on `staging` is what `main` will do.

**Never commit `frontend/dist` by hand** — it is rebuilt and committed automatically on a push to either branch, and a hand-built bundle turns a clean merge into a binary conflict. [`sync_staging.yml`](.github/workflows/sync_staging.yml) merges `main` into `staging` on every push to `main` and resolves that for you, so the promotion PR stays clean.

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
