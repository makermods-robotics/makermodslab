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

Test with `pytest` (install `.[test]`), lint with `ruff check` / `ruff format` (config for both in [pyproject.toml](pyproject.toml)). Tests in [tests/](tests/) cover request schemas, pure helpers, and idle/mutex branches; subprocess/thread happy paths and HF Jobs integration are **deliberately** not unit-tested — don't add coverage there. Contract/ratchet tests (see "API contract & versioning") are equality-asserted on purpose; state machines (the lease, the node registry) are tested with injected clocks and `httpx.MockTransport` — never sleeps, never new dependencies. There is no Python build step; for end-to-end validation, run `makermodslab` and exercise endpoints.

Frontend checks (run from `frontend/`): `npm run lint`, `npx tsc --noEmit -p tsconfig.app.json`, `npx tsc --noEmit -p tsconfig.node.json`, `npm run build`. **The `-p` is not optional, and there are two projects** — the root `tsconfig.json` is a solution file (`"files": []` plus project references), so a bare `npx tsc --noEmit` checks nothing and always exits 0; `tsconfig.app.json` covers `src`, `tsconfig.node.json` covers the build configs (`vite.config.ts`, `vitest.config.ts`, `tailwind.config.ts`). Some type/lint errors pre-date any given change — record the baseline before you start and compare against it; don't fix pre-existing errors unrelated to your change.

### Branches and CI

`main` is the release branch; `staging` is a permanent integration branch in front of it — feature branches PR into `staging`, and `staging` is promoted to `main` by PR. Both run the same workflows, so what passes on `staging` is what `main` will do.

**Never commit `frontend/dist` by hand** — it is rebuilt and committed automatically on a push to either branch, and a hand-built bundle turns a clean merge into a binary conflict. [`sync_staging.yml`](.github/workflows/sync_staging.yml) merges `main` into `staging` on every push to `main` and resolves that for you, so the promotion PR stays clean.

## Architecture

### Backend module layout (`makermodslab/`)

[server.py](makermodslab/server.py) is the FastAPI router (~3400 lines; many request models are defined inline there). Each feature lives in its own module that owns its global state (module-level flags + per-feature locks) and exposes handler functions the router calls. Routes register on one of two routers — see "API contract & versioning" below before adding any.

**Robot flows:**

- [record.py](makermodslab/record.py) — dataset recording. The record loop is reimplemented as `record_with_web_events`, driven by a `web_events` dict (`exit_early` / `stop_recording` / `rerecord_episode`) that frontend buttons toggle — there is no keyboard listener.
- [teleoperate.py](makermodslab/teleoperate.py) — leader→follower teleoperation.
- [calibrate.py](makermodslab/calibrate.py) — step-by-step manual web calibration: `CalibrationManager` singleton with a `_step_complete` threading.Event.
- [auto_calibrate.py](makermodslab/auto_calibrate.py) — automatic calibration: runs the vendored Feetech autocal ([vendor/feetech_autocal/](makermodslab/vendor/feetech_autocal/)) as a subprocess that drives the arm under torque and **writes servo EEPROM**.
- [rollout.py](makermodslab/rollout.py) — inference: runs a trained policy on the follower via a `lerobot-rollout` subprocess; single global session.
- [sessions.py](makermodslab/sessions.py) — the `/api/v1/sessions` surface, THE front door for starting robot flows (legacy start endpoints remain for external clients only). `SessionTracker` gives the one live session an identity by observing the seam below; the start wrappers resolve a robot-record NAME into the features' request models server-side; the **lease** (owner + heartbeat + expiry watchdog) safety-stops abandoned sessions — stopping is deliberately never owner-gated.
- [session_events.py](makermodslab/session_events.py) — the transition seam: every feature emits claim/phase/release through it; the WS broadcast and the SessionTracker both subscribe. Emissions swallow subscriber exceptions — a broadcast hiccup must never break a hardware flow.

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

**Multi-node:**

- [nodes.py](makermodslab/nodes.py) — peer-node registry (static/manual source for now): verify-on-add against a peer's `/api/v1/health` identity document, TTL liveness with an injected clock, `nodes.json` persistence. Discovered peers are hints — always re-verified, never trusted.
- [runners/lan_node.py](makermodslab/runners/lan_node.py) — `runner: "lan_node"`: offloads a training job to another node by driving that peer's own v1 jobs API; tolerates network blips (120s peer-lost grace), relays the peer's terminal verdict so a remote stop never classifies as failure. Datasets travel via the Hub ([runners/\_dataset.py](makermodslab/runners/_dataset.py)) — a LAN peer can no more see this machine's LeRobot cache than an HF pod can.

**utils/:**

- [utils/config.py](makermodslab/utils/config.py) — shared paths and persistence. **Import shared constants from here, do not hardcode paths in feature modules.**
- [utils/robot_factory.py](makermodslab/utils/robot_factory.py) — the single place `SO101LeaderConfig`/`SO101FollowerConfig`/`BiSO*Config` objects are assembled (`build_single_configs` / `build_bimanual_configs`); rollout.py builds CLI args separately.
- [utils/hf_auth.py](makermodslab/utils/hf_auth.py) (cached `whoami`, offline detection), [utils/devices.py](makermodslab/utils/devices.py) (force-close serial ports/cameras), [utils/errors.py](makermodslab/utils/errors.py) (error-text → plain-language hints), [utils/system.py](makermodslab/utils/system.py) (optional-extra pip installs as subprocess).

### API contract & versioning

Every route lives on one of two routers in server.py: `router` is mounted **twice** (flat, and under `/api/v1`) — the flat mount exists only for external/legacy clients and only ever shrinks; `v1_router` is mounted under `/api/v1` alone, and **all new surface goes there**. v1 operation ids are the bare handler function names (they become SDK method names, so handlers must be uniquely named); route `tags` are SDK namespaces (`system`/`datasets`/`models`/`jobs`/`sessions`/`nodes`).

[docs/api/openapi.json](docs/api/openapi.json) is the committed OpenAPI snapshot — the reviewable record of every surface change. Regenerate after any backend change with `uv run python -m makermodslab.scripts.export_openapi` (a pre-commit hook does it locally; CI's Quality job deliberately SKIPs that hook — the lint runner has no uv — and the Tests workflow's `test_openapi_snapshot_is_fresh` is the real enforcement).

[tests/test_api_contract.py](tests/test_api_contract.py) holds **equality-asserted ratchets**: `LEGACY_ROUTES` (the flat surface; shrink-only), `UNTYPED_V1_ROUTES` (v1 operations without a `response_model`; shrink-only — new routes ship typed), `RESPONSE_MODEL_EXEMPT` (file/stream/no-body routes, each with a reason), and `V1_ONLY_ROUTES` (the register of v1-only surface; grows). A failing ratchet means fix the route, not widen the list — the lists exist so migration progress is in the diff and cannot silently regress.

Error codes ([api_errors.py](makermodslab/api_errors.py)) follow `<domain>.<condition>[.<detail>]` — dots separate levels, underscores separate words; the domain set is closed and grammar-tested in [tests/test_api_errors.py](tests/test_api_errors.py), so extending the taxonomy starts there. Raise `ApiError` so bodies carry `code` beside the legacy string `detail`; the `robot.busy.<feature>` discriminants are equality-tested against the mutual-exclusion matrix. Schema-level 422s are coded APP-WIDE: a global `RequestValidationError` handler stamps `request.validation` beside FastAPI's untouched pydantic error list — necessary because FastAPI rejects the request before any endpoint code runs, so no per-route wiring can ever deliver that code.

Response models live in [schemas/](makermodslab/schemas/), one module per tag, and must describe the handlers' existing dicts **exactly**: `response_model` silently FILTERS undeclared fields and MATERIALIZES declared-but-absent optionals as `null`. Use `response_model_exclude_none` only where keys are absent-or-set (never legitimately null), `response_model_exclude_unset` where absent keys and legitimate nulls coexist; a route that can't be modeled faithfully stays in the ratchet with a `# why` comment rather than getting a lying model.

### State model & mutual exclusion

Each feature module owns module-level globals (`recording_active`, `teleoperation_active`, `inference_active`, `replay.replay_active`, plus `calibrate.calibration_is_active()`, `auto_calibrate.auto_calibration_is_active()`, and `wiggle.wiggle_active`) protected by per-feature locks. Teleoperation, recording, inference, replay, manual calibration, auto-calibration, and wiggle **are all mutually exclusive, enforced in code** — not by a shared lock, but by reciprocal active-flag checks at each feature's start (e.g. `handle_start_teleoperation` refuses while recording, inference, replay, calibration, auto-calibration, or a wiggle is active). Two layers sit on top of the flags without replacing them: the session_events seam (every real transition must emit — claim after the flag is set under its lock, phases, release after cleanup including error paths) and sessions.py's tracker + lease. A new robot-driving feature must therefore: add reciprocal checks against every existing feature (refusing with its `robot.busy.*` code), emit session_events at its transitions, register its discriminant in the mutex/taxonomy tests, and — if user-startable — join `STARTABLE_KINDS` with an options schema and a `_dispatch_stop` arm.

### WebSocket broadcast

server.py defines a single `ConnectionManager` with a background `_broadcast_worker` thread that drains a `queue.Queue` and forwards joint data to all `/ws/joint-data` clients via a thread-local asyncio loop. Feature modules get the manager passed in and call `manager.broadcast_joint_data_sync(data)` from their worker threads. Don't `await` from these threads — use the sync queue method. The same socket carries typed control events — `jobs_changed`/`job_progress` (JobRegistry) and `session_changed` (session_events) — as **droppable refetch hints**: clients refetch on a hint and never treat the payload as state (a missed broadcast self-heals on the next fetch). The frontend demuxes on `data.type` (`useJobsChangedSignal`, `useActiveSession`).

### Persistent state on disk

All under `~/.cache/huggingface/lerobot/` (managed in [utils/config.py](makermodslab/utils/config.py); writes are atomic):

- `calibration/teleoperators/so_leader/*.json`, `calibration/robots/so_follower/*.json` — named calibrations (leader = "teleop", follower = "robot")
- `robots/*.json` — per-robot records: arm layout (`mode: single|bimanual` with right-arm fields), ports, cameras, calibration names, `motor_power`
- `makermodslab_biso/` — bimanual calibration staging
- `ports/{leader,follower}_port.txt` — last-used serial ports
- `dismissed_hub_jobs.json`, `saved_custom_{datasets,models}.json`, `hidden_{datasets,models}.json` — UI-level bookkeeping
- `instance_id.txt` — this install's stable node identity (32-hex, minted on first read; how peers recognize this machine across restarts and address changes)
- `nodes.json` — saved peer nodes (url + name only; identity is re-verified on load, never trusted from disk)

`device_type` in API requests is `"teleop"` or `"robot"` (mapped to leader/follower paths). `robot_type` in port endpoints is `"leader"` or `"follower"`. Don't conflate the two vocabularies.

### Calibration files: dual-location pattern

`setup_calibration_files` ([utils/config.py](makermodslab/utils/config.py)) copies user-selected configs into LeRobot's expected locations under `calibration/`. Recording and teleoperation call it before starting (replay uses `setup_follower_calibration_file`). New features that drive a robot must do the same.

## Frontend layout (`frontend/src/`)

React + Vite + TypeScript with shadcn/radix primitives. Four pages (`Launchpad`, `Teleoperation`, `Training`, `NotFound`); ~100 components grouped by feature area (`calibration/`, `control/`, `dialogs/`, `studio/`, `library/`, `recording/`, `jobs/`, `launchpad/`, … plus shared `ui/`); state via React contexts (`ApiContext`, `StudioContext`, `InferenceSessionContext`, `UrdfContext`, …) and ~19 data/session hooks (`useRobots`, `useDatasets`, `useRealTimeJoints`, …). No Redux/Zustand.

Every frontend request targets `/api/v1/...` — the flat mount is for external clients only; don't reintroduce flat URLs (including `<img>`/`<video>` srcs and the WebSocket). Session flows start through [lib/sessionApi.ts](frontend/src/lib/sessionApi.ts) with a robot **name** + kind options (the server resolves ports/configs/cameras from the record), hold a lease heartbeated by `useSessionHeartbeat` (~20s), and show a courtesy `beforeunload` confirm via `useUnloadWarning`. There are deliberately **no client-side safety guards**: `SingleTabGuard`, `TeleopStopNotice`, and `useSessionExitGuard` were retired in favor of the server-side lease + `session.held` — do not resurrect browser stop-beacons or tab elections; an abandoned session is the expiry watchdog's job.

### Localization

The UI ships English and Simplified Chinese via `react-i18next`; catalogs live in [`frontend/src/i18n/locales/`](frontend/src/i18n/locales/), one namespace file per feature area. **Read [frontend/docs/localization.md](frontend/docs/localization.md) before touching user-facing strings** — it is written to be the only thing you need.

The governing rule is that localization is **cosmetic only**: no request/response, storage, form-value or on-disk change, and the Python backend is never localized (server prose renders in English in every language). A great many strings here are _data_ wearing a label — camera-name presets, codec ids, calibration file names, `formatDurationShort` output — and translating one corrupts a payload or a file on disk. Three tests enforce the invariants: catalog key parity, dynamic-key resolution, and frozen-English output for the helpers that had to be restructured.

## Hardware target

SO-101 leader/follower arms, single or **bimanual** (two leader/follower pairs via `BiSOLeaderConfig`/`BiSOFollowerConfig`). Robot config construction is centralized in [utils/robot_factory.py](makermodslab/utils/robot_factory.py) — adding a robot type means extending the factory, plus calibrate.py and rollout.py which build their configs/args themselves.
