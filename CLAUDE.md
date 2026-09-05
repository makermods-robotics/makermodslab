# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

MakerMods Lab is a FastAPI + React web UI for policy development, wrapping the [LeRobot](https://github.com/huggingface/lerobot) framework for the SO-101, Maker Arm v1 and Metal Arm leader/follower arms (each single or bimanual). It exposes teleoperation, dataset recording, curation, calibration, training, inference, and replay as HTTP/WebSocket endpoints, replacing LeRobot's CLI + keyboard-driven flows. It is a fork of Hugging Face's [leLab](https://github.com/huggingface/leLab), heavily extended by [makermods-robotics](https://github.com/makermods-robotics).

**Vocabulary:** a trained checkpoint is a **policy** in both the UI and the code. The term "skill" is retired — do not reintroduce it in identifiers, i18n keys, or user-facing copy.

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

`lerobot` is pinned by **SHA** to `makermods-robotics/lerobot@eaab69339` (branch `arm/makermods-combined`; the Star-leader unwrap fix on top of `b968c0c01`, the two-parent merge of `robot/makermods-maker-arm` and `arm/makermods-metal`), on an upstream `v0.6.2` base (see [pyproject.toml](pyproject.toml)). Pin by SHA, never by branch name — a moving pin makes every unrelated failure a bisect, and these fork branches are force-pushable. lerobot is the one dependency the app cannot run without, so treat a bump as a real change: expect import-path drift, adjust call sites, and exercise a rollout end-to-end before landing it. The extras are `core_scripts,feetech,training` (SO-101) plus `maker` (RobStride over CAN), `damiao` (the Metal arm's CAN stack), and `rebot` (motorbridge, the Star Arm 102's UART servo stack). **Never add the fork's `metal` extra** — it drags in Pinocchio (`pin`, no Windows wheels) that only the not-yet-integrated gravity-compensated `metal_leader` needs.

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

**Arm types — the CAN arms (Maker and Metal):**

Every robot record carries an `arm_type`: `so101` (the default, and what every record written before the CAN arms existed reads back as), `maker`, or `metal`. The SO-101 and the CAN arms share no bus protocol, no calibration procedure and no port-detection method, so it is the discriminant every hardware path branches on. The two CAN families share their integration seams (zero-pose calibration, MIT-setpoint rest return, the Star Arm 102 leader — with per-family joint-mapping presets) and differ in the follower's motor protocol: RobStride (Maker) vs Damiao (Metal). **The Damiao difference that matters everywhere: its bus handshake IS the per-motor enable command.** A "read-only" ping energizes a Metal arm; a handshake that raises partway has energized the motors that answered while `is_connected` reads False; a SIGKILLed process leaves the motors holding their last MIT command indefinitely. `torque.de_energize_can_bus` (reopen with `handshake=False`, broadcast the disable) is the recovery, wired into the CAN connect-failure paths and exposed as `POST /api/v1/arms/release-torque` ([can_recovery.py](makermodslab/can_recovery.py)) — deliberately not a session (it must work when session state is wrecked) but refused while `_held_by()` says a live session is driving.

- [arm_capabilities.py](makermodslab/arm_capabilities.py) — **the one place that answers "what can this arm type do?"**. `uses_feetech_bus` gates every helper that reads or writes a servo register by name (`arm_identity`, `motor_power`, `rest_pose`, `identify`/`wiggle` — all Feetech-only, all skipped for the CAN arms); `joints_per_arm` is 6 vs **7**; `supports_auto_calibration` / `uses_zero_calibration` pick the calibration flow; `ships_urdf` is True for SO-101 + Maker (a URDF ships in `frontend/public/`) and gates the teleop 3D-viewer broadcast; `supports_dagger` is False for the CAN arms and it is a HARDWARE limit — the Star Arm 102 leader's joints hold encoders and no motors, so there is nothing to back-drive during a handover. Import this instead of writing `arm_type == "maker"` inline.
- [zero_calibrate.py](makermodslab/zero_calibrate.py) — the CAN arms' ONLY calibration: torque off, the user poses the arm by hand, we set zero, and the ranges come from fixed config constants. No range sweep (their joint limits are measured constants) and no auto-calibration. Deliberately NOT a new session kind — it reuses kind `calibration`, and `calibrate.calibration_is_active()` ORs both managers so every existing reciprocal mutex check covers it with no new `robot.busy.*` discriminant. The request's `arm_type` picks the device configs and follower pose text (Maker folded/open, Metal upright/closed); both families' Star Arm 102 leaders share the same folded/closed physical zero pose. On Metal, the post-connect `disable_torque()` is the write that frees the follower the handshake just energized.
- [maker_rest_pose.py](makermodslab/maker_rest_pose.py) — **the CAN counterpart to rest_pose.py, and mandatory on every stop path (both CAN families — the return is device-level `send_action`, so it drives Maker and Metal alike).** A CAN arm has no brakes, so releasing torque anywhere but near its resting pose drops the whole arm under gravity; torque-off is the vendor's safe state only once it is somewhere safe to rest. A RobStride joint has no `Goal_Position`/`Goal_Velocity` to hand the motion to, so the return is shaped in software by interpolating the MIT setpoint at a bounded rate (`MAKER_RETURN_SPEED_DEG_S`, the vendor's own 30 deg/s) — the same shape lerobot's `RolloutStrategy._return_to_initial_position` uses, which is why inference already landed a Maker arm correctly and teleop/record/replay had to be fixed. Arrival is judged by **convergence, not tolerance**: a joint in MIT position control holds a standing error proportional to its load (measured 3-5 deg on `wrist_flex`, 3.8 deg on `shoulder_lift`), so waiting for zero burns the ceiling on every healthy stop.
- [maker_ports.py](makermodslab/maker_ports.py) — port detection for both CAN families (`arm_type` selects the follower protocol). The follower (CAN) and leader (FashionStar/UART) answer _different protocols_, so `probe_maker_ports` identifies both with no user gesture; `identify_maker_arm_by_motion` is the fallback for a bimanual rig, whose two arms per side are indistinguishable by probe. The Maker probes are strictly read-only. The **Metal follower probe is not**: the Damiao handshake energizes, so it briefly enables the gravity-neutral base joint and explicitly disables it after — and Metal follower motion-identify is refused outright (watching its joints would energize them mid-gesture; identify by the leaders instead).

Calibration libraries are **per arm type** and never merged: `so_leader`/`so_follower` for the SO-101, `maker_follower`/`metal_follower` for the CAN followers — and **one shared `rebot_102_leader` dir for BOTH CAN leaders**, because all the Star-leader presets are config-only variants of one class and lerobot derives the directory from the class's `name`. That sharing is why `default_slot_config_name` mints per-arm-type default ids (`<name>_maker` / `<name>_metal`): the physical zero pose is shared, but the two presets carry different direction/range mappings, so a name collision would silently reuse calibration metadata for the wrong follower. Thread `arm_type` through `calibration_dir_for_device`, `setup_calibration_files`, the bimanual stagers, and `is_robot_record_clean` — a hardcoded SO-101 path makes a fully set-up CAN robot look permanently unready.

Two deliberate gaps: **no Metal URDF ships** (the SO-101 and the Maker arm each ship one under `frontend/public/`), so a Metal teleop session broadcasts `joints_deg` (degrees by motor name) instead of URDF joints and the frontend renders `JointAngleReadout` in the 3D viewer's slot — a Maker session broadcasts both (`joints` in radians drives the model, `joints_deg` still carries the gripper, which the Maker URDF has no joint for); and **episode replay's approach/return mechanism differs by arm type**. `arm_capabilities.ships_urdf` is the predicate; the teleop 3D path (`_MAKER_URDF_JOINTS` / `get_maker_joint_positions_from_robot` in teleoperate.py, `frontend/src/lib/urdfConfigs.ts`) is Maker + SO-101 only. Studio-inference and replay viewers still show the readout for a Maker arm — a follow-up. The playback loop itself is arm-agnostic (plain `send_action` on the dataset's action column, as lerobot's own `lerobot-replay` does it); what differs is getting to frame 0 and back, which is `rest_pose.py` for an SO-101 and `maker_rest_pose.py` for the CAN arms.

**Every flow that energizes an arm must return it before releasing torque**, and all four now do: teleoperation, recording and replay call `return_maker_to_pose` / `return_maker_arms_to_rest` on their graceful-stop path (a second stop press aborts it via `abort_event`, leaving the arm nearer rest than it started), and inference gets the same behaviour for free from lerobot's `--return_to_initial_position=true`. Calibration is the one exception and correctly so — zero-calibration runs with torque off from start to finish, so there is nothing to return. Teleop and record exclude the **gripper** from the captured pose (it may be holding something at stop time); replay includes it, because the dataset drives the gripper and its start width is part of the pose being restored.

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

Two roots, split by who owns the data (all paths managed in [utils/config.py](makermodslab/utils/config.py); writes are atomic). **Import the constants; never spell a path.**

**MakerMods Lab's own state** lives under `MAKERMODSLAB_HOME` = `~/.makermods/makermodslab/` (env var `MAKERMODSLAB_HOME` overrides it — the test suite points it at a tmp dir before importing anything, which also switches the legacy migration off):

- `robots/*.json` — per-robot records: arm layout (`mode: single|bimanual` with right-arm fields), ports, cameras, calibration names, `motor_power`
- `biso_staging/` — bimanual calibration staging (lerobot reads it through an explicit `calibration_dir`, so it need not sit in lerobot's cache)
- `ports/{leader,follower}_port.txt` — last-used serial ports (legacy: read as a fallback, nothing writes them any more)
- `dismissed_hub_jobs.json`, `saved_custom_{datasets,models}.json`, `hidden_{datasets,models}.json`, `excluded_episodes.json` — UI-level bookkeeping
- `instance_id.txt` — this install's stable node identity (32-hex, minted on first read; how peers recognize this machine across restarts and address changes)
- `nodes.json` — saved peer nodes (url + name only; identity is re-verified on load, never trusted from disk)

**lerobot's data** stays under `~/.cache/huggingface/lerobot/` (`HF_LEROBOT_HOME`): datasets, models, `outputs/train/` (local policies and run history), and the calibration libraries — `calibration/teleoperators/so_leader/*.json`, `calibration/robots/so_follower/*.json` (leader = "teleop", follower = "robot") — because lerobot's device classes read them from there.

Versions before the split wrote the first group beside the second. `migrate_legacy_state` runs once at server startup: a file moves only when nothing exists at the new path (the newer file wins), a directory present at both places is merged name by name under the same rule, whatever is left behind is named in one warning, and nothing runs under a `MAKERMODSLAB_HOME` override — so setting the override on an install that already has state starts it fresh, including a newly minted instance id that peers will see as a new node. A new state file goes under `MAKERMODSLAB_HOME`; it does not need a migration row unless it existed before the split. Two log dirs deliberately still sit in the lerobot cache: rollout.py's `inference_logs/` and merge.py's sibling — logs, not state; move them with a constant when there is a reason to.

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

Three arm families, each single or **bimanual**:

- **SO-101** leader/follower — Feetech STS3215 servos on USB serial, 6 joints per arm (`SO101LeaderConfig`/`SO101FollowerConfig`, `BiSO*` when bimanual).
- **Maker Arm v1** — a **7-DOF** RobStride follower on classic CAN (via an slcan adapter; `port` is the adapter's serial device, e.g. `/dev/cu.usbmodem*` on macOS) driven by a **Star Arm 102** (reBot 102) leader on FashionStar UART servos (`MakerFollowerConfig`/`RebotArm102LeaderMakerTeleopConfig`, `BiMakerFollowerConfig`/`BiRebot102LeaderMakerConfig` when bimanual).
- **Metal arm** — a **7-DOF** Damiao follower on classic CAN, driven by the same Star Arm 102 leader with the Metal preset (`MetalFollowerConfig`/`RebotArm102LeaderMetalTeleopConfig`; bimanual is `BiMetalFollowerConfig` plus the GENERIC `BiRebot102LeaderConfig` carrying `RebotArm102LeaderMetalConfig` sub-configs — the fork registers no `bi_rebot_102_leader_metal`). Vendor-default `disconnect()` leaves Metal torque ON; MakerMods Lab's stop paths override that with return-to-rest + explicit release, keeping "stopped means de-energized" true for every family.

The Star leader preset MUST match the follower family — never the bare `rebot_102_leader`: each preset carries the joint directions and ranges of ITS follower, and a mismatched one runs joints the wrong way or saturates them against the follower's soft limits while teleop keeps reporting a healthy loop. That pairing is decided in the robot factory from `arm_type` alone; no caller picks a teleop type string.

Robot config construction is centralized in [utils/robot_factory.py](makermodslab/utils/robot_factory.py), which branches on `arm_type` — adding a robot type means extending the factory, plus calibrate.py/zero_calibrate.py and rollout.py which build their configs/args themselves.

The 7-vs-6 joint count is load-bearing: `rollout._arm_count_mismatch` reads the per-arm width from the arm type, because a 7-dim Maker checkpoint measured against the SO-101's 6 is neither `<= 6` nor a multiple of it and would silently disable the very guard it needs to trip.
