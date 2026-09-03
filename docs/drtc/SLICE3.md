# Slice 3 design record

## Decisions (2026-09-02)

Answers the user gave to the open questions below; these are settled, and the
design text after them is the analysis they were decided from (kept verbatim,
including the passages these decisions supersede).

1. **Naming — `remote_inference`.** Session kind `remote_inference`, busy code
   `robot.busy.remote_inference`, feature module `makermodslab/remote_inference.py`.
2. **New error domain `transport.*` — accepted.** It lands in
   `tests/test_api_errors.py`'s closed `DOMAINS` set first, per that module's
   own rule.
3. **Remote inference is REFUSED while a local training run holds the machine.**
   Symmetric and cheap; the "train locally while a Modal GPU drives the arm"
   capability is deliberately deferred and worth revisiting.
4. **Lifecycle option A in slice 3** (the session owns the robot side and
   verifies SFU + policy). **B next.** **C never as a session** — read-only
   transport status now, an SFU supervisor only if it earns itself.
5. **PR #111 merged to LOCAL staging only** (not to `origin`), so slice 3 builds
   on the absorbed preview.
6. **S3.1 scope includes the first-action ease-in AND `--livekit-room` on both
   Modal wrappers.**
7. **Adaptive-sync only.** `robot_rtc` / `policy_rtc` stay out of scope.

### Contract notes from the S3.1 implementation

These are the places the implementation had to depart from, or sharpen, the
design below. S3.2/S3.3 consume them.

- **`makermodslab/drtc_protocol.py` is the whole contract** and is
  self-contained (stdlib only). `STATS_KEYS` is the frozen key set; every key
  is always present, `null` where unknown, so an exact `response_model` is
  achievable. `format_stats` RAISES on an unknown key so a typo cannot ship.
- **`EASING` was added to the event vocabulary** (the design listed the ease-in
  as a risk, not as an event). It is emitted immediately before the first-action
  ramp begins; the natural phase name for the parent is `easing_in`.
- **The arbitrary-target ease-in primitive already exists.** The design assumed
  one was missing; in fact `rest_pose.return_to_rest_pose(bus, target,
normalize=True, tolerance=…, stall_min_progress=…)` is exactly it — it is what
  `replay` uses for its frame-0 approach — so `makermodslab/drtc/_pose.py` wraps
  it rather than reimplementing a ramp. No new rate/shape logic was written.
- **Ease-in is single-Feetech-bus only; return-to-rest is per bus.** A bimanual
  BiSO robot's action keys are `left_`/`right_` prefixed while each sub-arm's
  `bus.motors` are bare, so the action→bus mapping would silently match nothing;
  the ease-in refuses with `unsupported (2 buses)` rather than guess. The return
  works in raw ticks per bus and has no such problem. Koch/OMX (Dynamixel) get
  neither — the Feetech unit constants mean something else there. S3.2's
  preflight should therefore also refuse a **bimanual** robot for now, not just
  a CAN one.
- **`lerobot-rollout` does NOT ramp on entry.** Checked against the pinned fork:
  `ActionInterpolator.add` explicitly runs the first step raw ("First step: no
  previous action yet"), and `max_relative_target` defaults to `None`. So the
  pre-fix jump was PARITY with the local sibling, not a regression — but
  lerobot does ramp on EXIT (`RolloutStrategy._return_to_initial_position`, a
  3 s interpolation), which is the same shape in the other direction. The
  ease-in is therefore an improvement over both siblings, and `rollout.py`'s
  local inference still has the entry jump.
- **`_env.read_env()` returns the full resolved environment**, not just the
  four LiveKit keys — "what `load_env` would have produced in `os.environ`" is
  the only definition that cannot drift from `load_env`, which is now
  implemented on top of it.
- **`--ease_in` was added beside `--return_to_rest`**, both defaulting true, so
  the bench can A/B the ramp the way `--adaptive` / `--align` already allow.
  Neither belongs on the API surface.
- **draccus has no `--no-<flag>` form** — verified against the installed draccus,
  not assumed. `--no-adaptive` / `--no-align`, which the ported docstrings have
  claimed since the port, have never worked; the A/B form is `--<flag> false`.
  Docstrings in `robot_sync.py`, `_sync_player.py` and `docs/drtc/README.md`
  were corrected. S3.2's arg builder must not emit `--no-*`.

---

# Slice 3 — Remote inference (DRTC) as a first-class session

**Read-only run.** No files were written or edited, no git mutations (only `git log`/`git diff --stat`/`git status`), no installs, no servers, no network. Everything read came from `/Users/mokuroh54/Documents/MakerMods/MakerModsLab/.worktrees/drtc` at `227100b1`, with one exception disclosed for honesty: I ran read-only `ls`/`grep`/`sed` against `/Users/mokuroh54/Documents/MakerMods/MakerModsLab/.venv/lib/python3.*/site-packages/livekit/api/` to confirm the installed `livekit-api` 1.2.1 surface (`RoomService.list_participants`, `ws://`→`http://` normalization in `twirp_client.py`). Nothing was modified.

Baseline confirmed: `git diff --stat e8454e5b..227100b1` is 29 files, purely additive — `makermodslab/drtc/**`, `tools/drtc/**`, `docs/drtc/**`, `tests/test_drtc_{env,schedule}.py`, +22 lines in `utils/config.py` (the four `DRTC_*` paths), the `[drtc]` extra, `uv.lock`, `.gitignore`. `server.py`, `sessions.py`, `session_events.py`, `api_errors.py`, `schemas/**` are untouched. The preview claim holds exactly.

---

## 1. Session shape

### Options

**A — a new kind `remote_inference`,** with its own feature module `makermodslab/remote_inference.py` owning `remote_inference_active`, its own `_state_lock`, `handle_start_remote_inference` / `handle_stop_remote_inference` / `handle_remote_inference_status` / `remote_inference_is_active()`.

**B — a variant of `inference`,** selected by a `remote: bool` option on `InferenceOptions`, dispatched inside `rollout.py` to a fourth launcher beside `_launch_rollout_subprocess` / `_launch_eval_runner` / `_launch_dagger_runner`.

**REC: A.** The precedent that argues for B is coaching — a genuinely different runner (`dagger_runner`), different phases, different stop verb, all living under kind `inference` behind `coaching: bool`. But coaching earns that by _sharing the ladder_: `_run_inference_startup`'s download → `_prepare_robot` → spawn, one `_inference_meta`, one `_state_lock`, one `_last_result`. Remote inference shares only the middle step. There is no `_resolve_policy_path` (the checkpoint is loaded by `from_pretrained` inside the Modal container), no `--policy.device` / `_detect_device`, no `lerobot-rollout` argv, no `_classify_outcome` over lerobot's log vocabulary. What it _does_ share — `_prepare_robot`, `_session_cameras`, `_format_cameras_arg`, `_arm_count_mismatch`, `_terminate_tree` — is already factored as importable functions, so A reuses them without inheriting `rollout.py`'s 4413 lines of state. Three structural arguments seal it:

- **`_dispatch_stop` dispatches on kind alone.** Under B it would have to route `"inference"` to one of two entirely different stop state machines by peeking at a feature flag — and `check_expiry`'s watchdog goes through that same dispatcher, so the peek would sit on the safety path.
- **`_FOLLOWER_ONLY_KINDS`.** Inference is follower-only _except when coaching_, an exception `handle_start_session` already carries a paragraph of comment about ("lost when the branch was restacked, and losing it is not cosmetic"). Remote inference is follower-only unconditionally — a clean membership rather than a second conditional on an already-conditional line.
- **`robot.busy.<discriminant>` exists so a client can tell _what_ holds the arm.** Under B, a user whose Deploy panel is refused gets "inference" and a Stop button wired to `/inference-status` that will report idle. The frontend's `HOLDER_ACTIVITY_KEYS` map in `sessionApi.ts` is keyed by exactly this string.

Cost of A, exhaustively (this is the real "what a session kind costs here" list):

| Contract                                                                                                                        | Change                                                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `session_events.SESSION_KINDS`                                                                                                  | `+ "remote_inference"` (frozenset; `notify_session_changed` drops unknown kinds)                                                                                                                           |
| `api_errors.ErrorCode`                                                                                                          | `+ ROBOT_BUSY_REMOTE_INFERENCE = "robot.busy.remote_inference"` (grammar-legal: 3 levels, underscore within a level)                                                                                       |
| `sessions._held_by`                                                                                                             | one more branch, ordered before/after inference — order is arbitrary but must match the tests' expectations                                                                                                |
| `sessions.STARTABLE_KINDS`, `_FOLLOWER_ONLY_KINDS`, `_OPTIONS_MODELS`, `_REQUEST_BUILDERS`, `_dispatch_start`, `_dispatch_stop` | one entry each                                                                                                                                                                                             |
| `schemas/sessions.py`                                                                                                           | `RemoteInferenceOptions`, `SessionStartBody.kind` Literal, `__all__`                                                                                                                                       |
| Reciprocal checks                                                                                                               | `teleoperate`, `record`, `rollout`, `replay`, `calibrate`, `auto_calibrate`, `wiggle` each gain one guard returning the new code — **and the new module gains all seven plus `jobs.training_is_active()`** |
| `jobs.JobRegistry._robot_busy`                                                                                                  | must consult the new flag, or a queued training run starts under a live remote session                                                                                                                     |
| Tests (equality-asserted)                                                                                                       | `test_api_errors.BUSY_DISCRIMINANTS`, `test_session_events.test_session_kinds_match_the_mutex_features`                                                                                                    |
| Frontend                                                                                                                        | `SessionKind` union, `StartableSessionKind`, `HOLDER_ACTIVITY_KEYS`, en+zh catalogs                                                                                                                        |

### Options schema — `RemoteInferenceOptions` (`extra="forbid"`)

Exposed:

```
policy_ref: str                                # Lab-side ref (/jobs/{id}/checkpoints) — local metadata only
policy_hub_id: str = ""                        # "<owner>/<repo>" the GPU loads; advisory in slice 3, --policy-path in B
task: str = ""
camera_bindings: dict[str, str] = {}
camera_dims: dict[str, PolicyCameraDims] = {}
checkpoint_state_dim: int | None = None
duration_s: int = 60                           # -> robot_sync --duration_s (0 = unbounded)
horizon: int = 16                              # MUST match the GPU side
fps: int = 30                                  # MUST match the GPU side
video_codec: Literal["H264", "MJPEG"] = "H264" # MUST match the GPU side
skip_identity_check: bool = False
```

Two fields need a word. `policy_ref` and `policy_hub_id` are deliberately separate because they are different vocabularies: `policy_ref` is the opaque Lab ref that `_policy_ref_is_valid` / `_resolve_policy_path` understand and that the launch UI already has (it is what yields `checkpoint_state_dim`, `image_features` → `camera_dims`, and the task prefill); `--policy-path` on Modal is an HF repo id resolved in the container. Collapsing them would either force the Lab to download a checkpoint it never runs, or leave the arm-count guard with nothing to check.

**Fixed, not exposed** (constants in the arg builder, each with a `# why`): `adaptive=True`, `base_lead=2`, `s_min=4`, `align=True`, `action_delay=1`, `latency_alpha/beta/k`, `video_quality`, `video_bitrate_kbps`, `reliable_state` (it auto-follows the codec, and forcing it wrong head-of-line-blocks state behind H264 retransmits). These are knobs whose wrong values present as _"the arm freezes"_ or _"the arm snaps at every boundary"_ rather than as an error, and tonight's run validated the defaults end to end. `--no-adaptive` is a bench A/B flag and stays off the API entirely. **The one I'd revisit is `s_min`** — see Risk 3.

### Reciprocal mutex and the discriminant

`robot.busy.remote_inference`. The new module's own guard list must be all eight (seven features + local training), and the seven peers each gain one check. `tests/test_api_errors.py::test_busy_discriminants_cover_mutex_matrix` is an equality assertion over `BUSY_DISCRIMINANTS`, so this cannot be half-done.

### Phases

Wire strings on the meta, broadcast through `_set_phase` → `notify_session_changed`. Phase values are opaque to the tracker and the UI (`SessionInfo.phase: str | None`), so no central registry moves:

`resolving` → `transport_check` → `preflight` → `starting` → `connecting` → `warming_up` → `running` → `stopping` → `stopped` / `error`

`warming_up` is the phase where `active` is still `None` or `chunks == 0`; `running` is entered on the first correlated chunk. Return-to-rest runs **inside** `stopping` rather than under a new name — see §3 for why that matters to `_WINDING_DOWN_PHASES`.

---

## 2. Lifecycle ownership — the central decision

**REC: ship A in slice 3; B next; C never as a session-owned thing (read-only transport status now, an SFU supervisor only if it earns itself).**

### A — the session owns the robot side and verifies the other two

Verification is one call, not three. `livekit.api.LiveKitAPI(url, key, secret).room.list_participants(ListParticipantsRequest(room=ROOM))` answers everything the preflight needs at once:

- **SFU reachable** — connection error means the signaling endpoint is down. The twirp client normalizes `ws://` → `http://` itself (`twirp_client.py`: `scheme.replace("ws", "http")`), so `ws://127.0.0.1:7880` and `wss://x.livekit.cloud` both work unchanged. This is strictly better than hand-rolling `local_sfu_ts.sh`'s `curl -sf http://127.0.0.1:7880`, which only covers the local case.
- **Credentials valid** — an auth failure is distinguishable from a connection failure.
- **A policy is present** — a participant with identity `policy` (`policy.py`'s `IDENTITY`) / the `lk.portal.role` attribute Portal self-sets on connect. Zero operators, or a room that does not exist, is the empty-room case _caught before the arm is energized_.

Timeout: pass a short `aiohttp.ClientTimeout` (the default is 10 s; 3 s is right for a start-button preflight).

Cost: no `modal` dependency in the Lab, no CLI discovery, no cold start inside the session's critical path, no new failure mode where the arm is claimed while a GPU boots. The user keeps the two terminals they already run, and the session's promise is exactly the one it can keep: _"I own the arm; I verified the other two halves before I energized it."_ De-energize invariant: fully preserved — the Lab owns the only process touching the bus, and an orphaned GPU/SFU is a wasted Modal minute, not a hazard.

### B — the Lab also launches Modal

Mechanism: `modal run <abs>/makermodslab/drtc/modal_policy.py --policy-path … --horizon … --fps … --video-codec … [--tailscale --livekit-url … --livekit-api-key … --livekit-api-secret …]` as a second subprocess.

Three things to know before anyone schedules this:

1. **hf_cloud's secret model does not transfer.** `hf_cloud.py` builds `secrets = {"HF_TOKEN": token}` and passes the dict to `api.run_job(...)`. Modal secrets are _pre-created named objects_ baked into the app spec (`modal.Secret.from_name("LiveKit-cloud")`), and `modal_policy.py` explains at length why the two secret lists must be unconditional and identical locally and in-container. The Lab can inject nothing. The only per-run channel is the CLI flags.
2. **`LIVEKIT_ROOM` has no flag.** On the GPU side the room comes _only_ from the `LiveKit-cloud` secret. So B does **not** close the room-mismatch class that people assume it closes. Closing it is a genuinely small change — add `--livekit-room` to both wrappers' `_serve_impl`/`main` and thread it into `os.environ` before `policy.main()` — and it should be a prerequisite of B, not an afterthought.
3. **The `modal` CLI lives outside the venv.** Mirror `utils/system.py::_find_uv()`, which exists because of exactly this lesson ("venvs created with `uv venv` don't ship pip"): probe `shutil.which("modal")`, then `~/.local/bin/modal`, then `uv tool run modal`; if none, refuse with a coded error naming the install command. Never assume `sys.executable -m modal`.

Cold start (A100 + the image + `from_pretrained`) is realistically 60–180 s. That forces the ordering: **launch the GPU in a pre-claim phase, poll the room for the operator, and claim/connect the arm only once it is there.** This is the same discipline `_run_inference_startup` already encodes (download → preflight → spawn, so a stop during the download never opens the bus). Stop order is the mirror: stop `robot_sync` first (return-to-rest → torque off), _then_ terminate the `modal` process — never make the arm's release wait on a cloud API. And do **not** wire `/reset` into the session: it re-spawns the _last_ run globally from a `modal.Dict` with no session identity, so a Lab-driven reset could resurrect someone else's run. Keep `/reset` a documented human escape hatch.

### C — the Lab also owns the SFU

Would require install detection (`livekit-server`, `tailscale` with the App Store path fallback, `cloudflared`), config generation with a rotatable key/secret, and inheriting the documented top footgun: `livekit.local.env` _outlives the script_, so after a Ctrl-C the robot keeps dialing `ws://127.0.0.1:7880` and gets connection refused. Crucially it cannot be tied to a session — the SFU must be up _before_ the session starts (Modal dials it) and must survive between sessions. So it is a `system.*`-shaped supervised child, not a session kind. That is a lot of surface for something a 146-line shell script does well.

**REC for slice 3:** ship the read-only 10% that carries 90% of the operator value — `GET /api/v1/remote-inference/transport` reporting the _effective_ url/room, whether the source is cloud or the local override, whether `livekit.local.yaml` / `livekit.local.env` exist, whether the endpoint answers, and whether an operator is in the room. Add exactly one mutation: **"clear local SFU override"** (unlink `config.DRTC_LOCAL_ENV_PATH`), because that file outliving its script is the documented footgun and the file is one the Lab's own config module already names.

---

## 3. Stop semantics

**REC: a stdin line protocol, with return-to-rest running _inside_ the subprocess.**

**Why not a signal.** The subprocess owns the bus. The parent has no handle on the connected `Robot`, and opening the follower port from the parent while the child holds it is precisely the port-contention hazard `_inference_startup_thread`'s `is_alive()` guard exists to prevent. Worse, a SIGTERM would run `robot_sync.run()`'s existing `finally:` — `portal.disconnect()` → `portal.close()` → `robot.disconnect()` — which drops the arm from wherever the policy left it. Survivable on an SO-101; a fall on a CAN arm. This is the gap `docs/drtc/README.md` names in its own "Not yet done" list.

**New module `makermodslab/drtc_protocol.py`,** sibling of `eval_protocol.py` and for the identical reason stated in that file's docstring: both ends need it and they sit on opposite sides of a dependency wall — the parent must never import `livekit.portal` (an FFI dylib), the child is exactly the process that does.

- Commands (parent → child stdin, one bare word per line): `STOP` (leave the loop, return to the captured start pose, disconnect, exit 0), a second `STOP` (set the abort event — cut the return short), `QUIT` (immediate).
- Events (child → stdout, prefix `MAKERMODSLAB-DRTC`): `READY url=… room=…`, `CONNECTED`, `ACTIVE operator=…`, `STATS {json}`, `RETURNING`, `ERROR <msg>`, `BYE`. Reuse `eval_protocol`'s `format_event`/`parse_event` shape verbatim — including the "match the prefix anywhere in the line" rule, which exists because a log record flushed without its newline would otherwise swallow the following event.

**Return-to-rest lands in `robot_sync.py`.** Capture the start pose right after `robot.connect()` — `capture_rest_pose(robot.bus, normalize=False)` for Feetech, `capture_maker_pose(robot)` for CAN — and on the shutdown path call `return_to_rest_pose` / `return_maker_to_pose` with an `abort_event` before `robot.disconnect()`. **Exclude the gripper**, matching teleop and record rather than replay: the policy may have left it holding something at stop time. Gate it behind `--return_to_rest` defaulting to **true** — the default should be the safe one everywhere, and the bench keeps `--no-return_to_rest` for A/B work. This does make `robot_sync.py` import Lab modules (`rest_pose`, `maker_rest_pose`, `arm_capabilities`), so it stops being portable back to the standalone repo; that is a real, acceptable cost of integration and worth stating in the module docstring.

**Second stop press mid-return** mirrors `abort_event` exactly as `replay._replay_worker` and `teleoperate._return_followers_to_rest` do: the second `STOP` line sets the child's abort event, `return_*_to_pose` comes back `cut-short`, and torque releases where the arm is — _nearer rest than it started_. Parent-side: bound the wait at `RETURN_CEILING_S + 5` (`MAKER_RETURN_CEILING_S + 5` for CAN), then `_terminate_tree` (SIGTERM → SIGKILL over the process group, `start_new_session=True`). **Import `_signal_group`/`_terminate_tree` from `rollout`, or lift them into a shared module — do not copy them**; the process-group discipline there is load-bearing and was written after two orphaned runners survived SIGTERM by eight minutes.

**Lease expiry** is free: `check_expiry` → `_dispatch_stop("remote_inference")` → the same handler. The one thing to get right is `_WINDING_DOWN_PHASES = frozenset({"releasing", "stopping"})` — if the return-to-rest window carried a new phase name like `returning`, an expiry tick landing during it would dispatch a _second_ stop into an in-flight return. **REC: emit `stopping` during the return** and expose `returning_to_rest: bool` on the status for the UI. The alternative (add `"returning"` to the frozenset) is a one-line, tested change and equally defensible; reusing `stopping` is what every existing kind does and needs no edit to a safety-critical set.

**CAN arms: refuse outright.** `robot_sync.py` imports only `bi_so_follower`, `koch_follower`, `omx_follower`, `so_follower` — `maker_follower` / `metal_follower` are not registered with draccus there, so `--robot.type=maker_follower` fails at CLI-parse time _inside the child_, after the session has claimed and preflighted the arm. Add `supports_remote_inference(arm_type)` to `arm_capabilities.py` (the module whose stated purpose is "so the constraint is a value to read rather than a fact somebody has to rediscover"), return a synchronous 400 in the shape of the coaching refusal, and note in its docstring that unlike `supports_dagger` this is a **wiring** limit, not a hardware one.

---

## 4. Status / telemetry

Tonight's `[robot]` line is the spec:

```
[robot] t=  1s chunks=3 reqs=4 sched=6 lead=10 lat=8st/264ms holds=41
        chunk_age=320ms active=policy e2e=221.00ms/256.00ms (p50/p95) rtt=74.00ms
```

**REC: emit a structured JSON line _alongside_ the human line; pump it in the parent; expose it as fields on the typed status response; carry only `session_changed` on the WebSocket.**

```
MAKERMODSLAB-DRTC STATS {"t":1,"chunks":3,"reqs":4,"sched":6,"lead":10,"s_min":4,
 "lat_steps":8,"lat_ms":264,"holds":41,"degrade":false,"chunk_age_ms":320,
 "active":"policy","e2e_p50_us":221000,"e2e_p95_us":256000,"rtt_us":74000,"uncorr":0}
```

Keep the human line: it is the artifact that made tonight's run diagnosable, and it goes to the log file the same way `_pump_stdout` tees lerobot's output. Emit with `json.dumps(..., separators=(",", ":"))` — `format_event` collapses whitespace in the payload, which is safe for compact JSON and would mangle pretty-printed JSON. Pin that with a test.

**Why poll, not push.** The repo has both precedents, and they are explicitly distinguished: `session_changed` and `jobs_changed` are droppable _hints_ (consumers refetch, a miss self-heals), while `coaching_state` is the one event that carries _state_ — justified in `ConnectionManager.notify_coaching_state` because it is "safety-relevant and a refetch round-trip is most of the latency we are trying to remove: the operator has to know who is holding the arm now." Remote-inference telemetry does not clear that bar. `holds` climbing and `DEGRADE` mean the run is _degrading in quality_, and the operator's response is "stop the run" — not a millisecond decision. The line arrives at 1 Hz, the rate the status poll already runs at. And a droppable channel drops under queue pressure, which is exactly when the run is in trouble. So: `session_changed` on real transitions only (claim, connecting, warming*up, running, stopping, release), metrics on `GET /api/v1/remote-inference-status`. If the panel later proves it needs sub-second `active`/`DEGRADE`, name a typed event \_then*.

Response model `RemoteInferenceStatusResponse` (in `schemas/sessions.py`, tag `sessions`) with a nested `stats: RemoteInferenceStats | None`. Per the schema fidelity rule, the model must describe the handler dict _exactly_ — so have the child **always emit every key** (nulls where unknown), which makes an exact model achievable and keeps `response_model_exclude_none`/`_unset` out of it. Alongside `stats`: `phase`, `started_at`, `elapsed_s`, `duration_s`, `log_path`, `outcome` / `error` / `hint` (reuse `_classify_outcome`'s and `friendly_hint`'s contract verbatim so terminal handling matches its local sibling), `warning` (identity preflight), `returning_to_rest`, `shutting_down`, and a `transport` block (`url`, `room`, `source`, `operator_present`).

**Joint broadcast: no.** `broadcast_joint_data_sync` has exactly two callers — `replay.py` and `teleoperate.py`. `rollout.py` does not feed the 3D viewer, so remote inference matching that keeps the two inference modes visually identical. There is also a real cost if it were wanted: `robot_sync`'s control loop calls `robot.get_observation()` only when `want_obs` (once per chunk, not per tick), so a viewer fed from there would be jerky at ~2 Hz, and making it smooth means an extra bus read every tick. That is the reason, and it is worth writing down rather than rediscovering.

---

## 5. Credentials and preflight

**Read non-destructively.** `_env.load_env()` mutates `os.environ` and loads `livekit.local.env` with `override=True`. In a short-lived CLI that is correct; **in the long-lived FastAPI process it is a latent bug** — once the server has loaded a local-SFU override, deleting the file cannot un-set the variables, so the server points at a dead `ws://127.0.0.1:7880` until it restarts. **REC: add `_env.read_env() -> dict` implemented with `dotenv_values` in the same four-source order, and reimplement `load_env` on top of it.** `tests/test_drtc_env.py` stays the single authority on precedence and simply gains a parallel parametrization.

**Pin the transport explicitly.** Add `--livekit_url` / `--livekit_room` flags to `robot_sync.py`, mirroring what `modal_policy.py` already has for url/key/secret, with `load_env` as the fallback. The session then passes what its preflight actually verified, and the whole class of "parent verified room X, child's `.env.local` said room Y" disappears. The child echoes the effective values in `READY url=… room=…`; the parent compares and errors on mismatch — which also catches "the SFU script was restarted between preflight and spawn".

**Preflight ladder** (all cheap and synchronous in `handle_start_remote_inference`, before or immediately after the claim with a `_release_slot()`):

1. **Extra present** — `importlib.util.find_spec("livekit.portal")`, never an actual import (FFI dylib load). Missing → 400 `transport.extra_missing`.
2. **Credentials present** — the four vars → 400 `transport.not_configured`, naming which are missing and the path `~/.cache/huggingface/lerobot/livekit.env`.
3. **Room probe** — `list_participants`. Connection error → 400 `transport.unreachable`, and the message must branch: if the local override is active, say _"`livekit.local.env` points at 127.0.0.1:7880 and nothing is answering — is `tools/drtc/local_sfu_ts.sh` still running? Delete the override to go back to LiveKit Cloud."_ Auth error → 400 `transport.unauthorized`. Room absent or zero operators → 400 `transport.no_policy`, naming the room and the three causes (no `modal run` started; the `LiveKit-cloud` secret's `LIVEKIT_ROOM` differs from yours; an expired `TS_AUTHKEY` left the container unable to join).
4. **Arm type** — `supports_remote_inference` → 400.
5. **Arm count + cameras** — reuse `_arm_count_mismatch` and `_session_cameras` verbatim; a `CameraResolutionError` is a 400 in the launch panel, as it already is for inference.

**Does `utils/system.py`'s installer fit? No — deliberately.** `InstallManager` takes a single package string, and `POLICY_EXTRAS` maps a policy type to one `lerobot[extra]` target. The drtc extra is three packages, one of them exactly pinned (`livekit-portal==0.2.4`) for a reason that fails _silently_ if broken. The only single-string form is `makermodslab[drtc]`, which re-resolves the whole project including the SHA-pinned lerobot fork against a shared `.venv` — a multi-minute, potentially destructive operation triggered by a button. **REC: report, don't install**, with the exact command _and_ the worktree warning from `docs/drtc/README.md` (an editable install run from a worktree silently re-points every other session's `makermodslab`). Say in the code comment that this is a deliberate non-use of the InstallManager and why.

**The empty-room failure, surfaced fast** — three layers, cheapest first:

- (a) the preflight above refuses **before torque**;
- (b) a `warming_up` watchdog: no `active` within `_ACTIVE_TIMEOUT_S ≈ 15 s` of `CONNECTED` → `error` + stop;
- (c) a `no_chunks` watchdog: `active` present but `chunks == 0` within `_CHUNK_TIMEOUT_S ≈ 10 s` → `error` naming horizon / fps / video_codec / camera names.

Layer (c) is the one that earns its keep: it catches the **schema-fingerprint mismatch**, where the room matches, the operator joins, and Portal silently drops every packet. That failure is invisible by construction and is precisely what `docs/drtc/README.md` warns presents as "a healthy-looking session with 0 chunks and 0 observations."

**Room mismatch specifically:** say plainly that the Lab cannot detect it — the GPU's room comes only from the `LiveKit-cloud` secret, which the Lab cannot read, and there is no CLI flag for it. Layers (a) and (b) convert it from "silent forever with an energized arm" into "refused before torque" or "stopped in 15 seconds". The permanent fix is the `--livekit-room` flag on the Modal wrappers (see §2/Open questions).

**New error domain.** The `DOMAINS` set in `tests/test_api_errors.py` is closed and the grammar is tested. `transport.*` is the honest home: it is an external service the session depends on, exactly the argument that earned `hub` its own domain. Alternatives — folding into `hardware.connect_failed` (wrong: that is the serial bus) or `system.*` (wrong: it is not this process) — both lie. **This is a taxonomy decision and must be made in `tests/test_api_errors.py` first**, per the module docstring.

---

## 6. Frontend

_(Grounded in direct reads of `sessionApi.ts`, `useActiveSession.ts`, `lib/robotSetupGap.ts`, and `components/studio/deployGuards.ts`. Everything below the plumbing line is provisional — see the invalidation list.)_

**Stable plumbing** (safe to write against regardless of the rework):

- `lib/sessionApi.ts`: `"remote_inference"` into `StartableSessionKind`; a `RemoteInferenceSessionOptions` interface into the `SessionOptions` union; `remote_inference: "shared.sessionBusy.activity.remote_inference"` into `HOLDER_ACTIVITY_KEYS`. That map is explicitly a _static catalog_ so `keyUsage.test.ts` can verify resolution — the new key needs entries in **both** en and zh or catalog parity fails.
- `hooks/useActiveSession.ts`: `"remote_inference"` into `SessionKind`.
- `useSessionHeartbeat`: nothing — it is kind-agnostic and keys off `session.id`.
- New status hook polling `/api/v1/remote-inference-status` at 1 Hz, refetching eagerly on a `session_changed` hint, mirroring how the existing inference surfaces consume the seam.
- `lib/robotSetupGap.ts`: nothing — it already takes `scope: "all" | "follower"`, and remote inference passes `"follower"`, same as inference and replay.

**Start form.** The natural home is `DeployPanel`'s run-mode axis: `deployGuards.ts` declares `DeployRunMode = "single" | "eval" | "coach"`, and a fourth value `"remote"` slots in with new `DeployGuardContext` flags (`transportReady`, `armSupportsRemote`) and new keys `studio.deploy.blocked.transportNotReady` / `.remoteArmUnsupported`. `deployGuards.test.ts` is the existing pure test for exactly this and extends naturally. Fields: the existing robot selector and checkpoint picker (which already yields `camera_bindings`, `camera_dims`, `checkpoint_state_dim`, and the task prefill), plus a compact transport group — horizon, fps, codec, duration.

**The highest-value UI element in slice 3 is not the form — it is a copy-able `modal run` line.** Under lifecycle option A the panel must tell the user what to launch in the other terminal, built from the same options plus the `--tailscale --livekit-url/--livekit-api-key/--livekit-api-secret` flags read from the transport endpoint. That removes the documented "the quick tunnel hands out a new hostname every launch, so re-copy the flags each time" footgun. It becomes dead weight the day B lands, and that is fine.

**Status panel.** Phase, elapsed/duration, `active` operator, chunks/reqs, a `lead` vs `horizon − s_min` margin bar, a DEGRADE badge, `chunk_age`, e2e p50/p95, rtt. **Render `holds` as a rate, not the cumulative counter** — tonight's healthy run had `holds` _frozen_ after warm-up, so "holds not growing" is the health signal and the cumulative number (41 at t=1s) is misleading forever after.

**Localization.** Per `frontend/docs/localization.md`'s rule that many strings here are data wearing a label: the codec ids `H264`/`MJPEG`, the room name, the LiveKit URL, and every character of the generated `modal run` line are **data** and must not be translated.

**Conclusions the in-flight studio rework could invalidate.** The most recent commit is _"run-form opener, single-popover pickers, stretched libraries"_, which suggests the launch form is being lifted out of the panels. So treat as provisional: (i) that `DeployPanel` still hosts run modes at all; (ii) that `DeployRunMode` stays a three-value union rather than becoming a form-config object; (iii) that the checkpoint picker still assembles `camera_bindings`/`camera_dims` the way DeployPanel does today; (iv) that the status panel belongs beside the launch form rather than in `StudioOverlay` or `studio/panel/`; (v) the `studio.deploy.blocked.*` key namespace. **REC: land the backend first — it has no frontend dependency — and ship the UI as a separate PR after the rework merges.**

---

## 7. Tests and ratchets

**Unit-tested (pure), in `tests/test_remote_inference.py` unless noted:**

- `RemoteInferenceOptions` defaults and `extra="forbid"` rejections — mirroring `test_inference_request_has_expected_defaults` / `_rejects_unknown_engine`.
- The arg builder from a robot record: `--robot.type/port/id`, `--robot.cameras` keyed by policy names via `bind_robot_cameras`, and the horizon/fps/codec/duration flags — mirroring `test_single_robot_args_appends_bound_record_cameras` and `_captures_at_the_checkpoints_resolution`.
- All eight mutex refusals, parametrized — mirroring the eight `test_handle_start_inference_blocked_when_*`.
- The CAN refusal, and that it releases the slot (mirroring `test_handle_start_inference_arm_count_guard_releases_slot`).
- The STATS parser: valid, malformed JSON, truncated, and interleaved with lerobot INFO chatter (the "prefix anywhere in the line" case).
- The `warming_up` and `no_chunks` watchdogs **with an injected clock** — `tests/test_session_lease.py`'s `FakeClock` + `monkeypatch.setattr(..., "_clock", clock)` is the house pattern. Never sleeps.
- The stop state machine with a fake `Popen` from `tests/mocks.py`: first stop writes `STOP`; second stop while returning writes the second `STOP`; a dead child falls through to `_terminate_tree`; a stop before spawn abandons via the cancel event.
- The room probe behind a seam: put it in one function `_probe_room(url, key, secret, room) -> RoomProbe` and monkeypatch _that_. `livekit-api` is aiohttp-based, so `httpx.MockTransport` does not apply and **no new test dependency may be added**.
- `tests/test_drtc_protocol.py`: `format_event`/`parse_event` round-trip, including that compact JSON survives the whitespace collapse.
- `tests/test_arm_capabilities.py`: `supports_remote_inference`, both halves.
- `tests/test_sessions.py`: a `test_remote_inference_request_is_follower_only` sibling, `_FOLLOWER_ONLY_KINDS` membership, an options-422 parametrization row.
- `tests/test_session_lease.py`: one capture test that expiry routes to the new stop handler.

**Deliberately untested,** per repo policy: the subprocess happy path (spawn → connect → chunks), anything needing LiveKit, hardware, or a GPU, and the `modal run` launch when B lands.

**Ratchets and snapshots that must move:**

| Artifact                                                                    | Direction                                                                                                                                                                                            |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/test_api_errors.py::DOMAINS`                                         | `+ "transport"` (equality-asserted, taxonomy decision)                                                                                                                                               |
| `tests/test_api_errors.py::BUSY_DISCRIMINANTS`                              | `+ "remote_inference"` (equality-asserted)                                                                                                                                                           |
| `tests/test_session_events.py::test_session_kinds_match_the_mutex_features` | `+ "remote_inference"` (equality-asserted)                                                                                                                                                           |
| `tests/test_api_contract.py::V1_ONLY_ROUTES`                                | **grows** by `GET /api/v1/remote-inference-status`, `GET /api/v1/remote-inference/transport`, `POST /api/v1/remote-inference/clear-local-override` (and `GET /api/v1/remote-inference-log` if added) |
| `UNTYPED_V1_ROUTES`                                                         | **must not grow** — every new route ships a `response_model`                                                                                                                                         |
| `LEGACY_ROUTES`                                                             | **must not grow** — nothing on the flat mount                                                                                                                                                        |
| `docs/api/openapi.json`                                                     | regenerate: `uv run python -m makermodslab.scripts.export_openapi`; `test_openapi_snapshot_is_fresh` enforces                                                                                        |
| `test_v1_operation_ids_are_clean_and_unique`                                | handler names must be unique: `get_remote_inference_status`, `get_remote_inference_transport`, `clear_remote_inference_override`                                                                     |
| Frontend                                                                    | `keyUsage.test.ts`, catalog key-parity (en+zh), `deployGuards.test.ts`                                                                                                                               |
| Prose                                                                       | `CLAUDE.md` state-model + module-layout sections; `docs/drtc/README.md`'s "Not yet done" list                                                                                                        |

Note the pleasant consequence of the sessions-first design: **start and stop need no new routes at all** — they go through the existing `POST /api/v1/sessions` and `/sessions/{id}/stop`. The whole new HTTP surface is status and transport.

---

## Risks

1. **Portal's fingerprint mismatch is silent.** Horizon, fps, codec, camera names, state/action dims — any disagreement yields a healthy-looking session with zero chunks, no exception. Mitigated by the `no_chunks` watchdog and by generating the `modal run` line from the same options object.
2. **Camera-name mismatch is the likeliest instance of (1).** `robot_wire_schema` derives track names from the robot's `--robot.cameras` keys; `policy_wire_schema` derives them from the checkpoint's `observation.images.*` with the prefix stripped. `bind_robot_cameras` already returns a dict keyed by _policy-expected_ names — **that is exactly what makes the two agree, so `--robot.cameras` must be built no other way.**
3. **The degrade margin at horizon 16 is two steps.** With `s_min=4`, degrade triggers at `lead ≥ 12`; tonight ran at `lead=10`. Two steps at 30 fps is **67 ms of headroom** — a Wi-Fi hiccup, a Modal region change, or a bad GFW day tips a healthy run into freeze-per-boundary. Mitigations: default the horizon from the checkpoint's `n_action_steps` and prefer larger (ACT ships `chunk_size` 100); render `lead` against the threshold in the panel; and reconsider exposing `s_min` after all.
4. **Modal cold start** (60–180 s) is why B must launch the GPU _before_ claiming the arm — otherwise a session energizes an arm and idles it for minutes.
5. **Tailnet auth-key expiry.** `TS_AUTHKEY` is reusable+ephemeral with a finite life; when it lapses the container fails inside `_tailscale_up` and never joins. That is indistinguishable from an empty room, so the `transport.no_policy` message must name it.
6. **The first-action jump.** `robot_sync` sends the policy's first action with no ramp and no `max_relative_target` — the doc's own "Not yet done" item, and the same snap-to-pose family analysed for teleop/record on 2026-09-01. **I would gate slice 3 on fixing this**: ease in to the first chunk's step-0 pose using the very primitives replay already uses for its frame-0 approach (`return_to_rest_pose` / `return_maker_to_pose` with `EASE_ARRIVE_TOLERANCE`), giving the operator an `easing_in` phase to watch. Worth confirming first whether lerobot's own rollout ramps on entry — if it does, this is a regression against the local sibling rather than mere parity, which raises the priority further.
7. **`livekit.local.env` outliving its script** — handled by the preflight message and the clear-override action.
8. **`load_env()` permanently poisoning the server's `os.environ`** — handled by `read_env()`.
9. **The editable-install-from-a-worktree footgun** — the `transport.extra_missing` message must name the _primary checkout_, never "run this here".
10. **The CAN guard is the only thing** between a Metal record and a session that claims the arm, runs the identity preflight, spawns, and dies at draccus CLI parse. It must be synchronous and pre-claim.

---

## Open questions for the user

1. **Naming.** `remote_inference` (kind / `robot.busy.remote_inference` / `makermodslab/remote_inference.py`) vs `drtc`, `cloud_inference`, `split_inference`. This vocabulary is hard to change later — it appears in the error taxonomy, the WS `session_changed` kind, `last_ended`, and i18n keys in two catalogs.
2. **New error domain `transport`.** A closed-set taxonomy decision; confirm before it lands in `test_api_errors.py`.
3. **Should remote inference be allowed while a _local_ training run holds the machine?** It does not want the local GPU — only the USB bus. Refusing is symmetric and cheap; allowing is a genuine capability the split architecture buys (train locally while a Modal GPU drives the arm). REC: refuse in slice 3, revisit deliberately.
4. **Should PR #111 merge before the slice-3 branch starts?** **Recommend yes**, agreeing with the coordinator. Slice 3 edits `robot_sync.py` and `_env.py`, so building on an unmerged preview means either a long-lived stack or a rebase over moving ground. Merging first also makes "preview vs integrated" a real git boundary, so slice 3's diff is exactly _what integration costs_ — the reviewable artifact worth having. The preview is provably additive (nothing outside `makermodslab/drtc/`, `tools/drtc/`, `docs/drtc/`, plus 22 lines in `utils/config.py` and the extra), so it cannot regress anything.
5. **Does `modal_policy.py` get `--livekit-room` now?** It is a small change to two wrappers, it closes the room-mismatch class permanently, and it is a prerequisite for B. Now, or with B?
6. **Where does the horizon default come from?** If `/jobs/{id}/checkpoints/{step}/policy-config` does not expose `n_action_steps`, either add it or make the user type a number that must match the other terminal — worth deciding rather than defaulting to 16 forever.
7. **Frontend timing** — confirm backend-first, UI after the studio rework merges.
8. **ACT-only first?** `robot_rtc` / `policy_rtc` (the inpainting regime for flow/diffusion policies, with `--s_min` matching on both sides) is deliberately out of scope here. Confirm remote inference ships adaptive-sync only in slice 3.

---

## Implementation slices

### Renumbering (2026-09-03)

The plan below was written before S3.5 was repurposed, so the numbers past S3.4
no longer mean what the entries say. What actually happened, and what is left:

| Slice    | What it is                                                    | State                                                            |
| -------- | ------------------------------------------------------------- | ---------------------------------------------------------------- |
| **S3.5** | The RTC in-painting regime becomes a session ENGINE           | **done** — repurposed from the design's "the Lab launches Modal" |
| **S3.6** | Adopt the Lab-owned SFU (`makermodslab --sfu`) and its signer | **done** — this slice; the design's option C, earned             |
| **S3.7** | MolmoAct2 over DRTC                                           | next                                                             |
| **S3.8** | The Lab launches Modal (the design's old S3.5, lifecycle B)   | later                                                            |
| **S3.9** | CAN arms                                                      | later                                                            |

The two entries at the bottom of this section still carry their original
wording; read them through the table.

**S3.0 — Prerequisite.** Merge PR #111 (the preview). _Gate:_ existing CI green; no new checks.

**S3.1 — `robot_sync.py` hardening (no API surface).**
_Files:_ `makermodslab/drtc/robot_sync.py`, new `makermodslab/drtc_protocol.py`, `makermodslab/drtc/_env.py`, `docs/drtc/README.md`.
_Contracts:_ `--livekit_url`/`--livekit_room` flags; `--return_to_rest` (default true, gripper excluded); ease-in to the first action; the stdin `STOP` protocol and the `MAKERMODSLAB-DRTC` event vocabulary; `_env.read_env()`.
_Gates:_ `pytest tests/test_drtc_env.py tests/test_drtc_protocol.py tests/test_drtc_schedule.py`; `ruff check`/`format`; a manual bench run reproducing tonight's numbers **plus a Ctrl-C that returns the arm before going limp**.

**S3.2 — The feature module.**
_Files:_ new `makermodslab/remote_inference.py`; `makermodslab/arm_capabilities.py`; `makermodslab/api_errors.py`; reciprocal guards in `teleoperate.py`, `record.py`, `rollout.py`, `replay.py`, `calibrate.py`, `auto_calibrate.py`, `wiggle.py`; `makermodslab/jobs.py` (`_robot_busy`); `makermodslab/session_events.py`.
_Contracts:_ `remote_inference_active` + `remote_inference_is_active()`; `robot.busy.remote_inference`; the `transport.*` domain; `supports_remote_inference`; the phase vocabulary; the two watchdogs.
_Gates:_ `tests/test_remote_inference.py`, `tests/test_api_errors.py` (two equality lists edited), `tests/test_session_events.py`, `tests/test_arm_capabilities.py`.

**S3.3 — The API surface.**
_Files:_ `makermodslab/sessions.py`, `makermodslab/schemas/sessions.py`, `makermodslab/server.py`, `docs/api/openapi.json`, `CLAUDE.md`.
_Contracts:_ `STARTABLE_KINDS` / `_FOLLOWER_ONLY_KINDS` / `_OPTIONS_MODELS` / `_REQUEST_BUILDERS` / `_dispatch_start` / `_dispatch_stop`; `RemoteInferenceOptions`; `SessionStartBody.kind`; typed `GET /api/v1/remote-inference-status`, `GET /api/v1/remote-inference/transport`, `POST /api/v1/remote-inference/clear-local-override`.
_Gates:_ `tests/test_sessions.py`, `tests/test_api_contract.py` (`V1_ONLY_ROUTES` grows; `UNTYPED_V1_ROUTES` and `LEGACY_ROUTES` unchanged), `test_openapi_snapshot_is_fresh`, and **`pre-commit run --all-files`** before the PR.

**S3.4 — Frontend (after the studio rework merges).**
_Files:_ `frontend/src/lib/sessionApi.ts`, `hooks/useActiveSession.ts`, a new status hook, `components/studio/deployGuards.ts` + wherever the run form has landed, `i18n/locales/{en,zh}/*`.
_Gates:_ `npm run lint`; `npx tsc --noEmit -p tsconfig.app.json` **and** `-p tsconfig.node.json`; `npm run build`; vitest (`keyUsage`, catalog parity, `deployGuards`). Never hand-build `frontend/dist`.

**S3.5 — (later) Lifecycle option B.** Lab launches Modal: `--livekit-room` on both wrappers, `modal` CLI discovery mirroring `_find_uv`, pre-claim GPU launch with room polling, robot-first stop ordering.

**S3.6 — (later, only if earned) Lifecycle option C.** A supervised local SFU as a `system.*` feature, not a session.

---

## S3.2 as built (2026-09-03)

Only the deviations from the design above; everything not listed here landed as
written. Nothing in `sessions.py`, `schemas/`, `server.py`, `docs/api/` or
`frontend/` was touched — that is S3.3/S3.4.

- **The `_terminate_tree`/`_signal_group` lift did not happen, and is not
  needed.** §3 says "import them from `rollout`, or lift them into a shared
  module — do not copy them", and the import turned out to be enough: there is
  no import cycle to break. `rollout` reaches `remote_inference` through the
  same FUNCTION-LOCAL import block it already uses for its six other reciprocal
  guards (as do `record`, `replay`, `calibrate`, `auto_calibrate`, `wiggle`,
  `jobs._robot_busy` and `sessions._held_by`), so `remote_inference` can import
  `rollout` at module top. Lifting them would have meant a ~60-line delete in
  `rollout.py`, a file two open PRs (#113, #112) are also editing; the guard
  hunk there is now one import line plus one six-line block. The lift remains a
  clean follow-up refactor if `procs.py` is wanted for its own sake.
- **Reciprocal guards RETURN a refusal dict; they do not raise `ApiError`.**
  That is what all seven existing guards do in every peer (the route layer
  converts `{"success": False, "status_code", "message", "code"}`), and the
  instruction to keep these hunks trivially mergeable outranks introducing a
  second style beside them.
- **Phase name `easing`, not `easing_in`.** The contract note above calls
  `easing_in` "the natural phase name for the parent"; the implementation
  brief's phase list said `easing`. One constant
  (`remote_inference.PHASE_EASING`), no consumers yet — trivially flipped
  before the frontend keys off it.
- **The EASING phase arrives AFTER `warming_up`, not before.** The design lists
  the vocabulary as connecting → easing → warming_up → running, but the child
  emits `EASING` when it begins ramping into the FIRST CHUNK, which is
  necessarily after it joined the room. The causal order is therefore
  connecting (READY) → warming_up (CONNECTED) → easing (EASING) → running
  (first STATS with `chunks > 0`). The vocabulary itself is unchanged.
- **`_prepare_robot` is reached through a small adapter.** It and
  `_session_cameras` take an `InferenceRequest`, so `_robot_request()` builds
  one from the `RemoteInferenceRequest` rather than relying on the two models
  happening to share attribute names (they do not: `_prepare_robot` reads
  `.coaching` first thing). The adapter is pure and pinned by a test.
- **The watchdogs are not a thread.** They are one function
  (`_check_watchdogs`) called from the stdout pump on every line and from the
  status handler on every poll. The child logs at 1 Hz for as long as its
  control loop runs — which it does in BOTH empty-room failure modes — so there
  is nothing to watch that does not already wake one of those two. The verdict
  half (`_watchdog_failure_locked`) is split out and reads the injected
  `_clock`, so both are tested with a FakeClock and no sleeps. When one fires it
  WRITES STOP and returns rather than calling the stop handler: calling it from
  the pump would block the very thread that has to drain the child's stdout for
  the return-to-rest to finish.
- **The four `[drtc]` packages are one guarded top-level import.**
  `python-dotenv` is in the extra too, so `drtc._env` cannot be a hard import in
  a module the server loads at boot. `aiohttp`, `dotenv`, `livekit.api` and
  `drtc._env` are imported together in one `try/except ImportError` that sets
  all four to `None`; `_extra_missing()` reads that plus
  `importlib.util.find_spec("livekit.portal")` (never an import — it is the FFI
  dylib). Verified by importing the module with those packages blocked.
- **`transport_hint()` lives in `utils/errors.py` and is PURE.** The
  local-override branch §5 asks for takes `local_override: bool` as an argument
  rather than stat-ing `DRTC_LOCAL_ENV_PATH` itself, so the wording is testable
  without a filesystem. The handler passes
  `_transport_source(url) == "local_override"`. Hints are appended to the
  refusal `message` rather than carried as a new dict key, because the
  dict→HTTPException conversion the route layer performs only knows
  `message`/`code`.
- **`remote_inference_transport()` was added** (a pure read of `read_env` +
  provenance, no network) so S3.3's `GET /api/v1/remote-inference/transport` has
  something to call. Its shape is a suggestion, not a contract — no test pins
  it beyond the provenance helper.
- **Status-dict keys are pinned by an equality assertion** in
  `tests/test_remote_inference.py::STATUS_KEYS`, and the live, terminal and idle
  payloads all come from one builder. S3.3's `response_model` must describe
  exactly that set — `response_model` silently FILTERS undeclared fields.

### S3.1 leftovers closed here

- **Teardown is interrupt-shielded.** `contextlib.suppress(Exception)` does not
  cover `KeyboardInterrupt`, so a third Ctrl-C could unwind `run()`'s `finally`
  and skip `robot.disconnect()` — the call that releases torque — leaving the
  arm energized with no `BYE`. Every teardown step now goes through
  `robot_sync.shielded` (retry-once on an interrupt, `reraise=True` for the
  disconnect so a genuine failure still ends the process non-zero, as before).
  A source-level test pins that no teardown step is called bare.
- **`motor_power.reset_torque_limit(robot, FOLLOWER)` runs right after
  `robot.connect()`** on Feetech buses. `Torque_Limit` is RAM and survives
  between sessions on one power-up, so the bench record's `motor_power=38`
  would otherwise have throttled every hand-run `robot_sync` to 38% torque —
  sluggish, healthy-looking, and unexplainable from the logs. Every other Lab
  session already does this at start; this entrypoint was the one that did not.

---

## S3.3 as built (2026-09-03)

The API surface, and only the deviations from the design + the S3.3 plan.

- **The request builder passes NO `right_follower_*`.** The plan (written
  before S3.2 existed) had it forwarding the right half like
  `_build_inference_request` does. `RemoteInferenceRequest` has no right half:
  bimanual is refused outright by `supports_remote_inference`, so the fields
  would have had nowhere to land. `mode` still travels — it is what that
  refusal and rollout's `_arm_count_mismatch` read.
- **`remote_inference_transport()` was renamed to
  `handle_remote_inference_transport()` and reshaped.** S3.2 shipped it as an
  explicitly non-binding suggestion (no test pinned it). It now returns the
  exact `RemoteInferenceTransportStatusResponse` shape — every key always
  present, with `endpoint_reachable` / `operator_present` / `error_code` /
  `message` NULL when the probe did not run — and it runs the `_probe_room`
  probe, which the S3.2 version did not. The rename also keeps the v1 operation
  id free: operation ids are the ROUTE handlers' bare names, and the GET routes
  are `get_remote_inference_status` / `get_remote_inference_transport` (the
  `get_` prefix reads as the verb an SDK method wants, and keeps the route
  handler distinct from the feature function it delegates to).
- **That handler is SYNC, not `async def`.** The plan wrote it async.
  `_probe_room` calls `asyncio.run` internally, which RAISES inside a running
  loop — as a plain `def` the route is dispatched to FastAPI's threadpool,
  which is exactly the contract `_probe_room`'s docstring states. An `async
def` would have turned every configured call into a 500.
- **`handle_clear_local_override()` is new here.** S3.2 did not write it
  (SLICE3 assigned the routes to S3.3 and was silent on the handler bodies).
  It lives in `remote_inference.py` beside the transport read; `server.py` stays
  a thin delegation, like every other feature route.
- **Two `source` fields, deliberately different widths.** The STATUS's
  `transport.source` is `_transport_source`'s range (`cloud` /
  `local_override` / `cwd`) — that function answers "which FILE names this
  exact url" for a refusal message. The TRANSPORT route's `source` comes from
  the new `_resolved_transport_source`, which walks `read_env`'s precedence
  chain and can also say `process_env` and `none`. A pre-launch panel needs
  that distinction ("your shell exported LIVEKIT_URL" and "livekit.env says so"
  have different remedies); a running session does not.
- **`_WINDING_DOWN_PHASES` needed no edit**, as the plan predicted — S3.2 kept
  the phase at `stopping` through the return-to-rest. That is now pinned by
  `test_expiry_during_a_remote_return_to_rest_is_not_double_dispatched`.
- **Route tests redirect the `DRTC_*` paths themselves.** `tmp_lerobot_home`
  patches `utils.config`'s robot/calibration constants only, and
  `remote_inference` binds the DRTC paths by value at import; the tests patch
  them on the MODULE, which is what keeps the clear-override test from
  unlinking a real `livekit.local.env`. They also stub `_dotenv_values`, since
  CI installs `.[test]` and never the `[drtc]` extra.

Ratchets: `V1_ONLY_ROUTES` grew by exactly the three routes;
`LEGACY_ROUTES`, `UNTYPED_V1_ROUTES` and `RESPONSE_MODEL_EXEMPT` are unchanged
(all three routes ship typed, and none returns a file or a stream).

---

## S3.4 as built (2026-09-03)

The frontend, and only the deviations from §6 and the S3.3 plan's §7 touch list.

- **The whole surface is ONE mount point in `DeployPanel`.** §6 called the
  studio rework a live invalidation risk, so everything except the run-mode
  entry, the guard flags and a single `<RemoteInferenceBlock>` element lives
  under `frontend/src/components/remote-inference/`. `DeployPanel`'s own diff
  is a run-mode value, a verb entry, four `runMode !== "remote"` conditionals,
  two hook calls, two state values and that element — the rest rebases as a
  unit.
- **The verb row went from `grid-cols-3` to `grid-cols-2` (2x2).** A fourth
  verb on one row shrank every commitment line to a two-word wrap in a panel a
  third of the overlay wide, and the commitment travelling with the verb is
  the point of that row.
- **`transportReady` is FALSE while the probe has not answered.** §6 named the
  flag but not its pre-probe value. Treating "not checked yet" as ready would
  let an operator launch into an unverified transport and energize the arm for
  a run nothing may ever drive, so the guard's own message points at the
  transport read-out rather than trying to state which of the four conditions
  failed — the section below it says that exactly.
- **`inferenceActive` in `DeployGuardContext` is now fed `inferenceActive ||
remoteActive`,** and the camera previews pause on the same value. The two
  inference modes are mutually exclusive server-side; a remote run claims the
  cameras just as a local rollout does.
- **`temporalEnsembleInvalid` is suppressed in remote mode.** The ACT control
  is hidden there (no local rollout to configure) and the coeff is never sent,
  so leaving the guard armed would refuse a launch by naming a field the
  operator could not make appear — the dead end DeployPanel's own comments
  already warn about for `disabled` verbs.
- **The lease is heartbeated from `DeployPanel`, not from the block.** Remote
  inference opens no session dialog, and `StudioOverlay` keeps `DeployPanel`
  mounted for the whole visit, so the beat survives closing the studio. Without
  it the expiry watchdog safety-stops the run 60 s in, mid-rollout.
- **No deploy milestone for a remote run.** That banner is latched on the local
  `InferenceSessionDialog` closing, which a remote run never opens — it would
  have fired the instant the run started.
- **Stop resolves its own session id.** The block prefers the id this tab got
  from `startSession` and falls back to `GET /sessions/current` (accepting it
  only when the kind matches), so a reload or a run started from another tab
  still has a working Stop. Stopping is never owner-gated server-side, which is
  what makes that legitimate.
- **The generated `modal run` line is a tested pure function**
  (`components/remote-inference/modalCommand.ts`, asserted verbatim). It
  carries `--task` whenever the task is non-empty — §6's specified line did
  not, but sending the sentence to the robot side and not to the container
  does not fail loudly, it just makes a language-conditioned policy worse in
  ways that read as the policy being bad. The task is arbitrary user text that
  reaches a shell, so it is emitted as a double-quoted word with `\`, `"`, `$`
  and the backtick escaped (double quotes, not single: an apostrophe in an
  English task sentence is far likelier than any of those four). Flag order
  follows `modal_policy.py`'s own `local_entrypoint` signature.
- **`policy_hub_id` prefills from the selected job's `hf_repo_id`,** as a
  PLACEHOLDER rather than typed into the field, matching how the task and
  coaching-dataset fields already offer their defaults.
- **New i18n namespace `remoteInference`** (en + zh-CN), rather than growing
  `studio`. Only the four keys the Deploy panel itself resolves —
  `runMode.remote.*`, `runVerbs.remote`, `blocked.transportNotReady`,
  `blocked.remoteArmUnsupported` — landed in `studio`, again to keep the shared
  file's diff small. Data left untranslated as §6 requires: the codec ids, the
  room, the URL, the env-variable names, the error codes, the log path, the
  `DEGRADE` badge and every character of the generated command.
- **`useSessionHeartbeat` and `robotSetupGap` needed no change**, as predicted:
  the first keys off `session.id` alone, and remote inference passes
  `scope: "follower"` like inference and replay.

### After the studio-rework merge (2026-09-03)

§6 named the rework as slice 3's live invalidation risk. It landed
(`feat/studio-panels-rework`, PR #107) and the merge cost was three conflicted
files — `DeployPanel.tsx` and the two `studio` catalogs. The prediction held:
nothing under `components/remote-inference/` changed, and the mount point moved
rather than the surface. What moved:

- **The run modes still live in `RunVerbs`, in the same place.** The rework's
  opener slides a run FORM open under a `PanelEntryControl`; the verb row stays
  below the form, outside the collapsible, and still launches on press. The 2x2
  grid stands, so "Run it remotely" is where S3.4 put it.
- **`<RemoteInferenceBlock>` now lives INSIDE the run form** (the
  `CollapsibleContent`), between the engine picker and the camera list — the
  same relative position, one container deeper. Its own visibility rule is
  unchanged (`runMode === "remote" || remoteActive || remoteStatus?.exited`).
- **The form is forced open while `remoteActive`.** `open={formOpen ||
remoteActive}` on the `Collapsible`, and the same expression on the
  `PanelEntryControl` so the chevron does not claim the form is shut. Remote
  inference is the one mode whose live surface — telemetry and, crucially, its
  Stop — is inline in this panel rather than in the `InferenceSessionDialog` the
  local modes open, so a collapsed form would leave an energized arm with no
  Stop on screen. Cost: the entry control is inert for the length of a remote
  run. The alternative considered and rejected was mounting the block outside
  the collapsible, which would have popped the whole config block on a mere
  HOVER over the remote verb (the verb row arms on hover).
- **The "not visible until a skill is picked" limitation is gone.** The rework
  un-gated the run parameters from `policyConfig` having loaded, and the forced
  open above covers the rest: a remote run started from another tab or through
  the API now shows its status and Stop as soon as the Deploy panel renders.
  `docs/drtc/README.md`'s "Not yet done" bullet was corrected.
- **The camera derivation was NOT duplicated.** The rework replaced the
  per-feature binding dropdowns with name-based binding against Collect's
  read-only `SessionCameraList`; `cameraBindingPayload` / `cameraDimsPayload`
  are still built once in `handleStart`, above the remote fork, so the remote
  request consumes the rework's derivation verbatim.
- **`allCamerasBound` became `allCamerasReady`** in the guard context (the
  rework's rename — "matched and plugged in" rather than "picked from a
  dropdown"). `deployGuards.ts` and its test are untouched by the merge.
- **Catalogs: the rework's new `deploy` keys arrived in the retired "skill"
  vocabulary,** because PR #107 branched before the rebrand. They were merged as
  `deploy.entry` ("Run a policy"), `deploy.policy.label` (the rework called it
  `deploy.skill.label`) and `deploy.checkpoint.pickPolicyFirst` (its
  `pickSkillFirst`), in both en and zh-CN, with `DeployPanel` referencing the
  policy spellings. `deploy.picker.failedBadge` and `train.dataset.row.*` left
  for `landing.modelPicker.failedBadge` / `landing.datasetPicker.row.*`, as the
  rework moved the components that render them.

---

## S3.5 as built (2026-09-03)

**Scope change.** The slice list above reserved S3.5 for lifecycle option B
(the Lab launching Modal). A bench run displaced it: through the Lab, a SmolVLA
eraser-place run at horizon 48 held a clean adaptive-sync transport — holds
flat, no DEGRADE, e2e ~400 ms of which ~280 ms was inference — and the arm was
still visibly jerky at ~1 Hz. 33 chunks in 29 s says why: at ~400 ms latency the
adaptive-sync player aligns each arriving chunk by dropping its stale prefix, so
the plan switches every ~0.9 s, and two flow-policy plans made 400 ms apart
disagree at every seam. That is the failure in-painting exists to remove, and it
is invisible on ACT (80-110 ms e2e). Option B is still open and unstarted.

ANALYSIS §8.4 already had the RTC regime running live against
`modal_policy_rtc.py --slack 2` on this same GPU image pin, so no image change
was needed and none was made.

### The shared glue — and the two pieces that deliberately did NOT move

New `makermodslab/drtc/_session_glue.py`. It owns `emit` / `emit_stats` /
`note_first_operator` / `or_none`, the `LoopControl` dataclass (the three stdin
events, plus `start_command_pump`), `capture_start_poses_or_warn`,
`ease_into_first_action`, `return_step`, `shielded` and `_shielded_disconnect`,
and the four draccus flags. `robot_sync` and `robot_rtc` both import it;
`tests/test_drtc_robot_rtc.py` asserts neither redefines any of them locally.

- **The flags are FIELD FACTORIES, not a shared base dataclass.** Both configs
  open with `robot: RobotConfig` (no default), and inheriting defaulted fields
  from a base would order them before it and make the dataclass
  unconstructible. Factories keep the defaults and the help text identical.
- **The teardown's CALL SEQUENCE stayed written out in each entrypoint.** This
  is the one place the brief's "lift it, don't duplicate it" had to bend, and
  the reason is a test:
  `test_drtc_robot_sync.py::test_every_teardown_step_goes_through_the_shield`
  parses `run()`'s `finally:` and asserts every step is routed through
  `shielded` / `_shielded_disconnect` and that `disconnect` / `close` /
  `return_to_start_poses` are never called bare. That assertion is the ONLY
  guard the torque-release path has — the block only executes with a real arm
  attached — and collapsing the sequence into one `await teardown(...)` call
  would have retired it silently, on both engines at once. So the ~10-line
  sequence is stated twice and `tests/test_drtc_robot_rtc.py` applies the same
  assertion to the second copy, PLUS a new one that the two sequences are
  identical (compared as the ordered list of `shielded` step labels). Net: more
  guarded than before, not less.
- **`reset_torque_limit(robot, FOLLOWER)` likewise**, for the same reason (a
  source assertion per entrypoint, and `test_motor_power_call_sites`'s rule
  that a call site names its side).
- **Credential resolution stayed in the entrypoints** (three lines each).
  `_common` imports `livekit.api`; importing it from the glue would have made
  the glue need the extra and cost its CI-importability, which is what lets its
  pure parts be tested at all. The FLAGS are shared, so the two cannot drift on
  what a parent passes; only `load_env` / `required_env` / `mint_token` are
  restated.

### What `robot_rtc`'s loop needed that `robot_sync`'s did not

- **The `try` had to move up.** Everything from `codec = getattr(...)` to the
  loop was outside any `try`, so a bad codec or a camera that would not open
  raised with the arm already connected and energized and no teardown at all.
  It now opens right after the start-pose capture, with `portal = None` ahead of
  it, exactly as `robot_sync` has it.
- **The ease-in fits, and at the same moment.** `last_action` stays `None` until
  the first `schedule.pop_current()` returns something, and the send is gated on
  `last_action is not None` — so the first scheduled action IS the first thing
  ever commanded, same as `robot_sync`'s `cmd is not None` branch. It eases to
  the RAW target rather than the low-pass output; the filter (off by default)
  seeds its state to DC steady-state on its first sample, so its first output is
  that same pose either way.
- **One real difference, reported not papered over.** `robot_sync`'s player has
  a `pending` flag that guarantees at most ONE outstanding request, and pushing
  the first chunk clears it — so nothing is in flight while the ease blocks the
  loop. `robot_rtc` paces with a COOLDOWN instead, so one or two requests may
  already be out. Their chunks answer into the ease, and their round-trip
  samples are inflated by its duration, spiking the JK estimate. It is clamped
  to `[1, H//2]` and decays back over the next few samples (alpha 0.125), and
  `robot_sync` has the same exposure for any chunk that lands mid-ease, so this
  is parity rather than a new hazard — but it is real, and compensating for it
  would have been a semantics change, not a port. Noted at the call site.
- **Chunk alignment survives the ease** for the same reason it does on the other
  engine: `control_step` and the wall clock freeze together, and a chunk landing
  mid-ease is placed at `t_src + action_delay` where `t_src` is a frozen tick.

### STATS on a frozen key set

`drtc_protocol.STATS_KEYS` was NOT extended. `robot_rtc` fills every key:
`chunks` / `reqs` / `uncorr` / `t` / `s_min` / `horizon` / `lat_steps` /
`lat_ms` directly, `sched` from `ActionSchedule.remaining()`, `chunk_age_ms` /
`active` / `e2e_*` / `rtt_us` as the sync engine does, and a new `holds`
counter incremented on every dry tick — the exact quantity
`AdaptiveBlockPlayer.holds` counts, so the panel's rate means the same thing on
both engines. The two that needed a definition:

- **`lead` = `max(s_min, d)`**, the steps of an arriving chunk the round trip
  has already consumed. It is the RTC analogue of a prefetch lead, it is already
  the `s=` term in the human log line and the `overlap_end` input, and it reads
  correctly against the panel's existing `horizon - s_min` margin bar.
- **`degrade` = `lead >= horizon - s_min`**, which is simultaneously ANALYSIS
  §8.1's budget rule (`roundtrip < (H - s_min)/fps`) and, verbatim, the
  predicate `_sync_player.in_degrade` uses.

Everything with no analogue on the other engine — `merge_l2`, `prefix`,
`smooth`, `emit`, `ret`, `late`, `stale` — stays on the human `[robot]` line.
Nothing was smuggled in, and nothing the UI needs was unrepresentable.

### Session surface

- `engine: Literal["sync", "rtc"] = "sync"` and `s_min: int = 4` on
  `RemoteInferenceOptions` and `RemoteInferenceRequest`, threaded through
  `_build_remote_inference_request`. `_robot_sync_args` keeps its name (it is
  the child's arg builder either way) and gains one conditional flag;
  `_child_module(engine)` picks the module off `_CHILD_MODULES`.
- **`--s_min` is sent for `rtc` ONLY.** On that engine it is half a contract —
  the robot computes `overlap_end` from it and `policy_rtc` trusts the field.
  On `sync` it only tunes when the player calls itself degraded, and the arg
  builder's standing rule is to leave the scheduler knobs at the child's
  defaults. Cost, stated plainly: an API caller who sets `s_min` with
  `engine="sync"` has it ignored. The alternative (pass it always, since
  `robot_sync` accepts the flag and defaults to the same 4) is a one-line flip
  if that trade reads the wrong way.
- **The backend does not, and cannot, verify engine-vs-policy.** It never loads
  the checkpoint. The UI is the gate; the module docstring, the options schema
  and `deployGuards`' own comment all say so.
- `engine` is on the status dict and the response model (`str | None`, present
  in every branch — live, terminal and idle). No new routes; `V1_ONLY_ROUTES`,
  `LEGACY_ROUTES` and `UNTYPED_V1_ROUTES` all unchanged.

### The two S3.4 display bugs

- **`elapsed_s` no longer resets to 0 at the exit.** No new clock was needed:
  `_terminal_payload_locked` runs exactly once, at the exit, on top of
  `_payload_locked` — so its `time.time() - started_at` IS the run's length, and
  the fix was to delete the `"elapsed_s": 0.0` override. Two pure tests pin the
  value and that a later poll does not re-derive it.
- **`holds` no longer blanks to "—" at the exit.** The panel's effect reset the
  rate on `!remote_inference_active`, i.e. at exactly the moment someone reading
  a failed run wants to know whether the arm had been starving. It now resets on
  `started_at` changing (the run's identity, which survives into the terminal
  payload), and the sampling effect simply skips non-live payloads.
  `perSecondRate` already returns null for a dt of 0, so a repeated terminal
  sample cannot overwrite the last real rate with a false zero.

### UI

- `RemoteRunConfig` gains `engine` and `sMin`; `DEFAULT_HORIZON` is
  `{sync: 16, rtc: 50}` and switching engines re-seeds the horizon unless the
  operator has already typed their own.
- `defaultEngineForPolicyType` seeds from `policyConfig.policy_type`
  (`smolvla` / `pi0` / `pi05` / `diffusion` → rtc, everything else including an
  UNKNOWN type → sync), keyed on the resolved type so it re-seeds per checkpoint
  and never under a live run.
- **"≤ the checkpoint's `n_action_steps` if known" could not be implemented.**
  `PolicyConfigSummary` (`GET /policy-config`) carries `policy_type`,
  `image_features`, `requires_task`, `state_dim`, `action_dim` and
  `trained_on_robot_type` — no chunk-size field. So the rtc default is a flat
  50, which is the flow families' actual `chunk_size` and both `robot_rtc`'s and
  `modal_policy_rtc.py`'s own default. Exposing `n_action_steps` /
  `chunk_size` on that endpoint is the follow-up.
- `deployGuards` gains `remoteEngineSupported` →
  `studio.deploy.blocked.remoteEngineUnsupported`, ordered after the arm check
  and BEFORE the transport one: it is a fact about the checkpoint with a
  one-click remedy that "the transport isn't ready" would send the operator
  straight past.
- `modalCommand` switches script on the engine (`MODAL_WRAPPERS`) and emits
  `--s-min` for rtc only, between `--fps` and `--video-codec` where
  `modal_policy_rtc.py`'s `local_entrypoint` declares it. `modal_policy.py` has
  no `s_min` parameter at all, so emitting it there would make the line fail to
  parse. `slack` / `tolerance` / `max_guidance_weight` / `rtc_schedule` stay at
  the wrapper defaults and a test asserts none of them appears.
- New i18n namespace keys under `remoteInference.form.engine.*`,
  `remoteInference.form.sMin*` and `remoteInference.engine.*` (en + zh-CN).
  Deliberately NOT under `studio.deploy.engine.*` — that namespace already
  belongs to the LOCAL rollout's own sync/rtc picker (`inference_engine` on
  `InferenceOptions`), and they are different things.

## After the staging sync (128974b8, 2026-09-03)

Merging `origin/staging` into this branch conflicted in four files
(`DeployPanel.tsx`, both `studio` catalogs, `uv.lock`). What moved:

### The verb row is now three verbs, not four

Staging's `731ab8b` ("two rollout verbs, Run and Human in the loop") retired
**Score it** from `RUN_MODES` and relabelled the other two: `single` is now
**Run** (was "Just run it") and `coach` is **Human in the loop** (was "Coach
it"), with a commitment line that says what the operator actually does. The
eval MODE is untouched — it is still a valid `RunMode` and the scored-evaluation
engine still runs; it is simply no longer offered as a verb here.

S3.4's **Run it remotely** survives that unchanged, so the row is
`single` / `coach` / `remote`. The grid stays `grid-cols-2` (staging's shape for
two verbs): the two LOCAL verbs keep their side-by-side row and the remote verb
carries a new `wide` flag in `RUN_MODES` that renders it `col-span-2` beneath
them — the honest grouping, since it is the one verb that needs a second
machine. Nothing else about the remote mode moved: the guards (including
`remoteEngineSupported`), `useRemoteInferenceStatus` /
`useRemoteInferenceTransport`, the heartbeat, `open={formOpen || remoteActive}`,
the hidden-for-remote local controls, `RemoteInferenceBlock` inside the run
form, the engine / `s_min` fields and the start fork are all as S3.5 left them.

`RunVerbs.test.tsx` (staging's rewrite) enumerates the row, so it was extended:
three buttons, no "Score it", and the remote verb asserted `col-span-2`. Its
`/^Run/i` matcher had to change — "Run" is a prefix of "Run it remotely", and a
verb's accessible name is its label and commitment concatenated with no
separator ("Runhands off"), so the plain-run verb is now matched by
`/^Run(?! it remotely)/i`.

### #113's server-side RTC gate composes with the remote engine choice

Staging's `1581e5dc` added `supports_rtc` to `GET /policy-config` and a 400 from
`handle_start_inference` for an architecture that cannot run Real-Time Chunking.
Its frontend half — `rtcAvailable = policyConfig?.supports_rtc !== false`, the
stale-selection reset, the disabled `rtc` option and the `engine.rtcUnavailable`
line — was re-applied onto the reworked panel. It gates the **local** rollout's
`inference_engine` only. The remote run keeps asking `policySupportsRtc(policy_type)`
(`rtcSupported`, the frontend's own table) because that choice is read by the
GPU side, not by `handle_start_inference`. The two agree wherever the server has
an opinion; unifying them on `supports_rtc` is a cheap follow-up, not done here.

Also ported: the coaching note's last "skill" wording and its "Score it above to
check." tail (gone with the verb).

### Everything else

`uv.lock` was regenerated with `uv lock` rather than hand-merged; against
staging's lock it is additive only — the `[drtc]` extra's five packages
(`livekit-portal` 0.2.4, `livekit-api`, `livekit-protocol`, `pyjwt`,
`types-protobuf`) plus `provides-extras`. `docs/api/openapi.json` was
regenerated and was already up to date. The lerobot pin moved to `eaab69339`,
whose only change over `b968c0c01` is the Star-leader unwrap-window fix
(`rebot_102_leader`) — nothing the DRTC child imports — but the backend suite
ran against the OLD pin still installed in the venv, so the bump itself has to
be exercised on the bench.

---

## S3.6 as built (2026-09-03)

The design's option C ("the Lab also owns the SFU") said _"a lot of surface for
something a 146-line shell script does well"_ and deferred it to "only if
earned". It was earned by someone else: `feat/livekit-sfu` landed
`makermodslab --sfu` and `POST /api/v1/sfu/token` for remote TELEOPERATION, and
once the surface exists, keeping a second, worse local-SFU story for remote
inference is the expensive choice. This slice merges that branch and adopts it.

### What the session takes from the SFU, and what it stops reading

When `sfu.sfu_enabled()` (the launcher exported a key file), a remote-inference
start resolves its whole transport in-process and **reads no credential file at
all**:

| Piece           | Source                                                  |
| --------------- | ------------------------------------------------------- |
| url             | `sfu.local_url()` — `ws://127.0.0.1:7880`               |
| room            | `sfu.default_room(get_instance_id())` — `mml-<id[:12]>` |
| child's token   | `sfu.mint_token(role="robot", identity="robot", ...)`   |
| the probe's key | `sfu.api_keys()`, from the 0600 key file                |

The token rides a NEW `--livekit_token` flag, defined once in
`_session_glue.livekit_token_field()` so the two entrypoints cannot drift, and
consumed as `token = cfg.livekit_token or mint_token(IDENTITY, room)`.
`_common.mint_token` stays for the Cloud path. The consequence worth stating:
**the child process never holds an API secret** — a JWT scoped to one room and
one identity is what it needs, and that is all it gets.

**The Portal identities are CONTRACTS, not defaults.** The child is `robot` and
the GPU side is `policy`; `_probe_room` looks for the latter by name (or by
Portal's `lk.portal.role` attribute). `sfu.default_identity` mints
`<role>-<8 hex>` for a browser or a laptop and is deliberately NOT used here —
a random identity would make the room probe blind.

### What was retired

- `tools/drtc/local_sfu.sh`, `tools/drtc/local_sfu_ts.sh` (deleted).
- Three credential rungs in `drtc/_env`: cwd `.env`, cwd `.env.local`, and
  `livekit.local.env`. All were cwd-relative or `override=True`, and both
  properties are bugs in a long-lived server. `read_env` is now
  `livekit.env` < process environment, and `search_from` is accepted-and-ignored.
- `config.DRTC_LOCAL_ENV_PATH`, `config.DRTC_SFU_CONFIG_PATH`.
- `handle_clear_local_override`, `POST /api/v1/remote-inference/clear-local-override`,
  `ClearLocalOverrideResponse`, and its `V1_ONLY_ROUTES` row. Removing a row
  from that register is allowed — it is the record of what exists, not a
  shrink-only ratchet like `LEGACY_ROUTES`.
- `_resolved_transport_source`. There were two source walkers because the
  status model's `source` was narrower than the route's; with one enum for both
  (`sfu | cloud | process_env | none`), `_transport_source` serves both.
- `transport_hint`'s `local_override` branch, replaced by `sfu`.

### The transport endpoint

`GET /api/v1/remote-inference/transport` keeps every key it had except the
three local-script ones (`sfu_config_exists`, `local_env_exists`,
`local_env_path`) and gains six:

`sfu_enabled`, `sfu_url` (loopback), `sfu_modal_url` (the TAILNET url a Modal
container should dial — `ws://<tailscale ip -4>:7880`, null when tailscale is
absent or not logged in), `sfu_external_ip`, `sfu_key_id` and `sfu_key_file`,
plus `sfu_install_hint` when the binary is missing.

**`sfu_key_id` is the key's NAME and the secret is never returned.** The key ID
is the `--livekit-api-key` half of the generated `modal run` line and it
identifies rather than authorizes; the secret signs every room token for the
life of the install, so the panel names the file and a human reads it. A test
asserts the secret is absent from the raw response body.

`sfu_modal_url` is not `sfu_url`. A Modal container has no route to loopback
and none to a LAN address; offering it one would be a line that cannot work.
Null is the honest answer, and the panel says so.

### The change to sfu.py — `--sfu-external-ip`

`render_config` gained `external_ip: bool = False`. True writes
`use_external_ip: true` and **drops** the `rtc.node_ip` pin — the two are
mutually exclusive, because pinning the tailnet address is exactly what makes
the STUN-discovered public candidate unreachable.

Why it is needed at all: **signalling and media take different paths.** A Modal
container reaches signalling over the tailnet (`--tailscale` stands up a
loopback→SOCKS5 relay for it), but WebRTC media and data channels hole-punch
directly, and a tailnet address is not a route the container has. The public
`<ip>:7882` candidate is the only one it can punch to. Off stays the default:
the STUN self-probe stalls a station with no internet, and a LAN-only station
does not want it.

Threaded as launcher flag `--sfu-external-ip`, exported as
`MAKERMODSLAB_SFU_EXTERNAL_IP=1` so the app can REPORT it (it cannot act on it
— the config was rendered before the app started). Two additive helpers came
with it: `sfu.external_ip_enabled()` and `sfu.local_url()`, the app-side
counterpart of `sfu_url()` for a child with no request to derive a host from.

### `--bind` in `--dev` mode now reaches the SFU

As merged, `_run_dev` called `_start_sfu(sfu_bin, "127.0.0.1")` unconditionally
and `main()` warned that `--bind` was ignored in dev. That is right for Vite and
uvicorn (Vite serves localhost only), but the SFU is not a web server for the
developer's browser: a Modal container has to reach its SIGNALLING port, and a
loopback bind made a `--dev` session LiveKit-Cloud-only. The user runs `--dev`.

Fixed here, as narrowly as it goes: `_run_dev` takes `sfu_host="127.0.0.1"`,
`main()` passes the already-resolved `bind_host or "127.0.0.1"`, the one
`_start_sfu` call uses it, and the warning narrows to "`--bind` applies to the
SFU only in `--dev` mode (Vite and uvicorn serve localhost)". Vite and uvicorn
are untouched, no new flag, and the default is unchanged — a dev session that
never asked for a remote peer does not start advertising itself on an
interface. Two tests pin both halves.

### Notes worth keeping

- **The token is on the argv.** A JWT is visible in `ps` to this user's own
  processes. It is a deliberate trade — short-lived, one room, one identity —
  and it is what keeps the SECRET off the command line, which is the property
  `sfu.py`'s docstring actually promises. If the coordinator wants it tighter,
  the child's environment is the next rung down (`/proc/PID/environ` is 0600),
  at the cost of the flag being invisible in a log of the spawn.
- **`livekit-api` moved from the `[drtc]` extra to core dependencies**
  (`>=1.0`, his bound). The token route needs it at boot and so does the room
  probe; leaving a second copy in the extra would only invite the two bounds to
  drift. The extra is now `livekit-portal==0.2.4` + `python-dotenv>=1` — the
  FFI dylib the server must never load, plus the dotenv reader the Cloud
  fallback needs.
- **`DRTC_LOG_DIR` stays**, minus its SFU/cloudflared role: it is now just the
  parent of the sessions log dir.

## S3.8 as built (2026-09-03)

Lifecycle option B ("the Lab also launches Modal"), and its stated prerequisite
was already met: `--livekit-room` landed on both wrappers in S3.3/S3.4, so this
slice is smaller than §2 implies.

### Option B landed as a Lab-level RESOURCE, not as a session field

The obvious shape — `launch_gpu: bool` on `RemoteInferenceOptions` — was
rejected, and the reasons are all in existing code:

1. **The design's own pre-claim rule forbids it.** §2B wanted the GPU launched
   in a pre-claim phase. Under a session field, `handle_start_remote_inference`
   sets `remote_inference_active = True` before anything else (deliberately —
   it is the race protection), so a 60-180 s cold start would hold the claim
   with nothing touching hardware.
2. **The mutex would lie.** `robot.busy.remote_inference` would refuse teleop,
   record, replay and both calibrations for minutes **while the arm was
   completely free**.
3. **Three safety-critical sets would have to widen.** `_WINDING_DOWN_PHASES`,
   `_WATCHED_PHASES` and `_dispatch_stop` all assume a session that has (or is
   about to have) a child holding the bus. A phase with no child and no arm is
   outside every one of them.
4. **The lease's promise would break.** An expiry tick landing in the pre-claim
   window would have `_dispatch_stop("remote_inference")` doing something that
   is not de-energizing an arm — the exact opposite of what the lease is for.
5. **The precedent is already in the tree.** `jobs.training_is_active()` is a
   Lab-level resource the session consults but does not contain, and
   `utils/system.InstallManager` is the singleton-plus-two-routes shape this
   copies almost line for line.

So: `makermodslab/modal_launcher.py`, three v1 routes
(`POST /remote-inference/gpu/start`, `POST …/gpu/stop`, `GET …/gpu`, all
`tags=["sessions"]`), `GpuLaunchResponse` / `GpuStatusResponse` in
`schemas/sessions.py`, and a new `gpu.*` error domain. `SESSION_KINDS`,
`STARTABLE_KINDS`, `_OPTIONS_MODELS`, `_REQUEST_BUILDERS`, `_dispatch_start`
and `_dispatch_stop` are all **unchanged** — that is the point.

The clinching UX argument is that the readiness signal already existed and was
already being polled: `GET /remote-inference/transport` reports
`operator_present`, so "Start GPU" → progress → the existing poll flips
`operator_present` → "Run it remotely" unblocks, with zero new session surface.

### ONE transport resolver

`remote_inference.resolve_transport()` is now THE credential resolution, and
three callers use it and nothing else: the session's preflight, the read-only
transport endpoint, and `modal_launcher.resolve_transport_plan()`. A second
credential path is not a duplication smell here — the two halves meeting in
different rooms is invisible by construction (Portal drops the mismatched
stream in silence), so it is the failure mode itself.

The one thing callers legitimately differ on is which URL a given peer can
DIAL, so `sfu_modal_url()` (the tailnet address, ex-`_sfu_modal_url`) is public
and separate: it shells out to `tailscale ip -4`, and the session's own
preflight has no business paying for that. `TransportPlan.needs_tailscale` is
the SEAM for the open question `--sfu-external-ip` raises — if the Lab's SFU
ever becomes directly reachable from a container, that goes false in one place
and the argv, the child env and the tests all follow.

### The secret is not in `ps`

`modal run … --livekit-api-secret <secret>` would put a signing key in argv on
the operator's own machine. Both wrappers' `main()` therefore gained six
byte-identical lines:

```python
livekit_api_key = livekit_api_key or os.environ.get("LIVEKIT_API_KEY", "")
livekit_api_secret = livekit_api_secret or os.environ.get("LIVEKIT_API_SECRET", "")
```

A `@local_entrypoint` body runs on the USER'S machine, so this resolves locally
and the value then travels to the container as a `fn.remote(...)` kwarg over
Modal's own TLS channel. `build_argv` passes neither flag; `child_env` passes
both. The flag still wins when present, so every hand-typed invocation and
every line of `docs/drtc/README.md` is unchanged.

Scoped to the two CREDENTIALS only, deliberately not to `--livekit-url` /
`--livekit-room`: those are not secrets, and keeping them flag-only means
"which SFU, which room" stays a visible decision rather than one a stray
`LIVEKIT_ROOM` in an operator's shell can flip — the exact failure class
`--livekit-room` was added to close. The residual hazard is stated rather than
hidden: an operator with `LIVEKIT_API_SECRET` exported who previously fell
through to the `LiveKit-cloud` Modal secret now sends their local one.

### Attached only, and what that costs

No `--detach`. The local `modal run` process is the app's lifeline, so killing
its process group (`rollout._terminate_tree`, reused) stops the app — the
cost-safety property this slice wants. Detached would need `modal app stop`,
persisted app ids and reattachment, and would introduce the one genuinely
expensive failure mode in the design: an orphan A100 nobody knows about.
(`modal app list --json` lists every running app in the WORKSPACE, so
name-matching a reattach is a footgun, not a fallback.)

The trade, stated plainly: **the GPU dies with the Lab.** Acceptable because
the robot session dies with it too, and the child's `finally:` returns the arm
to rest before releasing torque.

### The two deadlines, and what does NOT happen on them

- `_COLD_START_TIMEOUT_S = 300.0`. 60-180 s is the realistic band, but a cold
  `hf-cache` volume plus a first-ever VLA `from_pretrained` can exceed 180, and
  a false failure at that moment is maximally annoying. The message names the
  last phase reached — "stuck at `loading`" and "stuck at `tailscale_up`" have
  nothing in common as remedies.
- `_GPU_IDLE_STOP_S = 600.0`, measured from whichever is LATER: reaching
  `ready`, or the end of the last remote session. An A100 is ~$2-4/hr and
  `_FN_KWARGS["timeout"] = 2h` already caps one forgotten run at ~$4-8; ten
  minutes is longer than a realistic gap between two runs and wastes at most
  ~$0.5. The panel says plainly that a ready GPU is billing, and shows the
  countdown. Visibility is the cheapest cost control there is.

Both are checked from the log pump and from the status poll — **not a thread**,
the same argument `_check_watchdogs` makes ("there is nothing to watch that
does not already wake one of those two").

**The launcher's exit does not stop a session**, and `_dispatch_stop` does not
stop the launcher. A lease expiry is a SAFETY stop whose one job is
de-energizing an arm; adding a slower, network-dependent action to that path is
exactly wrong. And `modal run`'s local exit is a weak signal — it can end on a
log-stream disconnect while the app is fine. The session's own watchdogs
already produce a _diagnosed_ message from the room itself, which is strictly
better; on a nonzero exit the two cards simply sit side by side.

**Readiness is a hint, never an authority.** `state: "ready"` is derived from
the container's stdout (`[policy] connected as` — NOT `claimed control as`,
which policy.py emits from a background, non-fatal task and a healthy run may
never print). The gate on energizing the arm stays `_probe_room`. If Modal ever
reformats its log lines the worst case is a misleading panel and a false
cold-start timeout, not a wrong energization.

**No refusal while a local training run holds the machine.** A Modal A100 is
not this machine's GPU, and CLAUDE.md already flags the existing remote-
inference↔training refusal as "the one asymmetry in the matrix, deliberate and
revisitable" — a second, weaker instance would entrench a rule we already
suspect.

### The failure is coded, and the stop is bounded

`classify_failure`'s verdict rides the status body as `code` (`gpu.*`, null in
every non-failed state), so an SDK dispatches on the FAILURE the way it already
dispatches on a coded refusal — `gpu.unauthenticated` (`modal token new`) and an
expired tailnet auth key are not the same problem, and the prose beside the code
stays free to improve. The panel shows the backend's own hint plus the raw code;
it deliberately does NOT key a translated hint off it, because that would be a
second, localized copy of server prose (CLAUDE.md: "the Python backend is never
localized").

`stopping` is bounded too. `_terminate_tree` escalates SIGTERM→SIGKILL, so the
PROCESS is always gone; what can still hang is the stdout PIPE, whose write end
an un-reaped grandchild may hold open — `readline` blocks forever, the pump
never runs its finalizer, and the launcher would sit in `stopping` with no way
out. So the terminate thread arms `_STOP_DRAIN_TIMEOUT_S = 10.0` **after the
kill returns** (the bound is on the drain, not on the kill, which has a ceiling
of its own), and the next status poll forces the terminal state — `idle` for an
operator or idle-timer stop, `failed` for a deadline, keeping its diagnosis and
the phase that diagnosis names. The message says the kill worked and the
LISTENING stopped, which is the distinction that matters to an operator.

Forcing also clears `_proc`, which ORPHANS the wedged pump: `_handle_line` and
`_handle_exit` both guard on `_proc is proc`, so a zombie thread that finally
wakes cannot write phases or a stale verdict into the next launch's state.

### The generated command stays

`ModalRunLine` / `modalCommand.ts` are kept, collapsed under "Run it yourself
instead". Three reasons: it is the only route when `modal` is missing or
unauthenticated; the only route to `--detach` or a hand-tuned
`--slack`/`--rtc-schedule`; and the ground truth an operator compares against
when the fingerprint watchdog fires. Its two invariants survive — the command
text is DATA and is never localized, and the API still never returns the API
secret (`LOCAL_SECRET_PLACEHOLDER` stays; the Lab-launched path reads the
secret from the key file into the child env, never through a response body).

### Ratchets and what was not touched

`V1_ONLY_ROUTES` +3; `DOMAINS` += `"gpu"`. `LEGACY_ROUTES`,
`UNTYPED_V1_ROUTES` and `RESPONSE_MODEL_EXEMPT` are all unchanged — the three
routes ship typed, and there is deliberately **no log-stream route**: the
status carries `log_path` and `last_line`, and a tail endpoint would be the
first thing in this group to need an exemption entry.

No `modal` Python dependency anywhere (the CLI is discovered as a binary), no
`/reset` wiring (it re-spawns the LAST run globally from a `modal.Dict` with no
session identity, so a Lab-driven reset could resurrect someone else's run — it
stays a documented human escape hatch), no Modal secret management, no image
changes, and **the Lab never reads `~/.modal.toml`**: it inherits the
environment and lets the CLI do its own thing, so a missing token surfaces as a
fast nonzero exit that `classify_failure` turns into `gpu.unauthenticated`
naming `modal token new`.
