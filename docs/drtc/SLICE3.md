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
