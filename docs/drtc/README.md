# Remote inference over LiveKit Portal (DRTC)

A split inference loop: this machine owns the arm and the cameras, a GPU
elsewhere owns the policy, and LiveKit Portal carries observations one way and
action chunks the other.

Both halves now live in this repo — it is a port of the standalone
[`livekit-drtc`](../../../livekit-drtc) repo, and nothing in that repo is
needed to run remote inference any more:

| Piece                                                     | Where                                                                                      |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Robot entrypoints                                         | `makermodslab/drtc/robot_sync.py`, `robot_rtc.py`                                          |
| GPU policy servers                                        | `makermodslab/drtc/policy.py`, `policy_rtc.py`                                             |
| Modal wrappers for the GPU servers                        | `makermodslab/drtc/modal_policy.py`, `modal_policy_rtc.py`                                 |
| Wire schemas / RTC core / latency estimator / offline sim | `makermodslab/drtc/_schema*.py`, `_rtc.py`, `_latency.py`, `_sync_player.py`, `_filter.py` |
| Credential loading                                        | `makermodslab/drtc/_env.py` (re-exported by `_common.py`)                                  |
| Parent↔child line protocol                               | `makermodslab/drtc_protocol.py`                                                            |
| Start-pose capture / ease-in / return-to-rest             | `makermodslab/drtc/_pose.py`                                                               |
| Local SFU scripts                                         | `tools/drtc/local_sfu.sh`, `tools/drtc/local_sfu_ts.sh`                                    |
| Transport-only probe (no robot, no GPU)                   | `makermodslab/drtc/transport_probe.py`                                                     |
| Design record                                             | `docs/drtc/ANALYSIS.md`, `SWEEP.md`, `SYNC_RESULTS.md` (verbatim from the source repo)     |

## Install

The dependencies are an **optional extra** — nothing in the Lab imports this
package at startup, so a normal install never pulls LiveKit:

```bash
uv pip install -e '.[drtc]'
```

**Run that from the primary checkout, never from a git worktree.** An editable
install re-targets the shared `.venv` at whatever directory it is run from, so
doing it inside a worktree silently re-points every other session's
`makermodslab` (and `frontend`) at that worktree. If it happens, the repair is
`uv pip install -e <primary-checkout> --no-deps`.

The extra is `livekit-portal==0.2.4`, `livekit-api>=0.7`, `python-dotenv>=1`.
`livekit-portal` is pinned **exactly** and must match the pin in the two Modal
wrappers' images. Portal fingerprints the wire schema and _silently drops_
packets whose fingerprint differs on the two peers, so a mismatch does not
raise — it presents as a healthy-looking session with 0 chunks and 0
observations.

Three modules are importable **without** the extra, which is what keeps their
tests in ordinary CI: `_env` (python-dotenv alone), `makermodslab/drtc_protocol.py`
(stdlib alone — it is the parent's half of the line protocol, and the parent must
never load the Portal dylib), and `_pose` (lerobot's motors, a hard dependency).
Everything else in `makermodslab/drtc/` imports `livekit.portal` at module top.

## Credentials

`LIVEKIT_URL`, `LIVEKIT_ROOM`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` are read
by `_env.load_env()` from four sources. Lowest precedence first; sources 3 and 4
are loaded with `override=True`, so they beat both the earlier files **and** the
process environment:

1. `~/.cache/huggingface/lerobot/livekit.env` (`config.DRTC_ENV_PATH`) — the
   saved credentials, beside the rest of the Lab's persistent state, so a wheel
   install and a source checkout read the same file. Start from
   [`livekit.env.example`](livekit.env.example).
2. `.env` in the current directory.
3. `~/.cache/huggingface/lerobot/livekit.local.env` (`config.DRTC_LOCAL_ENV_PATH`),
   **override** — written by `tools/drtc/local_sfu*.sh` while a local SFU is
   running. Delete it to go back to LiveKit Cloud.
4. `.env.local` in the current directory, **override** — the source repo's
   convention, kept so an existing `livekit-drtc` checkout still works as a
   working directory.

The process environment wins over 1 and 2 and loses to 3 and 4 — exactly as it
lost to `.env.local` in the source repo.

Source 3 is the one behavioural change from the source repo, which read the
local-SFU override from the **script's own directory**. Two live runs on
2026-09-02 failed with "connection refused" because the robot was started from
a different directory. `tests/test_drtc_env.py` pins the whole order down.

### `read_env` vs `load_env`

Two entry points, one precedence implementation:

- **`load_env()`** resolves the four sources and writes them into `os.environ`.
  For the CLI entrypoints, whose downstream code (`_common.mint_token`,
  `policy.py`) reads credentials from the environment.
- **`read_env() -> dict`** resolves the same four sources onto a **copy** of the
  process environment and hands it back, mutating nothing. `load_env` is
  implemented on top of it.

Use `read_env` from anything long-lived. `load_env`'s `override=True` on
sources 3 and 4 is a latent bug in a server process: once it has stamped a
local-SFU URL into `os.environ`, **deleting `livekit.local.env` can never
un-set it**, so the server keeps dialing a dead `ws://127.0.0.1:7880` until it
restarts. `read_env` re-resolves from disk on every call, so a deleted override
takes effect immediately. Both entry points are parametrized over the same
precedence cases in `tests/test_drtc_env.py`.

One deliberate narrowing: `${VAR}` interpolation inside these files resolves
against the process environment only, not against a value contributed by an
earlier source in the same chain. Nothing in `livekit.env.example` or the
local-SFU scripts interpolates.

## GPU side, on Modal

The Modal wrappers are invoked in **file form, from the repo root** — not as
`python -m`:

```bash
# adaptive-sync policy server (pairs with robot_sync)
modal run makermodslab/drtc/modal_policy.py --policy-path ${HF_USER}/my_policy --horizon 16

# full-DRTC + RTC in-painting server (pairs with robot_rtc)
modal run makermodslab/drtc/modal_policy_rtc.py --policy-path ${HF_USER}/my_pi0 \
    --horizon 50 --rtc-schedule linear --task "Put the lego brick in the box"
```

File form is deliberate. The `modal` CLI is typically installed as a uv tool
with its own interpreter, which cannot import `makermodslab`; so **the wrappers'
module top level imports only `modal` + stdlib**, and the package is shipped
into the container **by path**:

```python
_PACKAGE_DIR = Path(__file__).resolve().parents[1]      # .../makermodslab
... .add_local_dir(_PACKAGE_DIR, remote_path="/root/makermodslab", ignore=[...])
```

`from makermodslab.drtc import policy` then happens **inside** `_serve_impl`,
where it runs only in the container (`/root` is on `sys.path` there). Keep both
halves of that arrangement if you touch these files: a top-level
`makermodslab` import breaks `modal run` locally, and `add_local_python_source`
cannot be used because it resolves modules through the _local_ interpreter.
`add_local_dir` without `copy=True` is a runtime mount, so it must stay the
**last** build step in the image.

### Modal secrets

| Secret           | Keys                                                                   | Needed for                                                          |
| ---------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `LiveKit-cloud`  | `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_ROOM` | always                                                              |
| `huggingface`    | `HF_TOKEN`                                                             | private/gated checkpoints or base backbones (drop it if all public) |
| `tailscale-auth` | `TS_AUTHKEY` (REUSABLE + EPHEMERAL)                                    | `--tailscale` only                                                  |

```bash
modal secret create LiveKit-cloud \
    LIVEKIT_URL=wss://<your-project>.livekit.cloud \
    LIVEKIT_API_KEY=<key> LIVEKIT_API_SECRET=<secret> \
    LIVEKIT_ROOM=portal-lerobot-inference
modal secret create huggingface HF_TOKEN=hf_...
modal secret create tailscale-auth TS_AUTHKEY=tskey-...
```

`--livekit-url` / `--livekit-api-key` / `--livekit-api-secret` / `--livekit-room`
are all per-run flags; each unset one falls through to the `LiveKit-cloud`
secret, so every pre-existing invocation is unchanged. `--livekit-room` was
added on 2026-09-02 and closes a failure class that used to be silent: the room
came _only_ from the secret, and two peers in different rooms never see each
other — the robot reports a healthy connection with zero chunks forever. A
launcher that already knows which room the robot joined can now pin the GPU to
it. `modal_policy.py` records the room in its `/reset` `modal.Dict` too, so a
respawn lands in the same room.

`--fps` / `--horizon` / `--video-codec` must match the robot's flags, and for
the RTC pair `--s-min` must match `robot_rtc`'s `--s_min` too.

Each wrapper defines two otherwise-identical Modal functions, `serve` and
`serve_ts`; `--tailscale` picks the latter, which is the one carrying the
`tailscale-auth` secret. They are unconditional on purpose — Modal evaluates the
module both locally and in the container and the two dependency lists must
match exactly.

`modal_policy.py` additionally stashes each run's arguments in a `modal.Dict`
and publishes a `/reset` GET endpoint that re-spawns the last run, for when the
GPU side dies (LiveKit disconnect, OOM, bad checkpoint) and nothing is listening:

```bash
modal deploy makermodslab/drtc/modal_policy.py
curl https://<workspace>--lerobot-drtc-policy-reset.modal.run
```

The Modal app names (`lerobot-drtc-policy`, `lerobot-drtc-full-policy`) are
unchanged from the source repo so a deployed `/reset` URL stays valid — do not
rename them.

Note: the GPU image pins **upstream** `huggingface/lerobot@8414188`, not the
Lab's `makermods-robotics/lerobot` fork pin from `pyproject.toml`. That is
deliberate for now — the DRTC policy servers only need upstream policy code, and
switching the image to the fork pin is a separate, tested step.

## Robot side

Both entrypoints take the same `--robot.*` CLI as `lerobot-record`, and both are
`python -m` entrypoints so a feature module could spawn them as a subprocess the
way `rollout.py` spawns `lerobot-rollout`:

```bash
python -m makermodslab.drtc.robot_sync \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
    --horizon=16
```

### Safe start and stop (`robot_sync`, SO-101)

`robot_sync` no longer snaps into the policy's first action, and no longer drops
the arm at the end. Both behaviours are on by default; turning one off is for
bench A/B only.

**draccus has no `--no-<flag>` form.** Turn a boolean off with
`--<flag> false` (or `--<flag>=False`) — `--no-adaptive`, which this repo's
docstrings claimed since the port, has never worked. Same for `--align`,
`--return_to_rest` and `--ease_in`.

| Flag               | Default | What it does                                                                                                                                                                 |
| ------------------ | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--ease_in`        | on      | When the FIRST action chunk lands, ramp the arm from where it is to that chunk's step-0 pose at the gentle profile speed, then start executing. Emits `EASING`.              |
| `--return_to_rest` | on      | Capture the pose the arm starts in (right after `connect()`, **gripper excluded**) and drive it back there before torque is released, on every exit path. Emits `RETURNING`. |
| `--livekit_url`    | (env)   | Pin the SFU URL instead of taking it from the credential files. Echoed back in `READY`.                                                                                      |
| `--livekit_room`   | (env)   | Pin the room. Same rationale; pairs with `--livekit-room` on the Modal wrappers.                                                                                             |

The gripper is excluded from the captured pose for the same reason teleoperation
and recording exclude it: the policy may have left it holding something, and
driving it back to its (likely open) starting width would drop the object
mid-return. Replay includes it, because there the dataset drives the gripper.

Both use `rest_pose.return_to_rest_pose` — the ease-in with `normalize=True` and
`replay`'s normalized-unit tolerances (an arbitrary target), the return in raw
ticks. **SO-101 only:** the CAN arms are not registered with draccus here at
all, and Koch/OMX are Dynamixel, where those Feetech unit constants mean
something else — `_pose.feetech_buses` gates on the bus type and the helpers
no-op loudly otherwise. The ease-in additionally needs a SINGLE bus, so a
bimanual BiSO robot gets the return but not the ramp.

### Supervised operation: the stdin/stdout protocol

`makermodslab/drtc_protocol.py` is the contract a parent process (the future
`makermodslab/remote_inference.py`) drives `robot_sync` with. It is its own
module, free of heavy imports, for the same reason `eval_protocol.py` is: the
parent must never import `livekit.portal` (an FFI dylib behind the optional
extra) and the child is exactly the process that does.

Commands (parent → child stdin, one bare word per line):

| Command      | Effect                                                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `STOP`       | Leave the control loop, return to the captured start pose, disconnect, exit 0.                                                 |
| `STOP` again | Set the abort event: the in-flight return is **cut short** and torque releases where the arm is — nearer rest than it started. |
| `QUIT`       | Immediate: no return at all.                                                                                                   |

**Ctrl-C takes the same path as `STOP`**, and a second Ctrl-C during the return
cuts it short — a hand-run bench session gets the same teardown a supervised one
does. EOF on stdin does **not** stop the run: an abandoned session is the
server-side lease watchdog's job, and a child that died on a closed pipe would
drop an energized arm the moment a log pump hiccuped.

Events (child → stdout, prefix `MAKERMODSLAB-DRTC`, matched **anywhere** in the
line so a log record flushed without its newline cannot swallow the event behind
it):

```
MAKERMODSLAB-DRTC READY url=wss://x.livekit.cloud room=portal-lerobot-inference
MAKERMODSLAB-DRTC EASING
MAKERMODSLAB-DRTC CONNECTED
MAKERMODSLAB-DRTC ACTIVE operator=policy
MAKERMODSLAB-DRTC STATS {"t":1,"chunks":3,"reqs":4,...}
MAKERMODSLAB-DRTC RETURNING
MAKERMODSLAB-DRTC ERROR <message, whitespace collapsed to one line>
MAKERMODSLAB-DRTC BYE
```

`READY` is emitted **before the bus is opened** and echoes the EFFECTIVE url and
room the child resolved — not what the parent believes it passed — so a
transport mismatch (or an SFU restarted between preflight and spawn) is caught
before anything is energized.

`STATS` goes out once a second **beside** the human `[robot]` line, which stays:
it is the artifact that made the first live runs diagnosable, and it tees into
the parent's log file the way `rollout.py`'s `_pump_stdout` tees lerobot's.
Every key in `drtc_protocol.STATS_KEYS` is always present, `null` where unknown,
so a typed status model can be exact instead of `exclude_none`. It is serialized
with `json.dumps(..., separators=(",", ":"))` — `format_event` collapses
whitespace in the payload, which is lossless for compact JSON only, and
`tests/test_drtc_protocol.py` pins that.

### Two regimes — pick by policy type

| Regime                     | For                                             | Robot entrypoint | GPU server                        | How it stays smooth                                                                                                                                                                            |
| -------------------------- | ----------------------------------------------- | ---------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **adaptive-sync**          | ANY policy, especially **non-inpainting** (ACT) | `robot_sync`     | `policy` / `modal_policy`         | Plays each chunk to completion (one seam per boundary); the prefetch lead scales with measured round-trip latency so the next chunk lands before the current drains. Never re-plans mid-chunk. |
| **full DRTC + inpainting** | **flow/diffusion** (smolvla, pi0, pi05)         | `robot_rtc`      | `policy_rtc` / `modal_policy_rtc` | Ships the still-to-execute prefix and the inference delay so the server guides denoising — overlapping chunks are dynamically consistent, no hard seams.                                       |

## Local SFU (`tools/drtc/`)

Both scripts run a LiveKit server on this machine and expose only its
**signaling** endpoint to Modal; WebRTC media and data channels always
hole-punch straight to this machine's public IP on UDP 7882. If your NAT defeats
hole punching, forward UDP 7882 here.

```bash
tools/drtc/local_sfu.sh      # signaling over a Cloudflare quick tunnel
tools/drtc/local_sfu_ts.sh   # signaling over your tailnet (stable URL, private)
```

Both write their state beside the rest of the Lab's persistent state, **not**
next to the script, so the robot can be started from any directory:

- `~/.cache/huggingface/lerobot/livekit.local.yaml` — the SFU config with a
  random local API key/secret (generated once; delete to rotate).
- `~/.cache/huggingface/lerobot/livekit.local.env` — the robot-side override
  (source 3 above). **Delete it to return to LiveKit Cloud.** It outlives the
  script: after Ctrl-C the robot keeps dialing `ws://127.0.0.1:7880` and gets
  "connection refused" until the script is restarted or the file removed.
- `~/.cache/huggingface/lerobot/logs/drtc/` — `livekit-server` / `cloudflared`
  logs.

Each script prints the exact `modal run ... --livekit-url ... --livekit-api-key
... --livekit-api-secret ...` line to paste. The Cloudflare quick tunnel gets a
**new random hostname every launch**, so those flags must be re-copied each
time; the tailnet URL is stable. The tailnet variant also needs the
`tailscale-auth` Modal secret and Tailscale logged in on this machine.

## Transport probe

`transport_probe.py` verifies that a LiveKit path can carry the loop with **no
robot and no GPU** — a synthetic robot on one machine, an echo operator on the
other, so the reported `e2e` is pure transport with inference removed:

```bash
# machine A (the robot station; the SFU runs here)
python makermodslab/drtc/transport_probe.py robot --url ws://127.0.0.1:7880 \
    --api-key <key> --api-secret <secret>

# machine B (the would-be GPU peer), reaching the SFU over the tailnet
python makermodslab/drtc/transport_probe.py operator --url ws://100.x.y.z:7880 \
    --api-key <key> --api-secret <secret>
```

It is deliberately self-contained (no imports from this package), so the far end
needs only that one file plus
`pip install "livekit-portal==0.2.4" "livekit-api>=0.7" numpy`.

## Tests

```bash
pytest tests/test_drtc_schedule.py tests/test_drtc_env.py \
       tests/test_drtc_protocol.py tests/test_drtc_pose.py
```

- `tests/test_drtc_schedule.py` — the offline core: absolute-step alignment,
  last-write-wins merging, single-source prefix extraction, and the
  Jacobson-Karels estimator. It is the port of what used to be `_rtc.py`'s
  `__main__` self-test block.
- `tests/test_drtc_env.py` — credential precedence (the four sources above),
  parametrized over both `load_env` and `read_env`.
- `tests/test_drtc_protocol.py` — the line protocol: round-trip,
  prefix-anywhere, compact-JSON survival of the whitespace collapse, the
  always-present STATS key set, and the two-STOP rule.
- `tests/test_drtc_pose.py` — bus discovery, gripper exclusion, and the
  action→bus target mapping, with fake buses.

None needs LiveKit, hardware or a GPU, so all four run in ordinary CI without
the extra installed. The `return_to_rest_pose` poll loop itself is deliberately
NOT re-tested here — it settles for `RETURN_SETTLE_S` and is already covered by
the replay tests; driving it would mean sleeping.

The adaptive-sync simulation that produced `SYNC_RESULTS.md` is still a
`__main__` block, since it is a benchmark rather than a test:

```bash
python -m makermodslab.drtc._sync_player
```

## Done since the port (S3.1, 2026-09-02)

The design record is [`SLICE3.md`](SLICE3.md); this is what its S3.1 slice
landed, `robot_sync` only.

- **Return-to-rest on stop, for the SO-101** — see "Safe start and stop" above.
  Every exit path (STOP, Ctrl-C, duration elapsed, a crash) drives the arm back
  to its captured start pose before torque is released; a second STOP cuts the
  return short.
- **A first-action ease-in**, replacing the snap into the policy's first
  commanded pose.
- **A supervised stdin/stdout protocol** (`makermodslab/drtc_protocol.py`) with
  a 1 Hz machine-readable `STATS` line.
- **`--livekit_url` / `--livekit_room`** on `robot_sync`, **`--livekit-room`** on
  both Modal wrappers, and **`_env.read_env()`** for the long-lived server.

## Not yet done

- **The API surface is in.** S3.3 wired `remote_inference` into `sessions.py`
  (`STARTABLE_KINDS`, `_FOLLOWER_ONLY_KINDS`, `_OPTIONS_MODELS`,
  `_REQUEST_BUILDERS`, an EXPLICIT `_dispatch_start` branch above the replay
  fall-through, the `_dispatch_stop` arm the lease's expiry watchdog routes
  through, and `_held_by`), added `RemoteInferenceOptions` plus the exact
  status/transport response models to `schemas/sessions.py`, and registered
  three v1-only routes: `GET /api/v1/remote-inference-status`,
  `GET /api/v1/remote-inference/transport` and
  `POST /api/v1/remote-inference/clear-local-override`. Start and stop ride
  `POST /api/v1/sessions` (kind `remote_inference`) and
  `POST /api/v1/sessions/{id}/stop` like every other robot-driving kind, so no
  new start/stop verbs exist and the flat surface did not grow.
- **The UI is in.** S3.4 added a fourth Deploy run mode, "Run it remotely":
  the existing robot selector, checkpoint picker and camera bindings feed a
  remote run's options, with a compact transport group (horizon / fps / codec /
  duration) beside them. It generates the `modal run` line for the other
  terminal from those same values — the mitigation for the mismatch that
  Portal's schema fingerprint turns into a silently dropped stream — and shows
  a live status panel (phase ladder, operator, chunks, the lead-vs-margin
  bar, DEGRADE, chunk age, e2e p50/p95, rtt, and `holds` as a RATE) plus the
  transport read-out with the clear-local-override button. Everything lives in
  `frontend/src/components/remote-inference/`; the shared studio files carry
  only a run-mode entry, two guard flags and one mount point.
  - Known limitation: that block renders inside the Deploy panel's
    "a skill is selected" section, so a remote run started from another tab or
    through the API is not visible until a skill is picked here.
- **The Lab still does not launch Modal, and does not supervise the SFU.**
  Lifecycle option A: a human runs `modal run
makermodslab/drtc/modal_policy.py` in one terminal and (optionally)
  `tools/drtc/local_sfu_ts.sh` in another; the session VERIFIES both before it
  energizes anything and refuses with a coded `transport.*` otherwise. Option B
  (the Lab launches the GPU side) is S3.5; option C (a supervised local SFU) is
  S3.6, and only if earned.
- **`robot_rtc` is untouched.** It still calls `robot.disconnect()` straight out
  of its control loop, with no ease-in, no stdin protocol and no
  `--livekit_url`/`--livekit_room`. Slice 3 is adaptive-sync only; if the RTC
  regime is ever brought under a session it needs the same treatment.
- **No CAN-arm support here, and none planned in this slice.** `maker_follower`
  / `metal_follower` are not registered with draccus in either entrypoint, so
  `--robot.type=maker_follower` fails at CLI-parse time inside the child —
  after a parent would have claimed and preflighted the arm. S3.2 refuses both
  CAN arms and bimanual SO-101 synchronously and pre-spawn, via
  `arm_capabilities.supports_remote_inference` (bimanual gets the return but
  not the ease-in, so its first move would be a full-speed snap).
- **No `max_relative_target`.** The ease-in covers the entry jump; per-tick
  relative clamping during the run is still absent (as it is in
  `lerobot-rollout`, whose `max_relative_target` also defaults to `None`).
