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
| Supervised-session glue shared by BOTH entrypoints        | `makermodslab/drtc/_session_glue.py`                                                       |
| The SFU itself (`makermodslab --sfu`)                     | `makermodslab/sfu.py`, `makermodslab/scripts/makermodslab.py`                              |
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

The extra is `livekit-portal==0.2.4` and `python-dotenv>=1`. (`livekit-api` used
to be here too; since the bundled SFU landed it is a CORE dependency — the
`/api/v1/sfu/token` route signs with it, and so does this session's room probe.)
`livekit-portal` is pinned **exactly** and must match the pin in the two Modal
wrappers' images. Portal fingerprints the wire schema and _silently drops_
packets whose fingerprint differs on the two peers, so a mismatch does not
raise — it presents as a healthy-looking session with 0 chunks and 0
observations.

Four modules are importable **without** the extra, which is what keeps their
tests in ordinary CI: `_env` (python-dotenv alone), `makermodslab/drtc_protocol.py`
(stdlib alone — it is the parent's half of the line protocol, and the parent must
never load the Portal dylib), `_pose` (lerobot's motors, a hard dependency), and
`_session_glue` (the same footprint as `_pose`; its two portal-typed helpers
take the portal object as an argument and only call methods on it).
Everything else in `makermodslab/drtc/` imports `livekit.portal` at module top.

## Transport

Two, and the Lab picks between them with no configuration:

### The Lab's own SFU (`makermodslab --sfu`) — the normal path

The Lab runs `livekit-server` itself, as a child of the launcher (not of the
app — `uvicorn --reload` restarts the app on every save and the SFU must
outlive that). When it is up, a remote-inference session takes **everything**
from it in-process and reads no credential file at all:

- the url the child dials — `ws://127.0.0.1:7880` (`sfu.local_url()`);
- the room — `mml-<instance id prefix>` (`sfu.default_room`), one per station;
- the child's **token**, signed here with the key file's secret and passed on
  its argv as `--livekit_token`. The child therefore never holds an API secret;
  the secret lives only in `~/.cache/huggingface/lerobot/livekit_keys.yaml`
  (mode 0600, minted on the first `--sfu` run — delete it to rotate).

```bash
makermodslab --dev --sfu --bind <tailnet-ip> --sfu-external-ip   # or without --dev
```

Three flags, three jobs:

- `--sfu` runs the server. Missing binary ⇒ a one-line exit with the per-OS
  install hint (`brew install livekit` on macOS).
- `--bind <tailnet-ip>` puts **signalling** on an address a Modal container can
  reach. Without it the SFU binds loopback and nothing outside this machine can
  say hello. **In `--dev` mode `--bind` applies to the SFU alone** — Vite serves
  localhost only and uvicorn follows it, but the SFU is not a web server for
  your browser, and a loopback bind is what used to make a dev session
  LiveKit-Cloud-only.
- `--sfu-external-ip` lets the SFU STUN-discover this machine's public IP and
  advertise `<public>:7882` as an ICE candidate, instead of pinning the bound
  address. **Media is a separate problem from signalling**: a container reaches
  signalling over the tailnet but has to hole-punch for the video and action
  streams, and it has no route to a tailnet address. Without this flag you get
  a session that connects and receives nothing. It costs a STUN round trip at
  startup (so it is off by default — it stalls a station with no internet) and
  needs UDP 7882 reachable here; forward it if your NAT defeats hole punching.

The Deploy panel's transport section reports all of it — whether the SFU is
running, the key ID, the file its secret is in, whether the public media
address is advertised, and the **tailnet** URL a container should dial — and
folds them into the `modal run` line it generates. Copy that line; the only
thing to fill in by hand is the secret.

### LiveKit Cloud — the fallback

With no local SFU, `LIVEKIT_URL`, `LIVEKIT_ROOM`, `LIVEKIT_API_KEY` and
`LIVEKIT_API_SECRET` are read by `_env.read_env()` from **two** sources:

1. `~/.cache/huggingface/lerobot/livekit.env` (`config.DRTC_ENV_PATH`) — the
   saved credentials, beside the rest of the Lab's persistent state, so a wheel
   install and a source checkout read the same file. Start from
   [`livekit.env.example`](livekit.env.example).
2. The process environment, which wins.

Three rungs were **retired in S3.6** with the shell SFU scripts: a cwd `.env`, a
cwd `.env.local`, and `livekit.local.env`. All three were cwd-relative or
`override=True`; a cwd-relative source makes the answer depend on where the Lab
was started (two "connection refused" false starts on 2026-09-02), and an
override a long-lived process cannot un-set is a transport that survives the
deletion of the file naming it.

### `read_env` vs `load_env`

Two entry points, one precedence implementation:

- **`load_env()`** resolves the sources and writes them into `os.environ`. For
  the CLI entrypoints, whose downstream code (`_common.mint_token`, `policy.py`)
  reads credentials from the environment.
- **`read_env() -> dict`** resolves the same sources onto a **copy** of the
  process environment and hands it back, mutating nothing. `load_env` is
  implemented on top of it.

Use `read_env` from anything long-lived: it re-resolves from disk on every call,
so an edited file takes effect without a restart. Both are parametrized over the
same cases in `tests/test_drtc_env.py`.

One deliberate narrowing: `${VAR}` interpolation inside the file resolves
against the process environment only. Nothing in `livekit.env.example`
interpolates.

## GPU side, on Modal

**The Lab can now launch this for you.** Since S3.8 the remote-inference panel
has a "Start GPU" button (`makermodslab/modal_launcher.py`, three routes under
`/api/v1/remote-inference/gpu`): it finds the `modal` CLI on PATH, resolves the
room and credentials through the SAME function the session's preflight uses,
and runs the command below itself — attached, so stopping it stops the app. The
GPU is a Lab-level resource there, not part of the session: it does not hold the
arm, stopping a session does not stop it, and it stops itself after ten idle
minutes because a ready A100 is billing. Everything in this section stays true
and stays supported — it is the route when `modal` is missing or not signed in,
when you want `--detach` or a hand-tuned flag, and it is the line to compare
against when a run connects but receives nothing.

One thing the Lab-launched path does differently: it passes the API key and
secret in the CHILD'S ENVIRONMENT rather than as flags, because a
`@local_entrypoint` body runs on your machine and `--livekit-api-secret` would
put a signing key in `ps`. Both wrappers' `main()` falls back to
`LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` for that reason; the flags still win
when present, so every command in this file is unchanged.

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
| `tailscale-auth` | `TS_AUTHKEY` (REUSABLE, **non-ephemeral**)                              | `--tailscale` only                                                  |

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

### One tailnet node, not one per launch

The GPU joins the tailnet as a single node called `modal-policy`. Tailscale
identifies a node by its **node key**, which lives in tailscaled's state file,
so the mechanism is simply that the state file outlives the container: a
dedicated Modal Volume, `makermodslab-tailscale-state`, mounted at `/tailscale`,
and `tailscaled --state=/tailscale/tailscaled.state`. It is deliberately NOT the
`hf-cache` Volume — a 4 KB state commit should not re-commit a 20 GB model
cache. The Volume is created on first use (`create_if_missing=True`); there is
nothing to set up by hand.

Login goes: state file present → `tailscale up --hostname=modal-policy` with no
`--auth-key`, bounded to 20 s, which re-uses the saved node key; anything other
than success (expired node key, a `NeedsLogin` that printed a login URL and then
timed out — the wrapper does not try to tell those apart) falls back to the
`--auth-key` path, which is also what a first-ever run takes. The state is
committed once the backend reports Running and again on the way out, so a
rotated node key is kept.

Two rules follow from that, and the first one is a change:

- **The auth key must be REUSABLE and non-ephemeral.** Until 2026-09-04 this
  page asked for REUSABLE + EPHEMERAL, which was right when the wrapper ran
  `--state=mem:` and kept nothing. It is wrong now: the control plane *deletes*
  an ephemeral node as soon as it goes offline, so the persisted node key is
  dead on the next launch and the container re-registers as a new node. With an
  ephemeral key you still get one node while the GPU stays online — and a new
  one after every gap, which is the 21-rows-in-the-console problem again. Mint a
  fresh key with **Ephemeral** unticked and **Reusable** ticked, then
  `modal secret create tailscale-auth TS_AUTHKEY=tskey-auth-... --force`.
- **Two containers sharing the state log in as the same node**, and the later
  `tailscale up` displaces the earlier one from the tailnet. The Lab's
  one-launcher gate and its orphan reaper make that rare, and
  `DRTC_TS_EPHEMERAL=1` is the escape hatch for a run you deliberately want
  beside another: it restores `--state=mem:` and always-`--auth-key`, i.e. the
  pre-2026-09-04 behaviour, and gets its own throwaway `modal-policy-N` row.
  Set it in your own shell — `DRTC_TS_EPHEMERAL=1 modal run …` — the wrapper's
  `main()` reads it there and forwards it to the container as a kwarg (Modal
  does not ship the caller's environment). Nothing in the Lab's launcher sets
  it.

The 21 `modal-policy-N` rows a pre-2026-09-04 tailnet accumulated can simply be
deleted in the admin console; nothing references them.

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

The GPU image pins **the same lerobot SHA as `pyproject.toml`** — the Lab's
`makermods-robotics/lerobot` fork, with the extras `[pi,smolvla,molmoact2]`. It
used to pin upstream `huggingface/lerobot@8414188`; that changed in S3.7a for
two reasons. Upstream `8414188` is lerobot 0.5.2 and has no `supports_rtc`
anywhere in the tree, so the RTC server cannot ask a policy whether in-painting
is possible and MolmoAct2 cannot be served at all. And the fork is what the Lab
itself trains on — config compatibility follows the lerobot that WROTE a
checkpoint, so a GPU on a different lerobot is the very hazard the old pin
comment warned about, pointed the other way.

**The two pins move together or not at all.**
`tests/test_drtc_modal_wrappers.py` reads `pyproject.toml` and both wrappers as
text and asserts one URL, one SHA, one extras list and one `livekit-portal`
version across all three. Bumping the Lab's lerobot means editing three files,
and per CLAUDE.md it is a real bump: re-run a known-good SmolVLA RTC session on
the new image before declaring it good.

### MolmoAct2

Supported since S3.7a, on both servers. It is a ~7B vision-language policy, and
almost everything specific to it is about size and about the task string.

The public checkpoint is **`lerobot/MolmoAct2-SO100_101-LeRobot`** (ungated):

| What                    | Value                                          |
| ----------------------- | ---------------------------------------------- |
| cameras                 | `cam0`, `cam1`, both 3×224×224                 |
| state / action          | 6 / 6 (single SO-101 arm)                      |
| `n_action_steps`        | **30** — so `--horizon 30`, not the default 50 |
| `chunk_size`            | 30                                             |
| `inference_action_mode` | `continuous` → `supports_rtc()` is True        |
| `model_dtype`           | `float32`                                      |

**`--task` is REQUIRED, and the server refuses without it.** Not a style
preference: with no task MolmoAct2 does not fail, it renders the missing string
into a fixed template and prompts its VLM with the literal `"The task is to ."`,
then returns confidently wrong actions with nothing in any log to explain it.
Phrase the task as an infinitive completion — `--task "pick up the block and put
it in the box"`. The same rule now covers smolvla / pi0 / pi0_fast / pi05, from
one vocabulary (`makermodslab.utils.system.policy_requires_task`) that the Lab's
`requires_task` reads too.

**`--horizon` must be 30.** `predict_action_chunk` returns `n_action_steps`
steps; declaring 50 makes the two Portal peers disagree about the action-chunk
shape, the fingerprint stops matching, and every packet is dropped **in
silence** — a connected session that transfers nothing. `GET
…/policy-config` now reports `n_action_steps` so a launcher can stop guessing.

**Cold start downloads 21.8 GB.** The config's `checkpoint_path` is
`allenai/MolmoAct2-SO100_101` (5 safetensors shards), pulled into the `hf-cache`
Volume at `/cache/huggingface` — once, then every later run reads the Volume.
Budget several minutes before the first log line past model load, and note the
download starts EARLY: `_apply_norm_tag_metadata` snapshot-downloads the repo
before `_load_hf_model` does, just to read a 4.8 KB `norm_stats.json`. The image
itself grows only ~40 MB (peft + scipy).

**`model_dtype` is `float32` on a 40 GB `A100` — measure before changing it.**
~7B in fp32 is ~28 GB of weights before activations, and `predict_action_chunk`
only applies `autocast` for bf16/fp16, so fp32 runs full fp32 compute. If it
OOMs or cannot keep up, the operator's opt-in is:

```bash
modal run makermodslab/drtc/modal_policy_rtc.py \
    --policy-path lerobot/MolmoAct2-SO100_101-LeRobot \
    --task "pick up the block and put it in the box" \
    --horizon 30 --fps 30 --s-min 4 --video-codec H264 \
    --model-dtype bfloat16 \
    --livekit-room <the room the panel shows>
```

`--model-dtype` (float32 | bfloat16 | float16) is on BOTH wrappers and both
servers, defaults to unset, and when unset the checkpoint's saved value is used
exactly as saved. When set it replaces that value **before weights load** and
logs the override; the Lab never changes it silently. `gpu="A100-80GB"` is the
other lever — it respects the checkpoint but bills more on every run, including
the ones that did not need it.

**A discrete-mode checkpoint degrades instead of raising.** `enable_rtc()` now
consults `supports_rtc()` (MolmoAct2's is literally `inference_action_mode ==
"continuous"`), so a discrete checkpoint serves plain chunks with a logged
reason rather than reporting `RTC ENABLED` and then raising on every inference.

**Not yet: starting one from the UI.** The checkpoint's cameras are `cam0` /
`cam1`, no robot record has cameras by those names, and the Deploy panel binds
checkpoint cameras to robot cameras purely by name match with no manual picker —
so every binding lands in `unmatchedCameras` and Start stays disabled. Launch it
by hand (or from the panel's generated command line) until S3.7b lands the
per-role picker.

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

### Safe start and stop (BOTH entrypoints, SO-101)

Neither entrypoint snaps into the policy's first action, and neither drops the
arm at the end. Both behaviours are on by default; turning one off is for bench
A/B only. S3.1 built this for `robot_sync`; S3.5 lifted it into
`makermodslab/drtc/_session_glue.py` and wired `robot_rtc` to the same code —
shared, not copied, because two divergent copies of the logic that makes an
energized arm safe is exactly the bug worth designing out.

Two things deliberately stayed written out in each entrypoint rather than moving
into the glue: **the teardown's call sequence** and the
`reset_torque_limit(robot, FOLLOWER)` line. Both are pinned by source-level
assertions (`tests/test_drtc_robot_sync.py`, `tests/test_drtc_robot_rtc.py`)
that read the `finally:` block and check every step goes through `shielded` —
the only guard those paths have, since they only run with a real arm attached.
Hiding the sequence behind one helper call would have retired it on both engines
at once. A third test asserts the two sequences are IDENTICAL, so they cannot
drift.

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
| `--livekit_token`  | (mint)  | Join with a token the parent already signed, instead of minting one from `LIVEKIT_API_KEY`/`SECRET`. What the Lab passes under `--sfu`, so the child holds no API secret.    |

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

### Two regimes — both are session ENGINES, picked by policy type

Since S3.5 the two are not "the supported one and the bench script": they are
the `engine` option on a `remote_inference` session (`sync` / `rtc`), they run
the same session glue, and the Deploy panel picks the default from the
checkpoint's `policy_type`.

| Regime                     | `engine` | For                                                | Robot entrypoint | GPU server                        | How it stays smooth                                                                                                                                                                            |
| -------------------------- | -------- | -------------------------------------------------- | ---------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **adaptive-sync**          | `sync`   | ANY policy, especially **non-inpainting** (ACT)    | `robot_sync`     | `policy` / `modal_policy`         | Plays each chunk to completion (one seam per boundary); the prefetch lead scales with measured round-trip latency so the next chunk lands before the current drains. Never re-plans mid-chunk. |
| **full DRTC + inpainting** | `rtc`    | **flow/diffusion** (smolvla, pi0, pi05, diffusion) | `robot_rtc`      | `policy_rtc` / `modal_policy_rtc` | Ships the still-to-execute prefix and the inference delay so the server guides denoising — overlapping chunks are dynamically consistent, no hard seams.                                       |

**How to pick.** By policy family, and the UI does it for you: flow/diffusion
checkpoints default to `rtc` at horizon 50, everything else to `sync` at
horizon 16. Choosing `rtc` for a non-flow policy is BLOCKED client-side
(`deployGuards.remoteEngineSupported`) — and it has to be client-side, because
the backend never loads the checkpoint and therefore cannot tell the two apart.

**Why it matters even on a healthy transport** (bench, 2026-09-03): a SmolVLA
eraser-place run at horizon 48 held a flat `sync` transport — no DEGRADE,
e2e ~400 ms of which ~280 ms was inference — and the arm was still visibly jerky
at ~1 Hz. 33 chunks in 29 s: with ~400 ms latency the adaptive-sync player
aligns each arriving chunk by dropping its stale prefix, so the plan switches
every ~0.9 s, and two flow-policy plans made 400 ms apart disagree at every
seam. Nothing about the transport is wrong; the regime is. ACT at 80-110 ms e2e
never shows it, which is why `sync` remains the right default for everything
else.

**Both sides must agree on `s_min`, not just on horizon/fps/codec.** The robot
computes `overlap_end = H - max(s_min, d)` per request and ships it; the server
trusts that number and falls back to its own `H - s_min` only when the field is
absent. Two different values put the in-painting mask on a different boundary
than the guidance. Both default to 4, and the panel emits `--s-min` into the
generated `modal run` line from the same field the session is started with.

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
`pip install "livekit-portal==0.2.4" "livekit-api>=1.0" numpy`.

## Tests

```bash
pytest tests/test_drtc_schedule.py tests/test_drtc_env.py \
       tests/test_drtc_protocol.py tests/test_drtc_pose.py \
       tests/test_drtc_robot_sync.py tests/test_drtc_robot_rtc.py \
       tests/test_drtc_modal_wrappers.py
```

- `tests/test_drtc_schedule.py` — the offline core: absolute-step alignment,
  last-write-wins merging, single-source prefix extraction, and the
  Jacobson-Karels estimator. It is the port of what used to be `_rtc.py`'s
  `__main__` self-test block.
- `tests/test_drtc_env.py` — credential precedence (the two sources above, and
  that the three retired ones are no longer read), parametrized over both
  `load_env` and `read_env`.
- `tests/test_drtc_protocol.py` — the line protocol: round-trip,
  prefix-anywhere, compact-JSON survival of the whitespace collapse, the
  always-present STATS key set, and the two-STOP rule.
- `tests/test_drtc_pose.py` — bus discovery, gripper exclusion, and the
  action→bus target mapping, with fake buses.
- `tests/test_drtc_robot_sync.py` / `tests/test_drtc_robot_rtc.py` — the
  interrupt shield, plus the two SOURCE-level assertions per entrypoint that no
  runtime test can reach: every teardown step goes through `shielded`, and the
  torque cap is cleared right after `connect()`. The rtc file adds two more —
  that the two teardown sequences are IDENTICAL, and that neither entrypoint
  redefines a piece of `_session_glue` locally. The shield's own unit tests
  `importorskip` the extra; everything read off the source runs everywhere,
  which matters because those are the halves a refactor drops.
- `tests/test_drtc_modal_wrappers.py` — the GPU image's pins equal the Lab's
  own: one lerobot fork URL + SHA, one extras list including `molmoact2`, and
  one `livekit-portal` version across `pyproject.toml` and BOTH wrappers, plus
  that `--model-dtype` is forwarded end to end. Reads the files as TEXT, because
  the wrappers import `modal` at their top level and are not importable in CI —
  and because the pins are string literals, which is what a hand-edit breaks.

None needs LiveKit, hardware or a GPU, so all seven run in ordinary CI without
the extra installed. The `return_to_rest_pose` poll loop itself is deliberately
NOT re-tested here — it settles for `RETURN_SETTLE_S` and is already covered by
the replay tests; driving it would mean sleeping.

The adaptive-sync simulation that produced `SYNC_RESULTS.md` is still a
`__main__` block, since it is a benchmark rather than a test:

```bash
python -m makermodslab.drtc._sync_player
```

## Done since the port

The design record is [`SLICE3.md`](SLICE3.md).

### S3.7a (2026-09-03) — MolmoAct2, and the GPU image joins the Lab's pin

The backend half. Both wrapper images moved from upstream `lerobot@8414188`
(0.5.2, no `supports_rtc`) to the SAME fork SHA `pyproject.toml` pins, with
`[pi,smolvla,molmoact2]`; `tests/test_drtc_modal_wrappers.py` now asserts the
two pins are equal so they cannot drift again. Both servers gained an opt-in
`--model-dtype` applied BEFORE weights load, refuse to start when a
language-conditioned policy has no `--task` (one vocabulary shared with the
Lab's `requires_task`, which gained `molmoact2`), and fill a MolmoAct2
`inference_action_mode` only when the checkpoint saved none.
`policy_rtc.enable_rtc()` now consults the policy's own `supports_rtc()`, so a
discrete checkpoint degrades to plain chunks instead of raising once per
inference. `…/policy-config` reports `n_action_steps` and `chunk_size`. See the
MolmoAct2 section above for what an actual run needs; the UI half (camera role
binding) is S3.7b.

### S3.8 (2026-09-03) — the Lab launches the GPU

- **`makermodslab/modal_launcher.py`** plus three v1 routes
  (`POST /api/v1/remote-inference/gpu/start`, `POST …/gpu/stop`,
  `GET …/gpu`): the Lab shells out to the `modal` CLI, **attached** (no
  `--detach`), so killing the process group stops the app — and the GPU dies
  with the Lab, which is fine because the session does too.
- **A LAB-LEVEL resource, not a session field.** It holds no hardware, so it
  gets its own verbs: a `launch_gpu` option would hold
  `robot.busy.remote_inference` for a 1-3 minute cold start while the arm sat
  completely free. Its exit never stops a session; a session's stop (including
  a lease expiry) never stops it; `SESSION_KINDS` / `STARTABLE_KINDS` /
  `_dispatch_*` are untouched. See SLICE3.md "S3.8 as built" for the five
  arguments.
- **One transport resolver.** `remote_inference.resolve_transport()` is now the
  only credential path, shared by the session's preflight, the transport
  endpoint and the launcher — the two halves in different rooms is invisible by
  construction, so a second path is the bug rather than a smell.
- **The secret is never in argv.** Both wrappers' `main()` falls back to
  `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`, and the launcher passes them in the
  child's environment. The flags still win when present.
- **Two deadlines, no new thread:** a 300 s cold-start bound (its message names
  the last phase reached) and a 600 s idle auto-stop measured from `ready` or
  the last session's end, whichever is later. The panel says a ready GPU is
  billing and counts down. Both are checked from the log pump and the status
  poll.
- **Readiness is a hint.** `state: "ready"` comes from `[policy] connected as`
  in the container's stdout — never `claimed control as`, which a healthy run
  may never print. The gate on energizing the arm is still `_probe_room`.
- New `gpu.*` error domain (`cli_missing`, `unauthenticated`, `already_running`,
  `not_running`, `launch_failed`); `V1_ONLY_ROUTES` +3; the Lab still never
  reads `~/.modal.toml`.

### S3.6 (2026-09-03) — the Lab-owned SFU

- **`makermodslab --sfu` is the local-SFU story**, merged from
  `feat/livekit-sfu`: `sfu.py` (binary lookup, config rendering, the token
  broker) plus `POST /api/v1/sfu/token`. The two shell scripts under
  `tools/drtc/` are deleted.
- **The session adopts it.** With the SFU up it takes the url, the room and the
  child's TOKEN in-process and reads no credential file; the child gets
  `--livekit_token` and never holds an API secret. The Portal identities stay
  exactly `robot` and `policy` — the room probe looks for the latter by name.
- **`--sfu-external-ip`** (new, default off): the SFU advertises its
  STUN-discovered public IP instead of the pinned bind address, which is the
  only way a Modal container can hole-punch media to this machine.
- **The transport surface reshaped.** `source` is now
  `sfu | cloud | process_env | none`, the panel reports the SFU's state and the
  tailnet URL a container should dial, and the `modal run` line carries the
  whole transport. `clear-local-override` and the three retired credential
  rungs are gone.

### S3.5 (2026-09-03) — the RTC regime becomes an engine

- **`robot_rtc` hardened to `robot_sync`'s standard, on shared code.** New
  `makermodslab/drtc/_session_glue.py` owns the event emitters, the stdin
  command pump, the start-pose capture, the first-action ease-in, the interrupt
  shield and the four transport/safety flags; both entrypoints import it. It is
  importable WITHOUT the `drtc` extra, same rule as `_pose`.
- **`engine` + `s_min` on `RemoteInferenceOptions`.** `engine` picks the child
  module; `s_min` is sent only for `rtc`, where it is half a contract with the
  GPU side. `engine` is on the status payload too.
- **Two display fixes.** `elapsed_s` is FROZEN at the exit instead of reset to
  0 (a finished run reported "0s / 60", reading as one that never started), and
  the status panel keeps the last `holds` rate after a run ends instead of
  blanking it to "—" at exactly the moment a failed run needs it.
- **The UI picks the engine from the checkpoint** and blocks `rtc` for a
  non-flow policy, because the backend cannot make that check itself.

### S3.1 (2026-09-02) — `robot_sync` hardening

What that slice landed, `robot_sync` only (all of it is now shared):

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
  two v1-only routes: `GET /api/v1/remote-inference-status` and
  `GET /api/v1/remote-inference/transport` (a third, `POST
/api/v1/remote-inference/clear-local-override`, retired in S3.6 with the
  scripts whose dotenv override it deleted). Start and stop ride
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
  transport read-out. Everything lives in
  `frontend/src/components/remote-inference/`; the shared studio files carry
  only a run-mode entry, two guard flags and one mount point.
  Since the studio-panels rework merged, that block sits inside the Deploy
  panel's run form, which is held open for as long as a remote run is live —
  so a run started from another tab or through the API shows its status and
  its Stop as soon as this panel renders. (Before the rework it was gated on
  a policy being selected here, and was invisible until one was.)
- **The Lab owns the SFU, and now the GPU launch too.** S3.6 adopted
  the bundled `livekit-server` (`makermodslab --sfu`) and its token broker, so
  lifecycle option C is done: the SFU is a launcher child, not a session, and
  the session mints the robot's token from it. The GPU half then landed as
  S3.8 (above), and deliberately BESIDE the session rather than inside it —
  the session still owns only the robot side, and still VERIFIES an operator
  is in the room before it energizes anything, refusing with a coded
  `transport.*` otherwise. Running `modal run makermodslab/drtc/modal_policy.py`
  by hand in another terminal remains fully supported.
- **MolmoAct2 cannot be STARTED from the UI yet.** The backend is in (S3.7a,
  above), but the published checkpoint's cameras are `cam0` / `cam1` and the
  Deploy panel binds checkpoint cameras to robot cameras by NAME MATCH with no
  manual picker — every binding lands in `unmatchedCameras` and Start stays
  disabled. Launch it by hand, or from the panel's generated command line, until
  S3.7b adds the per-role picker. The panel also still defaults the RTC horizon
  to 50, where this checkpoint needs 30; `…/policy-config` now carries
  `n_action_steps` so S3.7b can read it instead of the operator typing it.
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
