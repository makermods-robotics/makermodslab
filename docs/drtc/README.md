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

Only `_env` is importable without the extra (it needs python-dotenv alone), so
the credential-precedence tests run in ordinary CI.

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

**`LIVEKIT_ROOM` has no CLI flag.** On the GPU side it comes _only_ from the
`LiveKit-cloud` secret, so it must equal the robot's `LIVEKIT_ROOM` in
`livekit.env`. `--livekit-url` / `--livekit-api-key` / `--livekit-api-secret`
_are_ per-run flags (that is how a run is pointed at a local SFU), but the room
is not — two peers in different rooms simply never see each other.

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
pytest tests/test_drtc_schedule.py tests/test_drtc_env.py
```

- `tests/test_drtc_schedule.py` — the offline core: absolute-step alignment,
  last-write-wins merging, single-source prefix extraction, and the
  Jacobson-Karels estimator. It is the port of what used to be `_rtc.py`'s
  `__main__` self-test block.
- `tests/test_drtc_env.py` — credential precedence (the four sources above).

Neither needs LiveKit, hardware or a GPU, so both run in ordinary CI without the
extra installed.

The adaptive-sync simulation that produced `SYNC_RESULTS.md` is still a
`__main__` block, since it is a benchmark rather than a test:

```bash
python -m makermodslab.drtc._sync_player
```

## Not yet done (this port deliberately stops here)

- **No API surface.** There is no session kind, no route, no mutex entry and no
  `session_events` emission. Nothing starts these entrypoints but a human at a
  shell. A robot-driving feature must add reciprocal checks against every
  existing feature, emit at its transitions, and join `STARTABLE_KINDS` — see
  the state-model section of the root `CLAUDE.md`.
- **No return-to-rest on stop.** Both entrypoints call `robot.disconnect()`
  straight out of the control loop. That is survivable for an SO-101 and is
  **not** safe for a CAN arm, which has no brakes and drops under gravity when
  torque is released anywhere but near its resting pose (`maker_rest_pose.py`).
  The registered robot types are currently SO-101/Koch/OMX only.
- **No startup ramp or `max_relative_target`.** The first `send_action` after
  connect goes straight to the policy's first action — the same snap-to-pose
  family of issue analysed for teleop/record on 2026-09-01.
