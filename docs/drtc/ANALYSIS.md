# DRTC over Portal — Analysis & Context Brief

> **Purpose.** A durable "regain-context-fast" brief for `lerobot_drtc_full/`. Read
> this cold and you should understand: what DRTC is, what the reference
> implementation actually does, what our first Portal imitation
> (`../lerobot_drtc/`) got right and wrong, why it "doesn't work well," and what
> now lives in this directory. Claims are cited to `file:line`; every citation
> below was verified against the source, not remembered.
>
> Reference implementation studied locally under
> `remote_inference_test/drtc/` (Jack Vial's DRTC:
> <https://jackvial.com/posts/distributed-real-time-chunking.html>). Local
> lerobot with RTC support under `makermods/lerobot/`.

---

## 1. DRTC is two separable halves

The single most important framing. DRTC is **not** one monolithic algorithm; it
is two layers stacked, and the lower one is a strict prerequisite of the upper.

**(a) The policy-agnostic scheduling / transport half.** Latency estimation,
execution-horizon control, absolute-step chunk alignment, and last-write-wins /
CRDT fault handling. It decides _when_ to request inference, _where_ on a shared
timeline an arriving chunk's actions land, and _which_ action wins when two
chunks overlap. It works for **any** action-chunking policy — ACT, diffusion,
pi0, SmolVLA — because it never looks inside the model. The reference paper's
resilience claims (recovery from lost/reordered/duplicated messages, monotone
dataflow) all live here.

**(b) The flow/diffusion-only guided-generation half.** _Real-Time Chunking
(RTC) in-painting_ performed **inside the policy's denoising loop**: the previous
chunk's overlapping actions are fed back as a soft/hard mask so the newly
generated chunk is one the policy would actually have produced given that
prefix. The stitched trajectory is _dynamically consistent_ — no jerk at the
seam. This only exists for flow/diffusion policies because it manipulates the
velocity field of the sampler (`drtc/.../rtc_guidance.py`).

**Half (a) is a prerequisite for (b).** You cannot do guided in-painting until
you already know the freeze index `d`, the overlap prefix, and which absolute
steps the prefix occupies — all of which are outputs of half (a). Half (a) also
delivers real wins on its own (alignment, pacing, fault tolerance), which is why
our first cut built it first.

---

## 2. What the reference DRTC provides

### 2.1 Congestion / pacing — latency-adaptive cooldown gated by backpressure

The client (`drtc/src/lerobot/async_inference/robot_client_drtc.py`) runs a
control loop at `fps`. It only requests a fresh inference when **both** a
schedule-depth backpressure gate and a cooldown counter allow it:

```
trigger_threshold = H - s_min                                   # robot_client_drtc.py:1104
should_trigger    = schedule_size <= trigger_threshold          # :1106
                    and obs_cooldown == 0
```

On firing, cooldown is re-armed to the latency estimate plus a margin:

```
obs_cooldown = latency_steps + epsilon                          # robot_client_drtc.py:1153
```

and decremented once per tick otherwise (`max(obs_cooldown - 1, 0)`, `:1174`).
The cooldown is **seeded** at startup to `s_min + epsilon` (`:502`) so the gate
works before the first RTT sample. The important property: the countdown is
purely local and monotone, so **a lost action chunk cannot stall the client** —
cooldown still elapses and a new request fires (this is the paper's "recovery
from lost/delayed messages").

### 2.2 Latency estimator — Jacobson–Karels (RFC 6298), clamped to `H/2`

`make_latency_estimator(...)` (`robot_client_drtc.py:364`) builds the JK
estimator (the same RTT smoother TCP uses):

```
first sample:  smoothed = measured;  deviation = 0
thereafter:    error     = measured - smoothed
               smoothed  = (1-α)·smoothed + α·measured          # α = 0.125
               deviation = (1-β)·deviation + β·|error|          # β = 0.25
estimate_s     = smoothed + k·deviation                         # k = 1.5
estimate_steps = clamp(⌈estimate_s · fps⌉, 1, H//2)
```

The `H//2` ceiling is the RTC constraint: with execution horizon `s = d`, the
frozen/overlap prefix `d ≤ H - s` collapses to `d ≤ H/2`. Beyond that a chunk is
too stale to be worth stitching.

### 2.3 Fault tolerance — LWW / CRDT registers with two clocks

Thread hand-offs and schedule merges are modeled as **Last-Write-Wins registers**
(`drtc/.../lww_register.py`). The join keeps the larger `control_step`:

```python
def __or__(self, other):                                        # lww_register.py:32
    if other.control_step > self.control_step:                  # :38  strict >
        return other
    return self
```

`update_if_newer` (`:102`) applies this monotonically, so **stale or duplicate
updates cannot overwrite** a fresher value. Two clocks are used deliberately
(`lww_register.py:11-14`, `robot_client_drtc.py:354-360`):

| clock          | symbol | role                                                                                                                                                               |
| -------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `control_step` | `t`    | monotone per control-loop tick; the **freshness / ordering** key for LWW and watermarks. Increments even when no action executes, so drops never stall watermarks. |
| `action_step`  | `n`    | the **execution index** on the action timeline; where a chunk's actions physically land.                                                                           |

The schedule merge (`ActionSchedule.merge`, `robot_client_drtc.py:169`) uses the
same discipline: drop actions whose step `≤ current_action_step` (`:202`), insert
into empty steps, and on an occupied step overwrite **only if** the incoming
`src_control_step` is strictly greater (`:222`). Reordered/duplicate/stale chunks
are absorbed; a strictly fresher observation always wins.

### 2.4 In-painting — frozen / soft-mask / fresh regions + guidance weight

`drtc/.../rtc_guidance.py` (`AsyncRTCProcessor.denoise_step`, `:72`) wraps the
model's denoiser. It builds a per-timestep weight vector over the chunk
(`_get_prefix_weights`, `:217`):

| region    | steps              | weight                | meaning                                                                     |
| --------- | ------------------ | --------------------- | --------------------------------------------------------------------------- |
| frozen    | `[0, d)`           | `1.0`                 | already committed / being executed during inference; hard-pin to the prefix |
| soft mask | `[d, overlap_end)` | decaying (linear/exp) | blend toward the prefix, decaying to 0                                      |
| fresh     | `[overlap_end, H)` | `0.0`                 | fully policy-generated                                                      |

The correction is a guided-denoising term. Base velocity `v_t` is computed, an
error `err = (prev - x1_t) * weights` is formed, and the guided velocity is
`v_t - guidance_weight · correction` (`rtc_guidance.py:178-212`), where the
guidance weight follows the RTC/Alex-Soare formula with prior variance `σ_d`:

```
inv_r2          = ((1-τ)² + τ²·σ_d²) / ((1-τ)²·σ_d²)            # rtc_guidance.py:204
guidance_weight = min( ((1-τ)/τ) · inv_r2 , max_guidance_weight)  # :208-210
```

`full_trajectory_alignment` (`:183`) skips the autograd term and uses `err`
directly (faster / smoother when chunks are close).

### 2.5 Wire contract — what crosses the network

Observation carries (`robot_client_drtc.py`, sent via `__rtc__` at `:1134`):
`control_step` (t), `timestamp`, `chunk_start_step` (n*k), and an `__rtc__` dict
`{ latency_steps=d, overlap_end, action_schedule_spans }`. The spans are a list
of `(src_control_step, start_idx, end_idx)` produced by
`get_masking_chunk_spans` (`:98`) — they describe the prefix \_by reference*, not
by value.

The server (`policy_server_drtc.py`) caches **raw, pre-postprocess** action
chunks keyed by source `control_step` in an LRU `ActionChunkCache` (`:80`,
populated at `:765` _before_ postprocessing). On a fresh observation it
reconstructs the prefix tensor by concatenating cached slices per span
(`:699-737`), then calls the policy with RTC kwargs
(`inference_delay=d`, `prev_chunk_left_over=prefix_tensor`, `overlap_end`) at
`:738-744`. The returned dense chunk echoes `source_control_step`,
`chunk_start_step`, and server timestamps (`:818-827`) so the client can
correlate and measure latency. `Ready` resets all server session state
(`_reset_server`, `:224`).

Why raw chunks are cached (not post-processed): RTC guidance operates in **raw
model action space** (e.g. 32 padded dims) inside the denoise loop, while the
postprocessor maps to executable space (e.g. 6 dims). The two are dimensionally
incompatible; see the comment at `policy_server_drtc.py:388-392`.

---

## 3. What `../lerobot_drtc/` ported correctly (the policy-agnostic half)

Our first Portal example (`portal/examples/python/lerobot_drtc/`) implements
half (a) only, entirely on the robot, with **no protocol or `policy.py`
change**. It rides on one fact: the received `ActionChunk` already carries
`in_reply_to_ts_us`, the timestamp of the observation the policy answered
(`livekit/portal/__init__.py:256`; set by the policy via
`send_action_chunk(..., in_reply_to_ts_us=obs.timestamp_us)`,
`lerobot_drtc/policy.py:258`). What it got right:

- **LWW merge** (`_scheduler.py:166-200`). Keyed on the absolute step
  `chunk_start_step + j`; steps `< current_step` are frozen and dropped
  (`:181`); occupied steps overwrite only if the incoming observation is
  strictly fresher (`src_control_step >`, `:195`). This is a faithful port of
  the reference join (`robot_client_drtc.py:222`).
- **Alignment for free** (`_scheduler.py:166-189`, README §4). A chunk answering
  the observation from tick `t_src` is placed at `chunk_start_step =
t_src + action_delay`; by arrival its stale prefix is `< current_step` and gets
  dropped, so it resumes at "now." The dropped-prefix length _is_ the round-trip
  in ticks — never computed explicitly.
- **Latency-adaptive cooldown pacing** (`robot.py:234-256`). `should_request =
runway <= (H - s_min) and obs_cooldown == 0` (`:235`); on request,
  `obs_cooldown = estimate_steps + epsilon` (`:254`); else decrement (`:256`).
  Directly mirrors the reference gate + re-arm.
- **JK estimator** (`_scheduler.py:37-95`) with the `[1, H//2]` clamp
  (`estimate_steps`, `:86-95`) — matches §2.2.
- **Timestamp correlation** (`robot.py:173-194`). The robot records
  `sent_obs[ts_us] = control_step` (`:250`) and, on a chunk, looks up
  `t_src = sent_obs.get(reply_ts)` (`:180`). This is **verified correct** against
  the transport: Portal's `Observation.timestamp_us` is exactly the state's send
  timestamp — the synced observation inherits `ts` from the popped state buffer
  entry (`sync_buffer.rs:493` → `:525`). So the robot's `send_state(...,
timestamp_us=ts_us)` value round-trips as the observation timestamp the policy
  echoes in `in_reply_to_ts_us`, and the `sent_obs[ts] → control_step` map lines
  up.

---

## 4. Why it "doesn't work well" — ranked findings

These are the concrete reasons the first `lerobot_drtc/` cut underperforms.
Ranked by impact.

### #1 — No RTC in-painting; overlaps are a HARD SWITCH

The headline DRTC feature is simply **absent**. On overlap the scheduler keeps
the freshest action per step (`_scheduler.py:195-197`) — a hard switch, not a
blend. The comment says so plainly (`_scheduler.py:26-29`). `MergeStats.mean_l2`
(`_scheduler.py:112-124`, computed via `_l2` at `:132`) is **instrumentation
measuring the resulting discontinuity**, not a fix for it. For a mode-switching
flow/VLA policy, hard-switching between two chunks generated from different
observations reproduces _exactly_ the trajectory jerk that DRTC's in-painting
exists to remove. Half (a) can align and de-duplicate; it cannot make the seam
dynamically consistent.

### #2 — The pacing is inherited from the RTC/replan-often world and is actively wrong for a hard-switch scheduler

This is likely the **main observed problem**, and it is subtle. The cooldown
pacing was designed for a world where overlaps are _smoothed by in-painting_, so
requesting often is harmless (every seam is invisible). In our hard-switch
scheduler the incentives invert:

- low latency → short cooldown (`estimate_steps + epsilon`, `robot.py:254`) →
  frequent re-requests → **more overlapping chunks** → **more hard seams**.

The result is perverse: the "DRTC" robot can be **jerkier** than the simpler
`../lerobot_inference/robot.py` `BlockChunkPlayer`
(`lerobot_inference/robot.py:111-169`), which plays each chunk to completion and
only swaps at a true block boundary (`step`, `:144-151`; one request per block,
`should_request`, `:158-164`) — i.e. **one seam per boundary** instead of many.
Absent in-painting, "replan often" is the wrong regime; "replan rarely, play to
completion" produces fewer discontinuities.

### #3 — The wire protocol carries none of the RTC metadata

`lerobot_drtc/policy.py` just calls `predict_action_chunk(batch)`
(`policy.py:147`) with no RTC kwargs. None of `inference_delay`, `overlap_end`,
or the prefix/spans crosses the wire. So in-painting is **not even possible**
without protocol work — half (b) is blocked on transport, not just on the model.
(Feasibility of that transport work is discussed in §6.)

### #4 — Fault-tolerance gaps versus the reference

Half (a) ported the _happy-path_ resilience but not all failure handling:

- **No stall detection / escalation.** On a dead or silent server the robot
  holds `last_action` forever (`robot.py:228-229`) — position is held, but
  there is no timeout, no re-request escalation, no surfacing of the stall.
- **Uncorrelated-chunk fallback can promote a stale chunk to "fresh now."**
  When `reply_ts` matches no recorded observation, the fallback stamps
  `t_src = schedule.current_step` (`robot.py:181-186`). Because `sent_obs` is a
  FIFO evicted at `obs_ttl = 4*horizon` (`robot.py:163`, `:250-252`), a _late_
  chunk whose observation has already been evicted is treated as freshest-now
  and can overwrite genuinely fresher scheduled actions — the opposite of the
  LWW guarantee. (Counted as `uncorr` in the log.)
- **No server-side session reset on reconnect.** `policy.py` has no analogue of
  the reference `_reset_server` on `Ready`
  (`policy_server_drtc.py:224-259`); a fresh robot re-joining relies only on the
  operator re-claiming control (`policy.py:226-232`).

---

## 5. Key feasibility fact for the full port

**The full DRTC port does not need to reimplement guided denoising — local
lerobot already ships RTC.** Verified in `makermods/lerobot`:

- `PI0Pytorch.__init__(self, config, rtc_processor: RTCProcessor | None = None)`
  attaches the processor to `policy.model` (`policies/pi0/modeling_pi0.py:569-572`).
- `_rtc_enabled()` gates on `config.rtc_config is not None and
config.rtc_config.enabled` (`modeling_pi0.py:625-626`); `RTCConfig` lives at
  `policies/rtc/configuration_rtc.py` (`enabled`, `prefix_attention_schedule`,
  `max_guidance_weight`, `execution_horizon`).
- The sampler calls `self.rtc_processor.denoise_step(..., prev_chunk_left_over,
inference_delay, execution_horizon)` inside the flow loop
  (`modeling_pi0.py:879-885`).
- `predict_action_chunk(batch, **kwargs: Unpack[ActionSelectKwargs])`
  (`modeling_pi0.py:1265`) accepts `inference_delay`, `prev_chunk_left_over`,
  `execution_horizon` (`ActionSelectKwargs`, `:66-69`).

So the server side of the full port **drives the existing processor** rather than
porting the reference's `AsyncRTCProcessor`. The **canonical local driver** is
`makermods/lerobot/src/lerobot/rollout/inference/rtc.py`
(`RTCInferenceEngine._rtc_loop`): it computes `delay = ⌈latency /
time_per_chunk⌉` (`rtc.py:277`), pulls the unconsumed prefix via
`queue.get_left_over()` (`:274`), and calls
`predict_action_chunk(preprocessed, inference_delay=delay,
prev_chunk_left_over=prev_actions)` (`:307-309`). The `ActionQueue`
(`policies/rtc/action_queue.py`) then `_replace_actions_queue`s, discarding the
first `real_delay` actions (`:175-194`) — that is the reference "frozen prefix"
handling, already implemented. `rollout/inference/sync.py` is the non-RTC
per-tick baseline (inline `select_action`) for comparison.

---

## 6. What lives in `lerobot_drtc_full/` now

This directory is being built out concurrently by sibling agents; files below
are described by **intent** (line numbers for files this brief can't yet see are
intentionally omitted). Three regimes share common scaffolding:

| file(s)                                                                             | regime                      | intent                                                                                                                                                                                                                                                                                          |
| ----------------------------------------------------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ANALYSIS.md`                                                                       | —                           | this brief: context, not code                                                                                                                                                                                                                                                                   |
| `policy.py`, `_schema.py`, `_common.py`, `modal_policy.py`                          | shared scaffolding          | plain policy-agnostic Portal plumbing carried over from `lerobot_inference` / `lerobot_drtc` (checkpoint load, wire schema, env/token helpers, Modal deploy)                                                                                                                                    |
| `robot_sync.py`, `_sync_player.py`                                                  | **adaptive-sync**           | a novel adaptive-prefetch, block-execution player for **non-in-painting** policies under latency — the "replan rarely, play to completion" regime that §4 #2 argues is correct when there is no in-painting. Descends from `BlockChunkPlayer` but adapts its prefetch lead to measured latency. |
| `policy_rtc.py`, `robot_rtc.py`, `_rtc.py`, `_schema_rtc.py`, `modal_policy_rtc.py` | **full DRTC + in-painting** | the flow/diffusion port that adds half (b): server drives the local `RTCProcessor` (§5), robot sends the freeze index + prefix spans, and the wire schema carries the extra RTC metadata half (a) lacked.                                                                                       |

**Transport note for the in-painting regime.** Portal has **no opaque
side-channel** on state/action/chunk messages. The two ways to add RTC metadata
are (a) declaring **extra scalar schema fields** on the chunk/state (the intended
mechanism; `add_state_typed` / `add_action_chunk`,
`livekit/portal/__init__.py:800`, `:820`) or (b) the out-of-band **RPC** channel
(`perform_rpc`, `:1177`). Note a full microsecond timestamp does **not** fit one
scalar: Portal dtypes top out at `F64`/`F32` with no I64/U64 (`_INT_DTYPES` /
`_FLOAT_DTYPES`, `:103-106`), and F64's 53-bit mantissa is too small — so
spans/freeze-index must be split across fields or sent via RPC. The client must
also **share `d` (`ℓ̂_Δ`)** with the server so the freeze boundary and the
client's execution horizon agree.

---

## 7. One-paragraph summary

DRTC = (a) a policy-agnostic scheduler/transport layer + (b) flow-only guided
in-painting that depends on (a). Our first Portal cut (`../lerobot_drtc/`)
ported (a) faithfully — LWW merge, free alignment, JK-paced cooldown requests,
correct `in_reply_to_ts_us` correlation — but shipped **zero** of (b). Without
in-painting, overlaps hard-switch (jerk), and the inherited "replan often"
pacing _maximizes_ the number of hard seams, so it can underperform the
dead-simple play-to-completion `BlockChunkPlayer`. The fix is not to reimplement
RTC: local lerobot already exposes `predict_action_chunk(inference_delay=,
prev_chunk_left_over=)` driven canonically by `rollout/inference/rtc.py`. This
directory splits the work into an adaptive-sync regime (correct for
non-in-painting policies) and a full DRTC+in-painting regime (correct for
flow/diffusion), over the same Portal scaffolding — the remaining hard part
being the RTC metadata transport, which Portal supports only via extra scalar
schema fields or RPC.

---

## 8. Field notes — 2026-08-03 live run (LiveKit Cloud, horizon sweep)

Setup: SO-101 follower + front/wrist cams @ 30fps, `robot_rtc.py` ↔
`modal_policy_rtc.py` over **LiveKit Cloud** (no local SFU; `.env.local`
deleted). Modal app `ap-glo3K1ilsjgEbtLxijHWBb`.

### 8.1 Horizon 16 "freeze" = starvation by roundtrip > chunk coverage

Symptom: robot executed ~1 chunk then held position indefinitely; Modal log
showed the server healthy and answering ~4 obs/s (`chunks_sent` 1→170+,
`obs_seen` ≈ `chunks_sent`, `rtc_applied=0` throughout).

Mechanism: chunks merge onto the **absolute** step axis anchored at the
observation that produced them (`robot_rtc.py` on_chunk:
`chunk_start_step = t_src + action_delay`). At 30fps a horizon-16 chunk covers
533 ms from obs capture. Measured obs→chunk e2e at h=16 ≈ **550–610 ms p50**
(790 ms p95). Every chunk therefore lands with all 16 steps already in the
past: LWW merge writes nothing executable, `pop_current()` is dry every tick,
robot re-commands last action (the "freeze"), and the cooldown
(`s_min+epsilon` = 5 ticks) makes it re-request ~4–6×/s forever. Server
happily infers on each — infinite stale-chunk loop, `rtc_applied=0` because
in-painting needs an overlap with a still-live schedule, which never exists.

**CORRECTED (same night): transport is NOT the fat.** The local-SFU h=16 run
measured e2e p50 ≈ 550–610 ms — and the user confirms LiveKit **Cloud** at
h=16 measured the same ~550–600 ms. Cloud-vs-local made no measurable e2e
difference. The earlier "~800 ms" figure came from **horizon=50** runs (and
plausibly from quoting the robot's `lat=` readout, which is the JK estimate
srtt + k×dev with k=1.5 (robot_rtc.py latency_k) — tonight it printed ~700 ms
while true p50 was ~600 ms; confirmed by code read 2026-08-03). e2e decomposition at h=16, local SFU: ≈270 ms inference

- ≈130 ms portal rtt + ≈200 ms unexplained (state path / robot loop /
  portal internals — the state-vs-video sync skew logged operator-side was
  "up to 600 ms" via cloud vs "up to 200 ms" local; burst maxima, not means,
  and e2e-invariant, so their meaning is unresolved).

Budget rule of thumb: **RTC needs roundtrip < (H − s_min)/fps** — at H=16,
s_min=4, 30fps that is 400 ms vs measured ~600 ⇒ starves. H=24 (800 ms
coverage) runs but **jerks wildly** (diagnosis pending — suspects: prefix=0
hard seams, LWW mid-execution overwrites, JK step clamp [1, H//2] saturating
below true latency and corrupting the shipped inference_delay). H=50 "worked"
but executes up-to-1.7 s-stale actions — the perceived sluggishness. The
attackable terms are inference (270 ms) and the ~200 ms robot/portal-side
mystery, NOT WAN transport.

### 8.2 livekit-portal 0.2.4: server-side Rust panic on robot disconnect (upstream bug)

Every robot disconnect panics a receiver-side tokio thread on the policy
server:

    thread 'async-compat/tokio-1' panicked at livekit-portal/src/video.rs:181:88:
    video frame missing user_timestamp — sender must enable PacketTrailerFeatures.user_timestamp

Observed twice in one session (23:51:15Z, 23:51:46Z), each immediately after
`participant 'robot' disconnected`. The blame message is wrong: the robot's
signal URL shows `capabilities=CAP_PACKET_TRAILER` and video carried
user_timestamps fine for minutes — this is a **teardown race** (a queued frame
processed after the packet-trailer handler detached). Process survives: portal
re-attaches handlers and the policy re-claims control on reconnect
("control gate is now None; re-claiming"). Severity: cosmetic-ish today, but a
panicking thread on a peer-lifecycle event is upstream-report-worthy
(github.com/livekit/portal; we pin `livekit-portal==0.2.4`). Not MakerLab code
— do not file in MakerLab bug lists.

### 8.3 Misc observations from the same logs

- Policy-side sync buffer under cloud latency: recurring
  `[sync-drop] no frame within ±50ms (video up to 600ms ahead)` +
  state/video buffer overflows — states arrive well behind video; worth
  rechecking on the direct-UDP path before tuning any ±50 ms pairing window.
- `local_sfu.sh` gotcha (cost one confused run): `.env.local` outlives the
  script, so after Ctrl-C the robot still dials `ws://127.0.0.1:7880` →
  connection refused. Restart `local_sfu.sh` or `rm .env.local`.
- NAT preflight for the local-SFU path (checked 2026-08-03): this network is
  cone-NAT (STUN mapped port identical — and preserved — across 3 servers),
  public IP is a real Comcast address, no CGNAT ⇒ ICE hole punching from Modal
  should work with no UDP 7882 port-forward.

### 8.4 Later same night — slack, in-painting engaged, Tailscale hybrid live

- `--slack 2` (new passthrough in modal_policy_rtc.py): e2e p50 550–610 → ~480 ms.
  Slack term real, ≈60–100 ms.
- h=32 + slack 2: RTC in-painting ENGAGED for the first time — prefix=14–16 on
  every request, merge_l2 dropping to ~0 (frozen-prefix agreement) or small
  values (soft re-planning). Arm smooth; staleness visible. h=24 still marginal
  at this e2e (usable ≈ request period − 1).
- Instrumentation (robot_rtc.py, uncommitted): emit=8–14 ms, late=0 always,
  ret ≈ 25 ms true return leg. ret= is clock-offset-contaminated ACROSS runs
  (each Modal container has its own NTP offset; observed −48 ms offset) —
  within-run trends only; e2e is offset-immune (both stamps robot-side).
- **Tailscale hybrid worked end-to-end on first corrected run** (twin-function
  fix for Modal's conditional-object error; see README `## Tailscale hybrid`).
  Signaling via tailnet ws://100.81.37.0:7880 (stable URL, no more quick-tunnel
  churn), media unchanged direct UDP. Best e2e of the night: p50 ~430 ms,
  p95 ~559 ms. Budget now ≈ 135 forward + 270 inference (assumed) + 25 return.
- Next: measure Modal inference directly (timer around engine.run in
  policy_rtc.py) + print portal SyncMetrics (match_delta p50/p95,
  states_dropped); then video publish cadence; then walk horizon down.

### 8.5 Transport attribution, verified from SFU logs (2026-09-01)

Re-read every `logs/local_sfu*.log` from the runs above and extracted the ICE
pair the SFU actually selected, per participant (`grep 'participant active'`,
then the `[selected]` entries in `publisherCandidates` / `subscriberCandidates`).

**Media has ALWAYS gone over the public internet — the Tailscale run included.**
For the `policy` participant (the Modal container) every selected pair is this
Mac's public IPv4 `73.15.141.80:7882` against a public-cloud `srflx` address on
the Modal side (OCI `137.131.x` / `129.146.x` / `132.226.x`, AWS `34.221.x`,
each carrying a `related` RFC1918 container address). The `robot` participant
stayed on `192.168.86.30` / link-local IPv6 throughout, as expected — it shares
a machine with the SFU.

**Tailnet candidates were gathered and advertised, and NEVER won.** Both logs
contain the SFU's own tailnet host candidate (`fd7a:115c:a1e0::b401:25d2:7882`,
Tailscale's ULA range) and the robot side's `100.81.37.0`. Not one selected pair
uses either. The mechanism is the one `modal_policy*.py` already documents from
the other direction: a Modal container has no _media_ route to a tailnet address
(userspace tailscaled exposes SOCKS5 = TCP only), so every ICE connectivity
check against that candidate fails and only the public pair survives. Merely
having Tailscale up does not move media; the candidate has to be reachable.

Two corrections to §8.4 follow:

1. **The 480 → 430 ms improvement cannot be credited to Tailscale.** The media
   path was byte-for-byte identical before and after — same local candidate,
   same remote public cloud srflx. `--slack 2` is the surviving explanation and
   the residual is run-to-run variance. §8.4's phrasing ("media unchanged direct
   UDP. Best e2e of the night") is literally true but reads as if the hybrid
   bought the delta; it did not.
2. **Signaling is the ONLY thing Tailscale buys on the Modal path**, and that is
   a URL-stability/exposure win, not a latency one. Any plan that wants the 30fps
   loop itself on the tailnet needs a far end with a real TUN device — i.e. a
   MakerMods Lab peer, never a Modal container.

Unverified and now parked: whether two REAL tailnet nodes carry the media loop
over WireGuard at acceptable cost. `transport_probe.py` exists to answer it
(synthetic robot + echo operator, no robot hardware and no GPU) together with a
tailnet-pinned SFU config (`node_ip: <tailnet v4>`, `use_external_ip: false`,
which is what forces ICE to stop preferring the public pair). Blocked only on a
second machine; deferred until the GPU peer that would sit at that end exists.
