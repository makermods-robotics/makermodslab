# Remote-inference parameter sweep — plan, axes, ranges

Companion to [`ANALYSIS.md`](./ANALYSIS.md) (esp. §8, the 2026-08-03 field notes,
which supplies every latency number used to pick the ranges below).

**Objective.** Eraser-place task success on the SO-101 with the smolvla
checkpoint that already runs smoothly under RTC.

Success rate is the _verdict_, not the _search signal_ — it's flat-zero across
the whole infeasible region and needs N≳24 per config to resolve a 25-point gap.
So the sweep ranks on a surrogate and spends episodes only on finalists:

```
minimize   staleness p95      (ms of age on the action being executed now)
s.t.       starvation < 1% of ticks          (feasibility gate, binary)
           RTC engaged on >90% of requests   (prefix_len > 0)
           jerk <= incumbent config's jerk   (SOFT ceiling — see below)
```

Jerk is a **constraint, not a term** — as an objective it is trivially won by
`lpf_hz`, which converges the sweep on a maximally-smooth, maximally-stale
config (the h=50 "worked but sluggish" trap, §8.1).

**Jerk ceiling softened (2026-08-31, partner review).** Jerk is uniquely
repairable downstream: it's pure post-processing on the command stream (the
axis-9 LPF exists for exactly this, and stronger smoothers — Kalman, or a
small learned model — remain available later), whereas staleness and
starvation are structural and cannot be filtered away. So the ceiling breaks
ties and flags configs, but does NOT exclude a config that wins clearly on
staleness — and when jerk differences between configs are inconclusive,
ignore the criterion entirely rather than letting noise in a repairable
metric veto an unrepairable win. The anti-`lpf_hz` trap still holds: jerk
never enters the _objective_.

**Open question, parked (2026-08-31)**: whether worse surrogate numbers and
better task results can co-occur — i.e. surrogate validity beyond pruning.
Revisit before treating any stage-3/4 ranking as more than a shortlist;
stage 6 remains the only verdict.

---

## Derived quantities — everything below follows from these

| Quantity        | Formula                                       | Source                                 |
| --------------- | --------------------------------------------- | -------------------------------------- |
| Feasibility     | `e2e < (H - s_min) / fps`                     | §8.1 budget rule                       |
| Freeze index    | `d = min(ceil((srtt + k*dev) * fps), H // 2)` | [`_rtc.py:96`](_rtc.py:96)             |
| Overlap window  | `overlap_end = max(1, H - max(s_min, d))`     | [`robot_rtc.py:435`](robot_rtc.py:435) |
| Cooldown        | `d + epsilon` ticks between requests          | [`robot_rtc.py:460`](robot_rtc.py:460) |
| Staleness       | `(d + position_in_chunk) / fps`               | measure per-tick                       |
| LPF group delay | `order / (2*pi*cutoff_hz)` seconds            | [`_filter.py:11`](_filter.py:11)       |

At 30 fps one tick = 33.3 ms. Measured e2e: **p50 ~430 ms, p95 ~559 ms**
(Tailscale hybrid + slack 2, best of §8.4).

**Two consequences worth internalizing before sweeping:**

1. **`d` is probably pinned at the clamp today.** At H=32 the ceiling is
   `H//2 = 16`. With e2e ~430–480 ms and `k=1.5` inflating the estimate, raw `d`
   lands at 16–17 — i.e. **saturated**. That means the shipped `inference_delay`
   under-reports true latency, which is one of §8.1's named suspects for the
   h=24 jerk. Verify from telemetry before tuning anything downstream.
2. **`s_min` is currently inert on `overlap_end`.** It only binds when
   `s_min > d`, and `d` saturates at `H//2` — so at H=32, `s_min` must exceed
   **16** to change the overlap at all. Every value in `{4 … 16}` is the same
   config on that axis. It still binds on the pacing trigger
   (`runway <= H - s_min`), in the same direction.

---

## Independent axes

22 raw knobs collapse to **10 axes**, because coupled pairs move together, six
knobs should be pinned, and four sweep entirely offline. (Revised 2026-08-31,
partner review: axes 5 and 6 pinned — their sections below record the decision —
and axis 3b, video codec, added; 9 axes live.)

### Live-run axes (need a robot — sim or real)

#### 1. `horizon` — the primary axis

Set identically on both sides ([`robot_rtc.py:135`](robot_rtc.py:135),
[`policy_rtc.py:354`](policy_rtc.py:354)). Bounded above by the model's
`chunk_size` (smolvla = 50).

| H   | Coverage `(H-4)/30` | Margin over p95 | Note                                  |
| --- | ------------------- | --------------- | ------------------------------------- |
| 20  | 533 ms              | **−26 ms**      | infeasible — excluded                 |
| 24  | 667 ms              | 107 ms          | §8.4: "still marginal"; jerks at h=24 |
| 28  | 800 ms              | 240 ms          | **untested gap** — include            |
| 32  | 933 ms              | 373 ms          | incumbent; RTC engaged, prefix 14–16  |
| 40  | 1200 ms             | 640 ms          | staleness starts to dominate          |
| 50  | 1533 ms             | 973 ms          | reference only — known 1.7 s stale    |

**Sweep `{24, 28, 32, 40}`**, with 32 as incumbent and 50 as a stale reference.

#### 2. Replan conservatism — `s_min` + `epsilon`

Both control how often a fresh chunk is requested, and therefore how many hard
seams per second. Do **not** grid them independently; walk a single ladder.
Rationale is §4-#2: absent perfect in-painting, replanning more often _creates_
seams.

| Rung            | `s_min`    | `epsilon` | Effect                                                   |
| --------------- | ---------- | --------- | -------------------------------------------------------- |
| aggressive      | 4          | 0         | max replan rate, max seams                               |
| baseline        | 4          | 1         | current default                                          |
| conservative    | 4          | 3         | longer cooldown, staler, fewer seams                     |
| overlap-binding | `H//2 + 4` | 1         | first rung where `s_min` actually shortens `overlap_end` |

The last rung is the only one that tests `s_min`'s overlap role at all — see
consequence (2) above.

#### 3. Uplink load — `video_bitrate_kbps` + camera fps

§8.3's `[sync-drop]` / state-buffer-overflow failure mode lives here: video
saturates the uplink and head-of-line-blocks the state channel.

Sweep `video_bitrate_kbps ∈ {1500, 2500, 4000, 6000}` at fixed camera settings,
scoring on sync-drop rate **and** e2e, not e2e alone.

> **Constraint — do not sweep camera count or resolution as a tuning knob.**
> The checkpoint was trained with a specific camera set (front + wrist) at a
> specific resolution. Dropping a camera isn't a cheaper config, it's a
> different observation space, and the policy will simply fail. Capture _fps_
> reduction (30 → 15) is the only safe payload lever, and it changes
> observation freshness, so treat it as its own axis if tested at all.

#### 3b. Video codec — H264 vs MJPEG (added 2026-08-31, partner review)

Categorical, and it **forks axis 3** rather than extending it — the two codecs
have disjoint load knobs and different transport topology:

|                                   | H264 (incumbent)                                   | MJPEG                                                                                             |
| --------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Wire path                         | WebRTC RTP, lossy, best-effort                     | per-frame byte-stream on reliable SCTP, **shared with state**                                     |
| Uplink-load knob                  | `video_bitrate_kbps` (axis 3)                      | `video_quality` (`video_bitrate_kbps` is dead)                                                    |
| State channel                     | unreliable (real-time)                             | reliable — auto-follows the codec ([`robot_rtc.py`](robot_rtc.py), mirrored from `robot_sync.py`) |
| §8.3 failure mode                 | congestion **desyncs** video/state → `[sync-drop]` | congestion **stalls both together** — frames+joints stay matched but the whole observation ages   |
| Wire cost @ trained 640×480×30fps | 1.5–6 Mbps (swept ceiling)                         | q90 ≈ 10–20 Mbps, q70 ≈ half that — all reliable                                                  |

**Cells: MJPEG × `video_quality ∈ {70, 90}`**, run against the H264 × bitrate
cells of axis 3, same surrogate (e2e p95 + sync-drop rate + starvation). The
q70 cell exists so MJPEG doesn't lose purely on bandwidth it didn't need; if
q90 saturates tailscale, only q70's verdict matters. `--video-codec` MUST match
on both sides — a mismatch subscribes the operator on the wrong transport and
presents as silent starvation, not an error.

What MJPEG buys if it survives the load gate: per-frame sync built in (the
livekit-portal pitch), frame independence, sub-ms decode, no encoder latency
or GOP artifacts. What it risks: head-of-line blocking now stalls the _entire_
observation stream, so network trouble shows up as staleness spikes instead of
sync-drops — the surrogate already scores both.

#### 4. Operator buffering — `--slack` + `--tolerance`

Server-side ([`policy_rtc.py:357`](policy_rtc.py:357), [`:363`](policy_rtc.py:363)).
`slack` adds `slack/fps` of buffering directly to e2e — the ~100 ms that §8.4
recovered going 5 → 2 is exactly `3/30`.

| `slack` | added e2e | risk                    |
| ------- | --------- | ----------------------- |
| 1       | 33 ms     | sync-drops under jitter |
| **2**   | 67 ms     | §8.4's measured win     |
| 3       | 100 ms    |                         |
| 5       | 167 ms    | default; over-buffered  |

Sweep `slack ∈ {1, 2, 3}` with `tolerance ∈ {1.0, 1.5}`, scoring sync-drop rate
alongside e2e — slack 1 is only a win if drops stay at zero.

#### 5. Transport path — PINNED 2026-08-31: `local SFU + tailscale`

Was `LiveKit Cloud` | `local SFU + cloudflare` | `local SFU + tailscale`.
Decided (tailscale: p50 430 / p95 559, best measured; partner concurs). No
LiveKit Cloud reference runs either — the axis is closed, not just narrowed.

#### 6. GPU — PINNED 2026-08-31: `A100`

Was an A10G-vs-A100 comparison ("measure this first"). Partner call: skip the
comparison, run everything on A100 — every bit of inference latency matters,
and cost was never the constraint (~3× hourly on a few hours of container
time). Default flipped at [`modal_policy_rtc.py:374`](modal_policy_rtc.py:374).

**One calibration run survives the pin**: every latency number in this doc
(e2e p50 ~430 / p95 ~559, inference ~270 ms) was measured on A10G. The A100
e2e budget must be re-measured once, task-free, before deriving the feasible
horizon set — if it shaves ~100 ms, the feasible-H table, the `d` saturation
argument, and the axis-1 sweep set `{24, 28, 32, 40}` all shift down.

#### 7. In-painting quality — `--rtc-schedule` + `--max-guidance-weight`

Only sweep if failure modes come back tagged as seam artifacts (jitter at
contact). `rtc-schedule ∈ {linear, exp}` with `zeros` as an ablation control
(≈ in-painting off); `max_guidance_weight ∈ {2, 5, 10, 20}`.

### Offline axes — sweep from recorded traces, zero robot runs

#### 8. `latency_k` ∈ `{0.5, 0.75, 1.0, 1.25, 1.5}`

The deviation multiplier in the JK estimator: `estimate = srtt + k*dev`
([`robot_rtc.py:186`](robot_rtc.py:186)). It sets the jitter margin above mean
RTT, and therefore `d` — which is shipped as the server's `inference_delay`,
sets `overlap_end`, and sets the cooldown. Too low and the server unfreezes a
prefix the arm is still executing (seam); too high and you freeze more than
needed and act staler.

**Only values below the default are distinguishable.** §8.4's `lat=~700 ms` vs
true p50 ~600 ms back-solves to `dev ≈ 67 ms` (single-point — confirm from
telemetry). At srtt ≈ 430 ms, H=32, clamp `H//2 = 16`:

| `k` | estimate | raw steps | after clamp                 |
| --- | -------- | --------- | --------------------------- |
| 0.5 | 463 ms   | 14        | 14                          |
| 1.0 | 497 ms   | 15        | 15                          |
| 1.5 | 530 ms   | 16        | 16                          |
| 2.0 | 564 ms   | 17        | **16** — same config as 1.5 |

Replay recorded e2e traces through [`_rtc.py`](_rtc.py)'s estimator — no robot
needed.

#### 9. LPF — `lpf_hz` × `lpf_order`

Pure post-processing on the commanded stream ([`robot_rtc.py:382`](robot_rtc.py:382)),
so it sweeps offline against a recorded raw-action trace. Walk it as a
lag ladder, not a 2-D grid:

| `lpf_hz` | `lpf_order` | group delay | ticks @30fps |
| -------- | ----------- | ----------- | ------------ |
| 0 (off)  | —           | 0           | 0            |
| 12       | 2           | 27 ms       | 0.8          |
| 8        | 2           | 40 ms       | 1.2          |
| 5        | 2           | 64 ms       | 1.9          |
| 5        | 4           | 127 ms      | 3.8          |
| 3        | 2           | 106 ms      | 3.2          |

Must satisfy `0 < lpf_hz < fps/2 = 15` ([`_filter.py:90`](_filter.py:90)).
Use this to bring a _fast_ config's jerk under the incumbent's ceiling — not as
a quality knob in its own right.

#### 10. Scheduler micro-behaviour

`epsilon` and `action_delay` interactions, merge/LWW behaviour under reorder and
loss. Replay-only, already partly covered by [`_rtc.py`](_rtc.py)'s self-tests.

---

## Pinned — do not sweep

| Knob                             | Value                 | Why                                                                                                                        |
| -------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `fps`                            | 30                    | Both sides; changes every ms-denominated quantity                                                                          |
| `action_delay`                   | 1                     | Structural — where a chunk lands on the step axis                                                                          |
| `latency_alpha` / `latency_beta` | 0.125 / 0.25          | RFC 6298 constants                                                                                                         |
| `raw_cache_size`                 | 32                    | Needs to be big enough; not a tradeoff                                                                                     |
| `video_quality`                  | —                     | UNPINNED 2026-08-31: now axis 3b's MJPEG load knob, swept {70, 90}                                                         |
| `reliable_state`                 | auto                  | Follows the codec in both robots (unreliable w/ H264, reliable w/ byte-stream); never set manually in a sweep cell         |
| `pacing`                         | True                  | `False` is a baseline, not a config                                                                                        |
| camera count / resolution        | as trained            | Changes the observation space — see §3                                                                                     |
| transport path                   | local SFU + tailscale | Pinned 2026-08-31 (was axis 5) — best measured, partner concurs; no Cloud reference runs                                   |
| GPU                              | A100                  | Pinned 2026-08-31 (was axis 6) — comparison skipped, but ONE e2e calibration run still required (all doc numbers are A10G) |

---

## Staged execution order

Each stage's outcome constrains the next, so do not reorder.

| Stage | What                                                                                           | Where             | Runs            | Gate to next                                     |
| ----- | ---------------------------------------------------------------------------------------------- | ----------------- | --------------- | ------------------------------------------------ |
| 0     | Per-tick telemetry + offline scorer                                                            | —                 | —               | Can compute staleness/jerk/starvation from a log |
| 1     | A100 e2e calibration (GPU pinned)                                                              | live, task-free   | 1               | e2e budget fixed                                 |
| 2     | codec × load (H264 × bitrate, MJPEG × quality) + slack/tolerance (transport pinned: tailscale) | live, task-free   | ~10             | e2e p95 known + codec verdict → feasible H set   |
| 3     | Offline axes (`latency_k`, LPF, epsilon) — exhaustive grid, no search                          | replay            | 90, free        | Candidate shortlist                              |
| 4     | `horizon` × replan ladder                                                                      | sim, free-running | ~20             | Surrogate ranking                                |
| 5     | In-painting knobs (if seam failures)                                                           | sim               | ~8              | —                                                |
| 6     | **Verdict** — incumbent + sim winner + one notch conservative                                  | real arm          | 3 × 20 episodes | —                                                |

Stage 6 carries three configs, not two: sim systematically **under-penalizes
jerk** (perfect actuators) and **under-measures observation latency** (no camera
pipeline), and both biases push its winner toward being too aggressive. The
extra rung hedges in the known direction of the error.

**Sim status (2026-08-31, partner)**: the real policy runs in the partner's
sim — imperfect, gap-closing in progress, but usable for stage 4/5's purpose
(surrogate ranking, not verdicts). Partner will send a sim update when testing
approaches stage 4/5; refresh the bias assumptions above against it then.

## Protocol for stage 3 — exhaustive, not searched (2026-08-31, partner review)

No search algorithm (evolutionary or otherwise) — enumerate the full offline
subspace. The multiplication says search machinery can't pay for itself:
`latency_k` (5) × LPF ladder (6) × `epsilon` (3) = **90 configs**, each a
deterministic replay of a recorded trace (~1500 ticks) through
[`_rtc.py`](_rtc.py)'s estimator / [`_filter.py`](_filter.py) — well under a
second per config, minutes total on the Mac, zero supervision. Exhaustive also
returns the full response surface rather than a single winner, which stage 4
needs (ladder monotonicity, and the clamp collapses `latency_k` cells —
§axis 8 — so a searcher would waste evals on identical configs without
noticing). Requires only stage 0's scorer + traces recorded during stage 2.

**Why replay is valid at all**: the trace records quantities these knobs are
causally downstream of — RTTs (network doesn't care what `latency_k` is) and
policy actions (the LPF consumes them, doesn't shape them). Replay recomputes
pure functions of fixed inputs; it is NOT a closed-loop simulation — it holds
recorded actions fixed and so can never measure task success, only the
surrogate, which was chosen because it is open-loop computable. Closed-loop
effects are stage 4/6's job. Membership test for this stage: a knob is
replayable iff it leaves the recorded quantities exogenous (codec/bitrate
change the network, horizon changes what the policy is asked — both fail; they
sweep live/sim). Caveat: `epsilon` shifts request timing, so replayed requests
sample the RTT process at slightly different moments — acceptable under
within-trace stationarity, and hedged by the multi-trace aggregate.

**One eval per config × trace — but several traces.** Replay is deterministic,
so repeating a (config, trace) cell recomputes the same number: never repeat.
The variance lives in the _trace_ (one recording = one draw of RTT/jitter
conditions), so replay the identical 90-cell grid over 3–5 traces spanning
conditions (different times of day; include one loaded-uplink trace from the
stage-2 stress cells). Rank on the aggregate: worst-case across traces for
feasibility metrics (starvation, sync), mean for staleness. 270–450 replays,
still minutes, still unsupervised.

Why so few traces suffice (they are NOT like training seeds): one trace is
~1500 ticks, so the scored metrics are already aggregates over hundreds of
within-trace events — the residual variance is _between regimes_ (quiet vs
loaded), a small discrete set that wants coverage, not count. 3–5 is a floor
for regime coverage, not a ceiling: every stage-2 live run records a trace for
free, so use all of them — but never spend extra bench time collecting traces
just for stage 3. Configs that rank-swap across traces are ties; carry both to
stage 4 rather than resolving with more data. (With many traces, switch
worst-case aggregation to a quantile, or one freak event dominates the
ranking.)

## Protocol for stage 6

- Grade each episode 0–3 (reached / grasped / lifted / placed), not pass-fail —
  ordinal scores have much lower variance than Bernoulli at the same N.
- Pair on ~10 marked initial eraser poses, run each config over the same set ×2,
  and compare per-pose. Paired analysis removes pose-difficulty variance, which
  otherwise dominates.
- Tag the failure mode every episode: missed grasp / knocked / released early /
  never converged / oscillated at contact. The modes map onto axes — late grasp →
  staleness, jitter → seams, never converged → starvation — and carry several
  times the information of a boolean for the same arm time.
- Interleave configs (A/B/A/B), never block. Servo heating, battery sag and
  lighting drift over a session will otherwise confound config with time.

## Inclusion criteria — what gets excluded, and on what basis

Exclusion is allowed **only** on the pre-registered mechanistic gate (starvation,
RTC engagement, late ticks), measured from telemetry and decided _before_
episodes are run. That gates on a covariate.

**Never exclude a config because it produced few successes.** That is selection
on the dependent variable, and it costs two specific things here:

- **The feasibility wall disappears.** A config at 0/20 is not a failed
  measurement, it is the measurement that locates the cliff — h=16 in §8.1 was
  the data point that explained the entire failure mode.
- **Range restriction distorts the knob→success relationships** among the
  survivors, which is precisely what the sweep is trying to estimate.

The graded 0–3 score largely dissolves the problem: a config that never places
the eraser still reports whether it reached, grasped and lifted, so it retains a
position on the scale and never needs discarding. Handle imprecision by
reporting the paired per-pose comparison and its CI — a wide interval means
imprecise, not inadmissible.

**Protect this case explicitly:** a config that passes every mechanistic gate and
still scores near zero is the most informative result the sweep can produce. It
says the remote-inference configuration is _not_ the bottleneck — the checkpoint
or the task setup is — and that further sweeping is wasted effort. Filtering it
out as "insufficient successes" discards the finding that saves the most time.

## Cross-run measurement traps

- **Score on `e2e`, never `ret`.** `ret` is clock-offset contaminated across
  containers (−48 ms observed, §8.4); a sweep is inherently cross-run, so it
  would rank configs by NTP skew. `e2e` is stamped robot-side at both ends.
- **`lat=` in the robot log is the JK estimate, not measured latency** — it read
  ~700 ms when true p50 was ~600. Score the logged distribution.
- **The current 1 Hz log line is not sufficient.** `dmax_raw`/`dmax_sent` are
  last-second _maxima_, reset every print ([`robot_rtc.py:491`](robot_rtc.py:491)).
  Stage 0 exists because of this.

## Sim transfer

Sim (partner's env + his checkpoint) ranks; the arm confirms. What carries:

- **Exactly:** feasibility boundary, RTC engagement, staleness distribution,
  merge/LWW behaviour — all pure scheduler arithmetic over a latency trace.
- **Shape only:** the staleness→performance curve's form and the direction of
  the horizon tradeoff. Not the location of the eraser-place cliff.
- **Not at all:** task success, which failure mode dominates, anything mediated
  by real actuators (compliance and backlash turn a seam into overshoot; a sim
  position controller tracks it nearly perfectly), real camera-pipeline latency.

**Precondition:** the sim must free-run on a wall clock at 30 fps and execute
whatever the schedule holds — including holding the last action when starved. A
synchronous `step → wait for action → step` harness has zero staleness by
construction, so every config looks feasible and the sweep measures nothing.

**Calibration:** measure 3 anchor configs (one expected to starve, the h=32
incumbent, one over-conservative) in _both_ worlds. If sim and arm agree on the
ordering, the sim ranking over the rest of the space has earned credit; if not,
discard it rather than shipping a config it picked.
