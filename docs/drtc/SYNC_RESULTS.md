# Adaptive-sync vs fixed-lead: does it help non-inpainting policies under latency?

## The question

Remote robot inference over LiveKit Portal sends observations to a policy and
gets back an **action chunk** (a short open-loop block). A **non-flow-matching /
non-inpainting** policy (ACT, diffusion, or any policy you do not want to guide
with dynamically-consistent denoising) must play each chunk to _completion_ — one
seam per chunk boundary — to stay in the distribution it was evaluated in. DRTC's
answer to latency (re-plan fast, merge overlapping chunks) adds a hard seam per
re-plan, which is only safe if the policy in-paints. So for these policies we keep
the **play-to-completion block player** and ask a narrower question:

> Given play-to-completion (one seam per boundary), does making the _prefetch
> lead_ adaptive to measured round-trip latency execute better under network
> latency than a fixed small lead?

`_sync_player.py::AdaptiveBlockPlayer` sets
`lead = JKLatencyEstimator.estimate_steps + base_lead` (clamped to `[1, H-1]`)
instead of a constant. Low latency → small lead → request near the end of the
chunk (minimal prefetch, smoothest). High latency → larger lead → request earlier
so the next block lands before the current one drains. Latency beyond one chunk
(`lead >= H - s_min`) → degrade to play-to-drain-and-**hold** (freeze) rather than
jump. It never adds a second seam — it only moves _when_ the robot asks.

## Method (simulation — no policy, no hardware, no network)

Fully headless and deterministic (seeded numpy), in `_sync_player.py`'s
`__main__`. Reproduce with:

```
python3 _sync_player.py
```

- Control loop at **fps=30, horizon H=16, action_dim=6**, 1500 ticks (~50 s).
- **Ground truth**: smooth per-joint sinusoids (sub-second frequencies), fixed
  across runs so both players are graded on the identical target trajectory.
- **Policy**: when asked at observation tick `t_obs`, returns a chunk = ground
  truth sampled for the H steps starting at `t_obs`, delivered after a network
  delay (`mean RTT + Gaussian jitter`, jitter = 25% of RTT, min 5 ms). A
  configurable fraction of returned chunks is **silently dropped**.
- Two players per setting: **adaptive** and the **fixed-lead baseline**
  (`base_lead = 2`, the `../lerobot_inference/robot.py` default).
- Sweep: mean RTT ∈ {30, 60, 120, 250, 400} ms × drop ∈ {0%, 5%}.
- Metrics: **starv** = frozen/held ticks (schedule dry); **seams** = chunk-boundary
  swaps and mean/max **L2 discontinuity** across each swap; **stale mean/p95** =
  executed-action age (ticks) vs its obs-time; **reqs** = inference requests.

Timestamps in the sim are integer ticks, not wall-clock, so runs are exactly
reproducible.

## Results (captured output)

```
  RTT  drop   player |  starv seams seamL2 mean seamL2 max stale mean stale p95  reqs
-------------------------------------------------------------------------------------
   30    0% adaptive |      1    93      0.0592     0.1071      11.45      19.0    94
   30    0%    fixed |      1    93      0.0586     0.0779      10.46      18.0    94
   30    5% adaptive |     36    91      0.0980     0.6769      11.22      19.0    99
   30    5%    fixed |     36    91      0.0857     0.5999      10.30      18.0    99
   60    0% adaptive |      2    93      0.0582     0.1293      13.43      21.0    94
   60    0%    fixed |      3    93      0.0587     0.1205      10.48      18.0    94
   60    5% adaptive |     30    91      0.0979     0.7533      13.19      21.0    98
   60    5%    fixed |      7    93      0.0613     0.3513      10.45      18.0    95
  120    0% adaptive |      5    93      0.0580     0.1529      15.92      23.0    94
  120    0%    fixed |     82    88      0.1107     0.2726      11.37      19.0    89
  120    5% adaptive |     41    91      0.0990     0.9378      15.37      23.0    97
  120    5%    fixed |    106    87      0.1269     0.8026      11.27      18.0    93
  250    0% adaptive |      7    93      0.0583     0.1071      18.39      26.0    94
  250    0%    fixed |    239    78      0.5359     1.2912      18.39      31.0    80
  250    5% adaptive |     48    90      0.0918     1.0773      18.40      26.0    95
  250    5%    fixed |    253    77      0.5095     1.5581      17.91      31.0    80
  400    0% adaptive |    145    84      0.1992     1.2737      21.24      31.3    89
  400    0%    fixed |    316    73      0.6250     1.4806      22.78      34.0    82
  400    5% adaptive |    236    78      0.2804     2.1174      21.51      32.0    89
  400    5%    fixed |    355    71      0.7424     2.0346      23.29      35.0    83
```

Headline deltas (no-drop): starvation cut **120 ms: 82→5 (−94%)**,
**250 ms: 239→7 (−97%)**, **400 ms: 316→145 (−54%)**; and max-seam L2 falls with
it (**250 ms: 1.29→0.11**, **400 ms: 1.48→1.27**) because the fixed player's
starvation forces big catch-up jumps that adaptive avoids.

## Verdict (plain English)

**Yes, adaptive-sync clearly wins under high latency, and it wins on both axes
that matter for a non-inpainting policy** — it starves far less _and_ its seams
stay small. From ~120 ms RTT upward the fixed small lead can no longer hide the
round-trip: the runway drains before the next chunk lands, the arm freezes for
tens-to-hundreds of ticks, and each recovery is a large discontinuity. Adaptive
requests earlier in proportion to measured latency, so the next block is staged
in time: at 250 ms it removes ~97% of starvation and keeps max seam ~12× smaller.
At an extreme 400 ms (RTT > H/2 ticks, so the estimator's `estimate_steps` clamps
and even adaptive cannot fully cover a chunk) it still halves starvation and, at
that point, also has _lower_ staleness than the fixed player — the fixed player is
so starved that its held actions age worse.

**Be honest about where it does not help:**

- **Low latency (≤ 60 ms), no loss:** essentially a tie. Both players rarely
  starve; adaptive carries a slightly larger runway, so its actions are a couple
  of ticks more stale (p95 ~21 vs ~18) for no starvation benefit. This is the
  intended trade — adaptive is _neutral_ here, not better.
- **Low latency (30–60 ms) WITH packet loss:** adaptive can be _slightly worse_
  (60 ms/5%: 30 vs 7 starve ticks). When RTT is tiny the fixed lead of 2 is
  already sufficient, and adaptive's larger lead scales up its drop-retry deadline,
  so it waits a few extra ticks before re-requesting a dropped chunk. The absolute
  cost is small (~2% of ticks) and it reverses by 120 ms, where adaptive is ahead
  even under 5% loss (41 vs 106).
- **Staleness is the price of prefetching earlier.** Adaptive is deliberately a
  few ticks more stale in the mid-latency band; that is the cost of never
  starving. For open-loop block execution this is the right trade.

Net: use adaptive-sync (the default) whenever the link can exceed ~100 ms RTT;
below that it is a wash, and only under low-latency-plus-loss is the fixed lead
marginally better.

## Caveats & running the real thing

This is a **simulation with a perfect, deterministic "policy"** (ground truth
sampled directly) and a synthetic network. It isolates the scheduling behaviour
of the player; it does **not** include a real ACT/diffusion checkpoint, image
observations, GPU inference-time variance, or real LiveKit transport. Real-world
seam magnitudes depend on the policy's own chunk-to-chunk consistency, and real
`estimate_steps` sees true e2e latency (queueing + inference + transport), not a
clean RTT. Treat the numbers as directional, not absolute.

To run the real robot-side loop against the unchanged `policy.py`:

```
# adaptive (default):
uv run robot_sync.py --robot.type=so101_follower --robot.port=/dev/ttyACM1 \
    --robot.id=my_arm --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
    --horizon=16

# fixed-lead baseline for comparison:
uv run robot_sync.py ... --no-adaptive --base_lead=2
```

`robot_sync.py` mirrors `../lerobot_inference/robot.py` (same draccus config,
Portal `Robot` setup, wire schema, control-loop shape, and logging) and only
swaps in `AdaptiveBlockPlayer`. JK-estimator knobs (`--latency_alpha/beta/k`),
`--base_lead`, and `--s_min` are exposed on the CLI in the style of
`../lerobot_drtc/robot.py`.
