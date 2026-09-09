# Comparing Molmo RTC latency

Restart the GPU worker and start a fresh RTC session after updating the checkout.
An already-running Modal worker retains its uploaded code. These diagnostics
write to the existing GPU/session log files; the wire schema and policy settings
are unchanged.

## What each measurement means

- `[startup]` separates policy construction/loading, transfer to the GPU, and
  input/output processor initialization. The weight progress bar can finish
  long before `policy.from_pretrained` returns. Warm-up has its existing timer.
- `[policy] loaded parameter elements` counts the actual loaded parameters by
  dtype and device. Mixed dtypes are reported, not collapsed to a BF16 claim.
  This is parameter storage precision, not proof that every operation uses it.
- `build_ms` is CPU image decode/resize/batch construction. `pre_ms` includes
  preprocessing and RTC-prefix preparation. `infer_ms` covers action prediction.
  `post_ms` includes caching, output processing and copying actions to the CPU.
  CUDA completion is synchronized at preprocessing/prediction boundaries so
  enqueue time is not mistaken for computation time. The barriers can introduce
  measurement overhead; compare runs using the same instrumentation.
- `rtc_used` reports whether this measured prediction received an RTC prefix.
  Plain and guided predictions can have different costs.
- `match_delta_us`, `blocker`, `vbuf`, and `fjitter_us` expose Portal's
  synchronization and buffering diagnostics. They are not additional independent
  terms to blindly add to end-to-end latency.
- `[camera-timing]` reports each existing camera buffer's dimensions, age since
  the driver's last completed read, and observed update rate over up to five
  seconds. It polls non-blockingly at 200 Hz, does not open cameras, and does not
  read motors. `observed_hz` is a lower bound because polling can miss updates;
  it counts timestamp changes, not visual changes. Buffer age excludes unknown
  sensor exposure and USB/driver delay. Unsupported camera buffers are explicitly
  `unavailable`. This cannot establish physical exposure-to-action latency.
- `[tailscale]` distinguishes local SOCKS readiness, login (including the saved
  identity retry), and total startup. `[tailscale-relay]` separately reports when
  the policy actually asks to dial the station and the TCP connection duration.
  A listening local relay does not establish station reachability. Neither TCP
  success nor Tailscale's Connected badge establishes LiveKit media readiness.

## MolmoAct2 startup now loads the final weights once

Both the RTC and sync policy entrypoints now automatically use the Lab's
single-pass loader for full MolmoAct2 LeRobot checkpoints. No new flag or
dependency publication is required; launch a fresh GPU worker from this checkout.
Look for `[policy] MolmoAct2 single-pass loader: skipping base weight loads`.

It fetches base configuration/tokenizer/normalization assets without base weight
shards, constructs the architecture with empty parameters, validates every final
checkpoint key and shape, and materializes each tensor once in the requested
dtype/device. The saved input processor reuses those local assets. Adapter
configurations retain the existing loader; incomplete full checkpoints fail
instead of silently mixing in base weights. Small nonpersistent rotary caches
are rebuilt and persistent rotary buffers come from the final checkpoint.

Startup logs separately time metadata, architecture, final checkpoint resolution,
and final weight materialization. Disk caching still matters for the final
checkpoint, and a new container still needs to put its weights in GPU memory.
The previous 144.7-second total is the baseline, not a promised savings figure.

Validation compared every parameter/persistent buffer and seeded action output
against the original loader on a tiny real Molmo architecture in FP32 and BF16,
with and without RTC guidance. Base-weight reads were forbidden and each final
tensor was read exactly once. All 1,295 keys/shapes of the actual SO100_101 LeRobot
checkpoint at `a93b5fcdce4c1e3cc7688c939fc943bf4b092a8a` matched the full empty
architecture against base metadata at `152569fe57914d97be91055800035f54e250d009`.
Full-model GPU loading time and output comparison still require a fresh run.

## Controlled comparison

For comparison with the successful September 4 afternoon run, keep the same
Molmo checkpoint, task, two camera-role bindings, saved 640x480/30 fps capture
settings, 30 Hz control rate, horizon 30, codec, H100 request and BF16 selection.
Set **Advanced → Minimum budget** to **1** to match that earlier robot session,
and launch the GPU with the same setting. For manual commands the equivalent
is `--s-min 1` on the GPU and `--s_min=1` on the robot. Do not change flow steps,
camera settings and scheduling together. These notes do not change any defaults.

Wait for `[policy] connected` before starting the robot session. Under operator
supervision, compare equal-duration runs with the same starting pose, object
placement and lighting, retaining at least 30 seconds of steady operation if
the task permits. Use separate warm-up/steady windows rather than comparing
the first five seconds to a previous 29-second aggregate. Compare model time,
camera age/update rates, synchronization, and the same end-to-end percentiles.
Negative cross-host `ret` values require clock/timestamp investigation before
that quantity can be used to attribute network delay.

## Next E2E tests, in order

The instrumented run `gpu/1788584408.log` sampled about 563 ms total model-side
processing (534 ms prediction), while `sessions/1788584627.log` reported about
1209/1303 ms E2E p50/p95. These are different aggregates, not a per-request
decomposition. Healthy 640x480 camera buffers updated around 30 Hz. The leading
unresolved terms are observation synchronization/transport and waiting for the
model. Startup's 144.7-second loader delay is a separate issue.

Use the new per-request diagnostics below for a fresh baseline. Keep H100,
BF16, the same two cameras, task, horizon 30, 30 Hz, minimum budget, and tolerance
1.5 fixed. For this baseline keep minimum budget 4 as in the recent slow run;
the earlier afternoon comparison with minimum budget 1 is a separate test.
Set flow steps explicitly to 10 for reproducibility: the locally cached SO100_101
backbone config specifies `flow_matching_num_steps=10`; the policy config's
`num_flow_timesteps=8` is a training parameter, not the inference count.

Stop the robot session before changing settings. Relaunch the GPU for each
changed GPU setting, wait until it connects, then start the matching robot
session under operator supervision. Retain at least 40 seconds when feasible:
discard the first 10 seconds and compare the following 30 seconds. Repeat a
promising comparison with baseline/test order reversed to check network drift.
Do not select a faster setting if grasping or motion quality worsens.

| Run | Change from baseline | Where to set it | What the result tests |
| --- | --- | --- | --- |
| A | None: H264, slack 5, 10 flow steps | Codec in Advanced; flow steps on the Modal card | Fresh paired baseline |
| B | Slack 5 → 2 only | Modal card → Sync slack (ticks) → 2; the generated command also includes `--slack 2` | Whether less sync buffering lowers residual delay without increasing drops/starvation |
| C | H264 → MJPEG only; restore slack 5 | Advanced → Codec, then restart GPU and robot with that selection | Whether the video/state transport and matching path accounts for the delay; MJPEG defaults to quality 90 |
| D | Flow steps 10 → 6 only; restore H264/slack 5 | Modal card → Flow steps; restart GPU | How much prediction and queue time can shrink; evaluate grasp quality as well as latency |

Both codec selections must match the GPU and robot. The slack selector remembers
2, 3, 4, or 5 ticks (default 5), applies to both RTC and sync, and is locked while
the GPU runs. Restart the Lab after this update to load the new backend and
bundled UI; then stop the existing GPU, choose the value, and start the GPU.
The running GPU's echoed slack participates in the settings-mismatch warning.
When using manual commands, copy a freshly generated command so its room and
other settings match the session. Do not
launch a second GPU alongside the existing one. A codec change also changes
state reliability automatically; run C compares the complete transport mode,
not just compression CPU cost. MJPEG may worsen latency on a constrained uplink.
Reducing slack by three ticks is a nominal 100 ms at 30 Hz; actual savings must
be measured because emissions are paced and jitter varies. Reducing flow steps
does not reduce all processing proportionally: image/prompt encoding and other
work remain. Six steps is available in the current UI; five requires the manual
`--flow-steps 5` flag and is not needed for this initial comparison.

Record paired RTT p50/p95, queue/service/residual timings, `rtc_used`, sync-drop
messages, stale observations, schedule starvation, and task outcome for each
run. Compare guided and unguided predictions separately if their proportion
changes substantially. If B or C lowers residual while prediction stays similar,
prioritize transport/sync tuning. If D reduces prediction and also queue time,
model throughput is contributing to request backlog. If `queue_ms` is tiny in
all runs, stop blaming the Python inference queue.

## Pairing the next run's logs

New `[policy-timing]` lines contain an observation ID plus `queue_ms` from the
Python observation callback to selection, and `service_ms` from selection through
worker dispatch, model work, and the local action-send call. The observation has
already passed Portal synchronization when that callback runs. New
`[robot-timing]` lines measure `rtt_ms` with the robot's monotonic clock, from
the source control tick to its correlated action callback. This includes the
robot's acquisition/emission work but excludes sensor exposure before that tick
and action execution after receipt. These clocks do not have to agree across hosts.

After collecting a run, analyze its matching files (replace both paths):

```sh
.venv/bin/python -m makermodslab.drtc.analyze_timing \
  --gpu-log "$HOME/.cache/huggingface/lerobot/logs/drtc/gpu/GPU_RUN.log" \
  --robot-log "$HOME/.cache/huggingface/lerobot/logs/drtc/sessions/ROBOT_RUN.log" \
  --skip-seconds 10 --window-seconds 30
```

The analyzer computes `residual_ms = rtt_ms - queue_ms - service_ms` for each
matched request before calculating percentiles. Residual includes robot-side
acquisition/encoding, both transport directions, Portal synchronization, and
callback dispatch; it is **not** a direct network ping measurement. Host clock
offset cannot explain this residual. Do not subtract independently aggregated
p50/p95 values. The observation ID uses the robot's wall-clock stamp only for
matching and window selection; a wall-clock jump during a run can affect window
selection, so rerun after any clock adjustment.

Repeated IDs are excluded as ambiguous; missing, truncated, or unmatched lines
are not silently filled in. Fewer than 30 pairs produces a warning. Existing
older logs lack these new records and cannot yield this decomposition. Logging
occurs once per returned chunk; use the same instrumentation for every test.

The scheduler currently reuses an estimate capped at half the horizon for
request cooldown. At H=30/30 Hz that ceiling is 500 ms even when observed RTT
exceeds a second. This is a reason to measure queuing, not evidence that changing
the RTC prefix constraint is safe. Do not turn pacing off as a latency test:
that emits every control tick and can increase load substantially.

## Findings from the saved September 4 logs

The 21:31 run joined Tailscale at approximately 21:31:45 and did not attempt
LiveKit signalling until 21:34:35. Most of that wait occurred during policy
startup. A separate 21:25 run spent about 20 seconds on a failed saved-identity
login before succeeding with the auth key. That is the launcher's bounded
resume fallback, not a three-minute peer-discovery timeout. Deletion/revocation
of the saved identity is consistent with those authentication logs, but its
cause was not established.

Both older and newer runs contain MJPG-setting fallback warnings. The newer
logs contain synchronization drops and negative cross-host return timestamps.
Those observations motivate measurements; they do not isolate the latency
increase. Port-8000 refusals occur on an incoming connection to the Modal worker;
the actual station signalling destination is port 7880. The caller of the
port-8000 requests remains unverified.
