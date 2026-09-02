"""Policy SERVER for the FULL DRTC port — RTC in-painting over Portal (GPU side).

This is ``policy.py`` with Real-Time Chunking (RTC) guided in-painting turned on.
It loads a flow-matching / diffusion policy (smolvla, pi0, pi05), enables RTC on
it, and drives ``predict_action_chunk(batch, inference_delay=d,
prev_chunk_left_over=prefix, execution_horizon=overlap_end)`` per the reference
inference driver ``lerobot/rollout/inference/rtc.py`` (in the installed lerobot
package) and the upstream DRTC reference implementation's
``async_inference/policy_server_drtc.py`` (another codebase — neither path is a
file in this repo).

What RTC does here (why in-paint at all)
----------------------------------------
Without RTC, every new chunk is a hard switch: overlapping steps from the old
plan get clobbered, producing a seam the robot feels as a jerk. RTC treats the
new chunk as an *in-painting* problem — during flow-matching denoising it softly
pulls the near-future of the new chunk toward the still-to-execute prefix of the
old chunk (frozen for the first ``d`` steps, decaying to free by ``overlap_end``),
so consecutive chunks join smoothly. The robot supplies ``d`` (its measured
latency in steps) and the prefix; the policy's own ``RTCProcessor`` does the
guided denoising (``lerobot/policies/rtc/modeling_rtc.py``, in the installed
lerobot package). We only DRIVE it.

How RTC is enabled on a loaded policy (the load-bearing bit)
-----------------------------------------------------------
Local lerobot flow policies expose a clean, canonical hook — we use it rather
than the reference server's manual ``setattr`` dance (that was needed only
because DRTC ships its *own* ``AsyncRTCProcessor``):

    policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=…, …)
    policy.init_rtc_processor()   # builds RTCProcessor(config.rtc_config) and
                                  # attaches it to BOTH policy.rtc_processor and
                                  # policy.model.rtc_processor

``policy._rtc_enabled()`` then returns True (it checks
``config.rtc_config.enabled``), and inside ``model.sample_actions`` the denoise
loop routes through ``rtc_processor.denoise_step(...)``. See
``pi0/modeling_pi0.py`` ``init_rtc_processor`` (~L1156), ``_rtc_enabled`` (~L1169),
``predict_action_chunk`` (~L1265), and ``sample_actions`` (~L874); smolvla/pi05
mirror this exactly. Policies WITHOUT this hook (e.g. ACT) fall back to plain
``predict_action_chunk`` with a warning — the robot still sends the RTC state
fields (schema must match), they are simply ignored.

Raw vs executable action dim
----------------------------
``predict_action_chunk`` already unpads its output to the executable action dim
(pi0: ``actions[:, :, :original_action_dim]``), so the RAW chunk we cache is at
executable dim — the SAME dim the robot's prefix slice references, and the same
tensor lerobot's ``rollout/inference/rtc.py`` caches as ``original`` and later feeds back
as ``prev_chunk_left_over``. The ``RTCProcessor`` re-pads the prefix up to the
model's ``max_action_dim`` internally (lerobot's ``modeling_rtc.denoise_step``
L196), so we
never touch ``max_action_dim`` here. We cache the raw (pre-postprocess) chunk and
send the per-timestep post-processed (unnormalized) chunk, exactly like
``policy.py``.

Wire protocol
-------------
Extended state schema from ``_schema_rtc.py`` (base ``s0..s{D-1}`` + five RTC
control fields). The returned ``act`` chunk carries
``in_reply_to_ts_us=obs.timestamp_us`` so the robot can key freshness/provenance
and measure true e2e latency. The raw-chunk cache is keyed by that same
``obs.timestamp_us`` — the robot's ``rtc_src_ts`` field references it.

    # smolvla / pi0 / pi05 (RTC-capable), started by hand on a GPU box you own
    # (`modal run makermodslab/drtc/modal_policy_rtc.py ...` is the usual path):
    python -m makermodslab.drtc.policy_rtc --policy-path=${HF_USER}/my_pi0 --device=cuda \
        --fps 30 --horizon 16 --video-codec H264 --task "put the block in the box"
    # ACT etc.: runs, logs a warning, serves plain chunks (no in-painting).

Config is CLI-only. Only the LiveKit connection (LIVEKIT_URL / LIVEKIT_API_KEY /
LIVEKIT_API_SECRET / LIVEKIT_ROOM) is read from the environment / `_env.load_env`; every
behavior knob below is a command-line flag with no env fallback, so there is a
single, unambiguous place to set each one. `--fps` / `--horizon` / `--video-codec`
MUST match the robot's flags.

CLI knobs:
  --fps / --horizon           control rate + chunk horizon (match robot)
  --video-codec               H264 (default) | MJPEG | PNG | RAW (match robot)
  --duration                  run seconds; 0 = until robot leaves / Ctrl-C
  --s-min                     execution-horizon floor (RTCConfig.execution_horizon)
  --max-guidance-weight       RTCConfig.max_guidance_weight (default 10.0)
  --rtc-schedule              linear | exp | ones | zeros (prefix_attention_schedule)
  --raw-cache-size            LRU size for the raw-chunk cache (default 32)
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import OrderedDict, deque

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812  (the universal torch alias)
from livekit.portal import Observation, Operator, OperatorConfig, VideoCodec, frame_bytes_to_numpy_rgb

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from ._common import load_env, mint_token, required_env
from ._schema_rtc import (
    CHUNK_NAME,
    IMAGE_PREFIX,
    RTC_D,
    RTC_OVERLAP_END,
    RTC_PREFIX_LEN,
    RTC_PREFIX_START,
    RTC_SRC_TS,
    policy_wire_schema,
    rtc_state_fields,
)

IDENTITY = "policy"


def pick_device(requested: str) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def enable_rtc(
    policy, execution_horizon: int, max_guidance_weight: float, prefix_attention_schedule: str
) -> bool:
    """Turn on RTC guided in-painting on a loaded policy, in place.

    Returns True if RTC is now active, False if the policy doesn't support it
    (ACT and other non-flow policies) — in which case the caller falls back to
    plain chunk prediction. Mirrors the local canonical path
    (``policy.init_rtc_processor``) rather than the reference server's manual
    ``setattr`` (that was for DRTC's own AsyncRTCProcessor).
    """
    cfg = getattr(policy, "config", None)
    has_hook = hasattr(policy, "init_rtc_processor")
    # ``rtc_config`` is a field on flow-policy configs (pi0/pi05/smolvla). Its
    # presence (even if None) is the capability signal.
    supports = has_hook and cfg is not None and hasattr(cfg, "rtc_config")
    if not supports:
        print(
            f"[policy] WARNING: {type(policy).__name__} has no RTC support "
            "(no init_rtc_processor/rtc_config); serving PLAIN chunks. RTC "
            "in-painting requires a flow policy (smolvla/pi0/pi05)."
        )
        return False

    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    try:
        schedule = RTCAttentionSchedule[prefix_attention_schedule.upper()]
    except KeyError:
        print(f"[policy] unknown --rtc-schedule={prefix_attention_schedule!r}; falling back to LINEAR")
        schedule = RTCAttentionSchedule.LINEAR

    cfg.rtc_config = RTCConfig(
        enabled=True,
        prefix_attention_schedule=schedule,
        max_guidance_weight=max_guidance_weight,
        execution_horizon=execution_horizon,
    )
    # Builds RTCProcessor(cfg.rtc_config) and attaches to policy.rtc_processor
    # AND policy.model.rtc_processor. After this, policy._rtc_enabled() is True.
    policy.init_rtc_processor()
    ok = bool(getattr(policy, "_rtc_enabled", lambda: False)())
    print(
        f"[policy] RTC {'ENABLED' if ok else 'FAILED to enable'}: "
        f"schedule={schedule.name} max_guidance_weight={max_guidance_weight} "
        f"execution_horizon={execution_horizon}"
    )
    return ok


def load_policy(policy_path: str, device: torch.device):
    """Load any lerobot policy + its pre/post processors from a checkpoint."""
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(policy_path)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor


class RTCInference:
    """Turns a Portal ``Observation`` (state + RTC metadata + frames) into a
    post-processed ``(horizon, A)`` chunk, driving RTC guided in-painting.

    Owns the raw-chunk LRU cache keyed by observation timestamp (µs), the exact
    analogue of ``policy_server_drtc.ActionChunkCache`` (keyed by source step) in
    the upstream DRTC reference implementation.
    """

    def __init__(
        self,
        policy,
        preprocessor,
        postprocessor,
        image_features,
        state_dim,
        horizon,
        device,
        task,
        rtc_enabled: bool,
        raw_cache_size: int,
        s_min: int,
    ):
        self.policy = policy
        self.pre = preprocessor
        self.post = postprocessor
        self.image_features = image_features
        self.state_dim = state_dim
        self.horizon = horizon
        self.device = device
        self.task = task
        self.rtc_enabled = rtc_enabled
        self.s_min = s_min
        # obs_ts (µs) -> RAW (pre-postprocess) chunk tensor, shape (horizon, A).
        self.raw_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.raw_cache_size = max(1, raw_cache_size)

    # -- cache -------------------------------------------------------------
    def _cache_put(self, obs_ts: int, raw_chunk: torch.Tensor) -> None:
        if obs_ts in self.raw_cache:
            del self.raw_cache[obs_ts]
        self.raw_cache[obs_ts] = raw_chunk.detach().clone()
        while len(self.raw_cache) > self.raw_cache_size:
            self.raw_cache.popitem(last=False)  # evict oldest (LRU)

    # -- batch build (only the s0..s{D-1} fields feed the model) -----------
    def _build_batch(self, obs: Observation) -> dict:
        state_vec = np.array([obs.state[f"s{i}"] for i in range(self.state_dim)], dtype=np.float32)[
            None, :
        ]  # (1, D) — RTC control fields are NOT part of the model state
        batch = {"observation.state": torch.from_numpy(state_vec)}

        for full_key, feature in self.image_features.items():
            cam = full_key[len(IMAGE_PREFIX) :] if full_key.startswith(IMAGE_PREFIX) else full_key
            frame = obs.frames.get(cam)
            if frame is None:
                raise KeyError(f"policy expects camera '{cam}' but the robot sent none")
            rgb = frame_bytes_to_numpy_rgb(frame.data, frame.width, frame.height)  # HWC uint8
            img = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0  # CHW [0,1]
            _, exp_h, exp_w = feature.shape
            if img.shape[1] != exp_h or img.shape[2] != exp_w:
                img = F.interpolate(
                    img.unsqueeze(0), size=(exp_h, exp_w), mode="bilinear", align_corners=False
                ).squeeze(0)
            batch[full_key] = img.unsqueeze(0).contiguous()  # (1, C, H, W)

        if self.task:
            batch["task"] = self.task
        return batch

    # -- RTC kwargs from the obs's control fields --------------------------
    def _rtc_kwargs(self, obs: Observation) -> dict:
        """Build the RTC kwargs for ``predict_action_chunk`` from the obs's RTC
        state fields + our raw cache. Empty dict => plain prediction (first
        inference, no prefix, cache miss, or RTC unsupported)."""
        if not self.rtc_enabled:
            return {}
        state = obs.state
        prefix_len = int(round(state.get(RTC_PREFIX_LEN, 0.0)))
        if prefix_len <= 0:
            return {}
        src_ts = int(round(state.get(RTC_SRC_TS, 0.0)))
        cached = self.raw_cache.get(src_ts)
        if cached is None:
            # Cache miss (evicted / never seen): degrade gracefully to plain.
            return {}
        start = int(round(state.get(RTC_PREFIX_START, 0.0)))
        d = int(round(state.get(RTC_D, 0.0)))
        overlap_end = int(round(state.get(RTC_OVERLAP_END, 0.0)))
        if overlap_end <= 0:
            overlap_end = max(self.s_min, d)

        # Slice the cached RAW chunk -> (len, A). Clamp to what's actually there.
        end = min(start + prefix_len, cached.shape[0])
        start = max(0, min(start, cached.shape[0]))
        prev = cached[start:end]  # 2D [T_prev, A], matches lerobot's rollout/inference/rtc.py
        if prev.shape[0] == 0:
            return {}
        prev = prev.to(self.device)
        # execution_horizon == overlap_end: the local RTCProcessor's prefix
        # weights are built as get_prefix_weights(start=d, end=execution_horizon,
        # total=H) — leading ones [0,d), linear decay [d, overlap_end), trailing
        # zeros. This is the same semantics the reference server calls
        # "overlap_end" (rtc_guidance) — see lerobot's
        # modeling_rtc.get_prefix_weights.
        return {
            "inference_delay": d,
            "prev_chunk_left_over": prev,
            "execution_horizon": overlap_end,
        }

    def warmup(self, rounds: int = 2) -> None:
        """A few forward passes on synthetic zero inputs before connecting, to
        pay CUDA init / autotune / JIT off the control loop (a cold first
        inference otherwise poisons the robot's latency estimator). Best-effort.
        RTC guidance is inert here (no prefix), so this is safe with RTC on."""
        batch = {"observation.state": torch.zeros((1, self.state_dim), dtype=torch.float32)}
        for full_key, feature in self.image_features.items():
            c, h, w = feature.shape
            batch[full_key] = torch.zeros((1, c, h, w), dtype=torch.float32)
        if self.task:
            batch["task"] = self.task
        with torch.no_grad():  # NOT inference_mode: RTC guidance needs enable_grad
            for _ in range(rounds):
                b = self.pre(dict(batch))
                chunk = self.policy.predict_action_chunk(b)
                if chunk.ndim == 2:
                    chunk = chunk.unsqueeze(0)
                for t in range(min(chunk.shape[1], self.horizon)):
                    self.post(chunk[:, t, :])
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def run(self, obs: Observation) -> np.ndarray:
        """Predict, cache the raw chunk under this obs's timestamp, and return
        the post-processed (executable) chunk. Caches BEFORE postprocess so the
        robot's future prefix references the raw (model-space) actions — exactly
        the tensor RTC guidance compares against."""
        batch = self._build_batch(obs)
        rtc_kwargs = self._rtc_kwargs(obs)
        # NOT torch.inference_mode(): RTC guidance temporarily re-enables grad
        # for the in-painting correction term, which inference_mode forbids.
        with torch.no_grad():
            batch = self.pre(batch)
            chunk = self.policy.predict_action_chunk(batch, **rtc_kwargs)  # (B, H, A) normalized
            if chunk.ndim == 2:
                chunk = chunk.unsqueeze(0)
            chunk = chunk[:, : self.horizon, :]

            raw = chunk.squeeze(0)  # (H, A) RAW / normalized -> cache this
            self._cache_put(obs.timestamp_us, raw)

            # Post-process per timestep (post expects (B, A)); unnormalize.
            steps = [self.post(chunk[:, t, :]) for t in range(chunk.shape[1])]
            processed = torch.stack(steps, dim=1).squeeze(0)  # (H, A_out)
        return processed.detach().cpu().numpy().astype(np.float32)


async def main() -> None:
    # Build probe: if you DON'T see this exact line in the policy logs, the
    # container is running a stale policy_rtc.py (Modal shipped an old mount) and
    # any "fix" here isn't live yet. The background-claim fix and this probe ship
    # together, so seeing this line == the non-fatal claim is active.
    print("[policy] >>> policy_rtc build marker: bg-claim-v2 (non-fatal background claim) <<<")
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", default="", help="HF repo id or local dir of the pretrained policy")
    parser.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    parser.add_argument("--task", default="", help="Language instruction for VLA policies")
    parser.add_argument(
        "--fps", type=int, default=30, help="Control/stream frequency. MUST match robot --fps."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="Action-chunk horizon. MUST match robot --horizon. Must not "
        "exceed the policy's chunk_size (smolvla/pi0/pi05: 50).",
    )
    parser.add_argument(
        "--slack",
        type=int,
        default=5,
        help="Portal sync-buffer headroom, in ticks (default 5 ~ 167ms "
        "@30fps). The observation matcher runs on THIS (operator) side, "
        "so slack/tolerance are tuned here, not on the robot. Higher "
        "absorbs jitter but adds slack/fps of buffering to e2e; lower "
        "shaves e2e latency at the cost of jitter tolerance. Floor 2.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.5,
        help="Portal state<->frame match window, in ticks (search_range = "
        "tolerance/fps). 0.5 for strict real-time (prefer explicit drops "
        "over silent misalignment); 1.5 (default) for lossy/wireless links "
        "(materially fewer drops).",
    )
    parser.add_argument(
        "--video-codec",
        default="H264",
        help="H264 (default) | MJPEG | PNG | RAW. MUST match robot --video-codec.",
    )
    parser.add_argument(
        "--duration", type=float, default=0.0, help="Run seconds; 0 = run until the robot leaves / Ctrl-C."
    )
    parser.add_argument(
        "--s-min",
        type=int,
        default=4,
        help="Execution-horizon floor; seeds RTCConfig.execution_horizon "
        "and the plain-prediction overlap fallback. Match robot's --s_min.",
    )
    parser.add_argument(
        "--max-guidance-weight", type=float, default=10.0, help="RTCConfig.max_guidance_weight"
    )
    parser.add_argument(
        "--rtc-schedule", default="linear", help="prefix attention schedule: linear | exp | ones | zeros"
    )
    parser.add_argument("--raw-cache-size", type=int, default=32, help="LRU size for the raw-chunk cache")
    args = parser.parse_args()
    if not args.policy_path:
        raise SystemExit("--policy-path is required")

    load_env()
    url = required_env("LIVEKIT_URL")
    room = required_env("LIVEKIT_ROOM")
    token = mint_token(IDENTITY, room)
    fps = args.fps
    horizon = args.horizon
    duration = args.duration
    raw_cache_size = args.raw_cache_size

    device = pick_device(args.device)
    print(f"[policy] loading '{args.policy_path}' on {device} ...")
    policy, pre, post = load_policy(args.policy_path, device)

    # Enable RTC in-painting. execution_horizon default = H - s_min (the RTC
    # constraint region); per-request overlap_end from the robot overrides it.
    rtc_on = enable_rtc(
        policy,
        execution_horizon=max(1, horizon - args.s_min),
        max_guidance_weight=args.max_guidance_weight,
        prefix_attention_schedule=args.rtc_schedule,
    )

    schema, image_features = policy_wire_schema(policy)
    print(
        f"[policy] policy schema: state={schema.state_dim} "
        f"action={schema.action_dim} cameras={schema.cameras} horizon={horizon} "
        f"rtc={'on' if rtc_on else 'OFF (plain fallback)'} "
        f"slack={args.slack} tolerance={args.tolerance}"
    )

    codec = getattr(VideoCodec, args.video_codec.upper())
    cfg = OperatorConfig(room)
    for cam in schema.cameras:
        cfg.add_video(cam, codec=codec)
    cfg.add_state_typed(rtc_state_fields(schema))  # base state + RTC control fields
    cfg.add_action_chunk(CHUNK_NAME, horizon=horizon, fields=schema.action_fields())
    cfg.set_fps(fps)
    # Sync-buffer knobs live on the operator: this side matches state<->frames and
    # emits Observations, so slack/tolerance take effect here (the robot never runs
    # the matcher). Defaults reproduce Portal's built-ins (5 / 1.5).
    cfg.set_slack(args.slack)
    cfg.set_tolerance(args.tolerance)

    op = Operator(cfg)
    engine = RTCInference(
        policy,
        pre,
        post,
        image_features,
        schema.state_dim,
        horizon,
        device,
        args.task,
        rtc_enabled=rtc_on,
        raw_cache_size=raw_cache_size,
        s_min=args.s_min,
    )

    # Warm before connecting (see policy.py rationale). Best-effort.
    try:
        t0 = time.monotonic()
        print("[policy] warming up model ...")
        engine.warmup()
        print(f"[policy] warmup done in {time.monotonic() - t0:.1f}s")
    except Exception as exc:  # noqa: BLE001 — warmup is optional
        print(f"[policy] warmup skipped ({type(exc).__name__}: {exc})")

    obs_queue: deque[Observation] = deque(maxlen=2)
    obs_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def on_observation(obs: Observation) -> None:
        obs_queue.append(obs)
        loop.call_soon_threadsafe(obs_event.set)

    op.on_observation(on_observation)

    # Keep strong refs to fire-and-forget tasks so the loop can't GC them.
    bg_tasks: set = set()

    def spawn(coro) -> None:
        t = asyncio.create_task(coro)
        bg_tasks.add(t)
        t.add_done_callback(bg_tasks.discard)

    async def claim_control(reason: str) -> None:
        """Claim the robot's control gate, retrying until a robot is present.

        Operator-side `set_active_operator` is an RPC to the robot; with no robot
        in the room it raises (e.g. NoPeer) after a ~1.5s internal poll. Routine,
        not fatal — the policy is often started before the robot, and on LiveKit
        Cloud the robot's role attribute can take a moment to propagate. Retry so
        startup never crashes; the robot accepts our chunks once the claim lands.
        See policy.py for the full rationale.
        """
        me = op.local_identity()
        while True:
            try:
                await op.set_active_operator(me)
                print(f"[policy] claimed control as {me!r} ({reason})")
                return
            except Exception as exc:  # noqa: BLE001 — robot not present/propagated yet
                print(
                    f"[policy] control claim pending ({reason}: {type(exc).__name__}: {exc}); retrying in 1s"
                )
                await asyncio.sleep(1.0)

    # Re-claim control whenever the robot's control gate moves off us (a fresh
    # robot resets the gate to None and drops our chunks).
    def on_active_operator_changed(active: str | None) -> None:
        if active != op.local_identity():
            print(f"[policy] control gate is now {active!r}; re-claiming")
            spawn(claim_control("gate moved"))

    op.on_active_operator_changed(on_active_operator_changed)

    print(f"[policy] connecting to {url} as '{IDENTITY}' in room '{room}' ...")
    await op.connect(url, token)
    # Claim in the background — don't block or crash if the robot hasn't joined,
    # and so a slow claim-RPC ACK over a high-latency link (robot in CN, SFU in
    # SG) can't take down the server. claim_control retries; the robot applies the
    # gate even when the ACK is slow to return. [build-marker: bg-claim v2]
    spawn(claim_control("startup"))
    print(f"[policy] connected as '{op.local_identity()}'; claiming control in background ...")

    chunks_sent = 0
    rtc_applied = 0
    start = time.monotonic()
    last_log = start

    try:
        while duration <= 0 or (time.monotonic() - start) < duration:
            try:
                await asyncio.wait_for(obs_event.wait(), timeout=1.0)
                obs_event.clear()
            except TimeoutError:
                continue
            if not obs_queue:
                continue

            obs = obs_queue[-1]  # always run on the freshest observation
            had_prefix = int(round(obs.state.get(RTC_PREFIX_LEN, 0.0))) > 0 and rtc_on
            # Inference is blocking (GPU/CPU-bound); keep it off the event loop.
            chunk = await asyncio.to_thread(engine.run, obs)
            op.send_action_chunk(CHUNK_NAME, chunk, in_reply_to_ts_us=obs.timestamp_us)
            chunks_sent += 1
            if had_prefix:
                rtc_applied += 1

            now = time.monotonic()
            if now - last_log >= 1.0:
                m = op.metrics()
                print(
                    f"[policy] t={int(now - start):>3}s chunks_sent={chunks_sent} "
                    f"rtc_applied={rtc_applied} obs_seen={m.sync.observations_emitted} "
                    f"cache={len(engine.raw_cache)}"
                )
                last_log = now
    finally:
        print(f"[policy] sent {chunks_sent} chunks ({rtc_applied} in-painted); disconnecting...")
        await op.disconnect()
        op.close()


if __name__ == "__main__":
    asyncio.run(main())
