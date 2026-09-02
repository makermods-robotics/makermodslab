"""Robot side of the ADAPTIVE-SYNC remote-inference loop (runs ON the robot).

This is a sibling of ``robot.py`` in the earlier ``lerobot_inference`` prototype
(a separate project — NOT a directory in this repo). Same robot/policy split
over LiveKit Portal, same unchanged ``policy.py`` on the other end (one plain
``act`` chunk per observation, correlated by ``in_reply_to_ts_us``). The only
thing that changes is the on-robot chunk player.

Why a third player at all
-------------------------
``lerobot_inference``'s ``robot.py`` (``BlockChunkPlayer``) plays each chunk to
completion — exactly ONE seam per chunk boundary — and prefetches the next chunk
at a FIXED ``prefetch_lead``. That is the right regime for a policy WITHOUT
inpainting (ACT, diffusion, or any policy you do not want to guide): a
play-to-completion block keeps the arm in the open-loop distribution the policy
was evaluated in, and a single seam per boundary is the minimum a chunked remote
loop can achieve. Its weakness is the *fixed* lead:

  * too small -> under network latency the runway drains before the next chunk
    lands, so the arm STARVES (freezes) at every boundary;
  * too large -> at low latency it prefetches far too early, holding a stale plan
    and wasting round-trips.

``lerobot_drtc``'s ``robot.py`` (the other earlier prototype, also not in this
repo; ``ActionSchedule``) fixes latency the other way:
re-plan as fast as latency allows and merge overlapping chunks last-write-wins.
But every merge is a HARD switch, and only the flow/diffusion (inpainting) half
of DRTC can make those switches dynamically consistent. For a NON-inpainting
policy, faster re-planning just multiplies hard seams — the opposite of smooth.

Adaptive-sync (this file, via ``AdaptiveBlockPlayer`` in ``_sync_player.py``)
keeps the play-to-completion / one-seam-per-boundary contract but makes the
prefetch lead ADAPTIVE to measured round-trip latency, tracked with the same
Jacobson–Karels estimator DRTC uses:

    lead = estimator.estimate_steps + base_lead        # clamped to [1, H-1]

  * LOW latency  -> lead small -> request near the END of the chunk -> minimal
    prefetch, near-zero overlap, smoothest execution.
  * HIGH latency -> lead grows -> request EARLIER so the next chunk arrives
    before the current one drains -> no starvation.
  * latency beyond one chunk (``lead >= H - s_min``) -> degrade to strict
    play-to-drain: request at the boundary and HOLD the last action (freeze)
    rather than jump to a stale plan.

In one line: adaptive-sync tunes *WHEN* the robot asks for the next block, never
*HOW OFTEN* it overwrites the plan — so the seam count stays at one per boundary
while the arm stops starving under latency. See ``SYNC_RESULTS.md`` for the
simulation numbers, and ``_sync_player.py`` to reproduce them.

Run it with the same ``--robot.*`` CLI you already use for `lerobot-record`::

    python -m makermodslab.drtc.robot_sync \
        --robot.type=so101_follower \
        --robot.port=/dev/ttyACM1 \
        --robot.id=my_awesome_follower_arm \
        --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
        --horizon=16

    # baseline (fixed lead, like the lerobot_inference prototype's robot.py):
    python -m makermodslab.drtc.robot_sync --robot.type=so101_follower ... \
        --no-adaptive --base_lead=2

``--horizon`` MUST match policy.py's ``--horizon`` and should equal the
checkpoint's ``n_action_steps`` (and be ``<=`` its ``chunk_size``) so one
transmitted chunk is exactly one open-loop block. LiveKit creds + room come from
`~/.cache/huggingface/lerobot/livekit.env` (a local SFU's override comes from
`~/.cache/huggingface/lerobot/livekit.local.env`, written by
`tools/drtc/local_sfu*.sh`; see `_env.load_env` for the full precedence). The
remote policy runs in this package's `policy.py`, launched on Modal by
`modal run makermodslab/drtc/modal_policy.py` (see docs/drtc/README.md).

NOTE: no `from __future__ import annotations` here — it would stringize the
`main(cfg: RobotSideConfig)` signature and break draccus's config-type lookup.
"""

import asyncio
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import draccus
from livekit.portal import (
    ActionChunk,
    Robot,
    RobotConfig as PortalRobotConfig,
    VideoCodec,
)

# Importing the camera/robot configs registers them with draccus so
# `--robot.type=...` and `--robot.cameras='{...}'` resolve on the CLI. SO100 /
# SO101 followers live in `so_follower`; add more here as needed (same list as
# lerobot's async_inference.robot_client).
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins

from ._common import fmt_us, load_env, mint_token, required_env
from ._latency import JKLatencyEstimator
from ._schema import CHUNK_NAME, robot_wire_schema
from ._sync_player import AdaptiveBlockPlayer

# Register any third-party robot/camera plugins (entry points) BEFORE draccus
# parses `--robot.type`. Built-in so100/so101 register on import above.
register_third_party_plugins()

IDENTITY = "robot"


@dataclass
class RobotSideConfig:
    robot: RobotConfig = field(metadata={"help": "lerobot robot configuration"})
    fps: int = field(default=30, metadata={"help": "Control/stream frequency"})
    horizon: int = field(
        default=16,
        metadata={
            "help": "Action-chunk horizon; MUST match policy.py's --horizon. "
            "Set to the policy's n_action_steps (<= chunk_size) so one "
            "transmitted chunk is one open-loop block."
        },
    )
    duration_s: float = field(default=0.0, metadata={"help": "Run duration in seconds; 0 = run until Ctrl-C"})
    video_codec: str = field(
        default="H264",
        metadata={
            "help": "Portal video codec: H264 (default; WebRTC RTP, inter-frame — the right "
            "choice at >=480p or multi-camera, where it compresses to ~1 chunk/frame) "
            "or MJPEG (per-frame byte-stream, lowest latency ONLY when each frame fits "
            "one 15KB SCTP chunk, i.e. <=~256px inference resolution; at 480x640 it "
            "congests the shared data channel and doubles e2e). MUST match policy.py's "
            "--video-codec. Rule of thumb: MJPEG for <=256px policies (e.g. SmolVLA), "
            "H264 for 480x640 policies (e.g. ACT)."
        },
    )
    video_quality: int = field(default=90, metadata={"help": "MJPEG quality 1..100"})
    video_bitrate_kbps: int = field(
        default=4000,
        metadata={
            "help": "H264 max-bitrate ceiling (kbps). The portal default of 10000 "
            "can saturate a thin uplink and head-of-line-block the state "
            "data channel, so state lags the real-time video and the policy "
            "drops it in sync. Ignored for MJPEG/PNG/RAW."
        },
    )
    reliable_state: bool = field(
        default=False,
        metadata={
            "help": "Force state onto the reliable data channel. Leave False to let it "
            "AUTO-follow the codec: reliable for byte-stream video (MJPEG/PNG/RAW, "
            "which shares the SCTP transport with state) and unreliable for H264 "
            "(separate RTP transport, so state stays real-time instead of queuing "
            "behind retransmits). This flag only forces reliable ON regardless."
        },
    )

    # --- adaptive-sync knobs ----------------------------------------------
    adaptive: bool = field(
        default=True,
        metadata={
            "help": "Adapt the prefetch lead to measured round-trip latency "
            "(lead = estimate_steps + base_lead). --no-adaptive falls "
            "back to a FIXED lead of base_lead, i.e. the "
            "lerobot_inference prototype's robot.py baseline."
        },
    )
    base_lead: int = field(
        default=2,
        metadata={
            "help": "Margin (epsilon) added to the latency estimate to form the "
            "prefetch lead; in --no-adaptive mode this IS the fixed lead. "
            "Request the next chunk once the runway drops to `lead`."
        },
    )
    s_min: int = field(
        default=4,
        metadata={
            "help": "Min execution budget: when estimate_steps + base_lead >= "
            "horizon - s_min the latency no longer fits one chunk, so the "
            "player degrades to play-to-drain-and-hold."
        },
    )
    align: bool = field(
        default=True,
        metadata={
            "help": "Resume each incoming chunk at the step aligned to NOW "
            "(drop the stale prefix) instead of index 0. Fixes the "
            "~lead-step backward SNAP at every block boundary. "
            "--no-align reverts to raw play-from-0 for A/B comparison."
        },
    )
    action_delay: int = field(
        default=1,
        metadata={
            "help": "Assumed offset (ticks) from an observation to its first "
            "action; sets where an aligned chunk resumes on the step axis."
        },
    )
    latency_alpha: float = field(default=0.125, metadata={"help": "JK estimator smoothing (RTT)"})
    latency_beta: float = field(default=0.25, metadata={"help": "JK estimator smoothing (deviation)"})
    latency_k: float = field(default=1.5, metadata={"help": "JK estimator deviation multiplier"})


async def run(cfg: RobotSideConfig) -> None:
    load_env()
    url = required_env("LIVEKIT_URL")
    room = required_env("LIVEKIT_ROOM")
    token = mint_token(IDENTITY, room)

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    schema, state_keys, action_keys = robot_wire_schema(robot)
    print(
        f"[robot] hardware schema: state={len(state_keys)} action={len(action_keys)} cameras={schema.cameras}"
    )
    print(f"[robot]   state keys : {state_keys}")
    print(f"[robot]   action keys: {action_keys}")

    codec = getattr(VideoCodec, cfg.video_codec.upper())

    portal_cfg = PortalRobotConfig(room)
    for cam in schema.cameras:
        portal_cfg.add_video(
            cam,
            codec=codec,
            quality=cfg.video_quality,
            max_bitrate_kbps=cfg.video_bitrate_kbps,
        )
    portal_cfg.add_state_typed(schema.state_fields())
    portal_cfg.add_action_chunk(CHUNK_NAME, horizon=cfg.horizon, fields=schema.action_fields())
    portal_cfg.set_fps(cfg.fps)
    # H264 (RTP) video and reliable state (SCTP) travel different transports; under
    # upload pressure the reliable channel lags and misaligns state/video at the
    # policy's sync buffer. So for H264 state stays UNRELIABLE (real-time). Byte-stream
    # video (MJPEG/PNG/RAW) shares the SCTP transport with state, so there state SHOULD
    # be reliable to stay in lockstep. Auto-follow the codec; --reliable_state forces on.
    _byte_stream_video = cfg.video_codec.upper() in ("MJPEG", "PNG", "RAW")
    portal_cfg.set_state_reliable(cfg.reliable_state or _byte_stream_video)

    portal = Robot(portal_cfg)

    # The player owns the JK estimator; we build it explicitly so the CLI knobs
    # flow through and so the log line can read its state each second.
    estimator = JKLatencyEstimator(
        fps=cfg.fps,
        action_chunk_size=cfg.horizon,
        s_min=cfg.s_min,
        alpha=cfg.latency_alpha,
        beta=cfg.latency_beta,
        k=cfg.latency_k,
    )
    player = AdaptiveBlockPlayer(
        horizon=cfg.horizon,
        fps=cfg.fps,
        adaptive=cfg.adaptive,
        margin=cfg.base_lead,
        s_min=cfg.s_min,
        align=cfg.align,
        action_delay=cfg.action_delay,
        estimator=estimator,
    )

    # push()/record_roundtrip() run on a Portal transport thread; step() /
    # should_request() run on the control loop below. Guard the shared player
    # (and its estimator) with a single lock.
    lock = threading.Lock()
    chunks_received = 0
    uncorrelated_chunks = 0
    # Maps an emitted observation's timestamp -> the control tick it was sent on,
    # so a returned chunk (which carries in_reply_to_ts_us) can be aligned onto
    # the absolute step axis. Bounded; oldest entries evicted FIFO.
    sent_obs: OrderedDict[int, int] = OrderedDict()
    obs_ttl = 4 * cfg.horizon

    def on_chunk(chunk: ActionChunk) -> None:
        # Correlate the returned chunk to the observation that produced it via
        # `in_reply_to_ts_us` (the obs timestamp the policy echoes back): feed the
        # true end-to-end round-trip to the estimator so the prefetch lead tracks
        # real latency, and recover the control tick that observation was sent on
        # so the player can align the swap to NOW. Then stage the chunk; step()
        # swaps it in when the current block drains (never truncating it).
        nonlocal chunks_received, uncorrelated_chunks
        now_us = int(time.time() * 1_000_000)
        reply_ts = chunk.in_reply_to_ts_us
        with lock:
            chunks_received += 1
            src_step = sent_obs.get(reply_ts) if reply_ts else None
            if reply_ts:
                player.record_roundtrip((now_us - reply_ts) / 1_000_000.0)
            if src_step is None:
                # No matching obs (evicted / no reply ts): fall back to play-from-0
                # for this chunk by leaving src_step None (align no-ops).
                uncorrelated_chunks += 1
            player.push(chunk, src_step)

    portal.on_action_chunk(CHUNK_NAME, on_chunk)

    print(f"[robot] connecting to {url} as '{IDENTITY}' in room '{room}' ...")
    await portal.connect(url, token)
    print(
        f"[robot] connected; {cfg.fps} fps, block horizon {cfg.horizon}, "
        f"adaptive={'on' if cfg.adaptive else 'off'} (base_lead={cfg.base_lead}, "
        f"s_min={cfg.s_min}), align={'on' if cfg.align else 'off'}"
    )

    interval = 1.0 / cfg.fps
    start = time.monotonic()
    next_tick = start
    last_log = start
    observations_emitted = 0
    control_step = 0
    last_action: dict[str, float] | None = None

    try:
        while cfg.duration_s <= 0 or (time.monotonic() - start) < cfg.duration_s:
            ts_us = int(time.time() * 1_000_000)

            # 1. Execute one action from the current block. On a dry schedule
            #    (the next block hasn't landed yet — the degrade/starvation case),
            #    hold position by re-commanding the last action rather than
            #    jumping to a stale plan.
            with lock:
                player.current_step = control_step  # for chunk alignment on swap
                cmd = player.step()
                want_obs = player.should_request()
                runway = player.remaining()
                lead = player.current_lead()
            if cmd is not None:
                action = {action_keys[idx]: cmd[f"a{idx}"] for idx in range(len(action_keys))}
                robot.send_action(action)
                last_action = action
            elif last_action is not None:
                robot.send_action(last_action)

            # 2. Request the NEXT block only when the runway drops to the ADAPTIVE
            #    lead — one inference per block, timed to the measured round-trip
            #    so the block lands just before the current one drains.
            if want_obs:
                obs = robot.get_observation()
                state = {f"s{idx}": float(obs[k]) for idx, k in enumerate(state_keys)}
                portal.send_state(state, timestamp_us=ts_us)
                for cam in schema.cameras:
                    frame = obs.get(cam)
                    if frame is not None:
                        portal.send_video_frame(cam, frame, timestamp_us=ts_us)
                # Remember which tick this obs went out on, so its returned chunk
                # can be aligned to NOW on swap (see on_chunk / player alignment).
                with lock:
                    sent_obs[ts_us] = control_step
                    while len(sent_obs) > obs_ttl:
                        sent_obs.popitem(last=False)
                observations_emitted += 1

            now = time.monotonic()
            if now - last_log >= 1.0:
                m = portal.metrics()
                with lock:
                    chunks = chunks_received
                    uncorr = uncorrelated_chunks
                    age = player.age_ms()
                    est_steps = estimator.estimate_steps
                    est_ms = estimator.estimate_seconds * 1000.0
                    degrade = player.in_degrade
                    holds = player.holds
                age_str = "-" if age is None else f"{age:.0f}ms"
                print(
                    f"[robot] t={int(now - start):>3}s chunks={chunks} reqs={observations_emitted} "
                    f"sched={runway} lead={lead} lat={est_steps}st/{est_ms:.0f}ms "
                    f"holds={holds}{' DEGRADE' if degrade else ''} chunk_age={age_str} "
                    f"active={portal.active_operator()} "
                    f"e2e={fmt_us(m.policy.e2e_us_p50)}/{fmt_us(m.policy.e2e_us_p95)} (p50/p95) "
                    f"rtt={fmt_us(m.rtt.rtt_us_last)}" + (f" uncorr={uncorr}" if uncorr else "")
                )
                last_log = now

            control_step += 1
            next_tick += interval
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
    finally:
        print("[robot] disconnecting...")
        await portal.disconnect()
        portal.close()
        robot.disconnect()


@draccus.wrap()
def main(cfg: RobotSideConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
