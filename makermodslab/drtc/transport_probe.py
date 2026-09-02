#!/usr/bin/env python3
"""Transport probe: verify a LiveKit path carries the DRTC loop, with NO robot
and NO GPU — a synthetic robot on one machine and an echo operator on another.

    # machine A (the robot station; SFU runs here, see tools/drtc/local_sfu*.sh)
    python makermodslab/drtc/transport_probe.py robot --url ws://127.0.0.1:7880 \
        --api-key <key> --api-secret <secret>

    # machine B (the would-be GPU peer), reaching the SFU over the tailnet
    python makermodslab/drtc/transport_probe.py operator --url ws://100.x.y.z:7880 \
        --api-key <key> --api-secret <secret>

The robot role publishes `--cameras` synthetic H264 streams at `--width x
--height` plus a state vector at `--fps`, and expects an action chunk back for
every observation. The operator role replies to each observation immediately
with a zero chunk tagged `in_reply_to_ts_us`, so the robot's e2e is pure
transport (encode + network + sync buffer), with inference removed. Both sides
print a 1 Hz line; the robot's `e2e` (p50/p95) is the number to compare across
transports (LiveKit Cloud vs local SFU vs tailnet).

Which path the media actually took is NOT visible from here — read the SFU
log's `participant active` line: the `[selected]` candidate pair names the
IPs in use (a 100.64.0.0/10 address = tailnet, a public IP = hole punch).

Self-contained on purpose: the far end needs only this file and
    pip install "livekit-portal==0.2.4" "livekit-api>=0.7" numpy
(python >= 3.12). Mirrors robot_sync.py / policy.py's Portal setup exactly —
same codec, same reliable-state rule, same schema shape — so a pass here
means the real scripts will connect over the same path.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import time
from collections import deque

import numpy as np
from livekit import api
from livekit.portal import (
    ActionChunk,
    DType,
    Observation,
    Operator,
    OperatorConfig,
    Robot,
    RobotConfig,
    VideoCodec,
)

CHUNK_NAME = "act"


def fmt_us(value) -> str:
    return "-" if value is None else f"{value / 1000:.0f}ms"


def mint_token(key: str, secret: str, identity: str, room: str) -> str:
    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_update_own_metadata=True,  # Portal sets lk.portal.role on connect
    )
    return (
        api.AccessToken(key, secret)
        .with_identity(identity)
        .with_grants(grants)
        .with_ttl(datetime.timedelta(hours=6))
        .to_jwt()
    )


def configure(cfg, args, horizon: int):
    """Shared Portal schema for both roles (must match, like the real scripts)."""
    codec = getattr(VideoCodec, args.video_codec.upper())
    for i in range(args.cameras):
        if isinstance(cfg, RobotConfig):
            cfg.add_video(f"cam{i}", codec=codec, quality=90, max_bitrate_kbps=args.video_bitrate_kbps)
        else:
            cfg.add_video(f"cam{i}", codec=codec)
    cfg.add_state_typed([(f"s{i}", DType.F64) for i in range(args.state_dim)])
    cfg.add_action_chunk(
        CHUNK_NAME,
        horizon=horizon,
        fields=[(f"a{i}", DType.F64) for i in range(args.action_dim)],
    )
    cfg.set_fps(args.fps)
    # Same rule as robot_sync.py / robot_rtc.py: state rides the reliable
    # channel only when video is a byte-stream codec sharing that transport.
    cfg.set_state_reliable(args.video_codec.upper() in ("MJPEG", "PNG", "RAW"))
    return cfg


def synth_frame(h: int, w: int, t: int) -> np.ndarray:
    """A moving gradient + noise so H264 has real motion to encode (a static
    frame would compress to nothing and understate the media load)."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = ((xx + yy + 4 * t) % 256).astype(np.uint8)
    rng = np.random.default_rng(t)
    noise = rng.integers(0, 32, size=(h, w), dtype=np.uint8)
    frame = np.stack([base, base[::-1], noise + base // 2], axis=-1)
    return np.ascontiguousarray(frame)


async def run_robot(args) -> None:
    cfg = configure(RobotConfig(args.room), args, args.horizon)
    portal = Robot(cfg)
    chunks = 0
    rtts: deque = deque(maxlen=300)

    def on_chunk(chunk: ActionChunk) -> None:
        nonlocal chunks
        chunks += 1
        if chunk.in_reply_to_ts_us:
            rtts.append((time.time() * 1e6 - chunk.in_reply_to_ts_us) / 1000.0)

    portal.on_action_chunk(CHUNK_NAME, on_chunk)
    token = mint_token(args.api_key, args.api_secret, "robot", args.room)
    print(
        f"[probe-robot] connecting {args.url} room={args.room} "
        f"cams={args.cameras}@{args.width}x{args.height} {args.video_codec} fps={args.fps}"
    )
    await portal.connect(args.url, token)
    print(f"[probe-robot] connected as '{portal.local_identity()}'; waiting for operator ...")

    # Pre-render a short loop of frames: the probe measures transport, not
    # numpy, so frame synthesis must not eat the 33 ms tick.
    loop_frames = [synth_frame(args.height, args.width, t) for t in range(30)]
    state = {f"s{i}": 0.0 for i in range(args.state_dim)}

    interval = 1.0 / args.fps
    start = time.monotonic()
    next_tick = start
    last_log = start
    sent = 0
    tick = 0
    try:
        while args.duration <= 0 or time.monotonic() - start < args.duration:
            ts_us = int(time.time() * 1_000_000)
            frame = loop_frames[tick % len(loop_frames)]
            for i in range(args.state_dim):
                state[f"s{i}"] = float(np.sin(tick / 30.0 + i))
            portal.send_state(state, timestamp_us=ts_us)
            for c in range(args.cameras):
                portal.send_video_frame(f"cam{c}", frame, timestamp_us=ts_us)
            sent += 1
            tick += 1

            now = time.monotonic()
            if now - last_log >= 1.0:
                m = portal.metrics()
                r = sorted(rtts)
                p50 = f"{r[len(r) // 2]:.0f}ms" if r else "-"
                p95 = f"{r[int(len(r) * 0.95)]:.0f}ms" if r else "-"
                print(
                    f"[probe-robot] t={int(now - start):>3}s sent={sent} chunks={chunks} "
                    f"active={portal.active_operator()} "
                    f"e2e={fmt_us(m.policy.e2e_us_p50)}/{fmt_us(m.policy.e2e_us_p95)} (p50/p95) "
                    f"reply_rtt={p50}/{p95} rtt={fmt_us(m.rtt.rtt_us_last)}"
                )
                last_log = now
            next_tick += interval
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
    finally:
        await portal.disconnect()
        portal.close()
        print(f"[probe-robot] done: sent={sent} chunks={chunks}")


async def run_operator(args) -> None:
    cfg = configure(OperatorConfig(args.room), args, args.horizon)
    if args.slack is not None:
        cfg.set_slack(args.slack)
    op = Operator(cfg)
    loop = asyncio.get_running_loop()
    obs_queue: deque = deque(maxlen=2)
    obs_event = asyncio.Event()
    zero_chunk = np.zeros((args.horizon, args.action_dim), dtype=np.float32)

    def on_observation(obs: Observation) -> None:
        obs_queue.append(obs)
        loop.call_soon_threadsafe(obs_event.set)

    op.on_observation(on_observation)
    token = mint_token(args.api_key, args.api_secret, "policy", args.room)
    print(f"[probe-operator] connecting {args.url} room={args.room}")
    await op.connect(args.url, token)
    me = op.local_identity()
    print(f"[probe-operator] connected as '{me}'; claiming control ...")
    try:
        await op.set_active_operator(me)
    except Exception as exc:  # noqa: BLE001 — robot may not be there yet
        print(f"[probe-operator] claim deferred ({exc}); retrying as robot appears")

    start = time.monotonic()
    last_log = start
    seen = 0
    replied = 0
    try:
        while args.duration <= 0 or time.monotonic() - start < args.duration:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(obs_event.wait(), timeout=1.0)
            obs_event.clear()
            while obs_queue:
                obs = obs_queue.popleft()
                seen += 1
                if op.active_operator() != me:
                    with contextlib.suppress(Exception):
                        await op.set_active_operator(me)
                op.send_action_chunk(CHUNK_NAME, zero_chunk, in_reply_to_ts_us=obs.timestamp_us)
                replied += 1
            now = time.monotonic()
            if now - last_log >= 1.0:
                m = op.metrics()
                vbuf = ",".join(f"{k}:{v}" for k, v in m.buffers.video_fill.items()) or "-"
                fjit = ",".join(f"{k}:{fmt_us(v)}" for k, v in m.transport.frame_jitter_us.items()) or "-"
                print(
                    f"[probe-operator] t={int(now - start):>3}s obs={seen} replied={replied} "
                    f"stale={m.sync.stale_observations_emitted} "
                    f"match_delta={fmt_us(m.sync.match_delta_us_p50)}/{fmt_us(m.sync.match_delta_us_p95)} "
                    f"blocker={m.sync.last_blocker_track or '-'} vbuf=[{vbuf}] fjitter=[{fjit}]"
                )
                last_log = now
    finally:
        await op.disconnect()
        op.close()
        print(f"[probe-operator] done: obs={seen} replied={replied}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("role", choices=["robot", "operator"])
    p.add_argument("--url", required=True, help="SFU signaling URL (ws:// or wss://)")
    p.add_argument("--api-key", required=True)
    p.add_argument("--api-secret", required=True)
    p.add_argument("--room", default="portal-transport-probe")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--cameras", type=int, default=2)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--video-codec", default="H264", help="H264 (default) or MJPEG/PNG/RAW")
    p.add_argument("--video-bitrate-kbps", type=int, default=4000)
    p.add_argument("--state-dim", type=int, default=6)
    p.add_argument("--action-dim", type=int, default=6)
    p.add_argument("--slack", type=int, default=None, help="operator sync-buffer slack in ticks")
    p.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until Ctrl-C")
    args = p.parse_args()
    asyncio.run(run_robot(args) if args.role == "robot" else run_operator(args))


if __name__ == "__main__":
    main()
