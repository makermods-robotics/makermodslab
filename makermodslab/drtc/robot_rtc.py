"""Robot CLIENT for the FULL DRTC port — RTC in-painting over Portal (robot side).

This is ``robot.py`` from the earlier ``lerobot_drtc`` prototype (a separate
project, NOT a directory in this repo — the half-DRTC robot: absolute-step LWW
scheduling + JK-paced observation emission) upgraded to drive the *full* DRTC
loop with RTC guided in-painting on the policy server (``policy_rtc.py``).

The policy-agnostic half is unchanged in spirit: every returned chunk is placed
on one absolute action-step axis so its stale prefix is dropped (alignment
offset for free), and overlapping chunks are merged last-write-wins by
observation freshness. What the full port ADDS is the in-painting request: when
the robot emits an observation it also tells the server (via extra STATE fields,
see ``_schema_rtc.py``) how to in-paint the next chunk against the still-to-
execute prefix of the current plan:

    rtc_d            = estimator.estimate_steps          (measured latency, steps)
    rtc_overlap_end  = H - max(s_min, d)                 (fresh-region boundary)
    rtc_src_ts       = source obs timestamp (µs) of the prefix's source chunk
    rtc_prefix_start = start index within that cached raw chunk
    rtc_prefix_len   = prefix length (steps); 0 = no prefix

The prefix is a SINGLE contiguous same-source run drawn from the schedule at/
after ``current_step`` (``_rtc.ActionSchedule.prefix_span``) — the single-source
simplification of the reference's multi-span ``get_masking_chunk_spans``
(documented in ``_schema_rtc.py``). ``rtc_src_ts`` is the source obs timestamp,
which is exactly the key the server cached the raw chunk under
(``in_reply_to_ts_us``) — so the server slices ``raw_cache[src_ts][start:start+len]``
with no extra correlation state.

Freshness / provenance ride on ``ActionChunk.in_reply_to_ts_us`` (the obs
timestamp the policy answered), same as the half-DRTC robot. The only new state
per scheduled action is that source timestamp + its index within the source
chunk, tracked so the next request's prefix descriptor can be recovered.

Run it with the same ``--robot.*`` CLI as ``lerobot-record`` / the half-DRTC
robot; RTC/pacing knobs are extra flags with sane defaults::

    python -m makermodslab.drtc.robot_rtc \
        --robot.type=so100_follower --robot.port=/dev/ttyACM1 \
        --robot.id=my_arm \
        --robot.cameras='{ up: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}}'

``--horizon`` MUST match ``policy_rtc.py`` (PORTAL_HORIZON), and ``--s_min``
should match the server's so the ``overlap_end`` math agrees. Only smolvla / pi0
/ pi05 policies actually in-paint on the server; ACT etc. serve plain chunks (the
robot still sends the RTC fields; they're ignored) — see ``policy_rtc.py``.

draccus has NO ``--no-<flag>`` form for booleans (it is not argparse's
BooleanOptionalAction): turn one off with ``--<flag> false`` / ``--<flag>=False``.

Supervised operation (S3.5)
---------------------------
This entrypoint is now a session's child exactly as ``robot_sync`` is, and it
runs the SAME glue (``._session_glue``) rather than a second copy of it: the
``READY``/``CONNECTED``/``ACTIVE``/``STATS``/``RETURNING``/``BYE`` events, the
``STOP``/``STOP``/``QUIT`` stdin protocol, the start-pose capture and
return-to-rest, the first-action ease-in, the torque-cap reset at connect and
the interrupt-shielded teardown. ``--livekit_url`` / ``--livekit_room`` pin the
transport a parent already verified; ``--return_to_rest`` / ``--ease_in``
default ON and exist for bench A/B only. Ctrl-C takes the same path as ``STOP``,
and a second one cuts the return short.

Why this file exists as a session engine at all: at ~400 ms end-to-end a
flow-policy run under ``robot_sync`` re-plans about once a second, and two
SmolVLA plans made 400 ms apart disagree at every seam — the arm twitches at
~1 Hz with a perfectly healthy transport. That is what in-painting removes, and
it is why the engine is a session option rather than a bench script.

Not portable back to the standalone repo
----------------------------------------
Same accepted cost of integration ``robot_sync`` took: this file now imports
``makermodslab.drtc_protocol`` and ``._session_glue`` (which reaches
``makermodslab.rest_pose``). ``_rtc.py``, ``_schema_rtc.py``, ``_latency.py``,
``_filter.py`` and ``policy_rtc.py`` remain dependency-free and still are.

The STATS key set is FROZEN (``drtc_protocol.STATS_KEYS``) and is shared with
the adaptive-sync engine, so this loop reports its own quantities through it:
``lead`` is ``max(s_min, d)`` (the steps of an arriving chunk the round trip has
already consumed — the RTC analogue of a prefetch lead, and the ``s=`` term in
the human log line), and ``degrade`` is ``lead >= horizon - s_min``, which is
ANALYSIS §8.1's budget rule (``roundtrip < (H - s_min)/fps``) and the very same
predicate ``_sync_player.in_degrade`` uses. Numbers with no analogue on the other
engine — ``merge_l2``, ``prefix``, ``smooth``, ``emit``, ``ret``, ``late``,
``stale`` — stay on the human ``[robot]`` line and are NOT smuggled into STATS.

NOTE: no `from __future__ import annotations` — it would stringize the
`main(cfg: RobotSideRTCConfig)` signature and break draccus's config-type lookup.
"""

import asyncio
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import draccus
import numpy as np
from livekit.portal import (
    ActionChunk,
    Robot,
    RobotConfig as PortalRobotConfig,
    VideoCodec,
)

# Importing the camera/robot configs registers them with draccus so
# `--robot.type=...` / `--robot.cameras='{...}'` resolve on the CLI.
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.motors import MotorNormMode
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins

from ..drtc_protocol import (
    EVENT_BYE,
    EVENT_CONNECTED,
    EVENT_ERROR,
    EVENT_READY,
    EVENT_RETURNING,
    format_ready,
)
from ..motor_power import FOLLOWER, reset_torque_limit
from ._common import fmt_us, load_env, mint_token, required_env
from ._filter import ButterworthLowpass
from ._latency import JKLatencyEstimator
from ._pose import feetech_buses
from ._rtc import ActionSchedule
from ._schema_rtc import (
    CHUNK_NAME,
    RTC_D,
    RTC_OVERLAP_END,
    RTC_PREFIX_LEN,
    RTC_PREFIX_START,
    RTC_SRC_TS,
    empty_rtc_meta,
    robot_wire_schema,
    rtc_state_fields,
)
from ._session_glue import (
    LoopControl,
    _shielded_disconnect,
    capture_start_poses_or_warn,
    ease_in_field,
    ease_into_first_action,
    emit,
    emit_stats,
    livekit_room_field,
    livekit_token_field,
    livekit_url_field,
    note_first_operator,
    or_none,
    return_step,
    return_to_rest_field,
    shielded,
)

register_third_party_plugins()

IDENTITY = "robot"


def action_spans(robot, action_keys) -> np.ndarray | None:
    """Full calibrated travel per action dim, in that dim's own native units.

    The action vector mixes units (SO-10x: body joints in degrees, gripper in
    0-100 %-of-range), so cross-joint comparisons must be scaled to %-of-range
    first. RANGE_* dims have a fixed span by definition; DEGREES dims span
    their calibrated travel (raw encoder ticks -> degrees via the model's
    resolution). Returns None if this robot doesn't expose a single bus with
    per-motor calibration (e.g. bimanual wrappers) — callers fall back to
    native units.
    """
    try:
        spans = []
        for key in action_keys:
            motor = key.split(".")[0]
            m = robot.bus.motors[motor]
            if m.norm_mode is MotorNormMode.RANGE_0_100:
                spans.append(100.0)
            elif m.norm_mode is MotorNormMode.RANGE_M100_100:
                spans.append(200.0)
            else:  # DEGREES
                cal = robot.calibration[motor]
                res = robot.bus.model_resolution_table[m.model]
                spans.append((cal.range_max - cal.range_min) * 360.0 / res)
        return np.array(spans)
    except (AttributeError, KeyError):
        return None


@dataclass
class RobotSideRTCConfig:
    robot: RobotConfig = field(metadata={"help": "lerobot robot configuration"})
    fps: int = field(default=30, metadata={"help": "Control/stream frequency"})
    horizon: int = field(
        default=50,
        metadata={
            "help": "Action-chunk horizon; MUST match policy_rtc.py --horizon. 50 = the "
            "full smolvla/pi0/pi05 chunk_size (the model always denoises 50; "
            "smaller values just truncate). Must not exceed the model's chunk_size."
        },
    )
    duration_s: float = field(default=0.0, metadata={"help": "Run duration in seconds; 0 = run until Ctrl-C"})
    video_codec: str = field(
        default="H264",
        metadata={"help": "Portal video codec: H264 (low-latency, default) or MJPEG/PNG/RAW (frame-exact)"},
    )
    video_quality: int = field(default=90, metadata={"help": "MJPEG quality 1..100"})
    video_bitrate_kbps: int = field(
        default=4000,
        metadata={
            "help": "H264 max-bitrate ceiling (kbps). The portal default of 10000 "
            "can saturate a thin uplink and head-of-line-block the state "
            "data channel, so state lags the real-time video and the policy "
            "drops it in sync ([sync-drop] 'video N ms ahead'). Ignored for MJPEG."
        },
    )
    reliable_state: bool = field(
        default=False,
        metadata={
            "help": "Force state onto the reliable data channel. Leave False to let it "
            "AUTO-follow the codec (same rule as robot_sync.py): reliable for "
            "byte-stream video (MJPEG/PNG/RAW, which shares the SCTP transport "
            "with state) and unreliable for H264 (separate RTP transport, so "
            "state stays real-time instead of queuing behind retransmits). "
            "This flag only forces reliable ON regardless."
        },
    )

    # --- DRTC scheduler / RTC knobs ---------------------------------------
    pacing: bool = field(
        default=True,
        metadata={"help": "Gate observation emission on runway (cooldown pacing). False = emit every tick."},
    )
    s_min: int = field(
        default=4,
        metadata={
            "help": "Min execution horizon: request a fresh chunk once the runway "
            "drops to horizon - s_min. Also sets overlap_end = H - max(s_min, d). "
            "Match policy_rtc.py --s-min."
        },
    )
    epsilon: int = field(
        default=1, metadata={"help": "Cooldown safety margin (extra ticks between requests)"}
    )
    action_delay: int = field(
        default=1,
        metadata={
            "help": "Assumed offset (ticks) from an observation to its first action; "
            "where a returned chunk lands on the step axis."
        },
    )
    latency_alpha: float = field(default=0.125, metadata={"help": "JK estimator smoothing (RTT)"})
    latency_beta: float = field(default=0.25, metadata={"help": "JK estimator smoothing (deviation)"})
    latency_k: float = field(default=1.5, metadata={"help": "JK estimator deviation multiplier"})

    # --- action smoothing --------------------------------------------------
    lpf_hz: float = field(
        default=0.0,
        metadata={
            "help": "Butterworth low-pass cutoff (Hz) applied to commanded actions; "
            "0 = off. Smooths chunk-seam jerk at the cost of ~order/(2*pi*fc) s "
            "of command lag (e.g. order 2 @ 5 Hz -> ~64 ms). Must be < fps/2."
        },
    )
    lpf_order: int = field(default=2, metadata={"help": "Butterworth order; roll-off is 6*order dB/octave"})

    # --- transport pinning + graceful stop ---------------------------------
    # Defined once in `_session_glue` so this entrypoint and `robot_sync` cannot
    # drift on the flags a supervising parent passes both of them.
    livekit_url: str = livekit_url_field()
    livekit_room: str = livekit_room_field()
    livekit_token: str = livekit_token_field()
    return_to_rest: bool = return_to_rest_field()
    ease_in: bool = ease_in_field()


async def run(cfg: RobotSideRTCConfig) -> None:
    # A supervising parent drives these three over stdin (see drtc_protocol):
    # STOP -> stop_event (leave the loop, return, exit 0); a second STOP ->
    # abort_event (cut the return short); QUIT -> all three (no return at all).
    control = LoopControl()

    load_env()
    # Flags win over the credential files so a parent can pin the transport it
    # actually verified. The EFFECTIVE values — not what anyone believes they
    # passed — are what READY reports and what we dial.
    url = cfg.livekit_url or required_env("LIVEKIT_URL")
    room = cfg.livekit_room or required_env("LIVEKIT_ROOM")
    # A token handed to us is already scoped to this room and identity and was
    # signed by whoever owns the SFU's secret (the Lab, under `--sfu`). Minting
    # our own is the LiveKit Cloud path, and needs the API secret in this
    # process's environment.
    token = cfg.livekit_token or mint_token(IDENTITY, room)
    # Emitted BEFORE the bus is opened, so a parent that spots a transport
    # mismatch can kill the child before anything is energized.
    emit(EVENT_READY, format_ready(url, room))

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # Un-throttle the servos before anything commands them. `Torque_Limit` is a
    # RAM register that SURVIVES between sessions on one power-up, so a cap an
    # earlier auto-calibration left behind would silently throttle this whole
    # run: the arm tracks the policy sluggishly, everything looks healthy, and
    # nothing anywhere says why. Feetech-only, like the register itself. Stated
    # here rather than lifted into the glue because a source assertion pins the
    # call site (see tests/test_drtc_robot_rtc.py, and the note in
    # _session_glue's docstring on what deliberately did not move).
    if feetech_buses(robot):
        for warning in reset_torque_limit(robot, FOLLOWER):
            print(f"[robot] WARNING: {warning}")

    # Only now is it safe to read stdin — see LoopControl.start_command_pump for
    # why the pump cannot start before connect().
    control.start_command_pump()

    schema, state_keys, action_keys = robot_wire_schema(robot)
    print(
        f"[robot] hardware schema: state={len(state_keys)} action={len(action_keys)} cameras={schema.cameras}"
    )
    print(f"[robot]   state keys : {state_keys}")
    print(f"[robot]   action keys: {action_keys}")

    # Capture where the operator left the arm, now — after connect, before
    # anything moves — so every exit path can drive it back there while torque
    # is still on.
    start_poses = capture_start_poses_or_warn(robot, cfg.return_to_rest)

    portal = None
    try:
        codec = getattr(VideoCodec, cfg.video_codec.upper())

        portal_cfg = PortalRobotConfig(room)
        for cam in schema.cameras:
            portal_cfg.add_video(
                cam,
                codec=codec,
                quality=cfg.video_quality,
                max_bitrate_kbps=cfg.video_bitrate_kbps,
            )
        # Extended state schema: base s0..s{D-1} + RTC control fields. MUST match
        # policy_rtc.py exactly (Portal fingerprints the schema).
        portal_cfg.add_state_typed(rtc_state_fields(schema))
        portal_cfg.add_action_chunk(CHUNK_NAME, horizon=cfg.horizon, fields=schema.action_fields())
        portal_cfg.set_fps(cfg.fps)
        # With H264 (RTP media) video and reliable state (SCTP) travel different
        # transports; under upload pressure the reliable channel lags, misaligning
        # state/video at the policy's sync buffer. So for H264 state stays UNRELIABLE
        # (real-time). Byte-stream video (MJPEG/PNG/RAW) shares the SCTP transport
        # with state, so there state SHOULD be reliable to stay in lockstep.
        # Auto-follow the codec (mirrors robot_sync.py); --reliable_state forces on.
        _byte_stream_video = cfg.video_codec.upper() in ("MJPEG", "PNG", "RAW")
        portal_cfg.set_state_reliable(cfg.reliable_state or _byte_stream_video)

        portal = Robot(portal_cfg)

        # --- DRTC scheduler state --------------------------------------------
        schedule = ActionSchedule()
        estimator = JKLatencyEstimator(
            fps=cfg.fps,
            action_chunk_size=cfg.horizon,
            s_min=cfg.s_min,
            alpha=cfg.latency_alpha,
            beta=cfg.latency_beta,
            k=cfg.latency_k,
        )
        # Maps an emitted observation's timestamp (µs) -> the control tick it was
        # sent on, so a returned chunk (carrying in_reply_to_ts_us) can be placed on
        # the absolute step axis. Bounded FIFO.
        sent_obs: OrderedDict[int, int] = OrderedDict()
        obs_ttl = 4 * cfg.horizon

        # Touched by both the control loop and the on_chunk callback (a Portal
        # transport thread) — guard with a lock.
        lock = threading.Lock()
        chunks_received = 0
        uncorrelated_chunks = 0
        last_merge_l2 = 0.0
        last_prefix_len = 0
        last_chunk_at: float | None = None
        # --- latency attribution (ANALYSIS §8.1: where does the non-inference
        # ~330ms of e2e hide?). ret splits e2e into its two legs:
        #   e2e = forward(stamp -> operator observation emit) + inference + ret
        # ret  : server publish -> on_chunk here. Cross-host wall clock, so it is
        #        only meaningful with NTP-synced peers (Modal is; treat ±5ms).
        # emit : obs state timestamp (tick top) -> last wire emission this tick.
        #        Bounds the robot-side capture/serialize cost charged to e2e.
        # late : ticks whose work overran `interval`, so `sleep_for <= 0`, the loop
        #        never awaited, and queued chunk callbacks (dispatched onto THIS
        #        event loop, not a transport thread) could not run. Must stay 0.
        last_ret_ms = 0.0
        last_emit_ms = 0.0
        late_ticks = 0

        def on_chunk(chunk: ActionChunk) -> None:
            nonlocal chunks_received, uncorrelated_chunks, last_merge_l2, last_chunk_at
            nonlocal last_ret_ms
            now_us = int(time.time() * 1_000_000)
            reply_ts = chunk.in_reply_to_ts_us  # obs timestamp (µs) the policy answered
            with lock:
                chunks_received += 1
                last_chunk_at = time.monotonic()
                last_ret_ms = (now_us - chunk.timestamp_us) / 1000.0
                t_src = sent_obs.get(reply_ts) if reply_ts is not None else None
                if t_src is None:
                    # No matching observation (evicted / uncorrelated / no reply ts):
                    # treat as "fresh now" so we still use the chunk. Use now_us as
                    # the freshness/provenance key too (a server cache miss then just
                    # means the NEXT request can't reference this chunk as a prefix).
                    t_src = schedule.current_step
                    uncorrelated_chunks += 1
                    src_ts = now_us if reply_ts is None else reply_ts
                else:
                    src_ts = reply_ts
                if reply_ts:
                    estimator.update((now_us - reply_ts) / 1_000_000.0)
                # LWW-merge on the absolute step axis. src_ts (µs) is the freshness
                # key AND the server raw-cache key; chunk_start_step positions the
                # chunk; each merged step remembers its within-chunk index j.
                stats = schedule.merge(
                    chunk,
                    src_ts=src_ts,
                    chunk_start_step=t_src + cfg.action_delay,
                )
                last_merge_l2 = stats.mean_l2

        portal.on_action_chunk(CHUNK_NAME, on_chunk)

        print(f"[robot] connecting to {url} as '{IDENTITY}' in room '{room}' ...")
        await portal.connect(url, token)
        emit(EVENT_CONNECTED)
        print(
            f"[robot] connected; {cfg.fps} fps, horizon {cfg.horizon}, "
            f"pacing={'on' if cfg.pacing else 'off'} (s_min={cfg.s_min}) RTC in-painting"
        )

        interval = 1.0 / cfg.fps
        start = time.monotonic()
        next_tick = start
        last_log = start
        control_step = 0
        observations_emitted = 0
        stale_obs = 0
        obs_cooldown = cfg.s_min + cfg.epsilon
        last_action: dict[str, float] | None = None
        # Optional Butterworth low-pass over the commanded action stream. Ticks
        # every send (fresh AND hold) so its state sees a uniform-rate signal.
        lpf = ButterworthLowpass(cfg.lpf_hz, cfg.fps, cfg.lpf_order) if cfg.lpf_hz > 0 else None
        if lpf is not None:
            print(f"[robot] action low-pass: Butterworth order {cfg.lpf_order} @ {cfg.lpf_hz:g} Hz")
        # Smoothness instrumentation: peak per-tick command delta (max over joints)
        # within each 1s log window, pre-filter (raw target) vs post-filter (sent).
        # Peak, not RMS — seam/starvation-recovery spikes are the thing being fixed.
        # Deltas are scaled to %-of-calibrated-travel per joint before the max, so
        # degree dims and the 0-100 gripper dim are commensurable; falls back to
        # native mixed units (smooth= unsuffixed) when spans can't be derived.
        spans = action_spans(robot, action_keys)
        joint_names = [k.split(".")[0] for k in action_keys]
        if spans is None:
            print("[robot] WARNING: no per-motor calibration spans; smooth= stays in native units")
            span_scale, smooth_unit = np.ones(len(action_keys)), ""
        else:
            span_scale, smooth_unit = spans / 100.0, "%"
            print(
                "[robot] joint spans: "
                + " ".join(f"{n}={s:.0f}" for n, s in zip(joint_names, spans, strict=True))
            )
        prev_raw: np.ndarray | None = None
        prev_sent: np.ndarray | None = None
        dmax_raw = 0.0
        dmax_sent = 0.0
        dmax_raw_joint = "-"
        dmax_sent_joint = "-"
        # Gate emission on operator presence (see the control loop). Start False so the
        # first policy join logs a "resuming emission" transition.
        operator_present_prev = False
        # Ticks the schedule was dry and the arm held its last command. The exact
        # counter `_sync_player.AdaptiveBlockPlayer.holds` keeps, so `holds` means
        # the same thing on both engines: it climbs during warm-up and FREEZES on
        # a healthy run, which is why the panel renders its derivative.
        holds = 0
        # The ease-in fires on the first SCHEDULED action, which is also the first
        # thing ever sent: `last_action` stays None until then, so nothing has
        # commanded the arm and it is still exactly where the operator left it.
        eased = False
        active_seen = False

        while cfg.duration_s <= 0 or (time.monotonic() - start) < cfg.duration_s:
            if control.stop_event.is_set():
                # STOP / QUIT / Ctrl-C. Leave the loop; the finally below owns
                # the return-to-rest and the release.
                print("[robot] stop requested")
                break
            ts_us = int(time.time() * 1_000_000)
            tick_t0 = time.monotonic()  # same instant as ts_us; for emit= below

            # 1. Execute the action scheduled for this control tick. On a dry
            #    schedule (starvation) hold position by re-commanding the last
            #    action rather than jumping.
            with lock:
                schedule.current_step = control_step
                cmd = schedule.pop_current()
            if cmd is not None:
                # last_action holds the RAW (pre-filter) target so starvation
                # holds converge to the planned pose, not a lagged one.
                last_action = {action_keys[idx]: cmd[f"a{idx}"] for idx in range(len(action_keys))}
                if not eased:
                    eased = True
                    # FIRST-ACTION EASE-IN (see _session_glue.ease_into_first_action).
                    # Eased to the RAW target, not the filtered one: the low-pass
                    # (off by default) seeds its state to DC steady-state on its
                    # first sample, so its first output IS this pose either way.
                    #
                    # The loop is BLOCKED for the duration, and control_step and
                    # the wall clock freeze TOGETHER — so a chunk that lands mid-ease
                    # still merges onto the absolute step axis coherently (it is
                    # placed at `t_src + action_delay`, and t_src is a frozen tick).
                    # Only the step->wall-time mapping gains an offset, absorbed here
                    # rather than letting the pacing loop burst to catch up.
                    #
                    # Unlike robot_sync there is no single-outstanding-request flag
                    # here — pacing is a cooldown — so one or two requests emitted
                    # before the ease may answer into it. Their round-trip samples
                    # are inflated by the ease's duration and the JK estimate spikes;
                    # it is clamped to [1, H//2] and decays back over the next few
                    # samples. Same exposure robot_sync has, and self-healing.
                    if cfg.ease_in:
                        eased_for, stopped = ease_into_first_action(robot, last_action, control)
                        start += eased_for
                        last_log += eased_for
                        next_tick = time.monotonic()
                        if stopped:
                            # A stop landed during the ease; it came back
                            # cut-short. Skip execution entirely.
                            break
            else:
                holds += 1
            if last_action is not None:
                raw = np.array([last_action[k] for k in action_keys])
                sent = raw if lpf is None else lpf.filter(raw)
                if prev_raw is not None:
                    d_raw = np.abs(raw - prev_raw) / span_scale
                    d_sent = np.abs(sent - prev_sent) / span_scale
                    i, j = int(d_raw.argmax()), int(d_sent.argmax())
                    if d_raw[i] > dmax_raw:
                        dmax_raw, dmax_raw_joint = float(d_raw[i]), joint_names[i]
                    if d_sent[j] > dmax_sent:
                        dmax_sent, dmax_sent_joint = float(d_sent[j]), joint_names[j]
                prev_raw, prev_sent = raw, sent
                robot.send_action(dict(zip(action_keys, sent.tolist(), strict=True)))

            # 2. Decide whether to request a fresh inference this tick. Never emit
            #    when no operator (policy) is connected: with no subscriber the
            #    observation state/video byte streams just pile into the reliable
            #    data channel's send buffer until it overflows ("unable to send
            #    packet"), and that backlog then floods the policy's sync buffer the
            #    instant it finally joins ("state buffer full"). Hold emission until
            #    a policy is present; the schedule is dry meanwhile so the arm holds.
            active_seen = note_first_operator(portal, active_seen)
            operator_present = len(portal.operators()) > 0
            if operator_present != operator_present_prev:
                print(
                    f"[robot] operator present: {portal.operators()}; resuming emission"
                    if operator_present
                    else "[robot] no operator connected; pausing observation emission"
                )
                operator_present_prev = operator_present
            with lock:
                runway = schedule.remaining()
            if not operator_present:
                should_request = False
            elif cfg.pacing:
                should_request = runway <= (cfg.horizon - cfg.s_min) and obs_cooldown == 0
            else:
                should_request = True

            # 3. Emit an observation only when requesting. Attach the RTC
            #    in-painting request (d, overlap_end, single-source prefix
            #    descriptor). Record ts -> tick so the chunk can be aligned; arm
            #    cooldown.
            if should_request:
                # A transient camera stall makes read_latest() raise TimeoutError
                # (frame older than its max age). That must NOT kill the session:
                # skip this request and retry shortly. The schedule keeps executing
                # (or holds the last action on starvation) meanwhile.
                try:
                    obs = robot.get_observation()
                except TimeoutError as e:
                    stale_obs += 1
                    print(f"[robot] stale observation, skipping request: {e}")
                    obs = None

                if obs is None:
                    # Decrement cooldown like a non-request tick so we re-attempt
                    # promptly once the camera catches up.
                    obs_cooldown = max(obs_cooldown - 1, 0)
                else:
                    state = {f"s{idx}": float(obs[k]) for idx, k in enumerate(state_keys)}

                    # Build the RTC request from the current schedule + latency est.
                    rtc_meta = empty_rtc_meta()
                    with lock:
                        d = estimator.estimate_steps
                        overlap_end = max(1, cfg.horizon - max(cfg.s_min, d))
                        # current_step's action was already popped this tick, so the
                        # prefix is the still-to-execute run at control_step+1 onward.
                        prefix = schedule.prefix_span(current_step=control_step, max_len=overlap_end)
                    prefix_len = prefix.length
                    rtc_meta[RTC_D] = float(d)
                    rtc_meta[RTC_OVERLAP_END] = float(overlap_end)
                    if prefix_len > 0:
                        rtc_meta[RTC_SRC_TS] = float(prefix.src_ts)
                        rtc_meta[RTC_PREFIX_START] = float(prefix.start)
                        rtc_meta[RTC_PREFIX_LEN] = float(prefix_len)
                    state.update(rtc_meta)

                    portal.send_state(state, timestamp_us=ts_us)
                    for cam in schema.cameras:
                        frame = obs.get(cam)
                        if frame is not None:
                            portal.send_video_frame(cam, frame, timestamp_us=ts_us)
                    last_emit_ms = (time.monotonic() - tick_t0) * 1000.0
                    with lock:
                        sent_obs[ts_us] = control_step
                        while len(sent_obs) > obs_ttl:
                            sent_obs.popitem(last=False)
                        last_prefix_len = prefix_len
                    observations_emitted += 1
                    obs_cooldown = estimator.estimate_steps + cfg.epsilon
            else:
                obs_cooldown = max(obs_cooldown - 1, 0)

            now = time.monotonic()
            if now - last_log >= 1.0:
                m = portal.metrics()
                with lock:
                    runway = schedule.remaining()
                    merge_l2 = last_merge_l2
                    chunks = chunks_received
                    uncorr = uncorrelated_chunks
                    plen = last_prefix_len
                    ret_ms = last_ret_ms
                    age = None if last_chunk_at is None else (now - last_chunk_at) * 1000.0
                s = max(cfg.s_min, estimator.estimate_steps)
                age_str = "-" if age is None else f"{age:.0f}ms"
                elapsed = int(now - start)
                operator = portal.active_operator()
                # The human line stays: it is the artifact that made the first
                # live runs diagnosable, and it carries the RTC-only numbers
                # (prefix / merge_l2 / smooth / emit / ret / late / stale) that
                # the FROZEN STATS key set has no room for.
                print(
                    f"[robot] t={elapsed:>3}s chunks={chunks} reqs={observations_emitted} "
                    f"sched={runway} s={s} lat={estimator.estimate_steps}st/"
                    f"{estimator.estimate_seconds * 1000:.0f}ms "
                    f"prefix={plen} chunk_age={age_str} merge_l2={merge_l2:.3f} "
                    f"smooth={dmax_raw:.1f}{smooth_unit}({dmax_raw_joint})->"
                    f"{dmax_sent:.1f}{smooth_unit}({dmax_sent_joint}) "
                    f"holds={holds} emit={last_emit_ms:.0f}ms ret={ret_ms:.0f}ms "
                    f"late={late_ticks} "
                    f"active={operator} "
                    f"e2e={fmt_us(m.policy.e2e_us_p50)}/{fmt_us(m.policy.e2e_us_p95)} (p50/p95) "
                    f"rtt={fmt_us(m.rtt.rtt_us_last)}"
                    + (f" uncorr={uncorr}" if uncorr else "")
                    + (f" stale={stale_obs}" if stale_obs else "")
                )
                # ... and beside it, the machine-readable twin, on the SAME
                # frozen key set the adaptive-sync engine emits (see
                # drtc_protocol.STATS_KEYS and this module's docstring for how
                # `lead` and `degrade` are defined here). Every key always
                # present, null where unknown, so the parent's status response
                # model stays exact for both engines.
                emit_stats(
                    {
                        "t": elapsed,
                        "chunks": chunks,
                        "reqs": observations_emitted,
                        "sched": runway,
                        # `s` = max(s_min, d): the steps of an arriving chunk the
                        # round trip has already consumed. The RTC analogue of a
                        # prefetch lead, read against the same `horizon - s_min`
                        # margin the sync engine's is.
                        "lead": s,
                        "s_min": cfg.s_min,
                        "horizon": cfg.horizon,
                        "lat_steps": estimator.estimate_steps,
                        "lat_ms": round(estimator.estimate_seconds * 1000.0, 1),
                        "holds": holds,
                        # ANALYSIS §8.1's budget rule (roundtrip < (H - s_min)/fps),
                        # which is also `_sync_player.in_degrade`'s predicate: past
                        # it the fresh region collapses, every chunk lands entirely
                        # stale, and the arm freezes rather than in-paints.
                        "degrade": s >= (cfg.horizon - cfg.s_min),
                        "chunk_age_ms": None if age is None else round(age, 1),
                        "active": operator,
                        "e2e_p50_us": or_none(m.policy.e2e_us_p50),
                        "e2e_p95_us": or_none(m.policy.e2e_us_p95),
                        "rtt_us": or_none(m.rtt.rtt_us_last),
                        "uncorr": uncorr,
                    }
                )
                dmax_raw = dmax_sent = 0.0
                dmax_raw_joint = dmax_sent_joint = "-"
                last_log = now

            control_step += 1
            next_tick += interval
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                late_ticks += 1  # no await this tick -> chunk callbacks starve
    except KeyboardInterrupt:
        # Ctrl-C takes exactly the STOP path, not the old raw disconnect: a
        # hand-run bench session gets the same return-before-release a
        # supervised one does. Swallowed so the process still exits 0.
        print("[robot] interrupted (Ctrl-C)")
        control.stop_event.set()
    except Exception as exc:
        emit(EVENT_ERROR, str(exc))
        raise
    finally:
        # Order is load-bearing: the return needs torque, and it is
        # robot.disconnect() that releases it. QUIT is the one path that skips
        # the return — it means "stop now", and the caller has accepted that
        # the arm stays where it is.
        #
        # EVERY step here is `shielded`: a Ctrl-C is most likely precisely
        # while this is running (that is usually what got us here), and an
        # unshielded one would unwind the whole block and skip the release. The
        # second Ctrl-C during the return is the second STOP press — cut it
        # short and release where the arm is, nearer rest than it started —
        # which is what setting `abort_event` before the retry does.
        if start_poses and not control.quit_event.is_set():
            emit(EVENT_RETURNING)
            print("[robot] returning to the start pose ...")
            shielded(
                "the return to the start pose",
                return_step(start_poses, control.abort_event),
            )
        print("[robot] disconnecting...")
        # `portal` is None when the failure landed before it was built (a bad
        # codec name, a camera that would not open) — the arm is connected and
        # energized by then, so the return above and this release still have to
        # run. Shielded individually so a transport teardown error (or an
        # interrupt inside it) can never cost the arm its torque release.
        if portal is not None:
            await _shielded_disconnect(portal)
            shielded("closing the transport", portal.close, attempts=1)
        # `reraise`: a disconnect that genuinely FAILS still ends the process
        # non-zero and without a BYE — the parent reads that as the child
        # dying, which is the truth. Only the interrupt is swallowed.
        shielded("the torque release (robot.disconnect)", robot.disconnect, reraise=True)
        shielded("the BYE event", emit, EVENT_BYE, attempts=1)


@draccus.wrap()
def main(cfg: RobotSideRTCConfig) -> None:
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
