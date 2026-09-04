"""Policy side of the general remote-inference loop (runs on the GPU machine).

Loads ANY pretrained lerobot policy by path (ACT, diffusion, pi0, smolvla, ...)
and serves it over Portal: subscribes to the robot's synced observations, runs
one forward pass per observation, and ships the resulting action *chunk* back
as a single LiveKit byte stream — correlated to the observation timestamp so
the robot can measure true end-to-end policy latency.

This is the Portal-transport analogue of the installed lerobot package's
``lerobot/async_inference/policy_server.py``. The heavy lifting (checkpoint loading,
normalization, tokenization, chunk prediction) is reused from lerobot; Portal
only replaces the transport.

Normally you do not run this by hand — `modal_policy.py` is the supported way to
start it (`modal run makermodslab/drtc/modal_policy.py ...` from the repo root).
On a GPU box you already own, it is also a plain `python -m` entrypoint::

    python -m makermodslab.drtc.policy --policy-path=${HF_USER}/my_policy --device=cuda

The policy type is auto-detected from the checkpoint. The wire schema (state
dim, action dim, camera names, horizon) is derived from the loaded policy — see
`_schema.py`. The LiveKit connection comes from the environment: `LIVEKIT_URL`,
`LIVEKIT_ROOM` and the station-signed `LIVEKIT_TOKEN` (what the Lab's launcher
and the Deploy panel's line hand a container). Run by hand on a bench,
`_env.load_env` layers `livekit.env` under the environment and an API
key/secret there can mint the token instead; the Lab itself never uses that
path (see `docs/drtc/README.md`).

`--task` is REQUIRED for a language-conditioned policy (see `load_policy`), and
`--model-dtype` optionally overrides the checkpoint's saved precision.

Why the interesting decisions are not made in this file
-------------------------------------------------------
This module imports `livekit.portal` and `torch`, so CI cannot import it and
nothing here can carry a unit test. Every judgement that has a right and a wrong
answer is therefore delegated to a PURE helper in
`makermodslab.utils.system` — which types need a task (`policy_requires_task`)
and what to do about a missing MolmoAct2 action mode
(`molmoact2_inference_action_mode`) — where `tests/test_utils_system.py` covers
it. What is left here is wiring: read a flag, call the helper, print, and either
refuse or proceed. Keep it that way; a decision that grows a branch belongs in
the helper, not below.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812  (the universal torch alias)
from livekit.portal import Observation, Operator, OperatorConfig, VideoCodec, frame_bytes_to_numpy_rgb

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from ..utils.system import molmoact2_inference_action_mode, policy_requires_task
from ._common import env_str, fmt_us, load_env, mint_token, required_env
from ._schema import CHUNK_NAME, IMAGE_PREFIX, policy_wire_schema

IDENTITY = "policy"


def pick_device(requested: str) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_policy(policy_path: str, device: torch.device, task: str = "", model_dtype: str = ""):
    """Load any lerobot policy + its pre/post processors from a checkpoint.

    Three things happen BEFORE the weights load, and the ordering is the point —
    the first checkpoint this matters for pulls a 21.8 GB base VLM, so a
    refusal or an override that lands after the download is worth nothing.

    * **A language-conditioned policy with no ``--task`` is refused outright.**
      MolmoAct2 is why: with an empty task it does not fail, it DEGRADES
      SILENTLY — its processor renders the missing string into a fixed template
      and prompts the VLM with the literal "The task is to .". Which types
      those are is ``utils.system.policy_requires_task``, the SAME vocabulary
      the Lab's ``requires_task`` reads, so the launch panel and the GPU can
      never disagree about whether a run needs a task.
    * **``--model-dtype``, when passed, replaces the config's saved dtype.** It
      has to happen here: the dtype is read while the base model is being
      constructed, so assigning it afterwards would change nothing. Never a
      default and never silent (see the flag's help).
    * **A missing ``inference_action_mode`` is filled in, and only a missing
      one.** ``utils.system.molmoact2_inference_action_mode`` decides, again so
      the Lab and the GPU answer identically.

    ``config=`` is handed to ``from_pretrained`` only when something actually
    changed, so a run that passes neither flag loads exactly as it did before.
    """
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_type = getattr(policy_cfg, "type", "")

    if policy_requires_task(policy_type) and not task:
        raise SystemExit(
            f"--task is required for a '{policy_type}' policy. It is language-conditioned, "
            "and an empty task does not fail — it degrades silently (MolmoAct2 prompts its "
            'VLM with the literal "The task is to ."). Pass --task "<what the robot should '
            'do>", phrased as an infinitive completion.'
        )

    overridden = False

    if model_dtype:
        if not hasattr(policy_cfg, "model_dtype"):
            raise SystemExit(
                f"--model-dtype={model_dtype!r} was passed, but {type(policy_cfg).__name__} "
                f"('{policy_type}') has no model_dtype field — this policy's precision is not "
                "configurable that way. Drop the flag rather than have it be silently ignored."
            )
        if not isinstance(getattr(torch, model_dtype, None), torch.dtype):
            raise SystemExit(
                f"--model-dtype={model_dtype!r} does not name a torch dtype. "
                "Use float32, bfloat16 or float16."
            )
        print(
            f"[policy] --model-dtype: OVERRIDING the checkpoint's saved "
            f"model_dtype={policy_cfg.model_dtype!r} with {model_dtype!r} before weights load "
            "(operator opt-in; without the flag the checkpoint's own value is used as saved)"
        )
        policy_cfg.model_dtype = model_dtype
        overridden = True

    # A GAP FILL, never an override. `MolmoAct2Config.inference_action_mode`
    # defaults to None and the policy raises on None, so a checkpoint that saved
    # no mode cannot be served at all; the published one saves "continuous" and
    # a discrete one saves "discrete", and forcing either would break the other.
    # No other config in the pin has this field, so the guard is self-scoping.
    if getattr(policy_cfg, "inference_action_mode", "unset") is None:
        mode = molmoact2_inference_action_mode(
            {"type": policy_type, "action_mode": getattr(policy_cfg, "action_mode", None)}
        )
        print(
            f"[policy] checkpoint saved no inference_action_mode; using {mode!r}, derived "
            "from its action_mode (lerobot raises on None, so there is no do-nothing option)"
        )
        policy_cfg.inference_action_mode = mode
        overridden = True

    policy_cls = get_policy_class(policy_cfg.type)
    policy = (
        policy_cls.from_pretrained(policy_path, config=policy_cfg)
        if overridden
        else policy_cls.from_pretrained(policy_path)
    )
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor


class Inference:
    """Turns a Portal ``Observation`` into an unnormalized ``(horizon, A)`` chunk.

    Mirrors lerobot's ``policy_server._predict_action_chunk``: build batch -> preprocess
    (normalize/tokenize/device) -> ``predict_action_chunk`` -> postprocess
    (unnormalize) per timestep.
    """

    def __init__(self, policy, preprocessor, postprocessor, image_features, state_dim, horizon, device, task):
        self.policy = policy
        self.pre = preprocessor
        self.post = postprocessor
        self.image_features = image_features
        self.state_dim = state_dim
        self.horizon = horizon
        self.device = device
        self.task = task
        # Most-recent per-call latency split (ms), read by the 1 Hz log to
        # localize e2e. build = CPU image decode+resize+tensor construction;
        # infer = preprocess + forward pass + postprocess + device->host copy.
        self.last_build_ms = 0.0
        self.last_infer_ms = 0.0

    def _build_batch(self, obs: Observation) -> dict:
        state_vec = np.array([obs.state[f"s{i}"] for i in range(self.state_dim)], dtype=np.float32)[
            None, :
        ]  # (1, D)
        batch = {"observation.state": torch.from_numpy(state_vec)}

        for full_key, feature in self.image_features.items():
            cam = full_key[len(IMAGE_PREFIX) :] if full_key.startswith(IMAGE_PREFIX) else full_key
            frame = obs.frames.get(cam)
            if frame is None:
                raise KeyError(f"policy expects camera '{cam}' but the robot sent none")
            rgb = frame_bytes_to_numpy_rgb(frame.data, frame.width, frame.height)  # HWC uint8
            # .copy(): the returned array is a read-only view over the frame's
            # immutable bytes; copy to a writable buffer torch can own.
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

    def warmup(self, rounds: int = 2) -> None:
        """Run a few forward passes on synthetic zero inputs before the control
        loop connects.

        The first real inference otherwise pays CUDA init, cuDNN autotuning, and
        first-pass JIT — often several seconds. On the robot that shows up as a
        single huge e2e sample that poisons the DRTC latency estimator (and the
        arm sits idle until the first chunk lands). Absorbing it here, while
        disconnected, makes the first *real* observation fast. Best-effort: a
        warmup failure must not stop us from serving.
        """
        batch = {"observation.state": torch.zeros((1, self.state_dim), dtype=torch.float32)}
        for full_key, feature in self.image_features.items():
            c, h, w = feature.shape
            batch[full_key] = torch.zeros((1, c, h, w), dtype=torch.float32)
        if self.task:
            batch["task"] = self.task
        with torch.inference_mode():
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
        t0 = time.monotonic()
        batch = self._build_batch(obs)
        t1 = time.monotonic()
        with torch.inference_mode():
            batch = self.pre(batch)
            chunk = self.policy.predict_action_chunk(batch)  # (B, chunk, A) normalized
            if chunk.ndim == 2:
                chunk = chunk.unsqueeze(0)
            chunk = chunk[:, : self.horizon, :]
            # Postprocessor expects (B, A) per action; unnormalize each timestep.
            steps = [self.post(chunk[:, t, :]) for t in range(chunk.shape[1])]
            chunk = torch.stack(steps, dim=1).squeeze(0)  # (horizon, A)
        # .cpu() forces a device sync, so t2 captures true GPU compute (no async
        # kernel-launch undercount) — the split is only meaningful because of it.
        out = chunk.detach().cpu().numpy().astype(np.float32)
        t2 = time.monotonic()
        self.last_build_ms = (t1 - t0) * 1e3
        self.last_infer_ms = (t2 - t1) * 1e3
        return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", default="", help="HF repo id or local dir of the pretrained policy")
    parser.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    parser.add_argument("--task", default="", help="Language instruction for VLA policies")
    parser.add_argument(
        "--model-dtype",
        default="",
        help="Override the checkpoint's saved model_dtype before weights load "
        "(float32 | bfloat16 | float16). OPT-IN: unset means the checkpoint's "
        "own value is used as saved. MolmoAct2's published checkpoint saves "
        "float32, which on a 40 GB A100 is ~28 GB of weights AND gets no "
        "autocast — measure first, then reach for bfloat16 if it OOMs or "
        "cannot keep up.",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Control/stream frequency. MUST match robot --fps."
    )
    parser.add_argument(
        "--horizon", type=int, default=16, help="Action-chunk horizon. MUST match robot --horizon."
    )
    parser.add_argument(
        "--video-codec",
        default="H264",
        help="H264 (default) | MJPEG | PNG | RAW. MUST match robot --video-codec.",
    )
    parser.add_argument(
        "--duration", type=float, default=0.0, help="Run seconds; 0 = run until the robot leaves / Ctrl-C."
    )
    args = parser.parse_args()
    if not args.policy_path:
        raise SystemExit("--policy-path is required")

    load_env()
    url = required_env("LIVEKIT_URL")
    room = required_env("LIVEKIT_ROOM")
    # The station-signed operator token (what the Lab's launcher and the Deploy
    # panel's line hand over) wins; minting from an API key/secret in the
    # environment is the hand-run bench fallback only.
    token = env_str("LIVEKIT_TOKEN", "") or mint_token(IDENTITY, room)
    fps = args.fps
    horizon = args.horizon
    duration = args.duration

    device = pick_device(args.device)
    print(f"[policy] loading '{args.policy_path}' on {device} ...")
    policy, pre, post = load_policy(args.policy_path, device, task=args.task, model_dtype=args.model_dtype)
    schema, image_features = policy_wire_schema(policy)
    print(
        f"[policy] policy schema: state={schema.state_dim} "
        f"action={schema.action_dim} cameras={schema.cameras} horizon={horizon}"
    )

    codec = getattr(VideoCodec, args.video_codec.upper())
    cfg = OperatorConfig(room)
    for cam in schema.cameras:
        cfg.add_video(cam, codec=codec)
    cfg.add_state_typed(schema.state_fields())
    cfg.add_action_chunk(CHUNK_NAME, horizon=horizon, fields=schema.action_fields())
    cfg.set_fps(fps)

    op = Operator(cfg)
    engine = Inference(policy, pre, post, image_features, schema.state_dim, horizon, device, args.task)

    # Warm the model before connecting so the first real inference isn't a
    # multi-second cold-start spike (which also poisons the robot's latency
    # estimator). Best-effort — never let warmup failure block serving.
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

    # Keep strong refs to fire-and-forget tasks so the event loop can't GC them
    # mid-flight (asyncio only holds a weak ref to a bare create_task result).
    bg_tasks: set = set()

    def spawn(coro) -> None:
        t = asyncio.create_task(coro)
        bg_tasks.add(t)
        t.add_done_callback(bg_tasks.discard)

    async def claim_control(reason: str) -> None:
        """Claim the robot's control gate, retrying until a robot is present.

        On the operator side `set_active_operator` is NOT local — it dispatches
        an RPC to the robot (portal.rs `resolve_robot_identity` -> perform_rpc).
        If no robot is in the room yet, that raises (e.g. NoPeer) after a short
        internal poll (~1.5s). That is routine, not fatal: the policy is often
        started before the robot, and against LiveKit Cloud the robot's role
        attribute can take a moment to propagate. So we retry instead of
        crashing — the serve loop runs meanwhile and the robot accepts our
        chunks the instant the claim lands.
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

    # Re-claim control whenever the robot's control gate moves off us. The robot
    # is the source of truth for whose chunks it accepts; when it restarts and
    # rejoins, that gate resets to None and every chunk we send is dropped. This
    # callback fires on the event loop (the Portal dispatcher hops onto it), so
    # spawning is safe. Claiming to our own identity fires this with active == us,
    # which no-ops — no loop.
    def on_active_operator_changed(active: str | None) -> None:
        if active != op.local_identity():
            print(f"[policy] control gate is now {active!r}; re-claiming")
            spawn(claim_control("gate moved"))

    op.on_active_operator_changed(on_active_operator_changed)

    print(f"[policy] connecting to {url} as '{IDENTITY}' in room '{room}' ...")
    await op.connect(url, token)
    # Claim control in the BACKGROUND — never block startup or crash if the robot
    # hasn't joined yet (the common case on LiveKit Cloud). In an HITL setup a
    # human could preempt with `await op.set_active_operator("human-id")`.
    spawn(claim_control("startup"))
    print(f"[policy] connected as '{op.local_identity()}'; claiming control in background ...")

    chunks_sent = 0
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
            # Inference is blocking (GPU/CPU-bound); keep it off the event loop.
            chunk = await asyncio.to_thread(engine.run, obs)
            op.send_action_chunk(CHUNK_NAME, chunk, in_reply_to_ts_us=obs.timestamp_us)
            chunks_sent += 1

            now = time.monotonic()
            if now - last_log >= 1.0:
                m = op.metrics()
                # Localize e2e's non-network cost, all measured HERE on the
                # policy (video-receive) side:
                #  infer/build  — GPU forward pass vs CPU image prep (engine.run).
                #  match_delta  — skew between the matched video frame and state
                #                 sample in each assembled observation; large =>
                #                 the sync buffer is waiting on the laggy stream.
                #  blocker      — which track the sync buffer waited on last
                #                 (the slow stream; usually a camera under H264).
                #  vbuf         — frames sitting in each camera's receive/jitter
                #                 buffer; high fill = latency parked in the buffer.
                #  fjitter      — per-camera frame arrival jitter.
                vbuf = ",".join(f"{k}:{v}" for k, v in m.buffers.video_fill.items()) or "-"
                fjit = ",".join(f"{k}:{fmt_us(v)}" for k, v in m.transport.frame_jitter_us.items()) or "-"
                print(
                    f"[policy] t={int(now - start):>3}s chunks_sent={chunks_sent} "
                    f"obs_seen={m.sync.observations_emitted} stale={m.sync.stale_observations_emitted} "
                    f"infer={engine.last_infer_ms:.0f}ms build={engine.last_build_ms:.0f}ms "
                    f"match_delta={fmt_us(m.sync.match_delta_us_p50)}/{fmt_us(m.sync.match_delta_us_p95)} (p50/p95) "
                    f"blocker={m.sync.last_blocker_track or '-'} "
                    f"vbuf=[{vbuf}] fjitter=[{fjit}]"
                )
                last_log = now
    finally:
        print(f"[policy] sent {chunks_sent} chunks; disconnecting...")
        await op.disconnect()
        op.close()


if __name__ == "__main__":
    asyncio.run(main())
