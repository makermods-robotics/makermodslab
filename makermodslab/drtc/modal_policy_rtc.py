"""Run the FULL-DRTC GPU policy server (`policy_rtc.py`) on a Modal serverless GPU.

Thin wrapper, identical in shape to ``modal_policy.py`` — the ONLY differences are
the entrypoint module (``policy_rtc`` instead of ``policy``) and the extra
RTC-in-painting env/CLI wiring (``--s-min``, ``--max-guidance-weight``,
``--rtc-schedule``). All DRTC scheduling lives on the robot; the server just
loads a flow policy, enables RTC guided in-painting, and serves over Portal.

``policy_rtc.py`` is a pure *outbound* WebRTC client (it dials into the LiveKit
SFU), so nothing needs to listen on a port and Modal's outbound-only networking
is a clean fit. The SFU is LiveKit Cloud (a separate, publicly reachable
service).

Topology:

    robot_rtc.py (on-prem) ─► LiveKit Cloud SFU ◄─ modal_policy_rtc.py (this, Modal GPU)

One-time setup (same as modal_policy.py)
----------------------------------------
1. LiveKit Cloud project -> `wss://` URL, API key, API secret.
2. Stash them + the room as a Modal secret (these become container env vars that
   `makermodslab.drtc._env.load_env()` / `_common.mint_token()` read directly):

       modal secret create LiveKit-cloud \
           LIVEKIT_URL=wss://<your-project>.livekit.cloud \
           LIVEKIT_API_KEY=<key> \
           LIVEKIT_API_SECRET=<secret> \
           LIVEKIT_ROOM=portal-lerobot-inference

3. Point the robot side at the SAME URL / key / secret / room — write them to
   `~/.cache/huggingface/lerobot/livekit.env` (see docs/drtc/livekit.env.example).
   `LIVEKIT_ROOM` is set ONLY by this secret on the GPU side, so it must match.

Run
---
    modal run makermodslab/drtc/modal_policy_rtc.py --policy-path ${HF_USER}/my_pi0
    modal run makermodslab/drtc/modal_policy_rtc.py --policy-path ${HF_USER}/my_pi0 \
        --task "Put the lego brick in the box" --rtc-schedule linear

    # local SFU over the tailnet (signaling only; media still direct UDP):
    modal run makermodslab/drtc/modal_policy_rtc.py --policy-path ${HF_USER}/my_pi0 \
        --tailscale --livekit-url ws://100.x.y.z:7880 \
        --livekit-api-key <key> --livekit-api-secret <secret>

`--horizon` MUST match robot_rtc.py's `--horizon`, and `--s-min` should match
robot_rtc.py's `--s_min`. `--slack` / `--tolerance` tune the operator-side Portal
sync buffer (see policy_rtc.py's help). RTC in-painting only engages for flow policies
(smolvla/pi0/pi05); ACT etc. serve plain chunks. Deploy (`modal deploy`) +
`.spawn()` to keep it running as a service.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import modal

# The makermodslab package directory, shipped into the container by PATH (see the
# add_local_dir note in the image). /root is on sys.path in a Modal container,
# so it lands as an importable `makermodslab` there.
_PACKAGE_DIR = Path(__file__).resolve().parents[1]

# --- Tailscale hybrid transport ----------------------------------------------
# `--tailscale` moves LiveKit *signaling* off the Cloudflare quick tunnel and
# onto the user's tailnet (no random URL per launch, no public unauthenticated
# endpoint). Media/data are UNCHANGED: still direct UDP 7882 between this
# container and the Mac via `use_external_ip: true` + ICE hole punch.
#
# Two constraints shape the implementation:
#   1. Modal containers cannot create a TUN device, so tailscaled must run in
#      userspace-networking mode, which exposes the tailnet only through a
#      SOCKS5 proxy.
#   2. The LiveKit Rust SDK's signaling WebSocket (tokio-tungstenite inside
#      livekit-portal's FFI dylib) speaks HTTP CONNECT proxies via
#      HTTP_PROXY/HTTPS_PROXY, but has NO SOCKS support at all (verified: zero
#      "socks" strings in liblivekit_portal_ffi.dylib).
# So we bridge them with the ~60 lines below: a loopback TCP listener that
# issues a SOCKS5 CONNECT to tailscaled and pipes bytes to
# <mac-tailnet-ip>:7880. The
# policy then just dials ws://127.0.0.1:7880 — zero SDK proxy awareness, and no
# process-wide *_PROXY env that would also hijack Hugging Face downloads.
_TS_SOCKS_PORT = 1055  # tailscaled's userspace SOCKS5 listener
_TS_RELAY_PORT = 7880  # loopback port the policy dials instead of the tailnet
_TS_SOCKET = "/tmp/tailscaled.sock"  # nosec B108 — tailscaled's own socket, inside a single-tenant Modal container
_TS_HOSTNAME = "modal-policy"  # node name in the tailnet admin console


def _looks_like_tailnet(host: str) -> bool:
    """True for a 100.64.0.0/10 CGNAT address, a MagicDNS name, or a bare name."""
    try:
        return ipaddress.ip_address(host) in ipaddress.ip_network("100.64.0.0/10")
    except ValueError:
        return host.endswith(".ts.net") or "." not in host


def _tailscale_up(timeout: float = 90.0) -> None:
    """Start userspace tailscaled and join the tailnet. Blocks until Running.

    Ephemeral-ness lives on the AUTH KEY, not the CLI: `tailscale up` has no
    `--ephemeral` flag (checked against current Tailscale docs), so the key made
    in the admin console must have "Ephemeral" ticked for dead containers to
    disappear from the tailnet by themselves. `--state=mem:` keeps that honest by
    persisting no node state in the container at all.
    """
    authkey = os.environ.get("TS_AUTHKEY")
    if not authkey:
        raise SystemExit(
            "[tailscale] --tailscale needs TS_AUTHKEY in the container env, and it "
            "isn't there. Create a REUSABLE + EPHEMERAL auth key in the Tailscale "
            "admin console (Settings -> Keys), then:\n"
            "    modal secret create tailscale-auth TS_AUTHKEY=tskey-auth-...\n"
            "and re-run with --tailscale (that flag routes to serve_ts, the twin "
            "function this secret is attached to)."
        )

    print(f"[tailscale] starting tailscaled (userspace networking, socks5 on 127.0.0.1:{_TS_SOCKS_PORT})")
    proc = subprocess.Popen(
        [
            "tailscaled",
            "--tun=userspace-networking",
            f"--socks5-server=localhost:{_TS_SOCKS_PORT}",
            "--state=mem:",
            f"--socket={_TS_SOCKET}",
        ]
    )

    # 1) tailscaled listening on its SOCKS port
    deadline = time.monotonic() + timeout
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"tailscaled exited early (rc={proc.returncode})")
        try:
            socket.create_connection(("127.0.0.1", _TS_SOCKS_PORT), timeout=0.5).close()
            break
        except OSError:
            if time.monotonic() > deadline:
                raise RuntimeError("tailscaled never opened its SOCKS5 port") from None
            time.sleep(0.25)

    # 2) join the tailnet. `tailscale up` blocks until the backend is Running, so
    #    this returning IS the "tailnet reachable" barrier the relay needs.
    print(f"[tailscale] tailscale up --hostname={_TS_HOSTNAME} --auth-key=<redacted>")
    up = subprocess.run(
        [
            "tailscale",
            f"--socket={_TS_SOCKET}",
            "up",
            f"--auth-key={authkey}",
            f"--hostname={_TS_HOSTNAME}",
            # Don't rewrite the container's /etc/resolv.conf: MagicDNS here would
            # only confuse Hugging Face / PyPI lookups. Tailnet names are resolved
            # by tailscaled itself, via the SOCKS5 domain address type below.
            "--accept-dns=false",
            f"--timeout={int(timeout)}s",
        ],
        capture_output=True,
        text=True,
    )
    if up.returncode != 0:
        raise RuntimeError(
            f"`tailscale up` failed (rc={up.returncode}): {up.stdout.strip()}{up.stderr.strip()}"
        )
    ip = subprocess.run(
        ["tailscale", f"--socket={_TS_SOCKET}", "ip", "-4"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"[tailscale] joined tailnet as {_TS_HOSTNAME} ({ip or 'no v4 address?'})")


async def _socks5_connect(host: str, port: int):
    """Open a TCP stream to host:port *through* tailscaled's SOCKS5 proxy.

    RFC 1928, no-auth, CONNECT only — the whole subset we need, hand-rolled to
    avoid pulling `python-socks` into the GPU image for ~40 lines of protocol.
    """
    reader, writer = await asyncio.open_connection("127.0.0.1", _TS_SOCKS_PORT)
    writer.write(b"\x05\x01\x00")  # VER=5, 1 method, NO AUTHENTICATION
    await writer.drain()
    greet = await reader.readexactly(2)
    if greet != b"\x05\x00":
        raise OSError(f"SOCKS5 greeting rejected: {greet!r}")

    try:  # ATYP=1 (IPv4) when the target is literal, else ATYP=3 (domain) so
        req = b"\x05\x01\x00\x01" + socket.inet_aton(host)  # tailscaled resolves it
    except OSError:
        name = host.encode("idna")
        if len(name) > 255:
            raise OSError(f"hostname too long for SOCKS5: {host!r}") from None
        req = b"\x05\x01\x00\x03" + bytes([len(name)]) + name
    writer.write(req + port.to_bytes(2, "big"))
    await writer.drain()

    rep = await reader.readexactly(4)
    if rep[0] != 5 or rep[1] != 0:
        raise OSError(
            f"SOCKS5 CONNECT to {host}:{port} failed (REP={rep[1]}) — is the Mac up on "
            "the tailnet and is livekit-server listening on that address?"
        )
    atyp = rep[3]  # drain BND.ADDR + BND.PORT
    if atyp == 1:
        await reader.readexactly(4)
    elif atyp == 3:
        await reader.readexactly((await reader.readexactly(1))[0])
    elif atyp == 4:
        await reader.readexactly(16)
    await reader.readexactly(2)
    return reader, writer


async def _pipe(reader, writer) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (OSError, asyncio.IncompleteReadError):
        pass
    finally:
        with contextlib.suppress(OSError):
            writer.close()


def _start_signaling_relay(host: str, port: int, timeout: float = 15.0) -> None:
    """Listen on 127.0.0.1:_TS_RELAY_PORT, forward to host:port over the tailnet.

    Runs on its OWN event loop in a daemon thread, deliberately: the policy's
    loop blocks for hundreds of ms per inference, and the signaling WebSocket
    (LiveKit pings on a timer) must not stall behind it.
    """
    ready = threading.Event()
    failure: list[BaseException] = []

    async def _handle(client_reader, client_writer) -> None:
        try:
            up_reader, up_writer = await _socks5_connect(host, port)
        except Exception as exc:  # one bad dial must not kill the listener
            print(f"[tailscale-relay] upstream dial failed: {exc}")
            client_writer.close()
            return
        await asyncio.gather(_pipe(client_reader, up_writer), _pipe(up_reader, client_writer))

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _serve() -> None:
            server = await asyncio.start_server(_handle, "127.0.0.1", _TS_RELAY_PORT)
            ready.set()
            async with server:
                await server.serve_forever()

        try:
            loop.run_until_complete(_serve())
        except BaseException as exc:  # noqa: BLE001 - surfaced on the main thread
            failure.append(exc)
            ready.set()

    threading.Thread(target=_run, name="tailscale-relay", daemon=True).start()
    if not ready.wait(timeout):
        raise RuntimeError("tailscale signaling relay did not start in time")
    if failure:
        raise RuntimeError(f"tailscale signaling relay failed to listen: {failure[0]}")
    print(f"[tailscale] relay 127.0.0.1:{_TS_RELAY_PORT} -> socks5 -> {host}:{port}")


def _tailscale_signaling_url(livekit_url: str) -> str:
    """Bring up the tailnet + relay; return the ws:// URL the policy should dial.

    Ordering is the whole point: tailscaled up -> `tailscale up` returns (node
    Running) -> relay listening -> only then does the caller hand the URL to
    policy_rtc and connect.
    """
    parsed = urllib.parse.urlsplit(livekit_url)
    if parsed.scheme not in ("ws", "http"):
        raise SystemExit(
            f"[tailscale] --tailscale expects a plaintext ws:// URL, got {livekit_url!r}. "
            "The relay rewrites the host to 127.0.0.1, so a wss:// certificate would fail "
            "hostname verification — and WireGuard already encrypts the tailnet hop, so "
            "plain ws:// is the correct choice here."
        )
    host, port = parsed.hostname, parsed.port or 7880
    if not host:
        raise SystemExit(f"[tailscale] no host in --livekit-url {livekit_url!r}")
    if not _looks_like_tailnet(host):
        print(
            f"[tailscale] WARNING: {host} doesn't look like a tailnet address "
            "(100.64.0.0/10 or a MagicDNS name). Continuing — it still has to be "
            "reachable *from inside the tailnet* for the relay to connect."
        )
    _tailscale_up()
    _start_signaling_relay(host, port)
    return f"ws://127.0.0.1:{_TS_RELAY_PORT}"


# --- GPU image ---------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    # curl + ca-certificates are for the Tailscale apt repo below (--tailscale).
    .apt_install(
        "git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential", "cmake", "curl", "ca-certificates"
    )
    .pip_install("uv")
    # lerobot pinned to the exact UPSTREAM commit pyproject.toml uses
    # (github.com/huggingface/lerobot @ 8414188d, main). Do NOT use bare `lerobot`:
    # PyPI resolves to 0.6.0, a release line whose PI05Config LACKS the
    # relative-action config fields (use_relative_actions, relative_exclude_joints,
    # action_feature_names, pretrained_revision) that main added in PR #2970 and
    # that your model's config.json was saved with — so 0.6.0 dies in draccus with
    # "fields ... are not valid for PI05Config". Config compatibility follows the
    # lerobot that WROTE the checkpoint, not the highest version number. Building
    # from git needs the `git` apt package (installed above). Bump this SHA
    # whenever you retrain on a newer lerobot.
    #
    # livekit-portal comes from PyPI, pinned to the same version as
    # pyproject.toml so robot and GPU sides speak the identical wire code.
    .run_commands(
        "uv pip install --system --compile-bytecode "
        '"livekit-api>=0.7" "python-dotenv>=1" "numpy>=1.24" '
        '"livekit-portal==0.2.4" '
        # [pi,smolvla] pulls the flow-policy runtime deps (transformers, scipy,
        # accelerate, num2words). pi0/pi05/smolvla all import transformers; without
        # the extra, from_pretrained fails with "transformers is required".
        '"lerobot[pi,smolvla] @ git+https://github.com/huggingface/lerobot.git'
        '@8414188db0b178b947985a7a9a91314708837315"'
    )
    # Tailscale, for the `--tailscale` hybrid transport (signaling over the
    # tailnet; media still direct UDP). Installed from Tailscale's own apt repo,
    # keyed to whatever Debian codename the base image is, because Tailscale
    # publishes no "latest" alias for its static tarballs and a hard-coded
    # version string rots. The version is therefore whatever `stable` was when
    # this layer was BUILT — Modal content-addresses the layer, so it's frozen
    # per image build, not per run. To pin explicitly, change the last apt-get to
    # `apt-get install -y tailscale=<version>`. Costs ~30 MB in the image and
    # nothing at runtime unless --tailscale is passed.
    .run_commands(
        'CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME:-bookworm}")" && '
        "curl -fsSL https://pkgs.tailscale.com/stable/debian/$CODENAME.noarmor.gpg "
        "-o /usr/share/keyrings/tailscale-archive-keyring.gpg && "
        "curl -fsSL https://pkgs.tailscale.com/stable/debian/$CODENAME.tailscale-keyring.list "
        "-o /etc/apt/sources.list.d/tailscale.list && "
        "apt-get update && apt-get install -y --no-install-recommends tailscale && "
        "rm -rf /var/lib/apt/lists/*"
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "0",
        }
    )
    # The whole makermodslab package (policy server + drtc core + utils.config),
    # added by PATH rather than by import: the local `modal` CLI is a uv tool
    # whose interpreter cannot import makermodslab, and add_local_python_source
    # resolves modules through the LOCAL interpreter.
    #
    # copy=False (the default) re-uploads the directory at every container
    # start, so an edited policy_rtc.py always reaches the container — the
    # staleness this used to work around came from an add_local_python_source
    # automount, not from copy=False per se. Being a runtime mount, this MUST
    # stay the LAST build step (Modal forbids build steps after an automount).
    .add_local_dir(
        _PACKAGE_DIR,
        remote_path="/root/makermodslab",
        ignore=["**/__pycache__", "**/*.pyc"],
    )
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

app = modal.App("lerobot-drtc-full-policy")


# Everything except `secrets=`, shared by the two Modal functions below so the
# GPU/region/timeout knobs stay in ONE place.
_FN_KWARGS = {
    "image": image,
    "gpu": "A100",  # pinned per SWEEP.md: inference is the largest e2e term, cost is not the constraint
    "timeout": 60 * 60 * 2,
    "volumes": {"/cache": hf_cache},
    "region": "us-west",
    # "min_containers": 1,  # pre-warm to avoid a cold start when the robot connects
}

# Secrets, split out for the same reason: `serve` and `serve_ts` differ ONLY in
# the tailscale entry.
_BASE_SECRETS = [
    modal.Secret.from_name("LiveKit-cloud"),
    modal.Secret.from_name("huggingface"),
]


def _serve_impl(  # nosec B107 — the empty `*_secret` defaults are "flag not passed", not credentials
    policy_path: str,
    task: str = "",
    horizon: int = 50,
    fps: int = 30,
    duration: float = 0.0,
    s_min: int = 4,
    slack: int = 5,
    tolerance: float = 1.5,
    max_guidance_weight: float = 10.0,
    rtc_schedule: str = "linear",
    video_codec: str = "H264",
    livekit_url: str = "",
    livekit_api_key: str = "",
    livekit_api_secret: str = "",
    tailscale: bool = False,
) -> None:
    # Shipped via add_local_dir; imported HERE (not at module top) because this
    # wrapper is also evaluated locally by the modal CLI, whose interpreter has
    # no makermodslab on its path.
    from makermodslab.drtc import policy_rtc

    # Optional per-run LiveKit override: point this run at a different SFU than
    # the LiveKit-cloud secret — e.g. the LOCAL SFU exposed through a Cloudflare
    # quick tunnel (`tools/drtc/local_sfu.sh` prints the exact flags; see
    # docs/drtc/README.md "Local SFU"). Quick-tunnel URLs are ephemeral, so this
    # rides per-run CLI args rather than a Modal secret. Unset flags fall through
    # to the secret.
    if livekit_url:
        os.environ["LIVEKIT_URL"] = livekit_url
    if livekit_api_key:
        os.environ["LIVEKIT_API_KEY"] = livekit_api_key
    if livekit_api_secret:
        os.environ["LIVEKIT_API_SECRET"] = livekit_api_secret

    # Tailscale hybrid: join the tailnet, stand up the loopback->SOCKS5 relay,
    # and point LIVEKIT_URL at the relay instead of the tailnet address. Media
    # and data channels are untouched — they still hole-punch straight to the
    # Mac's public IP on UDP 7882. `tools/drtc/local_sfu_ts.sh` prints the matching flags.
    if tailscale:
        target = livekit_url or os.environ.get("LIVEKIT_URL", "")
        if not target:
            raise SystemExit("[tailscale] --tailscale needs --livekit-url ws://<tailnet-ip>:7880")
        os.environ["LIVEKIT_URL"] = _tailscale_signaling_url(target)
        print(
            f"[tailscale] signaling {target} -> {os.environ['LIVEKIT_URL']} "
            "(media/data still direct UDP 7882)"
        )

    # HF auth for GATED base backbones. pi0/pi05/smolvla pull a gated base model
    # (pi0.5 -> google/paligemma-3b-pt-224); huggingface_hub/transformers read the
    # token from HF_TOKEN. Modal already exposes every key of the attached
    # "huggingface" secret as an env var, so HF_TOKEN is present IFF that secret
    # has an HF_TOKEN key. Mirror it to the alias older transformers read, and fail
    # loudly if it's absent — otherwise the only symptom is a downstream 403 on the
    # gated repo, which is easy to misread as "no access" rather than "no token".
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("[policy] HF_TOKEN present; gated base models will authenticate.")
    else:
        print(
            "[policy] WARNING: no HF_TOKEN in env. Gated base backbones "
            "(e.g. google/paligemma-3b-pt-224 for pi0.5) will 403. Ensure the "
            "Modal 'huggingface' secret has a key named exactly HF_TOKEN "
            "(`modal secret create huggingface HF_TOKEN=hf_...`)."
        )

    # policy_rtc.py is CLI-only for behavior config (no env fallback), so forward
    # every setting as a command-line flag. Only the LiveKit creds still ride the
    # environment (via the Modal secret), which is exactly what policy_rtc reads
    # from there. --fps / --horizon / --video-codec MUST match robot_rtc.py.
    #
    # video_codec selects the wire transport: H264/VP8/VP9/AV1 ride the WebRTC
    # media path; MJPEG/PNG/RAW ride the byte-stream. A mismatch with the robot
    # means the policy subscribes on the wrong path and sees no frames. Use H264 on
    # thin/long-haul links — real-time RTP + temporal compression won't back up the
    # publish queue the way per-frame MJPEG does.
    argv = [
        "policy_rtc.py",
        "--policy-path",
        policy_path,
        "--device",
        "cuda",
        "--fps",
        str(fps),
        "--horizon",
        str(horizon),
        "--video-codec",
        video_codec,
        "--duration",
        str(duration),
        "--s-min",
        str(s_min),
        # Portal sync-buffer knobs. The observation matcher runs on THIS
        # (operator) side, so these are tuned here, not on the robot: --slack is
        # buffer headroom in ticks (higher absorbs jitter, adds slack/fps to
        # e2e), --tolerance is the state<->frame pairing window in ticks.
        "--slack",
        str(slack),
        "--tolerance",
        str(tolerance),
        "--max-guidance-weight",
        str(max_guidance_weight),
        "--rtc-schedule",
        rtc_schedule,
    ]
    if task:
        argv += ["--task", task]
    sys.argv = argv
    asyncio.run(policy_rtc.main())


# TWO Modal functions, identical except that `serve_ts` also gets the
# `tailscale-auth` secret; `main()` picks one by the --tailscale flag.
#
# Why not one function whose secret list is built conditionally: Modal evaluates
# this module BOTH locally (to build the app spec) and inside the container (on
# module re-import), and the two dependency lists must match exactly. A list
# built from `sys.argv` — or from anything else that differs between the two
# environments — fails at container start with
#     ExecutionError: Function has 4 dependencies but container got 5 object ids
#     ... "defining Modal objects under a conditional statement that evaluates
#     differently in the local and remote environments."
# Both definitions below are unconditional, so they evaluate identically on both
# sides. Users who never created `tailscale-auth` simply never invoke serve_ts,
# so the LiveKit-Cloud and quick-tunnel paths are unaffected by its existence.
@app.function(**_FN_KWARGS, secrets=_BASE_SECRETS)
def serve(**kwargs) -> None:
    """Standard path: LiveKit Cloud, or a local SFU via the Cloudflare tunnel."""
    _serve_impl(**kwargs)


@app.function(
    **_FN_KWARGS,
    secrets=[
        *_BASE_SECRETS,
        # Only this function needs it, so only this function's callers need the
        # secret to exist: `modal secret create tailscale-auth TS_AUTHKEY=tskey-...`
        modal.Secret.from_name("tailscale-auth", required_keys=["TS_AUTHKEY"]),
    ],
)
def serve_ts(**kwargs) -> None:
    """Tailscale-hybrid path (--tailscale): same body, plus the TS_AUTHKEY secret."""
    _serve_impl(**kwargs)


@app.local_entrypoint()
def main(  # nosec B107 — the empty `*_secret` defaults are "flag not passed", not credentials
    policy_path: str,
    task: str = "",
    horizon: int = 50,
    fps: int = 30,
    duration: float = 0.0,
    s_min: int = 4,
    slack: int = 5,
    tolerance: float = 1.5,
    max_guidance_weight: float = 10.0,
    rtc_schedule: str = "linear",
    video_codec: str = "H264",
    livekit_url: str = "",
    livekit_api_key: str = "",
    livekit_api_secret: str = "",
    tailscale: bool = False,
) -> None:
    # --tailscale routes to the twin function that carries the tailscale-auth
    # secret; everything else is identical (see the note above serve()).
    fn = serve_ts if tailscale else serve
    fn.remote(
        policy_path=policy_path,
        task=task,
        horizon=horizon,
        fps=fps,
        duration=duration,
        s_min=s_min,
        slack=slack,
        tolerance=tolerance,
        max_guidance_weight=max_guidance_weight,
        rtc_schedule=rtc_schedule,
        video_codec=video_codec,
        livekit_url=livekit_url,
        livekit_api_key=livekit_api_key,
        livekit_api_secret=livekit_api_secret,
        tailscale=tailscale,
    )
