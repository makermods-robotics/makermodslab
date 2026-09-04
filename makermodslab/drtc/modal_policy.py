"""Run the GPU-side policy loop (`policy.py`) on a Modal serverless GPU.

Identical to the wrapper in the earlier `lerobot_inference` prototype (a separate
project, not a directory in this repo) — the DRTC example's `policy.py` and wire
protocol are unchanged; all DRTC scheduling lives on the robot side. So the
policy side deploys to Modal exactly the same way.

This is a thin wrapper: it does NOT reimplement anything. It builds a GPU image,
feeds LiveKit + policy config through environment variables, and calls
``policy.main()`` unchanged. `policy.py` is a pure *outbound* WebRTC client — it
dials into the LiveKit SFU — so nothing needs to listen on a port and Modal's
outbound-only networking is a clean fit. The SFU itself is **not** here: we use
LiveKit Cloud (or later a self-hosted server) as a separate, publicly reachable
service so media stays on UDP.

Topology:

    robot_sync.py (on-prem) ─► LiveKit Cloud SFU ◄─ modal_policy.py (this, on Modal GPU)

One-time setup
--------------
1. Create a LiveKit Cloud project (https://cloud.livekit.io) and grab its
   `wss://` URL, API key, and API secret.
2. Stash them (plus the room) as a Modal secret — these become container env vars
   that `makermodslab.drtc._env.load_env()` / `_common.mint_token()` read directly:

       modal secret create LiveKit-cloud \
           LIVEKIT_URL=wss://<your-project>.livekit.cloud \
           LIVEKIT_API_KEY=<key> \
           LIVEKIT_API_SECRET=<secret> \
           LIVEKIT_ROOM=portal-lerobot-inference

3. Point the robot side at the SAME `LIVEKIT_URL` / key / secret / room — write
   them to `~/.cache/huggingface/lerobot/livekit.env` (see
   docs/drtc/livekit.env.example). The secret's `LIVEKIT_ROOM` is the DEFAULT
   room on the GPU side; `--livekit-room` overrides it per run, which is how a
   launcher pins both peers to the same room without editing the secret.

Run
---
    modal run makermodslab/drtc/modal_policy.py --policy-path ${HF_USER}/my_policy

    # language-conditioned (VLA) policies:
    modal run makermodslab/drtc/modal_policy.py --policy-path ${HF_USER}/my_policy --task "Put the lego brick in the box"

    # local SFU reached over the tailnet (signaling only; media still direct UDP):
    modal run makermodslab/drtc/modal_policy.py --policy-path ${HF_USER}/my_policy \
        --tailscale --livekit-url ws://100.x.y.z:7880 \
        --livekit-api-key <key> --livekit-api-secret <secret>

    # pin the room explicitly (mirrors robot_sync.py's --livekit_room):
    modal run makermodslab/drtc/modal_policy.py --policy-path ${HF_USER}/my_policy \
        --livekit-room portal-lerobot-inference

`--horizon` MUST match the robot side's `--horizon`. DRTC knobs (`--s_min`,
`--epsilon`, `--pacing`, ...) are flags on whichever robot script you run
(`robot_sync.py` or `robot_rtc.py`) and need nothing here. Deploy
(`modal deploy`) + `.spawn()` if you'd rather keep it running as a service.

Resetting a stuck/crashed run
------------------------------
`serve()` runs `policy.main()` once per invocation with no retry: if it raises
(LiveKit disconnect, CUDA OOM, a bad checkpoint, ...) the container's run just
ends and nothing is listening anymore. `modal deploy makermodslab/drtc/modal_policy.py` publishes
a `/reset` HTTP GET endpoint that re-spawns `serve()` with whatever args were
used on the last run (stashed in a `modal.Dict` each time `serve()` starts), so
you can restart a dead run from a browser tab without re-running the CLI:

    modal deploy makermodslab/drtc/modal_policy.py
    # ... note the printed reset endpoint URL, then whenever it dies:
    curl https://<workspace>--lerobot-drtc-policy-reset.modal.run
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
# Verbatim copy of the block in modal_policy_rtc.py (these two wrappers are
# deliberate near-duplicates; there is no shared import path into both images).
# Fix bugs in both, or factor into a `_tailscale.py` added to both images.
#
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
# ONE node in the admin console, not one per launch. Tailscale identifies a node
# by its NODE KEY, which lives in tailscaled's state file, so a stable identity
# is simply a state file that outlives the container — hence a Modal Volume.
# DEDICATED, not the 20 GB hf-cache: a 4 KB state commit must stay a 4 KB
# commit. See _tailscale_up for the login sequence and the collision rule.
_TS_STATE_VOLUME = "makermodslab-tailscale-state"
_TS_STATE_DIR = "/tailscale"  # mount point (see _FN_KWARGS["volumes"], beside /cache)
_TS_STATE_FILE = "/tailscale/tailscaled.state"  # the node key lives in here
_TS_RESUME_TIMEOUT = 20.0  # bounded keyless `tailscale up` before falling back to the key

ts_state = modal.Volume.from_name(_TS_STATE_VOLUME, create_if_missing=True)

_TS_NO_AUTHKEY = (
    "[tailscale] --tailscale needs TS_AUTHKEY in the container env, and it "
    "isn't there. Create a REUSABLE, NON-EPHEMERAL auth key in the Tailscale "
    "admin console (Settings -> Keys), then:\n"
    "    modal secret create tailscale-auth TS_AUTHKEY=tskey-auth-...\n"
    "and re-run with --tailscale (that flag routes to serve_ts, the twin "
    "function this secret is attached to). NON-ephemeral is deliberate: the "
    "control plane deletes an ephemeral node the moment it goes offline, which "
    "would make the persisted node key dead on the next launch and register a "
    "fresh modal-policy-N node instead of reusing this one."
)


def _looks_like_tailnet(host: str) -> bool:
    """True for a 100.64.0.0/10 CGNAT address, a MagicDNS name, or a bare name."""
    try:
        return ipaddress.ip_address(host) in ipaddress.ip_network("100.64.0.0/10")
    except ValueError:
        return host.endswith(".ts.net") or "." not in host


def _ephemeral_node() -> bool:
    """True when DRTC_TS_EPHEMERAL asks for the pre-2026-09-04 throwaway node.

    The escape hatch for a DELIBERATE parallel run: two containers sharing one
    state file log in as the same node and the later one displaces the earlier
    from the tailnet, so a second concurrent GPU has to be a different node.
    Read from the container env; `main()` forwards the operator's own
    `DRTC_TS_EPHEMERAL=1 modal run ...` into it as a remote kwarg, the same way
    it forwards LIVEKIT_API_KEY (Modal does not ship the caller's environment).
    """
    return os.environ.get("DRTC_TS_EPHEMERAL", "").strip().lower() in ("1", "true", "yes", "on")


def _commit_ts_state(when: str) -> None:
    """Persist /tailscale back to the Volume; never fatal.

    A Modal Volume write is container-local until it is committed. The function
    exiting normally flushes one too, but a run that is killed (the Lab's stop,
    a `modal run` Ctrl-C, a container reap) never reaches that — and losing the
    file loses the node identity, which is the whole point. A failed commit
    costs one extra row in the admin console next launch, not the session.
    """
    if _ephemeral_node():
        return
    try:
        ts_state.commit()
    except Exception as exc:
        print(f"[tailscale] state commit ({when}) failed: {exc}")
    else:
        print(f"[tailscale] node state committed ({when})")


def _tailscale_up_cmd(authkey: str | None, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run `tailscale up`, with `--auth-key` only when one is actually being used.

    The key is never printed and never returned: callers log a redacted line,
    and the CompletedProcess deliberately carries a scrubbed `args`.
    """
    argv = [
        "tailscale",
        f"--socket={_TS_SOCKET}",
        "up",
        f"--hostname={_TS_HOSTNAME}",
        # Don't rewrite the container's /etc/resolv.conf: MagicDNS here would
        # only confuse Hugging Face / PyPI lookups. Tailnet names are resolved
        # by tailscaled itself, via the SOCKS5 domain address type below.
        "--accept-dns=false",
        f"--timeout={int(timeout)}s",
    ]
    if authkey:
        argv.insert(3, f"--auth-key={authkey}")
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout + 15)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            ["tailscale", "up"], 124, "", f"`tailscale up` did not return within {timeout + 15:.0f}s"
        )
    return subprocess.CompletedProcess(["tailscale", "up"], done.returncode, done.stdout, done.stderr)


def _tailscale_up(timeout: float = 90.0) -> None:
    """Start userspace tailscaled and join the tailnet as ONE stable node.

    Blocks until the backend is Running.

    Tailscale identifies a node by its NODE KEY, which lives in tailscaled's
    state file — so "the same node every launch" is exactly "the same state file
    every launch". Until 2026-09-04 this ran `--state=mem:`, deliberately
    keeping no state at all so that dead containers left nothing behind; the
    price was a fresh `modal-policy-N` row per launch (21 of them by the time
    this changed). The state file now lives on a dedicated Modal Volume mounted
    at /tailscale, and the login sequence is:

      1. state file present -> `tailscale up --hostname=modal-policy` with NO
         `--auth-key`, bounded to _TS_RESUME_TIMEOUT. This is the normal path
         and it re-uses the saved node key.
      2. anything less than rc=0 from that (an expired/stale node key, a
         NeedsLogin that printed a login URL and then hit the timeout — we do
         not try to tell them apart, the answer is the same) -> the original
         `--auth-key` path, which is also what a first-ever run takes.

    Two consequences worth knowing:

      * the auth key must be REUSABLE and **NON-ephemeral**. `tailscale up` has
        no `--ephemeral` flag (ephemeral-ness lives on the KEY), and the control
        plane deletes an ephemeral node as soon as it goes offline — a persisted
        node key for a deleted node is dead on the next launch and re-registers
        as a new node, which is the row-per-launch problem again.
      * two containers sharing this state log in as the SAME node, and the later
        `tailscale up` displaces the earlier one from the tailnet. The Lab's
        one-launcher gate and its orphan reaper make that rare;
        `DRTC_TS_EPHEMERAL=1` is the escape hatch for a deliberate parallel run
        (it restores `--state=mem:` and always-`--auth-key`, byte-for-byte the
        old behaviour).
    """
    authkey = os.environ.get("TS_AUTHKEY")
    ephemeral = _ephemeral_node()
    if ephemeral:
        state = "mem:"
        resume = False
        print("[tailscale] DRTC_TS_EPHEMERAL=1: throwaway node (--state=mem:, always --auth-key)")
    else:
        state = _TS_STATE_FILE
        Path(_TS_STATE_DIR).mkdir(parents=True, exist_ok=True)
        resume = Path(_TS_STATE_FILE).exists()
        print(f"[tailscale] node state {_TS_STATE_FILE}: {'found' if resume else 'not there yet'}")
    if not authkey and not resume:
        raise SystemExit(_TS_NO_AUTHKEY)

    print(f"[tailscale] starting tailscaled (userspace networking, socks5 on 127.0.0.1:{_TS_SOCKS_PORT})")
    proc = subprocess.Popen(
        [
            "tailscaled",
            "--tun=userspace-networking",
            f"--socks5-server=localhost:{_TS_SOCKS_PORT}",
            f"--state={state}",
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
    up = None
    if resume:
        print(f"[tailscale] tailscale up --hostname={_TS_HOSTNAME} (resuming the saved node key)")
        up = _tailscale_up_cmd(None, _TS_RESUME_TIMEOUT)
        if up.returncode != 0:
            print(
                f"[tailscale] keyless resume failed (rc={up.returncode}): "
                f"{up.stdout.strip()}{up.stderr.strip()} — falling back to the auth key"
            )
            up = None
    if up is None:
        if not authkey:
            raise SystemExit(_TS_NO_AUTHKEY)
        print(f"[tailscale] tailscale up --hostname={_TS_HOSTNAME} --auth-key=<redacted>")
        up = _tailscale_up_cmd(authkey, timeout)
    if up.returncode != 0:
        raise RuntimeError(
            f"`tailscale up` failed (rc={up.returncode}): {up.stdout.strip()}{up.stderr.strip()}"
        )

    # Backend is Running, so tailscaled has written the node key. Commit it now
    # rather than at function exit: a killed container never reaches an exit.
    if not ephemeral:
        if Path(_TS_STATE_FILE).exists():
            _commit_ts_state("after login")
        else:
            print(
                f"[tailscale] WARNING: {_TS_STATE_FILE} was not written; the next launch "
                "will register a new node. Is /tailscale mounted?"
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
    the policy and connect.
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
# lerobot pulls in torch; recent PyPI torch wheels are CUDA-enabled on linux, so
# this image runs on GPU as-is. ffmpeg / libgl are lerobot's system deps.
image = (
    modal.Image.debian_slim(python_version="3.12")
    # curl + ca-certificates are for the Tailscale apt repo below (--tailscale).
    .apt_install(
        "git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential", "cmake", "curl", "ca-certificates"
    )
    .pip_install("uv")
    # lerobot is pinned to the SAME SHA as pyproject.toml — the Lab's fork,
    # `makermods-robotics/lerobot`. THE TWO PINS MOVE TOGETHER OR NOT AT ALL;
    # `tests/test_drtc_modal_wrappers.py` reads both files and asserts it, so
    # they cannot drift apart again the way they had by S3.7.
    #
    # Until 2026-09-03 this said upstream `huggingface/lerobot@8414188d` and
    # argued for it. Two things make the fork the right pin now:
    #   * upstream 8414188d is lerobot **0.5.2**, which has no `supports_rtc`
    #     anywhere in the tree — the RTC server cannot ask a policy whether
    #     in-painting is even possible, and MolmoAct2 cannot be served at all.
    #     The fork is 0.6.2 and declares it on seven policies.
    #   * the fork is what the LAB itself runs, and config compatibility follows
    #     the lerobot that WROTE the checkpoint, not the highest version number.
    #     Training on the fork and serving on upstream is the same class of
    #     mismatch the old comment warned about for PI05Config, just pointed the
    #     other way.
    # Do NOT use bare `lerobot` (PyPI) for either reason. Building from git needs
    # the `git` apt package (installed above). A bump here is a real lerobot bump
    # per CLAUDE.md: re-run a known-good SmolVLA RTC session on the new image
    # before declaring it good.
    #
    # `pydantic` is explicit rather than transitive: the servers import
    # `makermodslab.utils.system` (the policy-requirement helpers), which imports
    # it, and nothing else in this image pulls it in.
    #
    # livekit-portal comes from PyPI, pinned to the same version as
    # pyproject.toml's `[drtc]` extra so robot and GPU sides speak the identical
    # wire code (Portal fingerprints the schema and drops mismatched packets
    # SILENTLY — a healthy-looking session with zero chunks).
    .run_commands(
        "uv pip install --system --compile-bytecode "
        # fastapi is this wrapper's alone: it serves the /reset endpoint.
        '"fastapi[standard]" '
        '"livekit-api>=0.7" "python-dotenv>=1" "numpy>=1.24" "pydantic>=2" '
        '"livekit-portal==0.2.4" '
        # [pi,smolvla,molmoact2] pulls the flow-policy runtime deps
        # (transformers, scipy, accelerate, num2words, peft). pi0/pi05/smolvla/
        # molmoact2 all import transformers; without the extras, from_pretrained
        # fails with "transformers is required". molmoact2 adds only peft (~1 MB)
        # and scipy (~35 MB) on top of what [pi,smolvla] already pulls, and all
        # three extras resolve the same transformers range — no conflict.
        '"lerobot[pi,smolvla,molmoact2] @ git+https://github.com/makermods-robotics/lerobot.git'
        '@eaab69339120787948776e4354dcee09f501fd16"'
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
    # Container env. HF_HOME points the Hugging Face cache at the persistent
    # Volume below, so checkpoints download once and survive cold starts /
    # policy switches. The *_OFFLINE=0 pins guarantee online lookups regardless
    # of any stray env — a fresh container's cache is empty, so offline is always
    # wrong here. Set before the Python process starts, so huggingface_hub reads
    # the right value at import (it freezes HF_HUB_OFFLINE into a constant then).
    #
    # This MUST come before add_local_dir below: files added via `add_local_*`
    # without copy=True are attached at container startup, not baked into the
    # image, so Modal forbids any further build step (.env, .run_commands, ...)
    # after them. Keep add_local_dir LAST.
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
    # resolves modules through the LOCAL interpreter. LAST build step (see the
    # .env note above).
    .add_local_dir(
        _PACKAGE_DIR,
        remote_path="/root/makermodslab",
        ignore=["**/__pycache__", "**/*.pyc"],
    )
)

# Persistent HF cache: first run of any policy downloads into this Volume; every
# run after reads from it (no re-download on cold start or when switching
# --policy-path). Mounted at /cache to match HF_HOME above.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Remembers the args `serve()` was last started with, so the /reset endpoint
# below can re-spawn an equivalent run without the caller having to know or
# repeat --policy-path / --task / etc.
run_config = modal.Dict.from_name("lerobot-drtc-policy-run-config", create_if_missing=True)

app = modal.App("lerobot-drtc-policy")


# Everything except `secrets=`, shared by the two Modal functions below so the
# GPU/region/timeout knobs stay in ONE place.
_FN_KWARGS = {
    "image": image,
    # A10G suffices for small policies (ACT); large VLAs (pi0 / SmolVLA) need
    # A100 VRAM — so "A100" is the pin, and it is what a hand-typed `modal run`
    # gets. DRTC_GPU is how the Lab's launcher overrides it per launch: this
    # dict is evaluated when `modal run` IMPORTS this file on the operator's own
    # machine, before Click parses a flag, so an env var is the only channel a
    # caller has to the decorator (see makermodslab/modal_launcher.py).
    "gpu": os.environ.get("DRTC_GPU") or "A100",
    "timeout": 60 * 60 * 2,  # hard session cap; PORTAL_DURATION_SECONDS can end sooner
    # /cache: persistent HF cache (see hf_cache / HF_HOME). /tailscale: the
    # tailnet node key, on its OWN Volume so a 4 KB state commit never drags the
    # 20 GB model cache with it (see _TS_STATE_FILE). Mounted on both twins —
    # `serve` never writes it, and identical kwargs keep the two declarations
    # from drifting.
    "volumes": {"/cache": hf_cache, _TS_STATE_DIR: ts_state},
    # Region: place the GPU near the LiveKit Cloud edge the robot uses, so media
    # doesn't cross an ocean. For a robot in China, "ap-southeast" (Singapore)
    # gives the best reachability of Modal's Asia regions; "ap-northeast"
    # (Seoul/Tokyo) is the alternative — pick whichever your robot's ISP routes
    # to with lower RTT. Any China<->Modal path still crosses the GFW (Modal has
    # no mainland-China presence), but co-locating in Asia cuts the ~440ms
    # trans-Pacific RTT to roughly robot<->Singapore (~60-150ms).
    # "region": "ap-southeast",
    "region": "us-west",
    # "min_containers": 1,  # pre-warm to avoid a cold start when the robot connects
    #                       # (keeps an idle GPU billed — opt in for latency-critical runs)
}

# Secrets, split out for the same reason: `serve` and `serve_ts` differ ONLY in
# the tailscale entry. --policy-path is an HF Hub repo id; from_pretrained
# downloads it in the container. huggingface_hub auto-reads HF_TOKEN from the
# env, so the "huggingface" secret is all that's needed to auth. REQUIRED only
# if the policy repo (or a VLA's base backbone, e.g. SmolVLM) is private/gated —
# drop it if everything you pull is public. Create it with:
#   modal secret create huggingface HF_TOKEN=hf_...
_BASE_SECRETS = [
    modal.Secret.from_name("LiveKit-cloud"),
    modal.Secret.from_name("huggingface"),
]


def _serve_impl(  # nosec B107 — the empty `*_secret` defaults are "flag not passed", not credentials
    policy_path: str,
    task: str = "",
    model_dtype: str = "",
    flow_steps: int = 0,
    horizon: int = 16,
    fps: int = 30,
    duration: float = 0.0,
    video_codec: str = "H264",  # MUST match robot_sync.py's --video-codec default
    livekit_url: str = "",
    livekit_api_key: str = "",
    livekit_api_secret: str = "",
    livekit_room: str = "",
    tailscale: bool = False,
    ts_ephemeral: bool = False,
) -> None:
    # Shipped via add_local_dir; imported HERE (not at module top) because this
    # wrapper is also evaluated locally by the modal CLI, whose interpreter has
    # no makermodslab on its path.
    from makermodslab.drtc import policy

    # `DRTC_TS_EPHEMERAL=1 modal run ...` is set in the OPERATOR's shell; Modal
    # does not ship the caller's environment, so main() resolves it locally and
    # it arrives here as a kwarg. Re-exporting it into the container env keeps
    # _ephemeral_node() the single reader (it is also honoured directly, for a
    # secret or an image env that sets it container-side).
    if ts_ephemeral:
        os.environ["DRTC_TS_EPHEMERAL"] = "1"

    # Optional per-run LiveKit override: point this run at a different SFU than
    # the LiveKit-cloud secret — e.g. the LOCAL SFU exposed through a Cloudflare
    # Lab's own SFU (`makermodslab --sfu`), whose url/key/secret the Deploy
    # panel prints as a ready-made line (see docs/drtc/README.md "Local SFU").
    # They ride per-run CLI args rather than a Modal secret because they are
    # per-station, not per-account. Unset flags fall through to the secret.
    if livekit_url:
        os.environ["LIVEKIT_URL"] = livekit_url
    if livekit_api_key:
        os.environ["LIVEKIT_API_KEY"] = livekit_api_key
    if livekit_api_secret:
        os.environ["LIVEKIT_API_SECRET"] = livekit_api_secret

    # `LIVEKIT_ROOM` used to be settable ONLY by the `LiveKit-cloud` secret, which
    # made "the two peers are in different rooms" a silent failure — they simply
    # never see each other, and the robot reports a healthy connection with zero
    # chunks forever. `--livekit-room` closes that class: a launcher that already
    # knows which room the robot is in can pin this run to it. Unset still falls
    # through to the secret, so every existing invocation is unchanged.
    if livekit_room:
        os.environ["LIVEKIT_ROOM"] = livekit_room

    # Tailscale hybrid: join the tailnet, stand up the loopback->SOCKS5 relay,
    # and point LIVEKIT_URL at the relay instead of the tailnet address. Media
    # and data channels are untouched — they still hole-punch straight to the
    # Mac's public IP on UDP 7882 (which is what `--sfu-external-ip` advertises).
    if tailscale:
        target = livekit_url or os.environ.get("LIVEKIT_URL", "")
        if not target:
            raise SystemExit("[tailscale] --tailscale needs --livekit-url ws://<tailnet-ip>:7880")
        os.environ["LIVEKIT_URL"] = _tailscale_signaling_url(target)
        print(
            f"[tailscale] signaling {target} -> {os.environ['LIVEKIT_URL']} "
            "(media/data still direct UDP 7882)"
        )

    # Remember these args so a later /reset call can re-spawn an equivalent run.
    # Stashed on every start (not just the first) so /reset always replays
    # whatever was launched most recently, including a prior reset. NOTE: a
    # /reset that replays a dead quick-tunnel livekit_url will fail to connect —
    # restart the tunnel and re-run the CLI instead.
    run_config["last"] = {
        "policy_path": policy_path,
        "task": task,
        "model_dtype": model_dtype,
        "flow_steps": flow_steps,
        "horizon": horizon,
        "fps": fps,
        "duration": duration,
        "video_codec": video_codec,
        "livekit_url": livekit_url,
        "livekit_api_key": livekit_api_key,
        "livekit_api_secret": livekit_api_secret,
        "livekit_room": livekit_room,
        # Recorded so /reset replays onto the right twin (serve_ts vs serve).
        "tailscale": tailscale,
        # So a /reset replays onto the same tailnet identity the run chose.
        # A dict written before 2026-09-04 has no such key and reads back as
        # False, i.e. the stable node — which is the right default.
        "ts_ephemeral": ts_ephemeral,
    }

    # policy.py is CLI-only for behavior config (no env fallback), so forward every
    # setting as a command-line flag. Only the LiveKit creds still ride the
    # environment (via the Modal secret), which is what policy.py reads from there.
    # --fps / --horizon / --video-codec MUST match robot_sync.py.
    argv = [
        "policy.py",
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
    ]
    if model_dtype:
        # OPERATOR OPT-IN, never a default: the server overrides the config's
        # saved `model_dtype` before weights load. The checkpoint's own value is
        # a deliberate choice by whoever trained it (the published MolmoAct2
        # saves float32), so the Lab does not second-guess it silently — but a
        # ~7B model in fp32 does not fit comfortably on a 40 GB A100 and gets no
        # autocast, so `--model-dtype bfloat16` is the escape hatch. Measure
        # first, then decide.
        argv += ["--model-dtype", model_dtype]
    if flow_steps:
        # Same shape and the same rule as --model-dtype: an OPERATOR OPT-IN the
        # server applies before weights load, never a default this side picks.
        # Unset is 0 — no sampler takes zero steps, so the absence of the flag
        # is the only way to say "leave the checkpoint's own count alone" — and
        # WHICH config field it writes is per policy family (see
        # utils.system.POLICY_FLOW_STEPS_FIELDS). Fewer steps is less GPU work
        # per chunk and a coarser action trajectory; measure before trusting it.
        argv += ["--flow-steps", str(flow_steps)]
    if task:
        argv += ["--task", task]
    sys.argv = argv
    try:
        asyncio.run(policy.main())
    finally:
        # Commit again on the way out: Tailscale rotates a node key periodically,
        # and the rotated one is only ours if it reaches the Volume. Runs on the
        # disconnect path AND on an exception, which is the case that matters.
        if tailscale:
            _commit_ts_state("shutdown")


# TWO Modal functions, identical except that `serve_ts` also gets the
# `tailscale-auth` secret; `main()` (and /reset) picks one by the --tailscale
# flag.
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


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def reset() -> dict:
    """Visit this URL to re-spawn `serve()` with the args it was last started
    with — the manual "kick it back to life" button for a run that died on
    the GPU side (LiveKit disconnect, OOM, bad checkpoint, ...).

    Fires a new run and returns immediately; does not touch any run already
    in flight (Modal just ends up with two `serve()` calls for a bit, which
    is harmless since the stale one is presumably already dead or stuck).
    """
    try:
        last = run_config["last"]
    except KeyError:
        return {
            "ok": False,
            "error": "no prior run recorded yet — start one with `modal run makermodslab/drtc/modal_policy.py --policy-path ...` first",
        }

    # Replay onto the same twin the run started on: a --tailscale run must go
    # back to serve_ts, which is the one carrying the tailscale-auth secret.
    use_ts = bool(last.get("tailscale"))
    call = (serve_ts if use_ts else serve).spawn(**last)
    return {
        "ok": True,
        "respawned_with": last,
        "call_id": call.object_id,
        "function": "serve_ts" if use_ts else "serve",
    }


@app.local_entrypoint()
def main(  # nosec B107 — the empty `*_secret` defaults are "flag not passed", not credentials
    policy_path: str,
    task: str = "",
    model_dtype: str = "",
    flow_steps: int = 0,
    horizon: int = 16,
    fps: int = 30,
    duration: float = 0.0,
    video_codec: str = "H264",  # MUST match robot_sync.py's --video-codec default
    livekit_url: str = "",
    livekit_api_key: str = "",
    livekit_api_secret: str = "",
    livekit_room: str = "",
    tailscale: bool = False,
) -> None:
    """Resolve the run's arguments locally, then fire the container.

    THIS BODY RUNS ON THE USER'S MACHINE: a @local_entrypoint is executed by
    the local `modal` CLI, never in the container. That is why the two
    credentials below may also arrive in the ENVIRONMENT — a
    `--livekit-api-secret <secret>` flag would put a signing key in `ps` for
    every process on that machine to read, so the Lab's launcher
    (makermodslab/modal_launcher.py) passes them as env instead. The resolved
    value then travels to the container as a `fn.remote(...)` kwarg over
    Modal's own TLS channel.
    """
    # The flag still wins when present, so every hand-typed invocation and
    # every line in docs/drtc/README.md is unchanged. Scoped to the two
    # credentials ONLY, deliberately not to --livekit-url / --livekit-room:
    # those are not secrets, and keeping them flag-only means "which SFU, which
    # room" stays a VISIBLE decision rather than one a stray LIVEKIT_ROOM in an
    # operator's shell can flip — which is the exact failure class
    # --livekit-room was added to close.
    livekit_api_key = livekit_api_key or os.environ.get("LIVEKIT_API_KEY", "")
    livekit_api_secret = livekit_api_secret or os.environ.get("LIVEKIT_API_SECRET", "")

    # Same channel, same reason: `DRTC_TS_EPHEMERAL=1 modal run ...` is read in
    # THIS body (the operator's shell) and travels as a kwarg, because Modal
    # does not ship the caller's environment into the container. Unset is the
    # stable-node default; see _tailscale_up.
    ts_ephemeral = _ephemeral_node()

    # --tailscale routes to the twin function that carries the tailscale-auth
    # secret; everything else is identical (see the note above serve()).
    fn = serve_ts if tailscale else serve
    fn.remote(
        policy_path=policy_path,
        task=task,
        model_dtype=model_dtype,
        flow_steps=flow_steps,
        horizon=horizon,
        fps=fps,
        duration=duration,
        video_codec=video_codec,
        livekit_url=livekit_url,
        livekit_api_key=livekit_api_key,
        livekit_api_secret=livekit_api_secret,
        livekit_room=livekit_room,
        tailscale=tailscale,
        ts_ephemeral=ts_ephemeral,
    )
