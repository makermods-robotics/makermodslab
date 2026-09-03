# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MakerMods Lab launcher.

Default mode: starts the FastAPI backend on :8000, which serves the
pre-built frontend at /. Opens the user's browser to the local app.

--dev mode: spawns the Vite dev server (frontend/, port 8080) for HMR
and starts uvicorn with --reload. Opens the browser to :8080.

--sfu (either mode): also runs a LiveKit SFU (`livekit-server`, from PATH)
alongside, bound where the API is bound, and hands the app the key file so
/api/v1/sfu/token can sign room tokens. The launcher — not the app — owns
that child: uvicorn --reload restarts the app process on every save. In --dev
mode --bind is honoured for the SFU ALONE (Vite and uvicorn stay on
localhost), because a remote peer has to reach its signalling port.
"""

import argparse
import contextlib
import ipaddress
import logging
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import psutil
import uvicorn

from makermodslab import sfu
from makermodslab.utils.config import LIVEKIT_CONFIG_FILE, LIVEKIT_KEY_FILE, load_or_create_livekit_keys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
FRONTEND_PATH = PROJECT_ROOT / "frontend"
FRONTEND_DIST = FRONTEND_PATH / "dist"
FRONTEND_PACKAGE_JSON = FRONTEND_PATH / "package.json"
BACKEND_PORT = 8000
FRONTEND_DEV_PORT = 8080
ENTRY_POINT_NAMES = ("makermodslab", "makermodslab-station")
# `uv tool install` lays down a symlink in ~/.local/bin that resolves into
# this tree (verified empirically: ~/.local/bin/<exe> ->
# ~/.local/share/uv/tools/<tool>/bin/<exe>). We use containment under this
# dir to recognise a uv-managed entry and refuse to clobber it.
UV_TOOLS_DIR = Path.home() / ".local" / "share" / "uv" / "tools"


def _is_uv_tool_link(link: Path, uv_tools_dir: Path = UV_TOOLS_DIR) -> bool:
    """True if `link` is a symlink whose target lives under uv's tools dir.

    That is the fingerprint of a `uv tool install` executable — a separate,
    self-contained flavor we must never silently overwrite with a venv link.
    """
    if not link.is_symlink():
        return False
    try:
        target = link.resolve()
        uv_root = uv_tools_dir.resolve()
    except OSError:
        return False
    return target == uv_root or uv_root in target.parents


def _ensure_path_symlinks(
    source_dir: Path | None = None,
    bin_dir: Path | None = None,
    uv_tools_dir: Path = UV_TOOLS_DIR,
) -> None:
    """Self-install the entry points onto PATH (idempotent, best-effort).

    pip has no post-install hook, so the first run by full path does the
    README.md symlink step itself: each venv entry point gets a symlink in
    ~/.local/bin. Correct links are left alone; stale symlinks (an old
    clone's venv) are repointed; anything that is NOT a symlink is never
    clobbered. A name already owned by a `uv tool install` (its symlink
    resolves under uv's tools dir) is left alone too — both flavors are
    present, and we tell the user how to pick one rather than fight it.
    Failures only log — PATH convenience must never block a server start.
    Set MAKERMODSLAB_NO_PATH_LINK=1 to opt out.
    """
    if os.name != "posix" or os.environ.get("MAKERMODSLAB_NO_PATH_LINK"):
        return
    try:
        source_dir = source_dir or Path(sys.executable).parent
        bin_dir = bin_dir or Path.home() / ".local" / "bin"
        created: list[str] = []
        for name in ENTRY_POINT_NAMES:
            source = source_dir / name
            if not source.is_file():
                continue  # partial env (entry point not installed here)
            link = bin_dir / name
            if _is_uv_tool_link(link, uv_tools_dir):
                logger.info(
                    "`%s` on your PATH is a `uv tool install` (%s), not a venv "
                    "symlink — leaving it. Both install flavors are present; pick "
                    "one: `uv tool uninstall %s` to prefer this checkout, or set "
                    "MAKERMODSLAB_NO_PATH_LINK=1 to keep the tool install and silence this.",
                    name,
                    link,
                    name,
                )
                continue
            if link.is_symlink():
                if link.resolve() == source.resolve():
                    continue
                link.unlink()  # stale: points into an old venv/clone
            elif link.exists():
                logger.warning(
                    "Not shadowing %s — it exists and is not a symlink; remove it "
                    "manually if `%s` should run this venv's copy.",
                    link,
                    name,
                )
                continue
            bin_dir.mkdir(parents=True, exist_ok=True)
            link.symlink_to(source)
            created.append(name)
        if created:
            logger.info(
                "🔗 Linked %s into %s — new shells can run them from any directory",
                ", ".join(created),
                bin_dir,
            )
            if str(bin_dir) not in os.environ.get("PATH", "").split(os.pathsep):
                logger.warning("%s is not on your PATH — add it in your shell profile", bin_dir)
    except Exception as exc:
        logger.debug("PATH symlink self-install skipped: %s", exc)


def _wait_for_port(port: int, timeout: int = 30) -> bool:
    for _ in range(timeout):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        if result == 0:
            return True
        time.sleep(1)
    return False


def _is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def _ensure_port_available(name: str, port: int, host: str = "127.0.0.1") -> None:
    """Fail fast if `port` is already taken on the host we're about to bind.

    A previous run that left an orphaned uvicorn/Vite holding the port is the
    common cause, so point the user at `makermodslab --stop` to reclaim it.
    """
    if not _is_port_open(port, host):
        return
    logger.error("❌ %s port %d is already in use on %s.", name, port, host)
    logger.error(
        "   If a previous MakerMods Lab run is still holding it, run `makermodslab --stop` "
        "to free it, then run the command again."
    )
    sys.exit(1)


# The console-script names pyproject installs for the launcher. The prod
# identity signal below matches on these EXACT basenames, never on a substring,
# so `vim makermodslab` or `tail -f makermodslab.log` can never qualify.
_LAUNCHER_SCRIPT_NAMES = frozenset({"makermodslab", "makermodslab-station"})


def _identity_reason(cmdline: str, proc: psutil.Process) -> str | None:
    """Why `proc` is recognisably one of ours, or None if it isn't.

    Three independent identity signals, deliberately narrow so `--stop` never
    touches an unrelated dev server that merely happens to hold :8000/:8080:
      1. cmdline runs `uvicorn ... makermodslab.server` — DEV mode only: the
         reload supervisor is a subprocess with the app string in its argv.
      2. an orphaned reload worker (`multiprocessing.spawn` / `spawn_main`)
         whose cwd is THIS project checkout.
      3. the prod launcher itself. Prod mode runs uvicorn IN-PROCESS
         (`uvicorn.Config("makermodslab.server:app", ...)`), so signal 1's
         string never appears in its argv — the cmdline is just
         `.../python3 .../bin/makermodslab --lan ...`, and before this signal
         existed `--stop` reported every prod server as a port stranger and
         refused to touch it. Only the EXECUTED SCRIPT counts: argv[0]
         directly, or argv[1] when argv[0] is a python interpreter (how a
         shebang script shows up in ps) — an unrelated process merely naming
         the launcher in an argument never qualifies.
    """
    if "makermodslab.server" in cmdline:
        return "uvicorn (makermodslab.server)"
    if sfu.BINARY_NAME in cmdline and LIVEKIT_CONFIG_FILE in cmdline:
        # The SFU child we spawned: livekit-server pointed at OUR generated
        # config. A user's own livekit-server (different config) is a stranger.
        return "livekit-server (--sfu)"
    if "multiprocessing.spawn" in cmdline or "spawn_main" in cmdline:
        with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            if Path(proc.cwd()) == PROJECT_ROOT:
                return "orphaned reload worker"
    argv = list(getattr(proc, "info", {}).get("cmdline") or [])
    if not argv:
        with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            argv = proc.cmdline() or []
    head = [Path(token).name for token in argv[:2]]
    if head and head[0] in _LAUNCHER_SCRIPT_NAMES:
        return "prod launcher (makermodslab)"
    if len(head) == 2 and head[0].startswith("python") and head[1] in _LAUNCHER_SCRIPT_NAMES:
        return "prod launcher (makermodslab)"
    return None


def _listening_port(proc: psutil.Process, ports: set[int]) -> int | None:
    """The first of `ports` that `proc` is LISTENING on, or None."""
    try:
        get_conns = getattr(proc, "net_connections", None) or proc.connections
        for conn in get_conns(kind="inet"):
            if conn.laddr and conn.laddr.port in ports and conn.status == psutil.CONN_LISTEN:
                return conn.laddr.port
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    return None


def _find_makermodslab_pids() -> tuple[dict[int, str], dict[int, tuple[int, str]]]:
    """Partition candidate processes into (kill_targets, port_strangers).

    kill_targets: pid -> reason, for anything matching an identity signal
      (see `_identity_reason`). These are safe to terminate.
    port_strangers: pid -> (port, name), for a process LISTENING on one of our
      ports that matches NO identity signal. We report these but never kill
      them — they might be someone else's server on the same port.
    """
    me = os.getpid()
    ports = {BACKEND_PORT, FRONTEND_DEV_PORT, sfu.SFU_HTTP_PORT, sfu.SFU_TCP_PORT}
    kill_targets: dict[int, str] = {}
    strangers: dict[int, tuple[int, str]] = {}
    for proc in psutil.process_iter(["pid", "cmdline", "name"]):
        pid = proc.info["pid"]
        if pid == me:
            continue
        cmdline = " ".join(proc.info.get("cmdline") or [])
        reason = _identity_reason(cmdline, proc)
        if reason is not None:
            kill_targets[pid] = reason
            continue
        listening = _listening_port(proc, ports)
        if listening is not None:
            strangers[pid] = (listening, proc.info.get("name") or "?")
    return kill_targets, strangers


def _terminate_tree(pid: int, timeout: int = 5) -> None:
    """Terminate a process and every descendant.

    Dev mode's children are themselves process trees (npm -> node -> vite, and
    uvicorn --reload -> reloader -> worker). Signalling only the direct child
    leaves grandchildren orphaned still holding :8000/:8080, so walk the whole
    tree: terminate → wait → kill any survivors.
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True)
    procs.append(parent)
    for proc in procs:
        with contextlib.suppress(psutil.NoSuchProcess):
            proc.terminate()
    _gone, alive = psutil.wait_procs(procs, timeout=timeout)
    for proc in alive:
        with contextlib.suppress(psutil.NoSuchProcess):
            proc.kill()


def _run_stop() -> None:
    """Stop a running MakerMods Lab and free :8000 / :8080, then return.

    The escape hatch for when a previous run left an orphaned Vite or uvicorn
    holding the ports. Identity-scoped: we only kill processes we recognise as
    ours, and merely warn about an unrelated stranger on the same port.
    """
    kill_targets, strangers = _find_makermodslab_pids()
    for pid, (port, name) in strangers.items():
        logger.warning(
            "port %d is held by pid %d (%s) — not a MakerMods Lab process; not touching it. "
            "Stop it manually if it's stale.",
            port,
            pid,
            name,
        )
    if not kill_targets:
        if not strangers:
            logger.info(
                "Nothing to stop: no MakerMods Lab process found on :%d / :%d / :%d.",
                BACKEND_PORT,
                FRONTEND_DEV_PORT,
                sfu.SFU_HTTP_PORT,
            )
        return
    for pid, reason in kill_targets.items():
        logger.info("🛑 Stopping pid %d (%s)...", pid, reason)
        _terminate_tree(pid)
    logger.info("✅ MakerMods Lab stopped.")


def _resolve_bind_host(value: str) -> str:
    """Resolve a --bind value to a bindable address.

    A literal IP (v4 or v6) is used as-is; anything else is treated as a
    network interface name and resolved to that interface's first IPv4 via
    psutil.net_if_addrs. Raises ValueError — an unknown interface (the error
    lists the ones that exist) or one without an IPv4 must fail fast before
    anything starts, never fall back to a broader bind than the user asked
    for.
    """
    with contextlib.suppress(ValueError):
        return str(ipaddress.ip_address(value))
    interfaces = psutil.net_if_addrs()
    if value not in interfaces:
        raise ValueError(
            f"{value!r} is neither an IP address nor a network interface on this machine "
            f"(interfaces here: {', '.join(sorted(interfaces))})"
        )
    for addr in interfaces[value]:
        if addr.family == socket.AF_INET:
            return addr.address
    raise ValueError(f"interface {value!r} has no IPv4 address to bind")


def _require_livekit_server() -> str:
    """Path to livekit-server, or a clean one-line exit with the per-OS
    install hint. Called from main() before anything starts, so a missing
    binary never leaves a half-started server behind."""
    binary = sfu.find_livekit_server()
    if binary:
        return binary
    override = os.environ.get(sfu.ENV_BIN)
    if override:
        logger.error("❌ --sfu: %s=%s is not a file.", sfu.ENV_BIN, override)
    else:
        logger.error("❌ --sfu needs `%s` on your PATH and it was not found.", sfu.BINARY_NAME)
    logger.error("   %s", sfu.install_hint(platform.system()))
    logger.error("   (or point %s at the binary)", sfu.ENV_BIN)
    sys.exit(1)


def _start_sfu(binary: str, host: str, external_ip: bool = False) -> subprocess.Popen:
    """Spawn livekit-server bound like the API, wait for its signalling port,
    and export the app-side settings (key file + port) into THIS process's
    environment — which both the prod uvicorn (same process) and the dev
    uvicorn subprocess (env copy) inherit.

    Ports are checked first so an orphan from a previous run is a clear
    `makermodslab --stop` hint rather than a livekit bind error. The child
    gets its own session so Ctrl+C reaches it through _terminate_tree and
    never as a stray SIGINT that races our own shutdown.

    `external_ip` (--sfu-external-ip) is passed straight to
    `sfu.render_config`, and exported so the app can REPORT it: a Modal
    container reaches the signalling URL over the tailnet but has to
    hole-punch for media, and only the STUN-discovered public candidate that
    flag turns on is punchable from there.
    """
    for name, port in (("SFU signalling", sfu.SFU_HTTP_PORT), ("SFU ICE/TCP", sfu.SFU_TCP_PORT)):
        _ensure_port_available(name, port, host)
    key_file = LIVEKIT_KEY_FILE
    load_or_create_livekit_keys(key_file)
    config_text = sfu.render_config(bind_host=host, key_file=key_file, external_ip=external_ip)
    Path(LIVEKIT_CONFIG_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(LIVEKIT_CONFIG_FILE).write_text(config_text)

    logger.info("📡 Starting LiveKit SFU on ws://%s:%d ...", sfu.public_host(host), sfu.SFU_HTTP_PORT)
    proc = subprocess.Popen(
        [binary, "--config", LIVEKIT_CONFIG_FILE, "--key-file", key_file],
        start_new_session=True,
    )
    if not _wait_for_port(sfu.SFU_HTTP_PORT, timeout=15):
        logger.error("❌ LiveKit SFU never came up on :%d (see its log lines above)", sfu.SFU_HTTP_PORT)
        _terminate_tree(proc.pid)
        sys.exit(1)
    os.environ[sfu.ENV_KEY_FILE] = key_file
    os.environ[sfu.ENV_PORT] = str(sfu.SFU_HTTP_PORT)
    os.environ[sfu.ENV_EXTERNAL_IP] = "1" if external_ip else "0"
    if external_ip:
        logger.info("   SFU advertising its STUN-discovered public IP (--sfu-external-ip)")
    logger.info(
        "   SFU ports: %d/tcp (signalling), %d/tcp + %d/udp (media) — open these for remote peers",
        sfu.SFU_HTTP_PORT,
        sfu.SFU_TCP_PORT,
        sfu.SFU_UDP_PORT,
    )
    return proc


def _watch_sfu(proc: subprocess.Popen, server: uvicorn.Server) -> None:
    """Daemon-thread body for prod: if the SFU child dies, stop uvicorn too.
    A silently missing SFU would leave /sfu/token handing out tokens for a
    server nobody can reach; better to exit loudly and let the operator
    (or systemd's Restart=) bring both back."""
    while not server.should_exit:
        if proc.poll() is not None:
            logger.error("❌ LiveKit SFU exited (code %s) — shutting down", proc.returncode)
            server.should_exit = True
            return
        time.sleep(1)


def _open_browser_when_ready():
    """Background-thread helper: poll the port, open the browser when up."""
    for _ in range(60):
        try:
            with socket.create_connection(("127.0.0.1", BACKEND_PORT), timeout=0.5):
                pass
        except OSError:
            time.sleep(0.5)
            continue
        logger.info("🌐 Opening browser...")
        webbrowser.open(f"http://localhost:{BACKEND_PORT}/")
        return


def _run_prod(
    lan: bool = False,
    no_ui: bool = False,
    host: str | None = None,
    sfu_bin: str | None = None,
    sfu_external_ip: bool = False,
):
    """Serve built frontend from backend on a single port.

    `lan` binds 0.0.0.0 for headless stations serving other machines on the
    network; it also skips the open-a-local-browser step (there is no local
    browser worth opening in that deployment). `host` is an already-resolved
    --bind address and takes precedence over the --lan/default choice (main()
    logs when both were given). `no_ui` skips serving (and requiring) the
    built frontend entirely — a pure API node. `sfu_bin` (--sfu) runs a
    LiveKit SFU alongside, bound to the same host, for the process lifetime.
    """
    if not no_ui and not FRONTEND_DIST.exists():
        logger.error(f"❌ Built frontend not found at {FRONTEND_DIST}")
        logger.error("   Run `npm run build` in frontend/ first, or use `makermodslab --dev`.")
        sys.exit(1)

    if host is None:
        host = "0.0.0.0" if lan else "127.0.0.1"  # noqa: S104  # nosec B104 — binds all interfaces only behind the explicit --lan opt-in; loopback otherwise
    _ensure_port_available("Backend", BACKEND_PORT, host)
    sfu_proc = _start_sfu(sfu_bin, host, sfu_external_ip) if sfu_bin else None
    if host == "127.0.0.1":
        logger.info("🚀 Starting MakerMods Lab on http://localhost:%d ...", BACKEND_PORT)
        if not no_ui:
            threading.Thread(target=_open_browser_when_ready, daemon=True).start()
    else:
        # A non-loopback bind (LAN or --bind) serves other machines: log the
        # real bind and don't open a browser at an address we may not answer.
        logger.info(
            "🚀 Starting MakerMods Lab on http://%s:%d%s ...",
            host,
            BACKEND_PORT,
            " (LAN)" if host == "0.0.0.0" else "",  # noqa: S104  # nosec B104 — log-label comparison, not a bind; the bind above carries its own justification
        )

    # Run uvicorn in the main thread so its native SIGINT handler works,
    # and bound graceful shutdown so a stuck WebSocket can't hang Ctrl+C.
    config = uvicorn.Config(
        "makermodslab.server:app",
        host=host,
        port=BACKEND_PORT,
        log_level="info",
        reload=False,
        timeout_graceful_shutdown=2,
    )
    server = uvicorn.Server(config)

    if os.name == "nt":
        # On Windows, uvicorn's graceful shutdown frequently hangs on Ctrl+C
        # (the asyncio Proactor loop doesn't wind down cleanly), leaving the
        # terminal stuck. Take over signal handling: stop hard and reap any
        # child subprocesses (training/recording/inference) so the prompt
        # always returns. On macOS/Linux uvicorn's native handlers give us the
        # bounded graceful shutdown above, so leave them in place.
        server.install_signal_handlers = lambda: None

        def _shutdown(_signum, _frame) -> None:
            logger.info("🛑 Shutting down...")
            try:
                for child in psutil.Process().children(recursive=True):
                    with contextlib.suppress(psutil.NoSuchProcess):
                        child.terminate()
            except Exception:
                pass
            os._exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        for _name in ("SIGTERM", "SIGBREAK"):
            _sig = getattr(signal, _name, None)
            if _sig is not None:
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(_sig, _shutdown)

    if sfu_proc is not None:
        threading.Thread(target=_watch_sfu, args=(sfu_proc, server), daemon=True).start()
    try:
        server.run()
    finally:
        # uvicorn's own graceful shutdown has run by now; the SFU child is
        # ours to reap. Idempotent if _watch_sfu saw it die already.
        if sfu_proc is not None:
            _terminate_tree(sfu_proc.pid)
            logger.info("  ✅ LiveKit SFU stopped")


def _run_dev(
    sfu_bin: str | None = None,
    sfu_external_ip: bool = False,
    sfu_host: str = "127.0.0.1",
):
    """Vite dev server (HMR) + uvicorn --reload (+ the LiveKit SFU with --sfu).

    `sfu_host` is the ONE thing `--bind` still means in dev mode. Vite and
    uvicorn stay on loopback — Vite serves localhost only — but the SFU is not
    a web server for this browser: a remote peer (a Modal container) has to
    reach its SIGNALLING port, and a loopback bind makes a dev session
    LiveKit-Cloud-only. So `--bind` is honoured for the SFU alone, and defaults
    to loopback like everything else here.
    """
    # --dev needs the frontend *source* (Vite config, package.json), which
    # only exists in a git checkout. A non-editable `uv tool install`
    # resolves PROJECT_ROOT into site-packages, where the shipped wheel has
    # only frontend/dist — no package.json — so `npm run dev` would fail with
    # a confusing path/npm error. Fail fast with a pointer to the fix instead.
    if not FRONTEND_PACKAGE_JSON.is_file():
        logger.error("❌ Dev mode needs the git checkout — %s not found.", FRONTEND_PACKAGE_JSON)
        logger.error(
            "   You're likely running a `uv tool install` copy (frontend source "
            "isn't shipped in the wheel). Clone the repo and run `makermodslab --dev` "
            "from there — see README.md."
        )
        sys.exit(1)

    _ensure_port_available("Frontend", FRONTEND_DEV_PORT)
    _ensure_port_available("Backend", BACKEND_PORT)

    logger.info("📦 Installing frontend deps...")
    subprocess.run(["npm", "install"], check=True, cwd=FRONTEND_PATH)

    logger.info("🎨 Starting Vite dev server (port %d)...", FRONTEND_DEV_PORT)
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_PATH,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    if not _wait_for_port(FRONTEND_DEV_PORT):
        logger.error("❌ Frontend never came up")
        _terminate_tree(frontend_process.pid)
        sys.exit(1)

    # Before the backend spawn: _start_sfu exports the app-side env the
    # reload supervisor copies into every worker it starts.
    children: list[tuple[str, subprocess.Popen]] = [("frontend", frontend_process)]
    if sfu_bin:
        try:
            children.insert(0, ("sfu", _start_sfu(sfu_bin, sfu_host, sfu_external_ip)))
        except SystemExit:
            _terminate_tree(frontend_process.pid)
            raise

    logger.info("🚀 Starting backend (port %d) with --reload...", BACKEND_PORT)
    backend_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "makermodslab.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(BACKEND_PORT),
            "--reload",
        ],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        start_new_session=True,
    )

    children.insert(0, ("backend", backend_process))

    if not _wait_for_port(BACKEND_PORT, timeout=15):
        logger.error("❌ Backend never came up")
        for _name, p in children:
            _terminate_tree(p.pid)
        sys.exit(1)

    logger.info("🌐 Opening browser...")
    webbrowser.open(f"http://localhost:{FRONTEND_DEV_PORT}/")

    logger.info("✅ Dev mode running — Ctrl+C to stop")
    logger.info("   Frontend: http://localhost:%d", FRONTEND_DEV_PORT)
    logger.info("   Backend:  http://localhost:%d", BACKEND_PORT)
    if sfu_bin:
        logger.info("   SFU:      ws://%s:%d", sfu.public_host(sfu_host), sfu.SFU_HTTP_PORT)

    def shutdown(signum, frame):
        logger.info("🛑 Shutting down...")
        # Walk each child's whole process tree (npm -> node -> vite, uvicorn
        # --reload -> reloader -> worker) so no grandchild outlives Ctrl+C and
        # keeps holding :8000/:8080 (or :7880).
        for name, p in children:
            _terminate_tree(p.pid)
            logger.info(f"  ✅ {name} stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        time.sleep(2)
        for name, p in children:
            if p.poll() is not None:
                logger.error("❌ %s died", name.capitalize())
                shutdown(None, None)


def main():
    parser = argparse.ArgumentParser(prog="makermodslab", description="Run MakerMods Lab")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode: Vite HMR + uvicorn --reload (requires Node.js)",
    )
    parser.add_argument(
        "--lan",
        action="store_true",
        help=(
            "Headless station mode: bind 0.0.0.0 (serve other machines), don't open a browser. "
            "For narrower exposure, bind one address/interface with --bind instead"
        ),
    )
    parser.add_argument(
        "--bind",
        metavar="ADDRESS_OR_INTERFACE",
        help=(
            "Bind this literal IP, or a network interface name (resolved to its first IPv4) — "
            "e.g. a tailnet-only interface. Overrides the host --lan/default would pick"
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1: every Hub call fails fast (all hardware flows work offline)",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Don't serve the built frontend: pure API node (same binary, headless role)",
    )
    parser.add_argument(
        "--discover-tailscale",
        action="store_true",
        help=(
            "Discover peer nodes over Tailscale (needs the tailscale CLI); "
            "candidates are verified against their /api/v1/health before being trusted"
        ),
    )
    parser.add_argument(
        "--sfu",
        action="store_true",
        help=(
            "Also run a LiveKit SFU (`livekit-server` from PATH) bound like the API, for remote "
            "teleoperation/inference peers; /api/v1/sfu/token then signs room tokens. Exits with "
            "install instructions if the binary is missing"
        ),
    )
    parser.add_argument(
        "--sfu-external-ip",
        action="store_true",
        help=(
            "With --sfu: let the SFU STUN-discover this machine's public IP and advertise it as "
            "an ICE candidate, instead of pinning the bound address. Needed for a peer with no "
            "route to the bound address (a Modal container reaching the signalling URL over the "
            "tailnet but hole-punching for media); needs UDP 7882 reachable here"
        ),
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop a running MakerMods Lab and free its ports (:8000/:8080/:7880), then exit.",
    )
    args = parser.parse_args()

    if args.stop:
        _run_stop()
        return

    # Resolve --bind before anything starts, so a typo'd interface name is a
    # clean one-line failure instead of a half-started server.
    bind_host: str | None = None
    if args.bind:
        try:
            bind_host = _resolve_bind_host(args.bind)
        except ValueError as exc:
            logger.error("❌ --bind %s: %s", args.bind, exc)
            sys.exit(1)
        if args.lan:
            logger.info("--bind wins over --lan: binding %s instead of 0.0.0.0", bind_host)  # noqa: S104

    # Same fail-fast rule as --bind: a missing livekit-server is a one-line
    # exit before anything starts, never a half-started stack.
    sfu_bin = _require_livekit_server() if args.sfu else None
    if args.sfu_external_ip and not args.sfu:
        logger.warning("--sfu-external-ip does nothing without --sfu")

    _ensure_path_symlinks()

    if args.offline:
        # Must land in the environment before makermodslab.server (and its
        # huggingface_hub import) loads — uvicorn imports the app lazily, so
        # setting it here covers both prod and the dev subprocess (env copy).
        os.environ["HF_HUB_OFFLINE"] = "1"
        logger.info(
            "HF_HUB_OFFLINE=1 (--offline): Hub features disabled (login/whoami/"
            "dataset push will fail fast), hardware flows unaffected."
        )

    if args.no_ui:
        # Like HF_HUB_OFFLINE above: must be in the environment before uvicorn
        # imports makermodslab.server, where ui_enabled() gates the SPA mount.
        os.environ["MAKERMODSLAB_NO_UI"] = "1"

    if args.discover_tailscale:
        # Same import-order rule: nodes.register_sources_from_env reads this
        # when makermodslab.server first imports the registry module.
        os.environ["MAKERMODSLAB_DISCOVER_TAILSCALE"] = "1"

    if args.dev:
        if args.lan:
            logger.warning("--lan is ignored in --dev mode (Vite serves localhost only)")
        if args.bind:
            logger.warning("--bind applies to the SFU only in --dev mode (Vite and uvicorn serve localhost)")
        _run_dev(
            sfu_bin=sfu_bin,
            sfu_external_ip=args.sfu_external_ip,
            # `bind_host` is None unless --bind was given and resolved.
            sfu_host=bind_host or "127.0.0.1",
        )
    else:
        _run_prod(
            lan=args.lan,
            no_ui=args.no_ui,
            host=bind_host,
            sfu_bin=sfu_bin,
            sfu_external_ip=args.sfu_external_ip,
        )


def station():
    """Entry point for headless robot stations: `makermodslab --lan --offline`.

    Installed as `makermodslab-station` (see pyproject.toml) so the posture is a
    first-class command. Extra CLI args still pass through.
    """
    sys.argv = [sys.argv[0], "--lan", "--offline", *sys.argv[1:]]
    main()


if __name__ == "__main__":
    main()
