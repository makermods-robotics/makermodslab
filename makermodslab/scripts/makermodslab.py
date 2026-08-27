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
"""

import argparse
import contextlib
import logging
import os
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


def _identity_reason(cmdline: str, proc: psutil.Process) -> str | None:
    """Why `proc` is recognisably one of ours, or None if it isn't.

    Two independent identity signals, deliberately narrow so `--stop` never
    touches an unrelated dev server that merely happens to hold :8000/:8080:
      1. cmdline runs `uvicorn ... makermodslab.server` (the reload supervisor / prod
         server — matches even a stale uv-tool snapshot from another venv).
      2. an orphaned reload worker (`multiprocessing.spawn` / `spawn_main`)
         whose cwd is THIS project checkout.
    """
    if "makermodslab.server" in cmdline:
        return "uvicorn (makermodslab.server)"
    if "multiprocessing.spawn" in cmdline or "spawn_main" in cmdline:
        with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            if Path(proc.cwd()) == PROJECT_ROOT:
                return "orphaned reload worker"
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
    ports = {BACKEND_PORT, FRONTEND_DEV_PORT}
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
                "Nothing to stop: no MakerMods Lab process found on :%d / :%d.",
                BACKEND_PORT,
                FRONTEND_DEV_PORT,
            )
        return
    for pid, reason in kill_targets.items():
        logger.info("🛑 Stopping pid %d (%s)...", pid, reason)
        _terminate_tree(pid)
    logger.info("✅ MakerMods Lab stopped.")


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


def _run_prod(lan: bool = False, no_ui: bool = False):
    """Serve built frontend from backend on a single port.

    `lan` binds 0.0.0.0 for headless stations serving other machines on the
    network; it also skips the open-a-local-browser step (there is no local
    browser worth opening in that deployment). `no_ui` skips serving (and
    requiring) the built frontend entirely — a pure API node.
    """
    if not no_ui and not FRONTEND_DIST.exists():
        logger.error(f"❌ Built frontend not found at {FRONTEND_DIST}")
        logger.error("   Run `npm run build` in frontend/ first, or use `makermodslab --dev`.")
        sys.exit(1)

    host = "0.0.0.0" if lan else "127.0.0.1"  # noqa: S104  # nosec B104 — binds all interfaces only behind the explicit --lan opt-in; loopback otherwise
    _ensure_port_available("Backend", BACKEND_PORT, host)
    if lan:
        logger.info("🚀 Starting MakerMods Lab on http://0.0.0.0:%d (LAN) ...", BACKEND_PORT)
    else:
        logger.info("🚀 Starting MakerMods Lab on http://localhost:%d ...", BACKEND_PORT)
        if not no_ui:
            threading.Thread(target=_open_browser_when_ready, daemon=True).start()

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

    server.run()


def _run_dev():
    """Vite dev server (HMR) + uvicorn --reload."""
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

    if not _wait_for_port(BACKEND_PORT, timeout=15):
        logger.error("❌ Backend never came up")
        for p in (backend_process, frontend_process):
            _terminate_tree(p.pid)
        sys.exit(1)

    logger.info("🌐 Opening browser...")
    webbrowser.open(f"http://localhost:{FRONTEND_DEV_PORT}/")

    logger.info("✅ Dev mode running — Ctrl+C to stop")
    logger.info("   Frontend: http://localhost:%d", FRONTEND_DEV_PORT)
    logger.info("   Backend:  http://localhost:%d", BACKEND_PORT)

    def shutdown(signum, frame):
        logger.info("🛑 Shutting down...")
        # Walk each child's whole process tree (npm -> node -> vite, uvicorn
        # --reload -> reloader -> worker) so no grandchild outlives Ctrl+C and
        # keeps holding :8000/:8080.
        for name, p in [("backend", backend_process), ("frontend", frontend_process)]:
            _terminate_tree(p.pid)
            logger.info(f"  ✅ {name} stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        time.sleep(2)
        if backend_process.poll() is not None:
            logger.error("❌ Backend died")
            shutdown(None, None)
        if frontend_process.poll() is not None:
            logger.error("❌ Frontend died")
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
        help="Headless station mode: bind 0.0.0.0 (serve other machines), don't open a browser",
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
        "--stop",
        action="store_true",
        help="Stop a running MakerMods Lab and free its ports (:8000/:8080), then exit.",
    )
    args = parser.parse_args()

    if args.stop:
        _run_stop()
        return

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
        _run_dev()
    else:
        _run_prod(lan=args.lan, no_ui=args.no_ui)


def station():
    """Entry point for headless robot stations: `makermodslab --lan --offline`.

    Installed as `makermodslab-station` (see pyproject.toml) so the posture is a
    first-class command. Extra CLI args still pass through.
    """
    sys.argv = [sys.argv[0], "--lan", "--offline", *sys.argv[1:]]
    main()


if __name__ == "__main__":
    main()
