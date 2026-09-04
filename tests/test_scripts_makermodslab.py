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
"""Tests for makermodslab.scripts.makermodslab — covers `_wait_for_port`,
`_ensure_path_symlinks`, `_resolve_bind_host`, and `main`'s arg handling (with
`_run_prod` stubbed out). The launcher's `_run_prod` / `_run_dev` bodies are
CLI/process glue (they call uvicorn.run, spawn npm, install SIGINT handlers)
and have no unit-testable seam without rewriting them; they are left to manual
smoke testing."""

from __future__ import annotations

import logging
import socket
import threading
import types

import pytest


def _bind_listener() -> tuple[socket.socket, int]:
    """Bind a real TCP listener on an ephemeral localhost port. Returns the
    socket (caller must close) and its actual port number."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    return server, server.getsockname()[1]


def test_wait_for_port_returns_true_when_port_is_open() -> None:
    from makermodslab.scripts.makermodslab import _wait_for_port

    server, port = _bind_listener()
    try:
        assert _wait_for_port(port, timeout=2) is True
    finally:
        server.close()


def test_wait_for_port_returns_false_when_port_never_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch sleep so we don't actually block for `timeout` seconds — the
    function's whole loop body is fast otherwise."""
    from makermodslab.scripts.makermodslab import _wait_for_port

    monkeypatch.setattr("makermodslab.scripts.makermodslab.time.sleep", lambda _s: None)
    # Pick an ephemeral port from the OS, then close it so it's not bound.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    assert _wait_for_port(port, timeout=2) is False


def test_wait_for_port_returns_true_immediately_for_already_open_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check that the success path doesn't sleep at all — guards
    against accidentally adding a leading delay."""
    from makermodslab.scripts.makermodslab import _wait_for_port

    sleep_calls = []
    monkeypatch.setattr("makermodslab.scripts.makermodslab.time.sleep", lambda s: sleep_calls.append(s))

    server, port = _bind_listener()
    # Drain any incoming connection so the listener stays healthy.
    accept_thread = threading.Thread(target=lambda: server.accept() if server else None, daemon=True)
    accept_thread.start()

    try:
        assert _wait_for_port(port, timeout=5) is True
        assert sleep_calls == []
    finally:
        server.close()


def test_wait_for_port_probes_the_host_it_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: `--bind <iface>` pins livekit's `bind_addresses` to that one
    interface, so probing loopback timed out on a healthy SFU and the launcher
    killed it. The probe must go to the address the child actually bound."""
    from makermodslab.scripts.makermodslab import _wait_for_port

    probed: list[tuple[str, int]] = []

    class _RecordingSocket:
        def settimeout(self, _t): ...

        def connect_ex(self, address):
            probed.append(address)
            return 0

        def close(self): ...

    monkeypatch.setattr(
        "makermodslab.scripts.makermodslab.socket.socket",
        lambda *_a, **_k: _RecordingSocket(),
    )

    assert _wait_for_port(7880, timeout=1, host="100.64.0.1") is True
    assert probed == [("100.64.0.1", 7880)]


def test_wait_for_port_defaults_to_loopback() -> None:
    """The default host keeps the loopback behaviour every other call site
    (dev-mode Vite and backend, both hardcoded to 127.0.0.1) relies on."""
    import inspect

    from makermodslab.scripts.makermodslab import _wait_for_port

    assert inspect.signature(_wait_for_port).parameters["host"].default == "localhost"


def _fake_entry_points(tmp_path):
    """A fake venv bin dir containing all three entry-point scripts."""
    from makermodslab.scripts.makermodslab import ENTRY_POINT_NAMES

    source_dir = tmp_path / "venv-bin"
    source_dir.mkdir()
    for name in ENTRY_POINT_NAMES:
        (source_dir / name).write_text("#!/bin/sh\n")
    return source_dir


def test_ensure_path_symlinks_links_all_entry_points(tmp_path) -> None:
    from makermodslab.scripts.makermodslab import ENTRY_POINT_NAMES, _ensure_path_symlinks

    source_dir = _fake_entry_points(tmp_path)
    bin_dir = tmp_path / "local-bin"  # deliberately absent: must be created

    _ensure_path_symlinks(source_dir=source_dir, bin_dir=bin_dir)

    for name in ENTRY_POINT_NAMES:
        link = bin_dir / name
        assert link.is_symlink()
        assert link.resolve() == (source_dir / name).resolve()


def test_ensure_path_symlinks_is_idempotent(tmp_path) -> None:
    from makermodslab.scripts.makermodslab import ENTRY_POINT_NAMES, _ensure_path_symlinks

    source_dir = _fake_entry_points(tmp_path)
    bin_dir = tmp_path / "local-bin"

    _ensure_path_symlinks(source_dir=source_dir, bin_dir=bin_dir)
    before = {name: (bin_dir / name).lstat().st_ino for name in ENTRY_POINT_NAMES}
    _ensure_path_symlinks(source_dir=source_dir, bin_dir=bin_dir)

    # Correct links are left alone, not unlinked and re-created.
    assert {name: (bin_dir / name).lstat().st_ino for name in ENTRY_POINT_NAMES} == before


def test_ensure_path_symlinks_repoints_stale_link(tmp_path) -> None:
    from makermodslab.scripts.makermodslab import _ensure_path_symlinks

    source_dir = _fake_entry_points(tmp_path)
    bin_dir = tmp_path / "local-bin"
    bin_dir.mkdir()
    old_venv = tmp_path / "old-venv-bin"
    old_venv.mkdir()
    (old_venv / "makermodslab").write_text("#!/bin/sh\n")
    (bin_dir / "makermodslab").symlink_to(old_venv / "makermodslab")

    _ensure_path_symlinks(source_dir=source_dir, bin_dir=bin_dir)

    assert (bin_dir / "makermodslab").resolve() == (source_dir / "makermodslab").resolve()


def test_ensure_path_symlinks_never_clobbers_regular_files(tmp_path) -> None:
    from makermodslab.scripts.makermodslab import _ensure_path_symlinks

    source_dir = _fake_entry_points(tmp_path)
    bin_dir = tmp_path / "local-bin"
    bin_dir.mkdir()
    foreign = bin_dir / "makermodslab"
    foreign.write_text("someone else's script\n")

    _ensure_path_symlinks(source_dir=source_dir, bin_dir=bin_dir)

    assert not foreign.is_symlink()
    assert foreign.read_text() == "someone else's script\n"
    # The other entry point is still linked.
    assert (bin_dir / "makermodslab-station").is_symlink()


def test_ensure_path_symlinks_skips_missing_entry_points(tmp_path) -> None:
    from makermodslab.scripts.makermodslab import _ensure_path_symlinks

    source_dir = tmp_path / "venv-bin"
    source_dir.mkdir()
    (source_dir / "makermodslab").write_text("#!/bin/sh\n")  # only one installed
    bin_dir = tmp_path / "local-bin"

    _ensure_path_symlinks(source_dir=source_dir, bin_dir=bin_dir)

    assert (bin_dir / "makermodslab").is_symlink()
    assert not (bin_dir / "makermodslab-station").exists()


def test_ensure_path_symlinks_env_opt_out(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from makermodslab.scripts.makermodslab import _ensure_path_symlinks

    monkeypatch.setenv("MAKERMODSLAB_NO_PATH_LINK", "1")
    source_dir = _fake_entry_points(tmp_path)
    bin_dir = tmp_path / "local-bin"

    _ensure_path_symlinks(source_dir=source_dir, bin_dir=bin_dir)

    assert not bin_dir.exists()


def _fake_uv_tool_link(tmp_path, name: str):
    """Simulate a `uv tool install` executable: a symlink in the fake bin dir
    that resolves into a fake uv tools dir (mirrors what uv creates —
    ~/.local/bin/<exe> -> ~/.local/share/uv/tools/<tool>/bin/<exe>). Returns
    the (bin_dir, uv_tools_dir) pair."""
    uv_tools_dir = tmp_path / "uv-tools"
    tool_bin = uv_tools_dir / name / "bin"
    tool_bin.mkdir(parents=True)
    (tool_bin / name).write_text("#!/bin/sh\n")  # the uv-managed executable

    bin_dir = tmp_path / "local-bin"
    bin_dir.mkdir()
    (bin_dir / name).symlink_to(tool_bin / name)
    return bin_dir, uv_tools_dir


def test_is_uv_tool_link_recognizes_uv_managed_symlink(tmp_path) -> None:
    from makermodslab.scripts.makermodslab import _is_uv_tool_link

    bin_dir, uv_tools_dir = _fake_uv_tool_link(tmp_path, "makermodslab")

    assert _is_uv_tool_link(bin_dir / "makermodslab", uv_tools_dir) is True


def test_is_uv_tool_link_false_for_venv_symlink(tmp_path) -> None:
    from makermodslab.scripts.makermodslab import _is_uv_tool_link

    source_dir = _fake_entry_points(tmp_path)
    bin_dir = tmp_path / "local-bin"
    bin_dir.mkdir()
    (bin_dir / "makermodslab").symlink_to(source_dir / "makermodslab")
    uv_tools_dir = tmp_path / "uv-tools"  # nonexistent / unrelated

    assert _is_uv_tool_link(bin_dir / "makermodslab", uv_tools_dir) is False


def test_is_uv_tool_link_false_for_regular_file(tmp_path) -> None:
    from makermodslab.scripts.makermodslab import _is_uv_tool_link

    bin_dir = tmp_path / "local-bin"
    bin_dir.mkdir()
    (bin_dir / "makermodslab").write_text("not a symlink\n")
    uv_tools_dir = tmp_path / "uv-tools"

    assert _is_uv_tool_link(bin_dir / "makermodslab", uv_tools_dir) is False


def test_station_injects_lan_and_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """`station()` prepends `--lan --offline` and defers to `main` — without
    starting a server (main is stubbed). Guards the systemd unit's posture."""
    from makermodslab.scripts import makermodslab as launcher

    captured: dict[str, list[str]] = {}

    def fake_main() -> None:
        captured["argv"] = list(launcher.sys.argv)

    monkeypatch.setattr(launcher, "main", fake_main)
    monkeypatch.setattr(launcher.sys, "argv", ["makermodslab-station"])

    launcher.station()

    assert captured["argv"] == ["makermodslab-station", "--lan", "--offline"]


def test_station_passes_extra_args_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ad-hoc flags after the injected posture still reach `main`."""
    from makermodslab.scripts import makermodslab as launcher

    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(launcher, "main", lambda: captured.setdefault("argv", list(launcher.sys.argv)))
    monkeypatch.setattr(launcher.sys, "argv", ["makermodslab-station", "--dev"])

    launcher.station()

    assert captured["argv"] == ["makermodslab-station", "--lan", "--offline", "--dev"]


def test_discover_tailscale_flag_sets_env_before_server_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--discover-tailscale` follows the MAKERMODSLAB_NO_UI precedent: the env
    var must be in place before uvicorn imports makermodslab.server (where
    nodes.register_sources_from_env reads it). OFF unless the flag is given."""
    import os

    from makermodslab.scripts import makermodslab as launcher

    monkeypatch.setenv("MAKERMODSLAB_DISCOVER_TAILSCALE", "0")  # restored by monkeypatch
    monkeypatch.delenv("MAKERMODSLAB_DISCOVER_TAILSCALE")
    monkeypatch.setattr(launcher, "_ensure_path_symlinks", lambda: None)
    seen_at_run: dict[str, str | None] = {}

    def fake_run_prod(**_kwargs) -> None:
        seen_at_run["env"] = os.environ.get("MAKERMODSLAB_DISCOVER_TAILSCALE")

    monkeypatch.setattr(launcher, "_run_prod", fake_run_prod)

    monkeypatch.setattr(launcher.sys, "argv", ["makermodslab"])
    launcher.main()
    assert seen_at_run["env"] is None  # off by default

    monkeypatch.setattr(launcher.sys, "argv", ["makermodslab", "--discover-tailscale"])
    launcher.main()
    assert seen_at_run["env"] == "1"


def test_entry_points_target_correct_functions() -> None:
    """`makermodslab` -> `main` (friendly default), `makermodslab-station` ->
    `station` (headless posture). The old `lelab*` / `makerlabs` / `makerlab*`
    names are gone.

    Reads the declared console_scripts so we never invoke the entry point
    (which would start a server). NOTE: this reflects the *installed*
    metadata, so after renaming in pyproject.toml you must `pip install -e .`
    for it to pass.
    """
    from importlib.metadata import entry_points

    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}

    assert scripts["makermodslab"] == "makermodslab.scripts.makermodslab:main"
    assert scripts["makermodslab-station"] == "makermodslab.scripts.makermodslab:station"
    assert "lelab" not in scripts
    assert "lelab-station" not in scripts
    assert "makerlabs" not in scripts
    assert "makerlab" not in scripts
    assert "makerlab-station" not in scripts


def test_ensure_path_symlinks_leaves_uv_tool_entry_untouched(tmp_path) -> None:
    """A name owned by `uv tool install` must be left exactly as-is — no
    clobber, no repoint — so the two install flavors don't fight."""
    from makermodslab.scripts.makermodslab import _ensure_path_symlinks

    source_dir = _fake_entry_points(tmp_path)
    bin_dir, uv_tools_dir = _fake_uv_tool_link(tmp_path, "makermodslab")
    uv_target_before = (bin_dir / "makermodslab").resolve()

    _ensure_path_symlinks(source_dir=source_dir, bin_dir=bin_dir, uv_tools_dir=uv_tools_dir)

    # MakerMods Lab still points at the uv tool, NOT the venv.
    assert (bin_dir / "makermodslab").resolve() == uv_target_before
    assert (bin_dir / "makermodslab").resolve() != (source_dir / "makermodslab").resolve()
    # The other name, not uv-owned, is linked to the venv as usual.
    assert (bin_dir / "makermodslab-station").resolve() == (source_dir / "makermodslab-station").resolve()


# --- --bind: literal address or interface name -------------------------------


def _fake_if_addrs(monkeypatch: pytest.MonkeyPatch, addrs: dict) -> None:
    import makermodslab.scripts.makermodslab as launcher

    monkeypatch.setattr(launcher.psutil, "net_if_addrs", lambda: addrs)


def _snic(family: int, address: str):
    """A psutil snicaddr stand-in (only family/address are read)."""
    return types.SimpleNamespace(family=family, address=address)


def test_resolve_bind_host_passes_literal_ip_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """A literal IP never consults the interface table."""
    import makermodslab.scripts.makermodslab as launcher

    monkeypatch.setattr(
        launcher.psutil, "net_if_addrs", lambda: pytest.fail("literal IPs must not hit psutil")
    )
    assert launcher._resolve_bind_host("192.168.1.10") == "192.168.1.10"
    assert launcher._resolve_bind_host("100.64.0.7") == "100.64.0.7"
    assert launcher._resolve_bind_host("::1") == "::1"


def test_resolve_bind_host_resolves_interface_to_first_ipv4(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.scripts.makermodslab as launcher

    _fake_if_addrs(
        monkeypatch,
        {
            "lo0": [_snic(socket.AF_INET, "127.0.0.1")],
            "tailscale0": [
                _snic(socket.AF_INET6, "fd7a:115c:a1e0::7"),  # v6 first, like real tables
                _snic(socket.AF_INET, "100.64.0.7"),
                _snic(socket.AF_INET, "100.64.0.8"),  # aliases: first IPv4 wins
            ],
        },
    )
    assert launcher._resolve_bind_host("tailscale0") == "100.64.0.7"


def test_resolve_bind_host_unknown_interface_fails_with_the_available_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import makermodslab.scripts.makermodslab as launcher

    _fake_if_addrs(monkeypatch, {"lo0": [_snic(socket.AF_INET, "127.0.0.1")]})
    with pytest.raises(ValueError, match="lo0"):
        launcher._resolve_bind_host("tailscale0")


def test_resolve_bind_host_interface_without_ipv4_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.scripts.makermodslab as launcher

    _fake_if_addrs(monkeypatch, {"utun3": [_snic(socket.AF_INET6, "fd7a:115c:a1e0::7")]})
    with pytest.raises(ValueError, match="no IPv4"):
        launcher._resolve_bind_host("utun3")


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> dict:
    """Drive main() with `argv`, capturing the _run_prod call instead of
    starting a server."""
    import makermodslab.scripts.makermodslab as launcher

    captured: dict = {}
    monkeypatch.setattr(launcher, "_ensure_path_symlinks", lambda: None)
    monkeypatch.setattr(launcher, "_run_prod", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(launcher.sys, "argv", ["makermodslab", *argv])
    launcher.main()
    return captured


def test_main_passes_resolved_bind_host_to_run_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_main(monkeypatch, ["--bind", "100.64.0.7"])["host"] == "100.64.0.7"


def test_main_without_bind_leaves_host_to_lan_or_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _run_main(monkeypatch, ["--lan"])
    assert captured["host"] is None
    assert captured["lan"] is True


def test_bind_wins_over_lan_and_says_so(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        captured = _run_main(monkeypatch, ["--lan", "--bind", "100.64.0.7"])
    assert captured["host"] == "100.64.0.7"
    assert "--bind" in caplog.text
    assert "--lan" in caplog.text


def test_bad_bind_fails_fast_before_anything_starts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import makermodslab.scripts.makermodslab as launcher

    _fake_if_addrs(monkeypatch, {"lo0": [_snic(socket.AF_INET, "127.0.0.1")]})
    monkeypatch.setattr(launcher, "_ensure_path_symlinks", lambda: pytest.fail("must fail before this"))
    monkeypatch.setattr(launcher, "_run_prod", lambda **_kw: pytest.fail("must not start"))
    monkeypatch.setattr(launcher.sys, "argv", ["makermodslab", "--bind", "no-such-if0"])

    with pytest.raises(SystemExit), caplog.at_level(logging.ERROR):
        launcher.main()
    assert "no-such-if0" in caplog.text


def test_bind_is_ignored_in_dev_mode_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import makermodslab.scripts.makermodslab as launcher

    monkeypatch.setattr(launcher, "_ensure_path_symlinks", lambda: None)
    monkeypatch.setattr(launcher, "_run_dev", lambda **_kwargs: None)
    monkeypatch.setattr(launcher.sys, "argv", ["makermodslab", "--dev", "--bind", "100.64.0.7"])

    with caplog.at_level(logging.WARNING):
        launcher.main()
    assert "--bind is ignored in --dev mode" in caplog.text


# --- Shutdown reliability: --stop, port preflight, process-tree teardown -----


class _FakeConn:
    """Minimal stand-in for a psutil connection tuple."""

    def __init__(self, port: int, status: str) -> None:
        self.laddr = types.SimpleNamespace(port=port)
        self.status = status


class _FakeProc:
    """A psutil.Process stand-in for _find_makermodslab_pids / _identity_reason.

    `process_iter` hands these back with `.info` populated; `.cwd()` and
    `.net_connections()` model the two other lookups the launcher performs.
    Set cwd=None to simulate a process whose cwd we can't read.
    """

    def __init__(
        self,
        pid: int,
        cmdline: list[str],
        name: str = "python",
        cwd=None,
        listening: tuple[int, ...] = (),
    ) -> None:
        self.pid = pid
        self.info = {"pid": pid, "cmdline": cmdline, "name": name}
        self._cwd = cwd
        self._listening = listening

    def cwd(self):
        import makermodslab.scripts.makermodslab as launcher

        if self._cwd is None:
            raise launcher.psutil.NoSuchProcess(self.pid)
        return self._cwd

    def net_connections(self, kind: str = "inet"):
        import makermodslab.scripts.makermodslab as launcher

        return [_FakeConn(port, launcher.psutil.CONN_LISTEN) for port in self._listening]


def test_stop_kills_identity_and_refuses_port_stranger(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The sharp-edge fix: `--stop` terminates the process we recognise as ours
    (cmdline runs `makermodslab.server`) but must NOT kill an unrelated stranger that
    merely happens to be listening on :8000 — it only warns about it."""
    import makermodslab.scripts.makermodslab as launcher

    ours = _FakeProc(100, ["python", "-m", "uvicorn", "makermodslab.server:app", "--reload"])
    stranger = _FakeProc(200, ["node", "some-other-server.js"], name="node", listening=(8000,))

    monkeypatch.setattr(launcher.psutil, "process_iter", lambda attrs=None: [ours, stranger])
    monkeypatch.setattr(launcher.os, "getpid", lambda: 999)
    terminated: list[int] = []
    monkeypatch.setattr(launcher, "_terminate_tree", lambda pid, timeout=5: terminated.append(pid))

    with caplog.at_level(logging.INFO):
        launcher._run_stop()

    # Only the identity-matched pid is terminated; the stranger is spared.
    assert terminated == [100]
    assert "held by pid 200" in caplog.text
    assert "(node)" in caplog.text
    assert "not a MakerMods Lab process" in caplog.text


def test_stop_kills_orphaned_reload_worker_in_this_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spawn worker whose cwd is THIS project is a kill target; the same
    worker in another directory is not ours and is left alone."""
    import makermodslab.scripts.makermodslab as launcher

    ours = _FakeProc(
        300,
        ["python", "-c", "from multiprocessing.spawn import spawn_main"],
        cwd=str(launcher.PROJECT_ROOT),
    )
    other = _FakeProc(
        400,
        ["python", "-c", "from multiprocessing.spawn import spawn_main"],
        cwd="/somewhere/else",
    )

    monkeypatch.setattr(launcher.psutil, "process_iter", lambda attrs=None: [ours, other])
    monkeypatch.setattr(launcher.os, "getpid", lambda: 999)
    terminated: list[int] = []
    monkeypatch.setattr(launcher, "_terminate_tree", lambda pid, timeout=5: terminated.append(pid))

    launcher._run_stop()

    assert terminated == [300]


def test_stop_kills_the_prod_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the long-standing sharp edge: prod mode runs uvicorn
    IN-PROCESS, so its argv is `.../python3 .../bin/makermodslab --lan ...`
    with no `makermodslab.server` substring anywhere — and `--stop` reported
    its own server as a port stranger and refused to touch it (observed on two
    real stations). The executed launcher script is now identity signal 3."""
    import makermodslab.scripts.makermodslab as launcher

    shebang_form = _FakeProc(
        500,
        ["/Users/x/MakerLab/.venv/bin/python3", "./.venv/bin/makermodslab", "--lan", "--portal"],
        listening=(8000,),
    )
    direct_form = _FakeProc(501, ["/Users/x/.local/bin/makermodslab-station", "--offline"])

    monkeypatch.setattr(launcher.psutil, "process_iter", lambda attrs=None: [shebang_form, direct_form])
    monkeypatch.setattr(launcher.os, "getpid", lambda: 999)
    terminated: list[int] = []
    monkeypatch.setattr(launcher, "_terminate_tree", lambda pid, timeout=5: terminated.append(pid))

    launcher._run_stop()

    assert sorted(terminated) == [500, 501]


def test_prod_launcher_signal_never_matches_a_mere_mention(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The signal matches the EXECUTED script only. A process that just names
    the launcher in an argument (an editor, a log tail) or in a longer
    basename must stay a stranger — `--stop` kills identity matches whether or
    not they hold a port, so a loose match here is a loose SIGTERM."""
    import makermodslab.scripts.makermodslab as launcher

    editor = _FakeProc(600, ["vim", "makermodslab"], name="vim")
    log_tail = _FakeProc(601, ["tail", "-f", "/Users/x/makermodslab.log"], name="tail")
    lookalike = _FakeProc(602, ["python3", "makermodslab_helper.py"], name="python")
    stranger_on_port = _FakeProc(603, ["node", "server.js"], name="node", listening=(8000,))

    monkeypatch.setattr(
        launcher.psutil,
        "process_iter",
        lambda attrs=None: [editor, log_tail, lookalike, stranger_on_port],
    )
    monkeypatch.setattr(launcher.os, "getpid", lambda: 999)
    terminated: list[int] = []
    monkeypatch.setattr(launcher, "_terminate_tree", lambda pid, timeout=5: terminated.append(pid))

    with caplog.at_level(logging.INFO):
        launcher._run_stop()

    assert terminated == []


def test_stop_reports_nothing_when_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import makermodslab.scripts.makermodslab as launcher

    monkeypatch.setattr(launcher.psutil, "process_iter", lambda attrs=None: [])
    monkeypatch.setattr(launcher.os, "getpid", lambda: 999)
    monkeypatch.setattr(launcher, "_terminate_tree", lambda *_a, **_k: pytest.fail("should not kill"))

    with caplog.at_level(logging.INFO):
        launcher._run_stop()

    assert "Nothing to stop" in caplog.text


def test_ensure_port_available_message_mentions_makermodslab_stop(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Port preflight fails fast and points the user at the actual command."""
    import makermodslab.scripts.makermodslab as launcher

    monkeypatch.setattr(launcher, "_is_port_open", lambda _port, _host="127.0.0.1": True)

    with pytest.raises(SystemExit), caplog.at_level(logging.INFO):
        launcher._ensure_port_available("Backend", 8000)

    assert "already in use" in caplog.text
    assert "makermodslab --stop" in caplog.text


def test_ensure_port_available_passes_when_free(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.scripts.makermodslab as launcher

    monkeypatch.setattr(launcher, "_is_port_open", lambda _port, _host="127.0.0.1": False)
    # Returns without raising.
    launcher._ensure_port_available("Backend", 8000)


def test_terminate_tree_terminates_parent_and_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tree teardown terminates the parent AND every descendant (so npm/vite
    and uvicorn reload workers can't outlive the parent and hold the ports)."""
    import makermodslab.scripts.makermodslab as launcher

    terminated: list[int] = []
    killed: list[int] = []

    class _TreeProc:
        def __init__(self, pid: int, kids: list[int] | None = None) -> None:
            self.pid = pid
            self._kids = kids or []

        def children(self, recursive: bool = False) -> list:
            return [_TreeProc(k) for k in self._kids]

        def terminate(self) -> None:
            terminated.append(self.pid)

        def kill(self) -> None:  # pragma: no cover - alive list is empty here
            killed.append(self.pid)

    monkeypatch.setattr(launcher.psutil, "Process", lambda pid: _TreeProc(pid, kids=[2, 3]))
    monkeypatch.setattr(launcher.psutil, "wait_procs", lambda procs, timeout=None: (procs, []))

    launcher._terminate_tree(1)

    # Parent (1) plus both children (2, 3) all get terminate(); nothing killed.
    assert sorted(terminated) == [1, 2, 3]
    assert killed == []


# --- --sfu: fail-fast binary check, handoff to the run functions, --stop identity


def test_sfu_flag_without_binary_exits_before_anything_starts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No livekit-server on PATH: a one-line exit with the per-OS install hint,
    BEFORE the PATH self-link or any server start — never a half-started
    stack."""
    import makermodslab.scripts.makermodslab as launcher

    calls: list[str] = []
    monkeypatch.setattr(launcher.sfu, "find_livekit_server", lambda *a, **k: None)
    monkeypatch.setattr(launcher, "_ensure_path_symlinks", lambda: calls.append("symlinks"))
    monkeypatch.setattr(launcher, "_run_prod", lambda **kwargs: calls.append("run_prod"))
    monkeypatch.setattr(launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(launcher.sys, "argv", ["makermodslab", "--sfu"])

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc:
        launcher.main()

    assert exc.value.code == 1
    assert calls == []
    assert "livekit-server" in caplog.text
    assert "brew install livekit" in caplog.text


def test_sfu_flag_hands_the_binary_to_run_prod_and_run_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    import makermodslab.scripts.makermodslab as launcher

    monkeypatch.setattr(
        launcher.sfu, "find_livekit_server", lambda *a, **k: "/opt/homebrew/bin/livekit-server"
    )
    assert _run_main(monkeypatch, ["--sfu"])["sfu_bin"] == "/opt/homebrew/bin/livekit-server"
    assert _run_main(monkeypatch, [])["sfu_bin"] is None

    captured: dict = {}
    monkeypatch.setattr(launcher, "_ensure_path_symlinks", lambda: None)
    monkeypatch.setattr(launcher, "_run_dev", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(launcher.sys, "argv", ["makermodslab", "--dev", "--sfu"])
    launcher.main()
    assert captured["sfu_bin"] == "/opt/homebrew/bin/livekit-server"


def test_identity_reason_recognises_our_sfu_child_but_not_a_foreign_livekit() -> None:
    """`--stop` must reap the livekit-server WE spawned (pointed at our
    generated config) and leave a user's own livekit-server alone."""
    import makermodslab.scripts.makermodslab as launcher

    ours = _FakeProc(300, ["/opt/homebrew/bin/livekit-server", "--config", launcher.LIVEKIT_CONFIG_FILE])
    foreign = _FakeProc(301, ["livekit-server", "--config", "/etc/livekit/livekit.yaml"])
    assert launcher._identity_reason(" ".join(ours.info["cmdline"]), ours) == "livekit-server (--sfu)"
    assert launcher._identity_reason(" ".join(foreign.info["cmdline"]), foreign) is None


def test_host_flag_requires_the_sfu_and_exports_the_robot(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    """`--host ROBOT` is station mode: it needs the SFU (--sfu or an external
    MAKERMODSLAB_SFU_URL) and hands the robot name to the app through the
    environment before the server import."""
    import os

    import makermodslab.scripts.makermodslab as launcher

    monkeypatch.delenv("MAKERMODSLAB_HOST_ROBOT", raising=False)
    monkeypatch.delenv(launcher.sfu.ENV_URL, raising=False)
    monkeypatch.setattr(launcher, "_ensure_path_symlinks", lambda: None)
    monkeypatch.setattr(
        launcher.sfu, "find_livekit_server", lambda *a, **k: "/opt/homebrew/bin/livekit-server"
    )
    seen: dict = {}
    monkeypatch.setattr(
        launcher,
        "_run_prod",
        lambda **kwargs: seen.update(kwargs, robot=os.environ.get("MAKERMODSLAB_HOST_ROBOT")),
    )

    monkeypatch.setattr(launcher.sys, "argv", ["makermodslab", "--host", "arm1"])
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit):
        launcher.main()
    assert "--sfu" in caplog.text

    monkeypatch.setattr(launcher.sys, "argv", ["makermodslab", "--sfu", "--host", "arm1"])
    launcher.main()
    assert seen["robot"] == "arm1"
    assert seen["sfu_bin"] == "/opt/homebrew/bin/livekit-server"
    assert os.environ.get("MAKERMODSLAB_STATION") == "1"

    # A bare --host is station mode with the robot chosen later (remembered,
    # auto-picked, or from the station's UI): the posture is set, the name empty.
    monkeypatch.delenv("MAKERMODSLAB_STATION", raising=False)
    monkeypatch.setattr(launcher.sys, "argv", ["makermodslab", "--sfu", "--host"])
    launcher.main()
    assert seen["robot"] == ""
    assert os.environ.get("MAKERMODSLAB_STATION") == "1"
