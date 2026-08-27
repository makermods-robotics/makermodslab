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
"""Tests for makermodslab.scripts.makermodslab — covers `_wait_for_port` and
`_ensure_path_symlinks`. The launcher's `_run_prod` / `_run_dev` / `main`
functions are CLI/process glue (they call uvicorn.run, spawn npm, install
SIGINT handlers) and have no unit-testable seam without rewriting them; they
are left to manual smoke testing."""

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
