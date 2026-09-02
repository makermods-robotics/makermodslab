"""Precedence of the LiveKit credential sources in `makermodslab.drtc._env`.

Pure-helper test: no LiveKit, no hardware. `_env` is deliberately importable
without the `drtc` extra (it needs only python-dotenv) so this runs in CI.

Why it exists: the source repo read `.env.local` from the SCRIPT's directory,
and the port moved the local-SFU override to the Lab's state dir. Two live
runs on 2026-09-02 failed with "connection refused" because the override was
being read from a directory the robot was not started in. These pin down which
file wins from where.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("dotenv")

from makermodslab.drtc import _env  # noqa: E402

KEYS = ("LIVEKIT_URL", "LIVEKIT_ROOM", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")


@pytest.fixture
def clean_env():
    """Start from an unset LIVEKIT_* environment, and put it back afterwards.

    `load_dotenv` writes `os.environ` directly, so monkeypatch cannot undo what
    `load_env()` sets — without this teardown the last test's credentials would
    leak into the rest of the session.
    """
    saved = {k: os.environ.get(k) for k in KEYS}
    for k in KEYS:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """Point every source at tmp_path and run from a fresh cwd there."""
    saved = tmp_path / "livekit.env"
    local = tmp_path / "livekit.local.env"
    monkeypatch.setattr(_env, "DRTC_ENV_PATH", str(saved))
    monkeypatch.setattr(_env, "DRTC_LOCAL_ENV_PATH", str(local))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return {"saved": saved, "local": local, "cwd": cwd}


def test_saved_credentials_load_when_nothing_else_exists(clean_env, paths):
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\nLIVEKIT_ROOM=r\n")

    _env.load_env()

    assert os.environ["LIVEKIT_URL"] == "wss://cloud"
    assert os.environ["LIVEKIT_ROOM"] == "r"


def test_process_environment_beats_the_saved_file(clean_env, paths, monkeypatch):
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\n")
    monkeypatch.setenv("LIVEKIT_URL", "wss://from-shell")

    _env.load_env()

    assert os.environ["LIVEKIT_URL"] == "wss://from-shell"


def test_local_sfu_override_beats_saved_and_process_environment(clean_env, paths, monkeypatch):
    """The override is 'the current transport', so it must win even over the shell."""
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\nLIVEKIT_ROOM=r\n")
    paths["local"].write_text("LIVEKIT_URL=ws://127.0.0.1:7880\n")
    monkeypatch.setenv("LIVEKIT_URL", "wss://from-shell")

    _env.load_env()

    assert os.environ["LIVEKIT_URL"] == "ws://127.0.0.1:7880"
    assert os.environ["LIVEKIT_ROOM"] == "r"  # room is NOT in the override; still from saved


def test_override_is_read_from_the_state_dir_not_the_cwd(clean_env, paths):
    """The regression the port fixes: the robot need not run from any directory."""
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\n")
    paths["local"].write_text("LIVEKIT_URL=ws://127.0.0.1:7880\n")
    elsewhere = paths["cwd"].parent / "elsewhere"
    elsewhere.mkdir()
    os.chdir(elsewhere)

    _env.load_env()

    assert os.environ["LIVEKIT_URL"] == "ws://127.0.0.1:7880"


def test_cwd_dotenv_files_still_work_as_in_the_source_repo(clean_env, paths):
    (paths["cwd"] / ".env").write_text("LIVEKIT_URL=wss://cloud\nLIVEKIT_ROOM=r\n")
    (paths["cwd"] / ".env.local").write_text("LIVEKIT_URL=ws://127.0.0.1:7880\n")

    _env.load_env()

    assert os.environ["LIVEKIT_URL"] == "ws://127.0.0.1:7880"
    assert os.environ["LIVEKIT_ROOM"] == "r"


def test_deleting_the_override_returns_to_the_saved_credentials(clean_env, paths):
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\n")
    paths["local"].write_text("LIVEKIT_URL=ws://127.0.0.1:7880\n")
    _env.load_env()
    assert os.environ["LIVEKIT_URL"] == "ws://127.0.0.1:7880"

    paths["local"].unlink()
    del os.environ["LIVEKIT_URL"]
    _env.load_env()

    assert os.environ["LIVEKIT_URL"] == "wss://cloud"


def test_required_env_names_the_missing_variable(clean_env):
    with pytest.raises(RuntimeError, match="LIVEKIT_ROOM"):
        _env.required_env("LIVEKIT_ROOM")
