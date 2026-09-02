"""Precedence of the LiveKit credential sources in `makermodslab.drtc._env`.

Pure-helper test: no LiveKit, no hardware. `_env` is deliberately importable
without the `drtc` extra (it needs only python-dotenv) so this runs in CI.

Why it exists: the source repo read `.env.local` from the SCRIPT's directory,
and the port moved the local-SFU override to the Lab's state dir. Two live
runs on 2026-09-02 failed with "connection refused" because the override was
being read from a directory the robot was not started in. These pin down which
file wins from where.

Every precedence case is parametrized over BOTH entry points — `load_env`
(mutates `os.environ`, for the CLI entrypoints) and `read_env` (resolves onto a
copy, for the long-lived server) — so this file stays the single authority and
the two can never drift. The cases that are specifically ABOUT the difference
between them live at the bottom.
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

    `load_env` writes `os.environ` directly, so monkeypatch cannot undo what it
    sets — without this teardown the last test's credentials would leak into the
    rest of the session.
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


@pytest.fixture(params=["load_env", "read_env"])
def resolve(request, clean_env):
    """Resolve the four sources and hand back the effective mapping.

    Both entry points must agree on every precedence rule; only their SIDE
    EFFECT differs, which is what the `read_env`-specific tests below cover."""
    if request.param == "load_env":

        def _resolve():
            _env.load_env()
            return dict(os.environ)

        return _resolve

    return _env.read_env


def test_saved_credentials_load_when_nothing_else_exists(resolve, paths):
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\nLIVEKIT_ROOM=r\n")

    env = resolve()

    assert env["LIVEKIT_URL"] == "wss://cloud"
    assert env["LIVEKIT_ROOM"] == "r"


def test_process_environment_beats_the_saved_file(resolve, paths, monkeypatch):
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\n")
    monkeypatch.setenv("LIVEKIT_URL", "wss://from-shell")

    assert resolve()["LIVEKIT_URL"] == "wss://from-shell"


def test_local_sfu_override_beats_saved_and_process_environment(resolve, paths, monkeypatch):
    """The override is 'the current transport', so it must win even over the shell."""
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\nLIVEKIT_ROOM=r\n")
    paths["local"].write_text("LIVEKIT_URL=ws://127.0.0.1:7880\n")
    monkeypatch.setenv("LIVEKIT_URL", "wss://from-shell")

    env = resolve()

    assert env["LIVEKIT_URL"] == "ws://127.0.0.1:7880"
    assert env["LIVEKIT_ROOM"] == "r"  # room is NOT in the override; still from saved


def test_override_is_read_from_the_state_dir_not_the_cwd(resolve, paths):
    """The regression the port fixes: the robot need not run from any directory."""
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\n")
    paths["local"].write_text("LIVEKIT_URL=ws://127.0.0.1:7880\n")
    elsewhere = paths["cwd"].parent / "elsewhere"
    elsewhere.mkdir()
    os.chdir(elsewhere)

    assert resolve()["LIVEKIT_URL"] == "ws://127.0.0.1:7880"


def test_cwd_dotenv_files_still_work_as_in_the_source_repo(resolve, paths):
    (paths["cwd"] / ".env").write_text("LIVEKIT_URL=wss://cloud\nLIVEKIT_ROOM=r\n")
    (paths["cwd"] / ".env.local").write_text("LIVEKIT_URL=ws://127.0.0.1:7880\n")

    env = resolve()

    assert env["LIVEKIT_URL"] == "ws://127.0.0.1:7880"
    assert env["LIVEKIT_ROOM"] == "r"


def test_the_saved_file_wins_over_a_cwd_dotenv(resolve, paths):
    """Among the two non-override sources the EARLIER one wins (dotenv's own
    `override=False` rule applied in order), which is what makes the saved
    credentials authoritative for a server started from an arbitrary cwd."""
    paths["saved"].write_text("LIVEKIT_ROOM=saved\n")
    (paths["cwd"] / ".env").write_text("LIVEKIT_ROOM=cwd\n")

    assert resolve()["LIVEKIT_ROOM"] == "saved"


def test_the_cwd_env_local_wins_over_the_state_dir_override(resolve, paths):
    """Among the two override sources the LATER one wins."""
    paths["local"].write_text("LIVEKIT_URL=ws://127.0.0.1:7880\n")
    (paths["cwd"] / ".env.local").write_text("LIVEKIT_URL=ws://from-env-local\n")

    assert resolve()["LIVEKIT_URL"] == "ws://from-env-local"


def test_deleting_the_override_returns_to_the_saved_credentials(clean_env, paths):
    """`load_env` needs the variable unset by hand — that is exactly the bug
    `read_env` exists to sidestep (see the next test)."""
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\n")
    paths["local"].write_text("LIVEKIT_URL=ws://127.0.0.1:7880\n")
    _env.load_env()
    assert os.environ["LIVEKIT_URL"] == "ws://127.0.0.1:7880"

    paths["local"].unlink()
    del os.environ["LIVEKIT_URL"]
    _env.load_env()

    assert os.environ["LIVEKIT_URL"] == "wss://cloud"


# --- what read_env is FOR ---------------------------------------------------


def test_read_env_does_not_mutate_the_process_environment(clean_env, paths):
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\n")
    paths["local"].write_text("LIVEKIT_URL=ws://127.0.0.1:7880\n")

    env = _env.read_env()

    assert env["LIVEKIT_URL"] == "ws://127.0.0.1:7880"
    assert "LIVEKIT_URL" not in os.environ


def test_read_env_sees_a_deleted_override_without_a_restart(clean_env, paths):
    """The whole reason it exists.

    `load_env`'s `override=True` stamps the local-SFU URL into `os.environ`;
    in a long-lived FastAPI process nothing can un-set it, so deleting
    `livekit.local.env` leaves the server dialing a dead `ws://127.0.0.1:7880`
    until it restarts. `read_env` re-resolves from disk every call."""
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\n")
    paths["local"].write_text("LIVEKIT_URL=ws://127.0.0.1:7880\n")
    assert _env.read_env()["LIVEKIT_URL"] == "ws://127.0.0.1:7880"

    paths["local"].unlink()

    assert _env.read_env()["LIVEKIT_URL"] == "wss://cloud"


def test_read_env_carries_the_rest_of_the_process_environment_through(clean_env, paths):
    """It returns what a freshly-started child would see, not just LIVEKIT_*."""
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\n")

    env = _env.read_env()

    assert env["PATH"] == os.environ["PATH"]


def test_a_bare_key_line_is_skipped_rather_than_read_as_empty(clean_env, paths):
    """Mirrors dotenv's own `set_as_environment_variables`: a `KEY` with no `=`
    has value None and must not overwrite anything with ""."""
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\n")
    paths["local"].write_text("LIVEKIT_URL\n")

    assert _env.read_env()["LIVEKIT_URL"] == "wss://cloud"


def test_required_env_names_the_missing_variable(clean_env):
    with pytest.raises(RuntimeError, match="LIVEKIT_ROOM"):
        _env.required_env("LIVEKIT_ROOM")
