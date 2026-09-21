"""Precedence of the LiveKit credential sources in `makermodslab.drtc._env`.

Pure-helper test: no LiveKit, no hardware. `_env` is deliberately importable
without the `drtc` extra (it needs only python-dotenv) so this runs in CI.

Since S3.6 there are two rungs, not five: `livekit.env` and the process
environment, which wins. The three that are gone — a cwd `.env`, a cwd
`.env.local`, and the `livekit.local.env` override the retired
`tools/drtc/local_sfu*.sh` scripts wrote — were cwd-relative or `override=True`,
and both properties were bugs in a long-lived server. The Lab hosts the SFU
itself now (`makermodslab --sfu`) and the session pins the child's transport on
its command line, so no file has to describe "the current SFU" at all.

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
    """Point the saved-credentials file at tmp_path and run from a fresh cwd.

    The `chdir` is not decoration: it is what proves the cwd no longer matters.
    A developer running pytest from a checkout carrying a `.env` must get the
    same answer CI does.
    """
    saved = tmp_path / "livekit.env"
    monkeypatch.setattr(_env, "DRTC_ENV_PATH", str(saved))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return {"saved": saved, "cwd": cwd}


@pytest.fixture(params=["load_env", "read_env"])
def resolve(request, clean_env):
    """Resolve the sources and hand back the effective mapping.

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


def test_a_missing_saved_file_is_not_an_error(resolve, paths):
    """A station running the Lab's own SFU never writes one, and starting a
    remote run there must not depend on a file nothing creates."""
    assert not paths["saved"].exists()

    assert "LIVEKIT_URL" not in resolve()


@pytest.mark.parametrize("name", [".env", ".env.local"])
def test_a_cwd_dotenv_is_no_longer_read(resolve, paths, name):
    """RETIRED in S3.6. Both cwd rungs made the answer depend on where the Lab
    happened to be started — the cause of two "connection refused" false starts
    on 2026-09-02 — and `.env.local` did it with `override=True`, so it beat
    even the process environment."""
    (paths["cwd"] / name).write_text("LIVEKIT_URL=ws://from-cwd\n")
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\n")

    assert resolve()["LIVEKIT_URL"] == "wss://cloud"


def test_the_local_sfu_override_file_is_no_longer_read(resolve, paths, tmp_path):
    """RETIRED in S3.6 with the scripts that wrote it. It loaded with
    `override=True`, so once a long-lived server had read it, deleting the file
    could never un-set what it stamped into `os.environ` — the server kept
    dialing a dead `ws://127.0.0.1:7880` until it restarted."""
    (tmp_path / "livekit.local.env").write_text("LIVEKIT_URL=ws://127.0.0.1:7880\n")
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\n")

    assert resolve()["LIVEKIT_URL"] == "wss://cloud"


# --- what read_env is FOR ---------------------------------------------------


def test_read_env_does_not_mutate_the_process_environment(clean_env, paths):
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\n")

    env = _env.read_env()

    assert env["LIVEKIT_URL"] == "wss://cloud"
    assert "LIVEKIT_URL" not in os.environ


def test_read_env_sees_an_edited_file_without_a_restart(clean_env, paths):
    """The whole reason it exists. With no override rung left it can no longer
    disagree with `load_env` about a deleted file, but it is still the only
    correct call from a server that must re-resolve on every request."""
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\n")
    assert _env.read_env()["LIVEKIT_URL"] == "wss://cloud"

    paths["saved"].write_text("LIVEKIT_URL=wss://other\n")

    assert _env.read_env()["LIVEKIT_URL"] == "wss://other"


def test_read_env_carries_the_rest_of_the_process_environment_through(clean_env, paths):
    """It returns what a freshly-started child would see, not just LIVEKIT_*."""
    paths["saved"].write_text("LIVEKIT_URL=wss://cloud\n")

    env = _env.read_env()

    assert env["PATH"] == os.environ["PATH"]


def test_a_bare_key_line_is_skipped_rather_than_read_as_empty(clean_env, paths):
    """Mirrors dotenv's own `set_as_environment_variables`: a `KEY` with no `=`
    has value None and must not be written as ""."""
    paths["saved"].write_text("LIVEKIT_URL\nLIVEKIT_ROOM=r\n")

    env = _env.read_env()

    assert "LIVEKIT_URL" not in env
    assert env["LIVEKIT_ROOM"] == "r"


def test_required_env_names_the_missing_variable(clean_env):
    with pytest.raises(RuntimeError, match="LIVEKIT_ROOM"):
        _env.required_env("LIVEKIT_ROOM")
