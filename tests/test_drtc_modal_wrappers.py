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
"""The GPU image's dependency pins must equal the Lab's own.

`makermodslab/drtc/modal_policy.py` and `modal_policy_rtc.py` build the image the
remote-inference policy server runs in. Two of its pins have a counterpart in
`pyproject.toml`, and a mismatch in either is a SILENT failure:

  * **lerobot** — config compatibility follows the lerobot that WROTE a
    checkpoint. The Lab trains on the fork; a GPU on a different lerobot loads a
    checkpoint whose config it does not fully understand, or lacks a capability
    (`supports_rtc` does not exist before 0.6) the server asks for. This is
    exactly what had happened by S3.7: the image was still on upstream
    `8414188d` (0.5.2) while `pyproject.toml` had moved to the fork twice.
  * **livekit-portal** — Portal FINGERPRINTS the wire schema and drops
    mismatched packets without an error, so the two sides speaking different
    versions surfaces as a healthy-looking session with zero chunks forever.

These files import `modal` at their top level and are not importable here, so
this reads them as TEXT. That is a feature, not a workaround: the pins are
string literals inside an image-builder expression, and comparing the strings is
what actually catches a hand-edit.

The same reading applies to the tailnet-identity tests at the bottom: the node
key's home is a path literal and a Volume name inside a decorator kwarg, so the
literals are the thing to assert. The one structural claim there — that
`--state=mem:` survives ONLY under the `DRTC_TS_EPHEMERAL` opt-out — is checked
with `ast` instead, because "in the else branch" is not something a regex can
say honestly. `ast.parse` still does not import `modal`.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WRAPPERS = (
    _REPO_ROOT / "makermodslab" / "drtc" / "modal_policy.py",
    _REPO_ROOT / "makermodslab" / "drtc" / "modal_policy_rtc.py",
)
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# `"lerobot[extras] @ git+https://…/<owner>/lerobot.git@<sha>"`. Split into the
# extras, the repo URL and the SHA so a test can assert each independently — a
# right SHA on the wrong fork is still wrong.
_LEROBOT_PIN_RE = re.compile(
    r"lerobot\[(?P<extras>[^\]]*)\]\s*@\s*git\+(?P<url>https://\S*?/lerobot\.git)"
    r"'?\s*'?@(?P<sha>[0-9a-f]{40})"
)
_PORTAL_PIN_RE = re.compile(r"livekit-portal==(?P<version>[0-9][^\"'\s]*)")


def _lerobot_pin(text: str) -> re.Match[str]:
    """The single lerobot git pin in `text`, or a failure naming the file.

    The wrapper's pin is split across two adjacent string literals (the URL on
    one line, `@<sha>` on the next) so the line stays under the formatter's
    width; the regex tolerates the quotes between them, which is why it can read
    both the wrapper and pyproject's one-line form.
    """
    matches = list(_LEROBOT_PIN_RE.finditer(text))
    assert len(matches) == 1, f"expected exactly one lerobot git pin, found {len(matches)}"
    return matches[0]


@pytest.fixture(scope="module")
def expected() -> dict[str, str]:
    """The Lab's own pins — pyproject.toml is the source of truth for both."""
    pyproject = _PYPROJECT.read_text(encoding="utf-8")
    lerobot = _lerobot_pin(pyproject)
    portal = _PORTAL_PIN_RE.search(pyproject)
    assert portal is not None, "pyproject.toml's [drtc] extra no longer pins livekit-portal"
    return {"url": lerobot["url"], "sha": lerobot["sha"], "portal": portal["version"]}


@pytest.mark.parametrize("wrapper", _WRAPPERS, ids=lambda p: p.name)
def test_gpu_image_pins_the_same_lerobot_as_the_lab(wrapper: Path, expected: dict[str, str]) -> None:
    """One SHA, two files. A bump edits pyproject.toml AND both wrappers, or it
    edits none of them — there is no state in between that is merely untidy."""
    pin = _lerobot_pin(wrapper.read_text(encoding="utf-8"))
    assert pin["url"] == expected["url"], (
        f"{wrapper.name} builds its image from {pin['url']}, but the Lab installs "
        f"{expected['url']}. The GPU must run the same fork the Lab trains on."
    )
    assert pin["sha"] == expected["sha"], (
        f"{wrapper.name} pins lerobot@{pin['sha'][:9]} but pyproject.toml pins "
        f"{expected['sha'][:9]}. Bump both together."
    )


@pytest.mark.parametrize("wrapper", _WRAPPERS, ids=lambda p: p.name)
def test_gpu_image_installs_the_molmoact2_extra(wrapper: Path) -> None:
    """MolmoAct2 imports transformers at construction time and `from_pretrained`
    dies with "transformers is required" without the extra. `pi` and `smolvla`
    are here for the same reason and are asserted alongside it so a future edit
    cannot quietly drop one — the two wrappers HAD drifted apart on this
    (`modal_policy.py` carried only `smolvla`)."""
    extras = {e.strip() for e in _lerobot_pin(wrapper.read_text(encoding="utf-8"))["extras"].split(",")}
    assert {"pi", "smolvla", "molmoact2"} <= extras, (
        f"{wrapper.name} installs lerobot{sorted(extras)}; the policy servers need pi, smolvla and molmoact2."
    )


@pytest.mark.parametrize("wrapper", _WRAPPERS, ids=lambda p: p.name)
def test_gpu_image_pins_the_same_livekit_portal_as_the_lab(wrapper: Path, expected: dict[str, str]) -> None:
    """A version skew here is invisible at runtime: Portal fingerprints the
    schema and silently drops what does not match, so the symptom is a connected
    session that transfers nothing."""
    found = _PORTAL_PIN_RE.search(wrapper.read_text(encoding="utf-8"))
    assert found is not None, f"{wrapper.name} no longer pins livekit-portal"
    assert found["version"] == expected["portal"], (
        f"{wrapper.name} pins livekit-portal=={found['version']} but pyproject.toml's "
        f"[drtc] extra pins {expected['portal']}. Robot and GPU must run identical wire code."
    )


def test_the_two_wrappers_pin_identically() -> None:
    """The wrappers are deliberate near-duplicates with no shared import path
    into either image, so every shared line is a hand-mirrored one. Asserting
    the pins against each OTHER (not only against pyproject) catches the case
    where someone updates the file they happened to be reading."""
    pins = [_lerobot_pin(w.read_text(encoding="utf-8")) for w in _WRAPPERS]
    assert pins[0]["url"] == pins[1]["url"]
    assert pins[0]["sha"] == pins[1]["sha"]
    assert pins[0]["extras"] == pins[1]["extras"]

    portals = [_PORTAL_PIN_RE.search(w.read_text(encoding="utf-8")) for w in _WRAPPERS]
    assert portals[0] is not None and portals[1] is not None
    assert portals[0]["version"] == portals[1]["version"]


@pytest.mark.parametrize("wrapper", _WRAPPERS, ids=lambda p: p.name)
def test_the_wrappers_forward_model_dtype(wrapper: Path) -> None:
    """`--model-dtype` is only useful end to end: the local entrypoint takes it,
    passes it to the container function, and the container turns it into a
    server flag. Dropping any one link leaves a flag that is accepted and
    ignored, which is the worst outcome for a precision override."""
    text = wrapper.read_text(encoding="utf-8")
    # Twice in the signatures (_serve_impl and main), once in the fn.remote call.
    assert text.count('model_dtype: str = ""') == 2, wrapper.name
    assert "model_dtype=model_dtype," in text, wrapper.name
    assert 'argv += ["--model-dtype", model_dtype]' in text, wrapper.name


@pytest.mark.parametrize("wrapper", _WRAPPERS, ids=lambda p: p.name)
def test_the_wrappers_forward_flow_steps(wrapper: Path) -> None:
    """`--flow-steps` rides the same three links as `--model-dtype`, and a break
    in any of them is worse here than an ignored flag: `modal_launcher.build_argv`
    emits `--flow-steps N` unconditionally once the knob is set, so a wrapper
    whose `local_entrypoint` has no such parameter fails the run on a Click
    usage error — after the cold start has been paid for."""
    text = wrapper.read_text(encoding="utf-8")
    # Twice in the signatures (_serve_impl and main), once in the fn.remote call.
    assert text.count("flow_steps: int = 0") == 2, wrapper.name
    assert "flow_steps=flow_steps," in text, wrapper.name
    assert 'argv += ["--flow-steps", str(flow_steps)]' in text, wrapper.name


# --- the tailnet node is ONE node -------------------------------------------
# Tailscale identifies a node by its node key, which lives in tailscaled's state
# file; the wrappers keep that file on a dedicated Modal Volume so every launch
# rejoins as `modal-policy` instead of minting a `modal-policy-N`. Everything
# below guards a piece of that arrangement whose loss is INVISIBLE at runtime —
# a session with a lost state file connects perfectly well, it just leaves
# another row in the admin console.
_TS_BLOCK_BOUNDS = ("_TS_SOCKS_PORT = 1055", "async def _socks5_connect")


def _tailscale_block(text: str) -> str:
    """The hand-mirrored tailscale bootstrap, from the constants to the SOCKS client."""
    start, end = (text.index(mark) for mark in _TS_BLOCK_BOUNDS)
    return text[start:end]


@pytest.mark.parametrize("wrapper", _WRAPPERS, ids=lambda p: p.name)
def test_the_wrappers_persist_the_tailnet_node_key(wrapper: Path) -> None:
    """State file on a DEDICATED Volume, mounted beside /cache, committed twice.

    Dedicated matters: putting the node key on `hf-cache` would make every
    4 KB state commit re-commit a 20 GB model cache. Mounted matters because
    tailscaled writes the file whether or not anything persists it. And the
    commits matter most of all — a Modal Volume write is container-local until
    one happens, so without them the file dies with the container and we are
    back to a row per launch.
    """
    text = wrapper.read_text(encoding="utf-8")
    assert '_TS_STATE_FILE = "/tailscale/tailscaled.state"' in text, wrapper.name
    assert '_TS_STATE_VOLUME = "makermodslab-tailscale-state"' in text, wrapper.name
    assert "modal.Volume.from_name(_TS_STATE_VOLUME, create_if_missing=True)" in text, wrapper.name
    assert 'tailscaled.state"' in text and 'f"--state={state}"' in text, wrapper.name
    assert re.search(r'"volumes":\s*\{[^}]*_TS_STATE_DIR:\s*ts_state', text), (
        f"{wrapper.name} no longer mounts the tailscale-state Volume in _FN_KWARGS"
    )
    # Once when the backend reports Running, once on the way out (a rotated node
    # key is only ours if it reaches the Volume).
    assert '_commit_ts_state("after login")' in text, wrapper.name
    assert '_commit_ts_state("shutdown")' in text, wrapper.name


@pytest.mark.parametrize("wrapper", _WRAPPERS, ids=lambda p: p.name)
def test_state_mem_survives_only_as_the_ephemeral_opt_out(wrapper: Path) -> None:
    """`--state=mem:` is the OLD behaviour and must be reachable only via the flag.

    Asserted structurally rather than by grepping for the string: the docstrings
    still discuss `--state=mem:` at length (they explain why it changed), so a
    text search cannot distinguish prose from a regression. What must hold is
    that the single `"mem:"` in `_tailscale_up` is the body of an `if` on the
    `DRTC_TS_EPHEMERAL` answer, and that the Popen arg is the variable those
    branches set.
    """
    tree = ast.parse(wrapper.read_text(encoding="utf-8"))
    fn = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_tailscale_up"
    )
    mem = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value == "mem:"
    ]
    assert len(mem) == 1, f"{wrapper.name}: expected one `mem:` assignment, found {len(mem)}"

    branches = [node for node in ast.walk(fn) if isinstance(node, ast.If) and mem[0] in node.body]
    assert len(branches) == 1, f"{wrapper.name}: the `mem:` assignment is not inside an if-branch"
    assert ast.unparse(branches[0].test) == "ephemeral", wrapper.name
    assert "ephemeral = _ephemeral_node()" in ast.unparse(fn), wrapper.name


@pytest.mark.parametrize("wrapper", _WRAPPERS, ids=lambda p: p.name)
def test_the_wrappers_read_the_ephemeral_opt_out(wrapper: Path) -> None:
    """DRTC_TS_EPHEMERAL=1 is the escape hatch for a deliberate parallel run —
    two containers sharing one state file log in as the same node and the later
    one displaces the earlier. It is read in the container, and forwarded there
    from the operator's shell by `main()` (Modal does not ship the caller's
    environment), so both halves are asserted."""
    text = wrapper.read_text(encoding="utf-8")
    assert 'os.environ.get("DRTC_TS_EPHEMERAL", "")' in text, wrapper.name
    assert 'os.environ["DRTC_TS_EPHEMERAL"] = "1"' in text, wrapper.name
    assert "ts_ephemeral = _ephemeral_node()" in text, wrapper.name
    assert "ts_ephemeral=ts_ephemeral," in text, wrapper.name


def test_the_two_tailscale_blocks_are_identical() -> None:
    """The wrappers are deliberate near-duplicates with no shared import path
    into either image, so the tailscale bootstrap is a hand-mirrored copy — the
    file itself says "fix bugs in both". Byte equality is the only thing that
    actually holds that, and it subsumes every per-file assertion above."""
    blocks = [_tailscale_block(w.read_text(encoding="utf-8")) for w in _WRAPPERS]
    assert blocks[0] == blocks[1], (
        "the tailscale block has drifted between modal_policy.py and "
        "modal_policy_rtc.py; they must stay verbatim copies"
    )
