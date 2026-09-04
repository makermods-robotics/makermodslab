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
"""

from __future__ import annotations

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
