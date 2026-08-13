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
"""Tests for makermodslab.utils.naming — the pure title/collision rules shared
by the jobs registry and the models listing. No filesystem, no Hub, no registry:
every case here is a string in and a string out."""

from __future__ import annotations

import pytest

from makermodslab.utils.naming import (
    dedupe_display_names,
    derive_imported_title,
    imported_name_suffixes,
    iso_time_suffixes,
    policy_type_from_name,
)

# ── derive_imported_title ────────────────────────────────────────────────────


def test_derive_title_peels_a_generated_repo_id_down_to_the_task() -> None:
    """The shape MakerMods Lab publishes: namespace, policy token, the dataset's
    namespace repeated as an infix, and a timestamp — none of which the title
    needs to carry (the chip, the policy row and the subtitle say all three)."""
    assert derive_imported_title("makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30") == "orange_box"
    assert (
        derive_imported_title("makermods/act_makermods_eraser_place_unblurry_real_2026-07-31_17-35-54")
        == "eraser_place_unblurry_real"
    )


def test_derive_title_strips_the_longest_policy_token() -> None:
    """pi0 must not swallow a pi0_fast repo's name (same longest-first rule as
    models._hub_policy_type)."""
    assert derive_imported_title("ns/pi0_fast_ns_sock_2026-01-01_10-00-00") == "sock"
    assert derive_imported_title("ns/pi0_ns_sock_2026-01-01_10-00-00") == "sock"


def test_derive_title_only_strips_the_repos_own_namespace_as_infix() -> None:
    """The infix is stripped because it repeats the namespace, not because it
    reads like an org — a dataset owned by someone else keeps its prefix."""
    assert derive_imported_title("alice/act_bob_orange_box_2026-08-03_12-53-30") == "bob_orange_box"


def test_derive_title_leaves_a_community_repo_alone_but_for_its_namespace() -> None:
    """No timestamp ⇒ a human named this, so every word is load-bearing. Notably
    the leading policy token stays: in `smolvla_base` it IS the name."""
    assert derive_imported_title("lerobot/smolvla_base") == "smolvla_base"
    assert derive_imported_title("lerobot/pi0") == "pi0"
    assert derive_imported_title("physical-intelligence/pi05_droid") == "pi05_droid"


def test_derive_title_handles_local_paths() -> None:
    """A local import's source is a path: the basename is the title, and a
    parent directory is never treated as a namespace to strip."""
    assert derive_imported_title("/Users/me/models/my_policy") == "my_policy"
    # The timestamp still marks the name as generated, so the policy token goes
    # — but the "me_" infix stays: with no namespace there is nothing saying it
    # is a repetition rather than part of the task.
    assert derive_imported_title("/Users/me/smolvla_me_orange_box_2026-08-03_12-53-30") == "me_orange_box"
    assert derive_imported_title("/Users/me/models/trailing/") == "trailing"


def test_derive_title_never_returns_empty() -> None:
    """Degenerate names must still label a card."""
    assert derive_imported_title("ns/_2026-08-03_12-53-30") == "_2026-08-03_12-53-30"
    assert derive_imported_title("ns/smolvla_2026-08-03_12-53-30") == "smolvla"


def test_policy_type_from_name() -> None:
    assert policy_type_from_name("pi0_fast_sock_2026-01-01_10-00-00") == "pi0_fast"
    assert policy_type_from_name("pi0_sock_2026-01-01_10-00-00") == "pi0"
    assert policy_type_from_name("orange_box") is None
    # A human-named repo can open with a policy token too — which is exactly why
    # derive_imported_title consults this only for a timestamped (generated)
    # name, and leaves `smolvla_base` whole.
    assert policy_type_from_name("smolvla_base") == "smolvla"


# ── suffix ladders ───────────────────────────────────────────────────────────


def test_imported_name_suffixes_ladder() -> None:
    assert imported_name_suffixes("ns/smolvla_ns_box_2026-08-03_12-53-30") == [
        "2026-08-03",
        "2026-08-03 12:53",
    ]
    # No timestamp to fall back on — the namespace separates two authors' copies.
    assert imported_name_suffixes("lerobot/pi0") == ["lerobot"]
    assert imported_name_suffixes("/Users/me/models/pi0") == []


def test_iso_time_suffixes_ladder() -> None:
    assert iso_time_suffixes("2026-07-31T17:35:54Z") == ["2026-07-31", "2026-07-31 17:35"]
    assert iso_time_suffixes("2026-07-31") == ["2026-07-31"]
    assert iso_time_suffixes(None) == []
    assert iso_time_suffixes("") == []


# ── dedupe_display_names ─────────────────────────────────────────────────────


def test_dedupe_leaves_unique_names_untouched() -> None:
    assert dedupe_display_names([("a", ["x"]), ("b", ["y"])]) == ["a", "b"]


def test_dedupe_separates_two_runs_of_one_policy_on_one_dataset() -> None:
    """The reported case: two repos, same policy, same dataset, so the listing
    served the same enriched name twice with nothing to tell them apart."""
    name = "SMOLVLA · makermods/eraser_place_unblurry_real"
    assert dedupe_display_names(
        [
            (name, iso_time_suffixes("2026-07-31T17:35:54Z")),
            (name, iso_time_suffixes("2026-08-02T12:22:54Z")),
        ]
    ) == [f"{name} (2026-07-31)", f"{name} (2026-08-02)"]


def test_dedupe_escalates_to_time_for_two_runs_on_one_day() -> None:
    """The date alone doesn't separate them, so the ladder's next rung does —
    and the shorter rung is not used for either, or one card would read as the
    more specific of a pair that isn't."""
    name = "SMOLVLA · makermods/eraser_place_unblurry_real"
    assert dedupe_display_names(
        [
            (name, iso_time_suffixes("2026-07-31T17:35:54Z")),
            (name, iso_time_suffixes("2026-07-31T12:22:54Z")),
        ]
    ) == [f"{name} (2026-07-31 17:35)", f"{name} (2026-07-31 12:22)"]


def test_dedupe_falls_back_to_an_ordinal_when_nothing_separates() -> None:
    """No candidates at all, and a tie on the most specific one, both resolve —
    a listing must never show one label twice."""
    assert dedupe_display_names([("box", []), ("box", [])]) == ["box (1)", "box (2)"]
    assert dedupe_display_names([("box", ["2026-08-03"]), ("box", ["2026-08-03"])]) == ["box (1)", "box (2)"]


def test_dedupe_does_not_suffix_two_policies_of_one_task() -> None:
    """An ACT and a SmolVLA of `eraser_place` derive the same title, but the
    card's Policy row already separates them — a date suffix there would spend
    the title's scarcest pixels restating a fact printed just below it, and
    imply the two differ by WHEN they ran rather than by what they are."""
    assert dedupe_display_names(
        [
            ("eraser_place", ["2026-07-31"]),
            ("eraser_place", ["2026-08-02"]),
        ],
        group_keys=["act", "smolvla"],
    ) == ["eraser_place", "eraser_place"]


def test_dedupe_still_separates_two_runs_of_one_policy() -> None:
    """The key narrows collisions, it doesn't abolish them: same title AND same
    policy is still two rows a user can't tell apart."""
    assert dedupe_display_names(
        [
            ("eraser_place", ["2026-07-31"]),
            ("eraser_place", ["2026-08-02"]),
        ],
        group_keys=["smolvla", "smolvla"],
    ) == ["eraser_place (2026-07-31)", "eraser_place (2026-08-02)"]


def test_dedupe_group_key_splits_only_within_a_shared_title() -> None:
    """Three of one task: the two SmolVLAs disambiguate against each other and
    the lone ACT stays bare — a suffix it doesn't need would read as a fourth
    thing to compare."""
    assert dedupe_display_names(
        [
            ("eraser_place", ["2026-07-31"]),
            ("eraser_place", ["2026-08-02"]),
            ("eraser_place", ["2026-08-05"]),
        ],
        group_keys=["smolvla", "smolvla", "act"],
    ) == ["eraser_place (2026-07-31)", "eraser_place (2026-08-02)", "eraser_place"]


def test_dedupe_treats_an_unknown_policy_as_its_own_key() -> None:
    """Two rows whose policy could not be read show the SAME Policy row (or
    none), so they are genuinely indistinguishable and must still be suffixed —
    the key groups on what the card renders, not on what is true."""
    assert dedupe_display_names(
        [("box", ["2026-08-03"]), ("box", ["2026-08-05"])],
        group_keys=[None, None],
    ) == ["box (2026-08-03)", "box (2026-08-05)"]


def test_dedupe_group_keys_length_is_checked() -> None:
    """A caller that zips a shorter key list would silently mis-group rows."""
    with pytest.raises(ValueError):
        dedupe_display_names([("a", []), ("a", [])], group_keys=["act"])


def test_dedupe_is_order_preserving_and_deterministic() -> None:
    entries = [
        ("box", ["2026-08-03"]),
        ("sock", ["2026-08-01"]),
        ("box", ["2026-08-04"]),
    ]
    assert dedupe_display_names(entries) == [
        "box (2026-08-03)",
        "sock",
        "box (2026-08-04)",
    ]
    assert dedupe_display_names(entries) == dedupe_display_names(entries)
