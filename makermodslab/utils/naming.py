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

"""Human-facing names derived from Hub repo ids, model paths and run metadata.

A leaf module by necessity as well as by taste: `models` already imports `jobs`
at module scope, so the naming rules the two share (the policy-type vocabulary,
the MakerMods Lab run-repo timestamp suffix, collision resolution) can only live
somewhere both can import without a cycle. Nothing here touches the filesystem,
the Hub, or the registry — every function is a pure string transform, which is
also what makes the derivation cheap to unit-test.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Hashable, Sequence

# Canonical policy types lerobot can load — mirrors the registry in
# lerobot/policies/factory.py (get_policy_class / make_policy_config). Defined
# once here as the single source of truth for recognizing a model's policy type
# from its tags / name / card; update alongside lerobot pin bumps.
#
# This is a RECOGNITION vocabulary, not a menu: it decides whether a Hub tag or
# a repo-name prefix names a policy, so a type missing from it makes a perfectly
# good model list with `policy_type: null`. It is therefore mirrored WHOLE from
# the pin rather than curated down to what MakerMods Lab can train — the
# training picker is server.py's `_POLICY_TYPE_TO_LEROBOT`, a separate and
# deliberately smaller list.
#
# Verified against the b968c0c01 pin with
# `PreTrainedConfig.get_known_choices()` (19 types); the eight below the blank
# line had drifted out of the mirror across pin bumps.
KNOWN_POLICY_TYPES = {
    "tdmpc",
    "diffusion",
    "act",
    "multi_task_dit",
    "vqbet",
    "pi0",
    "pi05",
    "gaussian_actor",
    "smolvla",
    "wall_x",
    "pi0_fast",
    "eo1",
    "evo1",
    "fastwam",
    "groot",
    "lingbot_va",
    "molmoact2",
    "vla_jepa",
    "xvla",
}

# Longest-first for name-prefix matching, so a shorter type can never shadow a
# longer one it prefixes (pi0 must not swallow a pi0_fast_... repo name).
POLICY_TYPES_BY_LENGTH = sorted(KNOWN_POLICY_TYPES, key=len, reverse=True)

# The `_<date>_<time>` tail every MakerMods Lab run id (and therefore every repo
# a run publishes to) carries — see jobs._generate_job_id. Its presence is what
# marks a name as machine-generated rather than human-chosen.
RUN_REPO_TIMESTAMP_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})$")


def _split_source(source: str) -> tuple[str, str]:
    """(namespace, basename) for an import source.

    Handles both shapes `register_imported` stores: a bare Hub repo id
    (`ns/name` — exactly one slash, no leading `/`, `~` or `.`) and a local
    directory path. Only a repo id yields a namespace; a path's parent
    directory is not one, and using it would strip real text out of titles.
    """
    src = source.strip().replace("\\", "/").rstrip("/")
    basename = src.rsplit("/", 1)[-1]
    is_repo_id = src.count("/") == 1 and not src.startswith(("/", "~", "."))
    namespace = src.split("/", 1)[0] if is_repo_id else ""
    return namespace, basename


def policy_type_from_name(name: str) -> str | None:
    """The policy type a MakerMods Lab-generated model name starts with, or None.

    Matched longest-first so `pi0` never shadows `pi0_fast`. Mirrors (and now
    backs) models._hub_policy_type's name-prefix signal.
    """
    for policy_type in POLICY_TYPES_BY_LENGTH:
        if name.startswith(policy_type + "_"):
            return policy_type
    return None


def derive_imported_title(source: str) -> str:
    """The card title for an imported model, derived from its repo id or path.

    A model MakerMods Lab trained is published under a fully machine-generated
    name — `{namespace}/{policy}_{dataset_slug}_{timestamp}` (jobs.start →
    _generate_job_id) — in which every segment but one is already stated
    elsewhere on the card: the provenance chip says it was imported, the Policy
    row says which policy, the subtitle carries the full source. The title's
    pixels are worth more spent on the one thing nothing else says, the TASK, so
    peel off the rest:

      makermods/smolvla_makermods_orange_box_2026-08-03_12-53-30 → orange_box

    namespace → policy token → the dataset's own namespace repeated as an infix
    (taken from the repo's namespace rather than hardcoded, so it strips for any
    user) → the trailing timestamp, which the subtitle keeps.

    The timestamp is the signature of a generated name, so a source WITHOUT one
    is treated as human-chosen and only loses its namespace — a community repo
    like `lerobot/smolvla_base` keeps every word its author wrote, including a
    leading policy token that here is part of the name rather than a prefix.
    Callers render long fallbacks with a middle ellipsis (see
    frontend/src/lib/modelNames.ts) so both ends stay readable.
    """
    namespace, basename = _split_source(source)
    match = RUN_REPO_TIMESTAMP_RE.search(basename)
    if match is None:
        return basename or source.strip()

    stem = basename[: match.start()]
    policy_type = policy_type_from_name(stem)
    if policy_type:
        stem = stem[len(policy_type) + 1 :]
    if namespace and stem.lower().startswith(namespace.lower() + "_"):
        stem = stem[len(namespace) + 1 :]
    return stem or basename


def imported_name_suffixes(source: str) -> list[str]:
    """Disambiguators for an imported title, least → most specific.

    Two imports of the same task — a rerun, or the same policy retrained on more
    steps — derive the same title, so the timestamp `derive_imported_title` drops
    comes back as the tiebreak: first its date, then date + time for two runs on
    one day. A source with no timestamp offers its namespace instead, which
    separates the common case of the same model pulled from two authors.
    """
    namespace, basename = _split_source(source)
    match = RUN_REPO_TIMESTAMP_RE.search(basename)
    if match is None:
        return [namespace] if namespace else []
    date, hour, minute, _second = match.groups()
    return [date, f"{date} {hour}:{minute}"]


def iso_time_suffixes(iso: str | None) -> list[str]:
    """The same ladder as `imported_name_suffixes`, read off an ISO timestamp.

    Used for listing rows, whose disambiguator is the repo/checkpoint's
    last-modified time rather than a name they may not carry one in.
    """
    if not iso or len(iso) < 10:
        return []
    date = iso[:10]
    if len(iso) >= 16 and iso[10] in " T":
        return [date, f"{date} {iso[11:16]}"]
    return [date]


def dedupe_display_names(
    entries: Sequence[tuple[str, Sequence[str]]],
    group_keys: Sequence[Hashable] | None = None,
) -> list[str]:
    """Make a listing's auto-generated display names unique.

    Auto-generated names collide by construction: two runs of one policy on one
    dataset are both named "SMOLVLA · ns/task" (jobs.start), and two imports of
    one task both derive to "task". The listing then shows several rows under a
    single label with nothing on the card to tell them apart.

    `entries` pairs each item's name with its candidate disambiguators, ordered
    least → most specific. A name that occurs once is returned untouched. For a
    colliding group the FIRST candidate level that separates every member is
    appended as " (<suffix>)"; if no level does — no candidates, or two items
    genuinely share even the most specific one — a 1-based ordinal is appended
    so the result is unique regardless.

    `group_keys`, when given, is a parallel sequence naming a SECOND axis the
    listing already renders in its own field — in practice the policy type. Two
    items only collide when they match on the name AND that key, because an ACT
    and a SmolVLA of one task are told apart by the Policy row a few pixels
    below the title: suffixing them would spend the title's scarcest space
    restating a fact the card already prints, and imply a difference in identity
    where the real one is in policy. The key affects GROUPING only; the label
    stays the bare name, since the key is never what the title should carry.

    Order-preserving and deterministic: the labels don't reshuffle between two
    requests over the same data, which is what makes them safe to show on a card
    the user is reading.
    """
    names = [name for name, _ in entries]
    keys: list[tuple[str, Hashable]] = (
        [(name, key) for name, key in zip(names, group_keys, strict=True)]
        if group_keys is not None
        else [(name, None) for name in names]
    )
    counts = Counter(keys)
    resolved = list(names)

    groups: dict[tuple[str, Hashable], list[int]] = {}
    for index, key in enumerate(keys):
        if counts[key] > 1:
            groups.setdefault(key, []).append(index)

    for (name, _key), indices in groups.items():
        depth = max(len(entries[i][1]) for i in indices)
        for level in range(depth):
            candidates = [entries[i][1][level] if level < len(entries[i][1]) else "" for i in indices]
            if all(candidates) and len(set(candidates)) == len(candidates):
                for index, suffix in zip(indices, candidates, strict=True):
                    resolved[index] = f"{name} ({suffix})"
                break
        else:
            for ordinal, index in enumerate(indices, start=1):
                resolved[index] = f"{name} ({ordinal})"

    return resolved
