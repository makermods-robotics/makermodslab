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

"""Post-hoc audit for the silent fine-tune architecture mismatch.

Background — the defect this audits for
---------------------------------------
MakerLab launches a fine-tune as ``--policy.type <A> --policy.pretrained_path
<checkpoint of architecture B>``. lerobot builds the policy class from
``--policy.type`` alone (``policies/factory.make_policy`` → ``get_policy_class(
cfg.type)``) and then calls ``<Class>.from_pretrained(config=cfg, ...)``.
Because ``config`` is passed explicitly, the checkpoint's own ``config.json`` is
never consulted; because ``PreTrainedPolicy.from_pretrained`` defaults
``strict=False`` and ``make_policy`` never overrides it, a state dict whose keys
don't match is dropped instead of raising.

When A != B the two key sets do not merely differ, they are *disjoint* —
measured on lerobot 0.6.0, a real smolvla checkpoint (500 tensors) loaded into a
fresh ACT policy (234 parameters) yields 234 missing keys, 500 unexpected keys
and **zero** loaded tensors. The run therefore trains a freshly-initialized A
while recording itself as a fine-tune of B.

What is and isn't detectable
----------------------------
The written weights carry NO evidence: a damaged fine-tune's checkpoint is
bit-for-bit the same kind of artifact a legitimate from-scratch run of type A
would produce, so inspecting ``model.safetensors`` can never settle it. Only two
things can:

1. **Provenance** — the checkpoint's ``train_config.json`` records both
   ``policy.type`` and ``policy.pretrained_path``. Comparing the declared type
   against the architecture of the recorded source is decisive whenever the
   source is still readable. This is `audit_train_config`.
2. **The run log** — lerobot's ``policies/utils.log_model_loading_keys`` emits
   ``Missing key(s) when loading model: [...]`` / ``Unexpected key(s) when
   loading model: [...]`` at WARNING level on every non-strict load, and
   MakerLab captures the trainer's stderr into ``log.jsonl``. This is
   `audit_load_warnings`, a corroborating signal.

Note that a *same*-architecture load cannot fail silently: matching keys with
mismatched shapes make ``torch.nn.Module.load_state_dict`` raise even under
``strict=False`` (verified), so such a run dies at startup rather than training
a lie. The silent window is exactly the cross-architecture case.

Everything here is a pure function over already-read artifacts — no filesystem,
no network, no registry coupling — so the caller decides what to read and how
(or whether) to surface the result.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# --- verdicts ---------------------------------------------------------------

#: The run never had a pretrained source, so this defect cannot apply to it.
NOT_A_FINETUNE = "not_a_finetune"
#: The run declared a pretrained source whose weights were silently dropped.
WEIGHTS_DISCARDED = "weights_discarded"
#: The run declared a pretrained source and did inherit its weights.
WEIGHTS_INHERITED = "weights_inherited"
#: The artifacts do not settle it either way. NEVER report this as "fine".
INDETERMINATE = "indeterminate"

# --- confidence -------------------------------------------------------------

#: Follows from the artifacts plus lerobot's documented loading semantics.
PROVEN = "proven"
#: Consistent with the artifacts but resting on MakerLab's own bookkeeping,
#: which the defect itself is known to have populated wrongly.
SUSPECTED = "suspected"
#: No usable evidence.
UNKNOWN = "unknown"


@dataclass(frozen=True)
class FinetuneAudit:
    """One run's verdict, with the evidence it rests on.

    `confidence` is part of the result on purpose: a false "your model is fine"
    is worse than an honest "cannot determine", so callers must be able to tell
    a proven clean bill of health from an unverified one.
    """

    verdict: str
    confidence: str
    reason: str
    declared_type: str | None = None
    source_type: str | None = None
    pretrained_path: str | None = None

    @property
    def is_damaged(self) -> bool:
        return self.verdict == WEIGHTS_DISCARDED

    @property
    def is_clear(self) -> bool:
        """True only for a *positively established* clean result.

        INDETERMINATE is deliberately not clear — callers that treat "not
        damaged" as "fine" would reintroduce the silent failure this module
        exists to surface.
        """
        return self.verdict in (NOT_A_FINETUNE, WEIGHTS_INHERITED)


def audit_train_config(
    train_config: dict,
    source_policy_type: str | None = None,
) -> FinetuneAudit:
    """Audit one run from its checkpoint's ``train_config.json``.

    `train_config` is the parsed file as lerobot wrote it. `source_policy_type`
    is the architecture of the checkpoint named by ``policy.pretrained_path``,
    read from *that* checkpoint's own ``config.json`` ``type`` — pass None when
    the source is gone or unreadable, which yields INDETERMINATE rather than a
    guess.
    """
    policy = train_config.get("policy")
    if not isinstance(policy, dict):
        return FinetuneAudit(
            INDETERMINATE,
            UNKNOWN,
            "train_config.json has no `policy` object, so neither the declared "
            "architecture nor the pretrained source can be read.",
        )

    declared = _clean(policy.get("type"))
    pretrained_path = _clean(policy.get("pretrained_path"))

    if declared is None:
        return FinetuneAudit(
            INDETERMINATE,
            UNKNOWN,
            "train_config.json records no `policy.type`, so the trained architecture is unknown.",
            pretrained_path=pretrained_path,
        )

    if pretrained_path is None:
        return FinetuneAudit(
            NOT_A_FINETUNE,
            PROVEN,
            f"No `policy.pretrained_path` recorded: this run trained {declared} "
            "from scratch, so there were never any inherited weights to drop.",
            declared_type=declared,
        )

    if train_config.get("resume") is True:
        # On the resume path lerobot reconstructs the policy from the
        # checkpoint's OWN train_config.json (`--config_path`), then points
        # pretrained_path at that same checkpoint — type and weights come from
        # one source, so they cannot disagree by construction. MakerLab's
        # resume branch reinforces this by never emitting --policy.type.
        return FinetuneAudit(
            WEIGHTS_INHERITED,
            PROVEN,
            f"Resumed run: lerobot rebuilt the {declared} policy from the "
            "checkpoint's own config, so the declared architecture and the "
            "loaded weights necessarily agree.",
            declared_type=declared,
            source_type=declared,
            pretrained_path=pretrained_path,
        )

    if source_policy_type is None:
        return FinetuneAudit(
            INDETERMINATE,
            UNKNOWN,
            f"Fine-tuned {declared} from {pretrained_path!r}, but that source's "
            "architecture could not be read, so whether its weights were "
            "loaded or silently dropped cannot be established.",
            declared_type=declared,
            pretrained_path=pretrained_path,
        )

    source = _clean(source_policy_type)
    if source is None:
        return FinetuneAudit(
            INDETERMINATE,
            UNKNOWN,
            f"Fine-tuned {declared} from {pretrained_path!r}, but the source "
            "reports a blank architecture, so nothing can be concluded.",
            declared_type=declared,
            pretrained_path=pretrained_path,
        )

    if source != declared:
        return FinetuneAudit(
            WEIGHTS_DISCARDED,
            PROVEN,
            f"Declared {declared} but the fine-tune source {pretrained_path!r} "
            f"is a {source} checkpoint. lerobot builds the policy from the "
            "declared type and loads the checkpoint non-strictly, and the two "
            "architectures share no parameter names — so none of the source's "
            f"weights were loaded. This artifact is a from-scratch {declared}, "
            "not a fine-tune.",
            declared_type=declared,
            source_type=source,
            pretrained_path=pretrained_path,
        )

    # Same architecture. A key-name-identical load can still fail on shape, but
    # torch raises on that even under strict=False, so a run that got far enough
    # to write this checkpoint did load the weights.
    return FinetuneAudit(
        WEIGHTS_INHERITED,
        PROVEN,
        f"Fine-tuned {declared} from a {source} checkpoint — architectures "
        "match, so the parameter names line up and the weights were loaded. "
        "(A shape disagreement would have raised at startup rather than "
        "training silently.)",
        declared_type=declared,
        source_type=source,
        pretrained_path=pretrained_path,
    )


# lerobot's policies/utils.log_model_loading_keys output, emitted on EVERY
# non-strict load. Anchored on the exact phrasing rather than a loose "key"
# match so unrelated log lines can't trip it.
_MISSING_KEYS_RE = re.compile(r"Missing key\(s\) when loading model:")
_UNEXPECTED_KEYS_RE = re.compile(r"Unexpected key\(s\) when loading model:")


@dataclass(frozen=True)
class LoadWarnings:
    """Whether a run's captured log shows lerobot dropping state-dict keys."""

    missing: bool
    unexpected: bool

    @property
    def any(self) -> bool:
        return self.missing or self.unexpected


def audit_load_warnings(messages: Iterable[str]) -> LoadWarnings:
    """Scan a run's captured log messages for non-strict-load key warnings.

    `messages` is the sequence of log lines (for MakerLab that is the
    ``"message"`` field of each ``log.jsonl`` entry — the trainer's stdout with
    stderr merged in, so lerobot's WARNING lines are present).

    Both flags set on a run that declared a pretrained source is the log-side
    fingerprint of the cross-architecture mismatch. Read this as CORROBORATING
    evidence only, and never invert it: a run whose log was truncated, or which
    died before ``make_policy``, shows no warnings while proving nothing.
    """
    missing = unexpected = False
    for message in messages:
        if not missing and _MISSING_KEYS_RE.search(message):
            missing = True
        if not unexpected and _UNEXPECTED_KEYS_RE.search(message):
            unexpected = True
        if missing and unexpected:
            break
    return LoadWarnings(missing=missing, unexpected=unexpected)


def _clean(value: object) -> str | None:
    """Trim a config string field to None when absent/blank/non-string."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
