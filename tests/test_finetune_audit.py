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
"""Tests for makerlab.finetune_audit — pure detection helpers for the silent
fine-tune architecture mismatch. No filesystem, no network."""

from __future__ import annotations

from makerlab.finetune_audit import (
    INDETERMINATE,
    NOT_A_FINETUNE,
    PROVEN,
    UNKNOWN,
    WEIGHTS_DISCARDED,
    WEIGHTS_INHERITED,
    audit_load_warnings,
    audit_train_config,
)


def _tc(policy: dict | None, **top) -> dict:
    tc: dict = dict(top)
    if policy is not None:
        tc["policy"] = policy
    return tc


# --- audit_train_config -----------------------------------------------------


def test_no_pretrained_path_is_provably_not_a_finetune() -> None:
    """The common case for every run the user has: trained from scratch, so
    there were never inherited weights that could be dropped."""
    audit = audit_train_config(_tc({"type": "act", "pretrained_path": None}))
    assert audit.verdict == NOT_A_FINETUNE
    assert audit.confidence == PROVEN
    assert audit.is_clear
    assert not audit.is_damaged


def test_mismatched_source_is_provably_damaged() -> None:
    audit = audit_train_config(
        _tc({"type": "act", "pretrained_path": "lerobot/smolvla_base"}),
        source_policy_type="smolvla",
    )
    assert audit.verdict == WEIGHTS_DISCARDED
    assert audit.confidence == PROVEN
    assert audit.is_damaged
    assert not audit.is_clear
    assert audit.declared_type == "act"
    assert audit.source_type == "smolvla"
    # The reason must say what the artifact actually is, not just that it's bad.
    assert "from-scratch" in audit.reason


def test_matching_source_is_provably_a_real_finetune() -> None:
    audit = audit_train_config(
        _tc({"type": "smolvla", "pretrained_path": "lerobot/smolvla_base"}),
        source_policy_type="smolvla",
    )
    assert audit.verdict == WEIGHTS_INHERITED
    assert audit.confidence == PROVEN
    assert audit.is_clear


def test_resume_is_provably_safe_without_knowing_the_source() -> None:
    """A resumed run rebuilds the policy from the checkpoint's own config, so
    type and weights agree by construction — no source lookup required."""
    audit = audit_train_config(
        _tc(
            {"type": "smolvla", "pretrained_path": "/tmp/ck/pretrained_model"},
            resume=True,
        )
    )
    assert audit.verdict == WEIGHTS_INHERITED
    assert audit.confidence == PROVEN


def test_unknown_source_is_indeterminate_not_clean() -> None:
    """The load-bearing case: an unreadable source must never be reported as
    fine. INDETERMINATE is neither damaged nor clear."""
    audit = audit_train_config(
        _tc({"type": "act", "pretrained_path": "someone/deleted-repo"}),
        source_policy_type=None,
    )
    assert audit.verdict == INDETERMINATE
    assert audit.confidence == UNKNOWN
    assert not audit.is_clear
    assert not audit.is_damaged


def test_blank_source_type_is_indeterminate() -> None:
    audit = audit_train_config(_tc({"type": "act", "pretrained_path": "x/y"}), source_policy_type="  ")
    assert audit.verdict == INDETERMINATE
    assert not audit.is_clear


def test_malformed_train_config_is_indeterminate() -> None:
    for tc in (_tc(None), _tc({"pretrained_path": "x/y"}), _tc({"type": ""})):
        audit = audit_train_config(tc)
        assert audit.verdict == INDETERMINATE, tc
        assert not audit.is_clear


def test_blank_pretrained_path_counts_as_absent() -> None:
    audit = audit_train_config(_tc({"type": "act", "pretrained_path": "  "}))
    assert audit.verdict == NOT_A_FINETUNE


# --- audit_load_warnings ----------------------------------------------------


def test_load_warnings_detects_lerobot_key_mismatch() -> None:
    """The log-side fingerprint: lerobot's log_model_loading_keys emits both
    lines when a cross-architecture checkpoint is dropped."""
    messages = [
        "INFO 2026-07-29 10:00:00 ot_train.py:232 {'batch_size': 8,",
        "WARNING 2026-07-29 10:00:01     utils.py:91 Missing key(s) when loading "
        "model: ['model.backbone.bn1.bias', 'model.action_head.weight']",
        "WARNING 2026-07-29 10:00:01     utils.py:93 Unexpected key(s) when loading "
        "model: ['model.action_in_proj.bias']",
    ]
    warnings = audit_load_warnings(messages)
    assert warnings.missing
    assert warnings.unexpected
    assert warnings.any


def test_load_warnings_absent_on_a_clean_log() -> None:
    warnings = audit_load_warnings(["INFO ot_train.py:232 {'batch_size': 8,", "INFO step:100 loss:0.1"])
    assert not warnings.any


def test_load_warnings_ignores_unrelated_key_talk() -> None:
    """Anchored on lerobot's exact phrasing so ordinary log chatter mentioning
    keys can't produce a false positive."""
    warnings = audit_load_warnings(
        [
            "INFO downloading key file",
            "ERROR missing keys in the dataset metadata",
            "WARNING unexpected keyword argument 'foo'",
        ]
    )
    assert not warnings.any


def test_load_warnings_accepts_a_generator() -> None:
    warnings = audit_load_warnings(m for m in ["Missing key(s) when loading model: []"])
    assert warnings.missing
    assert not warnings.unexpected
