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
"""Tests for makermodslab.utils.system — pip-extra install helpers + CUDA detection."""

from __future__ import annotations

import logging
import sys


def test_build_install_cmd_contains_pip_and_package() -> None:
    from makermodslab.utils.system import _build_install_cmd

    cmd = _build_install_cmd("lerobot[training]")
    # Command may use `uv pip install` or `python -m pip install` depending on env.
    assert "pip" in cmd
    assert "install" in cmd
    assert "lerobot[training]" in cmd


def test_build_install_cmd_uses_current_python_when_no_uv(monkeypatch) -> None:
    import shutil

    from makermodslab.utils import system
    from makermodslab.utils.system import _build_install_cmd

    # If uv is not on PATH (nor at any standard install location), the
    # command must use sys.executable.
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(system, "_UV_FALLBACK_PATHS", ())
    cmd = _build_install_cmd("lerobot[training]")
    assert cmd[0] == sys.executable
    assert "pip" in cmd
    assert "install" in cmd
    assert "lerobot[training]" in cmd


def test_install_manager_initial_state_is_idle() -> None:
    from makermodslab.utils.system import InstallManager

    # InstallManager requires a package name argument.
    mgr = InstallManager("some-package")
    status = mgr.get_status()
    assert status["state"] == "idle"
    assert status["error"] is None
    assert isinstance(status["logs"], list)


# --- CUDA / GPU mismatch detection (issue #30) --------------------------------


def test_detect_cuda_status_flags_mismatch_when_gpu_but_cpu_torch(monkeypatch) -> None:
    """GPU present + no CUDA in PyTorch should report a mismatch."""
    from makermodslab.utils import system

    monkeypatch.setattr(system, "_nvidia_gpu_present", lambda: True)
    monkeypatch.setattr(system, "_torch_cuda", lambda: (False, "2.10.0+cpu"))

    status = system.detect_cuda_status()
    assert status["gpu_present"] is True
    assert status["cuda_available"] is False
    assert status["mismatch"] is True
    assert status["torch_version"] == "2.10.0+cpu"
    assert status["docs_url"].startswith("https://pytorch.org")


def test_detect_cuda_status_no_mismatch_when_cuda_available(monkeypatch) -> None:
    from makermodslab.utils import system

    monkeypatch.setattr(system, "_nvidia_gpu_present", lambda: True)
    monkeypatch.setattr(system, "_torch_cuda", lambda: (True, "2.10.0+cu124"))

    assert system.detect_cuda_status()["mismatch"] is False


def test_detect_cuda_status_no_mismatch_without_gpu(monkeypatch) -> None:
    """No GPU (e.g. a Mac/CPU box) must not nag — CPU torch is expected there."""
    from makermodslab.utils import system

    monkeypatch.setattr(system, "_nvidia_gpu_present", lambda: False)
    monkeypatch.setattr(system, "_torch_cuda", lambda: (False, "2.10.0+cpu"))

    assert system.detect_cuda_status()["mismatch"] is False


def test_nvidia_gpu_present_false_when_smi_absent(monkeypatch) -> None:
    import shutil

    from makermodslab.utils import system

    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert system._nvidia_gpu_present() is False


def test_warn_if_cuda_mismatch_logs_on_mismatch(monkeypatch, caplog) -> None:
    from makermodslab.utils import system

    monkeypatch.setattr(system, "_nvidia_gpu_present", lambda: True)
    monkeypatch.setattr(system, "_torch_cuda", lambda: (False, "2.10.0+cpu"))

    with caplog.at_level(logging.WARNING, logger="makermodslab.utils.system"):
        system.warn_if_cuda_mismatch()
    assert any("use CUDA" in rec.message for rec in caplog.records)


def test_warn_if_cuda_mismatch_silent_when_ok(monkeypatch, caplog) -> None:
    from makermodslab.utils import system

    monkeypatch.setattr(system, "_nvidia_gpu_present", lambda: True)
    monkeypatch.setattr(system, "_torch_cuda", lambda: (True, "2.10.0+cu124"))

    with caplog.at_level(logging.WARNING, logger="makermodslab.utils.system"):
        system.warn_if_cuda_mismatch()
    assert caplog.records == []


def test_policy_extra_maps_policies_to_install_targets() -> None:
    """smolvla/pi0/pi0_fast/diffusion map to the right probe module + lerobot[extra]."""
    from makermodslab.utils.system import handle_get_policy_extra

    smol = handle_get_policy_extra("smolvla")
    assert smol["needs_extra"] is True
    assert smol["package"] == "transformers"
    assert smol["install_target"] == "lerobot[smolvla]"
    assert "lerobot[smolvla]" in smol["install_hint"]

    # pi0, pi0_fast, and pi05 share the lerobot[pi] extra; diffusion uses diffusers.
    assert handle_get_policy_extra("pi0")["install_target"] == "lerobot[pi]"
    assert handle_get_policy_extra("pi0_fast")["install_target"] == "lerobot[pi]"
    assert handle_get_policy_extra("pi05")["install_target"] == "lerobot[pi]"
    assert handle_get_policy_extra("diffusion")["package"] == "diffusers"
    assert handle_get_policy_extra("diffusion")["install_target"] == "lerobot[diffusion]"


def test_policy_extra_molmoact2_probes_transformers_not_peft_or_scipy() -> None:
    """MolmoAct2's extra is transformers + peft + scipy, but only transformers
    is a construction-time requirement for a rollout on this pin: peft is
    reached only under enable_lora_vlm, and scipy only when the checkpoint's
    action_mode is "discrete"/"both" (the released checkpoints save
    "continuous"). Probing either would report the extra missing for a
    checkpoint that runs fine, and DeployPanel refuses to launch on that."""
    from makermodslab.utils.system import handle_get_policy_extra

    molmo = handle_get_policy_extra("molmoact2")
    assert molmo["needs_extra"] is True
    assert molmo["package"] == "transformers"
    assert molmo["install_target"] == "lerobot[molmoact2]"
    assert "lerobot[molmoact2]" in molmo["install_hint"]


# ── policy runtime requirements (policy_inference_args & friends) ────────────
#
# Pure functions of a checkpoint's saved config.json — no filesystem here, the
# dicts below stand in for the file rollout.py reads.


def test_policy_inference_args_are_empty_for_every_other_policy() -> None:
    from makermodslab.utils.system import policy_inference_args

    assert policy_inference_args({"type": "act"}) == []
    assert policy_inference_args({"type": "smolvla"}) == []
    # An unreadable / missing config.json arrives as {} — add nothing rather
    # than guessing, and let the subprocess report whatever is really wrong.
    assert policy_inference_args({}) == []


def test_policy_inference_args_fill_in_a_missing_molmoact2_action_mode() -> None:
    """MolmoAct2Config.inference_action_mode defaults to None and the policy
    raises on None, so a checkpoint that saved no mode cannot be rolled out at
    all without this override."""
    from makermodslab.utils.system import policy_inference_args

    assert policy_inference_args({"type": "molmoact2", "action_mode": "both"}) == [
        "--policy.inference_action_mode=continuous"
    ]
    # A discrete-only checkpoint gets discrete: forcing continuous onto it
    # raises in MolmoAct2Config.__post_init__, which would swap one fatal
    # error for another.
    assert policy_inference_args({"type": "molmoact2", "action_mode": "discrete"}) == [
        "--policy.inference_action_mode=discrete"
    ]


def test_policy_inference_args_leave_an_explicit_molmoact2_mode_alone() -> None:
    """The released lerobot/MolmoAct2-*-LeRobot configs already save
    inference_action_mode; a rollout must run the policy the way its own config
    says, so there is nothing to override."""
    from makermodslab.utils.system import policy_inference_args

    saved = {"type": "molmoact2", "action_mode": "continuous", "inference_action_mode": "continuous"}
    assert policy_inference_args(saved) == []
    saved_discrete = {"type": "molmoact2", "action_mode": "both", "inference_action_mode": "discrete"}
    assert policy_inference_args(saved_discrete) == []


def test_molmoact2_rtc_conflict_only_blocks_discrete_checkpoints() -> None:
    """MolmoAct2Policy.supports_rtc() is `inference_action_mode ==
    "continuous"`, and build_rollout_context turns a False into a ValueError —
    but only after loading a multi-GB VLM. Answer from the saved config."""
    from makermodslab.utils.system import molmoact2_rtc_conflict

    assert molmoact2_rtc_conflict({"type": "molmoact2", "inference_action_mode": "continuous"}) is None
    assert molmoact2_rtc_conflict({"type": "molmoact2", "action_mode": "both"}) is None
    conflict = molmoact2_rtc_conflict({"type": "molmoact2", "inference_action_mode": "discrete"})
    assert conflict is not None
    assert "continuous" in conflict
    # Every other policy's RTC support is decided upstream — don't pretend to
    # know it here.
    assert molmoact2_rtc_conflict({"type": "act"}) is None
    assert molmoact2_rtc_conflict({}) is None


def test_the_flow_steps_field_is_per_family_and_verified_against_the_pin() -> None:
    """WHICH field a `--flow-steps` override writes, and the one that traps:
    MolmoAct2's `num_flow_timesteps` (8) is a TRAINING knob — how many flow
    timesteps are sampled per example to build the loss — and is never read on
    an inference path. `predict_action_chunk` reads `num_inference_steps`."""
    from makermodslab.utils.system import policy_flow_steps_field

    assert policy_flow_steps_field("smolvla") == "num_steps"
    assert policy_flow_steps_field("pi0") == "num_inference_steps"
    assert policy_flow_steps_field("pi05") == "num_inference_steps"
    assert policy_flow_steps_field("molmoact2") == "num_inference_steps"
    # No denoising loop to shorten: ACT regresses a chunk in one pass, and
    # pi0_fast decodes action TOKENS autoregressively.
    assert policy_flow_steps_field("act") is None
    assert policy_flow_steps_field("pi0_fast") is None
    # Same `object` tolerance policy_requires_task has: the type is read
    # straight out of a config.json, where it can be missing or corrupt.
    assert policy_flow_steps_field(None) is None
    assert policy_flow_steps_field(7) is None


def test_the_flow_steps_default_is_read_off_the_checkpoint_or_is_unknown() -> None:
    """The checkpoint's own saved value, with ONE documented fallback: a
    MolmoAct2 that saved `num_inference_steps: null` (which the published one
    does) runs at 10, because `modeling_molmoact2.py` resolves
    `steps = int(num_steps or self.config.flow_matching_num_steps)` against the
    backbone config, whose default is 10. It is NOT 8 — `num_flow_timesteps` is
    a training knob."""
    from makermodslab.utils.system import MOLMOACT2_FLOW_STEPS_DEFAULT, policy_flow_steps_default

    assert MOLMOACT2_FLOW_STEPS_DEFAULT == 10
    assert policy_flow_steps_default({"type": "smolvla", "num_steps": 10}) == 10
    assert policy_flow_steps_default({"type": "pi05", "num_inference_steps": 4}) == 4
    assert policy_flow_steps_default({"type": "molmoact2", "num_inference_steps": None}) == 10
    # Absent reads the same as null — a config that never wrote the key is in
    # exactly the state the container's `or` fallback answers.
    assert policy_flow_steps_default({"type": "molmoact2"}) == 10
    # A saved value still wins over the fallback.
    assert policy_flow_steps_default({"type": "molmoact2", "num_inference_steps": 4}) == 4
    # The fallback is MolmoAct2's alone: pi05's `num_inference_steps` has a
    # class-level default this file cannot see, so null there stays unknown.
    assert policy_flow_steps_default({"type": "pi05", "num_inference_steps": None}) is None
    # No such knob at all.
    assert policy_flow_steps_default({"type": "act", "n_action_steps": 100}) is None
    # A hand-edited config: "unknown" beats a number somebody sets a latency
    # budget from. `True` is an `int` subclass and would otherwise read as 1.
    assert policy_flow_steps_default({"type": "smolvla", "num_steps": 0}) is None
    assert policy_flow_steps_default({"type": "smolvla", "num_steps": "ten"}) is None
    assert policy_flow_steps_default({"type": "smolvla", "num_steps": True}) is None
    # Hand-edited MolmoAct2 too: only null (or absent) takes the fallback.
    assert policy_flow_steps_default({"type": "molmoact2", "num_inference_steps": "ten"}) is None
    assert policy_flow_steps_default({}) is None


def test_model_dtype_support_is_answered_from_the_saved_config() -> None:
    """A config.json is a dataclass dump, so key presence IS "the class
    declares this field" — which stays right through a pin bump that gives
    another family the knob. In this pin MolmoAct2 is the only one."""
    from makermodslab.utils.system import policy_supports_model_dtype

    assert policy_supports_model_dtype({"type": "molmoact2", "model_dtype": "float32"}) is True
    assert policy_supports_model_dtype({"type": "smolvla", "num_steps": 10}) is False
    assert policy_supports_model_dtype({}) is False


def test_the_variable_view_allowlist_is_closed_and_small() -> None:
    """Which checkpoints may be given a camera they were not published with
    (S3.8g). Answered off a TABLE rather than off key presence the way
    `model_dtype` is, because nothing in a config.json says "this family's
    vision tower takes any number of pictures" — that is a fact about its
    processor, established by reading one.

    MolmoAct2 is in it because its lerobot wrapper FIXED at two a list the
    allenai model takes any length of (`processor_molmoact2._extract_images`
    iterates whatever keys it resolves; the prompt and the sequence budget are
    both computed from `len(images)`). Everything else answers False, INCLUDING
    a type this pin has never heard of — which is the safe direction: the
    checkpoint then runs with the views it was published with, and the operator
    gets a refusal instead of a shape error inside a paid container."""
    from makermodslab.utils.system import (
        VARIABLE_VIEW_POLICY_TYPES,
        policy_supports_extra_image_roles,
    )

    assert frozenset({"molmoact2"}) == VARIABLE_VIEW_POLICY_TYPES
    assert policy_supports_extra_image_roles("molmoact2") is True
    for policy_type in ("smolvla", "pi0", "pi05", "act", "pi0_fast", "", "something_new"):
        assert policy_supports_extra_image_roles(policy_type) is False, policy_type
    # Read off a config dict whose "type" can be anything at all.
    assert policy_supports_extra_image_roles(None) is False
    assert policy_supports_extra_image_roles(7) is False


def test_a_camera_role_must_survive_four_journeys() -> None:
    """The role becomes a policy feature key, a Portal VIDEO TRACK name, a
    `--robot.cameras` dict key inside a draccus-parsed argv, and one element of
    a COMMA-separated flag. A comma, a space, a brace or a dot breaks one of
    those in a place that presents as "the session receives nothing" rather than
    as an error — hence a rule narrower than "any string"."""
    from makermodslab.utils.system import is_valid_image_role

    for good in ("cam2", "c", "wrist_cam", "cam_0", "a" * 32):
        assert is_valid_image_role(good) is True, good
    for bad in (
        "",
        "Cam2",  # uppercase: the feature key is spelled lowercase everywhere
        "2cam",  # must start with a letter
        "cam 2",  # a space splits the draccus dict
        "cam,2",  # a comma splits the flag itself
        "cam.2",  # a dot is the feature-key separator
        "cam-2",
        "{cam2}",
        "a" * 33,
        None,
        7,
        ["cam2"],
    ):
        assert is_valid_image_role(bad) is False, bad


def test_molmoact2_device_warning_is_advisory_and_names_the_device() -> None:
    """A WARNING, not a gate: nothing in this pin requires CUDA (the action-flow
    CUDA graph falls back off-CUDA), so this only sets expectations about a ~7B
    VLM in a 30 Hz loop. The device is injected — no real detection here."""
    from makermodslab.utils.system import molmoact2_device_warning

    molmo = {"type": "molmoact2", "inference_action_mode": "continuous"}
    assert molmoact2_device_warning(molmo, "cuda") is None

    for device in ("mps", "cpu"):
        warning = molmoact2_device_warning(molmo, device)
        assert warning is not None
        assert device in warning
        assert "CUDA" in warning

    # Every other policy is unaffected — ACT on MPS is an ordinary, fast run.
    assert molmoact2_device_warning({"type": "act"}, "mps") is None
    assert molmoact2_device_warning({}, "cpu") is None


# ── the task vocabulary ──────────────────────────────────────────────────────


def test_policy_requires_task_covers_every_language_conditioned_type() -> None:
    """ONE vocabulary, two consumers: jobs.py's `requires_task` (what the launch
    panel gates its task field on) and the two DRTC policy servers' startup
    refusal. A type in one and not the other is a run that the panel lets
    through and the GPU then rejects, or worse, accepts and degrades."""
    from makermodslab.utils.system import LANGUAGE_CONDITIONED_POLICY_TYPES, policy_requires_task

    assert {"smolvla", "pi0", "pi0_fast", "pi05", "molmoact2"} == LANGUAGE_CONDITIONED_POLICY_TYPES
    for policy_type in LANGUAGE_CONDITIONED_POLICY_TYPES:
        assert policy_requires_task(policy_type) is True


def test_policy_requires_task_is_false_for_everything_else() -> None:
    """Including the values a config.json can legitimately hand it: a missing
    "type" key reads as None, and a corrupt file can hold anything at all. An
    unreadable type only makes the task field optional — a policy that really
    wanted one still says so from inside the subprocess."""
    from makermodslab.utils.system import policy_requires_task

    assert policy_requires_task("act") is False
    assert policy_requires_task("diffusion") is False
    assert policy_requires_task("") is False
    assert policy_requires_task(None) is False
    assert policy_requires_task(42) is False


def test_molmoact2_needs_a_task_which_is_why_it_joined_the_set() -> None:
    """MolmoAct2 is the reason this became a hard requirement rather than a
    nudge. Its processor renders a missing task as the empty string into a fixed
    template, so the VLM is prompted with the literal "The task is to ." and
    returns confidently wrong actions with nothing in any log to explain it."""
    from makermodslab.utils.system import MOLMOACT2, policy_requires_task

    assert policy_requires_task(MOLMOACT2) is True


def test_policy_extra_core_policy_needs_nothing() -> None:
    from makermodslab.utils.system import handle_get_policy_extra

    act = handle_get_policy_extra("act")
    assert act["needs_extra"] is False
    assert act["available"] is True
    assert act["install_target"] == ""


def test_policy_extra_available_reflects_find_spec(monkeypatch) -> None:
    import importlib.util

    from makermodslab.utils import system

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert system.handle_get_policy_extra("smolvla")["available"] is True
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert system.handle_get_policy_extra("smolvla")["available"] is False


def test_training_extra_available_flips_with_find_spec(monkeypatch) -> None:
    """Availability is probed live: it flips within one process when find_spec
    starts returning a spec — no server restart required after install."""
    import importlib.util

    from makermodslab.utils import system

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert system.handle_get_training_extra()["available"] is False
    # Simulate the package appearing (e.g. an install finished mid-process).
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert system.handle_get_training_extra()["available"] is True
    assert system.handle_get_training_extra()["install_hint"] == "pip install accelerate"


def test_wandb_extra_available_flips_with_find_spec(monkeypatch) -> None:
    import importlib.util

    from makermodslab.utils import system

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert system.handle_get_wandb_extra()["available"] is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert system.handle_get_wandb_extra()["available"] is True
    assert system.handle_get_wandb_extra()["install_hint"] == "pip install wandb"


def test_extra_available_swallows_find_spec_errors(monkeypatch) -> None:
    import importlib.util

    from makermodslab.utils import system

    def _boom(name: str):
        raise ValueError("bad module name")

    monkeypatch.setattr(importlib.util, "find_spec", _boom)
    assert system._extra_available("whatever") is False


def test_install_success_invalidates_import_caches(monkeypatch) -> None:
    """On a successful install the InstallManager must call
    importlib.invalidate_caches() so the next find_spec sees the new package."""
    import importlib

    from makermodslab.utils import system

    calls: list[int] = []
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: calls.append(1))

    mgr = system.InstallManager("some-package")

    class _FakeProcess:
        returncode = 0

        def __init__(self) -> None:
            self.stdout = iter([])

        def wait(self) -> None:
            return None

    mgr.process = _FakeProcess()
    mgr._monitor()

    assert calls == [1]
    assert mgr.get_status()["state"] == "done"


def test_install_failure_does_not_invalidate_caches(monkeypatch) -> None:
    import importlib

    from makermodslab.utils import system

    calls: list[int] = []
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: calls.append(1))

    mgr = system.InstallManager("some-package")

    class _FailProcess:
        returncode = 1

        def __init__(self) -> None:
            self.stdout = iter([])

        def wait(self) -> None:
            return None

    mgr.process = _FailProcess()
    mgr._monitor()

    assert calls == []
    assert mgr.get_status()["state"] == "error"


def test_policy_extra_install_is_noop_for_core_policy() -> None:
    from makermodslab.utils.system import handle_install_policy_extra, handle_install_policy_extra_status

    assert handle_install_policy_extra("act")["started"] is False
    assert handle_install_policy_extra_status("act")["state"] == "done"


def test_policy_extra_route_known_and_core(client) -> None:
    smol = client.get("/system/policy-extra/smolvla").json()
    assert smol["needs_extra"] is True
    assert smol["install_target"] == "lerobot[smolvla]"
    core = client.get("/system/policy-extra/act").json()
    assert core["needs_extra"] is False


# --- self-restart (POST /api/v1/system/restart) --------------------------------


def test_build_install_cmd_falls_back_to_known_uv_locations(monkeypatch, tmp_path) -> None:
    """A headless server started over ssh/nohup gets a PATH without
    ~/.local/bin — uv must still be found at its standard install target,
    because a uv venv has no pip to fall back to."""
    import shutil

    from makermodslab.utils import system

    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/sh\n")
    fake_uv.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(system, "_UV_FALLBACK_PATHS", (str(fake_uv),))
    cmd = system._build_install_cmd("lerobot[smolvla]")
    assert cmd[0] == str(fake_uv)
    assert "--python" in cmd and sys.executable in cmd


def test_install_in_progress_reports_the_live_package(monkeypatch) -> None:
    from makermodslab.utils import system

    assert system.install_in_progress() is None
    mgr = system.InstallManager("lerobot[smolvla]")
    mgr.state = "installing"
    monkeypatch.setitem(system._policy_install_managers, "lerobot[smolvla]", mgr)
    assert system.install_in_progress() == "lerobot[smolvla]"


def test_restart_supported_only_for_entry_point_argv(monkeypatch) -> None:
    """Only a launch we KNOW re-runs the launcher (argv[0] is one of our entry
    points) may execv; a dev reload worker or a bare `python -m` must not."""
    from makermodslab.utils import system

    monkeypatch.setattr(sys, "argv", ["/some/.venv/bin/makermodslab", "--lan"])
    supported, why = system.restart_supported()
    assert supported is True and why == ""

    monkeypatch.setattr(sys, "argv", ["/some/.venv/bin/makermodslab-station"])
    assert system.restart_supported()[0] is True

    monkeypatch.setattr(sys, "argv", ["-c"])  # a multiprocessing spawn worker
    supported, why = system.restart_supported()
    assert supported is False
    assert "entry point" in why


def test_restart_supported_posix_only(monkeypatch) -> None:
    from makermodslab.utils import system

    monkeypatch.setattr(system.os, "name", "nt")
    monkeypatch.setattr(sys, "argv", ["C:/venv/Scripts/makermodslab"])
    supported, why = system.restart_supported()
    assert supported is False
    assert "platform" in why


def test_schedule_restart_execs_same_argv(monkeypatch) -> None:
    """The re-exec must reproduce this exact process: same interpreter, same
    argv — that is the whole restart contract."""
    from makermodslab.utils import system

    monkeypatch.setattr(sys, "argv", ["/venv/bin/makermodslab", "--lan", "--bind", "tailscale0"])
    calls: list[tuple[str, list[str]]] = []
    thread = system.schedule_restart(delay_s=0, execv=lambda exe, argv: calls.append((exe, argv)))
    thread.join(timeout=5)
    assert calls == [
        (sys.executable, [sys.executable, "/venv/bin/makermodslab", "--lan", "--bind", "tailscale0"])
    ]


def test_restart_route_refuses_while_a_feature_holds_the_robot(client, monkeypatch) -> None:
    """Killing the server mid-flow drops the hardware threads with it, so the
    refusal carries the holder's own busy discriminant."""
    from makermodslab import server

    monkeypatch.setattr(server, "held_by", lambda: "teleoperation")
    resp = client.post("/api/v1/system/restart")
    assert resp.status_code == 409
    assert resp.json()["code"] == "robot.busy.teleoperation"


def test_restart_route_refuses_while_training_runs(client, monkeypatch) -> None:
    from makermodslab import server

    monkeypatch.setattr(server, "held_by", lambda: None)
    monkeypatch.setattr(server, "training_is_active", lambda: "act_so101_run")
    resp = client.post("/api/v1/system/restart")
    assert resp.status_code == 409
    assert resp.json()["code"] == "robot.busy.training"


def test_restart_route_refuses_while_an_install_runs(client, monkeypatch) -> None:
    """Re-exec would orphan the pip subprocess mid-write — refuse until it
    finishes."""
    from makermodslab import server

    monkeypatch.setattr(server, "held_by", lambda: None)
    monkeypatch.setattr(server, "training_is_active", lambda: None)
    monkeypatch.setattr(server.job_registry, "list_queue", lambda: [])
    monkeypatch.setattr(server, "install_in_progress", lambda: "lerobot[smolvla]")
    resp = client.post("/api/v1/system/restart")
    assert resp.status_code == 409
    assert resp.json()["code"] == "system.install_in_progress"
    assert "lerobot[smolvla]" in resp.json()["detail"]


def test_restart_route_refuses_with_a_queued_run(client, monkeypatch) -> None:
    """The loader retires queued records on startup — a restart would silently
    eat the queue, so it refuses instead."""
    from makermodslab import server

    monkeypatch.setattr(server, "held_by", lambda: None)
    monkeypatch.setattr(server, "training_is_active", lambda: None)
    monkeypatch.setattr(server.job_registry, "list_queue", lambda: [object()])
    resp = client.post("/api/v1/system/restart")
    assert resp.status_code == 409
    assert resp.json()["code"] == "robot.busy.training"
    assert "queued" in resp.json()["detail"]


def test_restart_route_refuses_when_unsupported(client, monkeypatch) -> None:
    """Under pytest argv[0] is not a launcher entry point, so the REAL
    restart_supported refuses — no monkeypatching the gate itself."""
    from makermodslab import server

    monkeypatch.setattr(server, "held_by", lambda: None)
    monkeypatch.setattr(server, "training_is_active", lambda: None)
    monkeypatch.setattr(server.job_registry, "list_queue", lambda: [])
    resp = client.post("/api/v1/system/restart")
    assert resp.status_code == 409
    assert resp.json()["code"] == "system.restart_unsupported"


def test_restart_route_answers_then_schedules(client, monkeypatch) -> None:
    from makermodslab import server

    monkeypatch.setattr(server, "held_by", lambda: None)
    monkeypatch.setattr(server, "training_is_active", lambda: None)
    monkeypatch.setattr(server.job_registry, "list_queue", lambda: [])
    monkeypatch.setattr(server, "restart_supported", lambda: (True, ""))
    scheduled: list[int] = []
    monkeypatch.setattr(server, "schedule_restart", lambda: scheduled.append(1))
    resp = client.post("/api/v1/system/restart")
    assert resp.status_code == 200
    assert resp.json()["restarting"] is True
    assert scheduled == [1]


def test_restart_route_is_v1_only(client) -> None:
    """New surface never lands on the frozen flat mount."""
    resp = client.post("/system/restart")
    assert resp.status_code in {404, 405}


def test_torchcodec_probe_caches_and_reports(monkeypatch):
    from makermodslab.utils import system as sysmod

    calls = []
    monkeypatch.setattr(sysmod, "_torchcodec_cache", sysmod._TORCHCODEC_UNPROBED)
    monkeypatch.setattr(sysmod, "_probe_torchcodec_uncached", lambda: calls.append(1) or False)
    assert sysmod.torchcodec_loads() is False
    assert sysmod.torchcodec_loads() is False
    assert len(calls) == 1


def test_torchcodec_probe_subprocess_failure_means_unusable(monkeypatch):
    from makermodslab.utils import system as sysmod

    def boom(*a, **k):
        raise OSError("no interpreter")

    monkeypatch.setattr(sysmod.subprocess, "run", boom)
    assert sysmod._probe_torchcodec_uncached() is False
