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
