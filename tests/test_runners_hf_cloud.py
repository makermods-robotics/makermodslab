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
"""Tests for makermodslab.runners.hf_cloud — covers the host-side wandb credential
resolution, the pinned-lerobot spec derivation, and the cloud-boundary config
localization. HfCloudJobRunner itself talks to HF Jobs and is not unit-
testable without a heavy mock of HfApi; we intentionally leave it for
integration tests."""

from __future__ import annotations

import netrc
import re
import tomllib
from pathlib import Path

import pytest


def test_resolve_wandb_api_key_prefers_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.setenv("WANDB_API_KEY", "env-key-123")
    assert resolve_wandb_api_key() == "env-key-123"


def test_resolve_wandb_api_key_falls_back_to_netrc(monkeypatch: pytest.MonkeyPatch) -> None:
    """When WANDB_API_KEY is unset, the function must read the same place
    `wandb login` writes — ~/.netrc under machine api.wandb.ai."""
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    class _FakeNetrc:
        def authenticators(self, host):
            assert host == "api.wandb.ai"
            return ("login", "account", "netrc-key-456")

    monkeypatch.setattr(netrc, "netrc", lambda: _FakeNetrc())
    assert resolve_wandb_api_key() == "netrc-key-456"


def test_resolve_wandb_api_key_returns_none_when_netrc_has_no_wandb_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    class _FakeNetrc:
        def authenticators(self, host):
            return None

    monkeypatch.setattr(netrc, "netrc", lambda: _FakeNetrc())
    assert resolve_wandb_api_key() is None


def test_resolve_wandb_api_key_returns_none_when_netrc_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env var, no ~/.netrc — neither source has it, caller decides."""
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    def _raise_missing():
        raise FileNotFoundError("~/.netrc")

    monkeypatch.setattr(netrc, "netrc", _raise_missing)
    assert resolve_wandb_api_key() is None


def test_resolve_wandb_api_key_returns_none_when_netrc_parse_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    def _raise_parse():
        raise netrc.NetrcParseError("bad netrc", "~/.netrc", 1)

    monkeypatch.setattr(netrc, "netrc", _raise_parse)
    assert resolve_wandb_api_key() is None


def test_resolve_wandb_api_key_returns_none_when_password_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty password from netrc is treated as missing — the helper
    contract is 'returns the usable key or None', not 'returns whatever
    netrc happened to have'."""
    from makermodslab.runners.hf_cloud import resolve_wandb_api_key

    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    class _FakeNetrc:
        def authenticators(self, host):
            return ("login", "account", "")

    monkeypatch.setattr(netrc, "netrc", lambda: _FakeNetrc())
    assert resolve_wandb_api_key() is None


# -- pinned-lerobot spec derivation (version-skew fix) --


def _pyproject_lerobot_pin() -> str:
    """The raw lerobot dependency line from this repo's pyproject.toml."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    return next(d for d in deps if d.startswith("lerobot"))


def _spec_extras(spec: str) -> set[str]:
    m = re.match(r"lerobot\[(?P<extras>[^\]]*)\]", spec)
    assert m, f"no extras block in {spec!r}"
    return set(m.group("extras").split(","))


def test_cloud_lerobot_spec_carries_the_pyproject_pinned_ref() -> None:
    """The container install spec must reference the exact ref pinned in
    pyproject.toml — never a hardcoded second copy, never :latest."""
    from makermodslab.runners.hf_cloud import cloud_lerobot_spec

    pin = _pyproject_lerobot_pin()
    ref = pin.rsplit("@", 1)[1]  # the sha at the end of git+https://…@<sha>
    spec = cloud_lerobot_spec("act")
    assert ref in spec
    assert "latest" not in spec


def test_cloud_lerobot_spec_uses_archive_tarball_not_git() -> None:
    """A GitHub git+ pin is rewritten to the source archive tarball so pip in
    the container can install it without a git binary."""
    from makermodslab.runners.hf_cloud import cloud_lerobot_spec

    spec = cloud_lerobot_spec("act")
    assert "git+" not in spec
    # 0.6.0 pins by tag (v0.6.0), not a hex SHA, so the archive ref may contain
    # non-hex chars (v, dots). Match any non-slash ref, not just [0-9a-f].
    assert re.search(r"@ https://github\.com/.+/archive/[^/]+\.tar\.gz$", spec)


def test_cloud_lerobot_spec_drops_host_only_extras_and_adds_policy_extra() -> None:
    from makermodslab.runners.hf_cloud import cloud_lerobot_spec

    act = _spec_extras(cloud_lerobot_spec("act"))
    assert "feetech" not in act  # serial motor bus: host-only
    assert "training" in act
    assert "core_scripts" in act  # provides lerobot_train

    smolvla = _spec_extras(cloud_lerobot_spec("smolvla"))
    assert smolvla == act | {"smolvla"}

    pi0_fast = _spec_extras(cloud_lerobot_spec("pi0_fast"))
    assert pi0_fast == act | {"pi"}


def test_cloud_lerobot_spec_falls_back_to_pyproject_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running from a source tree without installed MakerMods Lab metadata must still
    derive the pin — from pyproject.toml directly."""
    from makermodslab.runners import hf_cloud

    monkeypatch.setattr(hf_cloud, "requires", lambda name: None)
    pin = _pyproject_lerobot_pin()
    ref = pin.rsplit("@", 1)[1]
    assert ref in hf_cloud.cloud_lerobot_spec("act")


# -- cloud-boundary config localization (device-leak fix) --


def _request(**overrides):
    from makermodslab.train import TrainingRequest

    return TrainingRequest(dataset_repo_id="user/ds", **overrides)


def test_localize_forces_flavor_device_over_host_detection() -> None:
    """The host's auto-detected device (mps on a Mac) must never reach a cloud
    job: GPU flavors force cuda, cpu tiers force cpu."""
    from makermodslab.runners.hf_cloud import localize_config_for_cloud
    from makermodslab.train import build_training_command

    for host_device in ("auto", "mps", "cpu", None):
        config = _request(policy_device=host_device)
        localize_config_for_cloud(config, "t4-small")
        assert config.policy_device == "cuda"
        cmd = build_training_command(config, "/tmp/out")
        assert cmd[cmd.index("--policy.device") + 1] == "cuda"

    cpu_config = _request(policy_device="mps")
    localize_config_for_cloud(cpu_config, "cpu-upgrade")
    assert cpu_config.policy_device == "cpu"


def test_localize_clears_host_local_dataset_root() -> None:
    from makermodslab.runners.hf_cloud import localize_config_for_cloud
    from makermodslab.train import build_training_command

    config = _request(dataset_root="/Users/someone/.cache/huggingface/lerobot/user/ds")
    localize_config_for_cloud(config, "t4-small")
    assert config.dataset_root is None
    assert "--dataset.root" not in build_training_command(config, "/tmp/out")


def test_localize_rejects_resume_from_host_checkpoint() -> None:
    from makermodslab.runners.hf_cloud import localize_config_for_cloud

    config = _request(
        resume=True, config_path="/host/run/checkpoints/5000/pretrained_model/train_config.json"
    )
    with pytest.raises(ValueError, match="[Rr]esum"):
        localize_config_for_cloud(config, "t4-small")


def test_localize_allows_cloud_resume_from_hub() -> None:
    """A cloud resume signals via resume_from_hub_repo (the wrapper downloads the
    checkpoint from the Hub), not a host-local config_path — so localization must
    NOT reject it. The host config_path stays unset here; the runner sets the
    container path later."""
    from makermodslab.runners.hf_cloud import localize_config_for_cloud

    config = _request(resume=True, resume_from_hub_repo="user/parent-run", resume_from_hub_step="005000")
    localize_config_for_cloud(config, "t4-small")  # no raise
    assert config.policy_device == "cuda"


def test_localize_rejects_a_resume_that_names_no_hub_checkpoint() -> None:
    """MT42's invariant at the cloud submission boundary: `resume` with nothing
    to resume FROM would fall through build_training_command's fresh-run branch
    and start over at step 0 on rented hardware while every UI surface reported a
    continuation. Refuse instead — silence is the failure mode here."""
    from makermodslab.runners.hf_cloud import localize_config_for_cloud

    config = _request(resume=True)
    with pytest.raises(ValueError, match="step 0"):
        localize_config_for_cloud(config, "t4-small")


def test_localize_rejects_local_pretrained_path_but_allows_hub_id() -> None:
    from makermodslab.runners.hf_cloud import localize_config_for_cloud

    local = _request(policy_pretrained_path="/host/checkpoints/000500/pretrained_model")
    with pytest.raises(ValueError, match="[Ff]ine-tun"):
        localize_config_for_cloud(local, "t4-small")

    hub = _request(policy_pretrained_path="user/some-model")
    localize_config_for_cloud(hub, "t4-small")  # no raise
    assert hub.policy_pretrained_path == "user/some-model"


def test_localize_allows_a_step_suffixed_hub_pretrained_ref() -> None:
    """MT2, cloud half: fine-tuning from a SPECIFIC Hub step travels as the ref
    'repo@checkpoints/<step_dir>' — the wrapper materializes it pod-side, the
    same way cloud resume already downloads its parent checkpoint. It must reach
    the container verbatim, not be rejected as a host path and not be rewritten
    here (a host path would be meaningless on the pod)."""
    from makermodslab.runners.hf_cloud import localize_config_for_cloud

    config = _request(policy_pretrained_path="user/some-model@checkpoints/003000")
    localize_config_for_cloud(config, "t4-small")  # no raise
    assert config.policy_pretrained_path == "user/some-model@checkpoints/003000"


# -- in-container installer ladder (the image's uv venv ships no pip) --


def test_install_plan_prefers_uv() -> None:
    """The lerobot-gpu image's venv is uv-created (no pip module), so uv on
    PATH must win, with --python pinning the install into this interpreter."""
    from makermodslab.runners.hf_cloud import _install_plan

    label, cmds = _install_plan("lerobot @ url", "/venv/bin/python", "/usr/local/bin/uv", True, True)
    assert label == "uv"
    assert cmds == [
        [
            "/usr/local/bin/uv",
            "pip",
            "install",
            "--python",
            "/venv/bin/python",
            "--no-cache",
            "lerobot @ url",
        ]
    ]


def test_install_plan_falls_back_to_pip_without_uv() -> None:
    from makermodslab.runners.hf_cloud import _install_plan

    label, cmds = _install_plan("spec", "/py", None, True, True)
    assert label == "pip"
    assert cmds == [["/py", "-m", "pip", "install", "--no-cache-dir", "spec"]]


def test_install_plan_bootstraps_pip_via_ensurepip_as_last_resort() -> None:
    from makermodslab.runners.hf_cloud import _install_plan

    label, cmds = _install_plan("spec", "/py", None, False, True)
    assert label == "ensurepip+pip"
    assert cmds == [
        ["/py", "-m", "ensurepip", "--upgrade"],
        ["/py", "-m", "pip", "install", "--no-cache-dir", "spec"],
    ]


def test_install_plan_reports_no_installer() -> None:
    from makermodslab.runners.hf_cloud import _install_plan

    assert _install_plan("spec", "/py", None, False, False) == (None, [])


# -- checkpoint-completeness check (partial-upload race fix) --


def test_checkpoint_step_ready_false_when_no_last_link_exists(tmp_path: Path) -> None:
    """No checkpoints/last symlink at all — e.g. before the first checkpoint
    of a run has completed — must not be mistaken for readiness."""
    from makermodslab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    step_dir.mkdir()
    assert _checkpoint_step_ready(step_dir) is False


def test_checkpoint_step_ready_false_when_last_points_to_an_earlier_step(tmp_path: Path) -> None:
    """The exact race this fixes: lerobot writes pretrained_model/config.json
    (and creates training_state/) well before the step is actually complete.
    Only checkpoints/last advancing to (or past) this step is proof the whole
    step finished — a step dir existing on its own proves nothing."""
    from makermodslab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "010000"
    step_dir.mkdir()
    (tmp_path / "005000").mkdir()
    (tmp_path / "last").symlink_to("005000")
    assert _checkpoint_step_ready(step_dir) is False


def test_checkpoint_step_ready_true_when_last_points_to_this_step(tmp_path: Path) -> None:
    from makermodslab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    step_dir.mkdir()
    (tmp_path / "last").symlink_to("005000")
    assert _checkpoint_step_ready(step_dir) is True


def test_checkpoint_step_ready_true_when_last_has_advanced_past_this_step(tmp_path: Path) -> None:
    """<=, not ==: two checkpoints can complete inside one 15s poll window, and
    a step must not go permanently unuploaded just because the poller missed
    the instant `last` pointed at it exactly."""
    from makermodslab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    step_dir.mkdir()
    (tmp_path / "010000").mkdir()
    (tmp_path / "last").symlink_to("010000")
    assert _checkpoint_step_ready(step_dir) is True


def test_checkpoint_step_ready_false_when_last_is_not_a_symlink(tmp_path: Path) -> None:
    """A plain file or directory named `last` (not the symlink lerobot writes)
    must not be misread as a target step number."""
    from makermodslab.runners.hf_cloud import _checkpoint_step_ready

    step_dir = tmp_path / "005000"
    step_dir.mkdir()
    (tmp_path / "last").mkdir()
    assert _checkpoint_step_ready(step_dir) is False


def test_wrapper_source_inlines_the_tested_checkpoint_ready_check() -> None:
    """The wrapper's checkpoint-completeness check is _checkpoint_step_ready's
    source inlined verbatim, so the in-container upload gate is exactly the
    function the tests above exercise — and _scan_and_upload must call it
    instead of checking config.json directly."""
    import inspect

    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE, _checkpoint_step_ready

    assert inspect.getsource(_checkpoint_step_ready) in WRAPPER_SOURCE
    assert "__CHECKPOINT_READY_SOURCE__" not in WRAPPER_SOURCE  # placeholder replaced
    assert "_checkpoint_step_ready(entry)" in WRAPPER_SOURCE


def test_lerobot_last_checkpoint_symlink_matches_our_readiness_check() -> None:
    """_checkpoint_step_ready's readiness signal is lerobot's own
    checkpoints/<LAST_CHECKPOINT_LINK> symlink, which lerobot_train.py points
    at a step directory via update_last_checkpoint() strictly after
    save_checkpoint() returns for that step. A lerobot pin bump that renames
    the link or reorders those two calls should fail here in CI, not ship a
    gate that silently never passes."""
    import inspect

    from lerobot.scripts import lerobot_train
    from lerobot.utils.constants import LAST_CHECKPOINT_LINK

    assert LAST_CHECKPOINT_LINK == "last"
    src = inspect.getsource(lerobot_train)
    assert src.index("save_checkpoint(") < src.index("update_last_checkpoint(checkpoint_dir)")


# -- wrapper sanity --


def test_wrapper_source_compiles_and_launches_an_argv_list() -> None:
    """The wrapper must pass the trainer argv to Popen as a LIST (splitting a
    joined string was the bug-3 hypothesis — it is not the case and must stay
    that way) and quote its log line so spaced values read unambiguously."""
    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE

    compile(WRAPPER_SOURCE, "<hf-jobs-wrapper>", "exec")  # syntactically valid
    assert "subprocess.Popen(list(trainer_argv)" in WRAPPER_SOURCE
    assert "shlex.join(trainer_argv)" in WRAPPER_SOURCE
    assert re.search(r"^import .*\bshlex\b", WRAPPER_SOURCE, re.MULTILINE)  # imported up top


def test_wrapper_source_handles_resume_download() -> None:
    """Cloud resume: the wrapper must parse --resume-from, download the parent
    checkpoint tree, refuse when the downloaded step is incomplete, and
    pre-seed `seen` so it never re-uploads the checkpoint it just pulled
    down."""
    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE

    compile(WRAPPER_SOURCE, "<hf-jobs-wrapper>", "exec")  # still valid with the resume block
    assert "--resume-from=" in WRAPPER_SOURCE
    assert "snapshot_download" in WRAPPER_SOURCE
    assert "seen.add(step_dir)" in WRAPPER_SOURCE


def test_wrapper_source_resume_checks_weights_and_training_state_not_just_a_bare_dir() -> None:
    """C4: a plain `training_state/` is_dir() check would pass a checkpoint
    that was itself only partially uploaded before this fix existed — the
    checkpoints/last symlink used by the live watcher isn't available here
    (it's never pushed to the Hub), so the resume path checks the files it
    actually needs directly instead of trusting the directory's existence."""
    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE

    assert 'any((dest / "pretrained_model").glob("*.safetensors"))' in WRAPPER_SOURCE
    assert '(dest / "training_state" / "training_step.json").is_file()' in WRAPPER_SOURCE
    assert '(dest / "training_state" / "rng_state.safetensors").is_file()' in WRAPPER_SOURCE


def test_wrapper_source_materializes_a_step_suffixed_finetune_base() -> None:
    """MT2, container half: a --policy.pretrained_path naming a Hub STEP is
    downloaded pod-side and rewritten to that local dir before the trainer runs.

    Two properties the block must keep:
      * it pulls ONLY that step's pretrained_model/ (weights-only — fine-tuning
        needs no training_state/), and
      * it uses the snapshot cache rather than <output_dir>/checkpoints/, which
        the uploader watches — a base checkpoint copied there would be
        republished as if this run had produced it.

    Source-level assertions: the block is top-level wrapper code (like the
    resume download it mirrors), so it has no import seam to exec against. The
    argv rewrite it depends on IS unit-tested below."""
    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE

    compile(WRAPPER_SOURCE, "<hf-jobs-wrapper>", "exec")
    assert '_arg("--policy.pretrained_path")' in WRAPPER_SOURCE
    assert '_set_arg("--policy.pretrained_path", str(base_dir))' in WRAPPER_SOURCE
    assert 'allow_patterns=[f"checkpoints/{step_dir}/pretrained_model/*"]' in WRAPPER_SOURCE
    # Never staged under the watched output dir.
    assert 'base_dir = Path(local_root) / "checkpoints" / step_dir / "pretrained_model"' in WRAPPER_SOURCE


def _wrapper_argv_helpers(trainer_argv: list[str]):
    """Exec the wrapper's own `_arg` / `_set_arg` over `trainer_argv`.

    Sliced out of WRAPPER_SOURCE by name and given the globals the wrapper would
    have, so a drift between the template and these tests fails loudly instead
    of silently testing a host-side paraphrase."""
    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE

    namespace: dict = {"trainer_argv": trainer_argv}
    for name in ("_arg", "_set_arg"):
        match = re.search(rf"^def {name}\(.*?(?=^\S)", WRAPPER_SOURCE, re.MULTILINE | re.DOTALL)
        assert match, f"{name} not found in WRAPPER_SOURCE"
        exec(compile(match.group(0), "<hf-jobs-wrapper>", "exec"), namespace)  # noqa: S102
    return namespace["_arg"], namespace["_set_arg"]


def test_wrapper_set_arg_rewrites_both_argv_spellings() -> None:
    """The rewrite must hit whichever form the argv builder used, and touch
    nothing else — this is what turns a Hub ref only the pod can resolve into a
    real path for the trainer."""
    joined = ["--policy.type=act", "--policy.pretrained_path=user/repo@checkpoints/003000", "--steps=10"]
    arg, set_arg = _wrapper_argv_helpers(joined)
    assert arg("--policy.pretrained_path") == "user/repo@checkpoints/003000"
    assert set_arg("--policy.pretrained_path", "/tmp/base") is True
    assert joined == ["--policy.type=act", "--policy.pretrained_path=/tmp/base", "--steps=10"]

    split = ["--policy.pretrained_path", "user/repo@checkpoints/003000", "--steps", "10"]
    arg, set_arg = _wrapper_argv_helpers(split)
    assert arg("--policy.pretrained_path") == "user/repo@checkpoints/003000"
    assert set_arg("--policy.pretrained_path", "/tmp/base") is True
    assert split == ["--policy.pretrained_path", "/tmp/base", "--steps", "10"]


def test_wrapper_set_arg_reports_a_missing_flag() -> None:
    """A bare-repo-id fine-tune (or no fine-tune at all) leaves the argv alone;
    the wrapper treats an unexpected miss as a hard error rather than launching
    a run that trains from the wrong weights."""
    argv = ["--policy.type=act"]
    arg, set_arg = _wrapper_argv_helpers(argv)
    assert arg("--policy.pretrained_path") is None
    assert set_arg("--policy.pretrained_path", "/tmp/base") is False
    assert argv == ["--policy.type=act"]


def test_cloud_resume_argv_keeps_lineage_in_parent_repo() -> None:
    """A cloud-resume config resolves to a --config_path at the container path and
    pushes into the parent's repo (same lineage), with resume essentials only."""
    from makermodslab.train import TrainingRequest, build_training_command

    req = TrainingRequest(
        dataset_repo_id="user/ds",
        resume=True,
        steps=20000,
        policy_push_to_hub=True,
        policy_repo_id="user/parent-run",
        config_path="/tmp/makermodslab/train/checkpoints/005000/pretrained_model/train_config.json",
    )
    cmd = build_training_command(req, output_dir="/tmp/makermodslab/train")
    assert (
        "--config_path=/tmp/makermodslab/train/checkpoints/005000/pretrained_model/train_config.json" in cmd
    )
    assert cmd[cmd.index("--policy.push_to_hub") + 1] == "true"
    assert cmd[cmd.index("--policy.repo_id") + 1] == "user/parent-run"
    assert cmd[cmd.index("--resume") + 1] == "true"
    # Inherited from the checkpoint — never re-passed on resume.
    assert "--dataset.repo_id" not in cmd
    assert "--policy.type" not in cmd


def _submitted_command(config, tmp_path, monkeypatch, job_id: str = "child_run"):
    """Drive HfCloudJobRunner.start with the Hub stubbed out; return the argv it
    submitted. No token, no network, no job — the run_job stand-in records and
    the worker threads are stubbed off."""
    from unittest.mock import MagicMock

    from makermodslab.jobs import TrainingMetrics
    from makermodslab.runners.hf_cloud import HfCloudJobRunner

    monkeypatch.setattr("makermodslab.runners.hf_cloud.get_token", lambda: "hf_fake")
    monkeypatch.setattr("makermodslab.runners.hf_cloud.cached_whoami", lambda: {"name": "alice"})
    runner = HfCloudJobRunner(TrainingMetrics(), tmp_path / "log.jsonl", "t4-small")
    api = MagicMock()
    api.run_job.return_value = MagicMock(id="hfjob-1", url="https://hf/jobs/1")
    runner._api = api
    monkeypatch.setattr(runner, "_ensure_dataset_on_hub", lambda repo_id: None)
    monkeypatch.setattr(runner, "_start_worker_threads", lambda label: None)
    runner.start(job_id, config, "/host/out")
    return api.run_job.call_args.kwargs["command"]


def test_cloud_resume_from_a_cloud_parent_publishes_into_the_parents_repo(tmp_path, monkeypatch) -> None:
    """Unchanged behaviour, pinned: a cloud→cloud continuation keeps the whole
    lineage in one repo."""
    config = _request(resume=True, resume_from_hub_repo="user/parent-run", resume_from_hub_step="005000")
    command = _submitted_command(config, tmp_path, monkeypatch)

    assert config.policy_repo_id == "user/parent-run"
    assert "--resume-from=user/parent-run@checkpoints/005000" in command


def test_cloud_resume_from_an_uploaded_local_checkpoint_gets_its_own_repo(tmp_path, monkeypatch) -> None:
    """F7, local→cloud: the source repo is a private STAGING repo holding the
    local parent's uploaded checkpoint, not an output repo. Publishing into it
    would put parent and child checkpoints in one tree again, so the run takes
    its own repo — while still resuming from the staged step."""
    config = _request(
        resume=True,
        resume_from_hub_repo="alice/src_checkpoints",
        resume_from_hub_step="000100",
        resume_from_uploaded_checkpoint=True,
    )
    command = _submitted_command(config, tmp_path, monkeypatch)

    assert config.policy_repo_id == "alice/child_run"
    assert "--resume-from=alice/src_checkpoints@checkpoints/000100" in command
    assert "--policy.repo_id" in command
    assert command[command.index("--policy.repo_id") + 1] == "alice/child_run"


def test_wrapper_source_inlines_the_tested_install_plan() -> None:
    """The wrapper's installer choice is _install_plan's source inlined
    verbatim, so the in-container code is exactly what the unit tests above
    exercised — uv first (shutil.which), pip / ensurepip as fallbacks."""
    import inspect

    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE, _install_plan

    assert inspect.getsource(_install_plan) in WRAPPER_SOURCE
    assert "__INSTALL_PLAN_SOURCE__" not in WRAPPER_SOURCE  # placeholder replaced
    assert 'shutil.which("uv")' in WRAPPER_SOURCE
    assert "no uv, pip, or ensurepip" in WRAPPER_SOURCE  # clear terminal message


# ---------------------------------------------------------------------------
# Checkpoint uploader hardening. The readiness GATE is covered above (it is
# _checkpoint_step_ready, exercised directly); what follows covers the two
# things the upload call itself must get right, by exec'ing the wrapper's own
# _scan_and_upload against fakes rather than a host-side paraphrase.
# ---------------------------------------------------------------------------


class _FakeUploadApi:
    """Records upload_folder calls; optionally fails the first `fail_times`."""

    def __init__(self, fail_times: int = 0) -> None:
        self.calls: list[dict] = []
        self._fail_times = fail_times

    def upload_folder(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("hub is having a day")


def _wrapper_scanner(output_dir: Path, api: _FakeUploadApi):
    """Exec the wrapper's own `_scan_and_upload` and return (call, seen).

    The function is sliced out of WRAPPER_SOURCE by name and given the globals
    the wrapper would have around it, so a drift between the template and this
    test surfaces as a KeyError/NameError rather than passing silently.
    """
    import os

    from makermodslab.runners.hf_cloud import WRAPPER_SOURCE, _checkpoint_step_ready

    match = re.search(r"^def _scan_and_upload\(.*?(?=^\S)", WRAPPER_SOURCE, re.MULTILINE | re.DOTALL)
    assert match, "_scan_and_upload not found in WRAPPER_SOURCE"
    namespace: dict = {
        "Path": Path,
        "os": os,
        "re": re,
        "api": api,
        "output_dir": str(output_dir),
        "repo_id": "user/run",
        "seen": set(),
        "waits": {},
        "_checkpoint_step_ready": _checkpoint_step_ready,
        "print": lambda *a, **k: None,  # keep the wrapper's logging out of pytest
    }
    exec(compile(match.group(0), "<hf-jobs-wrapper>", "exec"), namespace)  # noqa: S102
    return namespace["_scan_and_upload"], namespace["seen"]


def _write_checkpoint(output_dir: Path, step_dir: str) -> Path:
    """A complete lerobot checkpoint tree, with `checkpoints/last` advanced to
    it — the signal _checkpoint_step_ready reads to call the save finished."""
    import os

    ck = output_dir / "checkpoints" / step_dir
    pm = ck / "pretrained_model"
    pm.mkdir(parents=True, exist_ok=True)
    (pm / "config.json").write_text("{}")
    (pm / "model.safetensors").write_bytes(b"weights")
    (pm / "train_config.json").write_text("{}")
    ts = ck / "training_state"
    ts.mkdir(exist_ok=True)
    (ts / "training_step.json").write_text("{}")
    (ts / "rng_state.safetensors").write_bytes(b"rng")
    (ts / "optimizer_state.safetensors").write_bytes(b"optim")
    last = output_dir / "checkpoints" / "last"
    if last.is_symlink() or last.exists():
        last.unlink()
    os.symlink(step_dir, last)
    return ck


def test_wrapper_does_not_seal_a_checkpoint_whose_upload_failed(tmp_path: Path) -> None:
    """`seen.add` runs only after upload_folder returns, so a partial or failed
    commit must leave the step retryable instead of retiring it forever."""
    api = _FakeUploadApi(fail_times=1)
    scan, seen = _wrapper_scanner(tmp_path, api)

    _write_checkpoint(tmp_path, "002000")
    scan()  # attempts the upload, which raises
    assert len(api.calls) == 1
    assert seen == set()  # not retired

    scan()  # retried on the next tick, and this time it lands
    assert len(api.calls) == 2
    assert seen == {"002000"}


def test_wrapper_upload_excludes_safetensors_temp_files(tmp_path: Path) -> None:
    """A .tmp* file caught mid-rename has landed on the Hub before; keep it out
    of the commit even when the rest of the tree is complete."""
    api = _FakeUploadApi()
    scan, _seen = _wrapper_scanner(tmp_path, api)

    _write_checkpoint(tmp_path, "003000")
    scan()
    assert api.calls[0]["ignore_patterns"] == [".tmp*", "**/.tmp*"]


# ---------------------------------------------------------------------------
# Job-timeout precedence: request value wins (normalised to seconds), else the
# HF_JOB_TIMEOUT fallback constant.
# ---------------------------------------------------------------------------


def test_resolve_job_timeout_falls_back_to_constant_when_unset() -> None:
    from makermodslab.runners.hf_cloud import HF_JOB_TIMEOUT, resolve_job_timeout
    from makermodslab.train import TrainingRequest

    config = TrainingRequest(dataset_repo_id="x")
    assert config.hf_job_timeout is None
    assert resolve_job_timeout(config) == HF_JOB_TIMEOUT  # string passthrough, not seconds


def test_hf_job_timeout_constant_is_single_unit_and_covers_measured_runs() -> None:
    """The fallback is handed to run_job as a raw string, and run_job's parser is
    only `float(timeout[:-1]) * factor[timeout[-1]]` — a single unit suffix. A
    compound "improvement" like "1d12h" would not survive that, so pin the shape
    as well as the budget: it must clear the longest run we have measured
    (SmolVLA 15k steps at 2.24 s/step on a10g-small ≈ 8.8h)."""
    from makermodslab.runners.hf_cloud import HF_JOB_TIMEOUT

    assert isinstance(HF_JOB_TIMEOUT, str)
    factors = {"s": 1, "m": 60, "h": 3600, "d": 3600 * 24}  # run_job's own table
    assert HF_JOB_TIMEOUT[-1] in factors
    seconds = float(HF_JOB_TIMEOUT[:-1]) * factors[HF_JOB_TIMEOUT[-1]]  # no ValueError
    assert seconds == 24 * 3600
    assert seconds > 8.8 * 3600


def test_resolve_job_timeout_uses_request_value_normalised_to_seconds() -> None:
    """An explicit request value wins over the constant and is converted to an
    int of seconds — run_job's own str parser only handles a single unit, so
    compound forms like "3h30m" must be pre-resolved here."""
    from makermodslab.runners.hf_cloud import resolve_job_timeout
    from makermodslab.train import TrainingRequest

    assert resolve_job_timeout(TrainingRequest(dataset_repo_id="x", hf_job_timeout="45m")) == 2700
    assert resolve_job_timeout(TrainingRequest(dataset_repo_id="x", hf_job_timeout="3h30m")) == 12600
    assert resolve_job_timeout(TrainingRequest(dataset_repo_id="x", hf_job_timeout="2h")) == 7200


# --- stop(): distinguishing a cancel from a crash ---------------------------
#
# returncode() collapses every non-COMPLETED stage to 1, so the registry
# classifies cloud runs on terminal_stage() instead. These cover stop()'s own
# decisions only — no submission, no threads, no network — because the stage
# stop() leaves behind is what decides whether a stopped run reads as
# `interrupted` or as a failed model.


class _FakeStatus:
    def __init__(self, stage, message=None):
        self.stage = stage
        self.message = message


class _FakeJobInfo:
    def __init__(self, stage, message=None):
        self.status = _FakeStatus(stage, message)


class _FakeJobsApi:
    """Just the two calls stop() makes."""

    def __init__(self, *, cancel_raises=False, inspect=None, inspect_raises=False):
        self._cancel_raises = cancel_raises
        self._inspect = inspect
        self._inspect_raises = inspect_raises
        self.cancelled = []
        self.inspected = []

    def cancel_job(self, job_id):
        self.cancelled.append(job_id)
        if self._cancel_raises:
            raise RuntimeError("404 job already ended")

    def inspect_job(self, job_id):
        self.inspected.append(job_id)
        if self._inspect_raises:
            raise RuntimeError("network down")
        return self._inspect


def _runner_with(api, tmp_path, *, stage=None):
    from makermodslab.jobs import TrainingMetrics
    from makermodslab.runners.hf_cloud import HfCloudJobRunner

    runner = HfCloudJobRunner(TrainingMetrics(), tmp_path / "log.jsonl", "a10g-small")
    runner._api = api
    runner._hf_job_id = "job-1"
    runner._terminal_status = stage
    return runner


def test_stop_records_canceled_stage(tmp_path) -> None:
    api = _FakeJobsApi()
    runner = _runner_with(api, tmp_path)

    runner.stop()

    assert api.cancelled == ["job-1"]
    assert runner.terminal_stage() == "CANCELED"
    assert runner.is_running() is False
    # No corrective lookup needed when the cancel was accepted.
    assert api.inspected == []


def test_stop_does_not_overwrite_a_stage_the_poller_already_saw(tmp_path) -> None:
    """_set_terminal is idempotent, and that is what keeps a run which
    finished before the stop landed reported as COMPLETED."""
    api = _FakeJobsApi()
    runner = _runner_with(api, tmp_path, stage="COMPLETED")

    runner.stop()

    assert runner.terminal_stage() == "COMPLETED"
    assert runner.returncode() == 0


def test_stop_adopts_the_real_stage_when_cancel_is_refused(tmp_path) -> None:
    """cancel_job refusing usually means the job had ALREADY ended, so the
    pre-set CANCELED describes a run that finished on its own. Re-read it."""
    api = _FakeJobsApi(cancel_raises=True, inspect=_FakeJobInfo("COMPLETED"))
    runner = _runner_with(api, tmp_path)

    runner.stop()

    assert api.inspected == ["job-1"]
    assert runner.terminal_stage() == "COMPLETED"
    assert runner.returncode() == 0


def test_stop_adopts_an_error_stage_and_its_message(tmp_path) -> None:
    api = _FakeJobsApi(cancel_raises=True, inspect=_FakeJobInfo("ERROR", "Job timeout"))
    runner = _runner_with(api, tmp_path)

    runner.stop()

    assert runner.terminal_stage() == "ERROR"
    assert runner.terminal_message() == "Job timeout"


def test_stop_keeps_canceled_when_the_stage_cannot_be_re_read(tmp_path) -> None:
    """An unreachable Hub leaves CANCELED standing: our cancel is already out,
    so it's the best available account of the run."""
    api = _FakeJobsApi(cancel_raises=True, inspect_raises=True)
    runner = _runner_with(api, tmp_path)

    runner.stop()

    assert runner.terminal_stage() == "CANCELED"


def test_stop_keeps_canceled_when_the_job_is_still_running(tmp_path) -> None:
    """cancel_job can also fail transiently. A non-terminal stage is no
    evidence that the run ended on its own, so don't adopt it."""
    api = _FakeJobsApi(cancel_raises=True, inspect=_FakeJobInfo("RUNNING"))
    runner = _runner_with(api, tmp_path)

    runner.stop()

    assert runner.terminal_stage() == "CANCELED"


def test_stop_is_a_noop_before_submission(tmp_path) -> None:
    api = _FakeJobsApi()
    runner = _runner_with(api, tmp_path)
    runner._hf_job_id = None

    runner.stop()

    assert api.cancelled == []
    assert runner.terminal_stage() is None


# ---------------------------------------------------------------------------
# MT47: the log tail must not go permanently silent mid-run.
# ---------------------------------------------------------------------------


def test_is_replayed_skips_a_repeated_line_but_not_a_new_one(tmp_path) -> None:
    """The replay de-dupe, as a pure function. Content-based, so it is correct
    whether a reconnect replays the whole log, part of it, or none of it — the
    three cases the old positional counter could not tell apart."""
    from unittest.mock import MagicMock

    runner = _runner_with(MagicMock(), tmp_path)

    assert runner._is_replayed("first") is False
    assert runner._is_replayed("second") is False
    # The reconnect replays both, then delivers something genuinely new.
    assert runner._is_replayed("first") is True
    assert runner._is_replayed("second") is True
    assert runner._is_replayed("third") is False


def test_is_replayed_never_forgets_so_a_long_replay_cannot_thrash_it(tmp_path) -> None:
    """The de-dupe must not evict. The first version used an LRU window, and a
    replay LONGER than that window thrashed it end to end: the oldest line had
    aged out, so it read as novel, and accepting it evicted the next one, and so
    on through the whole history — every stale line accepted (review of PR #71).

    Never forgetting is what makes the length of a replay irrelevant.
    """
    from unittest.mock import MagicMock

    runner = _runner_with(MagicMock(), tmp_path)

    history = [f"line-{i}" for i in range(1001)]
    for line in history:
        assert runner._is_replayed(line) is False

    # A full replay of a history longer than the old 1,000-line window: every
    # single line must still be recognised, including the very first.
    assert [runner._is_replayed(line) for line in history] == [True] * len(history)


def test_is_replayed_stops_learning_at_the_cap_rather_than_forgetting(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap is a memory backstop, not a window: past it we stop remembering
    NEW lines instead of dropping old ones, so the early history a from-zero
    replay starts with stays recognised forever."""
    from unittest.mock import MagicMock

    from makermodslab.runners import hf_cloud

    monkeypatch.setattr(hf_cloud, "_TAIL_DEDUPE_MAX_LINES", 3)
    runner = _runner_with(MagicMock(), tmp_path)

    for line in ("a", "b", "c"):
        assert runner._is_replayed(line) is False
    assert len(runner._emitted_hashes) == 3

    # Past the cap: "d" is emitted and NOT remembered...
    assert runner._is_replayed("d") is False
    assert runner._is_replayed("d") is False  # so it can repeat — cosmetic
    # ...but nothing already learned was forgotten, which is the property that
    # matters: a replay still starts by hitting lines we recognise.
    assert runner._is_replayed("a") is True
    assert len(runner._emitted_hashes) == 3


def _drain(runner) -> list[str]:
    from queue import Empty as _Empty

    out = []
    while True:
        try:
            out.append(runner._log_queue.get_nowait().message)
        except _Empty:
            return out


def _run_tail_briefly(runner, monkeypatch, *, silence=30.0, seconds=1.0) -> list[str]:
    """Drive _tail_loop on a thread, then stop it from THIS thread.

    The stop always comes from the test, never from inside the fake stream: the
    loop consumes its reader through a queue, so a fake that set the event
    mid-yield would race the consumer and make the assertion meaningless."""
    import threading
    import time

    from makermodslab.runners import hf_cloud

    monkeypatch.setattr(hf_cloud, "_TAIL_CLEAN_END_WAIT_S", 0.01)
    monkeypatch.setattr(hf_cloud, "_TAIL_RECONNECT_BACKOFF_S", 0.01)
    monkeypatch.setattr(hf_cloud, "_TAIL_SILENCE_TIMEOUT_S", silence)

    thread = threading.Thread(target=runner._tail_loop, daemon=True)
    thread.start()
    time.sleep(seconds)
    lines = _drain(runner)
    runner._stop_event.set()
    thread.join(timeout=5)
    return lines


def test_tail_loop_keeps_delivering_after_a_reconnect_replays_less_than_it_had(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MT47, the exact regression. A reconnect that replays FEWER lines than
    were already processed used to leave a per-connection counter permanently
    behind a cross-connection total, so every later line was skipped — the job
    ran to completion while progress and charts froze at the drop."""
    import time
    from unittest.mock import MagicMock

    calls = {"n": 0}
    api = MagicMock()

    def fetch_job_logs(job_id, follow):
        calls["n"] += 1
        if calls["n"] == 1:
            yield from [f"L{i}" for i in range(1, 7)]
            raise RuntimeError("SSE dropped mid-run")
        # HF replays only a short tail, then the run continues.
        yield from ["L5", "L6", "L7", "L8", "L9"]
        while True:
            time.sleep(0.02)

    api.fetch_job_logs = fetch_job_logs
    runner = _runner_with(api, tmp_path)

    lines = _run_tail_briefly(runner, monkeypatch)

    assert calls["n"] == 2, "expected exactly one reconnect"
    # The new lines arrive...
    assert [x for x in lines if x in ("L7", "L8", "L9")] == ["L7", "L8", "L9"]
    # ...and the replayed prefix is not teed a second time.
    assert lines == ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]


def test_tail_loop_reconnects_when_a_connection_stops_yielding(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of MT47. `fetch_job_logs(follow=True)` can block inside a
    single read forever — no line, no StopIteration, no exception — and a plain
    `for` over it cannot even observe _stop_event, because the loop body never
    runs again. The silence timeout is what notices."""
    import time
    from unittest.mock import MagicMock

    calls = {"n": 0}
    api = MagicMock()

    def fetch_job_logs(job_id, follow):
        calls["n"] += 1
        if calls["n"] == 1:
            yield "L1"
            while True:  # half-open socket
                time.sleep(0.02)
        else:
            yield "L2"  # the reconnect recovers the stream
            while True:
                time.sleep(0.02)

    api.fetch_job_logs = fetch_job_logs
    runner = _runner_with(api, tmp_path)

    lines = _run_tail_briefly(runner, monkeypatch, silence=0.3, seconds=2.0)

    assert calls["n"] >= 2, "a silent connection must be abandoned and retried"
    assert lines == ["L1", "L2"]


def test_tail_loop_does_not_reconnect_while_a_connection_is_still_talking(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The silence timer must measure silence, not connection age — otherwise a
    long run would be torn down and re-replayed on a fixed cadence."""
    import time
    from unittest.mock import MagicMock

    calls = {"n": 0}
    api = MagicMock()

    def fetch_job_logs(job_id, follow):
        calls["n"] += 1
        for i in range(40):
            yield f"L{i}"
            time.sleep(0.05)

    api.fetch_job_logs = fetch_job_logs
    runner = _runner_with(api, tmp_path)

    _run_tail_briefly(runner, monkeypatch, silence=0.3, seconds=1.0)

    assert calls["n"] == 1


def test_tail_loop_still_scrapes_the_wandb_url_after_the_rewrite(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rewrite moved the de-dupe but must not have dropped the W&B scrape
    that already rode this loop."""
    import time
    from unittest.mock import MagicMock

    api = MagicMock()

    def fetch_job_logs(job_id, follow):
        yield "Track this run --> https://wandb.ai/me/proj/runs/abc123"
        while True:
            time.sleep(0.02)

    api.fetch_job_logs = fetch_job_logs
    runner = _runner_with(api, tmp_path)

    _run_tail_briefly(runner, monkeypatch)

    assert runner.wandb_run_url() == "https://wandb.ai/me/proj/runs/abc123"


# ---------------------------------------------------------------------------
# The two defects the PR #71 review found, as reproductions.
# ---------------------------------------------------------------------------


def _live_sse_threads() -> int:
    import threading

    return sum(1 for t in threading.enumerate() if t.name.endswith("-sse") and t.is_alive())


def test_repeated_silence_timeouts_do_not_strand_reader_threads(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviewer repro 1: four silent intervals produced four concurrent live
    reader threads.

    `_stop_event` is deliberately NEVER set here. It is the other arm of
    `_finished()`, so setting it would retire the parked readers on its own and
    the test would pass with the `abandoned` mechanism deleted entirely — which
    is exactly what an earlier version of this test did. The tail loop is left
    running and only the connections are abandoned, so the ONLY thing that can
    retire these readers is the per-connection flag.
    """
    import threading
    import time
    from unittest.mock import MagicMock

    from makermodslab.runners import hf_cloud

    before = _live_sse_threads()
    released = threading.Event()
    api = MagicMock()

    def fetch_job_logs(job_id, follow):
        # Silent long enough to be abandoned, then the stream "recovers" — the
        # exact shape that used to leave a reader consuming forever.
        released.wait(timeout=10)
        while True:
            yield "recovered-line"
            time.sleep(0.01)

    api.fetch_job_logs = fetch_job_logs
    runner = _runner_with(api, tmp_path)

    monkeypatch.setattr(hf_cloud, "_TAIL_CLEAN_END_WAIT_S", 0.01)
    monkeypatch.setattr(hf_cloud, "_TAIL_RECONNECT_BACKOFF_S", 0.01)
    monkeypatch.setattr(hf_cloud, "_TAIL_SILENCE_TIMEOUT_S", 0.2)
    thread = threading.Thread(target=runner._tail_loop, daemon=True)
    thread.start()
    time.sleep(1.6)  # several connections opened, gone silent, been abandoned

    # Wake every parked reader while the tail loop is STILL RUNNING.
    released.set()
    deadline = time.time() + 5
    stranded = _live_sse_threads()
    while time.time() < deadline and stranded > before + 1:
        time.sleep(0.05)
        # Captured INSIDE the loop: the still-running tail loop opens a fresh
        # connection every backoff, so re-reading after the loop can catch one
        # that was born in the gap and fail on a count that was fine.
        stranded = _live_sse_threads()

    runner._stop_event.set()
    thread.join(timeout=5)

    # At most the one live connection the still-running loop legitimately holds.
    assert stranded <= before + 1, "abandoned readers must not accumulate"


def test_an_abandoned_reader_retains_a_bounded_backlog_and_delivers_nothing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviewer repro 1, the memory half: an abandoned reader whose stream
    recovers consumed and retained 50,000 unobserved lines against the old
    unbounded queue.

    The reader is released BEFORE it is abandoned and produces without pause, so
    it genuinely runs at a queue nobody drains — the bound is what stops it, not
    the abandon flag arriving first. An earlier version of this test released
    the reader only after abandoning it, so it produced exactly one line and
    would have passed with the cap set to a million.
    """
    import time
    from unittest.mock import MagicMock

    from makermodslab.runners import hf_cloud

    monkeypatch.setattr(hf_cloud, "_TAIL_READER_PUT_TIMEOUT_S", 0.05)
    produced = {"n": 0}
    api = MagicMock()

    def fetch_job_logs(job_id, follow):
        while True:
            produced["n"] += 1
            yield f"stale-{produced['n']}"

    api.fetch_job_logs = fetch_job_logs
    runner = _runner_with(api, tmp_path)

    gen = runner._iter_job_logs()
    next(gen)  # one line delivered; the reader now races ahead of the consumer
    time.sleep(0.4)  # …and fills the queue, because nobody is draining it
    filled = produced["n"]
    assert filled >= hf_cloud._TAIL_READER_QUEUE_MAX, (
        f"reader never reached the bound ({filled}) — the test isn't exercising it"
    )
    # THE bound assertion. Without it the test passes with an unbounded queue:
    # a reviewer set _TAIL_READER_QUEUE_MAX = 0 and the reader retained
    # 1,069,101 lines while every other assertion here still held, because they
    # only constrain what happens AFTER abandonment.
    assert filled <= hf_cloud._TAIL_READER_QUEUE_MAX + 5, (
        f"queue did not bound retention: {filled} lines held"
    )

    gen.close()  # abandon the connection
    time.sleep(0.4)
    after = produced["n"]

    # Retention is bounded by the queue, not by the stream: once abandoned the
    # reader stops within a put-timeout instead of running on.
    assert after - filled <= 5, f"abandoned reader kept consuming: {filled} -> {after}"
    assert runner._log_queue.qsize() == 0, "an abandoned connection delivers nothing onward"


def test_a_full_replay_of_the_whole_history_is_recognised_and_never_rewinds(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviewer repro 2, at the size that broke the ORIGINAL 1,000-line window.

    Named for what it is: 1,001 lines is longer than the window this replaced,
    not longer than the 200k cap. Past-cap behaviour is covered separately
    below, where the cap is actually lowered.
    """
    import time
    from unittest.mock import MagicMock

    total = 1001
    history = [
        f"Training:  {i * 100 // total}%|##| {i}/{total} [00:10<00:10,  1.00step/s]"
        for i in range(1, total + 1)
    ]
    calls = {"n": 0}
    api = MagicMock()

    def fetch_job_logs(job_id, follow):
        calls["n"] += 1
        if calls["n"] == 1:
            yield from history
            raise RuntimeError("SSE dropped after the whole history")
        yield from history  # the reconnect replays EVERYTHING from line 1
        while True:
            time.sleep(0.02)

    api.fetch_job_logs = fetch_job_logs
    runner = _runner_with(api, tmp_path)

    lines = _run_tail_briefly(runner, monkeypatch, seconds=2.0)

    assert calls["n"] == 2, "expected exactly one reconnect"
    assert lines == sorted(set(lines), key=history.index)
    assert lines == history[-len(lines) :]
    assert runner._metrics.current_step == total
    assert runner._metrics.total_steps == total


def test_past_the_dedupe_cap_duplicates_slip_through_but_progress_still_cannot_rewind(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap's real failure mode, with the cap actually lowered.

    Past it we stop LEARNING new lines, so a replay of those un-learned lines is
    re-delivered — the acknowledged cosmetic cost. What must still hold is that
    the monotonic guard, which remembers no lines at all, refuses to let any of
    those duplicates drag progress backwards.
    """
    import time
    from unittest.mock import MagicMock

    from makermodslab.runners import hf_cloud

    monkeypatch.setattr(hf_cloud, "_TAIL_DEDUPE_MAX_LINES", 3)

    total = 8
    history = [
        f"Training:  {i * 10}%|##| {i}/{total} [00:10<00:10,  1.00step/s]" for i in range(1, total + 1)
    ]
    calls = {"n": 0}
    api = MagicMock()

    def fetch_job_logs(job_id, follow):
        calls["n"] += 1
        if calls["n"] == 1:
            yield from history
            raise RuntimeError("SSE dropped")
        # A PREFIX replay, not the whole history: this is what makes the final
        # assertion falsifiable. Replaying everything ends at the max step no
        # matter what the guard does, so `current_step == total` would hold even
        # with the guard deleted. Ending the replay at step 4 means an
        # unguarded parser finishes at 4, and only the guard keeps it at 8.
        yield from history[:4]
        while True:
            time.sleep(0.02)

    api.fetch_job_logs = fetch_job_logs
    runner = _runner_with(api, tmp_path)

    lines = _run_tail_briefly(runner, monkeypatch, seconds=1.5)

    assert calls["n"] == 2
    # Only the first 3 lines were ever learned, so the 4th repeats — cosmetic,
    # and the honest cost of a bounded memory.
    assert len(lines) > total, "expected un-learned lines to be re-delivered past the cap"
    # The guarantee that survives it: progress never moved backwards.
    assert runner._metrics.current_step == total


def test_parse_metrics_into_ignores_a_replayed_frame() -> None:
    """The monotonic guard as the backstop that remembers no lines: a stale
    frame moves nothing — not the step, and not the ETA/loss/LR that would
    otherwise ride along with it."""
    from makermodslab.jobs import StepFloor, TrainingMetrics, parse_metrics_into

    m, floor = TrainingMetrics(), StepFloor()
    parse_metrics_into(
        "Training:  90%|#########| 900/1000 [10:00<01:00,  1.00step/s]"
        "INFO step:900 loss:0.10 grdn:0.5 lr:1.0e-05",
        m,
        None,
        floor,
    )
    assert (m.current_step, m.current_loss, m.eta_seconds) == (900, 0.10, 60)

    parse_metrics_into(
        "Training:  10%|#| 100/1000 [01:00<09:00,  1.00step/s]INFO step:100 loss:9.99 grdn:9.9 lr:9.9e-01",
        m,
        None,
        floor,
    )
    assert m.current_step == 900, "a replayed frame must not rewind progress"
    assert m.current_loss == 0.10, "nor drag an obsolete loss along with it"
    assert m.eta_seconds == 60, "nor an obsolete ETA"


def test_the_first_frame_may_correct_an_overstated_resume_seed_downward() -> None:
    """The other direction, and the contract the guard must NOT break.

    `_initial_metrics` seeds `current_step` from the step the REQUEST named, and
    documents it as "a floor, not a claim about progress: the parser still owns
    the value from the first tqdm frame onwards" — because the bar reflects the
    checkpoint lerobot ACTUALLY restored. Guarding against the seed instead of
    against accepted progress would freeze a run whose seed overstated the
    restore (10,000 seeded, 8,000 restored ⇒ nothing moves for 2,000 steps),
    which is the MT47 symptom by another route.
    """
    from makermodslab.jobs import StepFloor, TrainingMetrics, parse_metrics_into

    # Seeded at 10,000; lerobot really restored 8,000 (bar total 2,000 of
    # 10,000 ⇒ 10000 − 2000 + 0).
    m = TrainingMetrics(current_step=10000, total_steps=10000)
    floor = StepFloor()

    parse_metrics_into("Training:   0%|| 0/2000 [00:00<20:00,  1.00step/s]", m, 10000, floor)
    assert m.current_step == 8000, "the first real frame must own the value"

    # …and from then on the guard is live: a replayed frame is still refused.
    parse_metrics_into("Training:   0%|| 0/2000 [00:00<20:00,  1.00step/s]", m, 10000, floor)
    parse_metrics_into("Training:  50%|#####| 1000/2000 [10:00<10:00,  1.00step/s]", m, 10000, floor)
    assert m.current_step == 9000
    parse_metrics_into("Training:  10%|#| 200/2000 [02:00<18:00,  1.00step/s]", m, 10000, floor)
    assert m.current_step == 9000, "a replay after real progress is still rejected"


def test_a_keepalive_only_connection_still_times_out_and_reconnects(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blank frames are not output.

    A half-open SSE connection commonly keeps dribbling empty keepalive frames.
    While those reset the silence deadline, such a connection is immortal — it
    never delivers a log line and never times out — which would have left the
    very run that motivated MT47 uncovered: its log simply stopped, and if the
    transport was still emitting keepalives nothing would have noticed.
    """
    import time
    from unittest.mock import MagicMock

    calls = {"n": 0}
    api = MagicMock()

    def fetch_job_logs(job_id, follow):
        calls["n"] += 1
        if calls["n"] == 1:
            yield "Training:  10%|#| 100/1000 [00:10<01:30,  1.00step/s]"
            while True:  # …and from here on, only keepalives
                yield ""
                time.sleep(0.01)
        else:
            yield "recovered"
            while True:
                time.sleep(0.02)

    api.fetch_job_logs = fetch_job_logs
    runner = _runner_with(api, tmp_path)

    lines = _run_tail_briefly(runner, monkeypatch, silence=0.3, seconds=1.5)

    assert calls["n"] >= 2, "a keepalive-only connection must still be abandoned"
    assert "recovered" in lines


def test_a_stream_error_surfaces_even_when_the_queue_is_full(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The highest-severity finding of the first review, pinned.

    The terminator used to be a queue item dropped under `suppress(Full)`. If
    the stream ended or errored while the bounded queue was full, both the error
    and the end-sentinel vanished; the consumer drained the backlog and then sat
    out the FULL silence timeout before raising "no log output for 600s" —
    inventing a stall and misreporting the cause of a real, immediately
    knowable error.

    Deliberately explicit about both halves: the real error arrives, and it
    arrives without the backlog being truncated to make room for it.
    """
    import time
    from unittest.mock import MagicMock

    from makermodslab.runners import hf_cloud

    monkeypatch.setattr(hf_cloud, "_TAIL_READER_QUEUE_MAX", 2)
    monkeypatch.setattr(hf_cloud, "_TAIL_SILENCE_TIMEOUT_S", 30.0)
    api = MagicMock()

    def fetch_job_logs(job_id, follow):
        yield from ["a", "b", "c"]
        raise RuntimeError("stream blew up while the queue was full")

    api.fetch_job_logs = fetch_job_logs
    runner = _runner_with(api, tmp_path)

    gen = runner._iter_job_logs()
    time.sleep(0.3)  # reader fills the 2-slot queue and then errors

    started = time.monotonic()
    delivered = []
    with pytest.raises(RuntimeError, match="stream blew up"):
        for line in gen:
            delivered.append(line)
    elapsed = time.monotonic() - started

    assert delivered == ["a", "b", "c"], "lines must not be dropped to make room"
    assert elapsed < 5.0, f"error waited on the silence timeout ({elapsed:.1f}s)"


def test_a_batched_burst_cannot_walk_progress_backwards_within_one_line(
    tmp_path,
) -> None:
    """One SSE message commonly batches a burst spanning several log_freq
    boundaries, so the LAST tqdm frame and the FIRST `step:` token describe
    different moments.

    The floor is read fresh at each comparison for exactly this: reading it once
    at the top let the earlier INFO segment be judged against a pre-tqdm floor,
    accepted, and applied — walking `current_step` backwards inside a single
    line and pulling the floor down with it, which reopened the skipped range to
    any genuine replay the content de-dupe misses.
    """
    from makermodslab.jobs import StepFloor, TrainingMetrics, parse_metrics_into

    m, floor = TrainingMetrics(), StepFloor()
    batched = (
        "Training:  25%|##  | 250/1000 [02:30<07:30,  1.00step/s]"
        "INFO step:250 loss:0.50 grdn:0.5 lr:1.0e-05"
        "Training:  50%|#### | 500/1000 [05:00<05:00,  1.00step/s]"
        "INFO step:500 loss:0.25 grdn:0.4 lr:9.0e-06"
    )
    parse_metrics_into(batched, m, None, floor)

    assert m.current_step == 500, "the last frame in the burst owns the step"
    assert floor.value == 500, "and the floor must not be dragged back with it"
