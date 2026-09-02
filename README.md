<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/makermods-robotics/makermodslab/raw/main/frontend/public/makermods/logo-mark-white.png" />
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/makermods-robotics/makermodslab/raw/main/frontend/public/makermods/logo-mark.png" />
    <img src="https://github.com/makermods-robotics/makermodslab/raw/main/frontend/public/makermods/logo-mark.png" alt="MakerMods" height="64" />
  </picture>
</p>

<h1 align="center">MakerMods Lab</h1>

<p align="center">
  <b>A web interface for <a href="https://github.com/huggingface/lerobot">LeRobot</a>, built for the SO-101 leader/follower arm.</b>
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

---

MakerMods Lab puts the full LeRobot workflow into one browser tab:

- Calibrate, teleoperate, record, train, and replay — plug in your arm, open the app, and go. No CLI gymnastics, no keyboard prompts.
- Hardware-safety guards, bimanual support, and a more guided setup and training flow than upstream.

Built by [makermods-robotics](https://github.com/makermods-robotics) and forked from Hugging Face's **[LeLab](https://github.com/huggingface/leLab)**.

## Demo

![MakerMods Lab demo](https://raw.githubusercontent.com/makermods-robotics/makermodslab/assets/readme-demo/demo.gif)

## Quick Start

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

**Just want to use it?** One line, no clone:

```bash
uv tool install "git+https://github.com/makermods-robotics/makermodslab"
makermodslab            # serves the UI + API on :8000, opens your browser
```

**Want to change the code?** Clone and install editable — your edits take effect on the next run, no reinstall needed:

```bash
git clone https://github.com/makermods-robotics/makermodslab
cd makermodslab
uv venv --python 3.12
uv pip install -e .
.venv/bin/makermodslab            # first launch also links `makermodslab` onto your PATH
```

That first launch symlinks the command into `~/.local/bin`, so from then on you can run it from anywhere:

```bash
makermodslab            # run the app: built UI + API on :8000
makermodslab --dev      # hack on it: Vite hot reload on :8080 + auto-reloading API on :8000
```

## What you can do

At a glance, MakerMods Lab wraps the following LeRobot workflow steps:

<div align="center">
  <table>
    <tr>
      <td>🎯 <b>Calibrate</b></td>
      <td>Guided web flow for both arms — manual or fully automatic, no keyboard prompts.</td>
    </tr>
    <tr>
      <td>🕹️ <b>Teleoperate</b></td>
      <td>Move the leader, the follower mirrors it. Live joint streaming into a 3D viewer.</td>
    </tr>
    <tr>
      <td>📹 <b>Record</b></td>
      <td>Capture episodes into a LeRobotDataset, with multiple cameras.</td>
    </tr>
    <tr>
      <td>🧠 <b>Train</b></td>
      <td>Kick off a LeRobot training job and watch the loss/lr chart live.</td>
    </tr>
    <tr>
      <td>🤖 <b>Run inference</b></td>
      <td>Execute a trained policy on the follower.</td>
    </tr>
    <tr>
      <td>📥 <b>Import</b></td>
      <td>Pull a dataset or model from the <a href="https://huggingface.co/">Hugging Face Hub</a> or your disk to get started.</td>
    </tr>
    <tr>
      <td>☁️ <b>Upload</b></td>
      <td>Push your dataset to the <a href="https://huggingface.co/">Hugging Face Hub</a> in one click.</td>
    </tr>
  </table>
</div>

## What MakerMods Lab adds

Opinionated extensions on top of the core workflow above.

### Hardware safety

Not letting a wiring mistake break a servo:

- 🛡️ **Arm-identity guard** — fingerprints each arm's EEPROM before energizing, so a swapped leader/follower port is caught rather than driven.
- ✋ **Hand-motion port detection** — hit _Detect_ and swing an arm's base to identify its serial port with no motor power. The legacy gripper-wiggle method is still available.
- 🛑 **Graceful stops** — teleop and auto-calibration freeze, return to the start pose, then release torque. Hit _Stop_ twice for an instant release.
- 🔋 **Motor power limiting** — cap per-robot motor power, with a live supply-voltage readout and session power telemetry.

The SO-101 split-host field trial has a separate, safety-gated runbook.
Maintainers can start with the
[PR review bundle](docs/architecture/remote-teleoperation-pr-summary.md), then
follow the [two-laptop quickstart](docs/remote-teleop/two-laptop-quickstart.md).
Live enable requires a matching secured-arm commissioning record, not merely a
saved remote-role configuration. The
[software validation record](docs/remote-teleop/software-validation.md) keeps
deterministic evidence separate from physical-arm acceptance.

### Robots & calibration

- 🤝 **Robots as first-class objects** — create a robot through a dialog with an immutable arm layout (single or bimanual), and reuse it across every feature.
- 🦾 **Bimanual mode** — two leader/follower pairs: 4-arm calibration, bimanual teleoperation with a dual-arm 3D viewer, and bimanual dataset recording.
- 🏷️ **Named calibrations** — save calibrations under names instead of overwriting; deleting one in use unassigns it rather than blocking. A start-pose guard rejects calibrations that didn't begin from the middle pose, and <code>wrist_roll</code> is handled as a full turn to match upstream <code>lerobot-calibrate</code>.

### Datasets

- 🪪 **Dataset info cards** — episodes, cameras, and tasks with per-task episode counts, plus warnings on unusable datasets.
- 🎬 **Episode viewer** — click any dataset, local or Hub-only, to open a synced camera grid, transport controls, and a joint-position chart tied to the playhead. Hub-only datasets stream chunk-by-chunk on demand, no full download required.
- 🔀 **Merge from the UI** — combine datasets (wraps LeRobot's <code>aggregate_datasets</code>), with legible errors and name validation.
- 🎥 **Preview before naming** — see all camera feeds before committing to a recording setup.

### Training

- 🧭 **Model-type-first entry** — pick the policy and dataset on the home page (availability-gated), frozen for the run thereafter; config guards, run names, and honest compute targets.
- ⏯️ **Continue from a checkpoint** — resume a saved run, with the lineage's loss chart stitched into one view and source checkpoints folded into the successor.
- 🗂️ **Job tooling** — checkpoint management, model display-name aliases, and idempotent imports with dedup.

## Resources

- **[LeRobot](https://github.com/huggingface/lerobot):** the underlying library — go here for everything beyond the UI.
- **[CLAUDE.md](CLAUDE.md):** architecture rundown for contributors.

## Community

- **[Discord](https://discord.gg/q8Dzzpym3f):** chat with the LeRobot community.
- **[GitHub Issues](https://github.com/makermods-robotics/makermodslab/issues):** bug reports, feature requests.

## Contribute

PRs welcome. Setup, the hot-reload dev loop, and the checks a PR has to pass are all in
**[CONTRIBUTING.md](CONTRIBUTING.md)**.

The short version:

```bash
uv pip install -e ".[dev]"   # ruff, pre-commit, pytest
pre-commit install           # wires the git hook — please don't skip this
makermodslab --dev           # Vite on :8080, uvicorn --reload on :8000
```

## Team

MakerMods Lab is maintained by [makermods-robotics](https://github.com/makermods-robotics).

## License

MakerMods Lab is released under the [Apache 2.0 License](LICENSE).
