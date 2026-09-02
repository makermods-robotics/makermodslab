<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/makermods-robotics/makermodslab/raw/main/frontend/public/makermods/logo-mark-white.png" />
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/makermods-robotics/makermodslab/raw/main/frontend/public/makermods/logo-mark.png" />
    <img src="https://github.com/makermods-robotics/makermodslab/raw/main/frontend/public/makermods/logo-mark.png" alt="MakerMods" height="64" />
  </picture>
</p>

<h1 align="center">MakerMods Lab</h1>

<p align="center">
  <b>A web UI interface for policy development.</b><br />
  Built for the Maker Arm, the Metal Arm, and the SO-101. Every robot in LeRobot. Single or Bimanual.
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

---

MakerMods Lab puts the full workflow for robotic policy development into one browser tab. Plug in an
arm, open the app, and go. Calibrate, teleoperate, record, curate, train, deploy, evaluate, then go
around again.

![MakerMods Lab demo](https://raw.githubusercontent.com/makermods-robotics/makermodslab/assets/readme-demo/demo.gif)

## "So what is this actually?"

A policy is never right the first time. You record demos, you train, you watch the robot fumble, you
work out which demos caused it, you collect more, you merge, you fine-tune, you run it again. That
loop is the job.

Most tools make the first pass easy and the tenth pass miserable. One CLI invocation per step, a
dataset directory you edit by hand, a checkpoint path you paste between five terminals. We built for
the tenth pass.

**The whole loop lives in one tab.** Record into a dataset, curate it, launch training, watch
the loss chart, deploy the checkpoint to the arm, evaluate it, and fine-tune from the result. No
terminal, no re-deriving a path.

**Correct the policy while it is running.** DAgger hands control back to the leader arm mid-rollout,
so you drive the robot through the exact motion it just failed. Those corrections land in the
training set, and the evaluation summary offers to merge them and fine-tune from the same screen.

**Curate before you spend a GPU on it.** Open any dataset, watch the episodes and deselect the bad ones for training. Nothing is ever deleted. Merge datasets from the UI when you want the combined set instead.

**Fine-tune from where you stopped.** Continue any run from a checkpoint, and also fine tune any public policies

**Powered by LeRobot and Huggingface.**

## Remote everything

**Training jobs on any node in your tailnet.** Point a run at this machine, at a peer node, or at a
Hugging Face Jobs GPU, from the same picker.

**Remote teleoperation over [LiveKit](https://livekit.io/).** Teleop any follower arm and view cameras over internet with livekit

**Remote inference on [Modal](https://modal.com/) GPUs.** Run bigger models with cloud GPUs

## Quick start

Requires Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/).

**Just want to use it?** One line, no clone:

```bash
uv tool install "git+https://github.com/makermods-robotics/makermodslab"
makermodslab            # serves the UI + API on :8000, opens your browser
```

**Want to change the code?** Clone and install editable. Your edits take effect on the next run, with
no reinstall:

```bash
git clone https://github.com/makermods-robotics/makermodslab
cd makermodslab
uv venv --python 3.12
uv pip install -e .
.venv/bin/makermodslab   # first launch also links `makermodslab` onto your PATH
```

That first launch symlinks the command into `~/.local/bin`, so from then on:

```bash
makermodslab            # run the app: built UI + API on :8000
makermodslab --dev      # hack on it: Vite hot reload on :8080 + auto-reloading API on :8000
```

## Server mode

Same binary whether it is your laptop or a headless station in the corner of the lab wired to the
arms.

```bash
makermodslab --lan                    # bind 0.0.0.0, no browser, serve the whole LAN
makermodslab --bind tailscale0        # or bind one interface, tailnet only
makermodslab --no-ui                  # pure API node, no frontend
makermodslab --discover-tailscale     # find peer nodes over Tailscale
```

Once a station is up, any client on the same tailnet can drive it from a browser, and any node can
hand a training job to any other.

**Peer nodes are verified, not trusted.** A node is only added once its `/api/v1/health` identity
document checks out, and a discovered peer gets re-verified every time.

**One browser drives the robot at a time.** The tab that started a session checks in every twenty
seconds. Go quiet for a minute, because the tab closed or the laptop shut, and the server stops the
session and releases the arm by itself.

## Every arm, single or bimanual

Three families, and the app knows the difference. Bus protocol, calibration flow, port detection,
joint count and safe-stop behaviour all branch on the arm type, so you never hand-configure it.

| Arm              | Follower                        | Leader        | Joints    |
| ---------------- | ------------------------------- | ------------- | --------- |
| **SO-101**       | Feetech STS3215 over USB serial | SO-101 leader | 6 per arm |
| **Maker Arm v1** | RobStride over CAN              | Star Arm 102  | 7 per arm |
| **Metal Arm**    | Damiao over CAN                 | Star Arm 102  | 7 per arm |

Every family runs single or bimanual: two leader/follower pairs, four-arm calibration, dual-arm
teleoperation and bimanual recording.

## Also in the box

**Guided calibration.** Manual step by step or fully automatic for the SO-101, zero-pose calibration
for the CAN arms. Save calibrations under names instead of overwriting them.

**Episode viewer.** Synced camera grid, transport controls, and a joint-position chart tied to the
playhead. Works on Hub-only datasets too, streaming chunk by chunk with no full download.

**Replay.** Play a recorded episode's motion back on the real robot.

**Import and upload.** Pull a dataset or policy from the Hugging Face Hub or your disk, push your
dataset back up in one click.

**English and Simplified Chinese**, throughout.

**A versioned API.** Everything the UI does is a documented `/api/v1` endpoint, with a committed
OpenAPI snapshot in [`docs/api/openapi.json`](docs/api/openapi.json). That snapshot is also why you
can point Claude Code or Codex at this repo and let an agent drive the robot: start a recording
session, launch a training run, deploy a checkpoint, all through the same endpoints the browser uses.
Or write your own client.

## Some notes

We are early. Expect bugs, expect the UI to move, and expect `staging` to be ahead of `main`. Feature
branches land on `staging` and get promoted to `main` in batches.

If you are running real hardware, start with the arm unclamped and clear of anything you care about.

## Docs

- [CONTRIBUTING.md](CONTRIBUTING.md) covers setup, the dev loop, and the checks a PR has to pass.
- [CLAUDE.md](CLAUDE.md) is the architecture rundown. Written for coding agents, useful for humans.
- [docs/api/openapi.json](docs/api/openapi.json) is the committed API snapshot.
- [frontend/docs/localization.md](frontend/docs/localization.md) is required reading before you touch
  a user-facing string.

## Community

- [Discord](https://discord.gg/q8Dzzpym3f) for chat with the LeRobot community.
- [GitHub Issues](https://github.com/makermods-robotics/makermodslab/issues) for bugs and feature
  requests.

## Contributing

PRs welcome. Branch off `staging` and open your PR against `staging`.

```bash
uv pip install -e ".[dev]"   # ruff, pre-commit, pytest
pre-commit install           # wires the git hook, please don't skip this
makermodslab --dev           # Vite on :8080, uvicorn --reload on :8000
```

The full version, including the two frontend typecheck projects that a bare `tsc` skips without
complaint, is in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0. See [LICENSE](LICENSE).

MakerMods Lab is maintained by [makermods-robotics](https://github.com/makermods-robotics). It began
as a fork of Hugging Face's [leLab](https://github.com/huggingface/leLab), also Apache 2.0, and it is
built on [LeRobot](https://github.com/huggingface/lerobot). Go there for everything beneath the UI.
