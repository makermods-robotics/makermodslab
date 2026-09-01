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
  Built for the Maker Arm, the Metal Arm, the SO-101 — every robot in LeRobot, single or bimanual.
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

</div>

---

MakerMods Lab puts the full workflow for robotic policy development into one browser tab. Plug in
an arm, open the app, and go: calibrate, teleoperate, record, curate, train, deploy, evaluate — and
then go around again, which is the part that actually matters.

![MakerMods Lab demo](https://raw.githubusercontent.com/makermods-robotics/makermodslab/assets/readme-demo/demo.gif)

## "So what is this actually?"

A policy is never right the first time. You record demos, you train, you watch the robot fumble, you
figure out *which* demos were the problem, you collect more, you merge, you fine-tune, you run it
again. That loop is the job. Every tool we have seen makes the first pass easy and the tenth pass
miserable — a CLI invocation per step, a dataset directory you edit by hand, a checkpoint path you
paste between five terminals.

So we built for the tenth pass.

- **The whole loop lives in one tab.** Record into a LeRobotDataset, curate it, launch training,
  watch the loss chart, deploy the checkpoint to the arm, evaluate it, and fine-tune from the result
  — without touching a terminal or re-deriving a path.
- **Curate before you burn a GPU on it.** Open any dataset, watch the episodes back in a synced
  camera grid, and untick the bad ones. The exclusions are per-dataset, they persist, and the
  training request only sees what you kept — nothing is ever deleted. Merge datasets from the UI
  when you want the combined set instead.
- **Fine-tune from where you stopped.** Continue any run from a checkpoint, with the lineage's loss
  chart stitched into one view. Deploy a policy, and the evaluation summary offers to merge what you
  just collected and fine-tune straight from it.
- **Training does not have to run here.** Point a run at this machine, at another node on your
  tailnet, or at a Hugging Face Jobs GPU, from the same picker.
- **Everything is LeRobot and Hugging Face all the way down.** Real `LeRobotDataset`s, real LeRobot
  policies, real Hub repos. Nothing is trapped in our format, and you can always drop back to the
  CLI.

## Quick start

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

**Just want to use it?** One line, no clone:

```bash
uv tool install "git+https://github.com/makermods-robotics/makermodslab"
makermodslab            # serves the UI + API on :8000, opens your browser
```

**Want to change the code?** Clone and install editable — your edits take effect on the next run,
no reinstall needed:

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

MakerMods Lab is the same binary whether it is your laptop or a headless station in the corner of
the lab wired to the arms.

```bash
makermodslab --lan                    # bind 0.0.0.0, no browser — serve the whole LAN
makermodslab --bind tailscale0        # or bind one interface: tailnet-only, nothing else
makermodslab --no-ui                  # pure API node, no frontend
makermodslab --discover-tailscale     # find peer nodes over Tailscale
```

Once a station is up, any client on the same tailnet can drive it from a browser, and any node can
hand a training job to any other:

- **Peer nodes are verified, not trusted.** A node is only added after its `/api/v1/health` identity
  document checks out, and a discovered peer is re-verified every time — never taken from disk.
- **Offload a run to a peer** and MakerMods Lab drives that peer's own jobs API, relays its progress,
  survives network blips, and reports the peer's own terminal verdict instead of guessing.
- **Datasets travel via the Hub**, because a LAN peer can no more see your local LeRobot cache than
  an HF pod can.
- **One live session, leased.** The server hands out a lease with a heartbeat; if a browser wanders
  off, a watchdog safety-stops the session and releases the arm. No tab elections, no stop beacons,
  no way to leave a robot energized because someone closed a laptop.

## Remote robots and remote GPUs

> [!NOTE]
> This section is shipping — the pieces below live on feature branches and land on `staging` first.
> Everything above is on `staging` today.

- 🎥 **Remote teleoperation over [LiveKit](https://livekit.io/).** Split the lab across two machines:
  the station owns the follower arm and the cameras, your laptop owns the leader. Leader joints go
  out over a LiveKit Portal session, camera frames come back, and the dataset is written on the
  station from the raw observation — video compression never touches what lands on disk.
- ⚡ **Remote inference on [Modal](https://modal.com/) GPUs.** Run your own policies against a cloud
  GPU when the machine next to the robot cannot keep up with the model you actually want to deploy.
- 🧑‍🏫 **DAgger, human in the loop.** Take the leader back mid-rollout, correct the policy where it is
  failing, and close the loop from the session summary: merge the corrections into the training set
  and fine-tune from the same screen.
- ✂️ **Episode trimming.** Cut the dead frames off the head and tail of an episode instead of
  throwing the whole take away.

## Every arm, single or bimanual

Three families, and the app knows the difference — bus protocol, calibration flow, port detection,
joint count and safe-stop behaviour all branch on the arm type, so you never hand-configure it.

| Arm | Follower | Leader | Joints |
|---|---|---|---|
| **SO-101** | Feetech STS3215 over USB serial | SO-101 leader | 6 per arm |
| **Maker Arm v1** | RobStride over CAN | Star Arm 102 | 7 per arm |
| **Metal Arm** | Damiao over CAN | Star Arm 102 | 7 per arm |

Every family runs single or **bimanual** — two leader/follower pairs, four-arm calibration, dual-arm
teleoperation and bimanual recording.

## Hardware safety

Policy work means a lot of stop/start cycles on real motors. These are not features so much as
things we refuse to ship without.

- 🛡️ **Arm-identity guard** — fingerprints each arm's servo EEPROM before energizing, so a swapped
  leader/follower port gets caught instead of driven.
- 🛑 **Every flow returns the arm before releasing torque.** A CAN arm has no brakes; cutting torque
  anywhere but near rest drops the whole arm under gravity. Teleop, recording, replay and inference
  all ease home first. Hit *Stop* twice to abort the return and release immediately.
- ⚡ **CAN crash recovery.** A Damiao bus handshake *is* the motor-enable command, so a killed process
  can leave motors holding their last command forever. `POST /api/v1/arms/release-torque` de-energizes
  the bus, and it deliberately works when session state is wrecked.
- ✋ **Hand-motion port detection** — hit *Detect* and swing an arm to identify its serial port with
  no motor power at all.
- 🔋 **Motor power limiting** — cap per-robot motor power, with a live supply-voltage readout and
  session power telemetry.
- 🔒 **Mutual exclusion in code.** Teleop, recording, inference, replay, calibration and auto-calibration
  cannot overlap, and a refusal tells you exactly which one holds the robot.

## Also in the box

- 🎯 **Guided calibration** — manual step-by-step or fully automatic for the SO-101; zero-pose
  calibration for the CAN arms. Save calibrations under names instead of overwriting them.
- 🎬 **Episode viewer** — synced camera grid, transport controls, and a joint-position chart tied to
  the playhead. Works on Hub-only datasets too, streaming chunk-by-chunk with no full download.
- ▶️ **Replay** — play a recorded episode's motion back on the real robot.
- 📥 **Import / ☁️ upload** — pull a dataset or policy from the Hugging Face Hub or your disk, push
  your dataset back up in one click.
- 🌏 **English and Simplified Chinese**, throughout.
- 🔌 **A versioned API.** Everything the UI does is a documented `/api/v1` endpoint — see
  [`docs/api/openapi.json`](docs/api/openapi.json). Build your own client.

## Some notes

We are early. Expect bugs, expect the UI to move, and expect `staging` to be ahead of `main` —
feature branches land on `staging` and get promoted to `main` in batches.

If you are running real hardware: read the safety section above, and start with the arm unclamped
and clear of anything you care about.

## Docs

- **[CONTRIBUTING.md](CONTRIBUTING.md)** — setup, the dev loop, and the checks a PR has to pass.
- **[CLAUDE.md](CLAUDE.md)** — the architecture rundown. Written for coding agents, useful for humans.
- **[docs/api/openapi.json](docs/api/openapi.json)** — the committed API snapshot.
- **[frontend/docs/localization.md](frontend/docs/localization.md)** — read before touching any
  user-facing string.

## Community

- **[Discord](https://discord.gg/q8Dzzpym3f)** — chat with the LeRobot community.
- **[GitHub Issues](https://github.com/makermods-robotics/makermodslab/issues)** — bugs and feature
  requests.

## Contributing

PRs welcome. Branch off `staging` and open your PR against `staging`.

```bash
uv pip install -e ".[dev]"   # ruff, pre-commit, pytest
pre-commit install           # wires the git hook — please don't skip this
makermodslab --dev           # Vite on :8080, uvicorn --reload on :8000
```

The full version, including the two frontend typecheck projects that a bare `tsc` silently skips,
is in **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## License

Apache 2.0 — see [LICENSE](LICENSE).

MakerMods Lab is maintained by [makermods-robotics](https://github.com/makermods-robotics). It began
as a fork of Hugging Face's [leLab](https://github.com/huggingface/leLab) (also Apache 2.0), and it
is built on [LeRobot](https://github.com/huggingface/lerobot) — go there for everything beneath the
UI.
