# SO-101 split-host two-laptop quickstart

This trial uses one SO-101 leader on the operator laptop and one SO-101
follower on the robot laptop. Both laptops run the same MakerMods Lab build.
The browser and normal API remain loopback-only; only the dedicated TLS control
and authenticated UDP action ports bind to the configured private interface.

> **Do not enable live control on an unsecured arm.** Saving a role
> configuration is not commissioning. The robot refuses live enable until the
> exact follower calibration, joint schema, limits, device identity, rig, and
> expected leader calibration have a matching owner-private secured-arm
> commissioning record. Any change invalidates that record.

> **The live robot host supports Linux and Apple Silicon macOS.** Before any
> Feetech torque or configuration write, the worker derives the adapter's USB
> VID/PID/serial identity from the already-open character-device descriptor.
> Linux resolves that device number through sysfs. macOS ties it to one unique
> IOKit serial client and USB parent, then repeats the descriptor and IOKit
> checks to reject unplug/replug races. Both implementations fail closed rather
> than trusting the configured pathname.

## 1. Install the PR on both laptops

Start in a clean checkout of this PR on each laptop. The shortest reproducible
path is:

```bash
scripts/remote-teleop-pr-check.sh
```

That command installs managed Python 3.12 and the frontend dependencies, builds the
UI, and runs the two-process TLS/UDP smoke and runtime-loss tests. It starts no
MakerMods application listener and opens no arm; the tests use only ephemeral
loopback sockets. On a checkout whose `.venv` and
`frontend/node_modules` are already prepared, use
`scripts/remote-teleop-pr-check.sh --verify-only`; the smoke gate should finish
in a few minutes. A clean dependency download takes longer according to network
and package-cache speed.

The script executes these reviewable commands rather than hiding another
installation path:

```bash
uv python install 3.12
uv venv --python 3.12 --managed-python
.venv/bin/python -c 'import platform; print(platform.machine())'
uv pip install -e ".[dev]"
npm --prefix frontend ci
npm --prefix frontend audit --audit-level=high
npm --prefix frontend run build
.venv/bin/python -m pytest -q \
  tests/test_remote_teleop_two_process.py \
  tests/test_remote_teleop_runtime_process.py
```

On Apple Silicon, the architecture command must print `arm64`. Stop if it
prints `x86_64`: PyTorch no longer publishes the required macOS x86_64 wheels,
so a Rosetta Python cannot resolve the pinned LeRobot dependency. The
`--managed-python` flag prevents an older Intel Homebrew Python from being
selected ahead of uv's native interpreter. Current Linux `x86_64` and
`aarch64` wheels are supported; this warning is specific to macOS x86_64.

For the physical trial, the robot laptop may run Linux with `/sys/dev/char`
available or native Apple Silicon macOS. Use a follower USB adapter that
exposes one globally unique, nonempty USB serial plus VID/PID. Missing or
duplicated identity fails closed before motor writes. The operator laptop may
also use supported Linux or Apple Silicon macOS.

The smoke commands open loopback TLS/UDP sockets and run clean subprocesses with
simulated arms. They must pass before attaching hardware. They cover pinned TLS,
one-time pairing, UDP endpoint proof, loss/reorder/duplicate packets, duplicate
sessions, acknowledged STOP, browser/operator/control/network loss, clock
drift, restart, rejection of a pre-restart action, and robot-local stop after
abrupt operator-process loss.

Do not copy `frontend/dist/` changes into the PR; the normal CI build owns that
artifact.

## 2. Join the private network

Install Tailscale using its current [macOS](https://tailscale.com/docs/install/mac)
or [Linux](https://tailscale.com/docs/install/linux) instructions. Sign both
laptops into the intended tailnet and apply the least-privilege policy in
[network-and-tailscale.md](network-and-tailscale.md).

Run on both laptops:

```bash
tailscale status
tailscale ip -4
```

Record the robot's `100.64.0.0/10` address. Do not use a public address, a
hostname, `0.0.0.0`, or the Wi-Fi interface address in the robot role.

## 3. Create the robot TLS identity

Run from the robot checkout. This deliberately stores the key in a sibling
private directory, outside the repository:

```bash
REMOTE_TLS_DIR="$(cd .. && pwd)/makermodslab-remote-private"
mkdir -p "$REMOTE_TLS_DIR"
chmod 700 "$REMOTE_TLS_DIR"
openssl req -x509 -newkey rsa:3072 -sha256 -nodes -days 30 \
  -subj "/CN=makermodslab-remote-robot" \
  -keyout "$REMOTE_TLS_DIR/robot-control-key.pem" \
  -out "$REMOTE_TLS_DIR/robot-control-cert.pem"
chmod 600 "$REMOTE_TLS_DIR/robot-control-key.pem"
REMOTE_CERT_SHA256="$(openssl x509 -in "$REMOTE_TLS_DIR/robot-control-cert.pem" -outform DER | openssl dgst -sha256 | awk '{print $2}')"
printf 'certificate=%s\nprivate-key=%s\nfingerprint=sha256:%s\n' \
  "$REMOTE_TLS_DIR/robot-control-cert.pem" \
  "$REMOTE_TLS_DIR/robot-control-key.pem" \
  "$REMOTE_CERT_SHA256"
```

Share only the `sha256:...` fingerprint with the operator tester. Never send
the private key or commit the sibling directory. Verify its absence from the
checkout before continuing:

```bash
git status --short
git grep -n "BEGIN PRIVATE KEY" -- . ':!frontend/dist'
```

The second command should print nothing.

## 4. Start both local applications

Robot laptop:

```bash
.venv/bin/makermodslab --offline
```

Operator laptop:

```bash
.venv/bin/makermodslab --offline
```

Both commands bind the ordinary UI/API to loopback and open the local browser.
No remote listener and no serial device opens merely because MakerMods Lab
started.

## 5. Configure and commission the robot laptop

Open **Remote teleoperation**, select **Remote robot**, then enter:

- the robot node ID and the existing follower robot record;
- the robot's exact Tailscale IPv4 address;
- TLS control port `7443` and UDP action port `7444`;
- the absolute certificate and private-key paths printed in step 3;
- the exact permitted leader calibration ID and digest; and
- conservative action rate, watchdog, velocity, and acceleration limits.

Save. Confirm the page still says **runtime disabled** and **listener stopped**.
Configuration alone must never expose the port or open the follower.

Complete [commissioning-worksheet.md](commissioning-worksheet.md) at the robot
laptop. The local commissioning action is intentionally no-motion: it verifies
the selected profile, proves the unique USB identity from the already-open
descriptor through Linux sysfs or macOS IOKit, connects with the follower
physically secured, reads the initial observation and torque-off state, and
closes the device. It writes an owner-private record only when all checks pass.
It does not alter firmware, servo IDs, baud rate, EEPROM, or calibration.

## 6. Pair the operator laptop

On the robot laptop, press **Open pairing window**. Keep the one-time code on
screen; do not put it in chat, shell history, screenshots, or the repository.

On the operator laptop, select **Remote operator**, then enter:

- an operator node ID;
- the existing leader robot record and calibration;
- `wss://ROBOT_TAILSCALE_IP:7443`; and
- the independently copied `sha256:...` certificate fingerprint.

Save, press **Pair**, and type the one-time code. The operator stores the issued
credential in its owner-private application data; the robot stores only a
salted digest. Tailscale membership alone never grants action authority.

## 7. Run the secured-arm trial

1. Robot tester presses **Enable** and verifies the displayed bind address,
   commissioned profile digest, owner, UDP state, watchdogs, and torque state.
2. Operator tester presses **Connect**. Confirm both screens show the same rig,
   leader/follower calibration identities, joint order, session ID, and
   executor generation.
3. Move one joint through the worksheet's smallest envelope. Confirm latency,
   last sequence, observation, and bounds before expanding.
4. Run every fault row in the worksheet. The robot tester watches local state;
   the operator display is not evidence that the robot stopped.
5. Press operator **STOP**, then robot-local **STOP**. Accept the result only if
   software dispatch halted, hardware stop and close completed, and Feetech
   torque-off readback is explicitly `true`. Unknown or false torque state is a
   fault lockout, not success.

If either process is killed before it can publish that terminal receipt, the
robot laptop must restart with a central hardware lockout. Follow the explicit
SO-101 procedure in [troubleshooting.md](troubleshooting.md); restarting,
disabling the role, or removing saved remote configuration must not clear it.

Perform a full quit/relaunch of both MakerMods Lab processes and repeat the
critical session. Runtime enablement must be cleared after restart.

## 8. Preserve the evidence

Attach the completed worksheet and redacted status/receipt output to the PR.
Remove pairing codes, credential secrets, private-key material, machine paths,
tailnet names, user names, and private IPs. Calibration IDs/digests, profile
digests, sequence counters, measured timings, and boolean safety receipts are
appropriate evidence when they do not identify a private machine or person.

The contributor-executed software baseline and the remaining physical boundary
are recorded separately in
[`software-validation.md`](software-validation.md). Do not substitute that
software record for this trial's completed worksheet.
