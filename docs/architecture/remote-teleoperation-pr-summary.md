# PR bundle: secured split-host SO-101 teleoperation

Suggested title:

> Add secured split-host SO-101 teleoperation and owner-fed servo health

This page is the maintainer entrypoint. It gives the shortest software proof,
the review order, and the boundary between deterministic evidence and the
still-required physical-arm trial.

## Fast review and simulated launch

From a clean checkout of this PR on either trial laptop:

```bash
scripts/remote-teleop-pr-check.sh
```

The script installs managed Python 3.12 and the frontend dependencies, builds
the UI, and runs the authenticated two-process smoke and abrupt-operator-loss
test. It opens only ephemeral loopback test sockets: no MakerMods application
listener or arm. Dependency download time is network-dependent.

If the checkout is already prepared, the few-minute path is:

```bash
scripts/remote-teleop-pr-check.sh --verify-only
```

For the complete backend/frontend regression after the smoke gate:

```bash
scripts/remote-teleop-pr-check.sh --verify-only --full
```

Do not include generated `frontend/dist/` changes in the PR; CI owns that
artifact.

## What this contribution adds

- One process-wide hardware registry for local teleoperation, calibration,
  recording, replay, inference, diagnostics, recovery, and both remote roles.
- Real SO-101 adapters split at the LeRobot action boundary: the operator owns
  only the leader; a killable robot child owns the follower Feetech bus; every
  admitted action reaches hardware only through `RemoteExecutor`.
- Pinned TLS control for pairing, clock synchronization, session creation,
  heartbeat, status, and acknowledged STOP; authenticated sequenced UDP for
  latest-value actions; robot-local action/control/browser watchdogs.
- Owner-private commissioning bound to follower device identity, both
  calibration identities, rig, ordered schema/units, and enforced limits.
- Truthful stop and recovery receipts: unknown torque or incomplete close is a
  durable fault lockout, never successful STOP evidence.

The UI provides role configuration, commissioning, pairing, health, registry
ownership, STOP/torque receipts, recovery, and rollback. Servo health is read
only by the current bus owner and published from cache; HTTP never becomes a
second serial owner.

## Five-part review map

| Review slice                    | Start here                                                                                                                                    | Invariant to verify                                                                                                                                  |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Ownership and crash recovery | `makermodslab/hardware_lease.py`, `hardware_recovery_identity.py`, `sessions.py`                                                              | Claim happens before construction; release requires complete evidence; restart restores unresolved ownership.                                        |
| 2. Follower execution and STOP  | `remote_teleop/adapters/follower_process.py`, `lerobot_follower.py`, `executor.py`, `robot_service.py`                                        | Only the child owns Feetech; all actions pass through `RemoteExecutor`; blocking calls are kill-bounded; torque is never inferred from process exit. |
| 3. Network authority            | `pairing.py`, `control_server.py`, `control_client.py`, `transport.py`, `watchdog.py`                                                         | Robot mints one owner/session; TLS is pinned; UDP is authenticated, sequenced, fresh, endpoint-bound; every loss mode stops locally.                 |
| 4. Identity and commissioning   | `calibration_identity.py`, `commissioning.py`, `serial_port_identity.py`, `macos_serial_registry.py`                                          | Live enable requires the exact commissioned profile; opened descriptor identity is proven through Linux sysfs or twice-checked macOS IOKit.          |
| 5. API, UI, and field proof     | `api_service.py`, `server.py`, `frontend/src/pages/RemoteTeleoperation.tsx`, `tests/test_remote_teleop_two_process.py`, `docs/remote-teleop/` | Saving config remains dormant; status is redacted; the simulated matrix and physical worksheet make different claims.                                |

## Non-negotiable safety invariants

1. Application start, role save, and restart open no arm and start no remote
   listener. Runtime enablement is memory-only and clears on restart.
2. The central hardware claim precedes adapter construction. Unknown close or
   torque state retains the lease and fault journal.
3. The robot stops locally when action, TLS control, browser proof,
   operator-process, or network freshness disappears.
4. `stop_accepted` and process termination are not torque proof. Success needs
   explicit Feetech torque-off readback and device-close evidence.
5. Tailscale provides private routing, not action authority. Pairing,
   credentials, certificate pinning, robot-minted sessions, and per-session UDP
   authentication remain mandatory.

## Compatibility and contribution hygiene

Existing robot/calibration files and local teleoperation remain the default.
New typed endpoints live only under `/api/v1/arms/remote-teleoperation` and
`/api/v1/arms/servo-health`.

TLS keys, pairing codes, credential material, private paths, and private
addresses are not contribution data or diagnostic output. The robot accepts
local TLS key paths but stores configuration owner-private and redacts status.
The contribution includes the Nori MotorLab source revision and MIT notice in
`THIRD_PARTY_NOTICES.md` and `licenses/NORI-MotorLab-MIT.txt`.

## Evidence already produced

- 2,436 backend tests and 156 frontend tests pass on the final contributor
  tree; both frontend typechecks and the production build pass.
- Two clean subprocesses pass pinned TLS, one-time pairing, authenticated UDP,
  packet loss/reorder/duplicates, stale/future actions, clock drift, duplicate
  sessions, acknowledged STOP, every required loss mode, and restart rejection.
- Separate real service subprocesses prove robot-local dispatch halt and
  simulated stop/close evidence after abrupt operator-process loss.
- Linux and macOS opened-descriptor identity tests reject swaps, duplicate
  identities, device-number ambiguity, disconnect, and unplug/replug races.
- Ruff, MyPy, Bandit, PyUpgrade, OpenAPI, Prettier, repository hooks, and a
  source-only Gitleaks scan pass.
- The clean lockfile has zero high/critical `npm audit` findings. Two moderate
  React Router 6 notices remain: the app has no SSR hydration and all current
  navigation targets are source-controlled literals; the available fix is a
  separate breaking Router 7 migration.

Two independent source-only adversarial reviews were also completed. Their
confirmed findings are closed in this tree: incomplete follower STOP attempts
report `hardware_stop_completed=false`; commissioning and listener enablement
require a stable unique USB binding before any device or listener opens; an
active pairing window cannot be silently replaced; unauthenticated control
errors are generic; and the UI defaults match the published `7443`/`7444`
trial ports. New regressions cover each boundary. The reviewers' proposed
credential-revocation race was rejected against the shared credential lock and
the existing blocked-open/revocation race test.

The detailed contributor receipt is
[`../remote-teleop/software-validation.md`](../remote-teleop/software-validation.md).

## What remains physically open

This contribution is software-ready; it is not self-certified on live arms.
Maintainers or a contributor with the hardware must run two supervised,
secured SO-101 sessions separated by a full restart and attach the redacted
worksheet.

Start at
[`../remote-teleop/two-laptop-quickstart.md`](../remote-teleop/two-laptop-quickstart.md),
then execute every row in
[`../remote-teleop/commissioning-worksheet.md`](../remote-teleop/commissioning-worksheet.md).
Unknown or false Feetech torque readback remains fault lockout. The operator UI
alone is never evidence that the follower stopped.

## Maintainer acceptance checklist

- [ ] Fast simulated gate passes on both intended laptops.
- [ ] Reviewers confirm the five safety invariants in code, not only tests.
- [ ] No `frontend/dist/`, secret, private path, credential, or personal
      network detail is present in the PR.
- [ ] Two secured-arm sessions pass the full fault matrix and rollback without
      firmware, EEPROM, servo-ID, baud, or calibration changes.
- [ ] The completed redacted worksheet distinguishes measured physical results
      from the contributor software record.
