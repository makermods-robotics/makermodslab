# ADR: split-host SO-101 teleoperation and owner-fed servo health

Status: implemented; physical secured-arm acceptance remains maintainer-gated

## Context

MakerMods Lab's local topology owns a leader and follower in one process and
calls `get_action()` followed by `send_action()`. That remains the default. In
the split-host topology, the operator laptop owns only the leader and the robot
laptop is the sole follower/Feetech bus owner.

MotorLab model metadata and telemetry conversions are useful, but its
standalone controller cannot run beside MakerMods Lab as another bus owner.

## Decision

1. Split at the LeRobot action/observation boundary; never transport serial
   bytes or device paths.
2. Use one process-wide hardware registry across every arm feature. Claim
   before adapter construction; release only after the owning finalizer reports
   device close and applicable torque evidence.
3. Let the robot mint session ids, generations, and per-session HMAC keys. A
   reconnect always creates new authority and a new sequence window.
4. Carry reliable control/status/STOP over pinned TLS WebSocket and
   latest-value sequenced actions over authenticated UDP.
5. Execute through one `RemoteExecutor` with exact identity/schema checks,
   robot-side position/velocity/acceleration bounds, and local action/control/
   browser watchdogs.
6. Put the live follower behind a timeout-bounded child process. Killing a
   stuck child halts software dispatch but never becomes torque-off evidence.
7. Require an owner-private, exact-profile secured-arm commissioning record
   before the robot listener can start. Persist incomplete stop/close evidence
   and re-latch it on restart until a matching local recovery succeeds.
8. Sample servo health only from the active bus owner and publish snapshots to
   cache-only HTTP readers.

## Transport choice

| Property                       | Sequenced UDP             | QUIC datagrams         | WebSocket/TCP       |
| ------------------------------ | ------------------------- | ---------------------- | ------------------- |
| Fresh action waits behind loss | No                        | No                     | Yes                 |
| Loss/reorder visible to app    | Yes                       | Yes                    | No                  |
| Secure session                 | HMAC key from TLS control | Built in               | Built in            |
| Added runtime                  | Standard library          | Additional QUIC stack  | Existing stack      |
| Role                           | **v1 action lane**        | Possible later adapter | Control/status/STOP |

## Safety invariants

- App start, configuration save, and process restart open no listener/device.
- A network packet cannot acquire authority; it can only exercise an active,
  authenticated, robot-minted session.
- STOP revokes dispatch and generation before requesting physical teardown.
- Action, control, browser, operator-process, and network loss stop the robot
  locally without depending on a remote acknowledgement.
- `stop_accepted` is not torque evidence. `hardware_stop_completed`,
  `hardware_close_completed`, and `torque_off_confirmed` remain independent.
- Unknown/false torque or incomplete close retains a fault lockout across
  process restart.
- Servo diagnostics never create another bus transaction owner.
- Other arm families require separate schemas, stop semantics, and physical
  commissioning; SO-101 evidence is not inherited by interface similarity.

## Consequences

The deterministic suite can prove authority, authentication, fault handling,
bounded stale execution, lifecycle races, durable lockout, and default
dormancy. It cannot prove a particular physical arm's gravity behavior,
direction, real bus latency, or power state. Those remain explicit rows in the
secured-arm worksheet and are prerequisites for physical acceptance.
