# Remote teleoperation troubleshooting

Stop first. Diagnose only after the robot reports software dispatch halted. If
torque-off is false or unknown, remove physical motor power and retain fault
lockout.

## Enable says the profile is not commissioned

This is expected after initial configuration or after changing a follower,
calibration, joint schema, rig, or limit. Secure the follower and rerun the
local no-motion commissioning flow. Never edit or copy the owner-private JSON
record; the profile digest must be produced from the live selected profile.

## Pairing fails

1. Open a new pairing window locally on the robot; codes are single-use and
   expire.
2. Verify the operator pinned the fingerprint printed from the robot
   certificate, not a certificate filename or private key.
3. Verify TCP control reachability is allowed only operator-to-robot on the
   configured private interface.
4. Revoke unexpected credentials from the robot UI before reopening pairing.

Repeated guesses intentionally close/rate-limit pairing. Do not paste a code
into an issue or log.

## TLS connects but UDP proof fails

Verify the UDP port matches on the robot configuration, tailnet grant, and host
firewall. UDP is endpoint-bound: a proxy/NAT/source-port change after proof is
rejected. Restart the session to mint a new key and prove the new endpoint.

Do not fall back to unauthenticated UDP, a wildcard bind, or raw serial
tunneling. The test `test_udp_receiver_refreshes_session_after_blocking_receive_begins`
guards the listener/session startup race.

## High latency, stale actions, or frequent watchdog stops

Run:

```bash
tailscale status
```

Check whether the pair is direct or relayed, then compare displayed clock
uncertainty and one-way action age with the commissioned budgets. Packet loss
may be tolerated; stale execution is not. Do not lengthen watchdogs during an
active trial. Stop, secure the arm, choose a new conservative profile, and
commission that changed profile.

## STOP is accepted but torque is unknown

`stop_accepted` and `software_dispatch_halted` prove that MakerMods revoked
future writes. They do not prove the Feetech bus disabled torque. Use robot-side
physical power removal, retain fault lockout, preserve the receipt, and inspect
the adapter's stop/close evidence. Process termination is never torque proof.
After the arm is secured and power removal is reachable, use **Run secured
recovery**. The durable latch clears only if the same commissioned profile and
device reconnect disarmed, the host derives the expected USB identity from the
already-open descriptor, an observation succeeds, Feetech torque-off readback
is `true`, and close completes. Linux performs that proof through sysfs; macOS
performs it through a device-number-bound, twice-verified IOKit chain. On any
other host, or when either platform cannot establish that proof, recovery
deliberately remains a physical lockout.
Restarting, editing the JSON file, or changing roles cannot clear the latch.

## The process died and only the central hardware lockout remains

Every hardware claim writes owner-private crash intent before a device may
open. If the process exits before it records a complete safe-close receipt,
the next process restores that claim as unresolved even when no remote-session
fault record survived. Do not delete or edit the hardware safety files.

Ordinary local SO-101 and Maker/Metal CAN claims are intentionally recorded as
**physical-only recovery**. Their normal upstream open paths do not expose a
universal descriptor-bound identity check, so no software endpoint may clear
their crash latch merely because another arm appears at the same path and
acknowledges torque-off. Secure the arm, remove physical motor power, preserve
the lockout, and follow the team's manual service procedure.

The dedicated remote SO-101 worker is narrower: on Linux and macOS it freezes
the unique pre-open adapter binding into the lease and child, proves that
binding from the already-open descriptor before any torque/configuration
write, then requires the exact commissioned servo calibration, torque-off
readback, and device close. Use **Run secured recovery** only for a remote fault
whose status offers that action. If the proof is missing, changed, duplicated,
unsupported, or any write/readback/close fails, the response remains locked
out and never exposes the raw path or USB serial.

## A second operator cannot connect

Only one action owner/session is allowed. Stop the existing owner and confirm
its hardware close/torque receipt before opening a replacement session. A stale
STOP from the first session must not stop the replacement.

## Restart does not restore motion

That is deliberate. Runtime enablement is memory-only and clears on every
restart. Reconcile any unresolved hardware lease/fault first, then explicitly
enable the commissioned robot profile. Never add an auto-enable service.
