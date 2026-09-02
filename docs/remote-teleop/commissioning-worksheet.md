# Secured-arm commissioning worksheet

One worksheet covers one exact profile only. Changing the follower device,
either calibration, joint order/units, rig, or enforced limits requires a new
commissioning record and worksheet.

## Trial identity

| Field                           | Recorded value                 |
| ------------------------------- | ------------------------------ |
| Date/time (UTC)                 |                                |
| MakerMods commit                |                                |
| Robot OS/Python                 |                                |
| Operator OS/Python              |                                |
| Follower record ID              |                                |
| Follower device-identity digest |                                |
| Follower calibration ID/digest  |                                |
| Leader record ID                |                                |
| Leader calibration ID/digest    |                                |
| Rig ID/digest                   |                                |
| Ordered joints/units            |                                |
| Limits digest                   |                                |
| Commissioned profile digest     |                                |
| Robot private-interface type    |                                |
| Control/action ports            | `7443` / `7444` unless changed |

Do not record serial paths, user directories, credentials, pairing codes,
private keys, tailnet names, private IPs, or personal names in PR evidence.

## A. Physical and local gates

Initial every row only after observing it at the robot laptop.

|   # | Required check                                                                                                      | Initials/result |
| --: | ------------------------------------------------------------------------------------------------------------------- | --------------- |
|   1 | Follower is mechanically secured against gravity/drop; no person or fragile object is in its workspace.             |                 |
|   2 | Physical power removal/E-stop is reachable by the robot-side tester without entering the workspace.                 |                 |
|   3 | Leader and follower independently pass their normal local calibration/identity checks.                              |                 |
|   4 | Open-descriptor USB identity, calibration digests, joint order/units, and local limits match the table above.       |                 |
|   5 | Tester acknowledges that live execution enables Feetech torque; the physical cutoff remains immediately reachable.  |                 |
|   6 | Local no-motion commissioning connects disarmed, reads observation, verifies Feetech torque off, and closes.        |                 |
|   7 | Owner-private commissioning record exists for the displayed profile digest; a changed profile is refused.           |                 |
|   8 | With runtime disabled, no remote listener exists and no leader/follower device is open.                             |                 |
|   9 | Robot-local STOP before remote action reports dispatch halted, stop/close complete, and torque-off readback `true`. |                 |

If torque-off is `false` or unknown, use physical power removal and stop. Do not
clear the fault lockout or reinterpret process termination as torque evidence.

## B. Conservative first motion

Record the initial limits used for this secured trial. They should be lower
than normal operation and may be raised only by producing a new commissioned
profile.

| Setting                           | Value |
| --------------------------------- | ----: |
| Action rate (Hz)                  |       |
| Action watchdog (ms)              |       |
| First-action deadline (ms)        |       |
| Control deadline (ms)             |       |
| Browser deadline (ms)             |       |
| Max velocity per second           |       |
| Max acceleration per second²      |       |
| First joint                       |       |
| First envelope from observed pose |       |

1. Establish the authenticated session without moving the leader.
2. Confirm the robot shows exactly one owner and one hardware-registry lease.
3. Move only the first joint inside the recorded envelope.
4. Confirm target, admitted action, follower observation, sequence, and latency.
5. Repeat one joint at a time; stop on any schema, direction, scale, or lag
   mismatch.

## C. Fault matrix

For every row, record robot-local elapsed time from last valid proof/action to
dispatch halt, stop completion, close completion, and torque-off readback.

| Injection                                                    | Expected local reason                                                                   | Dispatch halt ms | Stop ms | Close ms | Torque off | Fault lockout/notes |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------- | ---------------: | ------: | -------: | ---------- | ------------------- |
| Stop leader action samples while control/browser remain live | action watchdog                                                                         |                  |         |          |            |                     |
| Disable UDP forwarding while TLS heartbeat stays live        | action watchdog                                                                         |                  |         |          |            |                     |
| Stop TLS heartbeats while valid UDP continues                | control heartbeat timeout                                                               |                  |         |          |            |                     |
| Close the controlling browser tab                            | browser loss/timeout                                                                    |                  |         |          |            |                     |
| Terminate the operator process                               | operator/control loss                                                                   |                  |         |          |            |                     |
| Disconnect operator Tailscale/network                        | action or control loss, whichever trips first                                           |                  |         |          |            |                     |
| Inject packet loss, reorder, and duplicate datagrams         | high-water never rolls back; watchdog stops if freshness expires                        |                  |         |          |            |                     |
| Present a second authenticated operator                      | duplicate session refused; active owner unchanged                                       |              n/a |     n/a |      n/a | n/a        |                     |
| Present an over-budget clock drift                           | clock-sync violation                                                                    |                  |         |          |            |                     |
| Press operator STOP twice                                    | current STOP acknowledged; duplicate is harmless/stale STOP cannot affect a replacement |                  |         |          |            |                     |

Required receipt fields for every stop are:

```text
stop_accepted=true
software_dispatch_halted=true
disable_requested=true
hardware_stop_completed=true
hardware_close_completed=true
torque_off_confirmed=true
fault_lockout=false
faults=[]
```

Any differing field must remain visible and explained. An acknowledged STOP
means the robot accepted the request; it does not by itself prove torque off.

## D. Restart and repeat

| Check                                                                                     | Result |
| ----------------------------------------------------------------------------------------- | ------ |
| Quit both MakerMods processes after a confirmed stop.                                     |        |
| Relaunch both; runtime is disabled, listeners are stopped, and devices are closed.        |        |
| Old session/key/action replay is rejected.                                                |        |
| Existing pairing credential authenticates only if it was not revoked.                     |        |
| Explicit robot enable still requires the same matching commissioning record.              |        |
| Repeat first joint, operator STOP, browser loss, operator-process loss, and network loss. |        |
| Rehearse [rollback.md](rollback.md); firmware/calibration digests remain unchanged.       |        |

## Acceptance

- [ ] All deterministic subprocess tests pass on the contribution checkout.
- [ ] All rows above have robot-local evidence from two sessions separated by a full restart.
- [ ] No secret, private path/address, or personal information appears in PR artifacts.
- [ ] Rollback changed no firmware, servo ID, baud, EEPROM, or calibration.
- [ ] A maintainer can repeat the trial using only the repository and this package.
