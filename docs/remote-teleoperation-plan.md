# Live split-host teleoperation implementation plan

Status: proposed for maintainer review

This plan turns the existing hardware-free `makermodslab.remote-teleop.v1`
foundation into a live, two-laptop SO-101 trial. One laptop owns only the
leader; the other owns only the follower. The robot host remains the sole
authority for follower motion and stops locally when any required liveness
signal disappears.

The work should be reviewed as a stack of small pull requests. If maintainers
prefer one GitHub PR, retain the same commit boundaries and keep live listeners
disabled until the final commissioning gate.

## 1. Outcome and first-live scope

The first live contribution supports:

- one SO-101/Feetech leader on the operator laptop;
- one SO-101/Feetech follower on the robot laptop;
- a robot-authenticated TLS WebSocket control channel;
- an HMAC-authenticated, sequenced UDP latest-value action channel;
- robot-local action and control watchdogs;
- one-time pairing followed by a revocable operator credential;
- two local MakerMods Lab browser flows; and
- a reproducible two-process test and secured-arm field package.

The implementation stays generic at its interfaces, but Maker, Metal, and
bimanual live adapters remain disabled until they have family-specific joint,
stop, de-energization, and commissioning evidence. Raw serial tunneling,
public-internet exposure, cloud authority, remote inference, and servo
maintenance writes are out of scope.

Estimated effort is 18–28 engineering days plus two supervised hardware
sessions. The largest uncertainty is the behavior and maximum blocking time of
the real LeRobot/Feetech connect, read, write, torque-disable, and disconnect
calls.

## 2. Non-negotiable safety and authority invariants

1. **One hardware owner.** Local teleoperation, remote operator, remote robot,
   calibration, recording, replay, inference, and other arm-motion features
   claim one process-wide hardware registry before opening a device. A status
   tracker is not the mutex.
2. **Robot-minted authority.** Only the robot host creates the network session
   id, executor generation, UDP action key, and sequence window. A UDP packet
   cannot acquire authority.
3. **STOP is local and ungated.** STOP is never refused because of a lease,
   role, browser state, or motion-enable flag. The robot burns the generation
   and clears the latest action before touching hardware.
4. **No stale-source parking.** Watchdog, transport-loss, process-error, and
   ordinary STOP paths issue no new position trajectory and never return to a
   rest pose. A future graceful park is a separate, explicit, bounded command.
5. **Honest terminal state.** The system distinguishes STOP accepted, command
   advancement halted, torque-disable requested, disconnect completed, and
   torque-off confirmed. Missing confirmation becomes a fault lockout, not a
   successful stop.
6. **Identity before motion.** The robot independently verifies its follower
   calibration, the configured leader/follower pairing, joint order, units,
   limits, clock budget, authenticated operator, and observed UDP endpoint
   before admitting an action.
7. **Two independent robot watchdogs.** An action deadline and a control-channel
   deadline each stop the robot. UDP cannot keep a dead control session alive,
   and heartbeats cannot keep stale actions alive.
8. **Secrets stay local.** Long-lived credentials, TLS private keys, and
   per-session action keys never enter the repository, robot records, URLs,
   logs, status responses, recordings, or browser storage. The one-time pairing
   token may exist transiently in the two local setup views so it can be
   displayed/scanned, but it is never persisted, logged, or returned by status.
9. **Boot and restart disabled.** A process restart clears runtime enablement,
   invalidates all network sessions and action keys, and requires a new local
   enable action. Pairing survives until explicitly revoked.
10. **Local mode remains the default.** No network listener or live follower is
    opened without an explicit commissioned robot-role configuration and a
    local enable action.

## 3. Topology and responsibility split

| Component              | Owns                                                                                                        | Must never own                                                  |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Operator browser       | Role/calibration selection, latency/health display, STOP intent, local UI liveness                          | Leader device, follower device, operator credential, action key |
| Operator backend       | Local hardware lease, leader adapter, robot credential, control client, raw-sample binding, UDP sender      | Follower device or robot authority                              |
| Robot control service  | Pairing, authentication, clock sync, session state, control heartbeat, status/observations, STOP receipts   | Leader device                                                   |
| Robot UDP receiver     | One bound peer endpoint and one active session/key/generation                                               | Session creation or hardware ownership                          |
| `RemoteExecutor`       | Latest admitted action, robot clock validation, bounds, fixed-rate writes, action watchdog, recording hooks | Network pairing or browser state                                |
| Robot follower adapter | The normal MakerMods/LeRobot follower and its bus for one lease lifetime                                    | Sockets, network credentials, or leader hardware                |

Both browsers talk only to their local MakerMods Lab backend. Only the two
backends communicate across the network.

## 4. Required foundation repairs

These changes land before any live listener is enabled:

- parameterize `RemoteExecutor` status mode instead of returning
  `"simulation"` unconditionally;
- split a raw `LeaderAdapter.read()` result from the session-bound
  `ActionSample` constructed by the operator session client;
- validate joint membership, finite values, position limits, identities, and
  time bounds before atomically advancing the accepted sequence high-water;
- move follower stop/disconnect I/O outside the executor state lock;
- make teardown single-owner, idempotent, exception-safe, and observable;
- keep `RemoteSimulationService` physically separate from every live adapter
  and listener;
- add a bounded, non-blocking JSONL recorder queue rather than doing file I/O
  in the executor loop; and
- replace “connected means active” with the full state machine below.

An authenticated packet that is structurally or physically invalid does not
advance sequence high-water. It increments a reason-specific rejection counter.

## 5. Prerequisite: one atomic hardware registry

### Why this is PR 0

MakerMods Lab currently has feature-specific active flags and locks.
`SessionTracker` observes those flags; it does not atomically arbitrate hardware
across feature modules. Adding `remote_teleoperation_active` under another
feature-local lock would preserve the existing check-then-claim race.

Add `makermodslab/hardware_lease.py` with one `HardwareLeaseRegistry` and one
process-wide `RLock`.

```python
claim(kind, owner, resource="arm_hardware") -> HardwareLeaseToken
request_stop(token, reason) -> StopClaim
release(token, receipt) -> None
mark_unresolved(token, reason, receipt) -> None
snapshot() -> HardwareLeaseSnapshot
```

The token carries an unguessable id plus a monotonically increasing lease
generation. Existing feature flags remain compatibility/status projections,
not authority.

Every affected start path follows the same order:

1. validate its request without opening hardware;
2. atomically claim the registry;
3. set its legacy active/status projection;
4. open hardware and start its worker;
5. on startup failure, close any partial hardware, clear status, then release;
6. on STOP, request teardown without releasing the registry; and
7. release only in the hardware worker's finalizer after close and a safe
   receipt.

An expired session or heartbeat requests the owner's normal STOP callback. It
must never silently release a lease while a hardware worker may still be
running. An unconfirmed stop marks the lease unresolved and blocks all new
motion until evidence-backed recovery.

Migrate the start/finalizer paths for local teleoperation, recording, replay,
inference, calibration, auto/zero calibration, and any recovery or diagnostic
operation that opens or commands an arm. Update `sessions.py::held_by()` and
session status to read the registry first.

**PR 0 gate:** a barrier-based threaded test races every pair of start kinds;
exactly one claim wins, no loser opens hardware, STOP remains callable, and the
winner cannot release before its mocked device closes. All legacy status and
session tests remain green.

## 6. Robot-host adapter

Add `makermodslab/remote_teleop/adapters/lerobot_follower.py` and follower-only
config helpers in `makermodslab/utils/robot_factory.py`.

The current pair-building helpers stage both leader and follower calibrations.
The robot laptop has no leader port, so add explicit single-side helpers rather
than fabricating a missing leader:

```python
build_follower_config(record, *, cameras=None)
build_leader_config(record)
```

The first live adapter exposes a narrow lifecycle:

```python
class LiveFollowerDriver(Protocol):
    joint_schema: JointSchema
    device_identity: DeviceIdentity

    def connect_safe(self) -> InitialFollowerState: ...
    def enable_for_action(self) -> None: ...
    def observe(self) -> Mapping[str, float]: ...
    def execute(self, positions: Mapping[str, float]) -> Mapping[str, float]: ...
    def disable_torque(self, reason: str) -> StopHardwareReceipt: ...
    def close(self) -> CloseReceipt: ...
```

`connect_safe()` must either leave torque disabled or fail. If the installed
LeRobot class cannot prove that property, the adapter may not enter
`READY_WAITING_ACTION`. The first valid action enables execution locally; no
network message directly toggles torque.

Joint normalization is derived from the concrete SO-101 action contract, not
from observation keys by assumption. The adapter publishes an ordered
`JointSchema` containing canonical joint name, exact LeRobot action key, unit,
and hard/commissioned limit. It rejects missing, extra, duplicate, boolean,
non-finite, or differently ordered values.

Only this adapter imports the concrete SO-101 follower class within
`remote_teleop`. Add a static test that prevents other remote-teleoperation
modules from instantiating or directly commanding LeRobot hardware.

### Calibration and rig identity

Add `makermodslab/remote_teleop/calibration_identity.py`.

`calibration-digest-v1` is SHA-256 over strict canonical JSON: UTF-8, sorted
keys, compact separators, finite numbers only, and no duplicate keys. The robot
computes the follower digest from its local artifact. The operator computes the
leader digest from its local artifact. The robot accepts the leader digest only
if it matches a locally configured pairing entry.

The robot independently derives `rig_digest` from:

- protocol and digest algorithm versions;
- arm family and single/bimanual topology;
- ordered joint schema and units;
- leader and follower calibration ids/digests; and
- robot-enforced position, velocity, and acceleration limits.

The request never supplies authoritative limits or a trustworthy follower
digest. Mismatches return stable coded errors naming the failed identity class
without returning secret material.

### Robot STOP sequence

One teardown caller wins an atomic stop claim:

1. transition to `STOPPING`;
2. burn the executor generation and invalidate the action key;
3. clear the latest action and prevent further writes;
4. stop the UDP receiver from dispatching;
5. release the executor lock;
6. request torque disable through the follower adapter;
7. close the follower in `finally`;
8. record the stop/close receipts; and
9. release the hardware lease only when the safe terminal condition is
   confirmed; otherwise mark it unresolved and enter `FAULT_LOCKOUT`.

No return-to-rest or other new trajectory occurs in this sequence.

**Robot-adapter gate:** fake-adapter tests cover every lifecycle failure point;
an SO-101 secured-arm test proves device identity, initial observation, bounded
commands, action-watchdog behavior, torque-disable behavior, close, and the
honesty of the stop receipt.

## 7. Operator-host adapter

Add `makermodslab/remote_teleop/adapters/lerobot_leader.py` and
`makermodslab/remote_teleop/operator_service.py`.

The adapter owns only the local leader and returns raw provider-neutral values:

```python
@dataclass(frozen=True)
class RawLeaderSample:
    positions: Mapping[str, float]
    sampled_monotonic_ns: int

class LeaderAdapter(Protocol):
    joint_schema: JointSchema
    calibration_identity: CalibrationIdentity

    def connect(self) -> None: ...
    def read(self) -> RawLeaderSample: ...
    def close(self) -> None: ...
```

`OperatorSessionClient` owns the robot grant, sequence counter, time mapping,
action lifetime, and action key. It maps the raw leader schema to the exact
robot-granted order, constructs `ActionSample`, calls the existing canonical
encoder, and sends the datagram. The leader adapter never sees a robot session
id or key.

The operator backend claims its local hardware registry before connecting the
leader. On any pairing, control, schema, calibration, or robot-session failure,
it stops its action loop, closes the leader, and releases the local lease.

## 8. Control channel, authentication, and clock negotiation

### Transport and messages

Use the already-declared `websockets` dependency for a dedicated TLS WebSocket
listener bound to the explicitly configured private interface. Do not expose
the main MakerMods HTTP server merely to obtain the control socket.

Add:

- `remote_teleop/control_protocol.py` — strict versioned message values;
- `remote_teleop/control_server.py` — robot listener and state machine;
- `remote_teleop/control_client.py` — operator client; and
- `remote_teleop/clock_sync.py` — pure clock math and acceptance.

Control frames are strict JSON, capped at 64 KiB, and include protocol version,
message type, and request id. Session-bound messages also include the exact
session id and generation. The v1 message set is:

- `hello` / `hello_ack`;
- `pair` / `pair_ack`;
- `clock_probe` / `clock_reply`;
- `session_open` / `session_grant`;
- `udp_probe_ready` / `udp_endpoint_bound`;
- `heartbeat` / `heartbeat_ack`;
- `status` and rate-capped latest observations;
- `stop` / `stop_ack`; and
- structured `error`.

Reliable status and observations use a one-slot queue at no more than 10 Hz so
TCP backpressure cannot delay STOP or heartbeat handling.

### TLS and pairing

Core code accepts configured certificate and private-key paths; it does not
invent a certificate authority. The field guide documents and verifies two
supported sources: a Tailscale/MagicDNS certificate and a locally generated
self-signed certificate whose fingerprint is transferred out of band.

Pairing is allowed only while a loopback request from the robot laptop opens a
short-lived pairing window. The robot shows a QR/manual payload containing:

- robot address and control port;
- SHA-256 server-certificate fingerprint; and
- a random 256-bit, single-use pairing token.

The operator pins the fingerprint before sending the token. The robot rate
limits attempts, expires the window, consumes the token once, and issues a
random 256-bit operator credential with a public credential id. Store only the
credential hash and metadata on the robot; store the raw credential on the
operator. Both files use the platform's private application-data directory and
owner-only permissions. The robot UI can revoke the credential.

Pairing persists across ordinary reconnects until revoked. Every reconnect
still receives a new robot session id, generation, per-session action key, and
sequence window.

The operator credential authenticates the TLS control session. The per-session
action key is delivered only over that channel and authenticates UDP actions.
Tailscale connectivity or encryption alone never grants action authority.

### Clock synchronization

Monotonic clocks on different hosts are unrelated. During negotiation, collect
16 NTP-style samples:

```text
t0 = operator send
t1 = robot receive
t2 = robot send
t3 = operator receive

robot_minus_operator_offset = ((t1 - t0) + (t2 - t3)) / 2
rtt = (t3 - t0) - (t2 - t1)
uncertainty = ceil(rtt / 2) + scheduler_margin
```

Discard negative/invalid samples and select the valid sample with the lowest
RTT. The default session-open ceiling is 50 ms uncertainty. A large constant
offset is acceptable; high RTT, jitter, asymmetric uncertainty, or drift is
not.

Freeze the selected offset and uncertainty for the session. Periodic probes
check that a new uncertainty interval remains within the configured budget.
They never silently remap an active session. Exceeding the budget stops the
session and requires a new one.

## 9. UDP action lane

Extend `remote_teleop/transport.py` or add
`remote_teleop/udp_transport.py` with:

- `UdpActionReceiver` on the robot;
- `UdpActionSender` on the operator; and
- a deterministic fault-injecting proxy used only by tests.

The robot binds only the exact configured local private/tailnet address. Reject
wildcard, public, multicast, or unassigned addresses. Loopback is accepted only
for simulation and process tests.

After the control grant, the operator sends an authenticated UDP probe from the
same socket it will use for actions. The robot observes and binds the exact
source IP/port plus session id, generation, and key id. Other endpoints cannot
submit to that session.

The receiver reads at most `MAX_DATAGRAM_BYTES + 1`, drops oversize data before
JSON decoding, verifies HMAC in constant time, and rate limits invalid packets
per source. There is one receive path and one latest-value slot; packets are
never buffered for reordering.

Trial defaults, all bounded by config validation:

- executor rate: 50 Hz;
- action watchdog: 200 ms;
- first-action deadline after grant: 1 s;
- action source-age budget: 150 ms;
- maximum encoded action lifetime: 250 ms; and
- control heartbeat deadline: 1 s; and
- operator-browser liveness deadline: 2 s.

The operator backend sends control heartbeats more frequently than the deadline.
The operator browser also owns a local UI-liveness WebSocket; closing or losing
the controlling tab stops the operator action loop. The robot then trips its
action watchdog even if the control socket is otherwise healthy.

## 10. Session state machines

### Robot

```text
DISABLED
  -> IDLE
  -> PAIRING (first enrollment only; no hardware)
  -> AUTHENTICATED (no hardware)
  -> NEGOTIATING (clock + identities; no hardware)
  -> CLAIMING (atomic hardware lease)
  -> PREPARING_FOLLOWER (connect-safe + observation)
  -> READY_WAITING_ACTION (torque disabled; grant issued)
  -> ACTIVE (first valid action enables execution)
  -> STOPPING
  -> IDLE                 confirmed safe close
  -> FAULT_LOCKOUT        unconfirmed stop/close or unknown hardware state
```

Every failure after `CLAIMING` runs the same teardown path. `FAULT_LOCKOUT`
retains an unresolved registry claim and exposes only status, STOP retry, and a
future evidence-backed recovery flow.

### Operator

```text
IDLE
  -> CONTROL_CONNECTED
  -> AUTHENTICATED
  -> CLOCK_SYNCED
  -> LEADER_CLAIMED
  -> LEADER_READY
  -> SESSION_OPENING
  -> STREAMING
  -> STOPPING
  -> IDLE or FAULT
```

The operator stops reading/sending before it asks the robot to stop. If STOP
acknowledgement is lost, it reports “robot confirmation unavailable”; it never
claims the remote arm is safe. The robot-local watchdog remains authoritative.

### STOP acknowledgement

`stop_ack` includes:

- request id, session id, and burned generation;
- stop reason and robot monotonic acceptance time;
- action dispatch disabled;
- torque-disable requested;
- torque-off confirmation status and evidence source;
- follower close status;
- lease released or retained unresolved; and
- final state/fault code.

A stale session's STOP cannot stop a newer session. It returns a stale-session
receipt without changing current authority. The robot's local STOP button,
however, always targets the currently active session and bypasses remote
session ownership.

## 11. Local configuration, APIs, and UI

### Configuration

Add a platform-aware `remote_teleop` application-data directory with separate:

- non-secret role/node configuration;
- TLS certificate/key references;
- operator credential material; and
- bounded session recordings.

Do not store network credentials in robot records. A robot record may store the
non-secret expected leader calibration id/digest and rig pairing.

Robot configuration includes its configured role, exact bind
interface/address, ports, follower record/calibration, TLS paths, permitted
credential ids, watchdog/time budgets, and recording policy. Operator
configuration includes robot address, pinned certificate fingerprint,
credential id/secret reference, leader record/calibration, and local action
rate. Runtime motion/listener enablement is separate and is never persisted.

Missing, ambiguous, unreadable, wildcard, or permission-unsafe configuration
fails closed. Runtime enablement is cleared on startup even when the persisted
role remains configured.

### Typed local API

Keep all new HTTP surface under `/api/v1`. Extend the existing remote status
route and add typed local-only operations for:

- read/update non-secret local configuration;
- open/close a robot-local pairing window;
- pair/revoke an operator;
- start the configured operator or robot role through the normal session API;
- browser UI-liveness;
- current status/fault/metrics; and
- unconditional local STOP.

The dedicated remote TLS WebSocket is not a browser API. Update the v1 route
ratchet, response models, and committed OpenAPI snapshot in the same commit.

### Operator UI

1. Select **Remote operator**.
2. Select the local leader and calibration.
3. Enter/scan the robot pairing payload or choose an existing paired robot.
4. Connect and verify displayed certificate, calibration, joint schema, clock
   uncertainty, latency, and watchdog state.
5. Start streaming and keep the controlling tab open.
6. Provide a fixed STOP button and keyboard STOP shortcut.

### Robot UI

1. Select **Remote robot**.
2. Select the follower and its calibration.
3. Select an exact private interface; never offer `0.0.0.0` as a shortcut.
4. Configure the expected leader calibration pairing.
5. Open pairing only by a local action.
6. Display owner/credential id, state, watchdog remaining, last sequence,
   packet rejections, clock uncertainty, follower/torque receipt, faults, and
   STOP state.
7. Keep the local STOP control visible in every live state.

Do not overload the existing local teleoperation start/unmount behavior. Remote
operator and robot views have their own explicit lifecycle while reusing common
status components.

## 12. Recording and diagnostics

Add `remote_teleop/recording.py` with a bounded queue and one writer thread. A
session JSONL file contains a versioned header, bounded events, and a terminal
receipt. It records:

- robot monotonic receive/execute/observe times;
- frozen clock mapping and uncertainty;
- source, admitted, executed, and observed positions;
- sequence and rejection counters;
- watchdog/control transitions; and
- STOP/fault receipts.

It does not record action keys, credentials, pairing tokens, TLS key paths,
full network addresses, or browser data. Queue overflow increments an explicit
counter and drops diagnostics; it never delays execution or STOP.

Servo health remains owner-fed. The live follower adapter may advance the
existing sampler only while it owns the bus; HTTP handlers remain cache-only.

## 13. Pull-request sequence and gates

### PR F — current hardware-free foundation

Contracts, robot-minted authority, deterministic executor/simulation, HMAC
encoding, read-only servo health, API/status panel, docs, and tests. No live
listener or hardware adapter.

### PR 0 — atomic hardware ownership and executor hardening (3–5 days)

- central registry and all feature migrations;
- stop/finalizer/unresolved semantics;
- executor validation ordering and exception-safe teardown;
- mode/status and recorder-queue fixes.

Gate: mutual-exclusion race matrix, startup-unwind matrix, all existing tests.

### PR 1 — SO-101 robot adapter (3–4 days)

- follower-only config builder;
- SO-101 joint schema and calibration/rig identities;
- connect-safe/enable/execute/disable/close receipts;
- fake lifecycle tests and static adapter-boundary test.

Gate: no socket, listener, or automatic live enablement; secured-arm adapter
proof completed before the next live gate.

### PR 2 — SO-101 operator adapter (2–3 days)

- leader-only config builder;
- raw leader adapter;
- operator service and session-bound encoding;
- schema/calibration mismatch and cleanup tests.

Gate: deterministic leader-to-encoder round trip with no follower attached.

### PR 3 — TLS control, pairing, clock, and state machines (4–6 days)

- strict control protocol and TLS server/client;
- pinned-certificate, one-time pairing, credential persistence/revocation;
- NTP-style clock negotiation and frozen mapping;
- control heartbeat, STOP acknowledgement, and fault transitions.

Gate: two-process authenticated control test, secret/redaction/permission tests,
clock jitter/asymmetry tests, and control-loss stop test. Listener still disabled
by default.

### PR 4 — UDP action transport and observability (3–4 days)

- endpoint-bound UDP receiver/sender;
- HMAC/size/rate/sequence/time enforcement;
- action watchdog and browser-liveness propagation;
- robot-side bounded recording and metrics.

Gate: deterministic loss/reorder/duplicate/spoof/oversize/stale/future fault
matrix with maximum stale-execution assertions.

### PR 5 — two-laptop UI/configuration (3–4 days)

- role selection and private-interface validation;
- pairing and calibration flows;
- operator and robot status views;
- local and remote STOP controls; and
- typed APIs/OpenAPI/client updates.

Gate: frontend state/secret-boundary tests and full local simulation flow.

### PR 6 — field package and commissioning (2–4 days plus hardware sessions)

- two-instance process harness;
- exact install/launch commands for the merged CLI and current platforms;
- current official Tailscale ACL/firewall examples;
- secured-arm worksheet;
- rollback/uninstall procedure; and
- two supervised SO-101 sessions separated by a full stop/restart.

Gate: every mandatory acceptance row below has an attached result; live enable
remains unavailable when the field profile has not been commissioned.

### Later family PRs

Maker, Metal, and bimanual support are separate PRs. Each adds its own schema,
limits, de-energization receipt, drop/gravity assessment, and hardware test. No
family inherits SO-101 safety claims by interface compatibility alone.

## 14. Test and acceptance matrix

| Case                         | Deterministic process gate                                                                             | Hardware commissioning gate                                                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Normal session               | Two local MakerMods instances exchange real TLS WS and UDP using simulated devices                     | Leader/follower track within commissioned bounds                                                                               |
| UDP loss                     | Deterministic burst patterns; no command executes after its age/deadline; watchdog may truthfully stop | Pull/disable UDP path; robot stops inside measured budget                                                                      |
| Reorder/duplicate            | High-water never rolls back; duplicate does not execute twice                                          | Observe counters only                                                                                                          |
| Endpoint spoof               | Wrong IP/port/session/key/generation rejected                                                          | Optional second tailnet host probe                                                                                             |
| Control loss                 | Continue valid UDP while killing WS; control watchdog stops                                            | Disable control path; local stop confirmed                                                                                     |
| Browser loss                 | Drop operator UI-liveness while backend/network stay up                                                | Close controlling tab; action stream and robot stop                                                                            |
| Operator process/laptop loss | Kill process; robot action watchdog stops                                                              | Power off/sleep operator laptop                                                                                                |
| Tailscale/network loss       | Kill both channels; local robot state reaches stopped/fault without UI                                 | Disconnect tailnet/Wi-Fi                                                                                                       |
| Clock offset                 | Large constant offset accepted when uncertainty is low                                                 | Compare displayed mapping with observed latency                                                                                |
| Clock jitter/asymmetry/drift | Budget excess refuses open or stops; mapping never changes in place                                    | Network impairment test if available                                                                                           |
| Concurrent sessions          | Barrier-raced opens; exactly one robot lease/session wins                                              | Try a second operator laptop                                                                                                   |
| Partial startup              | Inject failure at every state; no leaked socket, key, device, or releasable lease                      | Unplug follower during preparation                                                                                             |
| STOP duplicate/stale         | Idempotent current STOP; stale STOP cannot hit replacement session                                     | Repeated button and reconnect cycle                                                                                            |
| Robot restart                | Old key/generation rejected; runtime enable cleared; unresolved journal reconciled                     | Full stop, restart, confirm disabled before re-enable                                                                          |
| Secret handling              | Logs/status/OpenAPI/recordings clean; files owner-only; malformed state fails closed                   | Inspect installed permissions                                                                                                  |
| Blocking adapter             | Hanging fake call cannot be reported as a confirmed stop                                               | Measure real SDK call maxima; if stop budget is not enforceable, require a killable worker architecture before live enablement |

Process tests use injected monotonic clocks, not changes to the laptops' wall
clocks. Packet corruption is distinct from reordering. Percentage loss tests do
not promise uninterrupted operation; they prove bounded stale execution and
truthful watchdog behavior.

## 15. Field-test package

Land these public artifacts with PR 6:

```text
docs/remote-teleop/two-laptop-quickstart.md
docs/remote-teleop/network-and-tailscale.md
docs/remote-teleop/commissioning-worksheet.md
docs/remote-teleop/troubleshooting.md
docs/remote-teleop/rollback.md
examples/remote-teleop/robot.example.json
examples/remote-teleop/operator.example.json
tests/test_remote_teleop_two_process.py
tests/test_remote_teleop_runtime_process.py
tests/helpers/remote_teleop_process_peer.py
tests/helpers/remote_teleop_runtime_service_peer.py
```

The quickstart contains commands that were executed from a clean isolated
macOS arm64 checkout. Linux wheel support and firewall policy are documented
for maintainer validation, but are not represented as contributor-executed.
Do not publish guessed systemd, firewall, or Tailscale commands. Link current
official Tailscale documentation and include tested ACL examples with
placeholders, while stating that ACLs restrict network reachability and do not
replace application authentication.

The secured-arm worksheet requires, in order:

1. follower physically secured against gravity/drop and workspace cleared;
2. physical power removal or E-stop reachable by the robot-side tester;
3. local leader and follower calibration independently proven;
4. follower device identity and joint schema captured;
5. no-motion connect and torque state verified;
6. local robot STOP proven before remote action;
7. minimum speed/acceleration/current profile selected;
8. first action performed one joint at a time inside a small envelope;
9. UDP, control, browser, operator-process, and network loss injected;
10. stop/torque/close receipts reviewed rather than inferred;
11. full process restart shows runtime motion disabled; and
12. second session repeats the critical gates after a clean restart.

## 16. Rollback and uninstall

Rollback is recoverable and does not delete calibration or robot records:

1. press robot-local STOP and review the receipt;
2. if torque-off is unconfirmed, use physical power removal and retain the fault
   lockout;
3. stop the operator action loop and close its leader;
4. disable the remote role on both laptops;
5. revoke the paired operator credential on the robot;
6. stop the two MakerMods Lab processes;
7. preserve remote configuration and logs as a timestamped backup;
8. remove only the optional service/autostart entries created by the field
   guide; and
9. relaunch the prior/local-only build and prove local teleoperation still owns
   the hardware registry correctly.

The feature makes no firmware, servo-id, baud, EEPROM, or calibration mutation,
so rollback never needs to reverse one.

## 17. Adapted prior-art patterns and deliberate non-copies

Useful patterns from an existing open-source robot authority are adapted here:

- one mechanically testable hardware adapter boundary;
- one-slot latest-value command delivery;
- session sequence, generation, and monotonic expiry;
- runtime motion gates cleared on restart;
- STOP accepted versus measured-confirmed receipts;
- a retained unresolved lease and durable fault latch;
- startup reconciliation of an operation with no terminal receipt;
- injected hanging hardware and one-owner static tests; and
- fail-closed secret and state-file handling.

Do not copy its device-specific STOP behavior, local bearer as remote pairing,
or in-process lease assumptions. The first implementation initially kept the
follower in process, then an injected blocking-call proof showed that an
uncancellable LeRobot/Feetech call could outlive the required STOP budget. The
live follower therefore runs behind a killable worker supervised by the robot
service. Killing that worker proves only that further software dispatch has
halted; it never proves torque-off, and it retains a durable fault lockout until
a matching local recovery confirms stop, close, and torque state. Real hardware
commissioning must still measure the normal and worst observed call budgets.

## 18. Adversarial review decisions

The independent proposal and counter-review resolved these material points:

- a central atomic hardware registry is a prerequisite; adding another active
  flag is not race-proof;
- watchdog STOP cannot move toward a rest pose;
- an enrollment flow without a one-time out-of-band authorization authenticates
  nobody;
- clock uncertainty is at least half the measured RTT plus scheduling margin;
- constant clock offset is not clock uncertainty;
- the clock mapping remains frozen for one session;
- leader hardware produces raw samples; the operator session binds authority;
- pairing credentials persist until revocation, while action credentials rotate
  every session;
- CI/process claims and physical hardware claims stay separate; and
- the first live scope is one SO-101 pair, with later arm families gated by
  their own evidence.

## 19. Definition of done

The live SO-101 contribution is done only when:

- all seven PR gates pass and the full existing backend/frontend suites remain
  green;
- two clean local instances pass the authenticated TLS/UDP fault matrix;
- no listener starts and no device opens by default;
- every feature uses the central hardware registry;
- the robot stops locally on action, control, browser, operator-process, and
  network loss;
- STOP and torque state are reported honestly, including fault lockout;
- secrets and private paths are absent from the contribution;
- exact two-laptop commands have been executed from the published guide;
- rollback has been rehearsed without changing firmware/calibration; and
- MakerMods maintainers can reproduce the secured-arm trial without access to
  the contributor's machines or network.
