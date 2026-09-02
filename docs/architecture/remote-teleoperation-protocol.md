# Remote teleoperation protocol and configuration

The v1 live scope is one SO-101 leader/follower pair on two laptops. Local
teleoperation remains the default. The remote runtime is dormant until a local
user saves a role and explicitly enables it; the robot additionally requires a
matching secured-arm commissioning record.

## Control channel

Pinned TLS WebSocket carries one-time pairing, credential authentication,
NTP-style monotonic clock samples, the robot-authored profile, session open,
heartbeat, status, UDP endpoint proof, and acknowledged STOP. The robot accepts
one credential/session owner. Reconnect creates a new session id, executor
generation, action key, and high-water mark.

The robot binds only the configured exact private IP. Wildcard, public,
hostname, and default-listener shortcuts are refused. Pairing opens only from a
robot-local request; Tailscale reachability alone grants no action authority.

## Action datagram

Each canonical JSON datagram is at most 4096 bytes and contains:

- protocol/message versions and robot-minted session/generation;
- source, rig, leader calibration, and follower calibration identities;
- an unsigned monotonic sequence and bounded source-clock expiry;
- exact ordered joint names/units and finite positions; and
- key id plus HMAC-SHA256 over the unsigned canonical body.

The receiver binds the first authenticated probe to its source endpoint. It
rejects unknown keys/sessions/generations, endpoint changes, oversized or
malformed bodies, invalid MACs, mismatched identity/schema, stale/future/
expired times, and sequences at or below the accepted high-water mark. Loss is
tolerated; reordering is never buffered. The executor consumes one latest slot
at a fixed rate under robot-side bounds.

## Local API lifecycle

- `GET /api/v1/arms/remote-teleoperation` — redacted status only.
- `PUT` or `DELETE .../configuration` — save/remove dormant role config.
- `POST .../commission` — local no-motion secured-arm proof.
- `POST .../recover-hardware` — evidence-backed durable-fault recovery.
- `POST .../enable` and `.../disable` — explicit runtime lifecycle.
- `POST .../pairing-window`, `.../pair`, and credential revocation — local
  authority management.
- `POST .../browser-heartbeat` — operator-tab liveness.
- `POST .../stop` — fail-safe STOP; not blocked by management authentication.

All management routes except STOP require a loopback caller. Simulation routes
remain available for deterministic tests and cannot discover/open hardware.

## Commissioning and restart

Commissioning binds the follower device-contract digest, follower and leader
calibration identities, rig, ordered joints/units, and enforced limits. The
follower is connected disarmed while physically secured, observed, stopped,
checked for Feetech torque-off readback, and closed. Saving robot configuration
invalidates the prior record; a runtime restart never restores enablement.

If stop/close/torque evidence is incomplete, an owner-private path-free record
is written and the central lease remains unresolved. On restart, the journal
reclaims the lease before another feature can open the arm. Only a local
recovery using the same commissioned profile and full safe-close evidence can
clear it.

## Servo health

The live follower child may sample one read-only register group while it owns
the bus and republish that snapshot to the parent. `GET
/api/v1/arms/servo-health` reads only the cache. Missing values remain null and
communication failures expose a sanitized error class, not vendor exception
text or device paths.
