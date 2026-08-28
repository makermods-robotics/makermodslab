# makermodslab-sdk behavior spec

The language-agnostic contract for this SDK and every port of it. The wire
surface is `docs/api/openapi.json` at the repo root; this file specifies the
BEHAVIOR a client must layer on top. The Python SDK in this directory is the
reference implementation; a port re-implements layer 1 idiomatically and
transcribes layers 2–3 (see README.md for the layering).

Written for the next port's author — who may well be an AI agent. Sections
marked *(reference)* name where the Python implementation lives.

## 1. Transport

- Base URL is the server root (default `http://<host>:8000`); every path in
  the OpenAPI snapshot is absolute from there. Only `/api/v1/...` paths may
  be used — the flat mount is legacy surface for other clients.
- A 2xx response is parsed JSON; 204 or an empty body is "no value".
- A non-2xx response MUST become a typed error carrying: HTTP status, the
  body's `detail` (normalized, §2), the body's `code` and `details` when
  present, and a remediation looked up from the code (§3). A body that isn't
  JSON still produces the error with status alone.
- A connection-level failure (no HTTP response) is a distinct error type and
  its message must name the base URL and say what to check.
- *(reference: `_transport.py`, `errors.py`)*

## 2. Error decoding

- `detail` may be: a string (most errors); a list of `{loc, msg, type}`
  objects (FastAPI 422s) — join the `msg` values with `"; "`; any other JSON
  — serialize it. Never surface a raw object repr.
- `code` follows `<domain>.<condition>[.<detail>]` (see
  `makermodslab/api_errors.py`; the domain set is closed). Branch on `code`
  or on error type — NEVER on the prose, which the server may reword.
- Typed subclasses (minimum set): `*.not_found` → NotFound;
  `request.*` or HTTP 422 → InvalidRequest; `robot.busy.*` → RobotBusy
  (expose the third segment as the discriminant); `session.held` →
  SessionHeld (expose `details.holder` = `{kind, session_id}`).
- Servers at older snapshots emit some errors uncoded (e.g. bare 404s);
  classification must degrade to the generic API error, never crash.

## 3. Remediation (agent-first errors)

- The SDK ships a table mapping codes (and code families, by prefix; exact
  entries win) to a one-sentence "next step" naming the literal next call.
  The rendered error text is `"<action> failed (<status>, <code>): <detail>"`
  plus `"Next step: <remediation>"` when one exists.
- Every request carries a human `action` label ("Start teleoperation
  session") — it is the first thing the reader of a failure sees.
- Method names used inside remediation texts are CONTRACT: implementations
  must provide them (`client.sessions.stop_current()`,
  `client.jobs.list()`, `client.system.hf_login(...)`, …).
- *(reference: `errors.py` REMEDIATIONS / FAMILY_REMEDIATIONS)*

## 4. Compatibility handshake

- Lazily, before the first real request, fetch `GET /api/v1/health` once.
  Warn — never fail — when the endpoint is missing or `version` parses lower
  than the SDK's minimum supported server version. Connection errors
  propagate (the real request would hit them too).
- Response models are tolerant everywhere: unknown keys are kept, never
  rejected (an older SDK must survive a newer server).
- *(reference: `client.py`)*

## 5. Sessions and the lease

- `POST /api/v1/sessions` starts any robot flow by robot RECORD NAME; the
  server resolves ports/configs/cameras. Kinds: teleoperation, recording,
  inference, replay, calibration, auto_calibration.
- Starting with an `owner` attaches a lease. Renewal is the owner's act
  alone: `POST /sessions/{id}/heartbeat` with the owner; reads never renew.
  Miss the heartbeats and the server's watchdog safety-stops the session
  (default timeout 60s; auto_calibration 90s).
- The SDK default is owner-attached (`sdk:<hostname>:<pid>:<token>`), with a
  background renewal at ~timeout/3. An abandoned client process must never
  leave an energized arm running — that is the lease's entire point.
- Stop is deliberately NEVER owner-gated (safety outranks ownership). A 404
  `session.not_found` on stop means already-gone: treat as success.
- 409 `session.held` carries `details.holder`; the client-side recovery is
  stop-current-then-retry, surfaced through the remediation, never done
  automatically.
- A 201 start response may carry `warnings` (warn-but-allow findings, e.g.
  arm identity). The session RUNS; the warnings must be surfaced verbatim
  (server prose, never localized/reworded).

## 6. Realtime hints

- One WebSocket (see the snapshot / server.py for the v1 path) carries joint
  telemetry plus typed control events: `jobs_changed`, `job_progress`,
  `session_changed`.
- Control events are DROPPABLE REFETCH HINTS: on receipt, refetch the
  resource; never treat the event payload as state. A missed event
  self-heals on the next fetch. Blocking waiters may use hints to wake early
  but must confirm terminal state via GET, and must work (by polling) when
  the realtime channel is unavailable.
- Unknown message types must be surfaced as an "unknown event" (forward
  compatibility), not an error.

## 7. Waiters

- Long-running work (jobs, downloads/uploads/merges, install extras) gets a
  blocking `wait`-style helper with a `timeout` and an injectable
  sleep/clock. Agents should never be made to write polling loops.
- Streams handed to agents are bounded by default (`sample_joints(duration)`
  returning a list); unbounded iterators are the explicitly-named variant.

## 8. Full-power principle

- The SDK exposes the BACKEND's full surface, not the web UI's subset — the
  UI narrows deliberately; the SDK must not. Where a request model is wider
  than the UI's form (training's ~45 knobs, chain-rewind lineage fields),
  every user-settable field is first-class, typed SDK surface.
- Client-side knob validation is strict (unknown field → immediate error
  naming close matches, nothing sent), with an explicitly-named unvalidated
  escape hatch for fields newer than the SDK.
- Request parity is RATCHETED: a test equality-asserts the SDK's field set
  against the server's request model minus a reasoned exclusion list
  (server-managed internals), so a new backend knob fails the build until
  typed. *(reference: test_jobs.py training-options parity test)*
- Server-managed fields (set by the registry/runners, never by clients) are
  excluded ON PURPOSE and each exclusion carries its reason.

## 9. Coverage discipline

- Every tagged v1 operation in the snapshot is either implemented (tied to
  its `operationId`) or listed as planned; the check is equality-asserted in
  both directions and the planned set only shrinks.
  *(reference: `tests/test_coverage_ratchet.py`)*
- The reference test harness runs the SDK against the real FastAPI app
  in-process. Ports without that luxury must test against recorded
  fixtures generated from the same snapshot.
- Tests never sleep, never touch the network, and NEVER call endpoints that
  energize hardware, write servo EEPROM, or start subprocesses.
