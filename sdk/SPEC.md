# makermodslab-sdk behavior spec

The language-agnostic contract for this SDK and every port of it. The wire
surface is `docs/api/openapi.json` at the repo root; this file specifies the
BEHAVIOR a client must layer on top. The Python SDK in this directory is the
reference implementation; a port re-implements layer 1 idiomatically and
transcribes layers 2–3 (see README.md for the layering).

Written for the next port's author — who may well be an AI agent. Sections
marked _(reference)_ name where the Python implementation lives.

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
- _(reference: `_transport.py`, `errors.py`)_

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
- _(reference: `errors.py` REMEDIATIONS / FAMILY_REMEDIATIONS)_

## 4. Compatibility handshake

- Lazily, before the first real request, fetch `GET /api/v1/health` once.
  Warn — never fail — when the endpoint is missing or `version` parses lower
  than the SDK's minimum supported server version. Connection errors
  propagate (the real request would hit them too).
- Response models are tolerant everywhere: unknown keys are kept, never
  rejected (an older SDK must survive a newer server).
- _(reference: `client.py`)_

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
- A heartbeat 404 after a finite session can mean normal completion. Read
  `sessions.current().last_ended`: a matching id with null `reason` is an end
  (including `phase="error"`), while `session.lease_expired` is a loss. A
  mismatched or absent end summary remains a loss. A blocking session waiter
  confirms terminal state by matching this id, not merely by seeing idle.

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
  typed. _(reference: test_jobs.py training-options parity test)_
- Server-managed fields (set by the registry/runners, never by clients) are
  excluded ON PURPOSE and each exclusion carries its reason.

## 9. Coverage discipline

- Every tagged v1 operation in the snapshot is either implemented (tied to
  its `operationId`) or listed as planned; the check is equality-asserted in
  both directions and the planned set only shrinks.
  _(reference: `tests/test_coverage_ratchet.py`)_
- The reference test harness runs the SDK against the real FastAPI app
  in-process. Ports without that luxury must test against recorded
  fixtures generated from the same snapshot.
- Tests never sleep, never touch the network, and NEVER call endpoints that
  energize hardware, write servo EEPROM, or start subprocesses.

## 10. Client-side flows

- `client.flows` composes existing methods; its methods have no operation IDs
  and are outside the tagged-operation coverage ratchet. Each docstring names
  its primitives, and the docs index points to a separate flows card.
- Recording a requested episode count owns a leased session. Its timeout
  stops that session; a normal early end with fewer saved episodes is a
  partial-result error. Per-episode prompts follow `current_episode`, because
  a pending take can leave `saved_episodes` unchanged at the next naming gate.
  Prompts and status use session-id-scoped server operations; a stale caller
  must never prompt or report a later recording. The result returns the
  server-stamped dataset id, terminal status (including warning/error detail),
  and start warnings. An expired flow must not submit another episode prompt.
- Local train-and-publish composes job creation/wait, checkpoint inspection,
  publish start/status. Training or publish timeout does not cancel server
  work; errors carry the job id for resumption. A failed or interrupted job
  never starts publishing. Status must match both the local job id and target
  repository before a flow reports success. Each accepted publish gets a
  transient server-generated UUID; the flow must also match that publish id
  on every status poll, so a later attempt for the same job and repository
  cannot be mistaken for the attempt the flow started. The server retains only
  the current in-memory slot; publish ids are not a durable history API.
- Each accepted Modal GPU start similarly returns a transient UUID in both the
  start response and current status. Flow cleanup passes that UUID to the stop
  operation; a changed slot raises 409 `gpu.launch_replaced` and the current
  GPU remains running. An omitted UUID means the explicit operator action
  "stop whichever GPU is current", preserving the UI and emergency control.
  The server reserves its singleton slot atomically before spawning so two
  concurrent starts cannot create an untracked billed process. Launch IDs are
  in-memory attribution only: no history endpoint and no restart recovery.
- The managed remote-inference flow is side-effect-free until its context is
  entered. Startup checks the transport, starts one GPU launch, matches its id
  on every readiness read, then starts the leased robot session. Shared wire
  options (engine, task, horizon, fps, codec and s_min) come from one argument
  set and are sent identically to both halves. GPU readiness is only a hint;
  the server's room probe remains the authority before arm energization. A
  startup failure or timeout conditionally stops the owned launch. Context
  exit always attempts the robot-session stop before the conditional GPU stop,
  and a cleanup failure never masks an exception from the context body.
