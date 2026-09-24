# makermodslab-sdk

Agent-first Python SDK for the MakerMods Lab robot server. Agent-first means:
every error carries the next call to make, long-running work gets blocking
`wait()`s instead of polling loops, and every docstring teaches by example —
the exception text and `help(client)` are the documentation that is guaranteed
to be in context when it's needed.

```python
from makermodslab_sdk import Client

client = Client("http://localhost:8000")
print(client.describe().summary())  # one-call orientation: server, session, jobs, nodes

with client.sessions.teleoperate("bench") as s:  # lease heartbeats + stop, automatic
    print(s.id, s.warnings)
```

## Docs: progressive disclosure

The built-in docs disclose in three tiers so an agent loads only what the
task needs — everything below the index is introspected from the code:

- `client.docs()` / `python -m makermodslab_sdk.docs` — the index: rules,
  namespace map, drill-down instructions (~1k tokens; start here)
- `client.docs("jobs")` / `... docs jobs` — one namespace's pattern intro
  (the resource module docstring) + method reference
- `client.docs("flows")` — passive hardware context plus short recording,
  training-to-publish and managed remote-inference sequences;
  each method names the primitive calls it composes
- `client.docs("jobs.create_training")` / `... docs jobs.create_training` —
  one method's full signature and docstring
- `client.docs(search="publish")` / `... docs --search publish` — find
  methods by name or docstring one-liner
- `python -m makermodslab_sdk.docs --all` — the full flat dump (unbudgeted;
  the tiers are the better start)

We deliberately ship no agent skill; a consumer can point one at the CLI in
five lines, e.g. `.claude/skills/makermodslab/SKILL.md`:

```markdown
---
name: makermodslab
description: Drive a MakerMods Lab robot server via the makermodslab-sdk Python SDK
---

Run `python -m makermodslab_sdk.docs` and follow its drill-down instructions
(`docs <tag>`, `docs <tag>.<method>`, `docs --search <q>`) before writing code.
```

`SPEC.md` is the language-agnostic behavior contract.

## Layering (port guide)

The SDK is three layers; a port to another language re-implements layer 1
idiomatically and transcribes layers 2–3:

1. **Transport core** (`_transport.py`, `errors.py`) — one request in, parsed
   JSON or a typed exception with remediation out.
2. **Resource namespaces** (`resources/`) — one module per API tag, thin and
   declarative; each method is tagged with the v1 `operationId` it covers.
3. **Ergonomics** — the leased-session context manager, waiters, realtime.
   `client.flows` lives here: compositions have no operation IDs and do not
   change the per-operation coverage ratchet.

For a fixed task, `client.flows.record_episodes("bench", "me/demo",
task="pick the cube", episodes=5)` starts a leased recording, reports
progress, and waits for all five saved episodes. Pass a list of task strings
to prompt separately for each episode. For a completed local run and Hub
publication in one call, use
`client.flows.train_and_publish("me/demo", steps=20000,
train_timeout=14400, publish_timeout=3600)`. Every flow exposes its full
signature at `client.docs("flows.<method>")`.

Setup runs through the same layering. `client.flows.auto_calibrate("bench",
arms=[{"device_type": "robot"}])` drives SO-101 auto-calibration to
completion — the arm moves itself, so nobody is in the loop — and reports
every arm's outcome rather than raising on a partial failure. The CAN
families' zero-pose wizard is a conversation instead:
`client.flows.calibrate_zero("bench", device_type="robot", confirm=ask)`
relays each instruction to `confirm`, which is required and has no default:
only a human can attest that the arm is physically posed, and confirming an
unposed arm records a wrong zero. The SO-101 manual sweep has no flow on
purpose — watch `client.calibration.status()` and narrate, or use the UI.

Before choosing a robot or discovery operation, get one passive snapshot:

```python
hardware = client.flows.inspect_hardware()
print(hardware.summary())
```

It combines the arm manifest, visible ports and cameras, and saved robot
records without opening an arm bus or changing a record. Missing assignments,
free ports, and shared record references remain structured on the result.
Suggested active discovery steps are literal SDK calls labelled
`read_only`, `may_energize`, or `moves_hardware`; the flow never runs them.

For remote inference, one context owns both the exact transient GPU launch and
the leased robot session. It sends the wire-shaping options to both halves from
one argument set, waits for that launch's readiness hint, then lets the
server's room probe make the authoritative decision before the arm energizes:

```python
with client.flows.remote_inference(
    "bench",
    policy_ref="me/act-pick",
    gpu="A10G",
    camera_bindings={"top": "top"},  # omitted = a camera-less run
    startup_timeout=180,
) as run:
    print(run.session_id, run.launch_id, run.start_warnings)
    run.session.wait(timeout=300)
```

As with `client.sessions.infer`, `camera_bindings` (policy camera name ->
robot-record camera name) is never inferred: omit it and the policy runs with
no cameras, even when the names match. `duration_s` defaults to 60 s on the
server; `duration_s=0` runs until stopped.

Exiting stops the robot session before conditionally stopping the GPU by its
`launch_id`; it cannot stop a replacement launch. Use the primitive
`client.sessions` methods when the GPU is launched elsewhere or should remain
available across multiple robot sessions.

The contract is `docs/api/openapi.json` at the repo root plus `SPEC.md`
(behavior semantics — leases, hints, error taxonomy; written alongside the
later stages). `tests/test_coverage_ratchet.py` equality-asserts
implemented-vs-planned against the snapshot, so surface coverage can only
move forward.

Tests run against the real FastAPI app through an in-process client — from
the repo root:

```bash
uv pip install -e ./sdk && .venv/bin/pytest sdk/tests
```
