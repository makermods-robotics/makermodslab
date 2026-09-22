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
