"""The shipped cheatsheet: the whole SDK surface in a few thousand tokens.

``python -m makermodslab_sdk.docs`` prints it. Load it into an agent's
context (or a human's terminal) and the SDK is drivable without reading
source: the patterns are hand-written, the per-namespace method reference is
INTROSPECTED from the live classes, so it can never drift from the code.
"""

from __future__ import annotations

import inspect

HEADER = """\
# makermodslab-sdk cheatsheet

Agent-first Python SDK for a MakerMods Lab robot server (SO-101 arms:
teleoperation, dataset recording, training, inference, replay, calibration).

    from makermodslab_sdk import Client
    client = Client("http://localhost:8000")   # the app's single port
    print(client.describe().summary())          # ALWAYS a good first call

## The rules that matter

- ERRORS ARE THE MANUAL. Every failure's text ends with "Next step: <the
  literal call to make>". Read it. Branch on `err.code` (e.g.
  "session.held") or exception type — never on the prose.
- SESSIONS HOLD THE ARM. One robot flow runs at a time. Start flows through
  the `with` form so the lease heartbeat + stop are automatic:

      with client.sessions.teleoperate("bench") as s:   # robot RECORD name
          print(s.id, s.warnings)      # warnings: warn-but-allow findings
          ...                          # arm is live inside the block
      # leaving the block stops the session; a lost lease raises
      # SessionLostError. Robot busy? -> SessionHeldError tells you the
      # holder; client.sessions.stop_current() is the (never owner-gated)
      # hammer, then retry.

  Other kinds: .record(robot, dataset_repo_id=..., single_task=...),
  .infer(robot, policy_ref=...), .replay(robot, repo_id=..., episode_index=...),
  .calibrate(robot, device_type="robot"|"teleop"), .auto_calibrate(robot, arms=[...]).
- NEVER WRITE POLLING LOOPS. Long-running work has blocking waiters:
  client.jobs.wait(job_id), client.datasets.wait_for_download(repo_id), ...
  All take timeout=; on timeout the error says how to keep waiting.
- FULL BACKEND POWER, wider than the web UI. create_training(...) accepts
  EVERY server training knob as a kwarg (help(makermodslab_sdk.TrainingOptions)
  is the catalog: wandb_*, optimizer_*, resume/fine-tune lineage, eval,
  device/AMP, hf_job_timeout, ...); a typo'd knob fails client-side with the
  fix named. client.robots manages the saved robot records sessions start
  from (create/update/rename/delete; mode is fixed at creation).
- REALTIME (optional extra `makermodslab-sdk[realtime]`):
  client.sample_joints(duration_s=2.0) -> bounded LIST of frames (empty =
  nothing moving, that's an answer). client.events() streams typed events;
  control events (jobs_changed/session_changed/...) are REFETCH HINTS,
  never state.
- Responses are pydantic models mirroring the server, `extra="allow"` —
  unknown server fields stay readable, never crash.

## Exceptions (all subclass MakerModsError)

ConnectionFailedError (server unreachable) / ApiError (any non-2xx; has
.status/.code/.detail/.details/.suggestion) with subclasses NotFoundError,
InvalidRequestError, RobotBusyError (.busy_with), SessionHeldError (.holder)
— plus JobWaitTimeout, WaitTimeoutError, OperationFailedError,
SessionLostError from the ergonomics layer.
"""


def _method_reference() -> str:
    from makermodslab_sdk.client import RESOURCE_CLASSES

    lines: list[str] = ["## Method reference (introspected — always current)"]
    for tag in sorted(RESOURCE_CLASSES):
        cls = RESOURCE_CLASSES[tag]
        lines.append(f"\n### client.{tag}")
        for name, member in sorted(vars(cls).items()):
            if name.startswith("_") or not callable(member):
                continue
            try:
                signature = str(inspect.signature(member)).replace("(self, ", "(").replace("(self)", "()")
            except (TypeError, ValueError):  # pragma: no cover - defensive
                signature = "(...)"
            doc = inspect.getdoc(member)
            first_line = doc.splitlines()[0] if doc else ""
            lines.append(f"- {name}{signature} — {first_line}")
    lines.append(
        "\n### client (top level)\n"
        "- describe() — one-call orientation snapshot; print(.summary())\n"
        "- events(kinds=None) / sample_joints(duration_s=2.0, max_frames=None) / "
        "stream_joints() — realtime extra\n"
        "- close() — or use Client as a context manager"
    )
    return "\n".join(lines)


def cheatsheet() -> str:
    """The full cheatsheet text (header + introspected method reference)."""
    return HEADER + "\n" + _method_reference() + "\n"


if __name__ == "__main__":
    print(cheatsheet())
