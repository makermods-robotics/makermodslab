"""The ``inference`` namespace: operator verbs for a coaching (DAgger) session.

Coaching is the third inference shape: start it with
``client.sessions.infer(robot, policy_ref=..., coaching=True,
coaching_dataset_name=...)`` — the policy drives the follower while the
LEADER arm stands armed for takeover; each takeover→handback correction is
recorded as an episode of the coaching dataset until ``target_corrections``
are collected.

These are the verb endpoints the operator (or an agent supervising the run)
fires while that session runs. They are flat one-shot commands with no body;
the session-scoped twin is ``client.sessions.coaching_command(session_id,
command)``, which routes the same verbs through the sessions surface —
prefer that one when you hold an ActiveSession (``s.coaching_command("takeover")``).
Every verb answers ``{success, message}`` — ``success=False`` with the
reason in ``message`` (e.g. no coaching session is running) rather than an
HTTP error.
"""

from __future__ import annotations

from makermodslab_sdk._operations import operation
from makermodslab_sdk.resources._base import Resource, SdkModel


class CoachingCommandResult(SdkModel):
    """The verbs' shared answer — check ``success``, read ``message``."""

    success: bool
    message: str


class InferenceResource(Resource):
    """``client.inference`` — coaching-run operator verbs.

    Example:
        >>> client.inference.takeover().success  # leader takes the arm
        True
        >>> client.inference.handback().success  # correction saved, policy resumes
        True
    """

    def _verb(self, name: str, action: str) -> CoachingCommandResult:
        return CoachingCommandResult.model_validate(
            self._transport.request("POST", f"/api/v1/coaching-{name}", action=action)
        )

    @operation("coaching_takeover")
    def takeover(self) -> CoachingCommandResult:
        """The operator takes control: the leader arm starts driving the
        follower and a correction recording begins."""
        return self._verb("takeover", "Coaching takeover")

    @operation("coaching_handback")
    def handback(self) -> CoachingCommandResult:
        """End the correction: save the recorded episode and hand the
        follower back to the policy."""
        return self._verb("handback", "Coaching handback")

    @operation("coaching_drop_last")
    def drop_last(self) -> CoachingCommandResult:
        """Discard the most recently saved correction episode."""
        return self._verb("drop-last", "Coaching drop last")

    @operation("coaching_hold")
    def hold(self) -> CoachingCommandResult:
        """Pause the policy (follower holds position; no recording)."""
        return self._verb("hold", "Coaching hold")

    @operation("coaching_resume")
    def resume(self) -> CoachingCommandResult:
        """Resume the policy from a hold."""
        return self._verb("resume", "Coaching resume")

    @operation("coaching_reset")
    def reset(self) -> CoachingCommandResult:
        """Reset the scene between attempts (policy pauses for the reset)."""
        return self._verb("reset", "Coaching reset")

    @operation("coaching_recovered")
    def recovered(self) -> CoachingCommandResult:
        """Declare the scene recovered after a reset; the policy resumes."""
        return self._verb("recovered", "Coaching recovered")

    @operation("coaching_cancel")
    def cancel(self) -> CoachingCommandResult:
        """Abandon the current correction WITHOUT saving it; the policy
        resumes."""
        return self._verb("cancel", "Coaching cancel")
