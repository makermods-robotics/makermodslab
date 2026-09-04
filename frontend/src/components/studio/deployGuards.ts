/**
 * Why a run mode cannot be launched, as a pure function.
 *
 * Extracted from DeployPanel so the preflight can be tested. Each of these
 * refusals stands between an operator and a physical robot session; getting one
 * wrong costs a hardware run, and the whole set was previously unreachable by
 * any test.
 *
 * Returns a TRANSLATION KEY, not prose, so the function stays pure and the
 * assertions below stay about which guard fired rather than about its wording.
 * The caller resolves it — see DeployPanel's `blockedReason`.
 */

export type DeployRunMode = "single" | "eval" | "coach" | "remote";

export interface DeployGuardContext {
  /** A robot record is selected. */
  hasRobot: boolean;
  /** The follower arm is configured and calibrated. */
  followerReady: boolean;
  /** A skill + checkpoint pair is chosen and its config has loaded. */
  hasCheckpoint: boolean;
  /** The checkpoint expects a different arm count than the robot has. */
  armMismatch: boolean;
  /** Every camera the checkpoint expects is bound to a device. */
  allCamerasBound: boolean;
  /** The temporal-ensemble coefficient is out of range. */
  temporalEnsembleInvalid: boolean;
  /** Another run already owns the robot. */
  inferenceActive: boolean;
  /** The leader arm is missing or its calibration is gone. */
  leaderMissing: boolean;
  /** This policy is language-conditioned and is steered by the task string. */
  requiresTask: boolean;
  /** The task string as typed. */
  task: string;
  /** REMOTE mode only. The LiveKit transport is installed, configured,
   * answering, AND already has the GPU-side operator in the room. False while
   * the probe has not answered yet, which is deliberate: launching into an
   * unverified transport energizes the arm for a run nothing will ever
   * drive. */
  transportReady: boolean;
  /** REMOTE mode only. The selected robot is a family the remote ease-in
   * supports — single-arm Feetech (SO-101). The CAN arms and bimanual rigs
   * are refused by the backend too; this only moves the refusal to before the
   * launch. */
  armSupportsRemote: boolean;
  /** REMOTE mode only. The chosen engine suits this checkpoint's policy type.
   *
   * Unlike every other flag here this one has NO backend twin, and cannot: the
   * server never loads the checkpoint (the GPU container does), so it cannot
   * tell a flow policy from an ACT one and accepts whichever engine it is
   * given. The UI is the only gate — `rtc` guides denoising, which an ACT
   * checkpoint cannot act on, and the run would simply be worse than `sync`
   * with nothing anywhere saying why. */
  remoteEngineSupported: boolean;
}

export function deployBlockedReason(
  mode: DeployRunMode,
  ctx: DeployGuardContext,
): string | null {
  if (!ctx.hasRobot) return "studio.deploy.blocked.noRobot";
  if (!ctx.followerReady) return "studio.deploy.blocked.followerNotReady";
  if (!ctx.hasCheckpoint) return "studio.deploy.blocked.noCheckpoint";
  if (ctx.armMismatch) return "studio.deploy.blocked.armMismatch";
  if (!ctx.allCamerasBound) return "studio.deploy.blocked.camerasUnbound";
  if (ctx.temporalEnsembleInvalid) return "studio.deploy.blocked.temporalEnsemble";
  if (ctx.inferenceActive) return "studio.deploy.blocked.runInProgress";

  // Applies to EVERY mode, not just coaching.
  //
  // A language-conditioned policy is steered by this string, and an empty one
  // does not fail loudly — it just makes the policy worse in ways that look
  // like the policy being bad. The task check used to live only under
  // `mode === "coach"`, so a Hub checkpoint (or any whose prefill lookup
  // failed) could be launched for a plain run or a scored eval with the field
  // blank, and the resulting success rate was measuring the wrong thing.
  if (ctx.requiresTask && ctx.task.trim() === "")
    return "studio.deploy.blocked.taskRequired";

  // Remote inference. The arm check comes first: it is a fact about the robot
  // that no amount of transport fixing changes, so saying "the transport isn't
  // ready" to someone holding a Metal arm would send them to the wrong problem.
  if (mode === "remote" && !ctx.armSupportsRemote)
    return "studio.deploy.blocked.remoteArmUnsupported";
  // Before the transport check, and for the same reason the arm check is: this
  // is a fact about the CHECKPOINT that no amount of transport fixing changes,
  // and it has a one-click remedy (switch the engine back to Adaptive sync)
  // that "the transport isn't ready" would send them right past.
  if (mode === "remote" && !ctx.remoteEngineSupported)
    return "studio.deploy.blocked.remoteEngineUnsupported";
  if (mode === "remote" && !ctx.transportReady)
    return "studio.deploy.blocked.transportNotReady";

  if (mode === "coach" && ctx.leaderMissing)
    return "studio.deploy.blocked.leaderMissing";
  // Coaching wants a task even when the policy does not read one: it is saved
  // with every correction and is the only record of what the session taught.
  if (mode === "coach" && ctx.task.trim() === "")
    return "studio.deploy.blocked.coachTaskRequired";
  return null;
}
