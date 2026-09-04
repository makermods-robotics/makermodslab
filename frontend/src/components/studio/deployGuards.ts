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

export type DeployRunMode = "single" | "eval" | "coach";

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
  /** The training dataset lists SEVERAL tasks and the operator has not picked
   * one. There is no defensible guess between them — the datasets this was
   * measured against separate near-identical task strings by a single episode —
   * so the panel asks instead of choosing. */
  taskAmbiguous?: boolean;
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
  // Ambiguity is refused wherever the task is actually used — the same two
  // cases that put the field on screen. A merged dataset carries several task
  // strings and nothing distinguishes them well enough to pick one: leaving the
  // box empty used to send whichever had one more episode than the rest, which
  // for a coaching run is then stamped into every recorded frame. The chips
  // below the field are the way out.
  //
  // BEFORE taskRequired, which is also true here and is the less useful of the
  // two: "describe the task" tells an operator to type a sentence when the
  // dataset is offering them a list to choose from.
  if (ctx.taskAmbiguous && (ctx.requiresTask || mode === "coach"))
    return "studio.deploy.blocked.taskAmbiguous";

  if (ctx.requiresTask && ctx.task.trim() === "")
    return "studio.deploy.blocked.taskRequired";

  if (mode === "coach" && ctx.leaderMissing)
    return "studio.deploy.blocked.leaderMissing";
  // Coaching wants a task even when the policy does not read one: it is saved
  // with every correction and is the only record of what the session taught.
  if (mode === "coach" && ctx.task.trim() === "")
    return "studio.deploy.blocked.coachTaskRequired";
  return null;
}
