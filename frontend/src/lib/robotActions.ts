import type { StartableSessionKind } from "@/lib/sessionApi";

/**
 * THE source of truth for "which buttons drive the robot arm".
 *
 * Two treatments, one map:
 *
 * - `robot` — the amber affordance (Button variant="robot", worn by
 *   `RobotActionButton`). Pressing it ENERGIZES an arm.
 * - `destructive` — the existing red variant (worn by `ReleaseActionButton`).
 *   Pressing it ends with the arm DE-ENERGIZED.
 *
 * Everything else keeps the default variants on purpose. Port detection,
 * identify, arm identity, robot settings, dataset/model browsing and training
 * are read-only or off-robot; painting them amber would dilute the signal
 * until it means nothing.
 *
 * The energizing session kinds are deliberately the backend's own startable
 * set (`SessionStartBody.kind` in docs/api/openapi.json, mirrored by
 * `StartableSessionKind` in lib/sessionApi.ts) — robotActions.test.ts asserts
 * the equality against the committed OpenAPI snapshot, so a new backend kind
 * fails here rather than silently shipping an unmarked start button.
 */

export type RobotActionTreatment = "robot" | "destructive";

/**
 * Session kinds whose start energizes an arm. Equal to the backend's
 * startable set — see the parity test.
 */
export const ENERGIZING_SESSION_KINDS = [
  "teleoperation",
  "recording",
  "inference",
  "replay",
  "calibration",
  "auto_calibration",
] as const satisfies readonly StartableSessionKind[];

/**
 * Non-session hardware actions that still drive a servo. `wiggle` is seconds
 * of open-loop gripper motion through the legacy flow endpoint (there is no
 * lease to hold), but it moves the arm, so it wears the same amber.
 */
export const ENERGIZING_DEVICE_ACTIONS = ["wiggle"] as const;

/** De-energizing actions: both end with torque released. */
export const RELEASE_ACTIONS = ["stop", "release_now"] as const;

export type EnergizingActionKey =
  | (typeof ENERGIZING_SESSION_KINDS)[number]
  | (typeof ENERGIZING_DEVICE_ACTIONS)[number];

export type ReleaseActionKey = (typeof RELEASE_ACTIONS)[number];

export type RobotActionKey = EnergizingActionKey | ReleaseActionKey;

export interface RobotActionSpec {
  treatment: RobotActionTreatment;
  /**
   * Catalog KEY, never resolved copy: this map is built once at import time,
   * so a `t()` here would freeze whichever language loaded first (see
   * frontend/docs/localization.md §5.1).
   */
  tooltipKey: string;
}

/**
 * Treatment + tooltip key per action. The keys themselves are identifiers —
 * they mirror the backend's session-kind enum and are never displayed.
 */
export const ROBOT_ACTIONS: Record<RobotActionKey, RobotActionSpec> = {
  teleoperation: {
    treatment: "robot",
    tooltipKey: "shared.robotAction.tooltip.teleoperation",
  },
  recording: {
    treatment: "robot",
    tooltipKey: "shared.robotAction.tooltip.recording",
  },
  inference: {
    treatment: "robot",
    tooltipKey: "shared.robotAction.tooltip.inference",
  },
  replay: {
    treatment: "robot",
    tooltipKey: "shared.robotAction.tooltip.replay",
  },
  calibration: {
    treatment: "robot",
    tooltipKey: "shared.robotAction.tooltip.calibration",
  },
  auto_calibration: {
    treatment: "robot",
    tooltipKey: "shared.robotAction.tooltip.auto_calibration",
  },
  wiggle: {
    treatment: "robot",
    tooltipKey: "shared.robotAction.tooltip.wiggle",
  },
  // First press of a session's stop: the backend returns the arm to the pose
  // it started from, THEN releases torque.
  stop: {
    treatment: "destructive",
    tooltipKey: "shared.robotAction.tooltip.stop",
  },
  // Second press while that return is in flight: abort it and release now.
  // No UI surfaces this yet — the entry exists so the one that does gets the
  // treatment and copy from here rather than inventing its own.
  release_now: {
    treatment: "destructive",
    tooltipKey: "shared.robotAction.tooltip.releaseNow",
  },
};
