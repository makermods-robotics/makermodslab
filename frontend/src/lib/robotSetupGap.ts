import type { TFunction } from "i18next";

/**
 * Diagnoses WHY a robot record is flagged unclean.
 *
 * The backend folds ports, calibration assignments, and on-disk calibration
 * files into one boolean, so warning surfaces can't tell what is missing from
 * the flag alone — and blaming "missing a calibration" when only a port is
 * unassigned sends the user to recalibrate an arm that is already calibrated.
 *
 * This used to return a ready-made English predicate assembled from fragments
 * ("is missing a calibration for the " + list + " arm" + optional "s"), which
 * no translator can reorder. It now returns STRUCTURE; rendering is a separate
 * step, in English (`robotSetupGap`) or via a catalog (`formatRobotSetupGap`).
 *
 * Lives outside useRobots.ts so it stays a pure function importable without
 * pulling in the API/localStorage side effects of the hook module.
 */

export type ArmKey =
  | "leader"
  | "follower"
  | "leftLeader"
  | "leftFollower"
  | "rightLeader"
  | "rightFollower";

/** The subset of RobotRecord this needs. Structural, so RobotRecord satisfies
 * it without an import (which would create a hooks <-> lib cycle). */
export interface RobotArmFields {
  mode: "single" | "bimanual";
  leader_port: string;
  follower_port: string;
  leader_config: string;
  follower_config: string;
  right_leader_port: string;
  right_follower_port: string;
  right_leader_config: string;
  right_follower_config: string;
}

export interface RobotSetupGaps {
  missingCalibration: ArmKey[];
  missingPort: ArmKey[];
  /** Every field is populated, so the backend must have flagged a referenced
   * calibration file that no longer exists on disk. */
  staleConfig: boolean;
}

/** English arm labels. Also the catalog's source of truth for `robot.arm.*`. */
export const ARM_LABEL_EN: Record<ArmKey, string> = {
  leader: "leader",
  follower: "follower",
  leftLeader: "left leader",
  leftFollower: "left follower",
  rightLeader: "right leader",
  rightFollower: "right follower",
};

/**
 * `scope` must match the flag being diagnosed: follower-only surfaces
 * (inference/replay) pass "follower" so the message never blames leader-arm
 * gaps their activity doesn't care about.
 */
export function robotSetupGaps(
  robot: RobotArmFields,
  scope: "all" | "follower" = "all",
): RobotSetupGaps {
  const allArms: {
    key: ArmKey;
    port: string;
    config: string;
    follower: boolean;
  }[] =
    robot.mode === "bimanual"
      ? [
          { key: "leftLeader", port: robot.leader_port, config: robot.leader_config, follower: false },
          { key: "leftFollower", port: robot.follower_port, config: robot.follower_config, follower: true },
          { key: "rightLeader", port: robot.right_leader_port, config: robot.right_leader_config, follower: false },
          { key: "rightFollower", port: robot.right_follower_port, config: robot.right_follower_config, follower: true },
        ]
      : [
          { key: "leader", port: robot.leader_port, config: robot.leader_config, follower: false },
          { key: "follower", port: robot.follower_port, config: robot.follower_config, follower: true },
        ];
  const arms = scope === "follower" ? allArms.filter((a) => a.follower) : allArms;
  const missingCalibration = arms.filter((a) => !a.config?.trim()).map((a) => a.key);
  const missingPort = arms.filter((a) => !a.port?.trim()).map((a) => a.key);
  return {
    missingCalibration,
    missingPort,
    staleConfig: missingCalibration.length === 0 && missingPort.length === 0,
  };
}

/**
 * English predicate to append after a robot's name, e.g.
 * "has no port assigned for the follower arm".
 *
 * Output is frozen by useRobots.setupGap.test.ts — it must stay byte-identical
 * to the pre-i18n implementation.
 */
export function robotSetupGap(
  robot: RobotArmFields,
  scope: "all" | "follower" = "all",
): string {
  const gaps = robotSetupGaps(robot, scope);
  if (gaps.staleConfig) {
    return "references a calibration file that no longer exists — reassign or recalibrate";
  }
  const armList = (keys: ArmKey[]) =>
    `${keys.map((k) => ARM_LABEL_EN[k]).join(" and ")} arm${keys.length > 1 ? "s" : ""}`;
  const parts: string[] = [];
  if (gaps.missingCalibration.length)
    parts.push(`is missing a calibration for the ${armList(gaps.missingCalibration)}`);
  if (gaps.missingPort.length)
    parts.push(`has no port assigned for the ${armList(gaps.missingPort)}`);
  return parts.join(" and ");
}

/**
 * Localized equivalent of `robotSetupGap`. Each clause is a whole sentence
 * fragment in the catalog, so a translator controls word order, the list
 * separator, and the plural — none of which survive string concatenation.
 */
export function formatRobotSetupGap(
  t: TFunction,
  robot: RobotArmFields,
  scope: "all" | "follower" = "all",
): string {
  const gaps = robotSetupGaps(robot, scope);
  if (gaps.staleConfig) return t("robot.setupGap.stale");

  const armList = (keys: ArmKey[]) =>
    t("robot.setupGap.armList", {
      count: keys.length,
      arms: keys
        .map((k) => t(`robot.arm.${k}` as never))
        .join(t("robot.setupGap.armJoin")),
    });

  const parts: string[] = [];
  if (gaps.missingCalibration.length)
    parts.push(
      t("robot.setupGap.missingCalibration", {
        arms: armList(gaps.missingCalibration),
      }),
    );
  if (gaps.missingPort.length)
    parts.push(
      t("robot.setupGap.noPort", { arms: armList(gaps.missingPort) }),
    );
  return parts.join(t("robot.setupGap.clauseJoin"));
}
