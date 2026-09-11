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

/**
 * The arm LAYOUT a robot record describes — what is plugged into THIS
 * machine. "both" is a leader+follower pair (every record written before
 * remote teleoperation reads back as this); "follower" is a robot station
 * that hosts its follower for an operator elsewhere; "leader" is a
 * controller — a leader arm that drives a remote robot. The values are data
 * (the record's `arms` field on disk); only their labels localize.
 */
export type RobotArms = "both" | "follower" | "leader";

/** Which arm(s) a setup-gap diagnosis should look at. */
export type SetupScope = "all" | "follower" | "leader";

/**
 * The scope that matches a layout: a station is ready once its follower
 * side is, a controller once its leader side is. Surfaces that summarise a
 * record as a whole (the picker's status dot, the settings footer) use this
 * so a follower-only station is not nagged about a leader it does not have.
 */
export function setupScopeForArms(arms: RobotArms | undefined): SetupScope {
  if (arms === "follower") return "follower";
  if (arms === "leader") return "leader";
  return "all";
}

/** The readiness flags the listing carries, beside the layout. Structural
 * for the same reason as RobotArmFields. */
export interface RobotReadinessFields {
  arms?: RobotArms;
  is_clean: boolean;
  follower_ready: boolean;
  leader_ready: boolean;
}

/**
 * "Is this record set up for what it is?" — the layout-aware readiness a
 * record-level summary shows. Activity-specific gates keep reading the flag
 * for the arm they drive (teleop → is_clean, inference → follower_ready,
 * remote driving → leader_ready); this is only for surfaces describing the
 * record rather than an activity.
 */
export function robotLayoutReady(robot: RobotReadinessFields): boolean {
  switch (setupScopeForArms(robot.arms)) {
    case "follower":
      return robot.follower_ready;
    case "leader":
      return robot.leader_ready;
    default:
      return robot.is_clean;
  }
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
 * (inference/replay/hosting) pass "follower" and leader-only ones (driving a
 * remote robot) pass "leader", so the message never blames gaps on an arm
 * their activity doesn't touch.
 */
export function robotSetupGaps(
  robot: RobotArmFields,
  scope: SetupScope = "all",
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
  const arms =
    scope === "follower"
      ? allArms.filter((a) => a.follower)
      : scope === "leader"
        ? allArms.filter((a) => !a.follower)
        : allArms;
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
  scope: SetupScope = "all",
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
  scope: SetupScope = "all",
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
