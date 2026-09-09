/**
 * What each arm type can and cannot do — the client mirror of the backend's
 * makermodslab/arm_capabilities.py, read from the GET /api/v1/arms manifest.
 * Import these instead of writing `arm_type === "maker"` comparisons inline:
 * a scattered check is a check somebody forgets to extend when the next arm
 * type arrives — and after TB5 the next arm type may arrive from an
 * extension the core has never heard of.
 *
 * Lives in lib/ (not useRobots.ts, ArmType's public home) so it stays a pure,
 * independently testable module — useRobots pulls in the API context on
 * import.
 */
import type { TFunction } from "i18next";
import type { ArmFamilyInfo, LeaderOptionInfo } from "./armsApi";

/**
 * A manifest id — no longer a closed union. The manifest (GET /api/v1/arms)
 * is the authority on which ids exist and what they can do; the built-in ids
 * "so101" | "maker" | "metal" appear in this codebase ONLY as the keys of the
 * bundled-asset and i18n maps.
 *
 * Records created before the Maker arm existed have no arm_type on disk; the
 * backend reads those back as "so101", so this is never undefined in practice.
 */
export type ArmType = string;

// ---------------------------------------------------------------------------
// Predicates over a manifest entry.
//
// Every one takes `ArmFamilyInfo | undefined`. `undefined` means the manifest
// has not loaded yet, or the record's arm type is not installed (its
// `arm_available` is false) — and answers with the SO-101 shape: 6 joints,
// range-sweep calibration, urdf telemetry, Feetech bus, no protocol probe.
// That is the same fallback the pre-manifest `?? "so101"` renders used, so a
// still-loading page draws exactly what it drew before. Callers MUST gate an
// unavailable arm on `record.arm_available` before any of these matter: the
// fallback keeps the page rendering, it does not make the arm usable.
// ---------------------------------------------------------------------------

export type CalibrationKind = ArmFamilyInfo["calibration"]["kind"];

/**
 * Which calibration flow the config dialog renders for this family:
 * "range_sweep" (the SO-101 sweep managers), "steps" (the generic step
 * wizard the CAN families' zero pose runs on) or "panel" (the extension
 * serves its own page). `undefined` answers "range_sweep" — the SO-101
 * shape, like every other fallback here.
 */
export function calibrationKind(
  info: ArmFamilyInfo | undefined,
): CalibrationKind {
  return info ? info.calibration.kind : "range_sweep";
}

export function supportsAutoCalibration(
  info: ArmFamilyInfo | undefined,
): boolean {
  return info ? info.capabilities.supports_auto_calibration : true;
}

export function supportsPortProbe(info: ArmFamilyInfo | undefined): boolean {
  return info ? info.capabilities.supports_port_probe : false;
}

export function usesFeetechBus(info: ArmFamilyInfo | undefined): boolean {
  return info ? info.capabilities.uses_feetech_bus : true;
}

export function supportsDagger(info: ArmFamilyInfo | undefined): boolean {
  return info ? info.capabilities.supports_dagger : true;
}

export function supportsRemoteInference(
  info: ArmFamilyInfo | undefined,
): boolean {
  return info
    ? (info.capabilities.supports_remote_inference ?? info.id === "so101")
    : true;
}

export function supportsGripperWiggle(
  info: ArmFamilyInfo | undefined,
): boolean {
  return info ? info.capabilities.supports_gripper_wiggle : false;
}

/**
 * The leader arms a family can be driven by, default first. Empty before
 * the manifest loads — callers treat that as "one leader, nothing to pick".
 */
export function leaderOptions(
  info: ArmFamilyInfo | undefined,
): LeaderOptionInfo[] {
  return info ? info.leader_options : [];
}

/**
 * The record's effective leader kind: its own when set, else the family's
 * default (a record written before leader kinds existed reads back as the
 * default server-side too, so this only matters for a still-loading page).
 */
export function effectiveLeaderKind(
  info: ArmFamilyInfo | undefined,
  recordKind: string | undefined,
): string {
  if (recordKind) return recordKind;
  return info?.default_leader_kind ?? "";
}

/** The manifest entry for a leader kind, or undefined for one the family
 * does not offer (a hand-edited record) or before the manifest loads. */
export function leaderOption(
  info: ArmFamilyInfo | undefined,
  kind: string,
): LeaderOptionInfo | undefined {
  return leaderOptions(info).find((o) => o.id === kind);
}

/** Flat proprioceptive width of ONE follower arm — one dim per joint. */
export function jointsPerArm(info: ArmFamilyInfo | undefined): number {
  return info ? info.joints_per_arm : 6;
}

export function telemetryKind(
  info: ArmFamilyInfo | undefined,
): "urdf" | "degrees" {
  return info ? info.telemetry_kind : "urdf";
}

/**
 * Best-effort arm family for a dataset's / checkpoint's raw `robot_type`
 * string (lerobot writes the robot's `.name`: "so101_follower",
 * "bi_maker_follower", …; datasets recorded elsewhere carry anything). The
 * client mirror of `arm_capabilities.arm_type_from_robot_type`: scans every
 * family's `robot_type_markers` in manifest order, EXCEPT the first (default)
 * family, which is checked LAST because its markers are the loosest.
 *
 * Returns null — NOT a default — when the string is missing or unrecognized:
 * the cross-arm warnings that call this must stay silent when the arm can't be
 * established rather than raise a false alarm.
 */
export function armTypeFromRobotType(
  arms: ArmFamilyInfo[],
  robotType: string | null | undefined,
): ArmType | null {
  if (!robotType || arms.length === 0) return null;
  const text = robotType.trim().toLowerCase();
  if (!text) return null;
  // The manifest's contract: the default family is served FIRST; the scan
  // checks it LAST so an extension whose name embeds a built-in's marker
  // ("so101_twin") is not swallowed by the default's loose "so101".
  const [first, ...rest] = arms;
  for (const family of [...rest, first]) {
    if (family.robot_type_markers.some((marker) => text.includes(marker))) {
      return family.id;
    }
  }
  return null;
}

/**
 * Display name for an arm type: the catalog's per-id label when it has one
 * (the built-ins do), else the manifest's own label, else the bare id.
 */
export function armLabel(
  info: ArmFamilyInfo | undefined,
  id: string,
  t: TFunction,
): string {
  // Built at runtime, so the typed t() cannot check it; the catalog's
  // robot.corner.armType.* entries are the per-id overrides for the built-ins
  // and defaultValue covers every id the catalog does not know.
  return t(`robot.corner.armType.${id}` as never, {
    defaultValue: info?.label ?? id,
  }) as string;
}
