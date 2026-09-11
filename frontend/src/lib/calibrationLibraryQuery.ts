import type { ArmType } from "./armTypes";

/**
 * The `?arm_type=…[&leader_kind=…]` query every calibration-library request
 * carries (list, delete, rename, upload). `leaderKind` is set ONLY for the
 * teleop side of a family with more than one leader (the Metal arm's own
 * leader keeps a library apart from the Star leader's); every other request
 * is byte-identical to what it was before leader kinds existed.
 */
export function libraryQuery(armType: ArmType, leaderKind?: string): string {
  const query = `?arm_type=${encodeURIComponent(armType)}`;
  return leaderKind
    ? `${query}&leader_kind=${encodeURIComponent(leaderKind)}`
    : query;
}
