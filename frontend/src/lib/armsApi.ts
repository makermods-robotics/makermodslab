import { apiRequest, type Fetcher } from "./apiClient";

/**
 * One entry of GET /api/v1/arms — ArmFamilyInfo in makermodslab/schemas/
 * system.py, mirrored field for field. The manifest is the client's ONLY
 * source of arm capabilities: an arm family the core did not ship (an
 * extension's) appears here like a built-in, and the predicates in
 * lib/armTypes.ts read these fields rather than branching on the id.
 */
export interface ArmZeroPose {
  leader: string;
  follower: string;
}

export interface ArmCalibrationInfo {
  kind: "range_sweep" | "zero_pose";
  /** Set for zero_pose families; null for range_sweep ones. */
  zero_pose: ArmZeroPose | null;
}

export interface ArmCapabilities {
  uses_feetech_bus: boolean;
  supports_auto_calibration: boolean;
  supports_dagger: boolean;
  supports_port_probe: boolean;
  motion_identify_energizes_follower: boolean;
}

export interface ArmFamilyInfo {
  id: string;
  label: string;
  short_label: string;
  /** "builtin" for the shipped families; an extension's name otherwise. */
  provided_by: string;
  joints_per_arm: number;
  /** Appended to a robot's name to mint its default calibration id ("" for
   * the SO-101, "_maker" / "_metal" for the CAN families — whose Star-leader
   * calibrations share one directory while their zero poses differ). Mirrors
   * each family's default_calibration_name on the server; the explicit
   * config_file the config dialog sends at calibration start must agree. */
  calibration_name_suffix: string;
  supports_bimanual: boolean;
  calibration: ArmCalibrationInfo;
  telemetry_kind: "urdf" | "degrees";
  capabilities: ArmCapabilities;
  /** [single_robot_type, bimanual_robot_type] — lerobot's config names. */
  robot_types: string[];
  /** Substrings that identify a dataset's raw robot_type as this family. */
  robot_type_markers: string[];
}

/** GET /api/v1/arms — the `arms` array, in manifest (registry) order: the
 * default family first. Throws on a non-OK response. */
export async function fetchArmFamilies(
  baseUrl: string,
  fetchWithHeaders: Fetcher = fetch,
): Promise<ArmFamilyInfo[]> {
  // apiRequest throws the house ApiError (status + detail + code) on non-OK,
  // so the provider can keep its last good manifest and report the failure
  // instead of rendering every arm as unavailable off an empty list.
  const body = await apiRequest<{ arms: ArmFamilyInfo[] }>(
    baseUrl,
    fetchWithHeaders,
    "/api/v1/arms",
    { action: "Load arm families" },
  );
  return body.arms;
}
