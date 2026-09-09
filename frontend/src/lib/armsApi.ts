import { apiRequest, type Fetcher } from "./apiClient";

/**
 * One entry of GET /api/v1/arms — ArmFamilyInfo in makermodslab/schemas/
 * system.py, mirrored field for field. The manifest is the client's ONLY
 * source of arm capabilities: an arm family the core did not ship (an
 * extension's) appears here like a built-in, and the predicates in
 * lib/armTypes.ts read these fields rather than branching on the id.
 */
export interface ArmCalibrationSide {
  /** What the config dialog shows for this side BEFORE Start. */
  text: string;
  /** A served image beside the text, or null for none. */
  image_url: string | null;
}

export interface ArmCalibrationSummary {
  leader: ArmCalibrationSide | null;
  follower: ArmCalibrationSide | null;
}

export interface ArmCalibrationInfo {
  /** range_sweep: the SO-101 sweep managers; steps: the generic step wizard
   * (the CAN families' zero pose); panel: the extension serves its own page
   * at `panel_url`. */
  kind: "range_sweep" | "steps" | "panel";
  /** Set for steps families; null for range_sweep (and panel) ones. */
  summary: ArmCalibrationSummary | null;
  /** The family's served calibration page, or null when it has none. */
  panel_url: string | null;
}

export interface ArmCapabilities {
  uses_feetech_bus: boolean;
  supports_auto_calibration: boolean;
  supports_dagger: boolean;
  supports_port_probe: boolean;
  motion_identify_energizes_follower: boolean;
  /** The family can jog ONE port's gripper so the user sees which arm it is
   * (POST /api/v1/maker/wiggle-gripper) — the identification of last resort
   * when neither the probe nor the gesture can tell two arms apart. */
  supports_gripper_wiggle: boolean;
}

/**
 * One leader arm a family can be driven by — LeaderOptionInfo in
 * makermodslab/schemas/system.py. `id` is what a robot record stores as
 * `leader_kind`; `available` is false when this install cannot drive it
 * (`unavailable_reason` names what to install); `energized` marks a leader
 * that holds torque while the human moves it (the Metal arm's
 * gravity-compensated leader): it answers the follower's protocol, so the
 * probe cannot tell the two apart, and refuses the gesture.
 */
export interface LeaderOptionInfo {
  id: string;
  label: string;
  available: boolean;
  unavailable_reason: string | null;
  energized: boolean;
  /** The pre-start calibration summary for THIS leader's side. */
  calibration_summary: ArmCalibrationSide | null;
}

export interface ArmFamilyInfo {
  id: string;
  label: string;
  short_label: string;
  /** "builtin" for the shipped families; an extension's name otherwise. */
  provided_by: string;
  /** A served photo for the create dialog; null for the built-ins, whose
   * photos the frontend bundles (ARM_PHOTOS). */
  image_url: string | null;
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
  /** What a record with no `leader_kind` reads as: `leader_options[0].id`. */
  default_leader_kind: string;
  /** The leader arms the family can be driven by, default first. */
  leader_options: LeaderOptionInfo[];
}

/**
 * Resolve a URL the manifest serves (`image_url`, a summary side's
 * `image_url`, `panel_url` — root-relative `/api/v1/ext/<name>/static/...`
 * paths) against the API origin. Under `makermodslab --dev` the page comes
 * from Vite, not the backend, so a bare root-relative `<img src>` would miss
 * the server — the same reason every camera and episode src is built on
 * `baseUrl`. Anything not root-relative passes through; null stays null.
 */
export function servedUrl(
  baseUrl: string,
  url: string | null | undefined,
): string | null {
  if (!url) return null;
  return url.startsWith("/") ? `${baseUrl}${url}` : url;
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
