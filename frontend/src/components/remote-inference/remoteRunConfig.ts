import type { RobotRecord } from "@/hooks/useRobots";
import { isCanArmType } from "@/lib/armTypes";

/**
 * The four transport knobs plus the Hub id, held as one object so the start
 * request and the generated `modal run` line are built from the SAME values.
 *
 * They are not defaults anyone should tune casually: horizon, fps and codec
 * MUST match the GPU side, and Portal fingerprints the wire schema — a
 * disagreement silently drops every packet instead of raising, so a mismatched
 * run looks healthy and does nothing. Everything else DRTC exposes (adaptive,
 * base_lead, s_min, align, action_delay, the latency coefficients, video
 * quality/bitrate, reliable_state) stays a backend constant precisely because
 * a wrong value there presents as "the arm freezes" rather than as an error.
 */
export interface RemoteRunConfig {
  /** "<owner>/<repo>" the GPU container loads. Advisory to the backend in
   * this slice; load-bearing for the generated command line. */
  policyHubId: string;
  horizon: number;
  fps: number;
  /** Codec IDENTIFIER — sent verbatim on both sides, never translated. */
  videoCodec: "H264" | "MJPEG";
  /** Seconds; 0 = unbounded. */
  durationS: number;
}

export const DEFAULT_REMOTE_RUN_CONFIG: RemoteRunConfig = {
  policyHubId: "",
  horizon: 16,
  fps: 30,
  videoCodec: "H264",
  durationS: 60,
};

/** Codec choices. Values are wire identifiers; only a label beside them could
 * ever be translated, and neither needs one. */
export const VIDEO_CODECS: RemoteRunConfig["videoCodec"][] = ["H264", "MJPEG"];

/**
 * Whether this robot can host a remote run at all — the client mirror of the
 * backend's `supports_remote_inference`.
 *
 * Single-arm SO-101 only. The CAN families (Maker, Metal) are out because the
 * ease-in is a single Feetech-bus procedure, and bimanual is out for the same
 * reason; both are refused server-side with a 400, so this only moves the
 * refusal to before the operator commits to a launch.
 */
export function armSupportsRemoteInference(
  robot: RobotRecord | null | undefined,
): boolean {
  if (!robot) return false;
  return !isCanArmType(robot.arm_type) && robot.mode !== "bimanual";
}
