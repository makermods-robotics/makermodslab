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
export type RemoteEngine = "sync" | "rtc";

export interface RemoteRunConfig {
  /** "<owner>/<repo>" the GPU container loads. Advisory to the backend in
   * this slice; load-bearing for the generated command line. */
  policyHubId: string;
  /** Which chunk player runs on the arm — and therefore which GPU server the
   * other terminal has to be running. Values are backend identifiers. */
  engine: RemoteEngine;
  horizon: number;
  fps: number;
  /** Codec IDENTIFIER — sent verbatim on both sides, never translated. */
  videoCodec: "H264" | "MJPEG";
  /** Seconds; 0 = unbounded. */
  durationS: number;
  /** RTC only. Minimum execution budget in steps; the robot computes
   * `overlap_end = H - max(s_min, d)` and the GPU server TRUSTS that number, so
   * the two `--s-min` / `--s_min` values must be the same. */
  sMin: number;
}

/** The GPU side's own default, and the robot's. Changing one without the other
 * puts the in-painting mask on a different boundary than the guidance. */
export const DEFAULT_S_MIN = 4;

/** Horizon defaults per engine. 16 is one open-loop ACT block; 50 is the full
 * chunk_size smolvla / pi0 / pi05 always denoise (a smaller value only
 * truncates), and it is what `robot_rtc` and `modal_policy_rtc.py` both
 * default to. */
export const DEFAULT_HORIZON: Record<RemoteEngine, number> = {
  sync: 16,
  rtc: 50,
};

export const DEFAULT_REMOTE_RUN_CONFIG: RemoteRunConfig = {
  policyHubId: "",
  engine: "sync",
  horizon: DEFAULT_HORIZON.sync,
  fps: 30,
  videoCodec: "H264",
  durationS: 60,
  sMin: DEFAULT_S_MIN,
};

/**
 * Policy families whose chunks can be IN-PAINTED, and therefore the only ones
 * the `rtc` engine helps.
 *
 * RTC ships the still-to-execute prefix so the server can GUIDE denoising —
 * which needs a policy that denoises. An ACT checkpoint serves a plain chunk
 * and ignores the extra state fields, so `rtc` buys it nothing and costs it the
 * play-to-completion contract it was evaluated in. Values are lerobot
 * `policy_type` identifiers, matched verbatim.
 */
export const FLOW_POLICY_TYPES = ["smolvla", "pi0", "pi05", "diffusion"] as const;

export function policySupportsRtc(
  policyType: string | null | undefined,
): boolean {
  if (!policyType) return false;
  return (FLOW_POLICY_TYPES as readonly string[]).includes(
    policyType.toLowerCase(),
  );
}

/**
 * The engine to preselect for a checkpoint.
 *
 * A flow policy defaults to `rtc` because that is the whole reason the engine
 * exists: at the ~400 ms round trip a bench run measures, the sync player
 * re-plans about once a second and two flow-policy plans made 400 ms apart
 * disagree at every seam — a visible ~1 Hz twitch on a perfectly healthy
 * transport. Everything else defaults to `sync`, which is correct for ANY
 * policy. An UNKNOWN policy type is "everything else": guessing `rtc` for one
 * would pair the arm with a GPU server the operator was never told to start.
 */
export function defaultEngineForPolicyType(
  policyType: string | null | undefined,
): RemoteEngine {
  return policySupportsRtc(policyType) ? "rtc" : "sync";
}

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
