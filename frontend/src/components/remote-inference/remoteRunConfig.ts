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
  // No duration here since S3.9: the panel offers ONE max-duration field for
  // both places a run can happen, and it lives in the panel's own state beside
  // the task. 0 still means unbounded to the backend for a remote run.
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
 *
 * FALLBACK ONLY since S3.7b. The server answers this question itself now
 * (`supports_rtc`, from `jobs.policy_type_supports_rtc`'s hand-mirrored read of
 * the pinned fork), and its answer is the primary one — this list is consulted
 * only when the field is null, i.e. a server too old to send it. The two lists
 * do disagree (this one carries `diffusion`, which the fork's classes say
 * cannot; the fork's carries `evo1`/`groot`/`molmoact2`, which this one has
 * never heard of), and that disagreement is precisely why the server's answer
 * wins wherever there is one.
 */
export const FLOW_POLICY_TYPES = [
  "smolvla",
  "pi0",
  "pi05",
  "diffusion",
] as const;

/**
 * The slice of a checkpoint's policy config every remote-engine decision reads.
 *
 * Structural on purpose: `PolicyConfigSummary` satisfies it without this module
 * importing the API types, and — more to the point — the default engine, the
 * ENABLED state of the rtc option, the sentence under the picker and the
 * `remoteEngineSupported` launch guard are all handed the SAME object, so they
 * cannot answer differently. Since S3.9's follow-up that is one rule rather
 * than two: the panel no longer keeps a separate, fail-open "the server won't
 * refuse this" reading for a local rollout.
 */
export interface PolicyRtcInfo {
  policy_type?: string | null;
  supports_rtc?: boolean | null;
}

/**
 * Whether this checkpoint's architecture can run Real-Time Chunking.
 *
 * `supports_rtc` from the server is the answer whenever it has one, including
 * `false`. `null` means "not established" — a policy type newer than the
 * server's table — and only then does the frontend's own family list decide.
 * An unknown type on both sides reads as "no", which is the safe direction:
 * guessing rtc pairs the arm with a GPU server the operator was never told to
 * start, and locally it trades the play-to-completion contract the checkpoint
 * was evaluated in for nothing. This is the ONLY answer the UI asks now — the
 * rtc option is disabled wherever it comes back false, on both the local and
 * the remote path, with one sentence under the picker saying why rather than a
 * greyed-out control explaining nothing.
 */
export function policySupportsRtc(
  policy: PolicyRtcInfo | null | undefined,
): boolean {
  if (!policy) return false;
  if (typeof policy.supports_rtc === "boolean") return policy.supports_rtc;
  const policyType = policy.policy_type;
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
 * policy — including a checkpoint nobody has classified.
 */
export function defaultEngineForPolicy(
  policy: PolicyRtcInfo | null | undefined,
): RemoteEngine {
  return policySupportsRtc(policy) ? "rtc" : "sync";
}

/**
 * The horizon to seed for `engine`, given what the checkpoint returns per
 * chunk (`n_action_steps`, null when the config doesn't say).
 *
 * `n_action_steps` is a CEILING, not a target. `predict_action_chunk` returns
 * exactly that many steps, so a horizon above it makes the two Portal peers
 * disagree about the action-chunk shape: the wire-schema fingerprint stops
 * matching and every packet is dropped IN SILENCE — a connected, healthy-
 * looking session that transfers nothing. MolmoAct2's published checkpoint is
 * 30 against an rtc default of 50, which is exactly that trap.
 *
 * So the seed is the engine default CLAMPED to the ceiling, never raised to it:
 * an ACT checkpoint declaring 100 still starts at one open-loop block (16), and
 * a flow checkpoint declaring 50 still starts at 50. Only a checkpoint that
 * returns FEWER steps than the engine default moves the number, which is the
 * only case where the default was unusable.
 */
export function horizonForEngine(
  engine: RemoteEngine,
  nActionSteps: number | null | undefined,
): number {
  const ceiling =
    typeof nActionSteps === "number" && nActionSteps > 0 ? nActionSteps : null;
  const base = DEFAULT_HORIZON[engine];
  return ceiling != null ? Math.min(base, ceiling) : base;
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
