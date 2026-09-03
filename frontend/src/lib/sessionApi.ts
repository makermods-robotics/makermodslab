import type { TFunction } from "i18next";
import { Fetcher, apiRequest, ApiError } from "./apiClient";
import type { SessionKind } from "@/hooks/useActiveSession";

/**
 * Client for the /api/v1/sessions surface — the named-session start/stop the
 * robot flows (teleoperation, recording, inference, remote inference, replay,
 * calibration, auto-calibration) go through.
 *
 * The frontend sends the robot's NAME plus kind-specific options only; ports,
 * configs, mode, right-arm fields and cameras all resolve server-side from
 * the saved robot record. Starting with an `owner` (lib/sessionOwner.ts)
 * attaches a lease the tab renews via useSessionHeartbeat — miss the
 * heartbeats and the server safety-stops the session. That lease is the
 * safety net that replaced the browser unload beacons and exit guards.
 */

/** The kinds startable through POST /api/v1/sessions (only wiggle still
 * starts through its legacy flow endpoint — seconds of open-loop motion,
 * nothing to lease). */
export type StartableSessionKind =
  | "teleoperation"
  | "recording"
  | "inference"
  | "remote_inference"
  | "replay"
  | "calibration"
  | "auto_calibration";

export interface SessionLeaseInfo {
  owner: string;
  timeout_s: number;
  expires_in_s: number;
}

export interface SessionInfo {
  id: string;
  kind: SessionKind;
  /** Null for sessions started through the legacy endpoints. */
  robot: string | null;
  owner: string | null;
  started_at: number;
  revision: number;
  phase: string | null;
  /** Null for owner-less and legacy-started sessions (never timeout-stopped). */
  lease: SessionLeaseInfo | null;
}

export interface EndedSessionInfo {
  id: string;
  kind: SessionKind;
  ended_at: number;
  phase: string | null;
  /** "session.lease_expired" when the expiry watchdog safety-stopped it. */
  reason: string | null;
}

// Kind-specific `options` payloads — mirror makermodslab/schemas/sessions.py
// (extra="forbid" server-side: an off-shape field is a coded 422, not an
// ignored knob).

export interface TeleoperationSessionOptions {
  skip_identity_check?: boolean;
}

export interface RecordingSessionOptions {
  dataset_repo_id: string;
  single_task: string;
  num_episodes?: number;
  episode_time_s?: number;
  reset_time_s?: number;
  fps?: number;
  video?: boolean;
  push_to_hub?: boolean;
  tags?: string[];
  private?: boolean;
  resume?: boolean;
  streaming_encoding?: boolean;
  skip_identity_check?: boolean;
}

export interface InferenceSessionOptions {
  policy_ref: string;
  task?: string;
  camera_bindings?: Record<string, string>;
  camera_dims?: Record<string, { width: number; height: number }>;
  duration_s?: number;
  checkpoint_state_dim?: number;
  eval_episodes?: number;
  inference_engine?: "sync" | "rtc";
  temporal_ensemble_coeff?: number;
  skip_identity_check?: boolean;
  // COACHING (DAgger) mode — the third session shape. The policy drives while
  // the operator watches; each takeover through the leader arm is recorded as
  // one episode of a new dataset. Mutually exclusive with `eval_episodes > 1`
  // (400) and refused alongside `inference_engine: "rtc"` (400). The LEADER
  // arms it needs are NOT sent from here: like the followers, they resolve
  // server-side from the named robot record.
  coaching?: boolean;
  // How many corrections to collect before the session ends on its own.
  // Clamped server-side to [1, 100].
  target_corrections?: number;
  // Dataset name for the corrections, WITHOUT the mandatory `rollout_` prefix
  // (applied server-side). lerobot then appends its own timestamp, so the name
  // on disk is NOT predictable from here — read `coaching_dataset` off the
  // status payload once the session reports it.
  coaching_dataset_name?: string;
}

/** Remote inference (DRTC) — mirrors schemas/sessions.py's
 * `RemoteInferenceOptions` field for field.
 *
 * The arm is driven by THIS machine; the policy runs on a remote GPU and the
 * two meet in a LiveKit room. Everything hardware-shaped still resolves
 * server-side from the named robot record, exactly as for `inference`.
 *
 * `policy_ref` and `policy_hub_id` are two different vocabularies and are
 * deliberately not collapsed: `policy_ref` is the opaque Lab ref the
 * checkpoint picker yields (and what `checkpoint_state_dim` / `camera_dims`
 * come from), while `policy_hub_id` is the "<owner>/<repo>" the GPU container
 * resolves with `from_pretrained`. The backend never reads `policy_hub_id` in
 * this slice — it is here so the panel can generate the other terminal's
 * `modal run` line from this same object.
 *
 * horizon / fps / video_codec MUST match the GPU side. Portal fingerprints the
 * wire schema and SILENTLY DROPS packets whose fingerprint differs, so a
 * disagreement presents as a healthy-looking session with zero chunks rather
 * than as an error — which is exactly why they are options here and not
 * constants buried in the backend. */
export interface RemoteInferenceSessionOptions {
  policy_ref: string;
  policy_hub_id?: string;
  task?: string;
  camera_bindings?: Record<string, string>;
  camera_dims?: Record<string, { width: number; height: number }>;
  checkpoint_state_dim?: number;
  /** 0 = unbounded. */
  duration_s?: number;
  horizon?: number;
  fps?: number;
  /** Codec IDENTIFIERS — sent verbatim, never translated. */
  video_codec?: "H264" | "MJPEG";
  skip_identity_check?: boolean;
}

export interface ReplaySessionOptions {
  repo_id: string;
  episode_index: number;
  skip_identity_check?: boolean;
}

export interface CalibrationSessionOptions {
  /** Which physical arm slot to calibrate: "robot" = follower, "teleop" =
   * leader; "left" is also the single-arm pair. Backend enum values — data,
   * never display copy. */
  device_type: "robot" | "teleop";
  arm?: "left" | "right";
  /** Calibration is the setup flow, so the port may ride here (the UI's
   * unsaved draft pick); omitted, the record's saved slot port is used. */
  port?: string;
  /** Save name; omitted, the slot's assigned config (else the robot's
   * default name for the slot) is used. */
  config_file?: string;
  overwrite?: boolean;
}

export interface AutoCalibrationArmSessionOption {
  device_type: "robot" | "teleop";
  arm?: "left" | "right";
  port?: string;
  config_file?: string;
}

export interface AutoCalibrationSessionOptions {
  /** 1-4 arm slots, all run CONCURRENTLY (a single arm is a batch of one). */
  arms: AutoCalibrationArmSessionOption[];
  /** Drive torque percent (10-100); omitted, the record's saved motor_power. */
  motor_power?: number;
  overwrite?: boolean;
}

export type SessionOptions =
  | TeleoperationSessionOptions
  | RecordingSessionOptions
  | InferenceSessionOptions
  | RemoteInferenceSessionOptions
  | ReplaySessionOptions
  | CalibrationSessionOptions
  | AutoCalibrationSessionOptions;

export interface StartSessionArgs {
  kind: StartableSessionKind;
  /** Robot RECORD name — the server resolves everything hardware-shaped. */
  robot: string;
  /** This tab's identity (lib/sessionOwner.ts). Attaches the lease. */
  owner: string;
  options: SessionOptions;
  /** Server default: 60s. Only pass to deviate. */
  lease_timeout_s?: number;
}

export interface StartedSession {
  session: SessionInfo;
  /** Warn-but-allow findings from the feature's start (teleoperation/replay
   * arm-identity checks): the session RUNS, but the caller should surface
   * them. Backend prose — rendered verbatim, never translated. */
  warnings: string[] | null;
}

export async function startSession(
  baseUrl: string,
  fetcher: Fetcher,
  args: StartSessionArgs
): Promise<StartedSession> {
  const { session, warnings } = await apiRequest<{
    session: SessionInfo;
    warnings?: string[] | null;
  }>(baseUrl, fetcher, "/api/v1/sessions", {
    method: "POST",
    body: args,
    action: `Start ${args.kind} session`,
  });
  return { session, warnings: warnings ?? null };
}

export async function getCurrentSession(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal
): Promise<{ session: SessionInfo | null; last_ended: EndedSessionInfo | null }> {
  return apiRequest(baseUrl, fetcher, "/api/v1/sessions/current", {
    signal,
    action: "Get current session",
  });
}

/** Renew the lease. 404 once the session is gone; 409 on owner mismatch or an
 * in-flight expiry stop — the caller treats every failure as non-fatal. */
export async function heartbeatSession(
  baseUrl: string,
  fetcher: Fetcher,
  sessionId: string,
  owner: string
): Promise<SessionInfo> {
  const { session } = await apiRequest<{ session: SessionInfo }>(
    baseUrl,
    fetcher,
    `/api/v1/sessions/${encodeURIComponent(sessionId)}/heartbeat`,
    { method: "POST", body: { owner }, action: "Heartbeat session" }
  );
  return session;
}

/** Stop by id — deliberately never owner-gated server-side (safety outranks
 * ownership). `result` is the kind's legacy stop-handler response verbatim
 * (teleoperation's `releasing`/`warning`, etc.). 404 session.not_found when
 * the id no longer names the current session — for a stop, that means the
 * session is already gone and there is nothing left to do. */
export async function stopSession(
  baseUrl: string,
  fetcher: Fetcher,
  sessionId: string
): Promise<{ session: SessionInfo; result: Record<string, unknown> }> {
  return apiRequest(
    baseUrl,
    fetcher,
    `/api/v1/sessions/${encodeURIComponent(sessionId)}/stop`,
    { method: "POST", action: "Stop session" }
  );
}

export type CoachingCommand =
  | "takeover"
  | "handback"
  | "cancel"
  | "hold"
  | "resume"
  | "reset"
  | "recovered"
  | "drop_last";

/** One coaching (DAgger) command for the current inference session, by id.
 *
 * Session-scoped and never owner-gated, exactly like `stopSession` — a
 * physical arm must stay controllable by whoever can reach the API. 404 once
 * the session is gone; the runner returns a 409 for a plain (non-coaching)
 * inference session or one that is still starting up. The verb is NOT
 * phase-checked client-side: the runner's phase is authoritative and the
 * browser only holds a copy that is one poll stale, so a mistimed command is a
 * harmless no-op the runner logs.
 *
 * The id is the whole point, and it is what a restack quietly dropped: without
 * it the dialog posts to a session-agnostic verb, so a dialog left open across
 * a session change commands whichever session happens to be current rather
 * than failing with a 404 the UI can report. */
export async function sendCoachingCommand(
  baseUrl: string,
  fetcher: Fetcher,
  sessionId: string,
  command: CoachingCommand
): Promise<{ result: Record<string, unknown> }> {
  return apiRequest(
    baseUrl,
    fetcher,
    `/api/v1/sessions/${encodeURIComponent(sessionId)}/coaching`,
    { method: "POST", body: { command }, action: "Coaching command" }
  );
}

// --- 409 session.held rendering ---------------------------------------------

/** The holder named by a 409 session.held error, or null when `e` is any
 * other failure. `kind` may be null when even the server couldn't name it. */
export function sessionHeldHolder(e: unknown): { kind: string | null } | null {
  if (!(e instanceof ApiError) || e.code !== "session.held") return null;
  const holder = (e.details as { holder?: { kind?: string | null } } | null)
    ?.holder;
  return { kind: holder?.kind ?? null };
}

// Static catalog keys (never a runtime-built template) so keyUsage.test.ts can
// verify each one resolves. The holder kind is backend data — matched on,
// never displayed raw except as the last-resort fallback.
const HOLDER_ACTIVITY_KEYS: Record<string, string> = {
  teleoperation: "shared.sessionBusy.activity.teleoperation",
  recording: "shared.sessionBusy.activity.recording",
  inference: "shared.sessionBusy.activity.inference",
  remote_inference: "shared.sessionBusy.activity.remote_inference",
  replay: "shared.sessionBusy.activity.replay",
  calibration: "shared.sessionBusy.activity.calibration",
  auto_calibration: "shared.sessionBusy.activity.auto_calibration",
  wiggle: "shared.sessionBusy.activity.wiggle",
};

/**
 * Localized "robot is busy (kind)" line for a 409 session.held, or null when
 * `e` is any other failure (the caller then falls back to its usual error
 * rendering). Every flow's start path funnels held-refusals through this so
 * the message is one string in the catalogs, not four ad-hoc variants.
 */
export function formatSessionHeld(t: TFunction, e: unknown): string | null {
  const held = sessionHeldHolder(e);
  if (!held) return null;
  const key = held.kind != null ? HOLDER_ACTIVITY_KEYS[held.kind] : undefined;
  if (!key) return t("shared.sessionBusy.generic");
  return t("shared.sessionBusy.message", { activity: t(key as never) });
}
