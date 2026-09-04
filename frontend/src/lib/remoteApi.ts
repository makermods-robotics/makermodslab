import type { TFunction } from "i18next";
import { Fetcher, apiRequest, ApiError } from "./apiClient";
import { formatSessionHeld } from "./sessionApi";

/**
 * Client for the remote-teleoperation status surface — the two halves of
 * driving one node's follower from another node's leader over LiveKit:
 *
 *  - the STATION hosts (`GET /api/v1/hosting`): its follower + cameras are
 *    published into its room and it executes the active operator's actions;
 *  - the OPERATOR drives (`GET /api/v1/remote-teleoperation`): its leader
 *    streams actions to a station picked from the node registry.
 *
 * Both sessions start through POST /api/v1/sessions (lib/sessionApi.ts,
 * kinds `hosting` / `remote_teleoperation`); this module only reads status
 * and renders the coded refusals those starts can answer with.
 */

// Mirrors makermodslab/schemas/remote.py.

export interface HostingCamera {
  name: string;
  width: number;
  height: number;
}

/** The station's resting/engaged state machine (remote_host.PHASES):
 * `parked` — arm at rest with torque off, room open, listening;
 * `engaging` — the 1 s soft start; `engaged` — following the seated
 * operator; `parking` — returning to rest. Ids are data, matched on. */
export type HostingPhase = "parked" | "engaging" | "engaged" | "parking";

/** What an operator needs to join this station's room. Every field is data
 * (robot name, room, motor names, codec id) — rendered verbatim. */
export interface HostingDescriptor {
  robot: string;
  arm_type: string;
  mode: string;
  room: string;
  url: string;
  fps: number;
  video_codec: "H264" | "MJPEG" | "PNG" | "RAW";
  motors: string[];
  cameras: HostingCamera[];
  joint_ranges_deg: Record<string, number>;
  /** The operator currently driving, or null while the station waits. */
  active_operator: string | null;
  phase: HostingPhase;
  /** True when the process was started with `--host`: hosting re-arms
   * itself a few seconds after any local session ends. */
  station_mode: boolean;
}

export interface HostingStatus {
  hosting_active: boolean;
  /** Rides only while a hosting session is live. */
  hosting: HostingDescriptor | null;
  /** First stop pressed: the follower is returning to rest before torque is
   * released (a second stop releases now — the teleoperation contract). */
  releasing: boolean;
  last_cleanup_error: string | null;
  outcome: string | null;
  error: string | null;
  hint: string | null;
  message: string;
}

export interface RemoteStation {
  instance_id: string;
  name: string | null;
  url: string;
}

/** Transport metrics in ms — null until the first sample. */
export interface RemoteTeleoperationMetrics {
  rtt_ms_last: number | null;
  rtt_ms_mean: number | null;
  rtt_ms_p95: number | null;
  observations: number;
  states_dropped: number;
}

export interface RemoteTeleoperationStatus {
  remote_teleoperation_active: boolean;
  station: RemoteStation | null;
  /** The station's live phase (its descriptor, re-read at most once a
   * second by the server); null when the station could not be read. */
  station_phase: HostingPhase | null;
  room: string | null;
  /** Camera names the station publishes — each one is re-streamed at
   * `remoteCameraUrl(baseUrl, name)`. Names are data. */
  cameras: string[];
  metrics: RemoteTeleoperationMetrics | null;
  last_cleanup_error: string | null;
  outcome: string | null;
  error: string | null;
  hint: string | null;
  message: string;
}

export async function getHostingStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<HostingStatus> {
  return apiRequest<HostingStatus>(baseUrl, fetcher, "/api/v1/hosting", {
    signal,
    action: "Get hosting status",
  });
}

export async function getRemoteTeleoperationStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<RemoteTeleoperationStatus> {
  return apiRequest<RemoteTeleoperationStatus>(
    baseUrl,
    fetcher,
    "/api/v1/remote-teleoperation",
    { signal, action: "Get remote teleoperation status" },
  );
}

/** Home / Engage from the operator side. Always a 200: a refusal is
 * `success: false` with the reason in `message` (server prose, verbatim). */
export interface RemoteCommandResponse {
  success: boolean;
  message: string;
}

/** Park the station's arm (return to rest, torque off) and HOLD it parked
 * until Engage. Only the seated operator's request is honoured. */
export async function remoteHome(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<RemoteCommandResponse> {
  return apiRequest<RemoteCommandResponse>(
    baseUrl,
    fetcher,
    "/api/v1/remote-teleoperation/home",
    { method: "POST", action: "Home remote arm" },
  );
}

/** Re-energize the station's arm after a Home, with a soft start. */
export async function remoteEngage(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<RemoteCommandResponse> {
  return apiRequest<RemoteCommandResponse>(
    baseUrl,
    fetcher,
    "/api/v1/remote-teleoperation/engage",
    { method: "POST", action: "Engage remote arm" },
  );
}

/** The MJPEG re-stream of one remote camera (multipart/x-mixed-replace, the
 * same shape as /api/v1/camera-preview/{index}) — for an `<img src>`. */
export function remoteCameraUrl(baseUrl: string, name: string): string {
  return `${baseUrl}/api/v1/remote-teleoperation/camera/${encodeURIComponent(name)}`;
}

/** The `remote` optional extra's availability — the same trio as the
 * training/wandb extras (GET …/remote-extra, POST …/install, GET
 * …/install-status; useInstallExtra drives the latter two). */
export interface RemoteExtraStatus {
  available: boolean;
  /** The backend's own pip command — data, shown verbatim. */
  install_hint: string;
}

export async function getRemoteExtra(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<RemoteExtraStatus> {
  return apiRequest<RemoteExtraStatus>(
    baseUrl,
    fetcher,
    "/api/v1/system/remote-extra",
    { signal, action: "Get remote extra" },
  );
}

// --- Coded refusals ---------------------------------------------------------

/** A start refusal rendered for a toast. `needsInstall` is true only for
 * 409 system.extra_missing, where the caller offers the install flow. */
export interface RemoteRefusal {
  message: string;
  needsInstall: boolean;
}

// Static catalog keys (never a runtime-built template) so keyUsage.test.ts
// can verify each one resolves. The codes are backend data — matched on,
// never displayed. The `robot.*` code is written as a template literal
// because that test scans quoted literals for anything shaped like a
// `robot.` catalog key, and an error code is not one.
const REFUSAL_KEYS: Record<string, string> = {
  "node.not_hosting": "robot.remote.refusal.notHosting",
  "node.not_found": "robot.remote.refusal.nodeNotFound",
  "node.unreachable": "robot.remote.refusal.nodeUnreachable",
  [`robot.schema_mismatch`]: "robot.remote.refusal.schemaMismatch",
  "sfu.disabled": "robot.remote.refusal.sfuDisabled",
  // The station's single operator seat is held by someone else (409).
  "sfu.seat_taken": "robot.remote.refusal.seatTaken",
  "system.extra_missing": "robot.remote.refusal.extraMissing",
};

/**
 * Localized line for a coded refusal of a `hosting` / `remote_teleoperation`
 * start, or null when `e` is not an ApiError (the caller then renders its
 * connection-error toast). 409 session.held goes through the shared
 * formatSessionHeld; every other coded refusal (robot.not_ready, hardware.*)
 * falls back to the server's own prose.
 */
export function formatRemoteRefusal(
  t: TFunction,
  e: unknown,
  fallback: string,
): RemoteRefusal | null {
  if (!(e instanceof ApiError)) return null;
  const held = formatSessionHeld(t, e);
  if (held) return { message: held, needsInstall: false };
  const key = e.code != null ? REFUSAL_KEYS[e.code] : undefined;
  if (key) {
    return {
      message: t(key as never),
      needsInstall: e.code === "system.extra_missing",
    };
  }
  return { message: e.detail ?? fallback, needsInstall: false };
}
