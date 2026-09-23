import { Fetcher, apiRequest } from "./apiClient";

// Starting a replay no longer lives here: launch goes through POST
// /api/v1/sessions (lib/sessionApi.ts startSession, kind "replay") — the
// request carries the robot NAME plus {repo_id, episode_index}, and the
// server resolves the follower port/config from the saved record. This module
// keeps the status polling and the kind-level fallback stop.

export type ReplayPhase = "idle" | "easing_in" | "playing" | "stopping" | "done" | "error";

export interface ThermalReplayStatus {
  result: string;
  message: string;
  experiment: "baseline" | "shoulder_kp_85";
  elapsed_s: number;
  duration_s: number;
  cycles: number;
  critical: boolean;
  rest_reached: boolean;
  log_dir: string;
  actuators: Array<{
    actuator: string;
    temperature_c: number | null;
    torque_nm: number | null;
    status: string;
  }>;
}

export interface ReplayStatus {
  replay_active: boolean;
  phase: ReplayPhase;
  episode_index: number | null;
  elapsed_s: number;
  duration_s: number | null;
  error?: string | null;
  hint?: string | null;
  thermal_test?: ThermalReplayStatus | null;
}

export async function stopReplay(
  baseUrl: string,
  fetcher: Fetcher,
  releaseNow = false,
): Promise<{ message: string }> {
  return apiRequest(baseUrl, fetcher, `/api/v1/stop-replay${releaseNow ? "?release_now=true" : ""}`, {
    method: "POST",
    action: "Stop replay",
  });
}

export async function getReplayStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<ReplayStatus> {
  return apiRequest<ReplayStatus>(baseUrl, fetcher, "/api/v1/replay-status", {
    signal,
    action: "Get replay status",
  });
}
