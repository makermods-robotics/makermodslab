import { Fetcher, apiRequest } from "./apiClient";

// Starting a replay no longer lives here: launch goes through POST
// /api/v1/sessions (lib/sessionApi.ts startSession, kind "replay") — the
// request carries the robot NAME plus {repo_id, episode_index}, and the
// server resolves the follower port/config from the saved record. This module
// keeps the status polling and the kind-level fallback stop.

export type ReplayPhase = "idle" | "easing_in" | "playing" | "stopping" | "done" | "error";

export interface ReplayStatus {
  replay_active: boolean;
  phase: ReplayPhase;
  episode_index: number | null;
  elapsed_s: number;
  duration_s: number | null;
  error?: string | null;
  hint?: string | null;
}

export async function stopReplay(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ message: string }> {
  return apiRequest(baseUrl, fetcher, "/api/v1/stop-replay", {
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
