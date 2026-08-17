import { Fetcher, apiRequest } from "./apiClient";

export interface StartReplayRequest {
  repo_id: string;
  episode_index: number;
  follower_port: string;
  follower_config: string;
  robot_name?: string;
}

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

export async function startReplay(
  baseUrl: string,
  fetcher: Fetcher,
  request: StartReplayRequest,
): Promise<{ message: string; warning?: string }> {
  return apiRequest(baseUrl, fetcher, "/start-replay", {
    method: "POST",
    body: request,
    action: "Start replay",
  });
}

export async function stopReplay(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ message: string }> {
  return apiRequest(baseUrl, fetcher, "/stop-replay", {
    method: "POST",
    action: "Stop replay",
  });
}

export async function getReplayStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<ReplayStatus> {
  return apiRequest<ReplayStatus>(baseUrl, fetcher, "/replay-status", {
    signal,
    action: "Get replay status",
  });
}
