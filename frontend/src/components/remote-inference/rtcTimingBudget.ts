import type { RemoteInferenceStats } from "@/hooks/useRemoteInferenceStatus";

/** Compare the unclamped prediction RTT estimate with the RTC delay window. */
export function rtcTimingBudget(stats: RemoteInferenceStats | null, fps: number | null | undefined) {
  if (!stats || stats.chunks < 1 || !fps || fps <= 0 || !Number.isFinite(fps) || !Number.isFinite(stats.lat_ms)) return null;
  const budgetMs = Math.min(Math.floor(stats.horizon / 2), stats.horizon - stats.s_min) / fps * 1000;
  if (budgetMs <= 0) return null;
  const delayMs = Math.max(0, stats.lat_ms);
  const ratio = delayMs / budgetMs;
  return { budgetMs, delayMs, percent: Math.min(100, ratio * 100), state: ratio >= 1 ? "over" : ratio >= 0.8 ? "near" : "within" } as const;
}
