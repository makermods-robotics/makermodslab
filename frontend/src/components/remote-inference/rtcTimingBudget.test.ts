import { describe, expect, it } from "vitest";
import type { RemoteInferenceStats } from "@/hooks/useRemoteInferenceStatus";
import { rtcTimingBudget } from "./rtcTimingBudget";

const stats = { chunks: 1, horizon: 30, s_min: 4, lat_steps: 15, lat_ms: 400 } as RemoteInferenceStats;
describe("RTC timing budget", () => {
  it("uses the actual run rate and unclamped delay", () => {
    expect(rtcTimingBudget(stats, 20)).toMatchObject({ budgetMs: 750, delayMs: 400, state: "within" });
    expect(rtcTimingBudget({ ...stats, lat_ms: 650 }, 20)?.state).toBe("near");
    expect(rtcTimingBudget({ ...stats, lat_ms: 900 }, 20)).toMatchObject({ state: "over", delayMs: 900, percent: 100 });
    expect(rtcTimingBudget(stats, 30)?.budgetMs).toBe(500);
  });
  it("respects the minimum execution budget", () => {
    expect(rtcTimingBudget({ ...stats, s_min: 20 }, 20)?.budgetMs).toBe(500);
  });
  it("does not report an unmeasured startup estimate as healthy", () => {
    expect(rtcTimingBudget({ ...stats, chunks: 0 }, 20)).toBeNull();
    expect(rtcTimingBudget(stats, undefined)).toBeNull();
    expect(rtcTimingBudget(null, 20)).toBeNull();
  });
});
