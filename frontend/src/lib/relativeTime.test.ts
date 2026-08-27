import { describe, expect, it } from "vitest";
import { relativeTimeAgo } from "./relativeTime";

// Freeze the ENGLISH output: it is interpolated pre-formatted into translated
// sentences ({{when}}), so its unit letters must never drift or be localized
// (localization.md §5.5 — the formatDurationShort rule).
describe("relativeTimeAgo", () => {
  const now = 1_700_000_000_000;

  it("renders a dash for a missing timestamp", () => {
    expect(relativeTimeAgo(0, now)).toBe("—");
  });

  it("renders seconds under a minute", () => {
    expect(relativeTimeAgo(now - 12_000, now)).toBe("12s ago");
  });

  it("renders minutes under an hour", () => {
    expect(relativeTimeAgo(now - 4 * 60_000, now)).toBe("4m ago");
  });

  it("renders hours under a day", () => {
    expect(relativeTimeAgo(now - 3 * 3_600_000, now)).toBe("3h ago");
  });

  it("renders days beyond that", () => {
    expect(relativeTimeAgo(now - 2 * 86_400_000, now)).toBe("2d ago");
  });

  it("clamps a future timestamp to zero", () => {
    expect(relativeTimeAgo(now + 60_000, now)).toBe("0s ago");
  });
});
