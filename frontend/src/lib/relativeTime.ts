/**
 * Short relative-time formatting ("4m ago"), shared by the node rows.
 *
 * ENGLISH ON PURPOSE, like `formatDurationShort` and the per-card
 * `relativeTime` helpers this mirrors: the output is interpolated
 * pre-formatted into translated sentences ({{when}}), never assembled from
 * translated fragments — see frontend/docs/localization.md §5.5. The exact
 * strings are frozen by lib/relativeTime.test.ts.
 */
export function relativeTimeAgo(epochMs: number, nowMs: number = Date.now()): string {
  if (!epochMs) return "—";
  const diff = Math.max(0, (nowMs - epochMs) / 1000);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}
