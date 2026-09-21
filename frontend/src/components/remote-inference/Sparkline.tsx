import React from "react";
import { cn } from "@/lib/utils";

/** How many 1 Hz samples a sparkline holds — half a minute, which is long
 * enough to see a transport degrade and short enough that the line still
 * moves. Exported so the status panel's ring buffer and the axis label agree. */
export const SPARKLINE_CAPACITY = 30;

/** Push one sample onto a ring buffer of `capacity`, oldest first. */
export function pushSample<T>(buffer: T[], sample: T, capacity: number): T[] {
  const next = buffer.length >= capacity ? buffer.slice(1) : buffer.slice();
  next.push(sample);
  return next;
}

/**
 * A trend line the width of its container and a couple of text lines tall.
 *
 * Nulls are GAPS, not zeros: the child reports `e2e_p50_us: null` until the
 * first correlated round trip lands, and drawing that as 0 ms would show a
 * transport that never happened. The y-range is the samples' own min/max so
 * the line uses the whole height — this is a shape to glance at, not a chart
 * to read numbers off (the number is the metric beside it).
 */
const Sparkline: React.FC<{
  values: (number | null)[];
  capacity?: number;
  className?: string;
}> = ({ values, capacity = SPARKLINE_CAPACITY, className }) => {
  const width = Math.max(1, capacity - 1);
  const height = 100;
  const present = values.filter((v): v is number => v != null);
  const min = present.length ? Math.min(...present) : 0;
  const max = present.length ? Math.max(...present) : 0;
  const span = max - min || 1;
  // Draw right-aligned, so a buffer still filling in grows from the right
  // edge leftwards and a full one scrolls — the newest sample is always at
  // the same place.
  const offset = capacity - values.length;
  const segments: string[] = [];
  let current: string[] = [];
  values.forEach((v, i) => {
    if (v == null) {
      if (current.length) segments.push(current.join(" "));
      current = [];
      return;
    }
    const x = offset + i;
    // 4% of padding top and bottom so the extremes are not clipped by the
    // stroke.
    const y = height - 4 - ((v - min) / span) * (height - 8);
    current.push(`${current.length ? "L" : "M"}${x} ${y.toFixed(1)}`);
  });
  if (current.length) segments.push(current.join(" "));

  return (
    <svg
      aria-hidden
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className={cn("h-6 w-full", className)}
    >
      {segments.map((d, i) => (
        <path
          key={i}
          d={d}
          fill="none"
          stroke="currentColor"
          strokeWidth={1.5}
          strokeLinejoin="round"
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </svg>
  );
};

export default Sparkline;
