import React, { useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useLanguage } from "@/contexts/LanguageContext";
import { isCaselessScript } from "@/i18n/config";
import { EpisodeJointSeries, EpisodeSummary } from "@/lib/replayApi";

// Six SO-101 joints, in the dataset's fixed column order — a validated
// categorical set (see dataviz skill palette) so adjacent lines stay
// distinguishable; cycles defensively if a feature set ever carries more.
const JOINT_COLORS = [
  "#2a78d6",
  "#eb6834",
  "#1baf7a",
  "#eda100",
  "#e87ba4",
  "#008300",
];

// Opacity for a joint's chart line when at least one *other* joint is
// selected — dimmed rather than removed, so the de-emphasized traces still
// give spatial context for the ones the user is isolating.
const DIMMED_LINE_OPACITY = 0.15;
// Legend chips dim less than the lines: the label still needs to be legible
// enough to read and click again to re-select it.
const DIMMED_LEGEND_OPACITY = 0.5;

/**
 * Joint-position chart synced to the playhead, with a clickable legend: click
 * a joint to isolate it (the rest dim), click a second to isolate both, click
 * either again to deselect. No selection means "show everything" — today's
 * default. Selection is local state, so it naturally persists across episode
 * changes (the parent doesn't remount this component when switching episodes)
 * and resets whenever the dialog itself is closed and reopened.
 */
const JointPositionChart: React.FC<{
  joints: EpisodeJointSeries | null;
  episode: EpisodeSummary | null;
  scrubFrac: number;
}> = ({ joints, episode, scrubFrac }) => {
  const { t } = useTranslation();
  const { language } = useLanguage();
  // `.eyebrow` bundles `uppercase` with letter-spacing: the uppercase is a
  // no-op on Chinese but the tracking is not, so both come off together.
  const eyebrow = isCaselessScript(language)
    ? "text-[11px] font-semibold text-muted-foreground"
    : "eyebrow";
  const [hoverT, setHoverT] = useState<number | null>(null);
  const [selectedJoints, setSelectedJoints] = useState<Set<number>>(new Set());

  const toggleJoint = (j: number) => {
    setSelectedJoints((prev) => {
      const next = new Set(prev);
      if (next.has(j)) next.delete(j);
      else next.add(j);
      return next;
    });
  };

  // Whether joint `j` is emphasized (full opacity) given the current
  // selection — everything is emphasized when nothing is selected.
  const isEmphasized = (j: number) => selectedJoints.size === 0 || selectedJoints.has(j);

  const jointRanges = useMemo(() => {
    if (!joints || joints.values.length === 0) return [];
    // Reduce, not Math.min(...vals)/Math.max(...vals) — spreading a large
    // array into a function call can throw "Maximum call stack size
    // exceeded" (engine-dependent, but well within reach of a long episode's
    // frame count), which would blank the whole chart.
    return joints.joint_names.map((_, j) => {
      let min = Infinity;
      let max = -Infinity;
      for (const frame of joints.values) {
        const v = frame[j];
        if (v < min) min = v;
        if (v > max) max = v;
      }
      return { min, max };
    });
  }, [joints]);

  const chartRef = useRef<HTMLDivElement>(null);
  const handleChartHover = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!episode || !joints) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    setHoverT(frac * episode.duration);
  };

  return (
    <div className="rounded-md border border-border bg-muted/40 p-2">
      <div className="mb-1 flex items-baseline justify-between select-none">
        <span className={eyebrow}>{t("dialogs.jointChart.heading")}</span>
        <span className={eyebrow}>
          {episode
            ? t("dialogs.jointChart.episode", {
                index: episode.episode_index,
              })
            : "—"}
        </span>
      </div>
      <div
        ref={chartRef}
        className="relative h-[72px]"
        onMouseMove={handleChartHover}
        onMouseLeave={() => setHoverT(null)}
      >
        {joints && episode && episode.duration > 0 ? (
          <svg viewBox="0 0 600 72" preserveAspectRatio="none" className="h-full w-full">
            {[0, 1, 2, 3, 4].map((i) => (
              <line
                key={i}
                x1={0}
                x2={600}
                y1={(i / 4) * 72}
                y2={(i / 4) * 72}
                stroke="hsl(var(--border))"
                strokeWidth={1}
              />
            ))}
            {joints.joint_names.map((_, j) => {
              const range = jointRanges[j];
              if (!range) return null;
              const span = range.max - range.min || 1;
              const points = joints.timestamps
                .map((t, i) => {
                  const x = (t / episode.duration) * 600;
                  const v = joints.values[i][j];
                  const y = 72 - ((v - range.min) / span) * 72;
                  return `${x.toFixed(1)},${y.toFixed(1)}`;
                })
                .join(" ");
              return (
                <polyline
                  key={j}
                  points={points}
                  fill="none"
                  stroke={JOINT_COLORS[j % JOINT_COLORS.length]}
                  strokeWidth={1.5}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  opacity={isEmphasized(j) ? 1 : DIMMED_LINE_OPACITY}
                />
              );
            })}
            <line
              x1={scrubFrac * 600}
              x2={scrubFrac * 600}
              y1={0}
              y2={72}
              stroke="hsl(var(--foreground))"
              strokeWidth={1}
              opacity={0.55}
            />
          </svg>
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
            {episode
              ? t("dialogs.jointChart.loading")
              : t("dialogs.jointChart.noEpisode")}
          </div>
        )}
        {hoverT != null && joints && episode && (
          <div
            className="pointer-events-none absolute top-1 rounded border border-border bg-popover px-1.5 py-1 text-[10px] shadow"
            style={{
              left: Math.min(
                Math.max((hoverT / episode.duration) * 100, 0),
                85,
              ) + "%",
            }}
          >
            <div className="mb-0.5 font-semibold">t = {hoverT.toFixed(1)}s</div>
            {joints.joint_names.map((name, j) => {
              if (!isEmphasized(j)) return null;
              const idx = joints.timestamps.reduce(
                (best, t, i) =>
                  Math.abs(t - hoverT) < Math.abs(joints.timestamps[best] - hoverT) ? i : best,
                0,
              );
              return (
                <div key={name} className="flex items-center gap-1 tabular-nums">
                  <span
                    className="h-1.5 w-1.5 shrink-0 rounded-sm"
                    style={{ background: JOINT_COLORS[j % JOINT_COLORS.length] }}
                  />
                  {name}: {joints.values[idx][j].toFixed(1)}
                </div>
              );
            })}
          </div>
        )}
      </div>
      {joints && (
        <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1">
          {joints.joint_names.map((name, j) => {
            const selected = selectedJoints.has(j);
            return (
              <button
                key={name}
                type="button"
                onClick={() => toggleJoint(j)}
                className={`flex items-center gap-1 rounded px-1 py-0.5 text-[10px] transition-colors ${
                  selected
                    ? "bg-accent text-foreground"
                    : "text-muted-foreground hover:bg-accent/50"
                }`}
                style={{ opacity: isEmphasized(j) ? 1 : DIMMED_LEGEND_OPACITY }}
                aria-pressed={selected}
              >
                <span
                  className="h-2 w-2 shrink-0 rounded-sm"
                  style={{ background: JOINT_COLORS[j % JOINT_COLORS.length] }}
                />
                {name}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default JointPositionChart;
