import React, { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { TrainingStatus } from "../types";
import { Activity, Clock, Gauge, TrendingDown } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { getJobMetricsHistory } from "@/lib/jobsApi";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface MonitoringStatsProps {
  jobId: string;
  trainingStatus: TrainingStatus;
  getProgressPercentage: () => number;
  formatTime: (seconds: number) => string;
}

interface LossPoint {
  step: number;
  loss: number;
}

interface LrPoint {
  step: number;
  lr: number;
}

const HISTORY_CAP = 2000;

// Roughly how many x-axis labels we aim for. The stride is snapped to a round
// number afterwards, so the realised count lands in ~4-9.
const TICK_TARGET = 6;

/**
 * Round x-axis ticks for a step domain: multiples of a 1/2/5×10ⁿ stride sized
 * from the span, clipped to [min, max].
 *
 * Without an explicit `ticks` array Recharts treats the x-axis as its
 * categorical axis and emits one tick per data point, then culls the labels by
 * width — so a 50/100/…/2100 run renders as "150, 300, 450 … 1500, 2100":
 * every label is a real step, but the set reads as arbitrary and reshuffles on
 * any pane resize. Deriving the ticks instead (rather than hardcoding a list)
 * keeps them round and lets the stride step up on its own as a live run
 * extends or a resume chain stitches more lineage in.
 *
 * Width culling still applies to these ticks, and the last one is still kept
 * and clamped inward — so a narrow pane drops labels, but it drops them to a
 * subset of round numbers.
 */
const roundTicks = (min: number, max: number): number[] => {
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    return Number.isFinite(min) ? [min] : [];
  }
  const raw = (max - min) / TICK_TARGET;
  const mag = 10 ** Math.floor(Math.log10(raw));
  // Snap to the nearest of 1/2/5/10 in log space (Heckbert's nice-number rule);
  // rounding to nearest rather than up keeps a 15k run at a 2k stride instead
  // of jumping to 5k.
  const norm = raw / mag;
  const nice = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  // Steps are integers, so never emit a fractional stride on a tiny span.
  const stride = Math.max(1, nice * mag);
  const ticks: number[] = [];
  // Index-based so the multiples stay exact instead of accumulating float drift.
  for (let i = Math.ceil(min / stride); i * stride <= max; i += 1) {
    ticks.push(i * stride);
  }
  return ticks;
};

/** 1500 → "1.5k", 2000 → "2k", 850 → "850". */
const formatStepTick = (value: number): string => {
  if (!Number.isFinite(value)) return "";
  if (Math.abs(value) < 1000) return String(value);
  // A multiple of 100 divides into an exact single decimal of k. Anything else
  // (only reachable on a degenerately narrow domain) prints in full rather
  // than rounding e.g. 1001 down to "1k".
  if (value % 100 === 0) return `${value / 1000}k`;
  return value.toLocaleString();
};

/**
 * X-axis config shared by the loss and lr charts so their tick logic can't
 * drift apart. `ticks` is derived per series because the two can end on
 * different steps (lr is only appended when the backend reports one).
 * `points` is append-only and step-monotonic, so first/last are the domain
 * bounds — same values Recharts resolves "dataMin"/"dataMax" to.
 */
const stepAxisProps = (points: readonly { step: number }[]) =>
  ({
    dataKey: "step",
    type: "number",
    scale: "linear",
    domain: ["dataMin", "dataMax"],
    ticks: roundTicks(
      points[0]?.step ?? 0,
      points[points.length - 1]?.step ?? 0
    ),
    tickFormatter: formatStepTick,
    tick: { fill: "hsl(var(--muted-foreground))", fontSize: 11 },
    stroke: "hsl(var(--border))",
  }) satisfies React.ComponentProps<typeof XAxis>;

const MonitoringStats: React.FC<MonitoringStatsProps> = ({
  jobId,
  trainingStatus,
  getProgressPercentage,
  formatTime,
}) => {
  const [lossHistory, setLossHistory] = useState<LossPoint[]>([]);
  const [lrHistory, setLrHistory] = useState<LrPoint[]>([]);
  const lastStepRef = useRef(0);
  // The last loss value we actually charted. The backend refreshes
  // current_loss/current_lr only on a real log line (every log_freq steps) and
  // forward-fills the stale values on the tqdm ticks in between. Keying appends
  // off a loss-value change lets us plot one point per real log emission
  // instead of a flat run of duplicated points across the intervening steps.
  const lastLossValRef = useRef<number | null>(null);
  const { baseUrl, fetchWithHeaders } = useApi();

  // Seed the curves from the persisted log on mount (and when the active job
  // changes). Without this, the chart starts empty on every page reload,
  // after navigating away and back, or after a MakerMods Lab restart re-attaches to
  // a still-running job. Live-append continues from the last seeded step.
  useEffect(() => {
    let cancelled = false;
    getJobMetricsHistory(baseUrl, fetchWithHeaders, jobId)
      .then((points) => {
        if (cancelled || points.length === 0) return;
        const lossSeed: LossPoint[] = points
          .filter((p) => p.loss != null)
          .map((p) => ({ step: p.step, loss: p.loss as number }))
          .slice(-HISTORY_CAP);
        const lrSeed: LrPoint[] = points
          .filter((p) => p.lr != null)
          .map((p) => ({ step: p.step, lr: p.lr as number }))
          .slice(-HISTORY_CAP);
        setLossHistory(lossSeed);
        setLrHistory(lrSeed);
        // Prime the loss ref so the first live tick (which forward-fills the
        // last logged value) doesn't re-append a point we already seeded.
        lastLossValRef.current = lossSeed[lossSeed.length - 1]?.loss ?? null;
        // Pin lastStepRef to the last seeded step so the first live tick
        // (whose step is >= the seed's last step) doesn't trigger the
        // step-regressed reset in the live-append effect below.
        const lastSeededStep = points[points.length - 1]?.step ?? 0;
        lastStepRef.current = lastSeededStep;
      })
      .catch(() => {
        // 404 or transient — fall through; live ticks will populate from empty.
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders, jobId]);

  // Append new metric points as they arrive; reset when a new run starts
  // (current_step resets back to 0).
  //
  // Live x-values and the seeded history must share one step basis, or appends
  // land at the wrong x (and a lower live step wipes the seed via the regress
  // reset below). They do: both come from the backend's single
  // parse_metrics_into, with the same resume rebasing applied — the live stream
  // via the runner's log tail, the seed via /jobs/{id}/metrics-history. Keep it
  // that way; a runner that forgets to pass resume_total desynchronises them.
  useEffect(() => {
    const step = trainingStatus.current_step;
    if (step < lastStepRef.current) {
      setLossHistory([]);
      setLrHistory([]);
      lastLossValRef.current = null;
    }
    lastStepRef.current = step;

    // A new log line refreshes current_loss (and current_lr alongside it).
    // Between log lines the backend forward-fills the stale values on every
    // tqdm tick, so we key off a loss-value change to detect a genuine new
    // emission and append both series once per real log point — not once per
    // step. Loss effectively never repeats to 4+ decimals, so this won't drop
    // real points, and lr gets a point at each log step even on a constant
    // schedule.
    if (
      step > 0 &&
      trainingStatus.current_loss != null &&
      trainingStatus.current_loss !== lastLossValRef.current
    ) {
      const loss = trainingStatus.current_loss;
      const lr = trainingStatus.current_lr;
      lastLossValRef.current = loss;
      setLossHistory((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.step === step) return prev;
        return [...prev, { step, loss }].slice(-HISTORY_CAP);
      });
      if (lr != null) {
        setLrHistory((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.step === step) return prev;
          return [...prev, { step, lr }].slice(-HISTORY_CAP);
        });
      }
    }
  }, [
    trainingStatus.current_step,
    trainingStatus.current_loss,
    trainingStatus.current_lr,
  ]);

  const progress = getProgressPercentage();
  // Until tqdm fires its first progress line, total_steps is 0 — show
  // "Training starting…" instead of a misleading 0/0 0% reading.
  const isStarting =
    trainingStatus.training_active && trainingStatus.total_steps === 0;
  const stepLabel = isStarting
    ? "Training starting…"
    : `${trainingStatus.current_step.toLocaleString()} / ${trainingStatus.total_steps.toLocaleString()}`;
  // Only a RUNNING job has time remaining. `eta_seconds` is the last value the
  // log parser extrapolated, and it survives on the record after the run ends —
  // so a finished job whose log stream died mid-flight (MT47) would otherwise
  // render a confident countdown next to its "Done" badge, which is how a stale
  // reading reads as a live one.
  const etaLabel =
    trainingStatus.training_active && trainingStatus.eta_seconds != null
      ? formatTime(trainingStatus.eta_seconds)
      : "—";

  return (
    <div className="space-y-6">
      <Card className="bg-card border-border rounded-md">
        <CardContent className="p-5">
          <div className="mb-3 flex items-baseline justify-between gap-3">
            <div>
              <h3 className="eyebrow flex items-center gap-1.5">
                <Activity className="h-3.5 w-3.5" /> Progress
              </h3>
              <div className="mt-1 text-base font-semibold tabular-nums text-foreground">
                {stepLabel}
              </div>
            </div>
            <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
              <Clock className="h-3.5 w-3.5" />
              <span>
                ETA{" "}
                <span className="font-semibold tabular-nums text-foreground">
                  {etaLabel}
                </span>
              </span>
            </div>
          </div>
          {/* Same bar family as JobCard's in-library progress strip. */}
          <div className="relative h-5 w-full overflow-hidden rounded-md bg-muted border border-border">
            <div
              className="h-full bg-info transition-[width] duration-500"
              style={{ width: `${progress}%` }}
            />
            <div className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-white tabular-nums drop-shadow">
              {isStarting ? "warming up…" : `${progress.toFixed(1)}%`}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-card border-border rounded-md">
          <CardHeader className="pb-2">
            <h3 className="eyebrow flex items-center gap-1.5">
              <TrendingDown className="h-3.5 w-3.5" /> Loss
            </h3>
            <div className="text-base font-semibold tabular-nums text-foreground">
              {trainingStatus.current_loss?.toFixed(4) ?? "—"}
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="h-48">
              {lossHistory.length === 0 ? (
                <div className="flex h-full items-center justify-center text-muted-foreground text-sm">
                  Waiting for first metric tick…
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={lossHistory}
                    margin={{ top: 8, right: 12, left: 0, bottom: 0 }}
                  >
                    <XAxis {...stepAxisProps(lossHistory)} />
                    <YAxis
                      tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                      stroke="hsl(var(--border))"
                      width={48}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "hsl(var(--popover))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: 8,
                      }}
                      labelStyle={{ color: "hsl(var(--muted-foreground))" }}
                      itemStyle={{ color: "hsl(var(--ok))" }}
                      formatter={(v: number) => v.toFixed(4)}
                    />
                    <Line
                      type="monotone"
                      dataKey="loss"
                      stroke="hsl(var(--ok))"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border rounded-md">
          <CardHeader className="pb-2">
            <h3 className="eyebrow flex items-center gap-1.5">
              <Gauge className="h-3.5 w-3.5" /> Learning rate
            </h3>
            <div className="text-base font-semibold tabular-nums text-foreground">
              {trainingStatus.current_lr?.toExponential(2) ?? "—"}
            </div>
          </CardHeader>
          <CardContent className="pt-0">
            <div className="h-48">
              {lrHistory.length === 0 ? (
                <div className="flex h-full items-center justify-center text-muted-foreground text-sm">
                  Waiting for first metric tick…
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={lrHistory}
                    margin={{ top: 8, right: 12, left: 0, bottom: 0 }}
                  >
                    <XAxis {...stepAxisProps(lrHistory)} />
                    <YAxis
                      tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                      stroke="hsl(var(--border))"
                      width={48}
                      tickFormatter={(v: number) => v.toExponential(0)}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "hsl(var(--popover))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: 8,
                      }}
                      labelStyle={{ color: "hsl(var(--muted-foreground))" }}
                      itemStyle={{ color: "hsl(var(--warn))" }}
                      formatter={(v: number) => v.toExponential(2)}
                    />
                    <Line
                      type="monotone"
                      dataKey="lr"
                      stroke="hsl(var(--warn))"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default MonitoringStats;
