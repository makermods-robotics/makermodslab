import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { AlertTriangle, Loader2 } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  perSecondRate,
  type RemoteInferenceStatus,
} from "@/hooks/useRemoteInferenceStatus";

/**
 * Live telemetry of a remote-inference run.
 *
 * Two presentation decisions here are load-bearing and were paid for on the
 * bench:
 *
 *  - **`holds` is rendered as a RATE, not the counter.** A healthy run's
 *    `holds` climbs during warm-up and then FREEZES. The cumulative number
 *    (41 at t=1s) stays on screen forever afterwards looking like a problem,
 *    while the thing that actually matters — is it still growing? — is
 *    invisible. The derivative says it directly.
 *  - **`lead` is shown against `horizon − s_min`, not on its own.** That
 *    difference is the scheduler's whole safety margin: lead falling into it
 *    is what precedes a hold, and a bare number tells the operator nothing
 *    about how close they are.
 *
 * Everything the backend calls an error, a hint or a warning is ITS prose and
 * is rendered exactly as sent.
 */

/** The phase ladder, in order. Values are backend enums — matched on, never
 * displayed raw except as the last-resort fallback. `stopped` / `error` are
 * terminal and sit outside the ladder. */
const PHASE_LADDER = [
  "resolving",
  "transport_check",
  "preflight",
  "starting",
  "connecting",
  "warming_up",
  "easing",
  "running",
  "stopping",
] as const;

/** µs → ms, at the one decimal the numbers actually justify. */
const usToMs = (us: number | null): string | null =>
  us == null ? null : (us / 1000).toFixed(1);

const Metric: React.FC<{
  label: string;
  value: React.ReactNode;
  tone?: "normal" | "warn" | "muted";
}> = ({ label, value, tone = "normal" }) => (
  <div className="min-w-0">
    <p className="truncate text-[10px] text-muted-foreground">{label}</p>
    <p
      className={cn(
        "truncate font-mono text-sm",
        tone === "warn" && "text-warn",
        tone === "muted" && "text-muted-foreground",
      )}
    >
      {value}
    </p>
  </div>
);

const RemoteInferenceStatusPanel: React.FC<{
  status: RemoteInferenceStatus;
  /** Null while this tab does not know which session to stop (a reload, or a
   * run started elsewhere) — the caller resolves it lazily. */
  onStop: (() => void) | null;
  stopping: boolean;
}> = ({ status, onStop, stopping }) => {
  const { t } = useTranslation();
  const stats = status.stats;

  // The previous sample, kept only to difference `holds` against.
  const [holdsRate, setHoldsRate] = useState<number | null>(null);
  const prevRef = useRef<{ t: number; value: number } | null>(null);

  // Reset on a NEW RUN, not on the run ending. Resetting on `!active` blanked
  // the rate to "—" the instant a run finished, at exactly the moment someone
  // reading a failed run wants to know whether the arm had been starving. The
  // last computed rate now stands as part of the terminal picture. `started_at`
  // is the run's identity: it survives into the terminal payload and only
  // changes when the next run claims the slot. Declared FIRST so that on the
  // render where a new run's first sample arrives, this clears before the
  // difference below is taken.
  useEffect(() => {
    prevRef.current = null;
    setHoldsRate(null);
  }, [status.started_at]);

  useEffect(() => {
    // Only a live run contributes samples. A terminal payload repeats its last
    // one on every poll, and differencing it against itself is a dt of 0 —
    // which `perSecondRate` reports as "no rate" rather than as a false zero.
    if (!status.remote_inference_active || !stats) return;
    const current = { t: stats.t, value: stats.holds };
    const rate = perSecondRate(current, prevRef.current);
    prevRef.current = current;
    if (rate != null) setHoldsRate(rate);
  }, [stats, status.remote_inference_active]);

  const phaseIndex = status.phase
    ? PHASE_LADDER.indexOf(status.phase as (typeof PHASE_LADDER)[number])
    : -1;
  const terminal = status.phase === "stopped" || status.phase === "error";

  // The margin the scheduler is working with. `lead` below it is the state
  // that precedes a hold.
  const margin = stats ? stats.horizon - stats.s_min : null;
  const marginPct =
    stats && margin != null && margin > 0
      ? Math.max(0, Math.min(100, (stats.lead / margin) * 100))
      : 0;

  return (
    <div className="space-y-3 rounded-lg border border-border p-3">
      {/* Phase --------------------------------------------------------- */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between gap-2">
          <p className="flex items-center gap-1.5 text-xs font-semibold text-foreground">
            {status.remote_inference_active && !terminal ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : null}
            {status.phase
              ? t(`remoteInference.phase.${status.phase}` as never, {
                  defaultValue: status.phase,
                })
              : t("remoteInference.phase.idle")}
            {/* Which chunk player is driving the arm. Shown even for a run this
                tab did not start — it is also which of the two `modal run`
                lines the other terminal has to be running. The engine VALUE is
                a backend identifier; only its label is translated. */}
            {status.engine ? (
              <span className="rounded border border-border px-1.5 py-0.5 text-[10px] font-normal text-muted-foreground">
                {t(`remoteInference.engine.${status.engine}` as never, {
                  defaultValue: status.engine,
                })}
              </span>
            ) : null}
          </p>
          <p className="font-mono text-xs text-muted-foreground">
            {t("remoteInference.status.elapsed", {
              elapsed: Math.round(status.elapsed_s),
              duration:
                status.duration_s && status.duration_s > 0
                  ? String(status.duration_s)
                  : "∞",
            })}
          </p>
        </div>
        {/* Dot ladder: position without a wall of labels. */}
        <div className="flex gap-1">
          {PHASE_LADDER.map((phase, i) => (
            <span
              key={phase}
              className={cn(
                "h-1 flex-1 rounded-full",
                status.phase === "error"
                  ? "bg-destructive/40"
                  : i <= phaseIndex || terminal
                    ? "bg-primary"
                    : "bg-muted",
              )}
            />
          ))}
        </div>
        {status.returning_to_rest ? (
          <p className="text-xs text-warn">
            {t("remoteInference.status.returningToRest")}
          </p>
        ) : null}
      </div>

      {/* Live sample ---------------------------------------------------- */}
      {stats ? (
        <>
          <div className="grid grid-cols-3 gap-x-3 gap-y-2">
            <Metric
              label={t("remoteInference.status.operator")}
              // The operator's identity is a LiveKit participant name — data.
              value={
                stats.active ?? t("remoteInference.status.noOperatorYet")
              }
              tone={stats.active ? "normal" : "muted"}
            />
            <Metric
              label={t("remoteInference.status.chunks")}
              value={`${stats.chunks} / ${stats.reqs}`}
            />
            <Metric
              label={t("remoteInference.status.chunkAge")}
              value={
                stats.chunk_age_ms == null
                  ? "—"
                  : `${Math.round(stats.chunk_age_ms)} ms`
              }
              tone={stats.chunk_age_ms == null ? "muted" : "normal"}
            />
            <Metric
              label={t("remoteInference.status.e2e")}
              value={
                stats.e2e_p50_us == null && stats.e2e_p95_us == null
                  ? "—"
                  : `${usToMs(stats.e2e_p50_us) ?? "—"} / ${
                      usToMs(stats.e2e_p95_us) ?? "—"
                    } ms`
              }
              tone={stats.e2e_p50_us == null ? "muted" : "normal"}
            />
            <Metric
              label={t("remoteInference.status.rtt")}
              value={
                stats.rtt_us == null ? "—" : `${usToMs(stats.rtt_us)} ms`
              }
              tone={stats.rtt_us == null ? "muted" : "normal"}
            />
            <Metric
              label={t("remoteInference.status.holdsRate")}
              // The DERIVATIVE. A frozen counter is the healthy state, and the
              // cumulative total misreports it forever after warm-up.
              value={
                holdsRate == null
                  ? "—"
                  : t("remoteInference.status.holdsPerSecond", {
                      rate: holdsRate.toFixed(1),
                    })
              }
              tone={
                holdsRate == null ? "muted" : holdsRate > 0 ? "warn" : "normal"
              }
            />
          </div>

          {/* lead vs the margin -------------------------------------- */}
          <div className="space-y-1">
            <div className="flex items-center justify-between gap-2 text-[10px] text-muted-foreground">
              <span>{t("remoteInference.status.leadLabel")}</span>
              <span className="font-mono">
                {t("remoteInference.status.leadValue", {
                  lead: stats.lead,
                  margin: margin ?? 0,
                })}
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
              <div
                className={cn(
                  "h-full rounded-full transition-[width]",
                  marginPct < 34 ? "bg-warn" : "bg-primary",
                )}
                style={{ width: `${marginPct}%` }}
              />
            </div>
            {stats.degrade ? (
              <span className="inline-flex items-center gap-1 rounded border border-destructive/50 px-1.5 py-0.5 text-[10px] font-semibold text-destructive">
                {/* DEGRADE is the child's own flag name — kept as an
                    identifier so the log line and the badge read alike. */}
                DEGRADE
                <span className="font-normal">
                  {t("remoteInference.status.degradeHint")}
                </span>
              </span>
            ) : null}
          </div>
        </>
      ) : status.remote_inference_active ? (
        <p className="text-xs text-muted-foreground">
          {t("remoteInference.status.noSampleYet")}
        </p>
      ) : null}

      {/* Warn-but-allow finding, and the terminal verdict --------------- */}
      {status.warning ? (
        <Alert className="border-warn/40 text-warn [&>svg]:text-warn">
          <AlertTriangle className="h-4 w-4" />
          {/* Backend prose — rendered exactly as sent. */}
          <AlertDescription>{status.warning}</AlertDescription>
        </Alert>
      ) : null}

      {status.exited && status.outcome ? (
        <div className="space-y-1.5">
          <p
            className={cn(
              "text-xs font-semibold",
              status.outcome === "ok" ? "text-foreground" : "text-destructive",
            )}
          >
            {t(`remoteInference.outcome.${status.outcome}` as never, {
              defaultValue: status.outcome,
            })}
          </p>
          {status.error ? (
            <pre className="overflow-x-auto rounded border border-border bg-muted/60 p-2 font-mono text-[11px] whitespace-pre-wrap">
              {status.error}
            </pre>
          ) : null}
          {status.hint ? (
            <p className="text-xs leading-relaxed text-muted-foreground">
              {status.hint}
            </p>
          ) : null}
        </div>
      ) : null}

      {/* The log the backend opened itself — a real path, shown verbatim. */}
      {status.log_path ? (
        <p className="font-mono text-[10px] break-all text-muted-foreground">
          {status.log_path}
        </p>
      ) : null}

      {status.remote_inference_active ? (
        <Button
          type="button"
          variant="destructive"
          size="sm"
          onClick={() => onStop?.()}
          disabled={stopping || onStop == null}
          className="w-full"
        >
          {stopping
            ? t("remoteInference.status.stopping")
            : t("remoteInference.status.stop")}
        </Button>
      ) : null}
    </div>
  );
};

export default RemoteInferenceStatusPanel;
