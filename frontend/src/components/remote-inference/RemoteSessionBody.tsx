import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { AlertTriangle, GraduationCap, Square } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import LogPanel from "@/components/LogPanel";
import {
  PHASE_DOT,
  PHASE_TEXT,
  PILL_BG,
  formatTime,
} from "@/components/inference/sessionFrame";
import { cn } from "@/lib/utils";
import {
  perSecondRate,
  type RemoteInferenceStatus,
} from "@/hooks/useRemoteInferenceStatus";
import Sparkline, { SPARKLINE_CAPACITY, pushSample } from "./Sparkline";

/**
 * A remote-inference (DRTC) run, inside the session dialog every local run
 * already opens.
 *
 * The FRAME is the shared one (`inference/sessionFrame`): the same pill, the
 * same big timer, the same policy line, the same full-width Stop and phase
 * line. Only the middle differs — where a coaching run shows its correction
 * tally, a remote run shows what the GPU is actually doing.
 *
 * Two presentation decisions here are load-bearing and were paid for on the
 * bench (they moved here verbatim from the retired inline status panel):
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
 * The GPU is NOT controlled from here. Start GPU / Stop GPU and the idle-stop
 * countdown stay on the Deploy panel, because the GPU outlives the run and is a
 * Lab-level resource beside the session rather than part of it. This card only
 * reports.
 *
 * Everything the backend calls an error, a hint or a warning is ITS prose and
 * is rendered exactly as sent; every value in the metrics is data.
 */

/** The status chip at the top of the dialog. `label` is already resolved — the
 * caller owns the wording, this owns the look (the same look the local run's
 * pill wears; see `sessionFrame`). */
const StatusPill: React.FC<{
  tone: "amber" | "green" | "red";
  label: string;
  /** Steady rather than pulsing once the run has stopped. */
  pulse?: boolean;
}> = ({ tone, label, pulse = true }) => (
  <div className="mb-6 text-center">
    <div
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-bold tracking-widest ${PILL_BG[tone]}`}
    >
      <span
        className={cn(
          "h-2 w-2 rounded-full",
          PHASE_DOT[tone],
          pulse && "animate-pulse",
        )}
      />
      {label}
    </div>
  </div>
);

/**
 * The big elapsed clock and the bar under it.
 *
 * `subtitle` is whatever belongs under the number — "/ 01:00", an unbounded
 * marker, or one fixed sentence during setup. The bar has three modes because
 * a remote run has three: `pulsing` while it is still setting up (progress is
 * unknowable), `idle` while it runs against no deadline at all, and `measured`
 * when there is a duration to fill.
 */
const SessionTimer: React.FC<{
  seconds: number;
  subtitle: React.ReactNode;
  tone: "amber" | "green" | "red";
  /** 0-100, or null when there is no deadline to measure against. */
  progress: number | null;
  barState: "pulsing" | "idle" | "measured";
}> = ({ seconds, subtitle, tone, progress, barState }) => (
  <>
    <div className="mb-4 text-center">
      <div
        className={`font-mono text-7xl leading-none font-bold ${
          tone === "red" ? "text-muted-foreground" : PHASE_TEXT[tone]
        }`}
      >
        {formatTime(seconds)}
      </div>
      <div className="mt-2 text-sm text-muted-foreground">{subtitle}</div>
    </div>
    <div className="mb-8 h-1.5 w-full rounded-full bg-muted">
      {barState === "pulsing" ? (
        <div className="h-1.5 w-full animate-pulse rounded-full bg-warn/40" />
      ) : barState === "idle" ? (
        <div className="h-1.5 w-full rounded-full bg-ok/35" />
      ) : (
        <div
          className={`h-1.5 rounded-full transition-all duration-500 ${
            tone === "red" ? "bg-destructive/35" : "bg-ok"
          }`}
          style={{ width: `${progress ?? 0}%` }}
        />
      )}
    </div>
  </>
);

/** The phase line under the action button — where the run is right now, in the
 * runner's own vocabulary. */
const PhaseLine: React.FC<{
  tone: "amber" | "green" | "red";
  label: string;
  pulse?: boolean;
}> = ({ tone, label, pulse }) => (
  <div className="mt-6 flex items-center gap-2 text-sm">
    <span
      className={cn(
        "h-2 w-2 rounded-full",
        PHASE_DOT[tone],
        pulse && "animate-pulse",
      )}
    />
    <span className={`font-medium ${PHASE_TEXT[tone]}`}>{label}</span>
  </div>
);

/** µs → ms, at the one decimal the numbers actually justify. */
const usToMs = (us: number | null): string | null =>
  us == null ? null : (us / 1000).toFixed(1);

const Metric: React.FC<{
  label: string;
  value: React.ReactNode;
  tone?: "normal" | "warn" | "muted";
  /** Optional trend under the number. Nulls in the series are gaps. */
  trend?: (number | null)[];
}> = ({ label, value, tone = "normal", trend }) => (
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
    {trend ? (
      <Sparkline values={trend} className="mt-0.5 text-muted-foreground" />
    ) : null}
  </div>
);

/** One 1 Hz sample of everything the sparklines draw. Held as one row so the
 * three lines are always the same length and the same instants. */
interface Trend {
  e2e: number | null;
  rtt: number | null;
  holds: number | null;
}

const RemoteSessionBody: React.FC<{
  status: RemoteInferenceStatus;
  /** The Modal profile of the GPU that is up, when the launcher knows one.
   * DATA — Modal's own name, shown verbatim. Null ⇒ the line is omitted
   * rather than guessed at. */
  gpuProfile: string | null;
  /** What the container ITSELF reported it is running on — e.g. "NVIDIA
   * A100-SXM4-40GB (39.6 GiB)". The only EVIDENCE about the hardware: the GPU
   * beside the profile is what the launch ASKED Modal for, and until this line
   * existed nothing anywhere could tell the two apart. Null until the policy
   * server prints it, which is when the billing line simply says less rather
   * than guessing. DATA — a vendor device string, never translated. */
  gpuDeviceName: string | null;
  onStop: () => void;
  stopping: boolean;
  /** Close the dialog once the run has ended. */
  onClose: () => void;
  /** "Policy failing? Coach it" — offered only when the launcher knows which
   * policy to hand on (a deploy prefill). Null hides the button. */
  onCoach: (() => void) | null;
}> = ({
  status,
  gpuProfile,
  gpuDeviceName,
  onStop,
  stopping,
  onClose,
  onCoach,
}) => {
  const { t } = useTranslation();
  const stats = status.stats;

  // The previous sample, kept only to difference `holds` against.
  const [holdsRate, setHoldsRate] = useState<number | null>(null);
  const [trend, setTrend] = useState<Trend[]>([]);
  const prevRef = useRef<{ t: number; value: number } | null>(null);

  // Reset on a NEW RUN, not on the run ending. Resetting on `!active` blanked
  // the picture the instant a run finished, at exactly the moment someone
  // reading a failed run wants to know whether the arm had been starving. The
  // last samples now stand as part of the terminal picture. `started_at` is the
  // run's identity: it survives into the terminal payload and only changes when
  // the next run claims the slot. Declared FIRST so that on the render where a
  // new run's first sample arrives, this clears before the difference below is
  // taken.
  useEffect(() => {
    prevRef.current = null;
    setHoldsRate(null);
    setTrend([]);
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
    setTrend((prev) =>
      pushSample(
        prev,
        {
          // Nulls stay null: the child reports no correlated round trip until
          // one lands, and drawing that as 0 ms would show a transport that
          // never happened.
          e2e: stats.e2e_p50_us,
          rtt: stats.rtt_us,
          holds: rate,
        },
        SPARKLINE_CAPACITY,
      ),
    );
  }, [stats, status.remote_inference_active]);

  const phase = status.phase;
  const terminal = status.exited === true;
  const outcome = status.outcome;
  const running = !terminal && phase === "running";
  const settingUp = !terminal && !running;

  const tone: "amber" | "green" | "red" = terminal
    ? outcome === "failed" || phase === "error"
      ? "red"
      : outcome === "ran_with_warning"
        ? "amber"
        : "green"
    : running
      ? "green"
      : "amber";

  const pillLabel = terminal
    ? outcome === "failed" || phase === "error"
      ? t("inference.pill.failed")
      : outcome === "ran_with_warning"
        ? t("inference.pill.ranWithWarning")
        : outcome === "ok"
          ? t("inference.pill.finished")
          : // Stopped with no verdict of its own: the run ended, and saying so
            // is more honest than borrowing "Finished".
            t("remoteInference.phase.stopped")
    : running
      ? t("inference.pill.running")
      : t("inference.pill.settingUp");

  // 0 / null is the backend's own unbounded contract for a remote run.
  const bounded = status.duration_s != null && status.duration_s > 0;
  const subtitle = settingUp
    ? // ONE fixed sentence for the whole setup. The live phase is named once,
      // on the phase line below — a subtitle that changed with it made the
      // number under the clock the noisiest thing on a screen whose job is to
      // say how long this has been going on.
      t("remoteInference.status.connectingSubtitle")
    : bounded
      ? `/ ${formatTime(status.duration_s ?? 0)}`
      : terminal
        ? t("remoteInference.status.unboundedDone")
        : t("remoteInference.status.unbounded");

  const barState = settingUp
    ? ("pulsing" as const)
    : terminal || bounded
      ? ("measured" as const)
      : ("idle" as const);
  const progress = terminal
    ? 100
    : bounded
      ? Math.min(100, (status.elapsed_s / (status.duration_s ?? 1)) * 100)
      : null;

  // The transport the RUNNING session resolved — the child's own echo, not
  // what the panel believed it passed. Every part is data; a part that is not
  // known is left out rather than guessed at.
  const room = status.transport?.room ?? "";
  const source = status.transport?.source ?? null;

  // The margin the scheduler is working with. `lead` below it is the state
  // that precedes a hold.
  const margin = stats ? stats.horizon - stats.s_min : null;
  const marginPct =
    stats && margin != null && margin > 0
      ? Math.max(0, Math.min(100, (stats.lead / margin) * 100))
      : 0;

  return (
    <div className="min-w-0">
      <StatusPill tone={tone} label={pillLabel} pulse={!terminal} />

      <SessionTimer
        seconds={status.elapsed_s}
        subtitle={subtitle}
        tone={tone}
        progress={progress}
        barState={barState}
      />

      {/* Which policy, where, and in which room — the line that tells two
          tabs apart. */}
      <div className="mb-6 text-xs break-all text-muted-foreground">
        {room && source
          ? t("remoteInference.status.policyLine", {
              ref: status.policy_ref ?? t("inference.unknownPolicy"),
              room,
              // The enum VALUE is data; only its label is translated, with the
              // raw value as the fallback for one this build doesn't know.
              source: t(`remoteInference.transport.source.${source}` as never, {
                defaultValue: source,
              }),
            })
          : t("remoteInference.status.policyLineNoRoom", {
              ref: status.policy_ref ?? t("inference.unknownPolicy"),
            })}
      </div>

      {/* The terminal verdict, when it is not a clean one. Backend prose. */}
      {terminal && outcome && outcome !== "ok" ? (
        <div
          className={`mb-6 rounded-lg border p-4 ${
            outcome === "ran_with_warning"
              ? "border-warn/40 bg-warn/10"
              : "border-destructive/40 bg-destructive/10"
          }`}
        >
          <div
            className={`flex items-center gap-2 text-sm font-semibold ${
              outcome === "ran_with_warning" ? "text-warn" : "text-destructive"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                outcome === "ran_with_warning" ? "bg-warn" : "bg-destructive"
              }`}
            />
            {t(`remoteInference.outcome.${outcome}` as never, {
              defaultValue: outcome,
            })}
          </div>
          {status.hint ? (
            <p
              className={`mt-2 text-sm leading-relaxed ${
                outcome === "ran_with_warning"
                  ? "text-warn/90"
                  : "text-destructive/90"
              }`}
            >
              {status.hint}
            </p>
          ) : null}
          {status.error ? (
            <pre className="mt-3 max-h-40 overflow-auto rounded bg-muted p-2 text-xs break-words whitespace-pre-wrap text-muted-foreground">
              {status.error}
            </pre>
          ) : null}
        </div>
      ) : null}

      {/* Warn-but-allow finding. The run happened; something about it was
          noisy. */}
      {status.warning ? (
        <Alert className="mb-6 border-warn/40 text-warn [&>svg]:text-warn">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{status.warning}</AlertDescription>
        </Alert>
      ) : null}

      {/* The GPU's telemetry, in the slot a coaching run gives its tally. */}
      <div className="mb-6 rounded-lg border border-border bg-muted/30 p-4">
        <div className="mb-3 flex items-center justify-between gap-2">
          <span className="flex items-center gap-2 text-sm font-semibold">
            {t("remoteInference.status.gpuCardTitle")}
            {/* Which chunk player is driving the arm — also which of the two
                `modal run` lines the GPU side is running. The engine VALUE is
                a backend identifier; only its label is translated. */}
            {status.engine ? (
              <span className="rounded border border-border px-1.5 py-0.5 text-[10px] font-normal text-muted-foreground">
                {t(`remoteInference.engine.${status.engine}` as never, {
                  defaultValue: status.engine,
                })}
              </span>
            ) : null}
          </span>
          {/* Only when the launcher actually knows the profile: this says who
              is being billed, and guessing at that is worse than silence. The
              profile name is Modal's own — data. */}
          {gpuProfile ? (
            // `title` rather than a second line: the card is dense, and this
            // is the answer to a question only asked when the billing line
            // looks wrong ("is it really on the card I picked?").
            <span
              className="text-xs text-muted-foreground"
              title={gpuDeviceName ?? undefined}
            >
              {t("remoteInference.status.gpuBilling", { profile: gpuProfile })}
            </span>
          ) : null}
        </div>

        {stats ? (
          <>
            <div className="grid grid-cols-3 gap-x-4 gap-y-2.5">
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
                trend={trend.map((s) => s.e2e)}
              />
              <Metric
                label={t("remoteInference.status.rtt")}
                value={stats.rtt_us == null ? "—" : `${usToMs(stats.rtt_us)} ms`}
                tone={stats.rtt_us == null ? "muted" : "normal"}
                trend={trend.map((s) => s.rtt)}
              />
              <Metric
                label={t("remoteInference.status.holdsRate")}
                // The DERIVATIVE. A frozen counter is the healthy state, and
                // the cumulative total misreports it forever after warm-up.
                value={
                  holdsRate == null
                    ? "—"
                    : t("remoteInference.status.holdsPerSecond", {
                        rate: holdsRate.toFixed(1),
                      })
                }
                tone={
                  holdsRate == null
                    ? "muted"
                    : holdsRate > 0
                      ? "warn"
                      : "normal"
                }
                trend={trend.map((s) => s.holds)}
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
                label={t("remoteInference.status.operator")}
                // The operator's identity is a LiveKit participant name — data.
                value={stats.active ?? t("remoteInference.status.noOperatorYet")}
                tone={stats.active ? "normal" : "muted"}
              />
            </div>

            {/* lead vs the margin ------------------------------------- */}
            <div className="mt-3 space-y-1">
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
        ) : (
          <p className="text-xs text-muted-foreground">
            {t("remoteInference.status.noSampleYet")}
          </p>
        )}
      </div>

      {/* One action, and it is the same shape a local run's is. */}
      {terminal ? (
        <div className="space-y-2">
          {onCoach ? (
            <Button
              onClick={onCoach}
              className="w-full py-6 text-lg font-semibold"
            >
              <GraduationCap className="mr-2 h-5 w-5" />
              {t("inference.coach.offer")}
            </Button>
          ) : null}
          <Button
            onClick={onClose}
            variant={onCoach ? "outline" : "secondary"}
            className="w-full py-6 text-lg font-semibold"
          >
            {t("inference.button.close")}
          </Button>
        </div>
      ) : (
        <Button
          onClick={onStop}
          disabled={stopping}
          variant="destructive"
          className="w-full py-6 text-lg font-semibold disabled:opacity-50"
        >
          <Square className="mr-2 h-5 w-5" />
          {stopping ? t("inference.button.stopping") : t("inference.button.stop")}
        </Button>
      )}

      {phase ? (
        <PhaseLine
          tone={tone}
          pulse={!terminal}
          label={t(`remoteInference.phase.${phase}` as never, {
            defaultValue: phase,
          })}
        />
      ) : null}

      {status.returning_to_rest ? (
        <p className="mt-2 text-xs text-warn">
          {t("remoteInference.status.returningToRest")}
        </p>
      ) : null}

      {/* The remote child streams no log to the browser — it writes one file
          and reports its PATH. So the log slot holds the path, verbatim, in
          the same panel a local run's output appears in: the operator opens it
          themselves. No new endpoint for that. */}
      <div className="mt-4">
        <LogPanel
          logs={status.log_path ?? t("remoteInference.status.noLogYet")}
          title={t("inference.log.title")}
          defaultCollapsed
          wrap={false}
        />
      </div>
    </div>
  );
};

export default RemoteSessionBody;
