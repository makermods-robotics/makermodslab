import React, { useEffect, useRef, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { CheckCircle2, Loader2, Play, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import {
  EpisodeResult,
  InferenceStatus,
  InferenceLogOwner,
  InferencePhase,
  getInferenceStatus,
  getInferenceLog,
  startNextInferenceEpisode,
  stopInference,
  stopInferenceEpisode,
} from "@/lib/inferenceApi";
import LogPanel from "@/components/LogPanel";
import { formatBytes } from "@/lib/formatBytes";
import { useSessionExitGuard } from "@/hooks/useSessionExitGuard";

const POLL_MS = 1000;

// Human-readable label + tone for each startup sub-phase. Drives the status
// line above the log panel so a slow startup names its substep ("Downloading
// model…", "Connecting to arm…") instead of an opaque spinner. `pulse` marks
// the still-working phases; terminal phases render steady.
const PHASE_META: Record<
  InferencePhase,
  { labelKey: string; tone: "amber" | "green" | "red"; pulse: boolean }
> = {
  downloading_model: { labelKey: "inference.phase.downloadingModel", tone: "amber", pulse: true },
  starting: { labelKey: "inference.phase.starting", tone: "amber", pulse: true },
  loading_policy: { labelKey: "inference.phase.loadingPolicy", tone: "amber", pulse: true },
  connecting: { labelKey: "inference.phase.connecting", tone: "amber", pulse: true },
  running: { labelKey: "inference.phase.running", tone: "green", pulse: true },
  stopping: { labelKey: "inference.phase.stopping", tone: "amber", pulse: true },
  stopped: { labelKey: "inference.phase.stopped", tone: "green", pulse: false },
  error: { labelKey: "inference.phase.error", tone: "red", pulse: false },
  resetting: { labelKey: "inference.phase.resetting", tone: "amber", pulse: false },
  finished: { labelKey: "inference.phase.finished", tone: "green", pulse: false },
  aborted: { labelKey: "inference.phase.aborted", tone: "amber", pulse: false },
};

// Per-episode verdict styling for the tally + the final per-episode list.
const RESULT_META: Record<
  EpisodeResult,
  { labelKey: string; dot: string; text: string }
> = {
  success: { labelKey: "inference.result.success", dot: "bg-ok", text: "text-ok" },
  failure: { labelKey: "inference.result.failure", dot: "bg-muted-foreground", text: "text-muted-foreground" },
  error: { labelKey: "inference.result.error", dot: "bg-destructive", text: "text-destructive" },
};

function tally(results: EpisodeResult[]): Record<EpisodeResult, number> {
  return {
    success: results.filter((r) => r === "success").length,
    failure: results.filter((r) => r === "failure").length,
    error: results.filter((r) => r === "error").length,
  };
}

const PHASE_DOT: Record<"amber" | "green" | "red", string> = {
  amber: "bg-warn",
  green: "bg-ok",
  red: "bg-destructive",
};

const PHASE_TEXT: Record<"amber" | "green" | "red", string> = {
  amber: "text-warn",
  green: "text-ok",
  red: "text-destructive",
};

// Pill (status chip) background + text per tone. Mirrors the dot/text maps so
// the finished-failed/warning states reuse the same palette as the phases.
const PILL_BG: Record<"amber" | "green" | "red", string> = {
  amber: "bg-warn/15 text-warn",
  green: "bg-ok/15 text-ok",
  red: "bg-destructive/15 text-destructive",
};

function formatTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

/**
 * The live inference run as a modal dialog over whatever launched it —
 * replaces the old /inference page (the polling/safety logic is ported
 * verbatim; every navigate-home became `onExit`). While the run is live
 * (including the connecting/setup window before the first status lands) the
 * dialog can't be dismissed by ESC / outside click / X: leaving stops the arm,
 * so the only exits are the explicit Stop flow, the clean-finish auto-close,
 * or the shared exit guard's confirmed leave. Once the run has ended,
 * dismissal is free.
 */
const InferenceSessionDialog: React.FC<{
  /** Called for every exit — closes the dialog, landing back where the run
   * was launched from. */
  onExit: () => void;
}> = ({ onExit }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { t } = useTranslation();
  const [status, setStatus] = useState<InferenceStatus | null>(null);
  const [logs, setLogs] = useState("");
  // Which run the fetched log belongs to. Never inferred from the text: the
  // backend says so explicitly, because a log file on disk carries no evidence
  // of which run wrote it.
  const [logOwner, setLogOwner] = useState<InferenceLogOwner>(null);
  const [stopping, setStopping] = useState(false);
  // Eval mode only: in-flight guards for the two per-episode controls, so a
  // double-click can't fire "succeeded" or "next episode" twice.
  const [endingEpisode, setEndingEpisode] = useState(false);
  const [startingNext, setStartingNext] = useState(false);
  const exitedRef = useRef(false);
  // Independent flag: we may request a stop (safety net) before the run
  // is actually inactive. We must not flip exitedRef yet — that
  // would block the natural completion path on the next tick.
  const stopRequestedRef = useRef(false);
  // Set once we've captured a finished (exited) payload we want to stay on —
  // a failure/warning we're surfacing inline. Freezes further polling so the
  // next idle status (which lacks outcome/error/hint, since the subprocess is
  // already reaped) can't clobber the error display.
  const doneRef = useRef(false);
  // The warn-but-allow arm-identity finding now arrives on the status payload
  // (the preflight runs server-side in the background), not the start response.
  // Toast it once when first seen so it isn't repeated on every poll.
  const warnedRef = useRef(false);

  // Safety net: a policy must never keep driving the arm with nobody watching.
  // While a session is active (any phase, INCLUDING downloading_model), an
  // unintentional exit stops the run — in-app back gets a blocking confirm, a
  // browser unload fires a best-effort stop beacon. There's no artifact and no
  // Done/Quit split here: the only semantic is STOP. After the run ends
  // (inference_active false) the guard disarms and navigation is free.
  const { markHandled } = useSessionExitGuard({
    active: status?.inference_active === true,
    confirmMessage: t("inference.leaveConfirm"),
    beaconUrl: `${baseUrl}/stop-inference`,
    onLeave: () => {
      stopInference(baseUrl, fetchWithHeaders).catch(() => {});
    },
    beaconFlagKey: "makermodslab:inference-stopped",
  });

  useEffect(() => {
    let cancelled = false;
    const stopIfHung = async () => {
      try {
        await stopInference(baseUrl, fetchWithHeaders);
      } catch {
        // The next status poll will surface the failure if it persists.
      }
    };
    const tick = async () => {
      // Once we've frozen on a finished-with-error payload, stop polling: a
      // later idle status would drop the outcome/error/hint we're showing.
      if (doneRef.current) return;
      try {
        const next = await getInferenceStatus(baseUrl, fetchWithHeaders);
        if (cancelled) return;
        setStatus(next);
        // Surface the server's warn-but-allow arm-identity finding once.
        if (next.warning && !warnedRef.current) {
          warnedRef.current = true;
          toast({
            title: t("inference.toast.startedWarningTitle"),
            description: next.warning,
            duration: 10000,
          });
        }
        // Pull the rollout log tail on the same tick so the panel stays live.
        // Best-effort: a log fetch failure must not disturb status handling.
        try {
          const log = await getInferenceLog(baseUrl, fetchWithHeaders);
          if (!cancelled) {
            setLogs(log.logs);
            setLogOwner(log.belongs_to);
          }
        } catch {
          // Ignore; the next tick retries.
        }
        // Handle a finished run.
        if (!next.inference_active && !exitedRef.current) {
          // A real failure or a cleanup-warning: keep the user here so the
          // hint + error snippet (rendered near the log panel) are readable
          // instead of flashing a toast and bouncing away. Freeze polling on
          // this payload.
          if (next.exited && next.outcome && next.outcome !== "ok") {
            doneRef.current = true;
            // Also surface a simple bottom-right toast (min_stable behavior):
            // the full hint + error snippet stay readable in the dialog, the
            // toast is the at-a-glance "it broke" signal.
            const failed = next.outcome === "failed";
            toast({
              title: failed
                ? t("inference.toast.failedTitle")
                : t("inference.toast.ranWithWarningTitle"),
              // next.hint / next.error are backend prose — shown as-is.
              description:
                next.hint ??
                next.error?.split("\n").at(-1) ??
                t("inference.toast.seeLog"),
              variant: failed ? "destructive" : undefined,
              duration: 10000,
            });
            return;
          }
          // An evaluation that ran its course (or was aborted) ends on its
          // SUMMARY, not by bouncing away: the accuracy and the per-episode
          // list are the entire point of the run. Freeze here and let the user
          // close. Checked after the failure branch above so a session-level
          // startup failure still renders as an error, not a summary.
          if (next.eval_mode && next.exited) {
            markHandled();
            exitedRef.current = true;
            doneRef.current = true;
            toast({
              title:
                next.phase === "aborted"
                  ? t("inference.toast.evalAbortedTitle")
                  : t("inference.toast.evalCompleteTitle"),
              description:
                next.phase === "aborted"
                  ? t("inference.toast.evalAbortedDescription")
                  : next.accuracy != null
                    ? t("inference.toast.evalAccuracy", {
                        percent: Math.round(next.accuracy * 100),
                      })
                    : t("inference.toast.evalNoScoreable"),
              duration: 10000,
            });
            return;
          }
          // A clean finish (completed / user stop): toast + auto-close.
          // Mark the exit handled so the leave guard doesn't fire a spurious
          // stop on the imminent unmount.
          markHandled();
          exitedRef.current = true;
          doneRef.current = true;
          if (next.exited) {
            toast({
              title: t("inference.toast.finishedTitle"),
              description: t("inference.toast.finishedDescription"),
            });
          }
          onExit();
          return;
        }
        // Safety net: only fire after the rollout *main loop* has actually
        // started (lerobot honours --duration there). Setup time — policy
        // load, snapshot_download, bus connect, camera connect — can take
        // 10–30s and must NOT count against the user's configured duration.
        if (
          next.inference_active &&
          next.rollout_started_at != null &&
          next.duration_s != null &&
          next.duration_s > 0 &&
          next.rollout_elapsed_s > next.duration_s + 10 &&
          !stopRequestedRef.current
        ) {
          stopRequestedRef.current = true;
          toast({
            title: t("inference.toast.hungTitle"),
            description: `Rollout past duration by ${Math.round(
              next.rollout_elapsed_s - next.duration_s,
            )}s. Stopping.`,
            variant: "destructive",
          });
          stopIfHung();
        }
      } catch (e) {
        if (!cancelled) {
          toast({
            title: t("inference.toast.lostConnectionTitle"),
            description: e instanceof Error ? e.message : String(e),
            variant: "destructive",
          });
        }
      }
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [baseUrl, fetchWithHeaders, onExit, toast, markHandled]);

  // Stops immediately — no confirmation dialog. The follower eases back to its
  // start pose and releases torque; `stopping` guards against double-fires
  // while the request is in flight.
  const handleStop = async () => {
    setStopping(true);
    // Explicit Stop — mark handled so the leave guard doesn't double-fire while
    // the run winds down.
    markHandled();
    try {
      await stopInference(baseUrl, fetchWithHeaders);
      // Status poll will catch the inactive state and close the dialog.
    } catch (e) {
      setStopping(false);
      toast({
        title: t("inference.toast.stopFailedTitle"),
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  // Eval mode: "the robot did the task". Ends THIS episode and scores it a
  // success; the session stays up and moves into its reset phase. Deliberately
  // not `handleStop` — that aborts the whole evaluation.
  const handleEpisodeSuccess = async () => {
    setEndingEpisode(true);
    try {
      await stopInferenceEpisode(baseUrl, fetchWithHeaders);
      // The status poll picks up the reset phase and the updated tally.
    } catch (e) {
      toast({
        title: t("inference.toast.endEpisodeFailedTitle"),
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setEndingEpisode(false);
    }
  };

  const handleNextEpisode = async () => {
    setStartingNext(true);
    try {
      await startNextInferenceEpisode(baseUrl, fetchWithHeaders);
    } catch (e) {
      toast({
        title: t("inference.toast.nextEpisodeFailedTitle"),
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setStartingNext(false);
    }
  };

  // Dismissal is blocked while the run is (or may still be) live — before the
  // first status lands we treat the session as live, since the launcher just
  // started it. A reset between episodes counts as live: the session still owns
  // the arm and the cameras.
  const live = status == null || status.inference_active === true;

  const setupElapsed = status?.elapsed_s ?? 0;
  const rolloutElapsed = status?.rollout_elapsed_s ?? 0;
  const duration = status?.duration_s ?? 0;
  // --- Multi-episode evaluation --------------------------------------------
  // `eval_mode` is the single flag to branch on: a plain run reports it false
  // with null companions, so everything below collapses to the old behaviour.
  const evalMode = status?.eval_mode === true;
  const episodesTotal = status?.episodes_total ?? null;
  const episodeIndex = status?.episode_index ?? null;
  const results = (status?.episode_results ?? []) as EpisodeResult[];
  const counts = tally(results);
  const accuracy = status?.accuracy ?? null;
  // Parked between episodes, waiting for the user to rearrange the scene.
  const isResetting = evalMode && status?.phase === "resetting";
  const evalFinished = evalMode && status?.phase === "finished";
  const evalAborted = evalMode && status?.phase === "aborted";
  const isEvalDone = evalFinished || evalAborted;
  // A crashed episode parks in the reset phase carrying its error — the reset
  // screen doubles as "this one broke, continue or abort?".
  const episodeCrashed = isResetting && !!status?.error;

  const isSettingUp =
    status != null &&
    status.inference_active &&
    !isResetting &&
    status.rollout_started_at == null;
  const isRunning =
    status != null && status.inference_active && status.rollout_started_at != null;

  // A finished run we're staying on to surface (see the tick): a real failure
  // (red) or a cleanup-only warning (amber). `ran_with_warning` must NOT read
  // as the red failed state — the run actually worked, only teardown was noisy.
  const isFinished = status?.exited === true && !status?.inference_active;
  const outcome = status?.outcome ?? null;
  const finishedWarn = isFinished && outcome === "ran_with_warning";
  const finishedFailed = isFinished && outcome === "failed";
  const showOutcome = finishedWarn || finishedFailed;
  // What to put in the log panel. `logs` is only THIS run's output when the
  // backend says the log belongs to the active session; a finished run's own log
  // is equally fine to show once the session has ended. Anything else means this
  // run has produced no output, and printing the text anyway is how a previous
  // run's log gets read as the current one — a live incident, where a failed run
  // showed a three-day-old run's output and the user concluded the wrong policy
  // had executed.
  const logIsThisRun =
    logOwner === "active" || (logOwner === "last_run" && !status?.inference_active);
  const logPlaceholder = finishedFailed
    ? t("inference.log.failedPlaceholder")
    : t("inference.log.emptyPlaceholder");
  // The live timer/progress block is replaced by the reset screen between
  // episodes and by the summary once an evaluation ends.
  const showTimer = !isFinished && !isResetting;

  // When setting up: progress is uncertain — show a soft pulsing bar.
  // When rolling out: progress is rolloutElapsed / duration.
  const pct =
    isRunning && duration > 0
      ? Math.min(100, (rolloutElapsed / duration) * 100)
      : 0;
  const pillTone: "amber" | "green" | "red" = finishedFailed
    ? "red"
    : finishedWarn
    ? "amber"
    : evalAborted
    ? "amber"
    : evalFinished
    ? "green"
    : isResetting
    ? "amber"
    : isSettingUp
    ? "amber"
    : "green";
  const pillLabel = finishedFailed
    ? t("inference.pill.failed")
    : finishedWarn
    ? t("inference.pill.ranWithWarning")
    : evalAborted
    ? t("inference.pill.aborted")
    : evalFinished
    ? t("inference.pill.evaluationComplete")
    : isResetting
    ? t("inference.pill.resetTheScene")
    : isSettingUp
    ? t("inference.pill.settingUp")
    : isRunning
    ? t("inference.pill.running")
    : t("inference.pill.finished");
  const timerSeconds = isRunning ? rolloutElapsed : setupElapsed;

  // Granular startup phase (from the same status poll). Suppressed once we're
  // showing the terminal outcome banner, which carries its own tone + label.
  // Null before any session has seeded a phase, or for an unrecognised value —
  // then we show nothing and let the timer/pill carry the state.
  const phaseMeta =
    !showOutcome && status?.phase ? PHASE_META[status.phase] ?? null : null;

  // Hub model download: show a real byte-progress bar during the
  // downloading_model phase. Indeterminate (pulsing) until the total is known —
  // the total can grow as file sizes are discovered, so the bar may legitimately
  // step backwards. Mirrors the sibling branch's DownloadProgressBar shape.
  const isDownloading = !showOutcome && status?.phase === "downloading_model";
  const dlDone = status?.download_bytes_done ?? null;
  const dlTotal = status?.download_bytes_total ?? null;
  const dlPercent = status?.download_percent ?? null;
  const dlDeterminate = dlPercent != null && dlTotal != null;

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !live) onExit();
      }}
    >
      <DialogContent
        hideClose
        onEscapeKeyDown={(e) => {
          if (live) e.preventDefault();
        }}
        onPointerDownOutside={(e) => {
          if (live) e.preventDefault();
        }}
        onInteractOutside={(e) => {
          if (live) e.preventDefault();
        }}
        // w-max, not w-fit: with left-1/2 positioning, fit-content shrink-wraps
        // into the half-viewport left by the offset; max-content sizes to the
        // log's longest line and the 95vw clamp does the capping.
        className="max-h-[92vh] w-max min-w-[min(36rem,95vw)] max-w-[95vw] gap-0 overflow-y-auto p-6"
        aria-describedby={undefined}
      >
        <DialogTitle className="sr-only">{t("inference.dialogTitle")}</DialogTitle>

        {!status ? (
          <div className="flex items-center justify-center py-20 text-muted-foreground">
            <Loader2 className="w-6 h-6 animate-spin mr-3" /> Connecting to
            inference…
          </div>
        ) : (
          // min-w-0 keeps the grid item from inheriting the log's unwrapped
          // line width — overflow scrolls inside the log panel, not the dialog.
          <div className="min-w-0">
            <div className="text-center mb-6">
              <div
                className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold tracking-widest ${PILL_BG[pillTone]}`}
              >
                <span
                  className={`w-2 h-2 rounded-full ${PHASE_DOT[pillTone]} ${
                    isFinished ? "" : "animate-pulse"
                  }`}
                />
                {pillLabel}
              </div>
            </div>

            {/* Evaluation header — which episode we're on, and the tally so
                far. Rendered on every eval screen (running, reset, summary) so
                the score is never more than a glance away. */}
            {evalMode && (
              <div className="mb-6 rounded-lg border border-border bg-muted/30 p-4">
                <div className="flex items-baseline justify-between gap-4">
                  <span className="text-sm font-semibold">
                    {isEvalDone
                      ? t("inference.eval.episodesTotal", {
                          count: episodesTotal ?? results.length,
                        })
                      : t("inference.eval.episodeProgress", {
                          index: episodeIndex ?? 1,
                          total:
                            episodesTotal ?? t("inference.eval.unknownTotal"),
                        })}
                  </span>
                  <span className="text-xs text-muted-foreground tabular-nums">
                    {t("inference.eval.done", { count: results.length })}
                  </span>
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-sm tabular-nums">
                  {(["success", "failure", "error"] as EpisodeResult[]).map(
                    (key) => (
                      <span key={key} className="flex items-center gap-1.5">
                        <span
                          className={`h-2 w-2 rounded-full ${RESULT_META[key].dot}`}
                        />
                        <span className={RESULT_META[key].text}>
                          {t(RESULT_META[key].labelKey as never)}
                        </span>
                        <span className="font-semibold">{counts[key]}</span>
                      </span>
                    ),
                  )}
                </div>
                {counts.error > 0 && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    {t("inference.eval.errorsExcluded")}
                  </p>
                )}
              </div>
            )}

            {/* Evaluation summary — the point of the whole run. */}
            {isEvalDone && (
              <div
                className={`mb-6 rounded-lg border p-4 ${
                  evalAborted
                    ? "border-warn/40 bg-warn/10"
                    : "border-ok/40 bg-ok/10"
                }`}
              >
                {evalAborted ? (
                  <p className="text-sm leading-relaxed text-warn">
                    {t("inference.eval.abortedSummary", {
                      done: results.length,
                      total: episodesTotal ?? t("inference.eval.unknownTotal"),
                    })}
                  </p>
                ) : accuracy != null ? (
                  <div className="text-center">
                    <div className="text-5xl font-mono font-bold leading-none text-ok">
                      {Math.round(accuracy * 100)}%
                    </div>
                    <div className="mt-2 text-sm text-muted-foreground tabular-nums">
                      {t("inference.eval.succeeded", {
                        success: counts.success,
                        scored: counts.success + counts.failure,
                      })}
                      {counts.error > 0
                        ? t("inference.eval.excludedAsErrors", {
                            count: counts.error,
                          })
                        : ""}
                    </div>
                  </div>
                ) : (
                  <p className="text-sm leading-relaxed text-warn">
                    {t("inference.eval.noScoreable")}
                  </p>
                )}
                {results.length > 0 && (
                  <ol className="mt-4 space-y-1 text-xs tabular-nums">
                    {results.map((r, i) => (
                      <li
                        key={i}
                        className="flex items-center gap-2 text-muted-foreground"
                      >
                        <span className="w-10 shrink-0">#{i + 1}</span>
                        <span
                          className={`h-1.5 w-1.5 rounded-full ${RESULT_META[r].dot}`}
                        />
                        <span className={RESULT_META[r].text}>
                          {t(RESULT_META[r].labelKey as never)}
                        </span>
                      </li>
                    ))}
                  </ol>
                )}
              </div>
            )}

            {/* Reset between episodes — user-ended, no timer. */}
            {isResetting && (
              <div
                className={`mb-6 rounded-lg border p-4 ${
                  episodeCrashed
                    ? "border-destructive/40 bg-destructive/10"
                    : "border-warn/40 bg-warn/10"
                }`}
              >
                {episodeCrashed ? (
                  <>
                    <div className="flex items-center gap-2 text-sm font-semibold text-destructive">
                      <span className="h-2 w-2 rounded-full bg-destructive" />
                      {t("inference.eval.episodeCrashed", {
                        index: results.length,
                      })}
                    </div>
                    <p className="mt-2 text-sm leading-relaxed text-destructive/90">
                      {t("inference.eval.episodeCrashedBody")}
                    </p>
                    {status.hint && (
                      <p className="mt-2 text-sm leading-relaxed text-destructive/90">
                        {status.hint}
                      </p>
                    )}
                    {status.error && (
                      <pre className="mt-3 max-h-40 overflow-auto rounded bg-muted p-2 text-xs text-muted-foreground whitespace-pre-wrap break-words">
                        {status.error}
                      </pre>
                    )}
                  </>
                ) : (
                  <p className="text-sm leading-relaxed text-warn">
                    <Trans
                      i18nKey="inference.eval.episodeRecorded"
                      values={{
                        index: results.length,
                        result: t(
                          RESULT_META[results.at(-1) ?? "failure"]
                            .labelKey as never,
                        ),
                      }}
                      components={[<span key="0" />, <strong key="1" />]}
                    />
                  </p>
                )}
              </div>
            )}

            {showTimer && (
              <>
                <div className="text-center mb-4">
                  <div
                    className={`text-7xl font-mono font-bold leading-none ${
                      isSettingUp ? "text-warn" : "text-ok"
                    }`}
                  >
                    {formatTime(timerSeconds)}
                  </div>
                  <div className="text-sm text-muted-foreground mt-2">
                    {isSettingUp
                      ? t("inference.settingUp")
                      : `/ ${formatTime(duration)}`}
                  </div>
                </div>

                <div className="w-full bg-muted rounded-full h-1.5 mb-8">
                  <div
                    className={`h-1.5 rounded-full transition-all duration-500 ${
                      isSettingUp ? "bg-warn/40 animate-pulse w-full" : "bg-ok"
                    }`}
                    style={isSettingUp ? undefined : { width: `${pct}%` }}
                  />
                </div>
              </>
            )}

            <div className="text-xs text-muted-foreground break-all mb-6">
              {t("inference.policyRef", {
                ref: status.policy_ref ?? t("inference.unknownPolicy"),
              })}
            </div>

            {showOutcome && (
              <div
                className={`mb-6 rounded-lg border p-4 ${
                  finishedWarn
                    ? "border-warn/40 bg-warn/10"
                    : "border-destructive/40 bg-destructive/10"
                }`}
              >
                <div
                  className={`flex items-center gap-2 text-sm font-semibold ${
                    finishedWarn ? "text-warn" : "text-destructive"
                  }`}
                >
                  <span
                    className={`w-2 h-2 rounded-full ${
                      finishedWarn ? "bg-warn" : "bg-destructive"
                    }`}
                  />
                  {finishedWarn
                    ? t("inference.outcome.ranWithWarning")
                    : t("inference.outcome.runFailed")}
                </div>
                {status.hint && (
                  <p
                    className={`mt-2 text-sm leading-relaxed ${
                      finishedWarn ? "text-warn/90" : "text-destructive/90"
                    }`}
                  >
                    {status.hint}
                  </p>
                )}
                {status.error && (
                  <pre className="mt-3 max-h-40 overflow-auto rounded bg-muted p-2 text-xs text-muted-foreground whitespace-pre-wrap break-words">
                    {status.error}
                  </pre>
                )}
              </div>
            )}

            {isFinished ? (
              <Button
                onClick={onExit}
                variant="secondary"
                className="w-full font-semibold py-6 text-lg"
              >
                {t("inference.button.close")}
              </Button>
            ) : isResetting ? (
              // Reset screen: continue is the primary action, abort stays
              // available alongside it.
              <div className="space-y-2">
                <Button
                  onClick={handleNextEpisode}
                  disabled={startingNext}
                  className="w-full font-semibold py-6 text-lg disabled:opacity-50"
                >
                  <Play className="w-5 h-5 mr-2" />
                  {startingNext
                    ? t("inference.button.starting")
                    : t("inference.button.startEpisode", {
                        index: Math.min(
                          results.length + 1,
                          episodesTotal ?? results.length + 1,
                        ),
                      })}
                </Button>
                <Button
                  onClick={handleStop}
                  disabled={stopping}
                  variant="outline"
                  className="w-full font-semibold disabled:opacity-50"
                >
                  <Square className="w-4 h-4 mr-2" />
                  {stopping
                    ? t("inference.button.aborting")
                    : t("inference.button.abortEvaluation")}
                </Button>
              </div>
            ) : evalMode ? (
              // Running an episode: calling it a success is the primary action,
              // and it is NOT the same button as aborting the whole run.
              <div className="space-y-2">
                <Button
                  onClick={handleEpisodeSuccess}
                  disabled={!isRunning || endingEpisode}
                  className="w-full font-semibold py-6 text-lg disabled:opacity-50"
                >
                  <CheckCircle2 className="w-5 h-5 mr-2" />
                  {endingEpisode
                    ? t("inference.button.endingEpisode")
                    : t("inference.button.taskSucceeded")}
                </Button>
                <Button
                  onClick={handleStop}
                  disabled={!status.inference_active || stopping}
                  variant="outline"
                  className="w-full font-semibold disabled:opacity-50"
                >
                  <Square className="w-4 h-4 mr-2" />
                  {stopping
                    ? t("inference.button.aborting")
                    : t("inference.button.abortEvaluation")}
                </Button>
              </div>
            ) : (
              <Button
                onClick={handleStop}
                disabled={!status.inference_active || stopping}
                variant="destructive"
                className="w-full font-semibold py-6 text-lg disabled:opacity-50"
              >
                <Square className="w-5 h-5 mr-2" />
                {stopping
                  ? t("inference.button.stopping")
                  : t("inference.button.stop")}
              </Button>
            )}

            {phaseMeta && (
              <div className="mt-6 flex items-center gap-2 text-sm">
                <span
                  className={`w-2 h-2 rounded-full ${PHASE_DOT[phaseMeta.tone]} ${
                    phaseMeta.pulse ? "animate-pulse" : ""
                  }`}
                />
                <span className={`font-medium ${PHASE_TEXT[phaseMeta.tone]}`}>
                  {t(phaseMeta.labelKey as never)}
                </span>
              </div>
            )}

            {isDownloading && (
              <div className="mt-3 space-y-1">
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                  {dlDeterminate ? (
                    <div
                      className="h-full rounded-full bg-warn transition-[width] duration-500"
                      style={{ width: `${dlPercent}%` }}
                    />
                  ) : (
                    <div className="h-full w-full animate-pulse rounded-full bg-warn/40" />
                  )}
                </div>
                <div className="text-[11px] tabular-nums text-muted-foreground">
                  {dlDeterminate
                    ? t("inference.download.progress", {
                        done: formatBytes(dlDone ?? 0),
                        total: formatBytes(dlTotal),
                      })
                    : dlDone != null
                      ? t("inference.download.soFar", {
                          done: formatBytes(dlDone),
                        })
                      : t("inference.download.starting")}
                </div>
              </div>
            )}

            <div className="mt-4">
              <LogPanel
                logs={logIsThisRun ? logs : logPlaceholder}
                title={t("inference.log.title")}
                defaultCollapsed
                wrap={false}
              />
            </div>
          </div>
        )}

      </DialogContent>
    </Dialog>
  );
};

export default InferenceSessionDialog;
