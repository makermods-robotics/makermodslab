import React, { useEffect, useRef, useState } from "react";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";

import { TrainingStatus, LogEntry } from "@/components/training/types";
import MonitoringStats from "@/components/training/monitoring/MonitoringStats";
import TrainingLogs from "@/components/training/monitoring/TrainingLogs";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { Loader2, Play, Square, Trash2, ArrowLeft } from "lucide-react";

import {
  JobRecord,
  getJob,
  getJobLogs,
  getJobLogFile,
  jobDisplayName,
  jobStateLabel,
  stopJob,
  deleteJob,
} from "@/lib/jobsApi";
import { JobCheckpoint, listJobCheckpoints } from "@/lib/checkpointsApi";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import { useStudio } from "@/contexts/StudioContext";

const POLL_INTERVAL_MS = 1000;
const MAX_LOG_LINES = 5000;

/**
 * Log lines the pre-step status strip must never speak for.
 *
 * `objc[<pid>]: Class … is implemented in both …` is the macOS Objective-C
 * runtime's duplicate-class warning, written to stderr by the dylibs pyav
 * bundles (libavdevice/libavformat) the moment the trainer imports them — that
 * is, squarely inside the `current_step === 0` window this strip exists to
 * narrate. It is harmless, it says nothing about progress, and at two absolute
 * dylib paths it is among the longest lines the run ever emits, which is what
 * let it crowd the header.
 *
 * Deliberately anchored and narrow: the runtime's own `objc[1234]:` prefix and
 * nothing else. Matching the body instead ("implemented in both", "duplicate")
 * would reach wider than the thing being suppressed and could swallow a real
 * failure that happens to use those words.
 *
 * This is DISPLAY SELECTION, not log rewriting — the line is still in the raw
 * log pane below, which is where a diagnostic like this belongs.
 */
const STATUS_NOISE_RE = /^objc\[\d+\]:/;

/**
 * The newest log line worth showing as status, or null when there is none.
 *
 * Walks backwards to the first line that is neither blank nor noise, so a
 * burst of objc warnings reveals the last real message underneath them rather
 * than blanking the strip. Blank lines are skipped for the same reason they
 * would be useless as status text.
 */
function newestStatusLine(logs: LogEntry[]): string | null {
  for (let i = logs.length - 1; i >= 0; i--) {
    const message = logs[i]?.message ?? "";
    if (message.trim() && !STATUS_NOISE_RE.test(message)) return message;
  }
  return null;
}

function jobToStatus(
  job: JobRecord | null,
  isStarting: boolean,
): TrainingStatus {
  // Adapter so MonitoringStats can keep its current prop shape.
  if (!job) {
    return {
      training_active: isStarting,
      current_step: 0,
      total_steps: 0,
      available_controls: {
        stop_training: false,
        pause_training: false,
        resume_training: false,
      },
    };
  }
  return {
    training_active: job.state === "running",
    current_step: job.metrics.current_step,
    total_steps: job.metrics.total_steps,
    current_loss: job.metrics.current_loss ?? undefined,
    current_lr: job.metrics.current_lr ?? undefined,
    grad_norm: job.metrics.grad_norm ?? undefined,
    eta_seconds: job.metrics.eta_seconds ?? undefined,
    available_controls: {
      stop_training: job.state === "running",
      pause_training: false,
      resume_training: false,
    },
  };
}

/**
 * The training-job monitor as a modal dialog over the studio's Train panel —
 * replaces the old /training/:jobId page (the polling/monitor logic is ported
 * verbatim; every navigate-home became `onExit`). Unlike the recording
 * session, dismissing is always safe — training keeps running in the
 * background — so ESC / outside click / the back button all exit to the
 * studio.
 */
const TrainingJobDialog: React.FC<{
  jobId: string;
  /** Called for every exit — closes the dialog, landing back in the studio. */
  onExit: () => void;
}> = ({ jobId, onExit }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

  const { openStudio } = useStudio();
  const [job, setJob] = useState<JobRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  // Whether selectedStep was chosen BY THE USER (dropdown) rather than
  // auto-seeded. An auto-seeded selection follows the newest checkpoint as
  // saves land during training; a user's explicit pick stays put (until the
  // checkpoint it names disappears, e.g. a deleted run).
  const userPickedStepRef = useRef(false);

  // Seed logs from the persistent on-disk file once on mount, so closing and
  // reopening the dialog (or coming in fresh on a finished/interrupted job)
  // shows the full log history. Polling /logs continues from this point — the
  // backend drains the live queue in the same /log-file call so we don't
  // double-display lines that were buffered when we landed.
  useEffect(() => {
    let cancelled = false;
    getJobLogFile(baseUrl, fetchWithHeaders, jobId)
      .then((seeded) => {
        if (!cancelled && seeded.length > 0) setLogs(seeded);
      })
      .catch(() => {
        // 404 or transient — fall through; live polling will fill in.
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders, jobId]);

  // Read latest job state from a ref so the polling intervals below stay
  // stable instead of tearing down/rebuilding on every state transition.
  const jobStateRef = useRef(job?.state);
  jobStateRef.current = job?.state;

  // Poll checkpoints — every 5s while the job is running.
  useEffect(() => {
    let cancelled = false;
    const tick = () => {
      listJobCheckpoints(baseUrl, fetchWithHeaders, jobId)
        .then((cks) => {
          if (cancelled) return;
          setCheckpoints(cks);
          if (cks.length > 0) {
            // Backend list is step-ascending, so the last entry is newest.
            const latest = cks[cks.length - 1].step;
            setSelectedStep((prev) => {
              const keepUserPick =
                userPickedStepRef.current &&
                prev != null &&
                cks.some((c) => c.step === prev);
              if (!keepUserPick) userPickedStepRef.current = false;
              return keepUserPick ? prev : latest;
            });
          } else {
            setSelectedStep(null);
            userPickedStepRef.current = false;
          }
        })
        .catch(() => {
          if (!cancelled) {
            setCheckpoints([]);
            setSelectedStep(null);
          }
        });
    };
    tick();
    const id = setInterval(() => {
      if (cancelled) return;
      if (jobStateRef.current && jobStateRef.current !== "running") return;
      tick();
    }, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [baseUrl, fetchWithHeaders, jobId]);

  // Poll the job + its logs while running. Caps log lines to avoid unbounded growth.
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const next = await getJob(baseUrl, fetchWithHeaders, jobId);
        if (cancelled) return;
        setJob(next);
        if (next.state === "running") {
          const newLogs = await getJobLogs(baseUrl, fetchWithHeaders, jobId);
          if (!cancelled && newLogs.length > 0) {
            setLogs((prev) => {
              const merged = [...prev, ...newLogs];
              return merged.length > MAX_LOG_LINES
                ? merged.slice(merged.length - MAX_LOG_LINES)
                : merged;
            });
          }
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    };
    tick();
    const id = setInterval(() => {
      if (cancelled) return;
      if (jobStateRef.current && jobStateRef.current !== "running") return;
      tick();
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [baseUrl, fetchWithHeaders, jobId]);

  // Auto-scroll the log panel as new lines arrive.
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, "0")}:${minutes
      .toString()
      .padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const getProgressPercentage = () => {
    if (!job || job.metrics.total_steps === 0) return 0;
    return (job.metrics.current_step / job.metrics.total_steps) * 100;
  };

  const handleStop = async () => {
    if (!job) return;
    if (!window.confirm("Stop this run?")) return;
    try {
      const next = await stopJob(baseUrl, fetchWithHeaders, job.id);
      setJob(next);
      toast({ title: "Stopping…" });
    } catch (e) {
      toast({
        title: "Stop failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handleDelete = async () => {
    if (!job) return;
    if (!window.confirm("Delete this run? This wipes the output directory."))
      return;
    try {
      await deleteJob(baseUrl, fetchWithHeaders, job.id);
      toast({ title: "Job removed" });
      onExit();
    } catch (e) {
      toast({
        title: "Delete failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const isRunning = job?.state === "running";

  // Before the trainer's first metric tick the stats panel can only say
  // "Training starting…" — which is wrong (and worryingly quiet) for the
  // minutes a local fine-tune spends downloading its base checkpoint. The
  // backend writes that progress into the job's log, so surface the newest line
  // as a status strip until real steps arrive; it then gets out of the way.
  // Newest USEFUL line — see newestStatusLine / STATUS_NOISE_RE: the trainer's
  // start-up stderr is exactly what a blind tail would pick up here.
  const preStepStatus =
    isRunning && job.metrics.current_step === 0
      ? newestStatusLine(logs)
      : null;

  const backButton = (
    <Button
      variant="ghost"
      size="sm"
      onClick={onExit}
      className="-ml-2 shrink-0 text-muted-foreground hover:text-foreground"
    >
      <ArrowLeft className="mr-1.5 h-4 w-4" /> Skill studio
    </Button>
  );

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onExit();
      }}
    >
      <DialogContent
        hideClose
        className="max-h-[92vh] w-[min(96vw,72rem)] max-w-none gap-0 overflow-y-auto p-6"
        aria-describedby={undefined}
      >
        <DialogTitle className="sr-only">Training job status</DialogTitle>

        {error && !job ? (
          <div className="space-y-4">
            {backButton}
            <p className="text-sm text-destructive">
              Couldn't load job {jobId}: {error}
            </p>
          </div>
        ) : !job ? (
          <div className="space-y-4">
            {backButton}
            <div className="flex items-center justify-center py-20 text-muted-foreground">
              <Loader2 className="mr-3 h-6 w-6 animate-spin" /> Loading job…
            </div>
          </div>
        ) : (
          <div className="space-y-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="flex min-w-0 items-start gap-2">
                {backButton}
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <h2 className="truncate text-base">
                      {jobDisplayName(job)}
                    </h2>
                    {job.runner === "hf_cloud" ? (
                      <span className="rounded border border-border bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                        HF · {job.hf_flavor ?? "cloud"}
                      </span>
                    ) : (
                      <span className="rounded border border-border bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                        Local
                      </span>
                    )}
                    {job.runner === "hf_cloud" &&
                      job.hf_repo_id &&
                      job.state === "done" && (
                        <a
                          href={`https://huggingface.co/${job.hf_repo_id}`}
                          target="_blank"
                          rel="noreferrer"
                          className="text-xs text-primary hover:underline"
                        >
                          View on Hub ↗
                        </a>
                      )}
                  </div>
                  {/* When aliased, keep the immutable run id visible as subtext. */}
                  {job.display_name ? (
                    <p className="font-mono text-[11px] text-muted-foreground">
                      {job.id}
                    </p>
                  ) : null}
                  {/* The label, not the wire value: this line rendered the raw
                      state, so a stopped run read "interrupted" here while the
                      card behind the dialog called it something else. */}
                  <p className="text-xs text-muted-foreground">
                    {jobStateLabel(job.state)}
                    {job.error_message ? ` — ${job.error_message}` : ""}
                  </p>
                </div>
              </div>
              {isRunning ? (
                <Button onClick={handleStop} variant="destructive" size="sm">
                  <Square className="mr-2 h-4 w-4" /> Stop
                </Button>
              ) : (
                <Button
                  onClick={handleDelete}
                  variant="ghost"
                  size="sm"
                  className="text-muted-foreground hover:text-foreground"
                >
                  <Trash2 className="mr-2 h-4 w-4" /> Delete
                </Button>
              )}
            </div>

            {/* One line, always — whatever the trainer prints. `truncate` on
                its own is inert here: a flex item's `min-width` defaults to
                `auto`, which for nowrap text resolves to the FULL text width,
                so the span could never shrink and a long line escaped the box
                instead of ellipsing inside it. `min-w-0 flex-1` is what makes
                the clamp real, and it holds for any future long line, not just
                objc's. `title` keeps the whole line one hover away. */}
            {preStepStatus ? (
              <div className="flex items-center gap-2 rounded-md border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
                <span
                  className="min-w-0 flex-1 truncate"
                  title={preStepStatus}
                >
                  {preStepStatus}
                </span>
              </div>
            ) : null}

            <MonitoringStats
              jobId={jobId}
              trainingStatus={jobToStatus(job, false)}
              getProgressPercentage={getProgressPercentage}
              formatTime={formatTime}
            />
            <div className="flex items-center gap-3 rounded-md border border-border bg-card p-4">
              <span className="eyebrow">Run inference</span>
              {checkpoints.length === 0 ? (
                <span className="text-xs text-muted-foreground">
                  No checkpoints yet — wait for the first save.
                </span>
              ) : (
                <>
                  <CheckpointDropdown
                    checkpoints={checkpoints}
                    // Single-job list: steps are unique here, so the step
                    // maps 1:1 onto the checkpoint's identifying ref.
                    selectedRef={
                      checkpoints.find((c) => c.step === selectedStep)?.ref ??
                      null
                    }
                    onChange={(c) => {
                      userPickedStepRef.current = true;
                      setSelectedStep(c.step);
                    }}
                  />
                  <Button
                    onClick={() => {
                      // Land on the Deploy panel with this job + checkpoint
                      // prefilled (DeployPanel consumes the prefill) instead
                      // of stacking the legacy InferenceModal over the dialog.
                      openStudio("deploy", {
                        deploy: {
                          source: "job",
                          id: jobId,
                          step: selectedStep ?? undefined,
                        },
                      });
                      onExit();
                    }}
                    disabled={selectedStep == null}
                    size="sm"
                  >
                    <Play className="mr-2 h-4 w-4" />
                    Run on robot
                  </Button>
                </>
              )}
            </div>
            <TrainingLogs logs={logs} logContainerRef={logContainerRef} />
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default TrainingJobDialog;
