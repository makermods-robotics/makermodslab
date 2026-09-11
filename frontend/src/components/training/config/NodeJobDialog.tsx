import React, { useEffect, useRef, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Loader2, Square, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { ApiError } from "@/lib/apiClient";
import {
  JobRecord,
  LogLine,
  isTerminalJobState,
  jobDisplayName,
  jobStateLabel,
} from "@/lib/jobsApi";
import {
  deleteNodeJob,
  getNodeJob,
  getNodeJobLogs,
  stopNodeJob,
} from "@/lib/nodesApi";

/** The logs proxy is incremental per call (the peer drains its live queue),
 * so a short interval costs little — each poll carries only new lines. */
const POLL_MS = 2000;
const MAX_LOG_LINES = 1000;

interface NodeJobDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The peer's routing key. */
  instanceId: string;
  /** The node's display name — data, rendered verbatim. */
  nodeName: string;
  /** The record the click was drawn against; the poll replaces it. Mount with
   * `key={job.id}` so a different run gets a fresh dialog. */
  job: JobRecord;
  /** Called after a successful stop/delete so the caller can refetch. */
  onChanged?: () => void;
}

/**
 * Minimal drill-in dialog for ONE run on a peer node, deliberately far
 * lighter than the local TrainingJobDialog: this is a PEER's job — the
 * record, an appending live log tail, Stop while it works, Delete once it is
 * terminal, all through this server's /nodes/{id}/jobs/{job_id} proxies (the
 * browser never talks to the peer). The header names the node so there is no
 * mistaking it for a local run.
 */
const NodeJobDialog: React.FC<NodeJobDialogProps> = ({
  open,
  onOpenChange,
  instanceId,
  nodeName,
  job,
  onChanged,
}) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [record, setRecord] = useState<JobRecord>(job);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [unreachable, setUnreachable] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const logEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const poll = () => {
      // Two proxy reads per tick: the record for state/progress, the log
      // tail for new lines. The logs endpoint is incremental — APPEND, never
      // replace, or lines drained by this very poll would be lost.
      Promise.all([
        getNodeJob(baseUrl, fetchWithHeaders, instanceId, job.id),
        getNodeJobLogs(baseUrl, fetchWithHeaders, instanceId, job.id),
      ])
        .then(([rec, lines]) => {
          if (cancelled) return;
          setRecord(rec);
          setUnreachable(false);
          if (lines.length > 0) {
            setLogs((prev) => [...prev, ...lines].slice(-MAX_LOG_LINES));
          }
        })
        .catch(() => {
          // node.unreachable / node.not_found / network — one honest word;
          // the last known record stays on screen.
          if (!cancelled) setUnreachable(true);
        });
    };
    poll();
    const timer = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [open, baseUrl, fetchWithHeaders, instanceId, job.id]);

  // Pin the tail to the newest line — a log viewer that doesn't follow is a
  // log viewer that gets scrolled by hand every two seconds.
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ block: "end" });
  }, [logs]);

  const stoppable = record.state === "running" || record.state === "queued";
  const deletable = isTerminalJobState(record.state);

  /** The peer's coded refusals worth our own words; everything else shows the
   * server's prose as it was written. */
  const refusalText = (e: unknown): string => {
    const code = e instanceof ApiError ? e.code : null;
    if (code === "job.state_changed")
      return t("jobs.jobsData.cancelStateChanged");
    if (code === "job.has_queued_dependents")
      return t("jobs.jobsData.cancelBlockedDependents");
    if (code === "node.unreachable")
      return t("training.target.nodeJob.unreachable");
    return e instanceof Error ? e.message : String(e);
  };

  const handleStop = async () => {
    // English on purpose: a native confirm's OK/Cancel come from the
    // BROWSER's locale (localization.md §5.10).
    if (!window.confirm("Stop this run?")) return;
    setStopping(true);
    try {
      // The precondition is the state this dialog is showing — the peer
      // answers 409 job.state_changed instead of killing a run that moved.
      const rec = await stopNodeJob(
        baseUrl,
        fetchWithHeaders,
        instanceId,
        job.id,
        record.state,
      );
      setRecord(rec);
      toast({ title: t("jobs.jobsData.stopping") });
      onChanged?.();
    } catch (e) {
      toast({
        title: t("jobs.jobsData.stopFailed"),
        description: refusalText(e),
        variant: "destructive",
      });
    } finally {
      setStopping(false);
    }
  };

  const handleDelete = async () => {
    // English on purpose — see the stop confirm above.
    if (
      !window.confirm(
        "Delete this run? This wipes its output directory on the node.",
      )
    )
      return;
    setDeleting(true);
    try {
      await deleteNodeJob(baseUrl, fetchWithHeaders, instanceId, job.id);
      toast({ title: t("jobs.jobsData.removed") });
      onChanged?.();
      onOpenChange(false);
    } catch (e) {
      toast({
        title: t("jobs.jobsData.deleteFailed"),
        description: refusalText(e),
        variant: "destructive",
      });
    } finally {
      setDeleting(false);
    }
  };

  const total = record.config?.steps || record.metrics.total_steps || 0;
  const current = record.metrics.current_step;
  const pct = total > 0 ? Math.min(100, (current / total) * 100) : 0;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <div className="space-y-1">
          {/* The run's name and number are data, rendered verbatim. */}
          <DialogTitle className="flex items-baseline gap-2 pr-6">
            {record.job_number > 0 ? (
              <span className="shrink-0 font-mono text-sm text-muted-foreground">
                #{record.job_number}
              </span>
            ) : null}
            <span className="truncate" title={jobDisplayName(record)}>
              {jobDisplayName(record)}
            </span>
          </DialogTitle>
          <p className="text-xs text-muted-foreground">
            {/* One whole sentence; <0> wraps the node's name (data). */}
            <Trans
              i18nKey="training.target.nodeJob.onNode"
              values={{ name: nodeName }}
              components={[
                <b key="0" className="font-medium text-foreground" />,
              ]}
            />
          </p>
        </div>

        {/* State + progress, straight off the polled record. */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-2 text-sm">
            <span className="font-medium">{jobStateLabel(record.state)}</span>
            {record.state === "running" && total > 0 ? (
              <span className="tabular-nums text-xs text-muted-foreground">
                {/* Numbers stay exactly as formatted — data, not copy. */}
                {`${current.toLocaleString()} / ${total.toLocaleString()} · ${pct.toFixed(1)}%`}
              </span>
            ) : null}
            {unreachable ? (
              <span className="text-xs text-warn">
                {t("training.target.nodeJob.unreachable")}
              </span>
            ) : null}
          </div>
          {record.state === "running" ? (
            <div className="h-1 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full bg-info transition-[width] duration-500"
                style={{ width: `${pct}%` }}
              />
            </div>
          ) : null}
        </div>

        {/* The live tail. Starts empty on open (the endpoint hands out only
            what arrived since the previous drain, wherever that happened). */}
        <div className="space-y-1">
          <p className="text-xs font-medium text-muted-foreground">
            {t("training.target.nodeJob.logsLabel")}
          </p>
          <div className="h-40 overflow-y-auto rounded-md border border-border bg-muted/30 p-2 font-mono text-[11px] leading-relaxed text-muted-foreground">
            {logs.length === 0 ? (
              <p className="text-muted-foreground/70">
                {t("training.target.nodeJob.logsEmpty")}
              </p>
            ) : (
              logs.map((line, i) => (
                <div key={i} className="whitespace-pre-wrap break-all">
                  {line.message}
                </div>
              ))
            )}
            <div ref={logEndRef} />
          </div>
        </div>

        <div className="flex justify-end gap-2">
          {stoppable ? (
            <Button
              variant="outline"
              size="sm"
              disabled={stopping}
              onClick={handleStop}
            >
              {stopping ? (
                <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
              ) : (
                <Square className="mr-1.5 h-3.5 w-3.5" />
              )}
              {stopping
                ? t("training.target.nodeJob.stopping")
                : t("training.target.nodeJob.stop")}
            </Button>
          ) : null}
          {deletable ? (
            <Button
              variant="destructive"
              size="sm"
              disabled={deleting}
              onClick={handleDelete}
            >
              {deleting ? (
                <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
              ) : (
                <Trash2 className="mr-1.5 h-3.5 w-3.5" />
              )}
              {deleting
                ? t("training.target.nodeJob.deleting")
                : t("training.target.nodeJob.delete")}
            </Button>
          ) : null}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default NodeJobDialog;
