import React, { useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Loader2, RotateCw } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { ApiError } from "@/lib/apiClient";
import { JobRecord, jobDisplayName } from "@/lib/jobsApi";
import {
  NodeEntry,
  getNodeJobs,
  getNodeQueue,
  nodeDisplayName,
  restartNode,
} from "@/lib/nodesApi";
import { relativeTimeAgo } from "@/lib/relativeTime";
import NodeJobDialog from "./NodeJobDialog";

interface NodeDetailPanelProps {
  /** The registry entry for the selected node — null when it has LEFT the
   * registry (the selection is kept and flagged, never silently dropped). */
  node: NodeEntry | null;
  /** The selected instance id, which outlives the entry. */
  instanceId: string;
  /** Bump to force an immediate workload refetch (the manual refresh button). */
  refreshToken?: number;
}

/** The selected node's workload, read through the server-to-server proxy.
 * `unreachable` covers both a refused proxy call and a network failure. */
type Workload =
  | { kind: "loading" }
  | { kind: "unreachable" }
  | {
      kind: "ok";
      running: JobRecord | null;
      /** The peer's EXACT queue records — already fetched for the count, and
       * kept whole so each entry can open the drill-in dialog. */
      queued: JobRecord[];
    };

const WORKLOAD_POLL_MS = 15_000;

/**
 * Contextual panel under the Compute selector when a LAN node is chosen: the
 * node's identity facts (instance id, version, last seen, URL — all data,
 * rendered verbatim), its CURRENT workload fetched on selection through
 * GET /api/v1/nodes/{id}/jobs (peers can queue now, so the queued count is
 * part of the answer), and the honest server-to-server sentence — the job runs
 * THERE, datasets sync via the Hub, the interface stays here.
 */
const NodeDetailPanel: React.FC<NodeDetailPanelProps> = ({
  node,
  instanceId,
  refreshToken,
}) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const [workload, setWorkload] = useState<Workload>({ kind: "loading" });
  // The run whose drill-in dialog is open, null when none. Kept as the full
  // record so the dialog opens instantly on what this panel already showed;
  // the dialog then polls its own fresher copy.
  const [openJob, setOpenJob] = useState<JobRecord | null>(null);
  // Bumped when the dialog stopped/deleted something, so the workload line
  // catches up without waiting out the 15s poll.
  const [changeToken, setChangeToken] = useState(0);
  // The remote-restart flow: an armed two-step button (idle → confirm), then
  // "requested" while the node bounces. The peer's refusal prose (its own 409
  // detail — server prose, rendered in English like all of it) lands in
  // `restartError`.
  const [restart, setRestart] = useState<"idle" | "confirm" | "requested">(
    "idle",
  );
  const [restartError, setRestartError] = useState<string | null>(null);

  const reachable = node != null && node.status === "ok";

  useEffect(() => {
    if (!reachable) return;
    let cancelled = false;
    setWorkload({ kind: "loading" });
    const fetchWorkload = () => {
      // Two reads: the jobs page for the running run, the queue endpoint for
      // the EXACT queue — the jobs page is limited and can undercount.
      Promise.all([
        getNodeJobs(baseUrl, fetchWithHeaders, instanceId),
        getNodeQueue(baseUrl, fetchWithHeaders, instanceId),
      ])
        .then(([jobs, queue]) => {
          if (cancelled) return;
          setWorkload({
            kind: "ok",
            running: jobs.find((j) => j.state === "running") ?? null,
            queued: queue,
          });
          // A successful read through the proxy means the node is answering
          // again — a pending restart has completed.
          setRestart((r) => (r === "requested" ? "idle" : r));
        })
        .catch(() => {
          // node.unreachable / node.not_found / network — one honest word.
          if (!cancelled) setWorkload({ kind: "unreachable" });
        });
    };
    fetchWorkload();
    const timer = setInterval(fetchWorkload, WORKLOAD_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [baseUrl, fetchWithHeaders, instanceId, reachable, refreshToken, changeToken]);

  // The armed confirm disarms itself: an unclicked "Confirm restart?" must
  // not linger as a landmine for a later stray click.
  useEffect(() => {
    if (restart !== "confirm") return;
    const timer = setTimeout(() => setRestart("idle"), 5_000);
    return () => clearTimeout(timer);
  }, [restart]);

  const handleRestart = async () => {
    if (restart === "idle") {
      setRestartError(null);
      setRestart("confirm");
      return;
    }
    setRestartError(null);
    setRestart("requested");
    try {
      await restartNode(baseUrl, fetchWithHeaders, instanceId);
    } catch (e) {
      // The peer's own refusal (409 robot.busy.* / system.restart_unsupported,
      // or a plain 404 from a peer without the endpoint) — its prose says why.
      setRestart("idle");
      setRestartError(
        e instanceof ApiError && e.detail
          ? e.detail
          : e instanceof Error
            ? e.message
            : String(e),
      );
    }
  };

  const name = node ? nodeDisplayName(node) : instanceId.slice(0, 8);

  if (node == null) {
    return (
      <div className="space-y-1.5 rounded-md border border-warn/40 bg-muted/30 p-3">
        <p className="text-xs text-warn">
          {t("training.target.detail.goneBody")}
        </p>
        <p className="font-mono text-[11px] text-muted-foreground">
          {instanceId}
        </p>
      </div>
    );
  }

  const running = workload.kind === "ok" ? workload.running : null;
  const runningPct =
    running && running.metrics.total_steps > 0
      ? Math.min(
          100,
          (running.metrics.current_step / running.metrics.total_steps) * 100,
        ).toFixed(0)
      : null;

  return (
    <div className="space-y-2 rounded-md border border-border bg-muted/30 p-3">
      {/* Identity facts — every value is data, rendered verbatim. */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        {node.instance_id ? (
          <span>
            {t("training.target.detail.instance")}{" "}
            <b
              className="font-mono font-medium text-foreground"
              title={node.instance_id}
            >
              {node.instance_id.slice(0, 8)}
            </b>
          </span>
        ) : null}
        {node.version ? (
          <span>
            {t("training.target.detail.version")}{" "}
            <b className="font-medium text-foreground">v{node.version}</b>
          </span>
        ) : null}
        {node.last_seen_at != null ? (
          <span>
            {t("training.target.detail.lastSeenLabel")}{" "}
            <b className="font-medium text-foreground">
              {/* Pre-formatted English relative time, like every {{when}}. */}
              {relativeTimeAgo(node.last_seen_at * 1000)}
            </b>
          </span>
        ) : null}
        {node.url ? (
          <span className="font-mono text-[11.5px] opacity-80">{node.url}</span>
        ) : null}
      </div>

      {/* Workload, live from the node (through THIS server's proxy). */}
      <div className="text-xs">
        {!reachable || workload.kind === "unreachable" ? (
          <span className="text-warn">
            {t("training.target.detail.workloadUnreachable")}
          </span>
        ) : workload.kind === "loading" ? (
          <span className="flex items-center gap-1.5 text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" />
            {t("training.target.detail.workloadLoading")}
          </span>
        ) : (
          <span className="text-muted-foreground">
            {running ? (
              // The whole running line opens the drill-in dialog for that
              // run — same words as before, now a button.
              <button
                type="button"
                onClick={() => setOpenJob(running)}
                title={t("training.target.nodeJob.openRunning")}
                className="text-left text-foreground underline decoration-dotted underline-offset-2 hover:text-info"
              >
                {runningPct != null
                  ? t("training.target.detail.workloadRunningPct", {
                      // The run's name is data; the percent is pre-formatted.
                      name: jobDisplayName(running),
                      pct: runningPct,
                    })
                  : t("training.target.detail.workloadRunning", {
                      name: jobDisplayName(running),
                    })}
              </button>
            ) : (
              t("training.target.detail.workloadIdle")
            )}
            {workload.queued.length > 0 ? (
              <span>
                {" · "}
                {t("training.target.detail.workloadQueued", {
                  // Named `total`, not `count`: a plain figure, no plural pick.
                  total: workload.queued.length,
                })}
              </span>
            ) : null}
          </span>
        )}
      </div>

      {/* The queued runs themselves — the records are already here for the
          count, so each gets a chip that opens the same drill-in dialog. */}
      {workload.kind === "ok" && workload.queued.length > 0 ? (
        <div className="flex flex-wrap gap-1">
          {workload.queued.map((q) => (
            <button
              key={q.id}
              type="button"
              onClick={() => setOpenJob(q)}
              title={t("training.target.nodeJob.openQueued")}
              className="rounded border border-border bg-background px-1.5 py-0.5 text-[11px] text-muted-foreground hover:border-ring/50 hover:text-foreground"
            >
              {/* Run number + name — both data, rendered verbatim. */}
              {q.job_number > 0 ? (
                <span className="mr-1 font-mono">#{q.job_number}</span>
              ) : null}
              {jobDisplayName(q)}
            </button>
          ))}
        </div>
      ) : null}

      <p className="text-xs text-muted-foreground">
        {/* One whole sentence; <0>/<1> wrap the node's name (data). */}
        <Trans
          i18nKey="training.target.detail.hubSyncHint"
          values={{ name }}
          components={[
            <b key="0" className="font-medium text-foreground" />,
            <b key="1" className="font-medium text-foreground" />,
          ]}
        />
      </p>

      {/* Remote restart: re-exec the node's server in place (e.g. after an
          environment change). Two-step arm/confirm; disabled while the node
          has work — the peer would refuse anyway, no point inviting the
          click. The peer's refusal prose renders verbatim. */}
      {reachable ? (
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={handleRestart}
            disabled={
              restart === "requested" ||
              running != null ||
              (workload.kind === "ok" && workload.queued.length > 0)
            }
            className="flex items-center gap-1 rounded border border-border bg-background px-1.5 py-0.5 text-[11px] text-muted-foreground hover:border-ring/50 hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
          >
            <RotateCw
              className={`h-3 w-3 ${restart === "requested" ? "animate-spin" : ""}`}
            />
            {restart === "confirm"
              ? t("training.target.detail.restartConfirm")
              : t("training.target.detail.restartAction")}
          </button>
          {restart === "requested" ? (
            <span className="text-[11px] text-muted-foreground">
              {t("training.target.detail.restartRequested", { name })}
            </span>
          ) : null}
          {restartError ? (
            <span className="text-[11px] text-warn">{restartError}</span>
          ) : null}
        </div>
      ) : null}

      {/* Drill-in dialog for the clicked run. Keyed by run id so switching
          runs remounts it with fresh state; it polls its own record + log
          tail through the same server-to-server proxies. */}
      {openJob ? (
        <NodeJobDialog
          key={openJob.id}
          open
          onOpenChange={(o) => {
            if (!o) setOpenJob(null);
          }}
          instanceId={instanceId}
          nodeName={name}
          job={openJob}
          onChanged={() => setChangeToken((v) => v + 1)}
        />
      ) : null}
    </div>
  );
};

export default NodeDetailPanel;
