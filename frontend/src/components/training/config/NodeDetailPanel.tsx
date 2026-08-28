import React, { useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Loader2 } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { JobRecord, jobDisplayName } from "@/lib/jobsApi";
import { NodeEntry, getNodeJobs, getNodeQueue, nodeDisplayName } from "@/lib/nodesApi";
import { relativeTimeAgo } from "@/lib/relativeTime";

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
      queuedCount: number;
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

  const reachable = node != null && node.status === "ok";

  useEffect(() => {
    if (!reachable) return;
    let cancelled = false;
    setWorkload({ kind: "loading" });
    const fetchWorkload = () => {
      // Two reads: the jobs page for the running run, the queue endpoint for
      // an EXACT queued count — the jobs page is limited and can undercount.
      Promise.all([
        getNodeJobs(baseUrl, fetchWithHeaders, instanceId),
        getNodeQueue(baseUrl, fetchWithHeaders, instanceId),
      ])
        .then(([jobs, queue]) => {
          if (cancelled) return;
          setWorkload({
            kind: "ok",
            running: jobs.find((j) => j.state === "running") ?? null,
            queuedCount: queue.length,
          });
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
  }, [baseUrl, fetchWithHeaders, instanceId, reachable, refreshToken]);

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
              <span className="text-foreground">
                {runningPct != null
                  ? t("training.target.detail.workloadRunningPct", {
                      // The run's name is data; the percent is pre-formatted.
                      name: jobDisplayName(running),
                      pct: runningPct,
                    })
                  : t("training.target.detail.workloadRunning", {
                      name: jobDisplayName(running),
                    })}
              </span>
            ) : (
              t("training.target.detail.workloadIdle")
            )}
            {workload.queuedCount > 0 ? (
              <span>
                {" · "}
                {t("training.target.detail.workloadQueued", {
                  // Named `total`, not `count`: a plain figure, no plural pick.
                  total: workload.queuedCount,
                })}
              </span>
            ) : null}
          </span>
        )}
      </div>

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
    </div>
  );
};

export default NodeDetailPanel;
