import { Fetcher, apiRequest } from "./apiClient";
import { JobRecord, JobState, LogLine } from "./jobsApi";
import type { HostingPhase } from "./remoteApi";

/** One entry of GET /api/v1/nodes (NodeEntry in makermodslab/schemas/nodes.py).
 *
 * Nulls are meaningful: `url` is null only on the self entry; `instance_id`,
 * `version`, `capabilities` and both timestamps are null on a saved peer that
 * hasn't completed a handshake since the server started — and on a discovered
 * candidate the verify handshake hasn't confirmed yet, which additionally
 * carries status "pending". `last_seen_at` is wall-clock epoch seconds of the
 * last successful handshake (null until one happens), the field "last seen X
 * ago" is computed from; `last_verified_at` is the server's monotonic registry
 * clock and is NOT a wall-clock time. */
export interface NodeEntry {
  name: string | null;
  url: string | null;
  instance_id: string | null;
  version: string | null;
  // The PEER's self-reported capability block — a plain dict on the wire so a
  // version-skewed peer can't fail validation. The keys this UI reads are
  // typed; everything else rides along.
  capabilities: (Record<string, unknown> & {
    accepts_jobs?: boolean;
    serves_ui?: boolean;
    gpu?: unknown;
    /** Present only when the node runs the LiveKit SFU (--sfu). */
    sfu?: { url: string };
    /** Present only while the node has a live hosting session — the robot
     * it offers for remote teleoperation. `robot`, `arm_type` and the
     * operator identity are data. `phase` / `active_operator` are optional
     * only for a version-skewed peer that predates them. */
    hosting?: {
      robot: string;
      arm_type: string;
      phase?: HostingPhase;
      active_operator?: string | null;
    };
  }) | null;
  status: "ok" | "unreachable" | "pending";
  source: "manual" | "tailscale";
  is_self: boolean;
  last_seen_at: number | null;
  last_verified_at: number | null;
}

/** GET /api/v1/nodes: the entries plus `sources` — the discovery-source ids
 * the server has registered (["tailscale"] when it was started with
 * --discover-tailscale, [] otherwise), so the UI can tell "no peers found"
 * apart from "discovery is off". */
export interface NodeListing {
  nodes: NodeEntry[];
  sources: string[];
}

export async function listNodes(
  baseUrl: string,
  fetcher: Fetcher,
  options: { force?: boolean; signal?: AbortSignal } = {},
): Promise<NodeListing> {
  // ?force=true is the manual-refresh contract: the server bypasses its TTL
  // for this one pass — discovery runs now, every entry is probed now. The
  // background poll stays un-forced so it rides the server's own cadence.
  return apiRequest<NodeListing>(
    baseUrl,
    fetcher,
    `/api/v1/nodes${options.force ? "?force=true" : ""}`,
    { signal: options.signal, action: "List nodes" },
  );
}

/** Verify-and-register a peer by URL. Coded refusals: 409 node.self /
 * node.duplicate, 502 node.unreachable, 422 request.validation — all thrown as
 * ApiError with `code` set; branch on the code, never the prose. */
export async function addNode(
  baseUrl: string,
  fetcher: Fetcher,
  url: string,
  name?: string,
): Promise<NodeEntry> {
  return apiRequest<NodeEntry>(baseUrl, fetcher, "/api/v1/nodes", {
    method: "POST",
    body: name ? { url, name } : { url },
    action: "Add node",
  });
}

/** The peer's own jobs listing, proxied server-to-server. 404 node.not_found
 * for an unknown instance id; 502 node.unreachable when the peer doesn't
 * answer. The body is the peer's GET /api/v1/jobs response verbatim. */
export async function getNodeJobs(
  baseUrl: string,
  fetcher: Fetcher,
  instanceId: string,
  signal?: AbortSignal,
): Promise<JobRecord[]> {
  const body = await apiRequest<{ jobs: JobRecord[] }>(
    baseUrl,
    fetcher,
    `/api/v1/nodes/${encodeURIComponent(instanceId)}/jobs`,
    { signal, action: "Get node jobs" },
  );
  return body.jobs;
}

/**
 * The peer's EXACT queue (its /api/v1/jobs/queue) — the jobs listing above is
 * a limited page and can undercount queued runs on a busy peer.
 */
export async function getNodeQueue(
  baseUrl: string,
  fetcher: Fetcher,
  instanceId: string,
  signal?: AbortSignal,
): Promise<JobRecord[]> {
  const body = await apiRequest<{ jobs: JobRecord[] }>(
    baseUrl,
    fetcher,
    `/api/v1/nodes/${encodeURIComponent(instanceId)}/jobs/queue`,
    { signal, action: "Get node queue" },
  );
  return body.jobs;
}

/** One run on the peer, through the drill-in proxy (the peer's own
 * GET /api/v1/jobs/{job_id}, verbatim). 404 node.not_found for an unknown
 * node; 502 node.unreachable for ANY failure to read the peer — its own 404
 * for an unknown job included. */
export async function getNodeJob(
  baseUrl: string,
  fetcher: Fetcher,
  instanceId: string,
  jobId: string,
  signal?: AbortSignal,
): Promise<JobRecord> {
  return apiRequest<JobRecord>(
    baseUrl,
    fetcher,
    `/api/v1/nodes/${encodeURIComponent(instanceId)}/jobs/${encodeURIComponent(jobId)}`,
    { signal, action: "Get node job" },
  );
}

/** The peer run's live log tail. The peer drains its runner's queue per call,
 * so this is inherently incremental — each call returns only the lines that
 * arrived since the last one. Append, never replace. */
export async function getNodeJobLogs(
  baseUrl: string,
  fetcher: Fetcher,
  instanceId: string,
  jobId: string,
  signal?: AbortSignal,
): Promise<LogLine[]> {
  const body = await apiRequest<{ logs: LogLine[] }>(
    baseUrl,
    fetcher,
    `/api/v1/nodes/${encodeURIComponent(instanceId)}/jobs/${encodeURIComponent(jobId)}/logs`,
    { signal, action: "Get node job logs" },
  );
  return body.logs;
}

/** Stop/cancel a run on the peer, forwarded server-to-server. `expectState`
 * is the same optimistic-concurrency precondition as the local stop: pass the
 * state the UI drew the button against. The peer's own coded refusals (409
 * job.state_changed / job.has_queued_dependents, 404 job.not_found) come back
 * with THEIR status and code — branch on `code`, never the prose; only
 * transport failure reads 502 node.unreachable. */
export async function stopNodeJob(
  baseUrl: string,
  fetcher: Fetcher,
  instanceId: string,
  jobId: string,
  expectState?: JobState,
): Promise<JobRecord> {
  const query = expectState
    ? `?expect_state=${encodeURIComponent(expectState)}`
    : "";
  return apiRequest<JobRecord>(
    baseUrl,
    fetcher,
    `/api/v1/nodes/${encodeURIComponent(instanceId)}/jobs/${encodeURIComponent(jobId)}/stop${query}`,
    { method: "POST", action: "Stop node job" },
  );
}

/** Delete a terminal run on the peer (its output directory there included).
 * Same passthrough stance as the stop: the peer's coded refusals keep their
 * status and code. */
export async function deleteNodeJob(
  baseUrl: string,
  fetcher: Fetcher,
  instanceId: string,
  jobId: string,
): Promise<void> {
  await apiRequest<void>(
    baseUrl,
    fetcher,
    `/api/v1/nodes/${encodeURIComponent(instanceId)}/jobs/${encodeURIComponent(jobId)}`,
    { method: "DELETE", action: "Delete node job" },
  );
}

/** The PEER's answer to "can your environment import what this policy
 * needs?" — its own GET /api/v1/system/policy-extra/{policy_type}, proxied
 * server-to-server. The offloaded run imports from the peer's site-packages,
 * so the local answer is irrelevant to it. */
export interface NodePolicyExtraStatus {
  policy_type: string;
  needs_extra: boolean;
  available: boolean;
  package: string;
  install_target: string;
  install_hint: string;
}

export async function getNodePolicyExtra(
  baseUrl: string,
  fetcher: Fetcher,
  instanceId: string,
  policyType: string,
  signal?: AbortSignal,
): Promise<NodePolicyExtraStatus> {
  return apiRequest<NodePolicyExtraStatus>(
    baseUrl,
    fetcher,
    `/api/v1/nodes/${encodeURIComponent(instanceId)}/policy-extra/${encodeURIComponent(policyType)}`,
    { signal, action: "Get node policy extra" },
  );
}

/** Ask the peer to restart its server process (re-exec in place), so a
 * just-installed extra environment or config change lands without a shell on
 * the node. 200 means the restart is SCHEDULED — expect the node to read
 * unreachable for a few seconds before the registry's probes pick it back up.
 * Coded refusals pass through from the peer (409 robot.busy.* /
 * session.held / system.restart_unsupported); a peer too old to have the
 * endpoint answers a plain 404. */
export async function restartNode(
  baseUrl: string,
  fetcher: Fetcher,
  instanceId: string,
): Promise<{ restarting: boolean; message: string }> {
  return apiRequest<{ restarting: boolean; message: string }>(
    baseUrl,
    fetcher,
    `/api/v1/nodes/${encodeURIComponent(instanceId)}/restart`,
    { method: "POST", action: "Restart node" },
  );
}

/** A node the Compute selector can offer a training to: reachable, verified
 * (it has an identity to route by), and advertising that it accepts jobs. */
export function isSelectableNode(node: NodeEntry): boolean {
  return (
    node.status === "ok" &&
    node.instance_id != null &&
    node.capabilities?.accepts_jobs === true
  );
}

/** The nodes the selector LISTS: peers only (never the self entry), and never
 * one that positively declares it does not accept jobs. An unreachable or
 * still-verifying peer has a null capability block — it stays visible
 * (disabled) rather than vanishing, so a node that drops off the network
 * reads as "unreachable", not as deleted. */
export function listableNodes(nodes: NodeEntry[]): NodeEntry[] {
  return nodes.filter(
    (n) => !n.is_self && n.capabilities?.accepts_jobs !== false,
  );
}

/** The stations a remote-teleop operator can pick: peers only (never the
 * self entry), reachable, verified (an instance id to route by), and
 * currently advertising a hosted robot. A station that stops hosting simply
 * drops out of this list — the capability is present only while its hosting
 * session is live. */
export function hostingNodes(nodes: NodeEntry[]): NodeEntry[] {
  return nodes.filter(
    (n) =>
      !n.is_self &&
      n.status === "ok" &&
      n.instance_id != null &&
      n.capabilities?.hosting != null,
  );
}

/** Display name for a node row: its name, else its host, else a short
 * instance id. All three are data — rendered verbatim, never translated. */
export function nodeDisplayName(node: NodeEntry): string {
  if (node.name) return node.name;
  if (node.url) {
    try {
      return new URL(node.url).host;
    } catch {
      return node.url;
    }
  }
  return node.instance_id ? node.instance_id.slice(0, 8) : "?";
}

/** The GPU chip's text, when the peer's capability block carries one.
 * `capabilities.gpu` is additive and self-reported: accept a plain string
 * verbatim, or an object with name/vram fields joined "name · vram". Data,
 * never translated; null ⇒ no chip. */
export function nodeGpuLabel(node: NodeEntry): string | null {
  const gpu = node.capabilities?.gpu;
  if (typeof gpu === "string" && gpu.trim()) return gpu;
  if (gpu && typeof gpu === "object") {
    const rec = gpu as Record<string, unknown>;
    const parts = [rec.name, rec.vram].filter(
      (v): v is string => typeof v === "string" && v.trim() !== "",
    );
    if (parts.length > 0) return parts.join(" · ");
  }
  return null;
}
