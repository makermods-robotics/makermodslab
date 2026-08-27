import { Fetcher, apiRequest } from "./apiClient";
import { JobRecord } from "./jobsApi";

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
  }) | null;
  status: "ok" | "unreachable" | "pending";
  source: "manual" | "tailscale";
  is_self: boolean;
  last_seen_at: number | null;
  last_verified_at: number | null;
}

export async function listNodes(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<NodeEntry[]> {
  const body = await apiRequest<{ nodes: NodeEntry[] }>(
    baseUrl,
    fetcher,
    "/api/v1/nodes",
    { signal, action: "List nodes" },
  );
  return body.nodes;
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
