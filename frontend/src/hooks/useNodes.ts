import { useCallback, useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { NodeEntry, listNodes } from "@/lib/nodesApi";

// Registry reads are cheap (the server answers from its table; probing happens
// on ITS schedule), so a modest poll keeps status dots honest without a
// dedicated WS channel. Matches the focus-refetch pattern JobsDataContext uses.
const POLL_MS = 30_000;

/**
 * The node registry, polled: GET /api/v1/nodes on mount, on window focus, and
 * every 30s while mounted. `nodes` is the raw listing (self entry included —
 * callers filter with `listableNodes`); `sources` the registered discovery
 * sources; `refresh` refetches (e.g. after adding a node). `forceRefresh` is
 * for the MANUAL refresh gesture only: it sends ?force=true, so the server
 * bypasses its TTL and probes everything now — the poll and the focus refetch
 * stay un-forced and ride the server's own cadence.
 *
 * A failed poll KEEPS the last listing rather than blanking it: the selector
 * degrades to stale-but-visible rows, the same honesty rule the unreachable
 * rows follow. `error` reports the failure alongside.
 */
export function useNodes() {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [nodes, setNodes] = useState<NodeEntry[]>([]);
  const [sources, setSources] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchNodes = useCallback(
    async (force: boolean) => {
      try {
        const listing = await listNodes(baseUrl, fetchWithHeaders, { force });
        setNodes(listing.nodes);
        setSources(listing.sources);
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    },
    [baseUrl, fetchWithHeaders],
  );

  // Bare callbacks (no forwarded args) so event listeners can't smuggle an
  // event object into the `force` parameter.
  const refresh = useCallback(() => fetchNodes(false), [fetchNodes]);
  const forceRefresh = useCallback(() => fetchNodes(true), [fetchNodes]);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, POLL_MS);
    const onVisible = () => {
      if (document.visibilityState === "visible") refresh();
    };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", refresh);
    return () => {
      clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("focus", refresh);
    };
  }, [refresh]);

  return { nodes, sources, loading, error, refresh, forceRefresh };
}
