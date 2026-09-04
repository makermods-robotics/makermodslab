import { useCallback, useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { useSessionEvent } from "@/hooks/useActiveSession";
import { getHostingStatus, type HostingStatus } from "@/lib/remoteApi";

/**
 * GET /api/v1/hosting, kept fresh two ways: a light poll (`intervalMs`,
 * 10s by default — the robot corner's chip) and a refetch on every
 * `session_changed` hint for kind `hosting` (claim, phase change, release),
 * which is what makes the chip flip parked ↔ engaged within a broadcast
 * rather than a poll. The hint is never treated as state — it only triggers
 * the fetch (CLAUDE.md "WebSocket broadcast").
 *
 * A failed read KEEPS the last status rather than blanking it, the same
 * stale-but-visible rule useNodes follows. `status` is null until the first
 * read lands.
 */
export function useHostingStatus(
  options: { enabled?: boolean; intervalMs?: number } = {},
) {
  const { enabled = true, intervalMs = 10_000 } = options;
  const { baseUrl, fetchWithHeaders } = useApi();
  const sessionEvent = useSessionEvent();
  const [status, setStatus] = useState<HostingStatus | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const next = await getHostingStatus(baseUrl, fetchWithHeaders);
      if (mountedRef.current) setStatus(next);
    } catch {
      /* best-effort; the next poll or hint retries */
    }
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    if (!enabled) return;
    refresh();
    const id = setInterval(refresh, intervalMs);
    return () => clearInterval(id);
  }, [enabled, intervalMs, refresh]);

  // Keyed on the event object: every hint is a fresh object, so a repeated
  // (kind, phase) pair still refetches.
  useEffect(() => {
    if (!enabled || sessionEvent?.kind !== "hosting") return;
    refresh();
  }, [enabled, sessionEvent, refresh]);

  return { status, refresh };
}
