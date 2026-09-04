import { useCallback, useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { useSessionEvent } from "@/hooks/useActiveSession";
import { getStationStatus, type StationStatus } from "@/lib/remoteApi";

/**
 * GET /api/v1/station — the `--host` posture: is this a station, which
 * robot it hosts (null until chosen), and which saved robots it could host.
 * Kept fresh the same two ways useHostingStatus is: a light poll
 * (`intervalMs`, 5s by default) and a refetch on every `session_changed`
 * hint for kind `hosting`, since choosing a robot re-arms hosting and the
 * chip should follow within a broadcast. The hint is never treated as state.
 *
 * `station_mode` is a launch flag, fixed for the process lifetime, so once a
 * read says this is NOT a station the poll stops — nothing here can change
 * until a restart, and the corner is mounted twice (launchpad + studio).
 *
 * A failed read KEEPS the last status; `status` is null until the first
 * read lands.
 */
export function useStationStatus(
  options: { enabled?: boolean; intervalMs?: number } = {},
) {
  const { enabled = true, intervalMs = 5_000 } = options;
  const { baseUrl, fetchWithHeaders } = useApi();
  const sessionEvent = useSessionEvent();
  const [status, setStatus] = useState<StationStatus | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const next = await getStationStatus(baseUrl, fetchWithHeaders);
      if (mountedRef.current) setStatus(next);
    } catch {
      /* best-effort; the next poll or hint retries */
    }
  }, [baseUrl, fetchWithHeaders]);

  // Poll until the first read lands; keep polling only on a station.
  const isStation = status === null || status.station_mode;

  useEffect(() => {
    if (!enabled) return;
    refresh();
    if (!isStation) return;
    const id = setInterval(refresh, intervalMs);
    return () => clearInterval(id);
  }, [enabled, isStation, intervalMs, refresh]);

  // Keyed on the event object: every hint is a fresh object, so a repeated
  // (kind, phase) pair still refetches.
  useEffect(() => {
    if (!enabled || !isStation || sessionEvent?.kind !== "hosting") return;
    refresh();
  }, [enabled, isStation, sessionEvent, refresh]);

  return { status, refresh };
}
