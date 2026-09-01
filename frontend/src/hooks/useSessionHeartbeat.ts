import { useEffect } from "react";
import { useApi } from "@/contexts/ApiContext";
import { heartbeatSession } from "@/lib/sessionApi";

/**
 * Renew a session's lease while this tab's flow is live.
 *
 * POSTs /api/v1/sessions/{id}/heartbeat every HEARTBEAT_INTERVAL_MS while
 * `active` and a session id is known; stops on unmount or when `active` flips
 * false. The interval (~20s) gives three attempts inside the server's default
 * 60s lease, so one dropped request never costs the session.
 *
 * A failed heartbeat is deliberately NON-FATAL: log and keep trying. A
 * transient network blip must not kill the flow's UX — the server is the
 * authority, and if the lease really did expire the flow's own status poll
 * shows the safety stop. (A 404/409 here most often just means the session
 * ended a beat before we did.)
 */
export const HEARTBEAT_INTERVAL_MS = 20_000;

export function useSessionHeartbeat(
  sessionId: string | null,
  owner: string | null,
  active: boolean
): void {
  const { baseUrl, fetchWithHeaders } = useApi();

  useEffect(() => {
    if (!active || !sessionId || !owner) return;
    let cancelled = false;
    const beat = async () => {
      try {
        await heartbeatSession(baseUrl, fetchWithHeaders, sessionId, owner);
      } catch (e) {
        if (!cancelled) {
          console.warn(`Session heartbeat for ${sessionId} failed:`, e);
        }
      }
    };
    // No immediate beat: creating (or renewing) the lease already set the
    // deadline; the first interval tick is comfortably inside it.
    const id = setInterval(beat, HEARTBEAT_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [active, sessionId, owner, baseUrl, fetchWithHeaders]);
}
