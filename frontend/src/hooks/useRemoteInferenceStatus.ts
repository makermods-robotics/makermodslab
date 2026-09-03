import { useCallback, useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { apiRequest } from "@/lib/apiClient";
import type { Fetcher } from "@/lib/apiClient";
import { useSessionEvent } from "@/hooks/useActiveSession";

/**
 * Live telemetry of the remote-inference (DRTC) session.
 *
 * Polls `GET /api/v1/remote-inference-status` — 1 Hz while a run is live (the
 * rate the child emits its STATS line at), a slow idle tick otherwise — and
 * refetches EAGERLY whenever a `session_changed` hint for this kind lands on
 * the shared socket.
 *
 * Metrics are deliberately not pushed over the WebSocket. That channel carries
 * droppable refetch hints, and it drops under queue pressure — which is
 * exactly when a run is in trouble. `holds` climbing and DEGRADE mean the run
 * is degrading in QUALITY and the operator's answer is "stop it", which is not
 * a millisecond decision.
 */

/** One 1 Hz STATS sample from the child (drtc_protocol.STATS_KEYS).
 *
 * Every key is ALWAYS present — the child fills unknowns with null and the
 * parent refills the full set on the way back — so a dropped or malformed line
 * degrades to "no sample this second" (`stats: null`), never to a
 * half-populated one the UI would render as real. The nulls below are
 * MEANINGFUL: no chunk yet, no operator yet, no correlated round trip yet. */
export interface RemoteInferenceStats {
  /** Seconds since the child started its loop. */
  t: number;
  chunks: number;
  reqs: number;
  sched: number;
  /** Actions still queued ahead of the executor. The health margin is
   * `lead` against `horizon - s_min`. */
  lead: number;
  s_min: number;
  horizon: number;
  lat_steps: number;
  lat_ms: number;
  /** CUMULATIVE count of held (repeated) actions. Render it as a RATE — a
   * healthy run has this FROZEN after warm-up, so "not growing" is the health
   * signal and the running total is misleading forever after. */
  holds: number;
  degrade: boolean;
  chunk_age_ms: number | null;
  /** The operator (publisher) currently driving, null until one joins. */
  active: string | null;
  e2e_p50_us: number | null;
  e2e_p95_us: number | null;
  rtt_us: number | null;
  uncorr: number;
}

/** The transport the RUNNING session actually resolved — the child's READY
 * echo, not what the parent believed it passed. `source` is narrower than the
 * transport ROUTE's field of the same name on purpose (see
 * useRemoteInferenceTransport). */
export interface RemoteInferenceRunTransport {
  url: string;
  room: string;
  source: "cloud" | "local_override" | "cwd";
  operator_present: boolean;
}

export interface RemoteInferenceStatus {
  remote_inference_active: boolean;
  /** resolving → transport_check → preflight → starting → connecting →
   * warming_up → easing → running → stopping → stopped | error. Backend
   * enum values — matched on, never displayed raw. */
  phase: string | null;
  policy_ref: string | null;
  started_at: number | null;
  elapsed_s: number;
  duration_s: number | null;
  log_path: string | null;
  exited: boolean;
  exit_code: number | null;
  outcome: string | null;
  /** Backend prose — rendered verbatim, never translated. */
  error: string | null;
  hint: string | null;
  warning: string | null;
  /** True inside `stopping` while the child eases the arm back to its captured
   * start pose. A FLAG rather than a phase name, because the session expiry
   * watchdog keys off the phase still reading "stopping". */
  returning_to_rest: boolean;
  shutting_down: boolean;
  stats: RemoteInferenceStats | null;
  transport: RemoteInferenceRunTransport | null;
}

export async function getRemoteInferenceStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<RemoteInferenceStatus> {
  return apiRequest<RemoteInferenceStatus>(
    baseUrl,
    fetcher,
    "/api/v1/remote-inference-status",
    { signal, action: "Get remote inference status" },
  );
}

/** The child emits one STATS line a second; polling faster only reports the
 * same sample twice. */
export const REMOTE_STATUS_ACTIVE_MS = 1000;
/** Idle tick. The `session_changed` hint is what actually catches a run
 * starting elsewhere; this is the self-heal for a dropped broadcast. */
export const REMOTE_STATUS_IDLE_MS = 5000;

export interface UseRemoteInferenceStatus {
  status: RemoteInferenceStatus | null;
  /** True once the first response (of this mount) has landed. */
  loaded: boolean;
  refresh: () => void;
}

/**
 * @param enabled gates only the IDLE poll. A live run is polled at 1 Hz
 * regardless — the surface that started it may well be hidden — and the
 * `session_changed` hint always refetches, which is how a run started from
 * another tab or the SDK becomes visible here.
 */
export function useRemoteInferenceStatus(
  enabled: boolean,
): UseRemoteInferenceStatus {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [status, setStatus] = useState<RemoteInferenceStatus | null>(null);
  const [loaded, setLoaded] = useState(false);
  const sessionEvent = useSessionEvent();

  const active = status?.remote_inference_active === true;

  const [nonce, setNonce] = useState(0);
  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  // Refetch on the session_changed HINT — never treat the event as state.
  // Claim, every phase transition and the final release each land here, which
  // is what makes the panel feel immediate without pushing metrics.
  useEffect(() => {
    if (sessionEvent?.kind === "remote_inference") refresh();
  }, [sessionEvent, refresh]);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const next = await getRemoteInferenceStatus(baseUrl, fetchWithHeaders);
        if (cancelled) return;
        setStatus(next);
        setLoaded(true);
      } catch {
        // Transient; the next tick retries. A failed poll must never blank a
        // panel the operator is reading mid-run.
      }
    };
    void tick();
    // A live run always ticks; an idle one only while the caller wants it.
    // (`active` is false on the very first pass, so an idle+disabled mount
    // still makes exactly ONE request — enough to learn a run is live and
    // re-arm at 1 Hz.)
    const id =
      active || enabled
        ? setInterval(
            () => void tick(),
            active ? REMOTE_STATUS_ACTIVE_MS : REMOTE_STATUS_IDLE_MS,
          )
        : null;
    return () => {
      cancelled = true;
      if (id !== null) clearInterval(id);
    };
    // `active` is in the deps so the interval re-arms at the other rate the
    // moment a run starts or ends.
  }, [baseUrl, fetchWithHeaders, enabled, active, nonce]);

  return { status, loaded, refresh };
}

/** Per-second rate of a cumulative counter between two samples, or null when
 * there is no earlier sample to difference against.
 *
 * Exported for the status panel AND its test: `holds` is the one counter whose
 * cumulative value actively misleads (a healthy run froze it at 41 during
 * warm-up and never touched it again), so the panel shows the derivative. */
export function perSecondRate(
  current: { t: number; value: number },
  previous: { t: number; value: number } | null,
): number | null {
  if (previous == null) return null;
  const dt = current.t - previous.t;
  if (dt <= 0) return null;
  return (current.value - previous.value) / dt;
}
