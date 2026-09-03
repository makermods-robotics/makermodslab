import { useCallback, useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { apiRequest } from "@/lib/apiClient";
import type { Fetcher } from "@/lib/apiClient";
import type { RemoteEngine } from "@/components/remote-inference/remoteRunConfig";

/**
 * The GPU half of a remote-inference run, which the Lab now launches itself
 * (`modal run makermodslab/drtc/modal_policy*.py`).
 *
 * A LAB-LEVEL RESOURCE, not a session: it holds no hardware, it has no lease,
 * and stopping it is not a safety action. That is why it lives on its own three
 * routes rather than as a field on the remote-run options — a 1-3 minute cold
 * start inside the session start would hold the robot-busy discriminant while
 * the arm sat completely free.
 *
 * It is also NOT what unblocks "Run it remotely". That gate stays the transport
 * probe's `operator_present`, which observes the ROOM; `state: "ready"` here is
 * derived from the container's own stdout and is a hint, not an authority.
 */

/** idle | starting | ready | failed | stopping. Backend enum values — matched
 * on, never displayed raw. */
export type GpuState = "idle" | "starting" | "ready" | "failed" | "stopping";

export interface GpuStatus {
  state: GpuState;
  /** The container's progress: tailscale_up | loading | warmup | connecting |
   * connected | claimed. Null before the first recognizable line. Backend enum
   * values. `connected` is what flips `state` to ready — `claimed` is a display
   * refinement, since the policy claims control in a background task and a
   * healthy run may never print it. */
  phase: string | null;
  /** Which wrapper is running. Backend identifiers ("sync" / "rtc"). */
  engine: string | null;
  policy_hub_id: string | null;
  /** The room the launcher pinned with --livekit-room. Data. */
  room: string | null;
  /** Survives an idle transition: after a failure this is the most useful
   * thing left. A path — data. */
  log_path: string | null;
  started_at: number | null;
  elapsed_s: number;
  /** Backend prose — rendered verbatim, never translated. */
  message: string | null;
  hint: string | null;
  /** The `gpu.*` code behind a FAILED state ("gpu.unauthenticated",
   * "gpu.launch_failed"); null in every other state. Backend data — matched
   * on and shown verbatim, never translated. Branch on this, never on the
   * prose beside it. */
  code: string | null;
  /** The most recent output line. DATA: a container's own log text. */
  last_line: string | null;
  /** Seconds until the idle auto-stop. Null unless `ready` AND no remote run is
   * live — a busy GPU is not idle, and a countdown that kept ticking through a
   * live run would be a lie. */
  idle_stop_in_s: number | null;
}

export interface GpuStartRequest {
  engine: RemoteEngine;
  policy_hub_id: string;
  task: string;
  horizon: number;
  fps: number;
  video_codec: "H264" | "MJPEG";
  s_min: number;
}

interface GpuLaunchResult {
  started: boolean;
  message: string;
  gpu: GpuStatus;
}

export async function getGpuStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<GpuStatus> {
  return apiRequest<GpuStatus>(
    baseUrl,
    fetcher,
    "/api/v1/remote-inference/gpu",
    { signal, action: "Read GPU status" },
  );
}

export async function startGpu(
  baseUrl: string,
  fetcher: Fetcher,
  body: GpuStartRequest,
): Promise<GpuLaunchResult> {
  return apiRequest<GpuLaunchResult>(
    baseUrl,
    fetcher,
    "/api/v1/remote-inference/gpu/start",
    { method: "POST", body, action: "Start the GPU" },
  );
}

export async function stopGpu(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<GpuStatus> {
  return apiRequest<GpuStatus>(
    baseUrl,
    fetcher,
    "/api/v1/remote-inference/gpu/stop",
    { method: "POST", action: "Stop the GPU" },
  );
}

/** A cold start moves through five phases in 1-3 minutes; 2s keeps the phase
 * strip honest without hammering a route that only reads memory. */
export const GPU_ACTIVE_POLL_MS = 2000;
/** Everything else. A ready GPU changes only when the idle auto-stop fires. */
export const GPU_IDLE_POLL_MS = 10000;

export interface UseGpuLauncher {
  status: GpuStatus | null;
  /** True while a start/stop request is in flight (distinct from the backend's
   * own `starting`/`stopping`, which outlive the request). */
  pending: boolean;
  /** The thrown error's own text, or null. Backend prose — shown as raised. */
  error: string | null;
  start: (body: GpuStartRequest) => Promise<void>;
  stop: () => Promise<void>;
  refresh: () => void;
}

/**
 * @param enabled poll while true. False leaves the last answer in place rather
 * than blanking it, so collapsing the form does not throw away what the
 * operator just read.
 */
export function useGpuLauncher(enabled: boolean): UseGpuLauncher {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [status, setStatus] = useState<GpuStatus | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  // Ref rather than state: the poll effect reads it to decide whether to keep
  // ticking, and putting it in the dep array would re-arm the interval on
  // every answer.
  const settled = useRef(false);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  const state = status?.state;
  // A settled failure (or a plain idle) changes only when the operator acts, so
  // polling it forever is pure noise. Any action below clears the flag.
  const quiet = state === "failed" || (state === "idle" && settled.current);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const next = await getGpuStatus(baseUrl, fetchWithHeaders);
        if (cancelled) return;
        setStatus(next);
        if (next.state === "failed" || next.state === "idle") settled.current = true;
      } catch {
        // Transient; the next tick retries. A failed poll must never blank a
        // panel the operator is reading mid-launch.
      }
    };
    void tick();
    if (quiet) return () => { cancelled = true; };
    const id = setInterval(
      () => void tick(),
      state === "starting" || state === "stopping"
        ? GPU_ACTIVE_POLL_MS
        : GPU_IDLE_POLL_MS,
    );
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [enabled, baseUrl, fetchWithHeaders, state, quiet, nonce]);

  const start = useCallback(
    async (body: GpuStartRequest) => {
      setPending(true);
      setError(null);
      settled.current = false;
      try {
        const result = await startGpu(baseUrl, fetchWithHeaders, body);
        setStatus(result.gpu);
      } catch (e) {
        // The backend's own coded refusal text (gpu.cli_missing names the
        // install line, gpu.launch_failed names the field or the tailnet).
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setPending(false);
        refresh();
      }
    },
    [baseUrl, fetchWithHeaders, refresh],
  );

  const stop = useCallback(async () => {
    setPending(true);
    setError(null);
    settled.current = false;
    try {
      setStatus(await stopGpu(baseUrl, fetchWithHeaders));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
      refresh();
    }
  }, [baseUrl, fetchWithHeaders, refresh]);

  return { status, pending, error, start, stop, refresh };
}
