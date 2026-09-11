import { useEffect, useRef } from "react";
import { useApi } from "@/contexts/ApiContext";
import { JobProgressSnapshot } from "@/lib/jobsApi";

/**
 * Subscribe to backend job events on the shared /ws/joint-data channel.
 *
 * Two event types flow on this socket:
 *  - `jobs_changed`  → fired on submit / watchdog finalisation / delete.
 *                       Triggers `onChange` so the caller can refetch /jobs.
 *  - `job_progress`  → fired by the watchdog (~1Hz) while jobs are running.
 *                       Triggers `onProgress` with per-job snapshots so the
 *                       UI can update the progress bar in place — no fetch.
 *
 * Callback refs are captured so identity changes don't tear down the socket.
 * Auto-reconnects with a 3s delay if the server bounces.
 */
export const useJobsChangedSignal = (
  onChange: () => void,
  onProgress?: (snapshots: JobProgressSnapshot[]) => void,
) => {
  const { wsBaseUrl } = useApi();
  const changeRef = useRef(onChange);
  changeRef.current = onChange;
  const progressRef = useRef(onProgress);
  progressRef.current = onProgress;

  useEffect(() => {
    let cancelled = false;
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (cancelled) return;
      try {
        ws = new WebSocket(`${wsBaseUrl}/api/v1/ws/joint-data`);
      } catch {
        reconnectTimer = setTimeout(connect, 3000);
        return;
      }
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data?.type === "jobs_changed") {
            changeRef.current();
          } else if (
            data?.type === "job_progress" &&
            progressRef.current &&
            Array.isArray(data?.jobs)
          ) {
            progressRef.current(data.jobs as JobProgressSnapshot[]);
          }
        } catch {
          /* ignore non-JSON or unexpected payloads */
        }
      };
      ws.onclose = () => {
        if (cancelled) return;
        reconnectTimer = setTimeout(connect, 3000);
      };
    };
    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) ws.close();
    };
  }, [wsBaseUrl]);
};
