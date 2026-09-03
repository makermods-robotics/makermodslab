import { createContext, useContext, useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";

/**
 * The robot-driving features of the backend's mutual-exclusion state model
 * (CLAUDE.md "State model & mutual exclusion") — the `kind` vocabulary of the
 * `session_changed` event (makermodslab/session_events.py). `hosting` is the
 * station side of remote teleoperation (holds the follower like teleop),
 * `remote_teleoperation` the operator side (holds the leader only).
 */
export type SessionKind =
  | "teleoperation"
  | "recording"
  | "inference"
  | "replay"
  | "calibration"
  | "auto_calibration"
  | "wiggle"
  | "hosting"
  | "remote_teleoperation";

/** The latest `session_changed` hint seen on the shared WS channel. */
export interface SessionChangedEvent {
  kind: SessionKind;
  active: boolean;
  /** Feature-specific phase ("releasing", "recording", "easing_in", …) or null. */
  phase: string | null;
  /** Client-side Date.now() when the event arrived. */
  receivedAt: number;
}

/**
 * Subscribe to backend `session_changed` events on the shared /ws/joint-data
 * channel and expose the LATEST one (null until any arrives).
 *
 * The backend broadcasts a hint whenever any robot-driving feature's session
 * state transitions — claim, phase change, final release — from whichever
 * flow triggered it (UI, SDK, watchdog, shutdown). The payload is a HINT:
 * consumers refetch the relevant status endpoint on it and never treat the
 * event itself as state (broadcasts are droppable, and polling still covers
 * every gap).
 *
 * Mirrors useJobsChangedSignal: raw WebSocket, demux on `data.type`,
 * auto-reconnect with a 3s delay if the server bounces.
 */
export const useActiveSession = (): SessionChangedEvent | null => {
  const { wsBaseUrl } = useApi();
  const [event, setEvent] = useState<SessionChangedEvent | null>(null);

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
      ws.onmessage = (message) => {
        try {
          const data = JSON.parse(message.data);
          if (data?.type === "session_changed" && data?.session?.kind) {
            setEvent({
              kind: data.session.kind as SessionKind,
              active: Boolean(data.session.active),
              phase: data.session.phase ?? null,
              receivedAt: Date.now(),
            });
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

  return event;
};

/**
 * App-wide access to the latest session event, provided by SessionProvider
 * (contexts/SessionContext.tsx) so the whole tree shares ONE socket. The
 * context object and consumer hook live here — not in the provider's file —
 * so that file exports only a component (keeps react-refresh lint clean).
 */
export const ActiveSessionContext = createContext<SessionChangedEvent | null>(null);

/** Latest `session_changed` event, or null before any arrived. */
export const useSessionEvent = (): SessionChangedEvent | null => useContext(ActiveSessionContext);
