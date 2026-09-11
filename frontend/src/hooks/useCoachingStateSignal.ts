import { useEffect, useRef } from "react";
import { useApi } from "@/contexts/ApiContext";
import { CoachingState } from "@/lib/inferenceApi";

/**
 * Subscribe to coaching state pushes on the shared /ws/joint-data channel.
 *
 * The coaching banner is the only thing in this app that tells a person whether
 * they or a robot is holding an arm, and it used to learn that from a 1 Hz poll.
 * The handover glide lasts about two seconds, so up to half of the window in
 * which the banner reads "the arm is moving — don't fight it" could elapse
 * before the operator could possibly have seen it. Worse, the operator looks up
 * at the moment they press the key, which is exactly the moment a poll is most
 * likely to be stale.
 *
 * So the backend pushes the whole coaching block (`type: "coaching_state"`) the
 * instant it changes — same shape as the poll's, so the caller merges it in
 * without deciding which of two objects wins.
 *
 * The poll stays. It is the reconciler: a dropped push, a socket that bounced,
 * or a browser tab that was asleep all heal within a second. This hook is
 * latency, never correctness — nothing here is the only path to any state.
 *
 * Callback ref is captured so identity changes don't tear down the socket.
 * Auto-reconnects with a 3s delay if the server bounces.
 */
export const useCoachingStateSignal = (
  onState: (state: CoachingState) => void,
  enabled: boolean,
) => {
  const { wsBaseUrl } = useApi();
  const stateRef = useRef(onState);
  stateRef.current = onState;

  useEffect(() => {
    // Only while a coaching session is live. A socket per dialog mount would
    // otherwise sit open through every non-coaching inference run.
    if (!enabled) return;

    let cancelled = false;
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (cancelled) return;
      try {
        ws = new WebSocket(`${wsBaseUrl}/ws/joint-data`);
      } catch {
        reconnectTimer = setTimeout(connect, 3000);
        return;
      }
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data?.type === "coaching_state" && data?.coaching === true) {
            stateRef.current(data as CoachingState);
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
  }, [wsBaseUrl, enabled]);
};
