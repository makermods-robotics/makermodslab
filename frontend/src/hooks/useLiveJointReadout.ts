import { useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";

interface ReplayJointMessage {
  type: "replay_joint_update";
  joints: Record<string, number>;
  timestamp: number;
}

const INITIAL_RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_DELAY_MS = 30000;

/**
 * Live numeric readout of the arm's actual position during a hardware
 * replay — a much smaller sibling of useRealTimeJoints: instead of driving
 * a 3D viewer's setJointValue, it just holds the latest `{jointName: value}`
 * dict in state. Filters on `type === "replay_joint_update"` specifically
 * (not the teleop-only "joint_update") so it never misreads a differently-
 * shaped payload from another feature sharing the same socket.
 */
export function useLiveJointReadout(enabled: boolean): {
  joints: Record<string, number>;
  isConnected: boolean;
} {
  const { wsBaseUrl } = useApi();
  const [joints, setJoints] = useState<Record<string, number>>({});
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelayRef = useRef(INITIAL_RECONNECT_DELAY_MS);
  const intentionallyClosedRef = useRef(false);

  useEffect(() => {
    if (!enabled) return;
    intentionallyClosedRef.current = false;

    const connect = () => {
      if (intentionallyClosedRef.current) return;
      let ws: WebSocket;
      try {
        ws = new WebSocket(`${wsBaseUrl}/api/v1/ws/joint-data`);
      } catch {
        scheduleReconnect();
        return;
      }
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        reconnectDelayRef.current = INITIAL_RECONNECT_DELAY_MS;
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
      };
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as ReplayJointMessage;
          if (data.type === "replay_joint_update" && data.joints) {
            setJoints(data.joints);
          }
        } catch {
          // Ignore a malformed frame; the next one is unaffected.
        }
      };
      ws.onclose = (event) => {
        setIsConnected(false);
        wsRef.current = null;
        if (intentionallyClosedRef.current || event.code === 1000) return;
        scheduleReconnect();
      };
      ws.onerror = () => setIsConnected(false);
    };

    const scheduleReconnect = () => {
      if (reconnectTimeoutRef.current) return;
      const delay = reconnectDelayRef.current;
      reconnectDelayRef.current = Math.min(delay * 2, MAX_RECONNECT_DELAY_MS);
      reconnectTimeoutRef.current = setTimeout(() => {
        reconnectTimeoutRef.current = null;
        connect();
      }, delay);
    };

    connect();

    return () => {
      intentionallyClosedRef.current = true;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      wsRef.current?.close(1000);
      wsRef.current = null;
      setIsConnected(false);
    };
  }, [enabled, wsBaseUrl]);

  return { joints, isConnected };
}
