import { useEffect, useRef, useState, useCallback } from "react";
import { URDFViewerElement } from "@/lib/urdfViewerHelpers";
import { useApi } from "@/contexts/ApiContext";

interface JointData {
  type: "joint_update";
  timestamp: number;
  // "joints" (primary/left arm) and, for bimanual, "joints_right".
  [key: string]: unknown;
}

interface UseRealTimeJointsProps {
  viewerRef: React.RefObject<URDFViewerElement>;
  enabled?: boolean;
  websocketUrl?: string;
  /** Which field of the joint message to apply — "joints" (default) or "joints_right". */
  jointsKey?: string;
}

const INITIAL_RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_DELAY_MS = 30000;

export const useRealTimeJoints = ({
  viewerRef,
  enabled = true,
  websocketUrl,
  jointsKey = "joints",
}: UseRealTimeJointsProps) => {
  const { wsBaseUrl } = useApi();
  const finalWebSocketUrl = websocketUrl || `${wsBaseUrl}/api/v1/ws/joint-data`;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectDelayRef = useRef(INITIAL_RECONNECT_DELAY_MS);
  const intentionallyClosedRef = useRef(false);
  const [isConnected, setIsConnected] = useState(false);

  const updateJointValues = useCallback(
    (joints: Record<string, number>) => {
      const viewer = viewerRef.current;
      if (!viewer || typeof viewer.setJointValue !== "function") return;
      Object.entries(joints).forEach(([jointName, value]) => {
        try {
          viewer.setJointValue(jointName, value);
        } catch (error) {
          console.warn(`Failed to set joint ${jointName}:`, error);
        }
      });
    },
    [viewerRef]
  );

  useEffect(() => {
    if (!enabled) return;

    intentionallyClosedRef.current = false;

    const connect = () => {
      if (intentionallyClosedRef.current) return;

      let ws: WebSocket;
      try {
        ws = new WebSocket(finalWebSocketUrl);
      } catch (error) {
        console.error("Failed to create WebSocket:", error);
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
          const data = JSON.parse(event.data) as JointData;
          const joints = data[jointsKey] as Record<string, number> | undefined;
          if (data.type === "joint_update" && joints) {
            updateJointValues(joints);
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
        }
      };

      ws.onclose = (event) => {
        setIsConnected(false);
        wsRef.current = null;
        if (intentionallyClosedRef.current) return;
        if (event.code === 1000) return; // clean close
        scheduleReconnect();
      };

      ws.onerror = () => {
        setIsConnected(false);
      };
    };

    const scheduleReconnect = () => {
      if (reconnectTimeoutRef.current) return;
      const delay = reconnectDelayRef.current;
      reconnectDelayRef.current = Math.min(
        delay * 2,
        MAX_RECONNECT_DELAY_MS
      );
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
      if (wsRef.current) {
        wsRef.current.close(1000);
        wsRef.current = null;
      }
      setIsConnected(false);
    };
  }, [enabled, finalWebSocketUrl, updateJointValues, jointsKey]);

  return { isConnected };
};
