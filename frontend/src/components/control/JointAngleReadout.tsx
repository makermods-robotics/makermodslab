import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Activity } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { cn } from "@/lib/utils";

interface JointAngleMessage {
  type: "joint_update";
  /** Degrees keyed by motor name — see teleoperate.get_maker_joint_degrees. */
  joints_deg?: Record<string, number>;
  joints_deg_right?: Record<string, number>;
  timestamp: number;
}

interface JointAngleReadoutProps {
  /** Which arm's stream to show: the primary/left arm, or the right one. */
  jointsKey?: "joints_deg" | "joints_deg_right";
  className?: string;
}

const INITIAL_RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_DELAY_MS = 30000;

/**
 * Live numeric joint readout — what a Maker arm shows in place of the 3D
 * viewer.
 *
 * The viewer cannot serve a Maker arm: the only URDF that ships is the
 * SO-101's (`frontend/public/so-101-urdf`), and the Maker arm is a different
 * 7-DOF geometry with a joint (`wrist_yaw`) the SO-101 model has no bone for.
 * Driving that model with Maker angles would animate the wrong arm with
 * silently wrong values, which is worse than showing no model — so the
 * backend sends `joints` empty for a Maker session and puts the real angles in
 * `joints_deg`, and this renders those.
 *
 * A sibling of useRealTimeJoints in shape (same socket, same reconnect
 * backoff) but it drives no 3D scene, so it holds the latest dict in state
 * rather than pushing into a viewer imperatively.
 */
const JointAngleReadout: React.FC<JointAngleReadoutProps> = ({
  jointsKey = "joints_deg",
  className,
}) => {
  const { t } = useTranslation();
  const { wsBaseUrl } = useApi();
  const [joints, setJoints] = useState<Record<string, number>>({});
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  const reconnectDelayRef = useRef(INITIAL_RECONNECT_DELAY_MS);
  const intentionallyClosedRef = useRef(false);

  useEffect(() => {
    intentionallyClosedRef.current = false;

    const scheduleReconnect = () => {
      if (intentionallyClosedRef.current) return;
      reconnectTimeoutRef.current = setTimeout(() => {
        reconnectDelayRef.current = Math.min(
          reconnectDelayRef.current * 2,
          MAX_RECONNECT_DELAY_MS,
        );
        connect();
      }, reconnectDelayRef.current);
    };

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
      };
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as JointAngleMessage;
          // Filter on the message type: this socket also carries the
          // droppable control events (jobs_changed, session_changed), and a
          // differently-shaped payload must not be misread as joint data.
          if (data.type !== "joint_update") return;
          const next = data[jointsKey];
          if (next && Object.keys(next).length > 0) setJoints(next);
        } catch {
          // A malformed frame is dropped; the next one self-heals.
        }
      };
      ws.onclose = () => {
        setIsConnected(false);
        scheduleReconnect();
      };
      ws.onerror = () => ws.close();
    };

    connect();

    return () => {
      intentionallyClosedRef.current = true;
      if (reconnectTimeoutRef.current)
        clearTimeout(reconnectTimeoutRef.current);
      wsRef.current?.close();
    };
  }, [wsBaseUrl, jointsKey]);

  const entries = Object.entries(joints);

  return (
    // Centred and width-capped rather than stretched: this panel occupies the
    // 3D viewer's slot, which is tall, and a seven-row list flush against the
    // top edge with the caption pinned to the bottom reads as a broken layout
    // rather than a deliberate one.
    <div
      className={cn(
        "flex h-full flex-col items-center justify-center overflow-auto p-6 text-foreground",
        className,
      )}
    >
      <div className="w-full max-w-sm space-y-3">
        <div className="flex items-center gap-2">
          <Activity
            className={cn(
              "h-4 w-4",
              isConnected ? "text-ok" : "text-muted-foreground",
            )}
          />
          <span className="text-sm font-medium">
            {t("shared.visualizer.jointAngles")}
          </span>
        </div>

        {entries.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            {t("shared.visualizer.waitingForJoints")}
          </p>
        ) : (
          <div className="grid grid-cols-1 gap-x-6 gap-y-1">
            {entries.map(([motor, angle]) => (
              <div
                key={motor}
                className="flex items-baseline justify-between gap-3 border-b border-border/50 py-1"
              >
                {/* Motor names are DATA — they key the calibration file and the
                  recorded dataset's feature columns — so they render verbatim
                  in every language. */}
                <span className="truncate font-mono text-xs text-muted-foreground">
                  {motor}
                </span>
                <span className="shrink-0 font-mono text-sm tabular-nums">
                  {angle.toFixed(1)}&deg;
                </span>
              </div>
            ))}
          </div>
        )}

        <p className="pt-1 text-xs text-muted-foreground">
          {t("shared.visualizer.noMakerModel")}
        </p>
      </div>
    </div>
  );
};

export default JointAngleReadout;
