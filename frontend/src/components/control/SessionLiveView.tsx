import { useTranslation } from "react-i18next";
import type { RobotRecord } from "@/hooks/useRobots";
import { useArms } from "@/hooks/useArms";
import { telemetryKind } from "@/lib/armTypes";
import UrdfViewer from "@/components/UrdfViewer";
import JointAngleReadout from "./JointAngleReadout";
import SessionCameraGrid from "./SessionCameraGrid";

/** Both session dialogs share the same arm-left, camera-grid-right layout. */
export default function SessionLiveView({ robot, recording = false, paused = false }: {
  robot?: RobotRecord | null; recording?: boolean; paused?: boolean;
}) {
  const { t } = useTranslation();
  const { byId } = useArms();
  const bimanual = robot?.mode === "bimanual";
  const readoutOnly = telemetryKind(byId(robot?.arm_type)) === "degrees";
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
      <div className={`grid min-w-0 gap-3 ${bimanual ? "grid-cols-2" : "grid-cols-1"}`}>
        {(bimanual ? ["left", "right"] : ["left"]).map(side => (
          <div key={side} className="relative h-[clamp(280px,40vh,440px)] min-w-0 overflow-hidden rounded-lg border border-border">
            {bimanual && <span className="absolute left-3 top-3 z-10 text-xs text-muted-foreground">{t(side === "right" ? "shared.visualizer.rightArm" : "shared.visualizer.leftArm")}</span>}
            {readoutOnly ? <JointAngleReadout jointsKey={side === "right" ? "joints_deg_right" : "joints_deg"} /> : (
              <UrdfViewer armType={robot?.arm_type ?? "so101"} jointsKey={side === "right" ? "joints_right" : "joints"} variant="light" compact />
            )}
          </div>
        ))}
      </div>
      <div className="h-[clamp(280px,40vh,440px)] min-w-0">
        <SessionCameraGrid robot={robot} recording={recording} paused={paused} />
      </div>
    </div>
  );
}
