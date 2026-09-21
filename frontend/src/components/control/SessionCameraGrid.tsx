import { useId, useState } from "react";
import { useTranslation } from "react-i18next";
import { Switch } from "@/components/ui/switch";
import type { RobotRecord } from "@/hooks/useRobots";
import CameraFeed from "./CameraFeed";
import RecordingCameraStream from "./RecordingCameraStream";

export default function SessionCameraGrid({
  robot, recording = false, paused = false,
}: { robot?: RobotRecord | null; recording?: boolean; paused?: boolean }) {
  const { t } = useTranslation();
  const toggleId = useId();
  const [enabled, setEnabled] = useState(true);
  const cameras = robot?.cameras ?? [];
  // robot_factory attaches every bimanual recording camera to the left arm;
  // LeRobot prefixes those observation keys independently of the device index.
  return (
    <section aria-label={t("shared.camera.title")} className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex shrink-0 items-center justify-between gap-2">
        <label htmlFor={toggleId} className="text-sm font-medium">{t("shared.camera.title")} <span className="text-muted-foreground">({cameras.length})</span></label>
        <Switch id={toggleId} checked={enabled} onCheckedChange={setEnabled} />
      </div>
      {paused && enabled && <p className="text-xs text-muted-foreground">{t("shared.camera.paused")}</p>}
      {!enabled ? <p className="text-sm text-muted-foreground">{t("shared.camera.off")}</p> : cameras.length ? (
        <div className={`grid min-h-0 auto-rows-max content-start gap-3 overflow-y-auto ${cameras.length === 1 ? "grid-cols-1" : "grid-cols-2"}`}>
          {cameras.map(camera => recording ? (
            <div key={camera.id} className="min-w-0">
              <div className="relative aspect-video">
                <RecordingCameraStream cameraName={`${robot?.mode === "bimanual" ? "left_" : ""}${camera.name}`} />
              </div>
              <div className="mt-1 truncate text-xs text-muted-foreground" title={camera.name}>{camera.name}</div>
            </div>
          ) : (
            <CameraFeed key={camera.id} cameraIndex={camera.camera_index} uniqueId={camera.unique_id} label={camera.name} />
          ))}
        </div>
      ) : <p className="text-sm text-muted-foreground">{t(robot ? "shared.camera.none" : "shared.camera.loadingRobot")}</p>}
    </section>
  );
}
