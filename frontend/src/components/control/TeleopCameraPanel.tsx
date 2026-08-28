import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useRobots } from "@/hooks/useRobots";
import CameraFeed from "./CameraFeed";

/**
 * Optional live camera panel for the teleoperation page. Off by default so we
 * never call getUserMedia just by landing on the page (same consent pattern as
 * the calibration camera toggle). Teleoperation opens no cv2 cameras, so the
 * browser can stream them directly while the arm runs.
 *
 * A strict mirror of the selected robot's configured cameras: one live feed per
 * camera on the robot record (e.g. "wrist_cam", "webcam"), stacked vertically.
 * If the robot has none configured it shows nothing — teleop never surfaces a
 * device that wasn't deliberately added to the robot.
 */
const TeleopCameraPanel: React.FC = () => {
  const { t } = useTranslation();
  const [enabled, setEnabled] = useState(false);
  // Bumped by the retry button to remount the feeds (a fresh getUserMedia
  // attempt) — useful if a camera was unplugged and reconnected.
  const [reloadKey, setReloadKey] = useState(0);
  const { selectedRecord, isLoading: robotsLoading } = useRobots();

  // Feeds come solely from the robot's configured cameras, streamed from the
  // backend by cv2 index (+ unique_id so the server re-anchors it across
  // replugs). A configured camera whose device is currently absent still shows
  // (name + failed-preview placeholder), so the user can tell it's expected but
  // not detected.
  const configured = selectedRecord?.cameras ?? [];
  const feeds = configured.map((c) => ({
    key: c.id,
    name: c.name,
    cameraIndex: c.camera_index,
    uniqueId: c.unique_id,
  }));

  return (
    <div className="bg-card border border-border rounded-lg p-4 flex flex-col gap-4 h-full">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-medium text-foreground">Cameras</h2>
        <div className="flex items-center gap-2">
          {enabled && feeds.length > 0 && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => setReloadKey((k) => k + 1)}
              className="h-9 w-9 text-muted-foreground hover:text-foreground flex-shrink-0"
              title={t("shared.camera.retryTitle")}
              aria-label={t("shared.camera.retry")}
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          )}
          <Label htmlFor="teleop-camera-toggle" className="text-sm text-muted-foreground">
            {enabled ? "On" : "Off"}
          </Label>
          <Switch
            id="teleop-camera-toggle"
            checked={enabled}
            onCheckedChange={setEnabled}
          />
        </div>
      </div>

      {enabled ? (
        feeds.length > 0 ? (
          <div className="flex flex-col gap-3 overflow-y-auto">
            {feeds.map((feed) => (
              <CameraFeed
                key={`${feed.key}:${reloadKey}`}
                cameraIndex={feed.cameraIndex}
                uniqueId={feed.uniqueId}
                label={feed.name}
              />
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            {robotsLoading
              ? t("shared.camera.loadingRobot")
              : t("shared.camera.none")}
          </p>
        )
      ) : (
        <p className="text-sm text-muted-foreground">
          Turn on to watch your cameras while you teleoperate.
        </p>
      )}
    </div>
  );
};

export default TeleopCameraPanel;
