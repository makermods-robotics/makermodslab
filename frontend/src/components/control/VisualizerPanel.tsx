import React from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { ArmType } from "@/lib/armTypes";
import UrdfViewer from "../UrdfViewer";
import JointAngleReadout from "./JointAngleReadout";
import Logo from "@/components/Logo";

interface VisualizerPanelProps {
  onGoBack: () => void;
  className?: string;
  /** Render a second arm viewer (driven by the "joints_right" stream). */
  bimanual?: boolean;
  /**
   * Show the numeric joint readout instead of the 3D model. Set for an arm
   * type with no shipped URDF (the Metal arm) — see JointAngleReadout.
   */
  readoutOnly?: boolean;
  /** Arm type whose URDF the 3D viewer should load (ignored when readoutOnly). */
  armType?: ArmType;
  /** Optional content rendered as a column beside the 3D viewer (e.g. a camera panel). */
  rightSlot?: React.ReactNode;
}

const VisualizerPanel: React.FC<VisualizerPanelProps> = ({
  onGoBack,
  className,
  bimanual = false,
  readoutOnly = false,
  armType = "so101",
  rightSlot,
}) => {
  const { t } = useTranslation();
  return (
    <div
      className={cn(
        "w-full p-2 sm:p-4 space-y-4 lg:space-y-0 lg:space-x-4 flex flex-col lg:flex-row",
        className
      )}
    >
      <div className="bg-card rounded-lg p-4 flex-1 flex flex-col">
        <div className="flex items-center gap-4 mb-4">
          <Logo iconOnly={true} />
          <div className="w-px h-6 bg-border" />
          <h2 className="text-xl font-medium text-foreground">
            {t("shared.visualizer.heading")}
          </h2>
          <Button
            onClick={onGoBack}
            className="ml-auto bg-destructive text-destructive-foreground hover:bg-destructive/90 flex-shrink-0"
          >
            {t("shared.visualizer.done")}
          </Button>
        </div>
        {/* No standing torque warning here: stops are graceful (the arm
            drives back to its session-start pose before torque releases) and
            the stop toast explains the behavior at the moment it happens.
            Only error stops release in place. */}
        {bimanual ? (
          <div className="flex-1 flex flex-col sm:flex-row gap-2 min-h-[50vh] lg:min-h-0">
            <div className="flex-1 flex flex-col">
              <span className="text-xs text-muted-foreground mb-1">
                {t("shared.visualizer.leftArm")}
              </span>
              <div className="flex-1 bg-background rounded border border-border min-h-[25vh]">
                {readoutOnly ? (
                  <JointAngleReadout jointsKey="joints_deg" />
                ) : (
                  <UrdfViewer jointsKey="joints" armType={armType} />
                )}
              </div>
            </div>
            <div className="flex-1 flex flex-col">
              <span className="text-xs text-muted-foreground mb-1">
                {t("shared.visualizer.rightArm")}
              </span>
              <div className="flex-1 bg-background rounded border border-border min-h-[25vh]">
                {readoutOnly ? (
                  <JointAngleReadout jointsKey="joints_deg_right" />
                ) : (
                  <UrdfViewer jointsKey="joints_right" armType={armType} />
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 bg-background rounded border border-border min-h-[50vh] lg:min-h-0">
            {readoutOnly ? (
              <JointAngleReadout />
            ) : (
              <UrdfViewer armType={armType} />
            )}
          </div>
        )}
      </div>
      {rightSlot && (
        <div className="lg:w-96 flex flex-col">{rightSlot}</div>
      )}
    </div>
  );
};

export default VisualizerPanel;
