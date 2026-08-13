import React from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import UrdfViewer from "../UrdfViewer";
import Logo from "@/components/Logo";

interface VisualizerPanelProps {
  onGoBack: () => void;
  className?: string;
  /** Render a second arm viewer (driven by the "joints_right" stream). */
  bimanual?: boolean;
  /** Optional content rendered as a column beside the 3D viewer (e.g. a camera panel). */
  rightSlot?: React.ReactNode;
}

const VisualizerPanel: React.FC<VisualizerPanelProps> = ({
  onGoBack,
  className,
  bimanual = false,
  rightSlot,
}) => {
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
          <h2 className="text-xl font-medium text-foreground">Teleoperation</h2>
          <Button
            onClick={onGoBack}
            className="ml-auto bg-destructive text-destructive-foreground hover:bg-destructive/90 flex-shrink-0"
          >
            Done
          </Button>
        </div>
        {/* No standing torque warning here: stops are graceful (the arm
            drives back to its session-start pose before torque releases) and
            the stop toast explains the behavior at the moment it happens.
            Only error stops release in place. */}
        {bimanual ? (
          <div className="flex-1 flex flex-col sm:flex-row gap-2 min-h-[50vh] lg:min-h-0">
            <div className="flex-1 flex flex-col">
              <span className="text-xs text-muted-foreground mb-1">Left arm</span>
              <div className="flex-1 bg-background rounded border border-border min-h-[25vh]">
                <UrdfViewer jointsKey="joints" />
              </div>
            </div>
            <div className="flex-1 flex flex-col">
              <span className="text-xs text-muted-foreground mb-1">Right arm</span>
              <div className="flex-1 bg-background rounded border border-border min-h-[25vh]">
                <UrdfViewer jointsKey="joints_right" />
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 bg-background rounded border border-border min-h-[50vh] lg:min-h-0">
            <UrdfViewer />
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
