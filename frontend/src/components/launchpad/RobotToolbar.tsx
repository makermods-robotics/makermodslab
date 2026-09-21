import React, { useState } from "react";
import RobotCorner from "./RobotCorner";
import RobotControls from "./RobotControls";
import CreateRobotDialog from "@/components/landing/CreateRobotDialog";
import RobotConfigDialog from "@/components/dialogs/RobotConfigDialog";
import { useRobots, type RobotMode, type ArmType } from "@/hooks/useRobots";

/** Shared setup dialogs, with distinct robot selection and session control groups. */
const RobotToolbar: React.FC<{ tourTarget?: boolean }> = ({ tourTarget = false }) => {
  const { availableNames, createRobot, refresh } = useRobots();
  const [createOpen, setCreateOpen] = useState(false);
  const [configOpen, setConfigOpen] = useState(false);
  const [configRobotName, setConfigRobotName] = useState<string | null>(null);
  // Open the Robot settings window for a robot. On close, re-fetch the shared
  // records — the window may have saved ports/cameras/torque or assigned
  // calibrations, and (unlike the old /calibration page) closing a dialog
  // doesn't remount anything that would refresh on its own.
  const openSettings = (name?: string | null) => {
    if (!name) return;
    setConfigRobotName(name);
    setConfigOpen(true);
  };

  const handleConfigOpenChange = (open: boolean) => {
    setConfigOpen(open);
    if (!open) refresh();
  };

  // Create → select (useRobots does this on success) → straight into the Robot
  // settings window so ports/calibration/cameras get configured (wireframe J1).
  const handleCreate = async (
    name: string,
    mode: RobotMode,
    armType: ArmType,
  ) => {
    const ok = await createRobot(name, mode, armType);
    if (ok) {
      setCreateOpen(false);
      openSettings(name);
    }
    return ok;
  };

  return (
    <>
      <div data-tour={tourTarget ? "launchpad-robot-corner" : undefined}>
        <RobotCorner onCreateRobot={() => setCreateOpen(true)} onOpenSettings={openSettings} />
      </div>
      <RobotControls onCreateRobot={() => setCreateOpen(true)} onOpenSettings={openSettings} />
      <CreateRobotDialog open={createOpen} onOpenChange={setCreateOpen}
        availableNames={availableNames} defaultMode="single" onCreateNew={handleCreate} />
      <RobotConfigDialog open={configOpen} onOpenChange={handleConfigOpenChange} robotName={configRobotName} />
    </>
  );
};

export default RobotToolbar;
