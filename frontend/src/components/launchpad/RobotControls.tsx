import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Radio, RadioTower, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { RobotActionButton } from "@/components/ui/robot-action-button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import TeleopDialog from "@/components/dialogs/TeleopDialog";
import HostingDialog from "@/components/dialogs/HostingDialog";
import StationRobotDialog from "@/components/dialogs/StationRobotDialog";
import RemoteTeleopDialog from "@/components/dialogs/RemoteTeleopDialog";
import RemoteExtraInstallDialog from "@/components/dialogs/RemoteExtraInstallDialog";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useRobots, type RobotRecord } from "@/hooks/useRobots";
import { useHostingStatus } from "@/hooks/useHostingStatus";
import { useStationStatus } from "@/hooks/useStationStatus";
import { ApiError } from "@/lib/apiClient";
import { startSession, formatSessionHeld } from "@/lib/sessionApi";
import type { HostingPhase } from "@/lib/remoteApi";
import { tabOwnerId } from "@/lib/sessionOwner";
import { formatRobotSetupGap } from "@/lib/robotSetupGap";
import { cn } from "@/lib/utils";

const HOSTING_PHASE_KEYS: Record<HostingPhase, string> = {
  parked: "robot.corner.hosting.phase.parked",
  engaging: "robot.corner.hosting.phase.engaging",
  engaged: "robot.corner.hosting.phase.engaged",
  parking: "robot.corner.hosting.phase.parking",
};

interface RobotControlsProps {
  onOpenSettings: (name: string) => void;
  onCreateRobot: () => void;
}

/** Session actions are separate from robot setup and follow the selected layout. */
const RobotControls: React.FC<RobotControlsProps> = ({ onOpenSettings, onCreateRobot }) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { selectedRecord } = useRobots();
  const [teleopStarting, setTeleopStarting] = useState(false);
  const [teleopOpen, setTeleopOpen] = useState(false);
  const [teleopSessionId, setTeleopSessionId] = useState<string | null>(null);
  const [hostOpen, setHostOpen] = useState(false);
  const [stationOpen, setStationOpen] = useState(false);
  const [remoteOpen, setRemoteOpen] = useState(false);
  const [remoteInstallOpen, setRemoteInstallOpen] = useState(false);
  const { status: hostingStatus, isError: hostingCheckFailed, refresh: refreshHosting } =
    useHostingStatus({ intervalMs: 3000 });
  const { status: stationStatus } = useStationStatus();

  // Start teleoperation through the sessions surface: the request carries the
  // robot NAME only — ports, configs, mode, right-arm fields all resolve
  // server-side from the saved record — plus this tab's owner id, which
  // attaches the lease TeleopDialog keeps renewed while it is open.
  const handleTeleop = async (robot: RobotRecord) => {
    setTeleopStarting(true);
    try {
      const { session, warnings } = await startSession(baseUrl, fetchWithHeaders, {
        kind: "teleoperation",
        robot: robot.name,
        owner: tabOwnerId(),
        options: {},
      });
      setTeleopSessionId(session.id);
      if (warnings?.length) {
        // A success can carry a warn-but-allow arm-identity finding (e.g. the
        // arm's servos hold a different saved calibration). Make it visible —
        // the warning text is backend prose, rendered verbatim.
        toast({
          title: t("robot.teleop.startedWarningTitle"),
          description: warnings.join(" "),
          duration: 10000,
        });
      } else {
        toast({
          title: t("robot.teleop.startedTitle"),
          description: t("robot.teleop.startedFallback", { name: robot.name }),
        });
      }
      setTeleopOpen(true);
    } catch (e) {
      if (e instanceof ApiError) {
        // 409 session.held renders as the shared localized "robot is busy"
        // line; every other coded refusal (robot.not_ready, hardware.*) shows
        // the server's own prose.
        toast({
          title: t("robot.teleop.failedTitle"),
          description:
            formatSessionHeld(t, e) ??
            e.detail ??
            t("robot.teleop.failedFallback"),
          variant: "destructive",
        });
      } else {
        toast({
          title: t("common.connectionError.title"),
          description: t("common.connectionError.description"),
          variant: "destructive",
        });
      }
    } finally {
      setTeleopStarting(false);
    }
  };

  // Remote teleoperation drives with the LEADER only — a record with no
  // follower is fine. Gate on leader_ready, the leader-side twin of
  // follower_ready, and diagnose the gap in the same scope.
  const remoteDisabledReason = !selectedRecord
    ? t("robot.corner.selectFirst")
    : !selectedRecord.leader_ready
      ? t("robot.remote.disabledReason", {
          name: selectedRecord.name,
          gap: formatRobotSetupGap(t, selectedRecord, "leader"),
        })
      : null;

  const teleopDisabledReason = !selectedRecord
    ? t("robot.corner.selectFirst")
    : !selectedRecord.is_clean
      ? t("robot.teleop.disabledReason", {
          name: selectedRecord.name,
          gap: formatRobotSetupGap(t, selectedRecord),
        })
      : null;

  const arms = selectedRecord?.arms ?? "both";
  const showStation = selectedRecord && arms === "follower";
  // Hosting belongs to the station; only label this selection as hosting if it matches.
  const hosting = showStation && !hostingCheckFailed && hostingStatus?.hosting_active &&
    hostingStatus.hosting?.robot === selectedRecord.name
      ? hostingStatus.hosting : null;
  const hostingPhaseLabel = hosting
    ? hosting.phase === "engaged" && hosting.active_operator
      ? t("robot.corner.hosting.engagedBy", { operator: hosting.active_operator })
      : t(HOSTING_PHASE_KEYS[hosting.phase] as never, { defaultValue: hosting.phase })
    : null;

  const selectedForHosting = showStation && stationStatus?.robot === selectedRecord.name;
  const hostingFailed = !hosting && selectedForHosting && hostingStatus?.outcome === "failed";
  const stationLabel = hostingCheckFailed
    ? t("robot.corner.station.unreachable")
    : !hostingStatus
      ? t("robot.corner.station.checking")
      : hostingFailed
        ? t("robot.corner.station.failed")
        : selectedForHosting
          ? t("robot.corner.station.waiting")
          : t("robot.corner.station.label");
  const stationHint = hostingCheckFailed
    ? t("robot.corner.station.unreachableTooltip")
    : hostingFailed
      ? hostingStatus?.hint ?? hostingStatus?.error ?? t("robot.corner.station.failedTooltip")
      : selectedForHosting
        ? t("robot.corner.station.waitingTooltip")
        : t("robot.corner.station.tooltip");

  /** One rounded secondary pill in the cluster's action position. */
  const actionButton = (
    label: string,
    icon: React.ReactNode,
    busy: boolean,
    disabledReason: string | null,
    onClick: () => void,
    tooltip: string | null,
  ) => (
    <Tooltip>
      <TooltipTrigger asChild>
        <span>
          <Button
            size="sm"
            variant="secondary"
            className="h-7 gap-1.5 rounded-full px-2.5"
            disabled={!!disabledReason || busy}
            onClick={onClick}
          >
            {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : icon}
            {label}
          </Button>
        </span>
      </TooltipTrigger>
      {(disabledReason ?? tooltip) && (
        <TooltipContent side="bottom">{disabledReason ?? tooltip}</TooltipContent>
      )}
    </Tooltip>
  );

  return (
    <div className="flex shrink-0 items-center gap-1 rounded-full border border-border bg-card p-0.5">
      {showStation && !hosting && actionButton(
        stationLabel,
        <RadioTower className={cn(
          "h-3.5 w-3.5",
          hostingCheckFailed ? "text-muted-foreground"
            : hostingFailed ? "text-destructive"
              : selectedForHosting ? "animate-pulse text-warn" : "text-muted-foreground",
        )} />,
        false,
        null,
        () => selectedForHosting || hostingCheckFailed || !stationStatus?.station_mode
          ? setHostOpen(true) : setStationOpen(true),
        stationHint,
      )}
      {hosting && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => setHostOpen(true)}
              className="h-7 gap-1.5 rounded-full px-2.5"
            >
              <RadioTower aria-hidden className="h-3.5 w-3.5 shrink-0 text-ok" />
              <span className="max-w-[220px] truncate">
                {t("robot.corner.hosting.chip")}
                {" · "}
                {hostingPhaseLabel}
              </span>
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            {t("robot.corner.hosting.tooltip", { robot: hosting.robot })}
          </TooltipContent>
        </Tooltip>
      )}

      {arms === "both" && (
        <Tooltip>
          <TooltipTrigger asChild>
            <span>
              <RobotActionButton
                action="teleoperation"
                size="sm"
                className="h-7 gap-1.5 rounded-full px-2.5 [&_svg]:size-3.5"
                disabled={!!teleopDisabledReason || teleopStarting}
                busy={teleopStarting}
                onClick={() => selectedRecord && handleTeleop(selectedRecord)}
                tooltipSide="bottom"
              >
                {t("robot.corner.teleop")}
              </RobotActionButton>
            </span>
          </TooltipTrigger>
          {teleopDisabledReason && (
            <TooltipContent side="bottom">{teleopDisabledReason}</TooltipContent>
          )}
        </Tooltip>
      )}
      {arms !== "follower" && actionButton(
        t("robot.corner.drive"),
        <Radio className="h-3.5 w-3.5" />,
        false,
        remoteDisabledReason,
        () => setRemoteOpen(true),
        t("robot.corner.remoteItemSub"),
      )}
      <TeleopDialog
        open={teleopOpen}
        onOpenChange={setTeleopOpen}
        sessionId={teleopSessionId}
      />

      <HostingDialog
        open={hostOpen}
        onOpenChange={setHostOpen}
        onChangeRobot={() => setStationOpen(true)}
      />

      <StationRobotDialog
        open={stationOpen}
        onOpenChange={(open) => {
          setStationOpen(open);
          if (!open) void refreshHosting();
        }}
        onOpenSettings={onOpenSettings}
        onCreateRobot={onCreateRobot}
      />

      <RemoteTeleopDialog
        open={remoteOpen}
        onOpenChange={setRemoteOpen}
        robot={selectedRecord ?? null}
        onInstallRequested={() => setRemoteInstallOpen(true)}
      />

      <RemoteExtraInstallDialog
        open={remoteInstallOpen}
        onOpenChange={setRemoteInstallOpen}
      />

    </div>
  );
};

export default RobotControls;
