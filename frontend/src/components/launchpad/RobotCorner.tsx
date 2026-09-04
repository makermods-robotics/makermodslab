import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Gamepad2,
  Plus,
  Radio,
  RadioTower,
  Settings,
  ChevronDown,
  Loader2,
  Pencil,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import CreateRobotDialog from "@/components/landing/CreateRobotDialog";
import TeleopDialog from "@/components/dialogs/TeleopDialog";
import HostingDialog from "@/components/dialogs/HostingDialog";
import RemoteTeleopDialog from "@/components/dialogs/RemoteTeleopDialog";
import RemoteExtraInstallDialog from "@/components/dialogs/RemoteExtraInstallDialog";
import StationRobotDialog from "@/components/dialogs/StationRobotDialog";
import RobotConfigDialog from "@/components/dialogs/RobotConfigDialog";
import RobotLayoutChip from "@/components/launchpad/RobotLayoutChip";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useHostingStatus } from "@/hooks/useHostingStatus";
import { useStationStatus } from "@/hooks/useStationStatus";
import { useRobots, RobotRecord, RobotMode, ArmType } from "@/hooks/useRobots";
import { ApiError } from "@/lib/apiClient";
import { startSession, formatSessionHeld } from "@/lib/sessionApi";
import type { HostingPhase } from "@/lib/remoteApi";
import { tabOwnerId } from "@/lib/sessionOwner";
import { formatRobotSetupGap, robotLayoutReady } from "@/lib/robotSetupGap";
import { useLanguage } from "@/contexts/LanguageContext";
import { isCaselessScript } from "@/i18n/config";
import { cn } from "@/lib/utils";

/** Status dot: calibrated (ok) vs needs setup (warn ring). */
const StatusDot: React.FC<{ ready: boolean; className?: string }> = ({
  ready,
  className,
}) => (
  <span
    aria-hidden
    className={cn(
      "inline-block h-2 w-2 shrink-0 rounded-full",
      ready ? "bg-ok" : "border border-warn bg-transparent",
      className,
    )}
  />
);

// Static catalog keys per hosting phase (never a runtime-built template) so
// keyUsage.test.ts can verify each one; the phase VALUE is data.
const HOSTING_PHASE_KEYS: Record<HostingPhase, string> = {
  parked: "robot.corner.hosting.phase.parked",
  engaging: "robot.corner.hosting.phase.engaging",
  engaged: "robot.corner.hosting.phase.engaged",
  parking: "robot.corner.hosting.phase.parking",
};

/**
 * The robot corner — Layout D's always-visible robot control, one pill
 * cluster so the pieces read as a single unit: "+ Robot", an icon-only
 * Settings button, a chip with the active robot + dropdown (instant switch,
 * create, rename, delete), and a Teleop button as the rightmost segment.
 * Mounted on the Launchpad header AND inside the studio overlay header,
 * sharing state through useRobots' module-level store.
 */
const RobotCorner: React.FC<{ className?: string }> = ({ className }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { t } = useTranslation();
  const { language } = useLanguage();
  const {
    records,
    selectedName,
    selectedRecord,
    availableNames,
    isLoading,
    refresh,
    selectRobot,
    createRobot,
    renameRobot,
    deleteRobot,
  } = useRobots();

  const [createOpen, setCreateOpen] = useState(false);
  // Robot settings window (ports, calibration, cameras, motor power).
  const [configOpen, setConfigOpen] = useState(false);
  const [configRobotName, setConfigRobotName] = useState<string | null>(null);
  const [teleopStarting, setTeleopStarting] = useState(false);
  const [teleopOpen, setTeleopOpen] = useState(false);
  // Session identity from POST /api/v1/sessions — TeleopDialog heartbeats it
  // and stops it by id.
  const [teleopSessionId, setTeleopSessionId] = useState<string | null>(null);
  // Remote teleoperation, both sides. Hosting is never STARTED here — a
  // station is launched with `makermodslab --sfu --host <robot>` and hosts
  // from startup — so this corner only surfaces a live hosting session as a
  // status chip that opens HostingDialog (the status view). Driving a
  // hosting station with this robot's leader is RemoteTeleopDialog, which
  // owns its own start.
  const [hostOpen, setHostOpen] = useState(false);
  const { status: hostingStatus } = useHostingStatus();
  // Station mode: which robot this station hosts is a runtime choice
  // (PUT /api/v1/station/robot) made in StationRobotDialog — a station may
  // boot with nothing chosen. Separate from the corner's SELECTED robot,
  // which drives local flows; the two may differ.
  const [stationOpen, setStationOpen] = useState(false);
  const { status: stationStatus } = useStationStatus();
  const [remoteOpen, setRemoteOpen] = useState(false);
  const [remoteInstallOpen, setRemoteInstallOpen] = useState(false);
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);

  const openRename = () => {
    setRenameValue(selectedName ?? "");
    setRenameOpen(true);
  };

  // useRobots owns validation, API errors, and toasts for rename/delete —
  // these handlers only manage the dialogs (same split as CreateRobotDialog).
  const handleRenameConfirm = async () => {
    if (!selectedName) return;
    setRenaming(true);
    try {
      const ok = await renameRobot(selectedName, renameValue);
      if (ok) setRenameOpen(false);
    } finally {
      setRenaming(false);
    }
  };

  const handleDeleteConfirm = async () => {
    if (!selectedName) return;
    await deleteRobot(selectedName);
    setDeleteOpen(false);
  };

  const hasRobots = availableNames.length > 0;

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

  // The arm LAYOUT decides which actions exist at all. Local teleop needs a
  // pair on this machine; driving a remote robot needs a leader. An action
  // the layout makes impossible is HIDDEN rather than disabled — a station
  // has no leader to set up, so "needs its leader" would be a dead end, not
  // a hint. The primary slot (Teleop's) goes to Drive remote on a
  // leader-only controller; a follower-only station has no local action at
  // all (it is hosted from the command line — the chip above says so). The
  // Remote menu exists for a pair, where Teleop holds the primary slot.
  const arms = selectedRecord?.arms ?? "both";
  const showTeleop = arms === "both";
  const showRemote = arms !== "follower";
  const showRemoteMenu = showTeleop;

  // The live hosting session, if this station has one. The chip reads
  // "Hosting · <phase>" (or "· Engaged by <operator>" — the identity is
  // data) and opens the status view. Kept fresh by a light poll plus the
  // session_changed hint for kind `hosting`.
  const hosting =
    hostingStatus?.hosting_active && hostingStatus.hosting
      ? hostingStatus.hosting
      : null;
  // While nothing is hosted on a station: no robot chosen yet (the chip asks
  // for one), or a chosen robot whose hosting is down right now (a local
  // session has the arm, or it is re-arming — the status view says which).
  const stationIdle =
    !hosting && stationStatus?.station_mode === true ? stationStatus : null;
  const hostingPhaseLabel = hosting
    ? hosting.phase === "engaged" && hosting.active_operator
      ? t("robot.corner.hosting.engagedBy", {
          operator: hosting.active_operator,
        })
      : t(HOSTING_PHASE_KEYS[hosting.phase] as never, {
          defaultValue: hosting.phase,
        })
    : null;

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
    <div
      className={cn(
        "flex items-center gap-0.5 rounded-full border border-border bg-card p-0.5",
        className,
      )}
    >
      <Tooltip>
        <TooltipTrigger asChild>
          {/* First run (no robots yet): the very first action in the app lives
              in this cluster, and studio copy points here — render it filled
              primary so "add a robot in the top-right corner" is findable at a
              glance instead of a ghost button to hunt for. */}
          <Button
            variant={hasRobots ? "ghost" : "default"}
            size="sm"
            onClick={() => setCreateOpen(true)}
            className="h-7 gap-1.5 rounded-full px-2.5"
          >
            <Plus className="h-3.5 w-3.5" />
            {t("robot.corner.create")}
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          {t("robot.corner.createTooltip")}
        </TooltipContent>
      </Tooltip>

      <Tooltip>
        <TooltipTrigger asChild>
          <span>
            <Button
              variant="ghost"
              size="sm"
              disabled={!selectedName}
              onClick={() => openSettings(selectedName)}
              aria-label={t("robot.corner.settings")}
              className="h-7 w-7 rounded-full p-0"
            >
              <Settings className="h-3.5 w-3.5" />
            </Button>
          </span>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          {selectedName
            ? t("robot.corner.settingsFor", { name: selectedName })
            : t("robot.corner.selectFirst")}
        </TooltipContent>
      </Tooltip>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 gap-2 rounded-full px-2.5 font-medium"
          >
            {isLoading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : hasRobots && selectedRecord ? (
              <>
                <StatusDot ready={robotLayoutReady(selectedRecord)} />
                <span className="max-w-[180px] truncate">
                  <span className="text-muted-foreground">
                    {t("robot.corner.activeLabel")}
                  </span>
                  {selectedRecord.name}
                </span>
                <RobotLayoutChip arms={selectedRecord.arms} />
              </>
            ) : hasRobots ? (
              <span>{t("robot.corner.selectRobot")}</span>
            ) : (
              <>
                <Plus className="h-3.5 w-3.5" />
                <span>{t("robot.corner.setUp")}</span>
              </>
            )}
            <ChevronDown className="h-3 w-3 text-muted-foreground" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-72">
          {hasRobots ? (
            <>
              <DropdownMenuLabel className="eyebrow">
                {t("robot.corner.robots")}
              </DropdownMenuLabel>
              {availableNames.map((name) => {
                const rec = records[name];
                if (!rec) return null;
                const selected = name === selectedName;
                return (
                  <DropdownMenuItem
                    key={name}
                    onSelect={() => selectRobot(name)}
                    className={cn("gap-2", selected && "bg-accent")}
                  >
                    <StatusDot ready={robotLayoutReady(rec)} />
                    <span className="flex-1 truncate">{name}</span>
                    <RobotLayoutChip arms={rec.arms} />
                    <span
                      className={cn(
                        "font-mono text-[10px] text-muted-foreground",
                        isCaselessScript(language)
                          ? ""
                          : "uppercase tracking-wider",
                      )}
                    >
                      {t(`robot.corner.armType.${rec.arm_type ?? "so101"}`)}
                      {" · "}
                      {rec.mode === "bimanual"
                        ? t("robot.corner.mode.bimanual")
                        : t("robot.corner.mode.single")}
                      {" · "}
                      {robotLayoutReady(rec)
                        ? t("robot.corner.status.ready")
                        : t("robot.corner.status.needsSetup")}
                    </span>
                  </DropdownMenuItem>
                );
              })}
              <DropdownMenuSeparator />
            </>
          ) : (
            <DropdownMenuLabel className="text-sm font-normal text-muted-foreground">
              {t("robot.corner.empty")}
            </DropdownMenuLabel>
          )}
          <DropdownMenuItem onSelect={() => setCreateOpen(true)} className="gap-2">
            <Plus className="h-4 w-4" />
            {t("robot.corner.createItem")}
          </DropdownMenuItem>
          <DropdownMenuItem
            disabled={!selectedName}
            onSelect={openRename}
            className="gap-2"
          >
            <Pencil className="h-4 w-4" />
            {t("robot.corner.renameItem")}
          </DropdownMenuItem>
          <DropdownMenuItem
            disabled={!selectedName}
            onSelect={() => setDeleteOpen(true)}
            className="gap-2 text-destructive focus:text-destructive"
          >
            <Trash2 className="h-4 w-4" />
            {t("robot.corner.deleteItem")}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      {stationIdle && stationIdle.robot === null && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              size="sm"
              variant="default"
              onClick={() => setStationOpen(true)}
              className="h-7 gap-1.5 rounded-full px-2.5"
            >
              <RadioTower className="h-3.5 w-3.5" />
              <span className="max-w-[260px] truncate">
                {t("robot.corner.station.chooseChip")}
              </span>
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            {t("robot.corner.station.chooseTooltip")}
          </TooltipContent>
        </Tooltip>
      )}

      {stationIdle && stationIdle.robot !== null && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => setHostOpen(true)}
              className="h-7 gap-1.5 rounded-full px-2.5"
            >
              <span
                aria-hidden
                className="h-2 w-2 shrink-0 rounded-full bg-muted-foreground/60"
              />
              <span className="max-w-[220px] truncate">
                {t("robot.corner.station.idleChip", { robot: stationIdle.robot })}
              </span>
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            {t("robot.corner.station.idleTooltip", { robot: stationIdle.robot })}
          </TooltipContent>
        </Tooltip>
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
              <span
                aria-hidden
                className={cn(
                  "h-2 w-2 shrink-0 rounded-full",
                  hosting.phase === "engaged"
                    ? "animate-pulse bg-destructive"
                    : hosting.phase === "parked"
                      ? "bg-ok"
                      : "animate-pulse bg-warn",
                )}
              />
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

      {showTeleop
        ? actionButton(
            t("robot.corner.teleop"),
            <Gamepad2 className="h-3.5 w-3.5" />,
            teleopStarting,
            teleopDisabledReason,
            () => selectedRecord && handleTeleop(selectedRecord),
            null,
          )
        : arms === "leader"
          ? actionButton(
              t("robot.corner.drive"),
              <Radio className="h-3.5 w-3.5" />,
              false,
              remoteDisabledReason,
              () => setRemoteOpen(true),
              t("robot.corner.remoteItemSub"),
            )
          : null}

      {showRemoteMenu && (
        <DropdownMenu>
          <Tooltip>
            <TooltipTrigger asChild>
              <DropdownMenuTrigger asChild>
                <Button
                  size="sm"
                  variant="secondary"
                  className="h-7 gap-1.5 rounded-full px-2.5"
                  disabled={!selectedRecord}
                >
                  <Radio className="h-3.5 w-3.5" />
                  {t("robot.corner.remote")}
                  <ChevronDown className="h-3 w-3 text-muted-foreground" />
                </Button>
              </DropdownMenuTrigger>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              {selectedRecord
                ? t("robot.corner.remoteTooltip")
                : t("robot.corner.selectFirst")}
            </TooltipContent>
          </Tooltip>
          <DropdownMenuContent align="end" className="w-80">
            {showRemote && (
              <DropdownMenuItem
                disabled={!!remoteDisabledReason}
                onSelect={() => setRemoteOpen(true)}
                className="flex-col items-start gap-0.5"
              >
                <span>{t("robot.corner.remoteItem")}</span>
                <span className="text-xs text-muted-foreground">
                  {remoteDisabledReason ?? t("robot.corner.remoteItemSub")}
                </span>
              </DropdownMenuItem>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      )}

      <CreateRobotDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        availableNames={availableNames}
        defaultMode="single"
        onCreateNew={handleCreate}
      />

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
        onOpenChange={setStationOpen}
        onOpenSettings={openSettings}
        onCreateRobot={() => setCreateOpen(true)}
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

      <RobotConfigDialog
        open={configOpen}
        onOpenChange={handleConfigOpenChange}
        robotName={configRobotName}
      />

      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{t("robot.rename.title")}</DialogTitle>
            <DialogDescription>
              {t("robot.rename.description")}
            </DialogDescription>
          </DialogHeader>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleRenameConfirm();
            }}
            className="space-y-4"
          >
            <div>
              <Label htmlFor="rename-robot-name">
                {t("robot.rename.newName")}
              </Label>
              <Input
                id="rename-robot-name"
                autoFocus
                value={renameValue}
                onChange={(e) => setRenameValue(e.target.value)}
                className="mt-1"
              />
            </div>
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setRenameOpen(false)}
              >
                {t("common.cancel")}
              </Button>
              <Button type="submit" disabled={renaming || !renameValue.trim()}>
                {renaming ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />{" "}
                    {t("robot.rename.submitting")}
                  </>
                ) : (
                  t("robot.rename.submit")
                )}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      <AlertDialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("robot.delete.title", {
                name: selectedName ?? t("robot.delete.fallbackName"),
              })}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t("robot.delete.description")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteConfirm}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {t("robot.delete.confirm")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default RobotCorner;
