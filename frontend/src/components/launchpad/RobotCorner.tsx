import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Gamepad2,
  Plus,
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
import RobotConfigDialog from "@/components/dialogs/RobotConfigDialog";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useRobots, RobotRecord, RobotMode, ArmType } from "@/hooks/useRobots";
import { ApiError } from "@/lib/apiClient";
import { startSession, formatSessionHeld } from "@/lib/sessionApi";
import { tabOwnerId } from "@/lib/sessionOwner";
import { formatRobotSetupGap } from "@/lib/robotSetupGap";
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

  const teleopDisabledReason = !selectedRecord
    ? t("robot.corner.selectFirst")
    : !selectedRecord.is_clean
      ? t("robot.teleop.disabledReason", {
          name: selectedRecord.name,
          gap: formatRobotSetupGap(t, selectedRecord),
        })
      : null;

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
                <StatusDot ready={selectedRecord.is_clean} />
                <span className="max-w-[180px] truncate">
                  <span className="text-muted-foreground">
                    {t("robot.corner.activeLabel")}
                  </span>
                  {selectedRecord.name}
                </span>
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
                    <StatusDot ready={rec.is_clean} />
                    <span className="flex-1 truncate">{name}</span>
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
                      {rec.is_clean
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

      <Tooltip>
        <TooltipTrigger asChild>
          <span>
            <Button
              size="sm"
              variant="secondary"
              className="h-7 gap-1.5 rounded-full px-2.5"
              disabled={!!teleopDisabledReason || teleopStarting}
              onClick={() => selectedRecord && handleTeleop(selectedRecord)}
            >
              {teleopStarting ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Gamepad2 className="h-3.5 w-3.5" />
              )}
              {t("robot.corner.teleop")}
            </Button>
          </span>
        </TooltipTrigger>
        {teleopDisabledReason && (
          <TooltipContent side="bottom">{teleopDisabledReason}</TooltipContent>
        )}
      </Tooltip>

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
