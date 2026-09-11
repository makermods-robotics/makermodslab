import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import {
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
import RobotLayoutChip from "@/components/launchpad/RobotLayoutChip";
import { useRobots } from "@/hooks/useRobots";
import { useArms } from "@/hooks/useArms";
import { armLabel } from "@/lib/armTypes";
import { robotLayoutReady } from "@/lib/robotSetupGap";
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

interface RobotCornerProps {
  className?: string;
  onCreateRobot: () => void;
  onOpenSettings: (name: string) => void;
}

/** Compact robot setup: add, settings, and the current robot picker. */
const RobotCorner: React.FC<RobotCornerProps> = ({ className, onCreateRobot, onOpenSettings }) => {
  const { t } = useTranslation();
  const { language } = useLanguage();
  const { byId: armById } = useArms();
  const {
    records,
    selectedName,
    selectedRecord,
    availableNames,
    isLoading,
    selectRobot,
    renameRobot,
    deleteRobot,
  } = useRobots();

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
            onClick={onCreateRobot}
            aria-label={t("robot.corner.createTooltip")}
            className="h-7 w-7 rounded-full p-0"
          >
            <Plus className="h-3.5 w-3.5" />
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
              onClick={() => selectedName && onOpenSettings(selectedName)}
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
            {isLoading && !selectedRecord ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : hasRobots && selectedRecord ? (
              <>
                <StatusDot ready={robotLayoutReady(selectedRecord)} />
                <span className="max-w-[180px] truncate sm:max-w-[280px]">
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
        <DropdownMenuContent align="end" className="w-[28rem] max-w-[calc(100vw-2rem)]">
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
                    className={cn("items-start gap-2 py-2", selected && "bg-accent")}
                  >
                    <StatusDot ready={robotLayoutReady(rec)} className="mt-1.5" />
                    <span className="min-w-0 flex-1 space-y-1">
                      <span className="block break-words font-medium">{name}</span>
                      <span className="flex flex-wrap items-center gap-2">
                        <RobotLayoutChip arms={rec.arms} />
                        <span
                          className={cn(
                            "font-mono text-[10px] text-muted-foreground",
                            isCaselessScript(language)
                              ? ""
                              : "uppercase tracking-wider",
                          )}
                        >
                          {armLabel(armById(rec.arm_type), rec.arm_type, t)}
                          {rec.arm_available === false ? (
                            <>
                              {" "}
                              <span className="rounded-sm bg-destructive/15 px-1 text-destructive">
                                {t("robot.corner.armUnavailable")}
                              </span>
                            </>
                          ) : null}
                          {" · "}
                          {rec.mode === "bimanual"
                            ? t("robot.corner.mode.bimanual")
                            : t("robot.corner.mode.single")}
                          {" · "}
                          {robotLayoutReady(rec)
                            ? t("robot.corner.status.ready")
                            : t("robot.corner.status.needsSetup")}
                        </span>
                      </span>
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
          <DropdownMenuItem onSelect={onCreateRobot} className="gap-2">
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
