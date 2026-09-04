import React, { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { MoreHorizontal, Pencil, Trash2, Upload, Wand2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useApi } from "@/contexts/ApiContext";
import { useLanguage } from "@/contexts/LanguageContext";
import { useToast } from "@/hooks/use-toast";
import type { ArmType } from "@/hooks/useRobots";
import { isCaselessScript } from "@/i18n/config";
import { cn } from "@/lib/utils";
import ImportCalibrationButton from "./ImportCalibrationButton";

interface ConfigEntry {
  name: string;
}

interface CalibrationLibraryProps {
  /** API device vocabulary: "teleop" (leader) or "robot" (follower). */
  device: "teleop" | "robot";
  /**
   * Which calibration library to list and act on. The SO-101 pair and the
   * Maker pair keep entirely SEPARATE directories on disk, so a name that
   * exists in one is a different file (or no file) in the other — listing,
   * deleting and renaming all have to be told which one they mean.
   */
  armType: ArmType;
  /** Config name currently assigned to the selected robot (marked "in use"). */
  assignedConfig?: string;
  /** Robot record to reassign when "Use for this robot" is clicked. */
  robotName?: string;
  /**
   * Which record field "Use for this robot" assigns to. Defaults to the
   * primary field for the device (leader_config / follower_config); bimanual
   * right-arm rows pass right_leader_config / right_follower_config.
   */
  configField?: string;
  /**
   * A config currently assigned to the OTHER same-side arm. Picking it here is
   * allowed but triggers a SWAP (this slot takes it; the other slot takes this
   * slot's config) so two physical arms never share one calibration.
   */
  excludeConfig?: string;
  /**
   * The record field the `excludeConfig` config lives in (the counterpart
   * same-side slot). Set together with `excludeConfig` in bimanual mode so the
   * swap can repoint both slots in a single upsert.
   */
  excludeConfigField?: string;
  /** Called after a successful reassignment so the parent can refetch the robot. */
  onAssigned?: () => void | Promise<unknown>;
  /**
   * Called after an operation that changes the FILE LIBRARY itself (rename /
   * delete / import). Each arm row renders its own CalibrationLibrary with a
   * private config list, so without this the SIBLING instances (e.g. the other
   * same-side arm in bimanual mode) keep showing stale filenames — the parent
   * should bump `reloadToken` here to refresh every instance.
   */
  onLibraryChanged?: () => void;
  /**
   * Bump to force a re-fetch of the saved-config list — e.g. after a
   * calibration completes and may have written a brand-new named file, or a
   * sibling instance renamed/deleted/imported one (see onLibraryChanged).
   */
  reloadToken?: number;
  /**
   * Opens the caller's new-calibration flow for this arm. Passing it adds a
   * "Calibrate" segment to the control group; omitting it leaves the group as
   * picker + overflow menu.
   *
   * The action lives INSIDE this component rather than beside it because the
   * three controls share a single border — a sibling button could not join
   * that group without the parent re-implementing its seams.
   */
  onCalibrate?: () => void;
  /** The caller's calibration panel is open for this arm; presses the segment. */
  calibrateOpen?: boolean;
  calibrateDisabled?: boolean;
}

/**
 * Per-side calibration "library" as a dropdown: picking a saved config
 * assigns it to this robot's slot immediately (no separate "Use for this
 * robot" confirmation), and the selection can then be Renamed, Deleted, or
 * supplemented via Import. Delete acts on the selected config (not per
 * dropdown entry); deleting an in-use config unassigns it server-side and
 * the affected arm returns to "needs calibration".
 */
const CalibrationLibrary: React.FC<CalibrationLibraryProps> = ({
  device,
  armType,
  assignedConfig,
  robotName,
  configField,
  excludeConfig,
  excludeConfigField,
  onAssigned,
  onLibraryChanged,
  reloadToken,
  onCalibrate,
  calibrateOpen,
  calibrateDisabled,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { t } = useTranslation();
  const { language } = useLanguage();
  // `uppercase` is a no-op on caseless scripts but the tracking that rides
  // along with it is not — both are dropped together on the chips below.
  const isCJK = isCaselessScript(language);

  const [configs, setConfigs] = useState<ConfigEntry[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [assigning, setAssigning] = useState(false);
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const [renameError, setRenameError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/api/v1/calibration-configs/${device}?arm_type=${armType}`,
      );
      const data = await res.json();
      if (data.success) {
        setConfigs(
          (data.configs ?? []).map((c: { name: string }) => ({ name: c.name })),
        );
      }
    } catch {
      // Non-fatal; leave the list as-is.
    }
  }, [baseUrl, fetchWithHeaders, device, armType]);

  useEffect(() => {
    refresh();
  }, [refresh, reloadToken]);

  // Keep a valid selection: prefer the current pick, then the in-use config,
  // then the first available. Exception: when the ASSIGNMENT changes (a manual
  // or auto calibration completed under a new name and the backend repointed
  // the robot), snap the selection to it — otherwise the dropdown would keep
  // showing the previous pick with a stale "Use … for this robot" button.
  // The assignment change is only "consumed" once the new name is actually in
  // the list, so it survives fetchRobot() and the list refresh landing in
  // either order.
  const lastAssignedRef = useRef(assignedConfig);
  useEffect(() => {
    const assignedChanged = lastAssignedRef.current !== assignedConfig;
    const assignedInList =
      !!assignedConfig && configs.some((c) => c.name === assignedConfig);
    if (!assignedChanged || assignedInList) {
      lastAssignedRef.current = assignedConfig;
    }
    setSelected((prev) => {
      if (assignedChanged && assignedInList) return assignedConfig;
      if (prev && configs.some((c) => c.name === prev)) return prev;
      if (assignedInList) return assignedConfig;
      // Nothing assigned: show the placeholder. Selecting now MEANS assigning,
      // so defaulting to an arbitrary first config would read as a choice the
      // user never made.
      return null;
    });
  }, [configs, assignedConfig]);

  const confirmDelete = useCallback(async () => {
    const name = pendingDelete;
    if (!name) return;
    setPendingDelete(null);
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/api/v1/calibration-configs/${device}/${encodeURIComponent(name)}?arm_type=${armType}`,
        { method: "DELETE" },
      );
      const data = await res.json().catch(() => ({}));
      if (data.success) {
        // Robots that referenced the deleted config were unassigned
        // server-side; those arms are back to "needs calibration".
        const unassigned = (data.unassigned ?? []) as { robot: string }[];
        toast({
          title: t("calibration.library.toast.deletedTitle"),
          // Two whole sentences, one per branch — never a shared stem with a
          // clause bolted on. `count` drives the verb agreement (a real
          // i18next plural); the config name and the joined robot-name list
          // are data and interpolate verbatim.
          description: unassigned.length
            ? t("calibration.library.toast.deletedUnassigned", {
                count: unassigned.length,
                name,
                robots: unassigned
                  .map((u) => u.robot)
                  .join(t("calibration.library.toast.robotJoin")),
              })
            : t("calibration.library.toast.deleted", { name }),
        });
        setConfigs((prev) => prev.filter((c) => c.name !== name));
        if (unassigned.length) {
          // Refetch the robot so the arm's status flips to uncalibrated.
          await onAssigned?.();
        }
        // Refresh sibling arm rows' config lists (see onLibraryChanged doc).
        onLibraryChanged?.();
      } else {
        toast({
          // `data.message` is backend prose — passed through untranslated.
          title: t("calibration.library.toast.deleteFailedTitle"),
          description: data.message,
          variant: "destructive",
        });
      }
    } catch (e) {
      toast({
        title: t("calibration.library.toast.deleteFailedTitle"),
        description: String(e),
        variant: "destructive",
      });
    }
  }, [
    baseUrl,
    fetchWithHeaders,
    device,
    armType,
    pendingDelete,
    toast,
    t,
    onAssigned,
    onLibraryChanged,
  ]);

  // Assign a config to this robot's slot. Called straight from the dropdown's
  // onValueChange — picking a config IS choosing it for this robot; there is
  // no separate "Use for this robot" confirmation step.
  const assignToRobot = useCallback(
    async (name: string) => {
      if (!name || !robotName) return;
      setAssigning(true);
      try {
        const field =
          configField ??
          (device === "teleop" ? "leader_config" : "follower_config");
        // If the picked config is the one the counterpart same-side slot holds,
        // SWAP: this slot takes `name`, the counterpart takes this slot's
        // current config. One upsert of both fields — the backend's
        // config-slot-conflict guard evaluates the merged record, so a two-slot
        // swap of distinct configs passes. Otherwise a plain single-field assign.
        const isSwap =
          !!excludeConfig &&
          !!excludeConfigField &&
          name === excludeConfig &&
          excludeConfigField !== field;
        const body = isSwap
          ? {
              [field]: name,
              [excludeConfigField as string]: assignedConfig ?? "",
            }
          : { [field]: name };
        const res = await fetchWithHeaders(
          `${baseUrl}/api/v1/robots/${encodeURIComponent(robotName)}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          },
        );
        const data = await res.json().catch(() => ({}));
        if (res.ok && data.status === "success") {
          toast({
            title: isSwap
              ? t("calibration.library.toast.swappedTitle")
              : t("calibration.library.toast.assignedTitle"),
            // Config names are data. `noConfig` is a DISPLAY placeholder for
            // the empty-string value the record actually holds — the stored
            // value is unaffected by the language.
            description: isSwap
              ? t("calibration.library.toast.swapped", {
                  name,
                  previous:
                    assignedConfig || t("calibration.library.toast.noConfig"),
                })
              : t("calibration.library.toast.assigned", { name }),
          });
          await onAssigned?.();
        } else {
          toast({
            // `data.message` is backend prose — passed through untranslated.
            title: t("calibration.library.toast.assignFailedTitle"),
            description: data.message,
            variant: "destructive",
          });
        }
      } catch (e) {
        toast({
          title: t("calibration.library.toast.assignFailedTitle"),
          description: String(e),
          variant: "destructive",
        });
      } finally {
        setAssigning(false);
      }
    },
    [
      robotName,
      device,
      t,
      configField,
      assignedConfig,
      excludeConfig,
      excludeConfigField,
      baseUrl,
      fetchWithHeaders,
      toast,
      onAssigned,
    ],
  );

  const openRename = useCallback(() => {
    if (!selected) return;
    setRenameValue(selected);
    setRenameError(null);
    setRenameOpen(true);
  }, [selected]);

  const renameConfig = useCallback(async () => {
    if (!selected) return;
    const next = renameValue.trim();
    if (!next) {
      setRenameError(t("calibration.library.rename.emptyName"));
      return;
    }
    if (next === selected) {
      setRenameOpen(false);
      return;
    }
    setRenaming(true);
    setRenameError(null);
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/api/v1/calibration-configs/${device}/${encodeURIComponent(selected)}/rename?arm_type=${armType}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ new_name: next }),
        },
      );
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.success) {
        toast({
          title: t("calibration.library.toast.renamedTitle"),
          // Both are calibration file names — data, rendered verbatim.
          description: t("calibration.library.toast.renamed", {
            from: selected,
            to: data.name,
          }),
        });
        setRenameOpen(false);
        await refresh();
        setSelected(data.name);
        // A robot referencing this config was repointed server-side; refetch it.
        await onAssigned?.();
        // Sibling arm rows hold their own (now stale) config lists — tell the
        // parent so it bumps reloadToken and every instance re-fetches.
        onLibraryChanged?.();
      } else {
        // 409/400 keep the dialog open with the message for a retry. The
        // backend's message is English prose we pass through; only the
        // client-side fallback beside it is translated.
        setRenameError(data.message || t("calibration.library.rename.failed"));
      }
    } catch (e) {
      setRenameError(String(e));
    } finally {
      setRenaming(false);
    }
  }, [
    armType,
    selected,
    renameValue,
    device,
    t,
    baseUrl,
    fetchWithHeaders,
    toast,
    refresh,
    onAssigned,
    onLibraryChanged,
  ]);

  const empty = configs.length === 0;

  // Opens the import file picker from the overflow menu; the input and the
  // naming dialog stay mounted inside ImportCalibrationButton at the row root.
  const importPick = React.useRef<(() => void) | null>(null);

  return (
    <div className="mt-1 ml-6 space-y-1">
      {/* One control, not three. The picker, Calibrate and the overflow menu
          share a single border with hairline seams, so the row reads as one
          object instead of a bordered select flanked by two buttons at two
          other visual weights. Every segment is h-10, matching SelectTrigger.

          Focus rings are moved INSIDE (ring-inset, no offset): the default
          shadcn ring draws 2px outside the element, which here would overlap
          the neighbouring segment and spill past the group's own border. */}
      <div className="flex items-stretch overflow-hidden rounded-md border border-input bg-background">
        <Select
          value={selected ?? ""}
          onValueChange={(name) => {
            setSelected(name);
            // Selecting IS choosing: assign immediately, no second
            // confirmation button. Re-picking the in-use config is a no-op.
            if (robotName && name && name !== assignedConfig) {
              void assignToRobot(name);
            }
          }}
          disabled={empty || assigning}
        >
          <SelectTrigger className="min-w-0 flex-1 rounded-none border-0 bg-transparent focus:ring-1 focus:ring-inset focus:ring-offset-0">
            <SelectValue
              placeholder={
                empty
                  ? t("calibration.library.placeholderEmpty")
                  : t("calibration.library.placeholder")
              }
            />
          </SelectTrigger>
          <SelectContent>
            {configs.map((c) => {
              // The counterpart same-side slot's config stays selectable now:
              // picking it swaps the two slots' assignments (see assignToRobot).
              const usedByOtherArm =
                !!excludeConfig && c.name === excludeConfig;
              return (
                <SelectItem key={c.name} value={c.name}>
                  {/* The item VALUE stays the config's file name — it is what
                      gets submitted and stored. Only the chips beside it are
                      display text. */}
                  <span className="flex items-center gap-2">
                    {c.name}
                    {c.name === assignedConfig && (
                      <span
                        className={cn(
                          "text-[10px] text-ok border border-ok/40 rounded px-1",
                          isCJK ? "" : "uppercase tracking-wide",
                        )}
                      >
                        {t("calibration.library.inUse")}
                      </span>
                    )}
                    {usedByOtherArm && (
                      <span
                        className={cn(
                          "rounded border border-warn/40 px-1 text-[10px] text-warn",
                          isCJK ? "" : "uppercase tracking-wide",
                        )}
                      >
                        {t("calibration.library.otherArm")}
                      </span>
                    )}
                  </span>
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>

        {/* The row's main action, and the reason anyone opens this section.
            It stays a labelled segment rather than folding into the menu
            below: a bare "+" beside a file picker reads as "add a file" when
            it actually starts a calibration run. */}
        {onCalibrate && (
          <Button
            type="button"
            variant="ghost"
            onClick={onCalibrate}
            disabled={calibrateDisabled}
            aria-expanded={calibrateOpen}
            title={t("robotConfig.files.newCalibrationTitle")}
            className={cn(
              "h-10 shrink-0 gap-1.5 rounded-none border-l border-input px-3 font-normal focus-visible:ring-inset focus-visible:ring-offset-0",
              calibrateOpen && "bg-accent text-accent-foreground",
            )}
          >
            <Wand2 className="h-4 w-4" />
            {t("robotConfig.files.calibrate")}
          </Button>
        )}

        {/* Rename, import and delete are rare next to picking a config, so
            they fold into one overflow menu and the row reads select,
            calibrate, more — instead of a strip of loose icons. */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              size="icon"
              variant="ghost"
              className="h-10 w-10 shrink-0 rounded-none border-l border-input text-muted-foreground hover:text-foreground focus-visible:ring-inset focus-visible:ring-offset-0"
              aria-label={t("calibration.library.moreAria")}
            >
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem
              disabled={!selected}
              onSelect={() => openRename()}
            >
              <Pencil className="mr-2 h-4 w-4" />
              {t("calibration.library.renameTooltip")}
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => importPick.current?.()}>
              <Upload className="mr-2 h-4 w-4" />
              {t("calibration.library.importShort")}
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              disabled={!selected}
              className="text-destructive focus:text-destructive"
              onSelect={() => selected && setPendingDelete(selected)}
            >
              <Trash2 className="mr-2 h-4 w-4" />
              {t("calibration.library.deleteTooltip")}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Renders nothing: the trigger lives in the menu above. Mounted OUTSIDE
          the control group so its hidden file input is never a flex child of
          a border-seamed row. */}
      <ImportCalibrationButton
        armType={armType}
        device={device}
        pickRef={importPick}
        onImported={async (name) => {
          await refresh();
          setSelected(name);
          // Refresh sibling arm rows' config lists (see onLibraryChanged doc).
          onLibraryChanged?.();
        }}
      />

      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("calibration.library.rename.title")}</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              {t("calibration.library.rename.description")}
            </DialogDescription>
          </DialogHeader>
          <Input
            value={renameValue}
            onChange={(e) => {
              setRenameValue(e.target.value);
              setRenameError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void renameConfig();
              }
            }}
            autoFocus
            placeholder={t("calibration.library.rename.placeholder")}
          />
          {renameError && (
            <p className="text-sm text-destructive">{renameError}</p>
          )}
          <DialogFooter className="flex gap-2 justify-end">
            <Button variant="outline" onClick={() => setRenameOpen(false)}>
              {t("common.cancel")}
            </Button>
            <Button
              disabled={
                renaming ||
                !renameValue.trim() ||
                renameValue.trim() === selected
              }
              onClick={renameConfig}
            >
              {renaming
                ? t("calibration.library.rename.submitting")
                : t("calibration.library.rename.submit")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={pendingDelete !== null}
        onOpenChange={(o) => !o && setPendingDelete(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {/* The quoted name is the config's file name — data. */}
              {t("calibration.library.delete.title", {
                name: pendingDelete ?? "",
              })}
            </DialogTitle>
            <DialogDescription className="text-muted-foreground">
              {t("calibration.library.delete.description")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex gap-2 justify-end">
            <Button variant="outline" onClick={() => setPendingDelete(null)}>
              {t("common.cancel")}
            </Button>
            <Button
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={confirmDelete}
            >
              {t("calibration.library.delete.confirm")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default CalibrationLibrary;
