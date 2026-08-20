import React, { useCallback, useEffect, useRef, useState } from "react";
import { Pencil, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
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
import { useToast } from "@/hooks/use-toast";
import ImportCalibrationButton from "./ImportCalibrationButton";

interface ConfigEntry {
  name: string;
}

interface CalibrationLibraryProps {
  /** API device vocabulary: "teleop" (leader) or "robot" (follower). */
  device: "teleop" | "robot";
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
  assignedConfig,
  robotName,
  configField,
  excludeConfig,
  excludeConfigField,
  onAssigned,
  onLibraryChanged,
  reloadToken,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

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
        `${baseUrl}/calibration-configs/${device}`,
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
  }, [baseUrl, fetchWithHeaders, device]);

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
        `${baseUrl}/calibration-configs/${device}/${encodeURIComponent(name)}`,
        { method: "DELETE" },
      );
      const data = await res.json().catch(() => ({}));
      if (data.success) {
        // Robots that referenced the deleted config were unassigned
        // server-side; those arms are back to "needs calibration".
        const unassigned = (data.unassigned ?? []) as { robot: string }[];
        toast({
          title: "Config deleted",
          description: unassigned.length
            ? `Removed "${name}". ${unassigned
                .map((u) => u.robot)
                .join(", ")} now needs calibration before use.`
            : `Removed "${name}".`,
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
          title: "Delete failed",
          description: data.message,
          variant: "destructive",
        });
      }
    } catch (e) {
      toast({
        title: "Delete failed",
        description: String(e),
        variant: "destructive",
      });
    }
  }, [baseUrl, fetchWithHeaders, device, pendingDelete, toast, onAssigned, onLibraryChanged]);

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
          ? { [field]: name, [excludeConfigField as string]: assignedConfig ?? "" }
          : { [field]: name };
        const res = await fetchWithHeaders(
          `${baseUrl}/robots/${encodeURIComponent(robotName)}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          },
        );
        const data = await res.json().catch(() => ({}));
        if (res.ok && data.status === "success") {
          toast({
            title: isSwap ? "Configs swapped" : "Config assigned",
            description: isSwap
              ? `"${name}" is now used for this arm; the other arm took "${assignedConfig || "(none)"}".`
              : `"${name}" is now used for this robot.`,
          });
          await onAssigned?.();
        } else {
          toast({
            title: "Assign failed",
            description: data.message,
            variant: "destructive",
          });
        }
      } catch (e) {
        toast({
          title: "Assign failed",
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
      setRenameError("Name cannot be empty.");
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
        `${baseUrl}/calibration-configs/${device}/${encodeURIComponent(selected)}/rename`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ new_name: next }),
        },
      );
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.success) {
        toast({
          title: "Config renamed",
          description: `"${selected}" → "${data.name}".`,
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
        // 409/400 keep the dialog open with the message for a retry.
        setRenameError(data.message || "Rename failed.");
      }
    } catch (e) {
      setRenameError(String(e));
    } finally {
      setRenaming(false);
    }
  }, [
    selected,
    renameValue,
    device,
    baseUrl,
    fetchWithHeaders,
    toast,
    refresh,
    onAssigned,
    onLibraryChanged,
  ]);

  const empty = configs.length === 0;

  return (
    <div className="mt-1 ml-6 space-y-1">
      <div className="flex items-center gap-1">
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
          <SelectTrigger className="flex-1">
            <SelectValue
              placeholder={empty ? "No saved configs" : "Select a config"}
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
                  <span className="flex items-center gap-2">
                    {c.name}
                    {c.name === assignedConfig && (
                      <span className="text-[10px] uppercase tracking-wide text-ok border border-ok/40 rounded px-1">
                        in use
                      </span>
                    )}
                    {usedByOtherArm && (
                      <span className="rounded border border-warn/40 px-1 text-[10px] uppercase tracking-wide text-warn">
                        other arm
                      </span>
                    )}
                  </span>
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>

        <Button
          size="icon"
          variant="ghost"
          className="shrink-0 text-muted-foreground hover:text-foreground"
          disabled={!selected}
          onClick={openRename}
          aria-label="Rename selected config"
          title="Rename"
        >
          <Pencil className="h-4 w-4" />
        </Button>
        <Button
          size="icon"
          variant="ghost"
          className="shrink-0 text-muted-foreground hover:text-destructive"
          disabled={!selected}
          onClick={() => selected && setPendingDelete(selected)}
          aria-label="Delete selected config"
          title="Delete"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
        <ImportCalibrationButton
          device={device}
          onImported={async (name) => {
            await refresh();
            setSelected(name);
            // Refresh sibling arm rows' config lists (see onLibraryChanged doc).
            onLibraryChanged?.();
          }}
        />
      </div>

      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename config</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              Renames the calibration file. Robots using it are updated
              automatically. Won't overwrite an existing name.
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
            placeholder="New name"
          />
          {renameError && <p className="text-sm text-destructive">{renameError}</p>}
          <DialogFooter className="flex gap-2 justify-end">
            <Button
              variant="outline"
              onClick={() => setRenameOpen(false)}
            >
              Cancel
            </Button>
            <Button
              disabled={
                renaming ||
                !renameValue.trim() ||
                renameValue.trim() === selected
              }
              onClick={renameConfig}
            >
              {renaming ? "Renaming…" : "Rename"}
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
            <DialogTitle>Delete config "{pendingDelete}"?</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              This permanently deletes the calibration file — you'd have to
              recalibrate the arm to recreate it. Any robot using it will need
              calibration before its next use.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex gap-2 justify-end">
            <Button
              variant="outline"
              onClick={() => setPendingDelete(null)}
            >
              Cancel
            </Button>
            <Button
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={confirmDelete}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default CalibrationLibrary;
