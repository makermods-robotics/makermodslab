import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Plus, Check, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ArmType, RobotMode } from "@/hooks/useRobots";
import { useArms } from "@/hooks/useArms";
import { cn } from "@/lib/utils";
import makerArmPhoto from "@/assets/arms/maker.jpg";
import metalArmPhoto from "@/assets/arms/metal.jpg";
import so101ArmPhoto from "@/assets/arms/so101.jpg";
import ArmTypePhoto from "./ArmTypePhoto";

interface CreateRobotDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  availableNames: string[];
  /** Layout the dialog preseeds to (mirrors the active filter). The user can
   * still change it in the dialog. */
  defaultMode: RobotMode;
  /** Optional name to seed the input with (e.g. a fresh name typed in the
   * selector's search box). */
  seedName?: string;
  onCreateNew: (
    name: string,
    mode: RobotMode,
    armType: ArmType
  ) => Promise<boolean>;
}

/**
 * The arm-layout options. `value` is LOGIC — it is what the form submits and
 * what the backend stores, so it stays the literal "single"/"bimanual". Only
 * the label/description halves are display, and they hold catalog KEYS rather
 * than resolved copy: this array evaluates at import time, so a resolved string
 * here would freeze whichever language happened to load first.
 */
const MODE_OPTIONS: {
  value: RobotMode;
  labelKey: string;
  descriptionKey: string;
}[] = [
  {
    value: "single",
    labelKey: "landing.createRobot.modes.single.label",
    descriptionKey: "landing.createRobot.modes.single.description",
  },
  {
    value: "bimanual",
    labelKey: "landing.createRobot.modes.bimanual.label",
    descriptionKey: "landing.createRobot.modes.bimanual.description",
  },
];

/**
 * The hardware-family cards come from the arms manifest (GET /api/v1/arms),
 * so a family an extension registers appears beside the built-ins with no
 * change here. The submitted `value` is the manifest id — what the backend
 * stores, verbatim. Display is the catalog's per-id label/description where
 * it has one (the built-ins), else the manifest's own label and no
 * description: manifest prose is backend text and renders in English in
 * every language, the same as any server message.
 *
 * Product photos are the one bundled asset keyed by built-in id. ArmTypePhoto
 * renders its same-sized placeholder for a family without one, so an
 * extension's arm lands before its photo does.
 */
const ARM_PHOTOS: Record<string, string> = {
  so101: so101ArmPhoto,
  maker: makerArmPhoto,
  metal: metalArmPhoto,
};

/** The card the dialog preselects: the SO-101 when the manifest has it, else
 * the first family served (the registry's default). */
const defaultArmType = (ids: string[]): ArmType =>
  ids.includes("so101") ? "so101" : (ids[0] ?? "so101");

/**
 * Name + arm-layout form for creating a new robot. Extracted from RobotSelector
 * so the same validated flow can be opened from either the selector's in-menu
 * row or a visible "New robot" button on the Landing card. useRobots owns
 * validation, API errors, and toasts; this component only manages the dialog.
 */
const CreateRobotDialog: React.FC<CreateRobotDialogProps> = ({
  open,
  onOpenChange,
  availableNames,
  defaultMode,
  seedName,
  onCreateNew,
}) => {
  const { t } = useTranslation();
  const { arms, loading: armsLoading, error: armsError } = useArms();
  // No manifest and not fetching: the load failed (ArmsProvider is retrying
  // on its own). There is nothing valid to submit, so Create is held.
  const armsFailed = arms.length === 0 && !armsLoading;
  const armIds = arms.map((a) => a.id);
  const [newName, setNewName] = useState("");
  const [newMode, setNewMode] = useState<RobotMode>(defaultMode);
  const [newArmType, setNewArmType] = useState<ArmType>(
    defaultArmType(armIds),
  );
  const [creating, setCreating] = useState(false);

  // The manifest can resolve after the dialog is already open (first load);
  // when the selection is not a card on screen, fall back to the default so
  // the form never submits an id the manifest does not list.
  React.useEffect(() => {
    if (armIds.length > 0 && !armIds.includes(newArmType)) {
      setNewArmType(defaultArmType(armIds));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [arms]);

  const nameExists = (name: string) =>
    availableNames.some((n) => n.toLowerCase() === name.toLowerCase());

  // Seed the form each time the dialog opens: carry a fresh typed name (if
  // any) and preseed the layout to the active filter.
  React.useEffect(() => {
    if (open) {
      const seed = (seedName ?? "").trim();
      setNewName(seed !== "" && !nameExists(seed) ? seed : "");
      setNewMode(defaultMode);
      setNewArmType(defaultArmType(armIds));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const trimmedNewName = newName.trim();
  const newNameExists = trimmedNewName !== "" && nameExists(trimmedNewName);
  const canConfirm =
    trimmedNewName !== "" && !newNameExists && !creating && arms.length > 0;

  const handleCreateConfirm = async () => {
    if (!canConfirm) return;
    setCreating(true);
    try {
      // useRobots handles validation, API errors, and toasts; on success it
      // also selects the new robot. We only manage the dialog here.
      const ok = await onCreateNew(trimmedNewName, newMode, newArmType);
      if (ok) {
        onOpenChange(false);
        setNewName("");
        setNewMode(defaultMode);
        setNewArmType(defaultArmType(armIds));
      }
    } finally {
      setCreating(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        onOpenChange(o);
        if (!o) {
          setNewName("");
          setNewMode(defaultMode);
          setNewArmType(defaultArmType(armIds));
        }
      }}
    >
      <DialogContent className="bg-popover border-border sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>{t("landing.createRobot.title")}</DialogTitle>
          <DialogDescription className="text-muted-foreground">
            {t("landing.createRobot.description")}
          </DialogDescription>
        </DialogHeader>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleCreateConfirm();
          }}
          className="space-y-4"
        >
          <div>
            <Label htmlFor="new-robot-name" className="text-foreground">
              {t("landing.createRobot.nameLabel")}
            </Label>
            <Input
              id="new-robot-name"
              autoFocus
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="my_robot"
              aria-invalid={newNameExists}
              className="mt-1 aria-[invalid=true]:border-destructive"
            />
            {newNameExists && (
              <p className="mt-1 text-xs text-destructive">
                {t("landing.createRobot.duplicate")}
              </p>
            )}
          </div>
          <div>
            <Label className="text-foreground">
              {t("landing.createRobot.armTypeLabel")}
            </Label>
            <div
              role="radiogroup"
              aria-label={t("landing.createRobot.armTypeLabel")}
              // Three across at EVERY width: the cards are small, and letting
              // them stack turns each 4:3 photo into a full-width block that
              // overflows the dialog past the viewport.
              className="mt-1 grid grid-cols-3 gap-2"
            >
              {arms.map((info) => {
                const selected = newArmType === info.id;
                // Runtime-built keys (the id is data), so `as never`; the
                // manifest's label is the default for an id the catalog does
                // not know, and such an id has no description of its own.
                const label = t(
                  `landing.createRobot.armTypes.${info.id}.label` as never,
                  { defaultValue: info.label },
                );
                const description = t(
                  `landing.createRobot.armTypes.${info.id}.description` as never,
                  { defaultValue: "" },
                );
                return (
                  <button
                    key={info.id}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    onClick={() => setNewArmType(info.id)}
                    className={cn(
                      "rounded-md border p-2 text-left transition-colors",
                      selected
                        ? "border-primary bg-accent"
                        : "border-border bg-card hover:bg-accent"
                    )}
                  >
                    <ArmTypePhoto src={ARM_PHOTOS[info.id] ?? null} alt={label} />
                    <div className="mt-2 flex items-start justify-between gap-1">
                      <span className="text-sm font-medium leading-tight text-foreground">
                        {label}
                      </span>
                      {selected && (
                        <Check className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                      )}
                    </div>
                    {description && (
                      <p className="mt-0.5 text-xs leading-snug text-muted-foreground">
                        {description}
                      </p>
                    )}
                    {info.provided_by !== "builtin" && (
                      <p className="mt-0.5 text-[10px] leading-snug text-muted-foreground">
                        {t("landing.createRobot.providedBy", {
                          extension: info.provided_by,
                        })}
                      </p>
                    )}
                  </button>
                );
              })}
            </div>
            {/* Nothing to pick yet: the manifest has not answered. The grid
                above is simply empty meanwhile — no card, no crash. */}
            {arms.length === 0 && armsLoading && (
              <p className="mt-1 text-xs text-muted-foreground">
                {t("landing.createRobot.armTypesLoading")}
              </p>
            )}
            {armsFailed && (
              <p className="mt-1 text-xs text-destructive">
                {t("landing.createRobot.armTypesFailed")}
                {/* Server / network prose, rendered verbatim after our line. */}
                {armsError ? ` (${armsError})` : ""}
              </p>
            )}
          </div>
          <div>
            <Label className="text-foreground">
              {t("landing.createRobot.armLayout")}
            </Label>
            <div
              role="radiogroup"
              aria-label={t("landing.createRobot.armLayout")}
              className="mt-1 grid grid-cols-2 gap-2"
            >
              {MODE_OPTIONS.map((opt) => {
                const selected = newMode === opt.value;
                return (
                  <button
                    key={opt.value}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    onClick={() => setNewMode(opt.value)}
                    className={cn(
                      "rounded-md border px-3 py-2 text-left transition-colors",
                      selected
                        ? "border-primary bg-accent"
                        : "border-border bg-card hover:bg-accent"
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-foreground">
                        {t(opt.labelKey as never)}
                      </span>
                      {selected && <Check className="h-4 w-4 text-primary" />}
                    </div>
                    <p className="mt-0.5 text-xs text-muted-foreground">
                      {t(opt.descriptionKey as never)}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
            >
              {t("common.cancel")}
            </Button>
            <Button type="submit" disabled={!canConfirm}>
              {creating ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />{" "}
                  {t("landing.createRobot.submitting")}
                </>
              ) : (
                <>
                  <Plus className="w-4 h-4 mr-2" />{" "}
                  {t("landing.createRobot.submit")}
                </>
              )}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default CreateRobotDialog;
