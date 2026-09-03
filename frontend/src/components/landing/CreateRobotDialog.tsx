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
 * The hardware-family options. Same logic/display split as MODE_OPTIONS above:
 * `value` is what the form submits and the backend stores, so it stays the
 * literal "so101"/"maker"/"metal"; the label/description halves hold catalog
 * KEYS.
 *
 * `image` is the card's product photo. ArmTypePhoto retains a same-sized
 * placeholder fallback so a future hardware family can land before its photo.
 */
const ARM_TYPE_OPTIONS: {
  value: ArmType;
  labelKey: string;
  descriptionKey: string;
  image: string | null;
}[] = [
  {
    value: "so101",
    labelKey: "landing.createRobot.armTypes.so101.label",
    descriptionKey: "landing.createRobot.armTypes.so101.description",
    image: so101ArmPhoto,
  },
  {
    value: "maker",
    labelKey: "landing.createRobot.armTypes.maker.label",
    descriptionKey: "landing.createRobot.armTypes.maker.description",
    image: makerArmPhoto,
  },
  {
    value: "metal",
    labelKey: "landing.createRobot.armTypes.metal.label",
    descriptionKey: "landing.createRobot.armTypes.metal.description",
    image: metalArmPhoto,
  },
];

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
  const [newName, setNewName] = useState("");
  const [newMode, setNewMode] = useState<RobotMode>(defaultMode);
  const [newArmType, setNewArmType] = useState<ArmType>("so101");
  const [creating, setCreating] = useState(false);

  const nameExists = (name: string) =>
    availableNames.some((n) => n.toLowerCase() === name.toLowerCase());

  // Seed the form each time the dialog opens: carry a fresh typed name (if
  // any) and preseed the layout to the active filter.
  React.useEffect(() => {
    if (open) {
      const seed = (seedName ?? "").trim();
      setNewName(seed !== "" && !nameExists(seed) ? seed : "");
      setNewMode(defaultMode);
      setNewArmType("so101");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const trimmedNewName = newName.trim();
  const newNameExists = trimmedNewName !== "" && nameExists(trimmedNewName);
  const canConfirm = trimmedNewName !== "" && !newNameExists && !creating;

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
        setNewArmType("so101");
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
          setNewArmType("so101");
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
              {ARM_TYPE_OPTIONS.map((opt) => {
                const selected = newArmType === opt.value;
                const label = t(opt.labelKey as never);
                return (
                  <button
                    key={opt.value}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    onClick={() => setNewArmType(opt.value)}
                    className={cn(
                      "rounded-md border p-2 text-left transition-colors",
                      selected
                        ? "border-primary bg-accent"
                        : "border-border bg-card hover:bg-accent"
                    )}
                  >
                    <ArmTypePhoto src={opt.image} alt={label} />
                    <div className="mt-2 flex items-start justify-between gap-1">
                      <span className="text-sm font-medium leading-tight text-foreground">
                        {label}
                      </span>
                      {selected && (
                        <Check className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                      )}
                    </div>
                    <p className="mt-0.5 text-xs leading-snug text-muted-foreground">
                      {t(opt.descriptionKey as never)}
                    </p>
                  </button>
                );
              })}
            </div>
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
