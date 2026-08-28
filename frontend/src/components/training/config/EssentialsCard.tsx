import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import {
  ConfigComponentProps,
  RESUME_INHERITED_NOTE_KEY,
  RESUME_INHERITED_SHORT_KEY,
} from "../types";
import WandbInstallDialog from "../WandbInstallDialog";
import { useApi } from "@/contexts/ApiContext";

/** The run's headline settings — steps, batch size, name, and W&B logging.
 * Flat: each control carries its own <Label> and the section has no eyebrow
 * heading, so nothing sits above a single field restating it. The policy
 * select lives in PolicyField, which renders earlier in the form.
 *
 * On a resume, `steps` stays editable (the resume branch passes --steps, and
 * raising it is the whole point of a continuation) while `batch_size` does not
 * — lerobot takes it from the checkpoint's train_config.json. The whole W&B
 * group is locked for the same reason: the resume branch emits no --wandb.*, so
 * lerobot logs (or doesn't) exactly as the parent run's config said. The one
 * live effect the toggle keeps is a bad one — HfCloudJobRunner reads
 * `wandb_enable` when assembling job secrets and 400s on a missing
 * WANDB_API_KEY, so leaving it enabled could only ever block a launch without
 * turning any logging on. */
const EssentialsCard: React.FC<ConfigComponentProps> = ({
  config,
  updateConfig,
  resumeLocked,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { t } = useTranslation();
  const [wandbDialogOpen, setWandbDialogOpen] = useState(false);
  const [wandbInstallHint, setWandbInstallHint] = useState("pip install wandb");

  const handleWandbToggle = async (checked: boolean) => {
    if (!checked) {
      updateConfig("wandb_enable", false);
      return;
    }
    // Check availability before flipping the switch on. If wandb isn't
    // importable in this MakerMods Lab process, surface the same install flow used
    // for the training extra (accelerate) instead of letting the user start
    // a run that will fail.
    try {
      const r = await fetchWithHeaders(`${baseUrl}/system/wandb-extra`);
      const data: { available: boolean; install_hint: string } = await r.json();
      if (data.available) {
        updateConfig("wandb_enable", true);
      } else {
        setWandbInstallHint(data.install_hint);
        setWandbDialogOpen(true);
      }
    } catch {
      // Backend unreachable — let the user proceed; training start will
      // surface the real error if wandb is genuinely missing.
      updateConfig("wandb_enable", true);
    }
  };

  // The step this continuation starts FROM, beside the name it continues.
  // Requested here specifically: the name is what the user recognises the run
  // by, and the starting step is the one number that says which attempt this
  // is — the two belong together.
  //
  // Training steps (above) is the TARGET; this is the floor. A resume turns on
  // the gap between them, so neither number means much alone.
  //
  // Step 0 is the whole-repo/single-model sentinel, not a real training step
  // (see CheckpointDropdown) — it and a missing step both read as "latest",
  // the same word the checkpoint picker uses for it.
  const resumeStep = config.resume_from_step;
  // The step number keeps its existing (non-locale-aware) formatting and is
  // passed in pre-formatted; only the words around it are translated.
  const resumedFrom = !resumeLocked
    ? null
    : resumeStep
      ? t("training.essentials.resumedFromStep", {
          step: resumeStep.toLocaleString(),
        })
      : t("training.essentials.resumedFromLatest");

  // On a resume, `steps` is a TOTAL the run trains UP TO — it is not added to
  // the steps already done. Nothing said so at the field itself (the "from step
  // N" note sits by the run name), so "20000" read equally well as "20k more".
  // Spell it out next to the input, with the arithmetic already worked out.
  const stepsHint = !resumeLocked
    ? null
    : resumeStep
      ? config.steps > resumeStep
        ? t("training.essentials.stepsTotalHint", {
            from: resumeStep.toLocaleString(),
            remaining: (config.steps - resumeStep).toLocaleString(),
          })
        : t("training.essentials.stepsTotalTooLow", {
            from: resumeStep.toLocaleString(),
          })
      : t("training.essentials.stepsTotalHintLatest");

  return (
    <section className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="steps">
            {resumeLocked
              ? t("training.essentials.stepsTotal")
              : t("training.essentials.steps")}
          </Label>
          <NumberInput
            id="steps"
            value={config.steps}
            onChange={(v) => {
              if (v !== undefined) updateConfig("steps", v);
            }}
          />
          {stepsHint ? (
            <p
              className={
                resumeStep && config.steps <= resumeStep
                  ? "text-xs text-destructive"
                  : "text-xs text-muted-foreground"
              }
            >
              {stepsHint}
            </p>
          ) : null}
        </div>

        <div className="space-y-2">
          <Label htmlFor="batch_size">{t("training.essentials.batchSize")}</Label>
          <NumberInput
            id="batch_size"
            value={config.batch_size}
            onChange={(v) => {
              if (v !== undefined) updateConfig("batch_size", v);
            }}
            disabled={resumeLocked}
          />
          {resumeLocked && (
            <p className="text-xs text-muted-foreground">
              {t(RESUME_INHERITED_SHORT_KEY)}
            </p>
          )}
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex items-baseline gap-2">
          <Label htmlFor="job_name">{t("training.essentials.runName")}</Label>
          {resumedFrom ? (
            <span className="truncate text-xs font-normal text-muted-foreground">
              {resumedFrom}
            </span>
          ) : null}
        </div>
        <Input
          id="job_name"
          value={config.job_name || ""}
          onChange={(e) => updateConfig("job_name", e.target.value)}
          /* The policy type and dataset id are DATA (rendered verbatim); only
             the stand-ins shown before either is chosen are copy. */
          placeholder={`${(
            config.policy_type || t("training.essentials.runNamePolicyFallback")
          ).toUpperCase()} · ${
            config.dataset_repo_id ||
            t("training.essentials.runNameDatasetFallback")
          }`}
        />
        <p className="text-xs text-muted-foreground">
          {t("training.essentials.runNameHint")}
        </p>
      </div>

      {/* W&B is one inherited group on a resume, so it gets a single
          section-level treatment like Advanced's locked sections. */}
      <div
        className={cn(
          "space-y-4",
          resumeLocked && "rounded-md border border-border bg-muted/30 p-4",
        )}
      >
        {resumeLocked && (
          <p className="text-xs text-muted-foreground">
            {t(RESUME_INHERITED_NOTE_KEY)}
          </p>
        )}
        <div className="flex items-center gap-3">
          <Switch
            id="wandb_enable"
            checked={config.wandb_enable}
            onCheckedChange={handleWandbToggle}
            disabled={resumeLocked}
            className="data-[state=checked]:bg-primary"
          />
          <Label htmlFor="wandb_enable">
            {t("training.essentials.wandbEnable")}
          </Label>
        </div>

        <WandbInstallDialog
          open={wandbDialogOpen}
          onOpenChange={setWandbDialogOpen}
          installHint={wandbInstallHint}
        />

        {config.wandb_enable && (
          <div className="space-y-4 border-l-2 border-border pl-4">
            <div className="space-y-2">
              <Label htmlFor="wandb_project">
                {t("training.essentials.wandbProject")}
              </Label>
              <Input
                id="wandb_project"
                value={config.wandb_project || ""}
                onChange={(e) =>
                  updateConfig("wandb_project", e.target.value || undefined)
                }
                // Sample W&B identifiers, left English on purpose: both fields
                // are sent to W&B verbatim, and the placeholder is showing the
                // SHAPE of the value (ASCII slug), not prose.
                placeholder="my-robotics-project"
                disabled={resumeLocked}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="wandb_entity">
                {t("training.essentials.wandbEntity")}
              </Label>
              <Input
                id="wandb_entity"
                value={config.wandb_entity || ""}
                onChange={(e) =>
                  updateConfig("wandb_entity", e.target.value || undefined)
                }
                placeholder="your-username"
                disabled={resumeLocked}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="wandb_notes">
                {t("training.essentials.wandbNotes")}
              </Label>
              <Input
                id="wandb_notes"
                value={config.wandb_notes || ""}
                onChange={(e) =>
                  updateConfig("wandb_notes", e.target.value || undefined)
                }
                placeholder={t("training.essentials.wandbNotesPlaceholder")}
                disabled={resumeLocked}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="wandb_mode">{t("training.essentials.wandbMode")}</Label>
              <Select
                value={config.wandb_mode || "online"}
                onValueChange={(value) => updateConfig("wandb_mode", value)}
                disabled={resumeLocked}
              >
                <SelectTrigger id="wandb_mode">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {/* Values are the wire settings sent to wandb — only the
                      labels are copy. */}
                  <SelectItem value="online">
                    {t("training.essentials.wandbModeOnline")}
                  </SelectItem>
                  <SelectItem value="offline">
                    {t("training.essentials.wandbModeOffline")}
                  </SelectItem>
                  <SelectItem value="disabled">
                    {t("training.essentials.wandbModeDisabled")}
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center gap-3">
              <Switch
                id="wandb_disable_artifact"
                checked={config.wandb_disable_artifact}
                onCheckedChange={(checked) =>
                  updateConfig("wandb_disable_artifact", checked)
                }
                disabled={resumeLocked}
                className="data-[state=checked]:bg-primary"
              />
              <Label htmlFor="wandb_disable_artifact">
                {t("training.essentials.wandbDisableArtifact")}
              </Label>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default EssentialsCard;
