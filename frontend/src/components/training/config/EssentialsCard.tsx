import React from "react";
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

/** The run's headline settings — steps, batch size, name, and W&B logging.
 * Flat: each control carries its own <Label> and the section has no eyebrow
 * heading, so nothing sits above a single field restating it. The policy
 * select lives in PolicyField, which renders earlier in the form.
 *
 * On a resume, `steps` stays editable (the resume branch passes --steps, and
 * raising it is the whole point of a continuation) while `batch_size` does not
 * — lerobot takes it from the checkpoint's train_config.json. The whole W&B
 * group is locked for a stronger reason: lerobot re-opens the PARENT's W&B run
 * (`wandb.init(resume="must")` with the run id stored in the checkpoint), so a
 * continuation cannot log anywhere else. JobRegistry.start inherits
 * enable/project/entity from the parent record and ignores what the form sends;
 * the values shown are the parent's, which is what the run really uses.
 *
 * On a FRESH run the toggle defaults ON once the backend reports a resolvable
 * W&B API key (see TrainingConfigurator); with no key it stays off, and a run
 * that enables it anyway is refused at submit time with the reason. */
const EssentialsCard: React.FC<ConfigComponentProps> = ({
  config,
  updateConfig,
  resumeLocked,
}) => {
  const { t } = useTranslation();

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
            onCheckedChange={(checked) => updateConfig("wandb_enable", checked)}
            disabled={resumeLocked}
            className="data-[state=checked]:bg-primary"
          />
          <Label htmlFor="wandb_enable">
            {t("training.essentials.wandbEnable")}
          </Label>
        </div>

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
                // Left English on purpose: this names lerobot's OWN default
                // W&B project, an identifier sent verbatim, not prose.
                placeholder="lerobot (default)"
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
                placeholder="your-username or team"
                disabled={resumeLocked}
              />
              {/* The 403 trap, stated as what the field IS rather than as a
                  warning: W&B rejects a run aimed at an entity you aren't a
                  member of, and it rejects it at run start, long after Start
                  was clicked. Naming "a team you belong to" is what stops
                  someone typing a placeholder word into it. */}
              <p className="text-xs text-muted-foreground">
                {t("training.essentials.wandbEntityHint")}
              </p>
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
