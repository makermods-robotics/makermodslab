import React from "react";
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
  RESUME_INHERITED_NOTE,
  RESUME_INHERITED_SHORT,
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
  const resumedFrom = !resumeLocked
    ? null
    : resumeStep
      ? `from step ${resumeStep.toLocaleString()}`
      : "from latest checkpoint";

  return (
    <section className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="steps">Training steps</Label>
          <NumberInput
            id="steps"
            value={config.steps}
            onChange={(v) => {
              if (v !== undefined) updateConfig("steps", v);
            }}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="batch_size">Batch size</Label>
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
              {RESUME_INHERITED_SHORT}
            </p>
          )}
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex items-baseline gap-2">
          <Label htmlFor="job_name">Run name</Label>
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
          placeholder={`${(config.policy_type || "policy").toUpperCase()} · ${
            config.dataset_repo_id || "dataset"
          }`}
        />
        <p className="text-xs text-muted-foreground">
          Optional — shown on the job card and searchable.
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
            {RESUME_INHERITED_NOTE}
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
          <Label htmlFor="wandb_enable">Log to Weights &amp; Biases</Label>
        </div>

        {config.wandb_enable && (
          <div className="space-y-4 border-l-2 border-border pl-4">
            <div className="space-y-2">
              <Label htmlFor="wandb_project">Project</Label>
              <Input
                id="wandb_project"
                value={config.wandb_project || ""}
                onChange={(e) =>
                  updateConfig("wandb_project", e.target.value || undefined)
                }
                placeholder="lerobot (default)"
                disabled={resumeLocked}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="wandb_entity">Entity</Label>
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
                Your W&amp;B username or a team you belong to. Blank = your
                personal account.
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="wandb_notes">W&amp;B notes (optional)</Label>
              <Input
                id="wandb_notes"
                value={config.wandb_notes || ""}
                onChange={(e) =>
                  updateConfig("wandb_notes", e.target.value || undefined)
                }
                placeholder="Training run notes..."
                disabled={resumeLocked}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="wandb_mode">W&amp;B mode</Label>
              <Select
                value={config.wandb_mode || "online"}
                onValueChange={(value) => updateConfig("wandb_mode", value)}
                disabled={resumeLocked}
              >
                <SelectTrigger id="wandb_mode">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="online">Online</SelectItem>
                  <SelectItem value="offline">Offline</SelectItem>
                  <SelectItem value="disabled">Disabled</SelectItem>
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
                Don't upload checkpoints to W&amp;B
              </Label>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default EssentialsCard;
