import React from "react";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { ConfigComponentProps, RESUME_INHERITED_SHORT } from "../types";

/** The run's headline settings — steps, batch size, and name.
 * Flat: each control carries its own <Label> and the section has no eyebrow
 * heading, so nothing sits above a single field restating it. The policy
 * select lives in PolicyField, which renders earlier in the form.
 *
 * On a resume, `steps` stays editable (the resume branch passes --steps, and
 * raising it is the whole point of a continuation) while `batch_size` does not
 * — lerobot takes it from the checkpoint's train_config.json. */
const EssentialsCard: React.FC<ConfigComponentProps> = ({
  config,
  updateConfig,
  resumeLocked,
}) => {
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
        <Label htmlFor="job_name">Run name</Label>
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
    </section>
  );
};

export default EssentialsCard;
