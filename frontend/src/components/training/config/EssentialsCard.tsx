import React from "react";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { validateModelName } from "@/lib/datasetName";
import { ConfigComponentProps, RESUME_INHERITED_SHORT } from "../types";

/** The run's headline settings — steps, batch size, and the model's name.
 * Flat: each control carries its own <Label> and the section has no eyebrow
 * heading, so nothing sits above a single field restating it. The policy
 * select lives in PolicyField, which renders earlier in the form.
 *
 * On a resume, `steps` stays editable (the resume branch passes --steps, and
 * raising it is the whole point of a continuation) while `batch_size` does not
 * — lerobot takes it from the checkpoint's train_config.json. The model name is
 * inert there too, and for a stronger reason: a continuation keeps the parent
 * run's identity (a cloud resume publishes into the parent's own repo), so
 * there is no new name to choose. */
const EssentialsCard: React.FC<ConfigComponentProps> = ({
  config,
  updateConfig,
  resumeLocked,
}) => {
  const modelName = config.job_name || "";
  // null when the name is usable; a message otherwise (incl. empty). Mirrors
  // the backend's validate_model_name, so Start can't fire a run the registry
  // would refuse. Suppressed on a resume, where the field is inert.
  const nameError = resumeLocked ? null : validateModelName(modelName);

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
          <Label htmlFor="job_name">Model name *</Label>
          {resumedFrom ? (
            <span className="truncate text-xs font-normal text-muted-foreground">
              {resumedFrom}
            </span>
          ) : null}
        </div>
        <Input
          id="job_name"
          value={modelName}
          onChange={(e) => updateConfig("job_name", e.target.value)}
          placeholder="eraser_stack"
          disabled={resumeLocked}
          aria-invalid={!!modelName.trim() && nameError !== null}
          className="aria-[invalid=true]:border-destructive"
        />
        {resumeLocked ? (
          <p className="text-xs text-muted-foreground">
            {RESUME_INHERITED_SHORT}
          </p>
        ) : modelName.trim() && nameError ? (
          <p className="text-xs text-destructive">{nameError}</p>
        ) : (
          <p className="text-xs text-muted-foreground">
            This is the model's identity — it trains, saves and publishes as{" "}
            <span className="font-mono text-foreground">
              {(modelName.trim() || "name") + "_<timestamp>"}
            </span>
            . Letters, numbers, <code>.</code> <code>_</code> <code>-</code>{" "}
            only; start and end with a letter or number.
          </p>
        )}
      </div>
    </section>
  );
};

export default EssentialsCard;
