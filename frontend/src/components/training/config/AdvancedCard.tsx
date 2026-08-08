import React, { useState, useEffect } from "react";
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
import { AdvancedSection } from "@/components/studio/panel/primitives";
import { cn } from "@/lib/utils";
import {
  ConfigComponentProps,
  POLICY_TYPE_OPTIONS,
  RESUME_INHERITED_NOTE,
  RESUME_INHERITED_SHORT,
} from "../types";
import { useApi } from "@/contexts/ApiContext";
import { isValidTimeout } from "@/lib/jobTimeout";

const SectionHeading: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => <h4 className="eyebrow">{children}</h4>;

interface OptimizerDefaults {
  optimizer: string;
  lr: number;
  weight_decay: number;
  grad_clip_norm: number;
}

// Render small floats readably: keep tiny/large magnitudes in exponential
// notation (1e-5, 1e-10) but show human-friendly decimals (0.01, 10) for the
// mid range, trimming any trailing zeros the browser tacks on.
const formatNum = (n: number): string => {
  if (n === 0) return "0";
  const abs = Math.abs(n);
  if (abs < 1e-3 || abs >= 1e6) {
    // toExponential(0) -> "1e-5" style; drop the "+" and leading zeros in exp.
    return n.toExponential().replace(/e\+?(-?)0*(\d)/, "e$1$2");
  }
  return String(Number(n.toPrecision(6)));
};

const OPTIMIZER_LABELS: Record<string, string> = {
  adam: "Adam",
  adamw: "AdamW",
  sgd: "SGD",
  multi_adam: "Multi Adam",
};

type OptimizerKnob = "lr" | "weight_decay" | "grad_clip_norm";

/**
 * Which optimizer knobs each policy actually exposes — the mirror of
 * `_POLICY_OPTIMIZER_FIELDS` in makermodslab/train.py. Keep the two in sync.
 *
 * Training always runs with the policy training preset on, so lerobot rebuilds
 * the optimizer from the POLICY config's own `optimizer_*` fields and discards
 * anything sent to the `--optimizer.*` namespace. The backend therefore emits
 * `--policy.optimizer_<knob>`, and only for knobs the selected policy declares
 * (an unknown one makes the CLI parser reject the run outright). A knob missing
 * here would silently do nothing, so the input is hidden rather than shown as a
 * control that can't take effect.
 *
 * This can NOT be derived from /policy-optimizer-defaults: that endpoint
 * reports the preset OBJECT's fields, and both AdamW and Adam presets carry a
 * grad_clip_norm regardless of whether the policy exposes a knob for it.
 */
const POLICY_OPTIMIZER_KNOBS: Record<string, readonly OptimizerKnob[]> = {
  act: ["lr", "weight_decay"],
  diffusion: ["lr", "weight_decay"],
  pi0: ["lr", "weight_decay", "grad_clip_norm"],
  smolvla: ["lr", "weight_decay", "grad_clip_norm"],
  tdmpc: ["lr"],
  vqbet: ["lr", "weight_decay"],
  pi0_fast: ["lr", "weight_decay", "grad_clip_norm"],
  // Its preset is a MultiAdam built from per-group settings — no scalar knobs.
  gaussian_actor: [],
};

// Matches the backend's fallback for a policy type absent from the table.
const DEFAULT_OPTIMIZER_KNOBS: readonly OptimizerKnob[] = ["lr"];

const policyShortLabel = (value: string): string =>
  POLICY_TYPE_OPTIONS.find((o) => o.value === value)?.label || value;

interface AdvancedCardProps extends ConfigComponentProps {
  /** Whether a W&B API key is resolvable on the BACKEND's host. null while the
   * credential probe is in flight (and if it failed) — the warning below is
   * gated on an explicit `false`, so an unanswered probe never claims a key is
   * missing. */
  wandbKeyAvailable?: boolean | null;
}

/** Advanced-parameters section of the training form. Uses the shared
 * AdvancedSection, so its trigger is the same eyebrow-level control as the
 * Collect form's "Advanced parameters" instead of a heavier heading that
 * outranked the sections above it.
 *
 * On a resume the first three sections (policy preset, training, optimizer) are
 * inherited wholesale from the checkpoint — build_training_command's resume
 * branch emits none of --policy.use_amp / --seed / --policy.optimizer_* (and
 * lerobot restores optimizer state from the checkpoint itself) — so they get
 * ONE section-level read-only treatment rather than a per-field repetition.
 * "Data loading", "Logging & checkpointing" and the cloud "Job timeout" below
 * stay live: --num_workers / --log_freq / --save_freq are on the resume argv,
 * and the timeout is a run_job submission parameter that never reaches lerobot
 * at all. Worker count sits outside the inherited block on purpose — it is a
 * host-capacity knob (like the cloud flavor), not part of the experiment, and a
 * continuation can land on different hardware than the parent run. */
const AdvancedCard: React.FC<AdvancedCardProps> = ({
  config,
  updateConfig,
  resumeLocked,
  wandbKeyAvailable = null,
}) => {
  const [expanded, setExpanded] = useState(false);
  const { baseUrl, fetchWithHeaders } = useApi();
  const [policyDefaults, setPolicyDefaults] = useState<
    Record<string, OptimizerDefaults | null>
  >({});

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetchWithHeaders(`${baseUrl}/policy-optimizer-defaults`);
        const data: { defaults: Record<string, OptimizerDefaults | null> } =
          await r.json();
        if (!cancelled) setPolicyDefaults(data.defaults || {});
      } catch {
        // Backend unreachable — fall back to the generic placeholders.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  const d = policyDefaults[config.policy_type] ?? null;
  const lrPlaceholder = d
    ? `${formatNum(d.lr)} (policy default)`
    : "Use policy default";
  const wdPlaceholder = d
    ? `${formatNum(d.weight_decay)} (policy default)`
    : "Use policy default";
  const gradPlaceholder = d
    ? `${formatNum(d.grad_clip_norm)} (policy default)`
    : "Use policy default";
  const defaultOptimizerLabel = d
    ? OPTIMIZER_LABELS[d.optimizer] ?? d.optimizer
    : null;

  // The optimizer CLASS is fixed by the policy preset and cannot be overridden
  // from the CLI, so it is shown, not chosen. Which scalar knobs remain
  // adjustable is likewise a property of the policy.
  const knobs =
    POLICY_OPTIMIZER_KNOBS[config.policy_type] ?? DEFAULT_OPTIMIZER_KNOBS;
  const has = (k: OptimizerKnob) => knobs.includes(k);
  const policyLabel = policyShortLabel(config.policy_type);

  // Cloud-only "Job timeout": the raw string drives both the input and the
  // (mirror-of-backend) inline validity check. Blank = the runner's
  // HF_JOB_TIMEOUT default (makermodslab/runners/hf_cloud.py).
  const isCloud = config.target.runner === "hf_cloud";
  const timeoutValue = config.hf_job_timeout ?? "";
  const timeoutInvalid =
    timeoutValue.trim() !== "" && !isValidTimeout(timeoutValue);

  return (
    <AdvancedSection
      open={expanded}
      onOpenChange={setExpanded}
      summary="Optimizer, learning rate, log frequency, checkpoints, W&B, and more"
    >
      <div className="space-y-6">
        {/* Policy preset + Training + Optimizer. On a resume these are one
            inherited block, so the explanation sits once at its head. */}
        <div
          className={cn(
            "space-y-6",
            resumeLocked && "rounded-md border border-border bg-muted/30 p-4",
          )}
        >
          {resumeLocked && (
            <p className="text-xs text-muted-foreground">
              {RESUME_INHERITED_NOTE}
            </p>
          )}
          {/* Policy */}
          <section className="space-y-3">
            <SectionHeading>Policy preset</SectionHeading>
            <div className="flex items-center gap-3">
              <Switch
                id="policy_use_amp"
                checked={config.policy_use_amp}
                onCheckedChange={(checked) =>
                  updateConfig("policy_use_amp", checked)
                }
                disabled={resumeLocked}
                className="data-[state=checked]:bg-primary"
              />
              <Label htmlFor="policy_use_amp">
                Use automatic mixed precision
              </Label>
            </div>
          </section>

          {/* Training */}
          <section className="space-y-3">
            <SectionHeading>Training</SectionHeading>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="seed">Random seed</Label>
                <NumberInput
                  id="seed"
                  value={config.seed}
                  onChange={(v) => updateConfig("seed", v)}
                  disabled={resumeLocked}
                />
              </div>
            </div>
          </section>

          {/* Optimizer */}
          <section className="space-y-3">
            <SectionHeading>Optimizer</SectionHeading>
            <div className="space-y-2">
              <Label>Optimizer</Label>
              <p className="text-sm">
                {defaultOptimizerLabel ?? "Set by the policy preset"}
              </p>
              <p className="text-xs text-muted-foreground">
                {defaultOptimizerLabel
                  ? `Set by the ${policyLabel} policy preset — the optimizer class isn't adjustable.`
                  : "The policy preset picks the optimizer class; it isn't adjustable."}
              </p>
            </div>
            {knobs.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                The {policyLabel} preset builds its optimizer from per-parameter-group
                settings, so there are no learning-rate or weight-decay knobs to set here.
              </p>
            ) : (
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                {has("lr") && (
                  <div className="space-y-2">
                    <Label htmlFor="optimizer_lr">Learning rate</Label>
                    <NumberInput
                      id="optimizer_lr"
                      integer={false}
                      step="0.0001"
                      value={config.optimizer_lr}
                      onChange={(v) => updateConfig("optimizer_lr", v)}
                      placeholder={lrPlaceholder}
                      disabled={resumeLocked}
                    />
                  </div>
                )}
                {has("weight_decay") && (
                  <div className="space-y-2">
                    <Label htmlFor="optimizer_weight_decay">Weight decay</Label>
                    <NumberInput
                      id="optimizer_weight_decay"
                      integer={false}
                      step="0.0001"
                      value={config.optimizer_weight_decay}
                      onChange={(v) => updateConfig("optimizer_weight_decay", v)}
                      placeholder={wdPlaceholder}
                      disabled={resumeLocked}
                    />
                  </div>
                )}
                {has("grad_clip_norm") && (
                  <div className="space-y-2">
                    <Label htmlFor="optimizer_grad_clip_norm">
                      Gradient clipping
                    </Label>
                    <NumberInput
                      id="optimizer_grad_clip_norm"
                      integer={false}
                      step="0.0001"
                      value={config.optimizer_grad_clip_norm}
                      onChange={(v) =>
                        updateConfig("optimizer_grad_clip_norm", v)
                      }
                      placeholder={gradPlaceholder}
                      disabled={resumeLocked}
                    />
                  </div>
                )}
              </div>
            )}
            {knobs.length > 0 && !has("grad_clip_norm") && (
              <p className="text-xs text-muted-foreground">
                The {policyLabel} policy exposes no gradient-clipping setting
                {has("weight_decay") ? "" : " or weight decay"}.
              </p>
            )}
          </section>
        </div>

        {/* Data loading. Outside the inherited block above even on a resume:
            --num_workers IS on the resume argv, and the host it runs on can
            differ from the parent run's. */}
        <section className="space-y-3">
          <SectionHeading>Data loading</SectionHeading>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="num_workers">Number of workers</Label>
              <NumberInput
                id="num_workers"
                value={config.num_workers}
                onChange={(v) => {
                  if (v !== undefined) updateConfig("num_workers", v);
                }}
              />
              <p className="text-xs text-muted-foreground">
                DataLoader processes feeding the GPU.
              </p>
            </div>
          </div>
        </section>

        {/* Logging & Checkpointing */}
        <section className="space-y-3">
          <SectionHeading>Logging &amp; checkpointing</SectionHeading>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="log_freq">Log frequency</Label>
              <NumberInput
                id="log_freq"
                value={config.log_freq}
                onChange={(v) => {
                  if (v !== undefined) updateConfig("log_freq", v);
                }}
              />
              {config.steps > 0 && config.log_freq > config.steps && (
                <p className="text-xs text-warn">
                  ⚠ Logging every {config.log_freq} steps exceeds the{" "}
                  {config.steps}-step run — no metrics will be logged.
                </p>
              )}
              <p className="text-xs text-muted-foreground">
                Steps between logged loss/lr points. Lower = higher-resolution
                charts (each point is a window average), but more log volume.
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="save_freq">Save frequency</Label>
              <NumberInput
                id="save_freq"
                value={config.save_freq}
                onChange={(v) => {
                  if (v !== undefined) updateConfig("save_freq", v);
                }}
              />
              {config.steps > 0 && config.save_freq > config.steps && (
                <p className="text-xs text-warn">
                  ⚠ Saving every {config.save_freq} steps exceeds the{" "}
                  {config.steps}-step run — no checkpoint will be saved.
                </p>
              )}
            </div>
          </div>
        </section>

        {/* Weights & Biases — sited here, directly after the log/save cadence,
            because W&B answers the same question those fields do: how loudly
            this run reports. It is NOT part of Compute — W&B works on both
            runners, and the only thing the runner changes is how the API key
            reaches the trainer, which is not the form's business.

            PLACEMENT (user decision, 2026-08-08): the upstream commit put this
            group inline and argued against a disclosure, on the grounds that
            whether a run is tracked should be visible without opening
            anything. That argument was written about the compressed panel's
            pane structure, where run settings sit behind no disclosure at all.
            It does not transfer here: on this panel EVERY run-reporting
            control — seed, cadence, AMP — lives inside Advanced, so placing
            W&B outside it would make it the anomaly rather than the peer of
            the AMP switch it was designed to be. Advanced is where the old
            panel's reporting controls live, so this is where W&B goes. */}
        <section className="space-y-3">
          <SectionHeading>Weights &amp; Biases</SectionHeading>
          <div className="flex items-center gap-3">
            <Switch
              id="wandb_enable"
              checked={config.wandb_enable}
              onCheckedChange={(checked) =>
                updateConfig("wandb_enable", checked)
              }
              disabled={resumeLocked}
              className="data-[state=checked]:bg-primary"
            />
            <Label htmlFor="wandb_enable">Log to Weights &amp; Biases</Label>
          </div>

          {config.wandb_enable && (
            <div className="space-y-4">
              {resumeLocked && (
                <p className="text-xs text-muted-foreground">
                  {RESUME_INHERITED_SHORT} A continuation re-opens the parent
                  run's W&amp;B run, so it can't log somewhere else.
                </p>
              )}

              {/* The one blocking condition, stated where it is caused. Start
                  is disabled on this too (TrainingConfigurator), so this is the
                  explanation rather than the enforcement. */}
              {wandbKeyAvailable === false && (
                <p className="text-xs text-warn">
                  ⚠ No W&amp;B API key found on this machine. Run{" "}
                  <code className="px-1 py-0.5 rounded bg-muted text-info">
                    wandb login
                  </code>{" "}
                  (or set WANDB_API_KEY) — W&amp;B can't sign in on its own from
                  a training job, so the run can't start without it.
                </p>
              )}

              <div className="grid grid-cols-2 gap-4">
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
              </div>

              {/* Shown on a resume too, read-only. These are NOT sent — the
                  resume branch emits no --wandb.mode/notes/disable_artifact —
                  but lerobot rebuilds them from the checkpoint's
                  train_config.json, which carries the parent's values. So they
                  are what the continuation actually runs with, and displaying
                  them is honest rather than decorative. (Verified against
                  lerobot v0.6.0.) */}
              <div className="grid grid-cols-2 gap-4">
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
        </section>

        {/* Cloud (HF Jobs) */}
        {isCloud && (
          <section className="space-y-3">
            <SectionHeading>Cloud</SectionHeading>
            <div className="space-y-2">
              <Label htmlFor="hf_job_timeout">Job timeout</Label>
              <Input
                id="hf_job_timeout"
                value={timeoutValue}
                onChange={(e) =>
                  updateConfig("hf_job_timeout", e.target.value)
                }
                placeholder="24h (default)"
                aria-invalid={timeoutInvalid}
                className={cn("w-32", timeoutInvalid && "border-destructive")}
              />
              {timeoutInvalid ? (
                <p className="text-xs text-destructive">
                  Use a duration like "2h", "45m", or "3h30m" (units: s, m, h,
                  d).
                </p>
              ) : (
                <p className="text-xs text-muted-foreground">
                  HF Jobs kills the run after this long. Leave blank for the
                  24h default.
                </p>
              )}
            </div>
          </section>
        )}

      </div>
    </AdvancedSection>
  );
};

export default AdvancedCard;
