import React, { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  AdvancedSection,
  useEyebrowClass,
} from "@/components/studio/panel/primitives";
import { cn } from "@/lib/utils";
import {
  ConfigComponentProps,
  POLICY_TYPE_OPTIONS,
  RESUME_INHERITED_NOTE_KEY,
} from "../types";
import { useApi } from "@/contexts/ApiContext";
import { isValidTimeout } from "@/lib/jobTimeout";

// `.eyebrow` bundles uppercase with letter-spacing; useEyebrowClass drops both
// on a caseless script, where the tracking would render visibly over-spaced.
const SectionHeading: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => <h4 className={useEyebrowClass()}>{children}</h4>;

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

/** Optimizer identifier (a backend value) → the catalog KEY of its display
 * name. Keys, not resolved copy: this is module-level, so a resolved string
 * would freeze whichever language loaded first. The names themselves are
 * algorithm names and read identically in every catalog. */
const OPTIMIZER_LABEL_KEYS: Record<string, string> = {
  adam: "training.advanced.optimizerName.adam",
  adamw: "training.advanced.optimizerName.adamw",
  sgd: "training.advanced.optimizerName.sgd",
  multi_adam: "training.advanced.optimizerName.multiAdam",
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
const AdvancedCard: React.FC<ConfigComponentProps> = ({
  config,
  updateConfig,
  resumeLocked,
}) => {
  const [expanded, setExpanded] = useState(false);
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const [policyDefaults, setPolicyDefaults] = useState<
    Record<string, OptimizerDefaults | null>
  >({});

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await fetchWithHeaders(`${baseUrl}/api/v1/policy-optimizer-defaults`);
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
  // The numbers keep formatNum's (deliberately non-locale-aware) rendering and
  // are passed in pre-formatted; only the parenthetical is copy.
  const policyDefault = (value: number) =>
    t("training.advanced.policyDefaultValue", { value: formatNum(value) });
  const lrPlaceholder = d
    ? policyDefault(d.lr)
    : t("training.advanced.usePolicyDefault");
  const wdPlaceholder = d
    ? policyDefault(d.weight_decay)
    : t("training.advanced.usePolicyDefault");
  const gradPlaceholder = d
    ? policyDefault(d.grad_clip_norm)
    : t("training.advanced.usePolicyDefault");
  // Falls back to the raw optimizer identifier for anything the table misses —
  // data, rendered verbatim.
  const defaultOptimizerLabel: string | null = d
    ? // `as never` because the map is keyed by a backend identifier, so the
      // lookup widens to `string` and loses the typed-key check.
      OPTIMIZER_LABEL_KEYS[d.optimizer]
      ? t(OPTIMIZER_LABEL_KEYS[d.optimizer] as never)
      : d.optimizer
    : null;

  // The optimizer CLASS is fixed by the policy preset and cannot be overridden
  // from the CLI, so it is shown, not chosen. Which scalar knobs remain
  // adjustable is likewise a property of the policy.
  const knobs =
    POLICY_OPTIMIZER_KNOBS[config.policy_type] ?? DEFAULT_OPTIMIZER_KNOBS;
  const has = (k: OptimizerKnob) => knobs.includes(k);
  const policyLabel = policyShortLabel(config.policy_type);

  // Cloud-only "Job timeout": the raw string drives both the input and the
  // (mirror-of-backend) inline validity check. Blank = HF Jobs default (2h).
  const isCloud = config.target.runner === "hf_cloud";
  const timeoutValue = config.hf_job_timeout ?? "";
  const timeoutInvalid =
    timeoutValue.trim() !== "" && !isValidTimeout(timeoutValue);

  return (
    <AdvancedSection
      open={expanded}
      onOpenChange={setExpanded}
      summary={t("training.advanced.summary")}
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
              {t(RESUME_INHERITED_NOTE_KEY)}
            </p>
          )}
          {/* Policy */}
          <section className="space-y-3">
            <SectionHeading>
              {t("training.advanced.sectionPolicyPreset")}
            </SectionHeading>
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
                {t("training.advanced.useAmp")}
              </Label>
            </div>
          </section>

          {/* Training */}
          <section className="space-y-3">
            <SectionHeading>{t("training.advanced.sectionTraining")}</SectionHeading>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="seed">{t("training.advanced.randomSeed")}</Label>
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
            <SectionHeading>{t("training.advanced.sectionOptimizer")}</SectionHeading>
            <div className="space-y-2">
              <Label>{t("training.advanced.optimizerLabel")}</Label>
              <p className="text-sm">
                {defaultOptimizerLabel ??
                  t("training.advanced.optimizerUnknown")}
              </p>
              <p className="text-xs text-muted-foreground">
                {defaultOptimizerLabel
                  ? t("training.advanced.optimizerFixedByPolicy", {
                      policy: policyLabel,
                    })
                  : t("training.advanced.optimizerFixedGeneric")}
              </p>
            </div>
            {knobs.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                {t("training.advanced.optimizerNoKnobs", {
                  policy: policyLabel,
                })}
              </p>
            ) : (
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                {has("lr") && (
                  <div className="space-y-2">
                    <Label htmlFor="optimizer_lr">
                      {t("training.advanced.learningRate")}
                    </Label>
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
                    <Label htmlFor="optimizer_weight_decay">
                      {t("training.advanced.weightDecay")}
                    </Label>
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
                      {t("training.advanced.gradientClipping")}
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
                {/* One complete sentence per case — English appended a
                    trailing " or weight decay" fragment, which no translation
                    can splice in the same place. */}
                {has("weight_decay")
                  ? t("training.advanced.noGradClip", { policy: policyLabel })
                  : t("training.advanced.noGradClipOrWeightDecay", {
                      policy: policyLabel,
                    })}
              </p>
            )}
          </section>
        </div>

        {/* Data loading. Outside the inherited block above even on a resume:
            --num_workers IS on the resume argv, and the host it runs on can
            differ from the parent run's. */}
        <section className="space-y-3">
          <SectionHeading>
            {t("training.advanced.sectionDataLoading")}
          </SectionHeading>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="num_workers">
                {t("training.advanced.numWorkers")}
              </Label>
              <NumberInput
                id="num_workers"
                value={config.num_workers}
                onChange={(v) => {
                  if (v !== undefined) updateConfig("num_workers", v);
                }}
              />
              <p className="text-xs text-muted-foreground">
                {t("training.advanced.numWorkersHint")}
              </p>
            </div>
          </div>
        </section>

        {/* Logging & Checkpointing */}
        <section className="space-y-3">
          <SectionHeading>
            {t("training.advanced.sectionLogging")}
          </SectionHeading>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="log_freq">{t("training.advanced.logFreq")}</Label>
              <NumberInput
                id="log_freq"
                value={config.log_freq}
                onChange={(v) => {
                  if (v !== undefined) updateConfig("log_freq", v);
                }}
              />
              {config.steps > 0 && config.log_freq > config.steps && (
                <p className="text-xs text-warn">
                  {t("training.advanced.logFreqExceeds", {
                    logFreq: config.log_freq,
                    steps: config.steps,
                  })}
                </p>
              )}
              <p className="text-xs text-muted-foreground">
                {t("training.advanced.logFreqHint")}
              </p>
            </div>
            <div className="space-y-2">
              <Label htmlFor="save_freq">{t("training.advanced.saveFreq")}</Label>
              <NumberInput
                id="save_freq"
                value={config.save_freq}
                onChange={(v) => {
                  if (v !== undefined) updateConfig("save_freq", v);
                }}
              />
              {config.steps > 0 && config.save_freq > config.steps && (
                <p className="text-xs text-warn">
                  {t("training.advanced.saveFreqExceeds", {
                    saveFreq: config.save_freq,
                    steps: config.steps,
                  })}
                </p>
              )}
            </div>
          </div>
        </section>

        {/* Cloud (HF Jobs) */}
        {isCloud && (
          <section className="space-y-3">
            <SectionHeading>{t("training.advanced.sectionCloud")}</SectionHeading>
            <div className="space-y-2">
              <Label htmlFor="hf_job_timeout">
                {t("training.advanced.jobTimeout")}
              </Label>
              <Input
                id="hf_job_timeout"
                value={timeoutValue}
                onChange={(e) =>
                  updateConfig("hf_job_timeout", e.target.value)
                }
                /* The duration literal is wire format (parseTimeoutSeconds
                   mirrors the backend regex) — only "(default)" is copy. */
                placeholder={t("training.advanced.jobTimeoutPlaceholder")}
                aria-invalid={timeoutInvalid}
                className={cn("w-32", timeoutInvalid && "border-destructive")}
              />
              {timeoutInvalid ? (
                <p className="text-xs text-destructive">
                  {t("training.advanced.jobTimeoutInvalid")}
                </p>
              ) : (
                <p className="text-xs text-muted-foreground">
                  {t("training.advanced.jobTimeoutHint")}
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
