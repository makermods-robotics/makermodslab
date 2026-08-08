export interface TrainingConfig {
  target: { runner: "local" | "hf_cloud"; flavor?: string };

  // Dataset configuration
  dataset_repo_id: string;

  // Policy configuration
  policy_type: string;

  // Optional user-supplied display name for the run.
  job_name?: string;

  // Core training parameters
  steps: number;
  batch_size: number;
  seed?: number;
  num_workers: number;

  // Logging and checkpointing
  log_freq: number;
  save_freq: number;
  save_checkpoint: boolean;

  // Output configuration
  resume: boolean;
  // Set by the "Continue training" flow (source run + checkpoint step).
  // `resume_from_job_id` is the LEAF being continued — the lineage edge.
  resume_from_job_id?: string;
  resume_from_step?: number;
  // Chain rewind: the ancestor whose storage holds that checkpoint, when it is
  // not the leaf's own. Provenance, not the edge.
  resume_from_checkpoint_job_id?: string;
  // Set by the "Fine-tune" flow: fresh run initialized from a source
  // checkpoint's weights (resume stays false).
  finetune_from_job_id?: string;
  finetune_from_step?: number;

  // Policy-specific parameters
  policy_device?: string;
  policy_use_amp: boolean;

  // Optimizer parameters
  optimizer_type?: string;
  optimizer_lr?: number;
  optimizer_weight_decay?: number;
  optimizer_grad_clip_norm?: number;

  // Advanced configuration
  use_policy_training_preset: boolean;

  // HF Cloud only: optional per-run override for the HF Jobs timeout, as a
  // duration string ("2h", "45m", "3h30m"). Undefined/blank ⇒ backend applies
  // its default. Ignored for local runs.
  hf_job_timeout?: string;
}

// The policy types the trainer supports. The model is chosen up-front on the
// landing page's "Create a model" card (one button per type, short `label`);
// the training config then shows it frozen using the full `display` name.
export const POLICY_TYPE_OPTIONS: {
  value: string;
  label: string;
  display: string;
}[] = [
  {
    value: "act",
    label: "ACT",
    display: "ACT (Action Chunking Transformer)",
  },
  { value: "diffusion", label: "Diffusion", display: "Diffusion Policy" },
  { value: "pi0", label: "PI0", display: "PI0" },
  { value: "smolvla", label: "SmolVLA", display: "SmolVLA" },
  { value: "tdmpc", label: "TD-MPC", display: "TD-MPC" },
  { value: "vqbet", label: "VQ-BeT", display: "VQ-BeT" },
  { value: "pi0_fast", label: "PI0 Fast", display: "PI0 Fast" },
  {
    value: "gaussian_actor",
    label: "Gaussian Actor",
    display: "Gaussian Actor",
  },
  // reward_classifier deliberately absent: it isn't a policy in the pinned
  // lerobot (separate RewardModelConfig registry — scores outcomes, doesn't
  // output actions) so lerobot-train can never construct it. Re-add alongside
  // a dedicated reward-model training pathway if that lands after a pin bump;
  // policyTypeDisplayName's fallback keeps any old records legible meanwhile.
];

// Full display name for a policy type value; falls back to the raw value so
// types coming from older job records still render something legible.
export function policyTypeDisplayName(value: string): string {
  return (
    POLICY_TYPE_OPTIONS.find((o) => o.value === value)?.display ||
    value.toUpperCase()
  );
}

// Short label for a policy type value (the picker-row form: "ACT", "SmolVLA",
// "Diffusion"…) — the same mapping as policyTypeDisplayName but the compact
// `label`. Same raw-value-uppercased fallback for unknown/older types.
export function policyTypeShortLabel(value: string): string {
  return (
    POLICY_TYPE_OPTIONS.find((o) => o.value === value)?.label ||
    value.toUpperCase()
  );
}

export interface TrainingStatus {
  training_active: boolean;
  current_step: number;
  total_steps: number;
  current_loss?: number;
  current_lr?: number;
  grad_norm?: number;
  epoch_time?: number;
  eta_seconds?: number;
  available_controls: {
    stop_training: boolean;
    pause_training: boolean;
    resume_training: boolean;
  };
}

export interface LogEntry {
  timestamp: number;
  message: string;
}

export interface ConfigComponentProps {
  config: TrainingConfig;
  updateConfig: <T extends keyof TrainingConfig>(
    key: T,
    value: TrainingConfig[T],
  ) => void;
  /** True on a RESUME entry (a Continue / Resume seed). lerobot rebuilds the
   * run's shape from the checkpoint's train_config.json, and
   * build_training_command's resume branch (makermodslab/train.py) passes it
   * only the continuation essentials — config_path, resume, output_dir, steps,
   * num_workers, log_freq, save_freq, save_checkpoint, the push-to-hub flags
   * and job_name.
   * Every other hyperparameter the form shows is silently discarded, so on a
   * resume those controls render read-only instead of pretending to matter. */
  resumeLocked?: boolean;
}

// Shown once where a card's inherited (read-only) fields begin. Deliberately
// text-only: the fine-tune flow is reached from a model's own card, not from
// here, so this points at it in prose rather than growing a button.
//
// Worded around the BEHAVIOUR ("rebuilt from the checkpoint") rather than the
// displayed value: the form does not prefill these controls from the parent
// run's persisted config, so the number beside the note is the form's default,
// not the parent's. Saying "inherited" of a control showing a default would be
// a fresh untruth — the lock's whole job is to stop the form claiming influence
// it does not have.
export const RESUME_INHERITED_NOTE =
  "Rebuilt from the parent run's checkpoint — a resume continues the same experiment, so changing these here has no effect. To train with different settings, fine-tune from this checkpoint instead.";

// The same idea for a lone locked control that sits directly under the resume
// banner (which already carries the full explanation) — repeating the whole
// sentence beside every field reads as nagging.
export const RESUME_INHERITED_SHORT =
  "Rebuilt from the parent run's checkpoint.";
