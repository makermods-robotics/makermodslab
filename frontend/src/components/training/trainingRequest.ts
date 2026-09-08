import { TrainingConfig } from "@/components/training/types";
import { TrainingRequest } from "@/lib/jobsApi";

/** Which local→cloud transfer a launch needs, if any. "resume" moves the whole
 * checkpoint of the run being continued (weights AND optimizer state);
 * "finetune" moves only the base checkpoint's weights, since a fine-tune starts
 * a fresh optimizer and never reads the rest. */
export type CheckpointUploadKind = "resume" | "finetune" | null;

/**
 * Map the training form's config to the backend's TrainingRequest. Pure and
 * extracted from TrainingConfigurator so the combine-mode override
 * (`dataset_repo_id` swapped for the merged output) is testable without
 * rendering the form.
 */
export function configToRequest(
  c: TrainingConfig,
  checkpointUploadKind: CheckpointUploadKind,
): TrainingRequest {
  // The backend's TrainingRequest has more optional fields; the form covers
  // the user-meaningful subset.
  return {
    target: c.target,
    dataset_repo_id: c.dataset_repo_id,
    dataset_episodes: c.dataset_episodes,
    policy_type: c.policy_type,
    job_name: c.job_name,
    steps: c.steps,
    batch_size: c.batch_size,
    seed: c.seed,
    num_workers: c.num_workers,
    log_freq: c.log_freq,
    save_freq: c.save_freq,
    save_checkpoint: c.save_checkpoint,
    resume: c.resume,
    resume_from_job_id: c.resume_from_job_id,
    resume_from_step: c.resume_from_step,
    resume_from_checkpoint_job_id: c.resume_from_checkpoint_job_id,
    // Consent, not a mode: sent only for the combinations that have to push
    // bytes to the Hub (a checkpoint only this machine has, needed by a run on
    // cloud compute), and the backend refuses those without it. Left undefined
    // otherwise so no other launch carries an upload permission it has no use
    // for. One field per MODE rather than one shared flag, because they consent
    // to different disclosures: the whole checkpoint of the run being continued,
    // versus the base model's weights.
    upload_resume_checkpoint:
      checkpointUploadKind === "resume" ? true : undefined,
    upload_finetune_checkpoint:
      checkpointUploadKind === "finetune" ? true : undefined,
    finetune_from_job_id: c.finetune_from_job_id,
    finetune_from_step: c.finetune_from_step,
    wandb_enable: c.wandb_enable,
    wandb_project: c.wandb_project,
    wandb_entity: c.wandb_entity,
    wandb_notes: c.wandb_notes,
    wandb_mode: c.wandb_mode,
    wandb_disable_artifact: c.wandb_disable_artifact,
    policy_device: c.policy_device,
    policy_use_amp: c.policy_use_amp,
    optimizer_type: c.optimizer_type,
    optimizer_lr: c.optimizer_lr,
    optimizer_weight_decay: c.optimizer_weight_decay,
    optimizer_grad_clip_norm: c.optimizer_grad_clip_norm,
    use_policy_training_preset: c.use_policy_training_preset,
    // Cloud-only; the backend validates the format and ignores it for local.
    // Send only a non-blank value so a stray "" doesn't reach the validator.
    hf_job_timeout:
      c.target.runner === "hf_cloud" && c.hf_job_timeout?.trim()
        ? c.hf_job_timeout.trim()
        : undefined,
  };
}
