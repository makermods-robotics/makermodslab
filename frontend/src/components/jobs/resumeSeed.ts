import { JobRecord, jobDisplayName } from "@/lib/jobsApi";
import type { JobCheckpoint } from "@/lib/checkpointsApi";
import type { ResumeSeed } from "@/components/training/TrainingConfigurator";

/** The step a "resume from the newest checkpoint" entry point should use, or
 * null when the run saved nothing.
 *
 * Explicitly a MAX over the steps rather than `cks[cks.length - 1]`: the
 * backend happens to list ascending, but two callers relying on that unstated
 * order is how the row-level quick-resume and the card's Continue drifted
 * apart in the first place. Reading the max is order-independent and identical
 * to the old tail-read on an ascending list, so this is a hardening, not a
 * behaviour change.
 *
 * Note this deliberately does NOT honour CheckpointDropdown's step-0-means-
 * "latest" sentinel (an imported single-model checkpoint, which lerobot never
 * writes at step 0). That sentinel exists for *display and inference*; there is
 * no optimizer state to resume from at step 0, so a mixed list must resume from
 * its highest real step, not from the sentinel. A list holding only the
 * sentinel still yields 0, which the callers' own "already reached its
 * target" / resume gates handle. */
export function latestResumableStep(
  checkpoints: JobCheckpoint[],
): number | null {
  if (checkpoints.length === 0) return null;
  return checkpoints.reduce((best, c) => (c.step > best ? c.step : best), -1);
}

/**
 * Build the seed that puts the Train panel's configurator into resume mode for
 * `job` continuing from `step`.
 *
 * THE one place a resume seed is assembled. Both entry points call it — the
 * job card's step-selectable Continue / Resume-cloud and the jobs library's
 * row-level quick-resume — because they previously built the same payload
 * independently, which is exactly the shape of duplication that drifts.
 *
 * Carries the parent run's configured shape forward. The registry already
 * holds it as `job.config` (the persisted TrainingRequest), so this needs no
 * extra fetch and no reading of the checkpoint's train_config.json.
 *
 * The runner is derived from the job, not passed in. `ResumeSeed.runner` is
 * the runner the PARENT ran on — F7 lets the continuation cross to the other
 * one, and the form needs the parent's side to know whether continuing on the
 * cloud has to upload a local checkpoint first. That fact lives on the record,
 * and both call sites already gate their buttons on `job.runner`, so a caller
 * supplying it could only ever disagree by mistake; which runner the
 * continuation actually launches on stays the form's choice. `imported`
 * records are not resumable and never reach here; they map to "local" for
 * exhaustiveness.
 *
 * Scope note: this fills exactly the fields `ResumeSeed` declares today. That
 * used to stop at identity, dataset, policy, the step budget, the log/save
 * cadence and the runner/flavor — the timeout and hyperparameters were left out
 * because the seed type had nowhere to put them. `main` has since widened the
 * type (NEW-12), so they are carried here now; leaving them out would have
 * quietly regressed a shipped fix the moment the two branches met. Two of them
 * matter for different reasons: `hfJobTimeout` changes what the continuation
 * gets (omitted, the form renders blank and the runner resolves its own
 * default, capping a run already known to need longer), while the
 * hyperparameters do not change what trains — lerobot rebuilds those from the
 * checkpoint's train_config.json — but keep the form from displaying fresh-run
 * defaults for a continuation, and keep the new record's persisted config
 * truthful about the run's shape.
 */
export function buildResumeSeed(job: JobRecord, step: number): ResumeSeed {
  const parent = job.config;
  const runner = job.runner === "hf_cloud" ? "hf_cloud" : "local";
  return {
    jobId: job.id,
    step,
    name: jobDisplayName(job),
    datasetRepoId: parent.dataset_repo_id,
    policyType: parent.policy_type,
    sourceSteps: parent.steps,
    logFreq: parent.log_freq,
    saveFreq: parent.save_freq,
    runner,
    flavor: runner === "hf_cloud" ? (job.hf_flavor ?? undefined) : undefined,
    hfJobTimeout:
      runner === "hf_cloud" ? (parent.hf_job_timeout ?? undefined) : undefined,
    batchSize: parent.batch_size,
    seed: parent.seed,
    numWorkers: parent.num_workers,
    policyDevice: parent.policy_device,
    policyUseAmp: parent.policy_use_amp,
    optimizerType: parent.optimizer_type,
    optimizerLr: parent.optimizer_lr,
    optimizerWeightDecay: parent.optimizer_weight_decay,
    optimizerGradClipNorm: parent.optimizer_grad_clip_norm,
  };
}
