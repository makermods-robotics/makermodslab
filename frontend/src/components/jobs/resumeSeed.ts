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
 * "latest" sentinel (a whole-repo/flat model with no step tree). That sentinel
 * exists for *display and inference*; there is no optimizer state to resume
 * from at step 0, so a mixed list must resume from its highest real step, not
 * from the sentinel. A list holding only the sentinel still yields 0, which the
 * callers' own "already reached its target"/canResumeEntry gates handle. */
export function latestResumableStep(checkpoints: JobCheckpoint[]): number | null {
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
 * independently and had already drifted: the row version omitted the parent's
 * HF Jobs timeout (silently capping a cloud continuation at the runner's 2h
 * default, the NEW-12 bug the card version was fixed for), the worker count,
 * and every hyperparameter the form displays read-only. Unifying means the row
 * path now carries the same full parent shape the card path does.
 *
 * Carries the parent run's whole configured shape forward. The registry
 * already holds it as `job.config` (the persisted TrainingRequest), so this
 * needs no extra fetch and no reading of the checkpoint's train_config.json.
 * The configurator PREFILLS from these, then renders read-only the ones
 * lerobot rebuilds from the checkpoint anyway (batch size, seed, device,
 * optimizer). Steps, the log/save cadence, the worker count, hardware and the
 * timeout stay editable — those a continuation can really change.
 *
 * The runner is derived from the job, not passed in: it is where the parent
 * RAN, which is both the form's default Compute and (since a continuation may
 * now cross runners — F7) the fact that decides whether the parent's checkpoint
 * has to be moved first. Both call sites already read `job.runner`, so a caller
 * supplying it could only ever disagree by mistake. `imported` records are not
 * resumable and never reach here; they map to "local" for exhaustiveness.
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
    // Cloud-only: without this a Continue fell back to the runner's 2h
    // default, capping the tail of a run already known to need longer.
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
