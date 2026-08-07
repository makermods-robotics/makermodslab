import { ApiError, Fetcher, apiRequest } from "./apiClient";

export type JobState = "running" | "done" | "failed" | "interrupted";

export interface TrainingMetrics {
  current_step: number;
  total_steps: number;
  current_loss: number | null;
  current_lr: number | null;
  grad_norm: number | null;
  eta_seconds: number | null;
}

export interface LogLine {
  timestamp: number;
  message: string;
}

export type MetricsHistoryPoint = {
  step: number;
  loss: number | null;
  lr: number | null;
  grad_norm: number | null;
};

// Mirror of the backend TrainingRequest. The frontend doesn't send all of
// these; defaults on the server fill in the rest.
export interface TrainingRequest {
  dataset_repo_id: string;
  policy_type: string;
  // Optional user-supplied display name; blank ⇒ backend auto-names the run.
  job_name?: string;
  steps: number;
  batch_size: number;
  seed?: number;
  num_workers: number;
  log_freq: number;
  save_freq: number;
  save_checkpoint: boolean;
  resume: boolean;
  // Set by the "Continue training" flow: source run + checkpoint step to
  // resume from. The backend resolves these into the checkpoint's config_path.
  resume_from_job_id?: string;
  resume_from_step?: number;
  // Consent for the one resume shape that has to publish something: continuing
  // a LOCAL run on cloud compute uploads its checkpoint to a private Hub repo
  // first, because the pod can't read this machine's disk (F7). The backend
  // refuses that combination unless this is true, so it can never be a silent
  // side effect of Continue.
  upload_resume_checkpoint?: boolean;
  // Set by the "Fine-tune" flow: start a fresh run whose weights are init'd
  // from an imported/existing checkpoint. The backend resolves these into
  // policy_pretrained_path (which it also accepts directly).
  finetune_from_job_id?: string;
  finetune_from_step?: number;
  policy_pretrained_path?: string;
  wandb_enable: boolean;
  wandb_project?: string;
  wandb_entity?: string;
  wandb_notes?: string;
  wandb_mode?: string;
  wandb_disable_artifact: boolean;
  policy_device?: string;
  policy_use_amp: boolean;
  optimizer_type?: string;
  optimizer_lr?: number;
  optimizer_weight_decay?: number;
  optimizer_grad_clip_norm?: number;
  use_policy_training_preset: boolean;
  // Optional target for runner dispatch; omitted ⇒ local.
  target?: { runner: "local" | "hf_cloud"; flavor?: string };
  // HF Cloud only: optional override for the HF Jobs timeout, as a duration
  // string ("2h", "45m", "3h30m"). Omitted ⇒ backend falls back to its
  // default. The backend validates the format and ignores it for local runs.
  hf_job_timeout?: string;
}

export interface JobRecord {
  id: string;
  name: string;
  // User-set display alias (rename is metadata-only; the id / output dir /
  // hub repo id never change). Null/absent ⇒ show `name`.
  display_name?: string | null;
  state: JobState;
  config: TrainingRequest;
  output_dir: string;
  started_at: number;
  ended_at: number | null;
  exit_code: number | null;
  error_message: string | null;
  metrics: TrainingMetrics;
  runner: "local" | "hf_cloud" | "imported";
  hf_job_id: string | null;
  hf_flavor: string | null;
  hf_repo_id: string | null;
  hf_job_url: string | null;
  wandb_run_url: string | null;
  checkpoint_count: number;
}

// Per-running-job snapshot pushed by the watchdog over WS at ~1Hz. Subset
// of JobRecord — just the fields that change during a running tick.
export interface JobProgressSnapshot {
  id: string;
  state: JobState;
  metrics: TrainingMetrics;
  wandb_run_url: string | null;
  checkpoint_count: number;
}

export async function listJobs(
  baseUrl: string,
  fetcher: Fetcher,
  limit = 10,
  signal?: AbortSignal,
): Promise<JobRecord[]> {
  const body = await apiRequest<{ jobs: JobRecord[] }>(
    baseUrl,
    fetcher,
    `/jobs?limit=${limit}`,
    { signal, action: "List jobs" },
  );
  return body.jobs;
}

export async function getJob(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
  signal?: AbortSignal,
): Promise<JobRecord> {
  return apiRequest<JobRecord>(baseUrl, fetcher, `/jobs/${id}`, {
    signal,
    action: "Get job",
  });
}

export async function getJobLogs(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
  signal?: AbortSignal,
): Promise<LogLine[]> {
  const body = await apiRequest<{ logs: LogLine[] }>(
    baseUrl,
    fetcher,
    `/jobs/${id}/logs`,
    { signal, action: "Get job logs" },
  );
  return body.logs;
}

export async function getJobLogFile(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
  signal?: AbortSignal,
): Promise<LogLine[]> {
  const body = await apiRequest<{ logs: LogLine[] }>(
    baseUrl,
    fetcher,
    `/jobs/${id}/log-file`,
    { signal, action: "Get job log file" },
  );
  return body.logs;
}

export async function getJobMetricsHistory(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
  signal?: AbortSignal,
): Promise<MetricsHistoryPoint[]> {
  const body = await apiRequest<{ points: MetricsHistoryPoint[] }>(
    baseUrl,
    fetcher,
    `/jobs/${id}/metrics-history`,
    { signal, action: "Get job metrics history" },
  );
  return body.points;
}

export async function startTrainingJob(
  baseUrl: string,
  fetcher: Fetcher,
  request: TrainingRequest,
): Promise<JobRecord> {
  const { target, ...config } = request;
  const body = target ? { config, target } : config;
  try {
    return await apiRequest<JobRecord>(baseUrl, fetcher, "/jobs/training", {
      method: "POST",
      body,
      action: "Start training",
    });
  } catch (e) {
    if (e instanceof ApiError && e.status === 409) {
      throw new Error("Another training is already running. Stop it first.");
    }
    throw e;
  }
}

// Importing an already-registered source is idempotent: the backend returns
// the EXISTING record with `already_imported: true` instead of a new entry.
export type ImportModelResult = JobRecord & { already_imported?: boolean };

export async function importModel(
  baseUrl: string,
  fetcher: Fetcher,
  source: string,
  name?: string,
): Promise<ImportModelResult> {
  return apiRequest<ImportModelResult>(baseUrl, fetcher, "/jobs/import", {
    method: "POST",
    body: name ? { source, name } : { source },
    action: "Import model",
  });
}

/** The name to display for a job: the user's alias, falling back to the
 * original (auto-generated or import-time) name. */
export function jobDisplayName(job: JobRecord): string {
  return job.display_name?.trim() || job.name;
}

/**
 * What each job state is CALLED in the UI.
 *
 * The wire values are the backend's and never change — `interrupted` is what
 * lands in job.json and what /jobs returns — so this maps them to the words a
 * person reads. Three are the state name capitalised; the fourth is the reason
 * this map exists. A run in `interrupted` is one the user pressed **Stop** on,
 * so it reads "Stopped": the plainer word, and the same one as the button that
 * produced the state. "Interrupted" suggests something happened TO the run.
 *
 * Lives here, beside JobState and jobDisplayName, rather than in whichever
 * component renders a badge. The badges keep their own colour and icon —
 * those legitimately differ — but the WORDS have to agree, and they did not:
 * the card said "Interrupted" while the monitor dialog showed the raw wire
 * string for the same run.
 */
export const JOB_STATE_LABELS: Record<JobState, string> = {
  running: "Running",
  done: "Done",
  failed: "Failed",
  interrupted: "Stopped",
};

/** `JOB_STATE_LABELS` for a state that may not be one — a state added by a
 * newer backend than this bundle falls back to the raw wire word rather than
 * rendering blank. */
export function jobStateLabel(state: JobState): string {
  return JOB_STATE_LABELS[state] ?? state;
}

/** Set a job's display alias. Metadata-only — the job id, output dir, and
 * hub repo id are immutable identity and never change on rename. */
export async function renameJob(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
  newName: string,
): Promise<JobRecord> {
  return apiRequest<JobRecord>(baseUrl, fetcher, `/jobs/${id}/rename`, {
    method: "POST",
    body: { new_name: newName },
    action: "Rename job",
  });
}

export async function stopJob(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
): Promise<JobRecord> {
  return apiRequest<JobRecord>(baseUrl, fetcher, `/jobs/${id}/stop`, {
    method: "POST",
    action: "Stop job",
  });
}

export async function deleteJob(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
): Promise<void> {
  await apiRequest<void>(baseUrl, fetcher, `/jobs/${id}`, {
    method: "DELETE",
    action: "Delete job",
  });
}

export interface RunnerFlavor {
  name: string;
  pretty_name: string;
  cpu: string;
  ram: string;
  accelerator: string | null;
  unit_cost_usd: number;
  unit_label: string;
}

export interface RunnerHardwareResponse {
  authenticated: boolean;
  username: string | null;
  flavors: RunnerFlavor[];
  // True when the backend is in HF_HUB_OFFLINE mode: every Hub write is
  // disabled, so a local-only dataset can't be uploaded for a cloud run. The
  // training page uses this to keep Start disabled + explain why. Absent on
  // older backends → treated as online (false).
  offline?: boolean;
}

const EMPTY_HARDWARE: RunnerHardwareResponse = {
  authenticated: false,
  username: null,
  flavors: [],
  offline: false,
};

export async function listRunnerHardware(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<RunnerHardwareResponse> {
  // Backend returns 401/403 for unauthenticated users; surface as "no flavors"
  // rather than throwing so the UI can render the "log in to use cloud" hint.
  try {
    return await apiRequest<RunnerHardwareResponse>(
      baseUrl,
      fetcher,
      "/jobs/runners/hardware",
      { signal, action: "List runner hardware" },
    );
  } catch (e) {
    if (e instanceof ApiError) return EMPTY_HARDWARE;
    throw e;
  }
}

export interface HubJob {
  id: string;
  created_at: string | null;
  docker_image: string | null;
  space_id: string | null;
  flavor: string | null;
  status: { stage: string; message: string | null } | null;
  owner: string | null;
  url: string;
}

// Hub stages still doing work. Anything outside this set (COMPLETED, FAILED,
// CANCELED, …) is a terminal leftover — demoted to UNTRACKED and dismissible.
// Mirrors _HUB_ACTIVE_STAGES on the backend.
export const HUB_ACTIVE_STAGES = new Set(["RUNNING", "QUEUED", "SCHEDULING"]);

export const isHubJobActive = (job: HubJob): boolean =>
  HUB_ACTIVE_STAGES.has((job.status?.stage ?? "").toUpperCase());

export interface HubModel {
  repo_id: string;
  last_modified: string | null;
  private: boolean;
}

export interface HubJobsResponse {
  authenticated: boolean;
  // False when the token is valid but lacks the job.read scope (jobs can't be
  // listed). Absent/true otherwise. Only meaningful when authenticated.
  jobs_permission?: boolean;
  jobs: HubJob[];
  models: HubModel[];
}

const EMPTY_HUB: HubJobsResponse = {
  authenticated: false,
  jobs: [],
  models: [],
};

/**
 * Permanently delete a model repo from the Hugging Face Hub. Scoped to the
 * caller's own namespace on the backend. A repo already gone (404) resolves
 * as success (idempotent), matching the backend semantics.
 */
export async function deleteHubModel(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
): Promise<void> {
  await apiRequest<void>(
    baseUrl,
    fetcher,
    `/jobs/hub/models/${repoId}`,
    { method: "DELETE", action: "Delete hub model" },
  );
}

/**
 * Hide a Hub job from the /jobs/hub listing. A local, persisted dismissal on
 * the backend — the HF Jobs API has no delete, so the job record on the Hub
 * itself is untouched.
 */
export async function dismissHubJob(
  baseUrl: string,
  fetcher: Fetcher,
  jobId: string,
): Promise<void> {
  await apiRequest<void>(
    baseUrl,
    fetcher,
    `/jobs/hub/jobs/${encodeURIComponent(jobId)}/dismiss`,
    { method: "POST", action: "Dismiss hub job" },
  );
}

export async function listHubJobs(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<HubJobsResponse> {
  // Same graceful degradation as listRunnerHardware.
  try {
    return await apiRequest<HubJobsResponse>(baseUrl, fetcher, "/jobs/hub", {
      signal,
      action: "List hub jobs",
    });
  } catch (e) {
    if (e instanceof ApiError) return EMPTY_HUB;
    throw e;
  }
}
