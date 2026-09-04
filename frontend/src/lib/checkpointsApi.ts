import { Fetcher, apiRequest } from "./apiClient";

export interface JobCheckpoint {
  step: number;
  source: "local" | "hub";
  ref: string;
  /** Which run in a resume chain actually saved this checkpoint. Present only
   * on a `lineage: true` listing, where a step is no longer unique; a
   * single-run listing omits it because every row would say the same thing.
   *
   * `(owner_job_id, step)` IS unique, so it is the safe way to address one
   * checkpoint by step — e.g. the policy-config endpoint, which takes a job id
   * and a step. */
  owner_job_id?: string;
  owner_job_number?: number;
  owner_name?: string;
}

export interface PolicyConfigSummary {
  policy_type: string | null;
  image_features: Record<string, { height: number; width: number }>;
  requires_task: boolean;
  /** Whether this checkpoint's ARCHITECTURE can run the Real-Time Chunking
   * inference engine. `false` means the server refuses `inference_engine:
   * "rtc"` for it with a 400 before any hardware is claimed, so the dialogs
   * take the option off the menu. `null` means "not established" — a policy
   * type newer than the server's table — and must be read as "offer it and let
   * the server decide", never as "no". */
  supports_rtc: boolean | null;
  // Flat proprioceptive state / action widths from the checkpoint. For an
  // SO-101 arm this is 6 (one per joint); a bimanual-trained checkpoint carries
  // 12 (two arms). The inference modal compares state_dim against the selected
  // robot's arm count to explain a single-arm/bimanual mismatch. Either can be
  // null when the checkpoint omits the feature.
  state_dim: number | null;
  action_dim: number | null;
  /** Raw lerobot robot_type of the dataset this checkpoint was trained on
   * (recovered via its train_config.json). null when it can't be
   * established — an imported flat model, a deleted training dataset, an
   * untagged one. The fine-tune panel normalises it with armTypeFromRobotType
   * and warns when it disagrees with the selected dataset's arm. */
  trained_on_robot_type?: string | null;
  /** The dataset this checkpoint was trained on, from its OWN
   * train_config.json. null when the lineage offers no real id (an imported
   * flat model repo with no train_config; the "(imported)" placeholder is
   * never reported as one).
   *
   * Prefer this over the owning job's `config.dataset_repo_id`, which is the
   * wrong source twice over: an import's record carries the placeholder, and
   * on a resume chain the tip's record does not describe a checkpoint owned by
   * an ancestor. Absent from an older backend's response — read it as
   * `?? job.config?.dataset_repo_id`, never as a bare value. */
  dataset_repo_id?: string | null;
}

/** Collapse checkpoint entries that point at the same underlying checkpoint.
 *
 * A cloud resume reuses its parent's output repo, so when a card merges the
 * checkpoint lists of a run and its ancestors, every checkpoint in the shared
 * repo is listed once per run — identical `ref`, identical step. The `ref`
 * encodes the checkpoint's true identity (owning repo/path plus step), so it
 * is the dedupe key; entries with the same step but different refs (two runs
 * that each saved a step N in their own repo) are distinct and all kept.
 * Keeps the first occurrence, so with the caller's this-run-before-ancestors
 * ordering the surviving entry is attributed to the nearest run in the
 * lineage. */
export function dedupeCheckpointEntries<T extends { ckpt: JobCheckpoint }>(
  entries: T[],
): T[] {
  const seen = new Set<string>();
  return entries.filter((entry) => {
    if (seen.has(entry.ckpt.ref)) return false;
    seen.add(entry.ckpt.ref);
    return true;
  });
}

export async function listJobCheckpoints(
  baseUrl: string,
  fetcher: Fetcher,
  jobId: string,
  signal?: AbortSignal,
  /** Widen to the whole resume chain — this run plus the runs it resumed.
   * A chain is ONE model trained across several records, so anything offering
   * "which checkpoint do you want to run" needs all of them; without this the
   * tip offers only the steps it personally saved. */
  lineage = false,
): Promise<JobCheckpoint[]> {
  const body = await apiRequest<{ checkpoints: JobCheckpoint[] }>(
    baseUrl,
    fetcher,
    `/api/v1/jobs/${jobId}/checkpoints${lineage ? "?lineage=true" : ""}`,
    { signal, action: "List checkpoints" },
  );
  return body.checkpoints;
}

export async function getCheckpointPolicyConfig(
  baseUrl: string,
  fetcher: Fetcher,
  jobId: string,
  step: number,
  signal?: AbortSignal,
): Promise<PolicyConfigSummary> {
  return apiRequest<PolicyConfigSummary>(
    baseUrl,
    fetcher,
    `/api/v1/jobs/${jobId}/checkpoints/${step}/policy-config`,
    { signal, action: "Load policy config" },
  );
}
