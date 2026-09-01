import { Fetcher, apiRequest } from "./apiClient";

export interface JobCheckpoint {
  step: number;
  source: "local" | "hub";
  ref: string;
}

export interface PolicyConfigSummary {
  policy_type: string | null;
  image_features: Record<string, { height: number; width: number }>;
  requires_task: boolean;
  // Flat proprioceptive state / action widths from the checkpoint. For an
  // SO-101 arm this is 6 (one per joint); a bimanual-trained checkpoint carries
  // 12 (two arms). The inference modal compares state_dim against the selected
  // robot's arm count to explain a single-arm/bimanual mismatch. Either can be
  // null when the checkpoint omits the feature.
  state_dim: number | null;
  action_dim: number | null;
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
): Promise<JobCheckpoint[]> {
  const body = await apiRequest<{ checkpoints: JobCheckpoint[] }>(
    baseUrl,
    fetcher,
    `/api/v1/jobs/${jobId}/checkpoints`,
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
