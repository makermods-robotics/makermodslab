import { JobRecord } from "@/lib/jobsApi";
import { ModelItem } from "@/lib/modelsApi";

/**
 * Pure model → launchable-job mapping for the "Run inference" entry points.
 *
 * A selected ModelItem isn't always a job:
 *   * a LOCAL RUN model's `id` IS its job-registry id (list_local_models builds
 *     rows straight from the registry), so it matches a JobRecord by id;
 *   * a model previously LAZY-IMPORTED via the Jobs section (or a tracked cloud
 *     run) is covered by an existing record whose `hf_repo_id` matches — the
 *     same case-insensitive repo matching JobsSection's trackedRepoIds uses
 *     (and the backend's find_imported dedup);
 *   * anything else (hub-only / downloaded-but-unregistered) has no job yet and
 *     goes through the same lazy auto-import the Jobs cards use before playing
 *     (see useInferenceLaunch.importSource).
 *
 * Kept as pure functions (no fetching) per the repo's frontend-testing stance:
 * the logic is inspectable and tsc-checked without a component harness.
 */
export function findJobForModel(
  model: ModelItem,
  jobs: JobRecord[],
): JobRecord | null {
  // A local-run (or "both"-collapsed) model keys on its registry job id.
  const byId = jobs.find((j) => j.id === model.id);
  if (byId) return byId;

  // A hub model already tracked by an imported/cloud record — match on the
  // repo id, case-insensitively (HF repo ids are unique case-insensitively;
  // mirrors the backend's find_imported and JobsSection's trackedRepoIds).
  const repo = (model.hf_repo_id ?? "").toLowerCase();
  if (repo) {
    // SEVERAL records can share one repo: a cloud resume reuses its parent's
    // output repo, so an N-deep chain is N records pointing at the same
    // weights. First-match took whichever the registry listed first — often a
    // failed early link, whose checkpoint list is not the finished model's.
    // Rank as the backend does (models.py `_job_outranks`): a run that reached
    // "done" published the final policy at the repo root and wins; among
    // equals the later-started run (the deepest link of the chain) wins.
    return jobs
      .filter((j) => j.hf_repo_id?.toLowerCase() === repo)
      .reduce<JobRecord | null>((best, j) => {
        if (!best) return j;
        const jDone = j.state === "done";
        const bestDone = best.state === "done";
        if (jDone !== bestDone) return jDone ? j : best;
        return j.started_at > best.started_at ? j : best;
      }, null);
  }
  return null;
}

/**
 * The `source` string to lazy-import when no job covers the model. Preference
 * order mirrors the existing flows: the Hub repo id when the model has one
 * (exactly what HubModelCard passes — and what the backend's find_imported
 * dedups on), else the local checkpoint path (a disk import/download with no
 * hub identity — register_imported stores the path pointer), else the id.
 */
export function importSourceForModel(model: ModelItem): string {
  return model.hf_repo_id ?? model.path ?? model.id;
}
