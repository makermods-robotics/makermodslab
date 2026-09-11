import { ModelItem } from "@/lib/modelsApi";

/**
 * The `source` string to lazy-import when no job covers the model.
 *
 * The companion `findJobForModel` is gone: it re-implemented the backend's
 * `_job_outranks` ranking in TypeScript over a separately-fetched page of
 * /jobs, which made it a second copy of "which run owns these weights" that had
 * to be kept in sync by hand — and one that could not see a run past the page
 * limit it scanned. The server stamps `job_id` on every skill row instead.
 * This function survives because the case it serves is real: a row NO run
 * tracks (a bare Hub repo, a scanned directory) still has to register one. Preference
 * order mirrors the existing flows: the Hub repo id when the model has one
 * (exactly what HubModelCard passes — and what the backend's find_imported
 * dedups on), else the local checkpoint path (a disk import/download with no
 * hub identity — register_imported stores the path pointer), else the id.
 */
export function importSourceForModel(model: ModelItem): string {
  return model.hf_repo_id ?? model.path ?? model.id;
}
