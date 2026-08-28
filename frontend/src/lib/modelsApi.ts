import { Fetcher, apiRequest } from "./apiClient";

/** Whether a model is a completed local run, a Hub policy repo, or both (a
 * local run that was also pushed to the Hub). Mirrors DatasetSource. */
export type ModelSource = "local" | "hub" | "both";

/**
 * One row in the merged /models listing. `id` is the local run id (for local /
 * both) or the Hub repo id (hub-only). `path` is the local checkpoint dir when
 * present; `hf_repo_id` the Hub repo when the model is (or was pushed) on the
 * Hub. `dataset` / `steps` / `policy_type` come from the checkpoint's
 * train_config and may be null for a Hub-only model that didn't record them.
 */
export interface ModelItem {
  id: string;
  name: string;
  policy_type: string | null;
  dataset: string | null;
  steps: number | null;
  path: string | null;
  last_modified: string | null;
  hf_repo_id: string | null;
  source: ModelSource;
  /** True for a Hub model the user pinned via the "Add model" chooser (not
   * their own namespace). Such a row is "removed" by unpinning
   * (removeCustomModel), never a destructive delete. Mirrors
   * DatasetItem.saved_custom. */
  saved_custom?: boolean;
  /** Whether the Hub repo is private (hub-derived rows only). Mirrors
   * DatasetItem.private; drives the picker's amber "private" badge. */
  private?: boolean;
}

/** GET /models — merged local + Hub listing, each with a `source`. Mirrors
 * listDatasets. */
export async function getModels(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<ModelItem[]> {
  return apiRequest<ModelItem[]>(baseUrl, fetcher, "/models", {
    signal,
    action: "List models",
  });
}

/** Where a skill's weights came from, as opposed to where they live now
 * (`source`). This is what the library filters on. */
export type SkillOrigin =
  | "trained-local"
  | "trained-cloud"
  | "imported"
  | "downloaded"
  | "hub-untracked";

/** Whether the weights can be loaded. "unverified" is a Hub-backed row the
 * server deliberately did NOT probe — confirming each one is a serial round
 * trip, so the claim is settled by the download the deploy path runs anyway. */
export type SkillWeights = "ready" | "unverified" | "none";

/** One row of GET /skills. A ModelItem plus the facts that decide whether it
 * can actually be run, and the job that runs it. */
export interface SkillItem extends ModelItem {
  origin: SkillOrigin;
  weights: SkillWeights;
  /** Terminal state of the run behind it, when there is one. "failed" rows are
   * listed and runnable — the weights exist — but must be badged. */
  state: string | null;
  /** The run that represents this one's resume chain; non-null means this row
   * is a link, not a skill, and `deployable` is false. */
  superseded_by: string | null;
  /** Derived server-side, never stored: weights are loadable AND nothing
   * supersedes it. */
  deployable: boolean;
  /** The registry record that deploys this row, stamped by the server. Null for
   * a row no run tracks (a bare Hub repo, a scanned directory) — those still go
   * through the lazy import on pick. */
  job_id: string | null;
  /** Set on a Hub row served from the last complete listing because this
   * refresh could not reach the Hub. */
  stale?: boolean;
}

/** Reachability of the Hub half of the listing. Without this an outage and an
 * empty shelf are the same screen. */
export interface SkillsHubStatus {
  ok: boolean;
  authenticated: boolean;
  degraded: boolean;
  /** Some rows are being served from the last complete listing. */
  stale_rows: boolean;
}

export interface SkillsEnvelope {
  skills: SkillItem[];
  hub: SkillsHubStatus;
}

/** GET /skills — every trained policy, each saying whether it can run.
 *
 * The deployable projection of the same build /models serves. Both the deploy
 * picker and the models library read THIS, so they cannot disagree about what a
 * skill is; /models stays the wider artifact listing the fine-tune base picker
 * needs. */
export async function getSkills(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<SkillsEnvelope> {
  return apiRequest<SkillsEnvelope>(baseUrl, fetcher, "/skills", {
    signal,
    action: "List skills",
  });
}

/** Detail view of one model. Adds `size_bytes` (null for a Hub-only model,
 * which isn't on disk) on top of the listing fields. */
export interface ModelInfo extends ModelItem {
  size_bytes: number | null;
}

/** GET /models/info?id=… — per-model detail card. `id` is a local run id or a
 * Hub repo id (repo ids contain "/", hence a query param). Throws ApiError with
 * status 404 when neither resolves. Mirrors getDatasetInfo. */
export async function getModelInfo(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
  signal?: AbortSignal,
): Promise<ModelInfo> {
  return apiRequest<ModelInfo>(
    baseUrl,
    fetcher,
    `/models/info?id=${encodeURIComponent(id)}`,
    { signal, action: "Model info" },
  );
}

/** POST /models/upload — push a local run's final checkpoint to the Hub as a
 * PUBLIC, MakerModsLab-tagged model repo. `id` is the local run id; `repoId` optionally
 * overrides the default namespaced repo id. MUTATES the Hub. Throws ApiError
 * (400 offline, 403 no write access, 404 no checkpoint, 502 Hub failure) with
 * the backend message in `.detail`. Returns {repo_id, url, tags}. */
export async function uploadModel(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
  repoId?: string,
): Promise<{ repo_id: string; url: string; tags: string[] }> {
  return apiRequest(baseUrl, fetcher, "/models/upload", {
    method: "POST",
    body: { id, ...(repoId ? { repo_id: repoId } : {}) },
    action: "Upload model",
  });
}

/** POST /models/delete — remove a local model's training-run output dir
 * (strictly sandboxed under outputs/train/). Never touches the Hub. Throws
 * ApiError (400 non-local/unsafe, 404 unknown, 409 still training, 502 delete
 * failure). Returns {deleted, id}. Mirrors deleteDataset. */
export async function deleteModel(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
): Promise<{ deleted: boolean; id: string }> {
  return apiRequest(baseUrl, fetcher, "/models/delete", {
    method: "POST",
    body: { id },
    action: "Delete model",
  });
}

/** Pin a typed Hub model repo id so it persists in the /models listing.
 * Idempotent; POST /models/custom. Mirrors saveCustomDataset. */
export async function saveCustomModel(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
): Promise<{ success: boolean; repo_id: string }> {
  return apiRequest(baseUrl, fetcher, "/models/custom", {
    method: "POST",
    body: { repo_id: repoId },
    action: "Save custom model",
  });
}

/** Hide a Hub model from the picker listing ("remove from list"). NEVER
 * deletes or mutates the Hub repo — a persistent local filter. Re-pinning via
 * saveCustomModel auto-unhides. POST /models/hide. Mirrors hideDataset. */
export async function hideModel(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
): Promise<{ success: boolean; repo_id: string }> {
  return apiRequest(baseUrl, fetcher, "/models/hide", {
    method: "POST",
    body: { repo_id: repoId },
    action: "Hide model",
  });
}

/** Unpin a saved custom model (does not touch the Hub or any local copy).
 * DELETE /models/custom. Mirrors removeCustomDataset. */
export async function removeCustomModel(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
): Promise<{ success: boolean; repo_id: string }> {
  return apiRequest(baseUrl, fetcher, "/models/custom", {
    method: "DELETE",
    body: { repo_id: repoId },
    action: "Remove custom model",
  });
}

export type ModelDownloadState = "idle" | "running" | "done" | "error";

/** Live status of the single background Hub-model download. Same shape as the
 * dataset download status (they share the backend state machine). */
export interface ModelDownloadStatus {
  state: ModelDownloadState;
  repo_id: string | null;
  message: string | null;
  error: string | null;
}

/** Kick off a background download of a Hub model checkpoint into the local
 * models dir. Returns immediately with {started, repo_id}; poll
 * getModelDownloadStatus for progress. Throws ApiError (400 bad id, 409 a
 * download is already running). POST /models/download. */
export async function downloadModel(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
): Promise<{ started: boolean; repo_id: string; message: string }> {
  return apiRequest(baseUrl, fetcher, "/models/download", {
    method: "POST",
    body: { repo_id: repoId },
    action: "Download model",
  });
}

/** Current state of the single background model download (survives navigation —
 * a card polls this on mount to re-attach). GET /models/download-status. */
export async function getModelDownloadStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<ModelDownloadStatus> {
  return apiRequest<ModelDownloadStatus>(
    baseUrl,
    fetcher,
    "/models/download-status",
    { action: "Model download status", signal },
  );
}

/** Import a policy checkpoint folder already on the server machine by COPYING
 * it into the local models dir (source left intact). `name` optionally overrides
 * the target id (bare or namespace/name; defaults to the folder's basename).
 * Throws ApiError (400 invalid source/name, 404 no such folder, 409 target
 * exists). POST /models/import. Mirrors importDataset — distinct from the jobs
 * importModel (jobsApi), which registers a POINTER to the source instead of
 * copying it. */
export async function importModelFromDisk(
  baseUrl: string,
  fetcher: Fetcher,
  path: string,
  name?: string,
): Promise<{ repo_id: string }> {
  return apiRequest(baseUrl, fetcher, "/models/import", {
    method: "POST",
    body: { path, name },
    action: "Import model",
  });
}
