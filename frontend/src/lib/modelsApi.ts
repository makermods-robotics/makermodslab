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
  /** What the local side of this row is — a training run or a copy pulled from
   * the Hub / imported from disk. Absent on hub-only rows. Read by
   * resolveDeleteAction, because deleting a run costs its unpublished
   * checkpoints while deleting a copy costs nothing irreplaceable. */
  local_kind?: "run" | "downloaded";
  /** Episode subset this model was trained on, from its checkpoint's
   * train_config.json. null/absent means every episode (no curation), OR the
   * training dataset didn't resolve as a public Hub dataset — the backend
   * redacts this field rather than expose it for a private/unresolvable
   * source (see models._gate_dataset_episodes); the frontend has no way to
   * tell those two cases apart, by design. */
  dataset_episodes?: number[] | null;
}

/** GET /models — merged local + Hub listing, each with a `source`. Mirrors
 * listDatasets. */
export async function getModels(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<ModelItem[]> {
  return apiRequest<ModelItem[]>(baseUrl, fetcher, "/api/v1/models", {
    signal,
    action: "List models",
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
    `/api/v1/models/info?id=${encodeURIComponent(id)}`,
    { signal, action: "Model info" },
  );
}

/** One of a local run's saved checkpoints as the publish picker sees it.
 * `published` is true when that step is already in the target Hub repo — the
 * picker leaves those unselected so a re-publish reads as "add the new ones". */
export interface RunCheckpoint {
  step: number;
  path: string;
  published: boolean;
}

/** GET /api/v1/models/checkpoints?id=… — everything the publish picker needs for one
 * run in a single call. `hf_repo_id` is set only once the run has actually been
 * published (so the UI can say "add more" instead of "publish");
 * `default_repo_id` is the target an upload with no explicit repo would pick,
 * shown as the input's placeholder. `legacy_root_checkpoint` flags a repo whose
 * first upload predates the step-addressed layout. */
export interface RunCheckpoints {
  id: string;
  default_repo_id: string;
  hf_repo_id: string | null;
  legacy_root_checkpoint: boolean;
  /** False when the Hub couldn't be asked (offline, or the call failed). Every
   * `published` flag is then `false` by default and means "unknown", not "not
   * published" — present it that way, or a user re-queues gigabytes they
   * already sent. */
  hub_readable: boolean;
  checkpoints: RunCheckpoint[];
}

export async function listRunCheckpoints(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
  signal?: AbortSignal,
): Promise<RunCheckpoints> {
  return apiRequest<RunCheckpoints>(
    baseUrl,
    fetcher,
    `/api/v1/models/checkpoints?id=${encodeURIComponent(id)}`,
    { signal, action: "List run checkpoints" },
  );
}

/** POST /api/v1/models/publish — START publishing a local run's checkpoints to
 * the Hub as ONE public, MakerModsLab-tagged model repo. `id` is the local run
 * id; `repoId` optionally overrides the default namespaced repo id; `steps`
 * names the checkpoints to publish (omitted ⇒ the final one only). Every step
 * lands in the SAME repo, so a later call adds to the same model card.
 *
 * A v1-only endpoint: the legacy flat POST /models/upload is the older,
 * synchronous single-checkpoint push and is left frozen for SDK clients.
 *
 * MUTATES the Hub, and returns as soon as the queue is accepted — the upload
 * itself runs in the background; poll getModelUploadStatus for progress and for
 * the failures (offline / permission / Hub) that surface there. Throws ApiError
 * 409 when a publish is already running. */
export async function uploadModel(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
  repoId?: string,
  steps?: number[],
): Promise<{ started: boolean; model_id: string; message: string }> {
  return apiRequest(baseUrl, fetcher, "/api/v1/models/publish", {
    method: "POST",
    body: {
      id,
      ...(repoId ? { repo_id: repoId } : {}),
      ...(steps ? { steps } : {}),
    },
    action: "Upload model",
  });
}

/** Progress of the single background publish. `done`/`total` are queue
 * position, `current_step` the checkpoint in flight, and `done_steps` the ones
 * already on the Hub — which stay meaningful on `error`, because a queue that
 * fails part-way keeps everything it published before it died. */
export interface ModelUploadStatus {
  state: "idle" | "running" | "done" | "error";
  model_id: string | null;
  repo_id: string | null;
  url: string | null;
  message: string | null;
  error: string | null;
  total: number;
  done: number;
  current_step: number | null;
  done_steps: number[];
}

/** GET /api/v1/models/publish-status — poll the background publish. */
export async function getModelUploadStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<ModelUploadStatus> {
  return apiRequest<ModelUploadStatus>(baseUrl, fetcher, "/api/v1/models/publish-status", {
    signal,
    action: "Model upload status",
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
  return apiRequest(baseUrl, fetcher, "/api/v1/models/delete", {
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
  return apiRequest(baseUrl, fetcher, "/api/v1/models/custom", {
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
  return apiRequest(baseUrl, fetcher, "/api/v1/models/hide", {
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
  return apiRequest(baseUrl, fetcher, "/api/v1/models/custom", {
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
  return apiRequest(baseUrl, fetcher, "/api/v1/models/download", {
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
    "/api/v1/models/download-status",
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
  return apiRequest(baseUrl, fetcher, "/api/v1/models/import", {
    method: "POST",
    body: { path, name },
    action: "Import model",
  });
}
