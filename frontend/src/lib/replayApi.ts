import { ApiError, Fetcher, apiRequest } from "./apiClient";

export type DatasetSource = "local" | "hub" | "both";

export interface DatasetItem {
  repo_id: string;
  last_modified: string | null;
  private: boolean;
  source: DatasetSource;
  /** True for a Hub dataset the user pinned by typing it into the picker (not
   * their own namespace, no local copy). Such a row is "removed" by unpinning
   * (removeCustomDataset), never a destructive delete. */
  saved_custom?: boolean;
}

export async function listDatasets(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<DatasetItem[]> {
  return apiRequest<DatasetItem[]>(baseUrl, fetcher, "/datasets", {
    signal,
    action: "List datasets",
  });
}

/** Pin a typed Hub dataset repo id so it persists in the picker listing.
 * Idempotent; POST /datasets/custom. */
export async function saveCustomDataset(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
): Promise<{ success: boolean; repo_id: string }> {
  return apiRequest(baseUrl, fetcher, "/datasets/custom", {
    method: "POST",
    body: { repo_id: repoId },
    action: "Save custom dataset",
  });
}

/** Hide a Hub dataset from the picker listing ("remove from list"). NEVER
 * deletes or mutates the Hub repo — a persistent local filter. Re-pinning via
 * saveCustomDataset auto-unhides. POST /datasets/hide. */
export async function hideDataset(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
): Promise<{ success: boolean; repo_id: string }> {
  return apiRequest(baseUrl, fetcher, "/datasets/hide", {
    method: "POST",
    body: { repo_id: repoId },
    action: "Hide dataset",
  });
}

/** Unpin a saved custom dataset (does not touch the Hub or any local copy).
 * DELETE /datasets/custom. */
export async function removeCustomDataset(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
): Promise<{ success: boolean; repo_id: string }> {
  return apiRequest(baseUrl, fetcher, "/datasets/custom", {
    method: "DELETE",
    body: { repo_id: repoId },
    action: "Remove custom dataset",
  });
}

export type DatasetDownloadState = "idle" | "running" | "done" | "error";

/** Live status of the single background Hub-dataset download. `error` is set
 * when it failed; `repo_id` says which dataset the download is for (so a card
 * knows whether it's *its* download). */
export interface DatasetDownloadStatus {
  state: DatasetDownloadState;
  repo_id: string | null;
  message: string | null;
  error: string | null;
}

/** Kick off a background download of a Hub dataset into the local cache. Returns
 * immediately with {started, repo_id}; poll getDatasetDownloadStatus for
 * progress. Throws ApiError (400 bad id, 409 a download is already running) with
 * the backend message in `.detail`. POST /datasets/download. */
export async function downloadDataset(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
): Promise<{ started: boolean; repo_id: string; message: string }> {
  return apiRequest(baseUrl, fetcher, "/datasets/download", {
    method: "POST",
    body: { repo_id: repoId },
    action: "Download dataset",
  });
}

/** Current state of the single background download (survives navigation — a
 * card polls this on mount to re-attach to an in-flight download).
 * GET /datasets/download-status. */
export async function getDatasetDownloadStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<DatasetDownloadStatus> {
  return apiRequest<DatasetDownloadStatus>(
    baseUrl,
    fetcher,
    "/datasets/download-status",
    { action: "Download status", signal },
  );
}

/** Import a LeRobot dataset folder already on the server machine by COPYING it
 * into the local cache. `name` is the optional target repo id (bare or
 * namespace/name); defaults to the source folder's basename. Throws ApiError
 * (400 invalid source/name, 404 no such folder, 409 target exists) with the
 * backend message in `.detail`. POST /datasets/import. */
export async function importDataset(
  baseUrl: string,
  fetcher: Fetcher,
  path: string,
  name?: string,
): Promise<{ repo_id: string }> {
  return apiRequest(baseUrl, fetcher, "/datasets/import", {
    method: "POST",
    body: { path, name },
    action: "Import dataset",
  });
}

/** One task string with how many episodes use it (0 = count unavailable). */
export interface DatasetTask {
  task: string;
  num_episodes: number;
}

export interface DatasetInfo {
  repo_id: string;
  total_episodes: number;
  total_frames: number;
  fps: number | null;
  robot_type: string | null;
  cameras: string[];
  tasks: DatasetTask[];
  /** On-disk size for a local dataset; null for a Hub summary (not on disk). */
  size_bytes: number | null;
  /** "local" = full detail from the local cache; "hub" = the meta/info.json
   * summary of a not-yet-downloaded Hub dataset (no tasks/size; rename not
   * applicable). Treat absent as "local". */
  source?: "local" | "hub";
}

/** Detail view of one dataset: full local detail, or a Hub meta/info.json
 * summary for a dataset with no local copy (see `source`). 404 only when
 * neither resolves (offline / unknown repo). */
export async function getDatasetInfo(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  signal?: AbortSignal,
): Promise<DatasetInfo> {
  return apiRequest<DatasetInfo>(
    baseUrl,
    fetcher,
    `/datasets/info?repo_id=${encodeURIComponent(repoId)}`,
    { signal, action: "Dataset info" },
  );
}

export interface EpisodeSummary {
  episode_index: number;
  length: number;
  duration: number;
  tasks: string[];
  /** Per-camera {from, to} seconds locating this episode's slice WITHIN its
   * (possibly shared) video file — v3.0 packs consecutive episodes into the
   * same mp4 per camera, so playback must seek to `from` and stop at `to`
   * rather than assume the file starts/ends at this episode's boundaries. */
  video_offsets: Record<string, { from: number; to: number }>;
}

/** Per-episode index/length/duration/tasks, for the dataset viewer window's
 * episode list. 404 when the dataset isn't local, or predates the v3.0
 * parquet episode layout the viewer reads (older jsonl-metadata datasets
 * aren't viewable — see makermodslab/datasets.py `_read_episode_rows`). */
export async function listEpisodes(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  signal?: AbortSignal,
): Promise<EpisodeSummary[]> {
  return apiRequest<EpisodeSummary[]>(
    baseUrl,
    fetcher,
    `/datasets/episodes?repo_id=${encodeURIComponent(repoId)}`,
    { signal, action: "List episodes" },
  );
}

interface EpisodeDeleteResult {
  success: boolean;
  repo_id: string;
  deleted_episode: number;
  total_episodes: number;
  trash_id: string;
}

interface EpisodeDeleteStatus {
  state: "idle" | "running" | "done" | "error";
  repo_id: string | null;
  episode_index: number | null;
  message: string | null;
  result: EpisodeDeleteResult | null;
}

const EPISODE_DELETE_POLL_MS = 500;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Delete one episode from a locally-cached dataset. Local-only — never
 * touches a Hub copy. The actual rewrite runs on the backend in a background
 * thread (it can take a moment for a shared-video-file episode); this
 * function starts it and polls until it finishes, so callers keep awaiting
 * one call exactly as before rather than holding one HTTP request open for
 * the whole rewrite. Throws ApiError (400 invalid index / dataset's only
 * episode, 404 not local, 409 busy, 507 not enough free disk space, or a
 * background rewrite failure) with the backend message in `.detail`.
 * POST /datasets/episode-delete, polls GET /datasets/episode-delete-status. */
export async function deleteEpisode(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  episodeIndex: number,
): Promise<EpisodeDeleteResult> {
  await apiRequest(baseUrl, fetcher, "/datasets/episode-delete", {
    method: "POST",
    body: { repo_id: repoId, episode_index: episodeIndex },
    action: "Delete episode",
  });

  // A 409 above means someone else already owns the single delete slot, so
  // once we get here nothing else can start and overwrite it before we see
  // our own outcome — no need to match on repo_id/episode_index below.
  for (;;) {
    let status: EpisodeDeleteStatus | undefined;
    try {
      status = await apiRequest<EpisodeDeleteStatus>(
        baseUrl,
        fetcher,
        "/datasets/episode-delete-status",
        { action: "Delete episode" },
      );
    } catch {
      // transient — retry next tick
    }
    if (status?.state === "done" && status.result) return status.result;
    if (status?.state === "error") {
      throw new ApiError(
        `Delete episode failed: ${status.message ?? "Unknown error"}`,
        500,
        status.message,
      );
    }
    await sleep(EPISODE_DELETE_POLL_MS);
  }
}

export interface DeletedEpisodeEntry {
  trash_id: string;
  episode_index: number;
  deleted_at: string;
  expires_at: number;
  length: number;
  duration_s: number;
}

/** Unexpired (< 24h) episode-delete trash entries for repoId, newest first.
 * GET /datasets/deleted-episodes. */
export async function listDeletedEpisodes(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
): Promise<DeletedEpisodeEntry[]> {
  return apiRequest<DeletedEpisodeEntry[]>(
    baseUrl,
    fetcher,
    `/datasets/deleted-episodes?repo_id=${encodeURIComponent(repoId)}`,
    { action: "List deleted episodes" },
  );
}

interface EpisodeUndoStatus {
  state: "idle" | "running" | "done" | "error";
  repo_id: string | null;
  trash_id: string | null;
  message: string | null;
  result: { success: boolean; repo_id: string } | null;
}

/** Restore a deleted episode from its trash entry. Starts the restore then
 * polls until it finishes, same start+poll shape as deleteEpisode. Throws
 * ApiError (404 unknown/expired trash_id, 409 busy, 507 not enough disk
 * space, or a background restore failure).
 * POST /datasets/undo-episode-delete, polls GET /datasets/episode-undo-status. */
export async function undoEpisodeDelete(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  trashId: string,
): Promise<{ success: boolean; repo_id: string }> {
  await apiRequest(baseUrl, fetcher, "/datasets/undo-episode-delete", {
    method: "POST",
    body: { repo_id: repoId, trash_id: trashId },
    action: "Undo episode delete",
  });

  for (;;) {
    let status: EpisodeUndoStatus | undefined;
    try {
      status = await apiRequest<EpisodeUndoStatus>(
        baseUrl,
        fetcher,
        "/datasets/episode-undo-status",
        { action: "Undo episode delete" },
      );
    } catch {
      // transient — retry next tick
    }
    if (status?.state === "done" && status.result) return status.result;
    if (status?.state === "error") {
      throw new ApiError(
        `Undo episode delete failed: ${status.message ?? "Unknown error"}`,
        500,
        status.message,
      );
    }
    await sleep(EPISODE_DELETE_POLL_MS);
  }
}

export interface DeletedDatasetEntry {
  trash_id: string;
  repo_id: string;
  deleted_at: string;
  expires_at: number;
}

/** Unexpired (< 24h) whole-dataset trash entries across the whole library,
 * newest first. GET /datasets/deleted-datasets. */
export async function listDeletedDatasets(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<DeletedDatasetEntry[]> {
  return apiRequest<DeletedDatasetEntry[]>(baseUrl, fetcher, "/datasets/deleted-datasets", {
    action: "List deleted datasets",
  });
}

/** Restore a whole dataset from its trash entry. Runs synchronously on the
 * backend (a rename, not a rewrite) so there's no polling here, unlike the
 * episode case. Throws ApiError (404 unknown/expired trash_id, or a name
 * collision if something now exists at the live path).
 * POST /datasets/undo-dataset-delete. */
export async function undoDatasetDelete(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  trashId: string,
): Promise<{ success: boolean; message: string }> {
  return apiRequest(baseUrl, fetcher, "/datasets/undo-dataset-delete", {
    method: "POST",
    body: { repo_id: repoId, trash_id: trashId },
    action: "Undo dataset delete",
  });
}

export interface EpisodeJointSeries {
  joint_names: string[];
  timestamps: number[];
  values: number[][];
}

/** Per-frame timestamp + joint (observation.state) values for one episode,
 * for the viewer's joint-position chart. */
export async function getEpisodeJoints(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  episodeIndex: number,
  signal?: AbortSignal,
): Promise<EpisodeJointSeries> {
  return apiRequest<EpisodeJointSeries>(
    baseUrl,
    fetcher,
    `/datasets/episode-joints?repo_id=${encodeURIComponent(repoId)}&episode_index=${episodeIndex}`,
    { signal, action: "Load episode joint data" },
  );
}

/** URL for one camera's mp4 for one episode — used directly as a <video> src
 * (the endpoint supports Range requests, so seeking doesn't need the whole
 * file). Not routed through `fetchWithHeaders`/apiRequest since a <video> tag
 * can't attach custom headers to its own request. */
export function episodeVideoUrl(
  baseUrl: string,
  repoId: string,
  episodeIndex: number,
  camera: string,
): string {
  const params = new URLSearchParams({
    repo_id: repoId,
    episode_index: String(episodeIndex),
    camera,
  });
  return `${baseUrl}/datasets/episode-video?${params.toString()}`;
}

/** Where a dataset with this id lives. "local_only" = a local copy exists but
 * it's not on the Hub (offer upload); "absent" = neither on the Hub nor local
 * (a stale pin / deleted / renamed selection); "unknown" is the
 * offline/unauthenticated degrade — the card shows no badge for it. */
export type HubStatusValue = "on_hub" | "local_only" | "absent" | "unknown";

export interface HubStatus {
  repo_id: string;
  status: HubStatusValue;
  url: string | null;
}

/** Hub existence check, fetched lazily/separately so it never blocks the
 * info card render. */
export async function getDatasetHubStatus(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  signal?: AbortSignal,
): Promise<HubStatus> {
  return apiRequest<HubStatus>(
    baseUrl,
    fetcher,
    `/datasets/hub-status?repo_id=${encodeURIComponent(repoId)}`,
    { signal, action: "Hub status" },
  );
}

/** Current Hub-side visibility + tags for a dataset already on the Hub, used to
 * pre-fill the post-upload editor. `tags` is the live card `tags:` list (org
 * tags included). */
export interface HubSettings {
  repo_id: string;
  private: boolean;
  tags: string[];
}

/** Read the current visibility + tags of a Hub dataset. Throws ApiError (400
 * offline, 403 no read/write access, 502 Hub failure) with the backend message
 * in `.detail`. */
export async function getDatasetHubSettings(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  signal?: AbortSignal,
): Promise<HubSettings> {
  return apiRequest<HubSettings>(
    baseUrl,
    fetcher,
    `/datasets/hub-settings?repo_id=${encodeURIComponent(repoId)}`,
    { signal, action: "Hub settings" },
  );
}

/** Flip a Hub dataset's visibility (public <-> private). MUTATES the live repo.
 * Throws ApiError (400 offline, 403 no write access, 502 Hub failure) with the
 * backend message in `.detail`. */
export async function setDatasetVisibility(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  isPrivate: boolean,
  signal?: AbortSignal,
): Promise<{ repo_id: string; private: boolean }> {
  return apiRequest(baseUrl, fetcher, "/datasets/visibility", {
    method: "POST",
    body: { repo_id: repoId, private: isPrivate },
    action: "Set visibility",
    signal,
  });
}

/** Replace a Hub dataset card's `tags:`. The backend re-adds the required org
 * tags, so the returned list may include tags beyond the ones passed. MUTATES
 * the live card. Throws ApiError (400 offline, 403 no write access, 502 Hub
 * failure) with the backend message in `.detail`. */
export async function setDatasetTags(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  tags: string[],
  signal?: AbortSignal,
): Promise<{ repo_id: string; tags: string[] }> {
  return apiRequest(baseUrl, fetcher, "/datasets/tags", {
    method: "POST",
    body: { repo_id: repoId, tags },
    action: "Set tags",
    signal,
  });
}

export type UploadState = "idle" | "running" | "done" | "error";

/** Live status of the single background upload. `dataset_url` is set once
 * done; `docs_url` accompanies a friendly auth error. `repo_id` says which
 * dataset the upload is for (so a card knows whether it's *its* upload). */
export interface UploadStatus {
  state: UploadState;
  repo_id: string | null;
  message: string | null;
  dataset_url?: string | null;
  docs_url?: string | null;
}

/** Kick off a background push of a locally-cached dataset to the Hub. Returns
 * immediately with {started, repo_id}; poll getDatasetUploadStatus for
 * progress. Throws ApiError (409) when an upload is already running or the
 * dataset is busy being written — the message is in `.detail`. */
export async function uploadDataset(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  tags: string[],
  isPrivate: boolean,
  signal?: AbortSignal,
): Promise<{ started: boolean; repo_id: string; message: string }> {
  return apiRequest(baseUrl, fetcher, "/upload-dataset", {
    method: "POST",
    body: { dataset_repo_id: repoId, tags, private: isPrivate },
    action: "Upload dataset",
    signal,
  });
}

/** Current state of the single background upload (survives navigation — the
 * card polls this on mount to re-attach to an in-flight upload). */
export async function getDatasetUploadStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<UploadStatus> {
  return apiRequest<UploadStatus>(baseUrl, fetcher, "/upload-status", {
    action: "Upload status",
    signal,
  });
}

export async function deleteDataset(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
): Promise<{ success: boolean; message?: string }> {
  return apiRequest(baseUrl, fetcher, "/delete-dataset", {
    method: "POST",
    body: { dataset_repo_id: repoId },
    action: "Delete dataset",
  });
}

/**
 * Rename a locally-cached dataset by moving its directory. `newName` is the
 * NAME PART ONLY — the namespace prefix stays fixed (so `ns/old` -> `ns/new`).
 * Returns the new repo_id. Throws ApiError on a rejected rename (invalid name,
 * target exists, dataset in use), with the backend's message in `.detail`.
 */
export async function renameDataset(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  newName: string,
): Promise<{ success: boolean; repo_id: string }> {
  return apiRequest(baseUrl, fetcher, "/datasets/rename", {
    method: "POST",
    body: { repo_id: repoId, new_name: newName },
    action: "Rename dataset",
  });
}

export type MergeState = "idle" | "running" | "done" | "error";

export interface MergeStatus {
  state: MergeState;
  error: string | null;
  output_repo_id: string | null;
  logs: { timestamp: number; message: string }[];
}

export async function startDatasetMerge(
  baseUrl: string,
  fetcher: Fetcher,
  sourceRepoIds: string[],
  outputRepoId: string,
): Promise<{ started: boolean; message: string }> {
  // apiRequest JSON.stringifies `body` itself — pass a raw object, not a string.
  return apiRequest(baseUrl, fetcher, "/datasets/merge", {
    method: "POST",
    body: { source_repo_ids: sourceRepoIds, output_repo_id: outputRepoId },
    action: "Merge datasets",
  });
}

export async function getDatasetMergeStatus(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<MergeStatus> {
  return apiRequest<MergeStatus>(baseUrl, fetcher, "/datasets/merge/status", {
    action: "Merge status",
  });
}
