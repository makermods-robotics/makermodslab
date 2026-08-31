import { Fetcher, apiRequest } from "./apiClient";

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
  /** Carries per-episode sampling weights (a weighted merge). Present for
   * datasets with a local copy; absent for Hub-only rows, where it is unknown
   * rather than false — so read it as `weighted === true`, never `!weighted`. */
  weighted?: boolean;
}

export async function listDatasets(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<DatasetItem[]> {
  return apiRequest<DatasetItem[]>(baseUrl, fetcher, "/api/v1/datasets", {
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
  return apiRequest(baseUrl, fetcher, "/api/v1/datasets/custom", {
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
  return apiRequest(baseUrl, fetcher, "/api/v1/datasets/hide", {
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
  return apiRequest(baseUrl, fetcher, "/api/v1/datasets/custom", {
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
  return apiRequest(baseUrl, fetcher, "/api/v1/datasets/download", {
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
    "/api/v1/datasets/download-status",
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
  return apiRequest(baseUrl, fetcher, "/api/v1/datasets/import", {
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
  /** Carries per-episode sampling weights. Local datasets only; absent means
   * unknown (Hub-only), not false. */
  weighted?: boolean;
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
    `/api/v1/datasets/info?repo_id=${encodeURIComponent(repoId)}`,
    { signal, action: "Dataset info" },
  );
}

export interface EpisodeSummary {
  episode_index: number;
  length: number;
  duration: number;
  tasks: string[];
  /** How often this episode is sampled during training, relative to a weight of
   * 1. Written at merge time; absent from an older backend's response, and
   * absent means 1 (see R3 in docs/weighted-sampling-plan.md) — so read it as
   * `sampling_weight ?? 1`, never as a bare number. */
  sampling_weight?: number;
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
    `/api/v1/datasets/episodes?repo_id=${encodeURIComponent(repoId)}`,
    { signal, action: "List episodes" },
  );
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
    `/api/v1/datasets/episode-joints?repo_id=${encodeURIComponent(repoId)}&episode_index=${episodeIndex}`,
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
  return `${baseUrl}/api/v1/datasets/episode-video?${params.toString()}`;
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
    `/api/v1/datasets/hub-status?repo_id=${encodeURIComponent(repoId)}`,
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
    `/api/v1/datasets/hub-settings?repo_id=${encodeURIComponent(repoId)}`,
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
  return apiRequest(baseUrl, fetcher, "/api/v1/datasets/visibility", {
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
  return apiRequest(baseUrl, fetcher, "/api/v1/datasets/tags", {
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
  return apiRequest(baseUrl, fetcher, "/api/v1/upload-dataset", {
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
  return apiRequest<UploadStatus>(baseUrl, fetcher, "/api/v1/upload-status", {
    action: "Upload status",
    signal,
  });
}

export async function deleteDataset(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
): Promise<{ success: boolean; message?: string }> {
  return apiRequest(baseUrl, fetcher, "/api/v1/delete-dataset", {
    method: "POST",
    body: { dataset_repo_id: repoId },
    action: "Delete dataset",
  });
}

/** What happened to the dataset's Hub copy during a rename: "renamed" — the
 * Hub copy was moved to match; "none" — the Hub was reachable and confirmed
 * it has no copy; "skipped" — the Hub step didn't run (offline, logged out,
 * or a namespace this account can't write to), so a Hub copy, if any, KEPT
 * ITS OLD NAME. The local rename always happens regardless. */
export type DatasetRenameHubResult = "renamed" | "none" | "skipped";

/**
 * Rename a locally-cached dataset by moving its directory. `newName` is the
 * NAME PART ONLY — the namespace prefix stays fixed (so `ns/old` -> `ns/new`).
 * Returns the new repo_id plus `hub`, which reports whether the Hub copy was
 * renamed too — the caller must surface it rather than claim a Hub rename
 * happened unconditionally (a "skipped" Hub copy is still live under the old
 * name). Throws ApiError on a rejected rename (invalid name, target exists,
 * dataset in use), with the backend's message in `.detail`.
 */
export async function renameDataset(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  newName: string,
): Promise<{ success: boolean; repo_id: string; hub: DatasetRenameHubResult }> {
  return apiRequest(baseUrl, fetcher, "/api/v1/datasets/rename", {
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

/** Largest per-source weight the backend accepts (mirrors
 * `MAX_SOURCE_WEIGHT` in makermodslab/merge.py — keep the two in step). */
export const MAX_SOURCE_WEIGHT = 20;

/** Start a merge. `sourceWeights` is a per-source integer repeat count,
 * positionally aligned with `sourceRepoIds`; omit it (or pass all 1s) for an
 * unweighted merge. A source with weight 3 contributes its episodes three
 * times, so training samples them three times as often. */
export async function startDatasetMerge(
  baseUrl: string,
  fetcher: Fetcher,
  sourceRepoIds: string[],
  outputRepoId: string,
  sourceWeights?: number[],
): Promise<{ started: boolean; message: string }> {
  // Only send weights when at least one is non-default, so an ordinary merge
  // keeps the exact request body it had before weights existed.
  const weighted = sourceWeights?.some((w) => w !== 1) ?? false;
  // apiRequest JSON.stringifies `body` itself — pass a raw object, not a string.
  return apiRequest(baseUrl, fetcher, "/api/v1/datasets/merge", {
    method: "POST",
    body: {
      source_repo_ids: sourceRepoIds,
      output_repo_id: outputRepoId,
      ...(weighted ? { source_weights: sourceWeights } : {}),
    },
    action: "Merge datasets",
  });
}

export async function getDatasetMergeStatus(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<MergeStatus> {
  return apiRequest<MergeStatus>(baseUrl, fetcher, "/api/v1/datasets/merge/status", {
    action: "Merge status",
  });
}
