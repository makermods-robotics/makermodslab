import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useApi } from "@/contexts/ApiContext";
import { ApiError } from "@/lib/apiClient";
import { Fetcher } from "@/lib/apiClient";

const POLL_MS = 1500;

/** The backend DownloadManager's status shape — shared by the dataset
 * (/datasets/download-status) and model (/models/download-status) downloaders,
 * which run the same server-side state machine. */
export interface HubDownloadStatus {
  state: "idle" | "running" | "done" | "error";
  repo_id: string | null;
  message: string | null;
  error: string | null;
}

interface UseHubDownloadArgs {
  /** The repo this hook cares about. It only reports "downloading" / fires
   * callbacks when the single global download is for THIS repo. */
  repoId: string;
  /** GET the poller's status endpoint (e.g. getDatasetDownloadStatus). */
  getStatus: (
    baseUrl: string,
    fetcher: Fetcher,
    signal?: AbortSignal,
  ) => Promise<HubDownloadStatus>;
  /** POST the download-start endpoint (e.g. downloadDataset). */
  startDownload: (
    baseUrl: string,
    fetcher: Fetcher,
    repoId: string,
  ) => Promise<{ started: boolean; repo_id: string; message: string }>;
  /** Fired once when this repo's download finishes. */
  onDone?: () => void;
  /** Fired once when this repo's download fails, with the message. */
  onError?: (message: string) => void;
}

export interface UseHubDownloadResult {
  /** True while the global download is running AND it's for this repoId. Drives
   * the row/card "Downloading…" state; survives navigation because the hook
   * re-attaches by polling the status endpoint on mount. */
  downloading: boolean;
  /** Kick off the download. Returns null on success, or an error message when
   * the start was refused (409: already running) or unreachable. */
  start: () => Promise<string | null>;
}

/**
 * Drives one repo's background Hub download against a DownloadManager-shaped
 * backend (one download at a time, start + pollable status). Parameterized by
 * the two API calls so the dataset and model downloaders share the client-side
 * state machine (see useDatasetDownload / useModelDownload): it starts the
 * download and, while it runs for THIS repoId, polls until done|error — then
 * fires onDone / onError exactly once. Polling only runs while a download for
 * this repo is in flight (no permanent loop), and the hook seeds itself from
 * the backend on mount so a card that mounts mid-download (e.g. after
 * navigating back) shows the live "Downloading…" state.
 */
export function useHubDownload({
  repoId,
  getStatus,
  startDownload,
  onDone,
  onError,
}: UseHubDownloadArgs): UseHubDownloadResult {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const [status, setStatus] = useState<HubDownloadStatus | null>(null);
  // Guards the one-shot callbacks so a lingering "done"/"error" status isn't
  // re-notified on every poll tick.
  const notified = useRef(false);

  // Latest callbacks/fns without retriggering the poll effect on every render.
  // `t` rides along for the same reason: the poll effect must not restart (and
  // re-arm its one-shot callbacks) just because the UI language changed.
  const onDoneRef = useRef(onDone);
  const onErrorRef = useRef(onError);
  const getStatusRef = useRef(getStatus);
  const startDownloadRef = useRef(startDownload);
  const tRef = useRef(t);
  onDoneRef.current = onDone;
  onErrorRef.current = onError;
  getStatusRef.current = getStatus;
  startDownloadRef.current = startDownload;
  tRef.current = t;

  const isMine = status?.repo_id === repoId;
  const downloading = isMine && status?.state === "running";

  // Seed from the backend on mount / repoId change so we re-attach to an
  // already-running download. If it's for a different repo, ignore it.
  useEffect(() => {
    let cancelled = false;
    notified.current = false;
    getStatusRef
      .current(baseUrl, fetchWithHeaders)
      .then((s) => {
        if (cancelled) return;
        setStatus(s.repo_id === repoId && s.state === "running" ? s : null);
      })
      .catch(() => {
        if (!cancelled) setStatus(null);
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders, repoId]);

  // Poll while this repo's download runs; on the running -> done|error edge fire
  // the one-shot callback and stop polling.
  useEffect(() => {
    if (!downloading) return;
    const id = setInterval(async () => {
      try {
        const s = await getStatusRef.current(baseUrl, fetchWithHeaders);
        if (s.repo_id !== repoId) return; // a different download took over
        setStatus(s);
        if (s.state === "done" && !notified.current) {
          notified.current = true;
          onDoneRef.current?.();
        } else if (s.state === "error" && !notified.current) {
          notified.current = true;
          // `s.error` / `s.message` are the backend's own text; only this
          // fallback is ours.
          onErrorRef.current?.(
            s.error ?? s.message ?? tRef.current("landing.hubDownload.failed"),
          );
        }
      } catch {
        // transient — retry next tick
      }
    }, POLL_MS);
    return () => clearInterval(id);
  }, [downloading, baseUrl, fetchWithHeaders, repoId]);

  const start = useCallback(async (): Promise<string | null> => {
    try {
      const res = await startDownloadRef.current(
        baseUrl,
        fetchWithHeaders,
        repoId,
      );
      if (!res.started) {
        return res.message ?? t("landing.hubDownload.couldNotStart");
      }
      notified.current = false;
      setStatus({
        state: "running",
        repo_id: repoId,
        message: res.message,
        error: null,
      });
      return null;
    } catch (e) {
      if (e instanceof ApiError && e.detail) return e.detail;
      return e instanceof Error
        ? e.message
        : t("landing.hubDownload.unreachable");
    }
  }, [t, baseUrl, fetchWithHeaders, repoId]);

  return { downloading, start };
}
