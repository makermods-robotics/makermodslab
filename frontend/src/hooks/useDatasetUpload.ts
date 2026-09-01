import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useApi } from "@/contexts/ApiContext";
import { ApiError } from "@/lib/apiClient";
import {
  UploadStatus,
  getDatasetUploadStatus,
  uploadDataset,
} from "@/lib/replayApi";

const POLL_MS = 1500;

interface UseDatasetUploadArgs {
  /** The dataset this hook cares about. It only reports "uploading" / fires
   * callbacks when the single global upload is for THIS repo. */
  repoId: string;
  /** Fired once when this dataset's upload finishes, with the Hub URL. */
  onDone?: (url: string) => void;
  /** Fired once when this dataset's upload fails, with the message (and an
   * optional setup-guide URL for auth failures). */
  onError?: (message: string, docsUrl?: string | null) => void;
}

interface UseDatasetUploadResult {
  /** True while the global upload is running AND it's for this repoId. Drives
   * the row/card "Uploading…" state; survives navigation because the hook
   * re-attaches by polling /upload-status on mount. */
  uploading: boolean;
  /** Kick off the upload. Returns null on success, or an error message when the
   * start was refused (409: already running / dataset busy) or unreachable. */
  start: (tags: string[], isPrivate: boolean) => Promise<string | null>;
}

/**
 * Drives one dataset's background Hub upload. There is a single server-side
 * upload at a time (see UploadManager); this hook starts it and, while it runs
 * for THIS repoId, polls /upload-status until done|error — then fires onDone /
 * onError exactly once. Polling only runs while an upload for this repo is in
 * flight (no permanent loop), and the hook seeds itself from the backend on
 * mount so a card that mounts mid-upload (e.g. after navigating back) shows the
 * live "Uploading…" state instead of a stale idle one.
 */
export function useDatasetUpload({
  repoId,
  onDone,
  onError,
}: UseDatasetUploadArgs): UseDatasetUploadResult {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const [status, setStatus] = useState<UploadStatus | null>(null);
  // Guards the one-shot callbacks so a lingering "done"/"error" status isn't
  // re-notified on every poll tick.
  const notified = useRef(false);

  // Latest callbacks without retriggering the poll effect on every render.
  // `t` rides along for the same reason: the poll effect must not restart (and
  // re-arm its one-shot callbacks) just because the UI language changed.
  const onDoneRef = useRef(onDone);
  const onErrorRef = useRef(onError);
  const tRef = useRef(t);
  onDoneRef.current = onDone;
  onErrorRef.current = onError;
  tRef.current = t;

  const isMine = status?.repo_id === repoId;
  const uploading = isMine && status?.state === "running";

  // Seed from the backend on mount / repoId change so we re-attach to an
  // already-running upload (survives navigation). If the running upload is for
  // a different repo, this card just ignores it.
  useEffect(() => {
    let cancelled = false;
    notified.current = false;
    getDatasetUploadStatus(baseUrl, fetchWithHeaders)
      .then((s) => {
        if (cancelled) return;
        // Only latch onto a status that's for THIS repo and in flight; a stale
        // done/error for this repo from a prior visit shouldn't re-toast.
        setStatus(s.repo_id === repoId && s.state === "running" ? s : null);
      })
      .catch(() => {
        if (!cancelled) setStatus(null);
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders, repoId]);

  // Poll while this repo's upload runs; on the running -> done|error edge fire
  // the one-shot callback and stop polling (state !== running ends the effect).
  useEffect(() => {
    if (!uploading) return;
    const id = setInterval(async () => {
      try {
        const s = await getDatasetUploadStatus(baseUrl, fetchWithHeaders);
        if (s.repo_id !== repoId) return; // a different upload took over
        setStatus(s);
        if (s.state === "done" && !notified.current) {
          notified.current = true;
          onDoneRef.current?.(
            s.dataset_url ?? `https://huggingface.co/datasets/${repoId}`,
          );
        } else if (s.state === "error" && !notified.current) {
          notified.current = true;
          onErrorRef.current?.(
            // `s.message` is the backend's own text; only this fallback is ours.
            s.message ?? tRef.current("landing.hubUpload.failed"),
            s.docs_url ?? null,
          );
        }
      } catch {
        // transient — retry next tick
      }
    }, POLL_MS);
    return () => clearInterval(id);
  }, [uploading, baseUrl, fetchWithHeaders, repoId]);

  const start = useCallback(
    async (tags: string[], isPrivate: boolean): Promise<string | null> => {
      try {
        const res = await uploadDataset(
          baseUrl,
          fetchWithHeaders,
          repoId,
          tags,
          isPrivate,
        );
        if (!res.started) {
          return res.message ?? t("landing.hubUpload.couldNotStart");
        }
        // Seed a running status so the poll effect attaches immediately.
        notified.current = false;
        setStatus({
          state: "running",
          repo_id: repoId,
          message: res.message,
          dataset_url: null,
        });
        return null;
      } catch (e) {
        // 409 (already running / dataset busy) and any other failure surface
        // their message; ApiError carries the backend detail.
        if (e instanceof ApiError && e.detail) return e.detail;
        return e instanceof Error
          ? e.message
          : t("landing.hubUpload.unreachable");
      }
    },
    [t, baseUrl, fetchWithHeaders, repoId],
  );

  return { uploading, start };
}
