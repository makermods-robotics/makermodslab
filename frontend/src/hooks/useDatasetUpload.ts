import { useCallback, useEffect, useRef, useState } from "react";
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
  /** Set by a caller that knows an upload for THIS repoId was just started
   * (moments ago, e.g. by CollectPanel's Finalize step) by someone other than
   * this hook instance. Normally the seed effect below only latches onto a
   * status that's still "running" at mount time, specifically so a stale
   * done/error left over from an unrelated, long-past visit doesn't re-toast.
   * That guard is wrong for this one case: the upload legitimately started
   * just before this component mounted, so a terminal status seen on the
   * very first fetch is that upload's real outcome, not stale history — it
   * still needs to reach the user. Defaults to false, which preserves
   * today's behavior exactly. */
  trustFirstSeed?: boolean;
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
 * live "Uploading…" state instead of a stale idle one. `trustFirstSeed` opts
 * a mount into also trusting a terminal (done/error) status on that first
 * seed, for the one case where the caller knows the upload it cares about was
 * started immediately beforehand by someone else — see its doc comment.
 */
export function useDatasetUpload({
  repoId,
  onDone,
  onError,
  trustFirstSeed = false,
}: UseDatasetUploadArgs): UseDatasetUploadResult {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [status, setStatus] = useState<UploadStatus | null>(null);
  // Guards the one-shot callbacks so a lingering "done"/"error" status isn't
  // re-notified on every poll tick.
  const notified = useRef(false);

  // Latest callbacks without retriggering the poll effect on every render.
  const onDoneRef = useRef(onDone);
  const onErrorRef = useRef(onError);
  onDoneRef.current = onDone;
  onErrorRef.current = onError;

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
        const mine = s.repo_id === repoId;
        // Only latch onto a status that's for THIS repo and in flight; a
        // stale done/error for this repo from a prior visit shouldn't
        // re-toast — UNLESS trustFirstSeed says an upload for this repo was
        // just started moments ago, in which case a terminal status seen
        // right here IS that upload's real (just-finished) outcome, not
        // stale history, and must still reach the user.
        if (!mine || (s.state !== "running" && !trustFirstSeed)) {
          setStatus(null);
          return;
        }
        setStatus(s);
        // The poll effect below only fires the one-shot callbacks on a
        // running -> done|error transition it observes itself. A trusted
        // first seed can land directly on a terminal state (the upload
        // finished before this hook ever saw it running), which the poll
        // effect never starts for — so fire the callback here too.
        if (s.state === "done" && !notified.current) {
          notified.current = true;
          onDoneRef.current?.(
            s.dataset_url ?? `https://huggingface.co/datasets/${repoId}`,
          );
        } else if (s.state === "error" && !notified.current) {
          notified.current = true;
          onErrorRef.current?.(s.message ?? "Upload failed.", s.docs_url ?? null);
        }
      })
      .catch(() => {
        if (!cancelled) setStatus(null);
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders, repoId, trustFirstSeed]);

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
            s.message ?? "Upload failed.",
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
          return res.message ?? "Upload could not be started.";
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
          : "Could not reach the backend to upload.";
      }
    },
    [baseUrl, fetchWithHeaders, repoId],
  );

  return { uploading, start };
}
