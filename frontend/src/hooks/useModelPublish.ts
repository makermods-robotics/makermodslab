import { useCallback, useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { ApiError } from "@/lib/apiClient";
import {
  ModelUploadStatus,
  getModelUploadStatus,
  uploadModel,
} from "@/lib/modelsApi";

const POLL_MS = 1500;

interface UseModelPublishArgs {
  /** The local run this hook cares about. It only reports progress / fires
   * callbacks when the single global publish is for THIS run. */
  modelId: string;
  /** Fired once when this run's publish finishes, with the final status (which
   * carries the repo id, url, and every step that landed). */
  onDone?: (status: ModelUploadStatus) => void;
  /** Fired once when this run's publish fails, with the message. Steps that
   * landed before the failure are in the status's `done_steps` and stay on the
   * Hub — a retry only needs the rest. */
  onError?: (message: string, status: ModelUploadStatus) => void;
}

export interface UseModelPublishResult {
  /** True while the global publish is running AND it's for this modelId. */
  publishing: boolean;
  /** Live status while publishing (null when idle / someone else's publish).
   * Carries done/total/current_step so the caller can render queue position
   * rather than an opaque spinner. */
  status: ModelUploadStatus | null;
  /** Kick off a publish of `steps` (omit for the final checkpoint only).
   * Returns null on success, or an error message when the start was refused
   * (409: a publish is already running) or the backend was unreachable. */
  publish: (repoId?: string, steps?: number[]) => Promise<string | null>;
}

/**
 * Drives one run's background multi-checkpoint publish (POST /models/publish +
 * polled /models/publish-status) — the models-publish twin of useHubDownload,
 * against the same one-at-a-time start/poll backend shape.
 *
 * Why a poller rather than an awaited request: a publish is a SEQUENTIAL QUEUE
 * of checkpoints that can run for minutes, so the queue lives on the server and
 * the UI re-attaches to it. That is what lets the training dialog be closed and
 * reopened mid-publish and still show "3 of 8 · step 30000" — the hook seeds
 * itself from the backend on mount instead of holding the only copy of the
 * progress.
 *
 * Polling runs only while a publish for this run is in flight, and the one-shot
 * callbacks fire exactly once per publish (guarded across the lingering
 * done/error status the endpoint keeps reporting afterwards).
 */
export function useModelPublish({
  modelId,
  onDone,
  onError,
}: UseModelPublishArgs): UseModelPublishResult {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [status, setStatus] = useState<ModelUploadStatus | null>(null);
  const notified = useRef(false);

  const onDoneRef = useRef(onDone);
  const onErrorRef = useRef(onError);
  onDoneRef.current = onDone;
  onErrorRef.current = onError;

  const isMine = status?.model_id === modelId;
  const publishing = Boolean(isMine) && status?.state === "running";

  // Seed from the backend on mount / modelId change so a dialog reopened
  // mid-publish re-attaches to the running queue. A publish for a different run
  // is ignored (this hook speaks only for `modelId`).
  useEffect(() => {
    let cancelled = false;
    notified.current = false;
    getModelUploadStatus(baseUrl, fetchWithHeaders)
      .then((s) => {
        if (cancelled) return;
        if (s.model_id !== modelId || s.state === "idle") {
          setStatus(null);
          return;
        }
        // A publish that finished before this mount: keep the terminal status
        // readable, but never re-fire the one-shot callbacks for it — the
        // toast belonged to the session that started the publish, and the
        // caller's own data refetches already cover the outcome.
        if (s.state !== "running") notified.current = true;
        setStatus(s);
      })
      .catch(() => {
        if (!cancelled) setStatus(null);
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders, modelId]);

  // Poll while this run's publish is in flight; fire the one-shot callback on
  // the running -> done|error edge and stop.
  useEffect(() => {
    if (!publishing) return;
    const id = setInterval(async () => {
      try {
        const s = await getModelUploadStatus(baseUrl, fetchWithHeaders);
        if (s.model_id !== modelId) {
          // The global slot moved on without us ever seeing our own terminal
          // state — another run took over, or the server restarted and reset
          // the manager to idle. Either way this publish is no longer
          // observable, so stop claiming it is running rather than spinning
          // forever on a status that will never mention us again.
          setStatus(null);
          return;
        }
        setStatus(s);
        if (s.state === "done" && !notified.current) {
          notified.current = true;
          onDoneRef.current?.(s);
        } else if (s.state === "error" && !notified.current) {
          notified.current = true;
          onErrorRef.current?.(s.error ?? s.message ?? "Publish failed.", s);
        }
      } catch {
        // transient — retry next tick
      }
    }, POLL_MS);
    return () => clearInterval(id);
  }, [publishing, baseUrl, fetchWithHeaders, modelId]);

  const publish = useCallback(
    async (repoId?: string, steps?: number[]): Promise<string | null> => {
      try {
        const res = await uploadModel(
          baseUrl,
          fetchWithHeaders,
          modelId,
          repoId,
          steps,
        );
        if (!res.started) return res.message ?? "Publish could not be started.";
        notified.current = false;
        // Optimistic running state so the row switches immediately instead of
        // waiting up to a poll interval for the first tick.
        setStatus({
          state: "running",
          model_id: modelId,
          repo_id: repoId ?? null,
          url: null,
          message: res.message,
          error: null,
          total: steps?.length ?? 1,
          done: 0,
          current_step: null,
          done_steps: [],
        });
        return null;
      } catch (e) {
        if (e instanceof ApiError && e.detail) return e.detail;
        return e instanceof Error
          ? e.message
          : "Could not reach the backend to publish.";
      }
    },
    [baseUrl, fetchWithHeaders, modelId],
  );

  return { publishing, status, publish };
}
