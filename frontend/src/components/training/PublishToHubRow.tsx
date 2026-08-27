import React, { useCallback, useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useCanUpload } from "@/hooks/useCanUpload";
import { useModelPublish } from "@/hooks/useModelPublish";
import { RunCheckpoints, listRunCheckpoints } from "@/lib/modelsApi";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Loader2, Upload as UploadIcon } from "lucide-react";

/** step 0 is the sentinel an imported single-model checkpoint carries (lerobot
 * never saves at step 0) — same label rule as CheckpointDropdown. */
const stepLabel = (step: number) => (step === 0 ? "latest" : `step ${step}`);

/**
 * The training dialog's "Publish to Hub" row: pick any subset of a finished
 * run's checkpoints and queue them to ONE Hub model repo.
 *
 * The one-repo-per-run rule is the whole point of the row's shape. Every step
 * lands under `checkpoints/<step>/pretrained_model` in the same repo, so a run
 * has a single model card no matter how many times it is published — which is
 * why the repo name is editable only before the first publish, and why a later
 * visit reads "add checkpoints" against a pinned repo instead of offering a
 * fresh target the user could accidentally fork the run across.
 *
 * The queue lives on the server (see useModelPublish), so closing the dialog
 * mid-publish doesn't cancel or lose it — reopening re-attaches to the same
 * progress.
 */
const PublishToHubRow: React.FC<{
  jobId: string;
  /** Refetch the job — a first publish pins its hf_repo_id, which the dialog
   * header renders as "View on Hub". */
  onPublished: () => void;
}> = ({ jobId, onPublished }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { auth } = useHfAuth();
  const canUpload = useCanUpload();
  const username = auth.status === "authenticated" ? auth.username : null;

  const [data, setData] = useState<RunCheckpoints | null>(null);
  const [repoId, setRepoId] = useState("");
  const [selected, setSelected] = useState<number[]>([]);
  const [open, setOpen] = useState(false);
  const submittingRef = useRef(false);

  const refresh = useCallback(
    (signal?: AbortSignal) =>
      listRunCheckpoints(baseUrl, fetchWithHeaders, jobId, signal)
        .then(setData)
        .catch(() => {
          // 404 while the run has no checkpoint yet, or a transient failure —
          // the row simply doesn't render rather than showing a broken state.
        }),
    [baseUrl, fetchWithHeaders, jobId],
  );

  useEffect(() => {
    const controller = new AbortController();
    refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  const { publishing, status, publish } = useModelPublish({
    modelId: jobId,
    onDone: (s) => {
      const n = s.done_steps.length;
      toast({
        title: `Published ${n} checkpoint${n === 1 ? "" : "s"}`,
        description: (
          <span>
            {s.repo_id} is on the Hub.{" "}
            {s.url && (
              <a
                href={s.url}
                target="_blank"
                rel="noopener noreferrer"
                className="underline font-medium"
              >
                View model
              </a>
            )}
          </span>
        ),
      });
      setSelected([]);
      void refresh();
      onPublished();
    },
    onError: (message, s) => {
      const landed = s.done_steps.length;
      toast({
        title: "Publish failed",
        description: landed
          ? `${message} (${landed} checkpoint${landed === 1 ? "" : "s"} already published — retry the rest.)`
          : message,
        variant: "destructive",
      });
      // Some steps may still have landed; re-read so their badges are right.
      void refresh();
      onPublished();
    },
  });

  const checkpoints = data?.checkpoints ?? [];
  const unpublished = checkpoints.filter((c) => !c.published);
  const publishedCount = checkpoints.length - unpublished.length;

  // Default selection, applied only on the OPEN transition: the newest
  // checkpoint that isn't on the Hub yet. Keeps the common case one click, and
  // never silently queues a whole run's worth of weights ("select all" makes
  // that an explicit, counted choice). Gated on a ref rather than on
  // `selected.length === 0` so that clearing the list stays cleared.
  const wasOpen = useRef(false);
  useEffect(() => {
    if (open && !wasOpen.current) {
      const newest = unpublished[unpublished.length - 1];
      setSelected(newest ? [newest.step] : []);
    }
    wasOpen.current = open;
  }, [open, unpublished]);

  // Drop steps that have since been published (or vanished) from the selection.
  // After a partial failure this turns the stale selection into exactly "the
  // rest", which is what the failure toast tells the user to retry — leaving
  // the landed steps selected would re-upload gigabytes to no effect.
  useEffect(() => {
    if (!data) return;
    const publishable = new Set(
      data.checkpoints.filter((c) => !c.published).map((c) => c.step),
    );
    setSelected((prev) => {
      const next = prev.filter((s) => publishable.has(s));
      return next.length === prev.length ? prev : next;
    });
  }, [data]);

  if (!canUpload || checkpoints.length === 0) return null;

  const pinnedRepo = data?.hf_repo_id ?? null;
  const placeholder =
    data?.default_repo_id ??
    (username ? `${username}/${jobId}` : "namespace/repo-name");

  const toggle = (step: number) =>
    setSelected((prev) =>
      prev.includes(step) ? prev.filter((s) => s !== step) : [...prev, step],
    );

  // "Select all" means all the checkpoints that AREN'T up there yet — the
  // published ones are exactly what the user doesn't need to send again.
  const allSelected =
    unpublished.length > 0 && unpublished.every((c) => selected.includes(c.step));
  const selectAll = () =>
    setSelected(allSelected ? [] : unpublished.map((c) => c.step));

  const onConfirm = async () => {
    // `publishing` only flips once the POST resolves, so a fast second click
    // would fire a second POST and collect a 409 toast. This guard is
    // synchronous.
    if (selected.length === 0 || submittingRef.current) return;
    submittingRef.current = true;
    setOpen(false);
    try {
      const err = await publish(
        pinnedRepo ?? (repoId.trim() || undefined),
        [...selected].sort((a, b) => a - b),
      );
      if (err) {
        toast({
          title: "Publish failed",
          description: err,
          variant: "destructive",
        });
      }
    } finally {
      submittingRef.current = false;
    }
  };

  const hubUnknown = data != null && !data.hub_readable;

  const total = status?.total ?? 0;
  const progressPct = total > 0 ? Math.round((status!.done / total) * 100) : 0;
  // One checkpoint has no meaningful intermediate progress — see the bar below.
  const indeterminate = total <= 1;
  const progressLabel =
    (total > 1
      ? `Uploading ${Math.min((status?.done ?? 0) + 1, total)} of ${total}`
      : "Uploading") +
    (status?.current_step != null ? ` · ${stepLabel(status.current_step)}` : "");

  return (
    <div className="rounded-md border border-border bg-card p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <span className="eyebrow">Publish to Hub</span>
          {pinnedRepo ? (
            <p className="mt-0.5 truncate text-xs text-muted-foreground">
              <a
                href={`https://huggingface.co/${pinnedRepo}`}
                target="_blank"
                rel="noreferrer"
                className="text-primary hover:underline"
              >
                {pinnedRepo}
              </a>{" "}
              {hubUnknown
                ? "· couldn't check which checkpoints are published"
                : `· ${publishedCount} of ${checkpoints.length} checkpoints published`}
            </p>
          ) : (
            <p className="mt-0.5 text-xs text-muted-foreground">
              Share this run's checkpoints as a public model on the Hub — every
              step you pick lands in one repo, under one model card.
            </p>
          )}
        </div>

        {publishing ? (
          <div className="w-52 shrink-0">
            <div
              className="flex items-center justify-between text-[11px] text-muted-foreground"
              aria-live="polite"
            >
              <span>{progressLabel}</span>
              <Loader2 className="h-3 w-3 animate-spin" />
            </div>
            <div
              className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-muted"
              role="progressbar"
              aria-label="Publishing checkpoints"
              aria-valuemin={0}
              aria-valuemax={status?.total ?? 1}
              // Omitted for a single checkpoint: with no byte-level progress
              // from the Hub the bar would sit at 0% for the whole upload, so
              // it runs indeterminate instead of lying about being stalled.
              aria-valuenow={indeterminate ? undefined : status?.done}
            >
              <div
                className={
                  indeterminate
                    ? "h-full w-1/3 animate-pulse rounded-full bg-teal-500"
                    : "h-full rounded-full bg-teal-500 transition-all"
                }
                style={indeterminate ? undefined : { width: `${progressPct}%` }}
              />
            </div>
          </div>
        ) : (
          <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
              <Button
                size="sm"
                variant="outline"
                className="h-8 shrink-0 gap-1.5 border-teal-500/50 text-teal-700 hover:bg-teal-500/10 dark:text-teal-300"
              >
                <UploadIcon className="h-3.5 w-3.5" />
                {pinnedRepo ? "Add checkpoints" : "Upload to Hub"}
              </Button>
            </PopoverTrigger>
            <PopoverContent align="end" className="w-80 text-xs">
              <div className="space-y-3">
                {pinnedRepo ? (
                  <p className="leading-snug text-muted-foreground">
                    Adding to{" "}
                    <span className="font-mono text-foreground">
                      {pinnedRepo}
                    </span>
                    . A run keeps one repo, so every checkpoint stays under the
                    same model card.
                  </p>
                ) : (
                  <div className="space-y-1">
                    <Label
                      htmlFor={`publish-repo-id-${jobId}`}
                      className="font-normal text-muted-foreground"
                    >
                      Repo name (optional)
                    </Label>
                    <Input
                      id={`publish-repo-id-${jobId}`}
                      value={repoId}
                      onChange={(e) => setRepoId(e.target.value)}
                      placeholder={placeholder}
                      className="h-7 text-xs"
                    />
                    <p className="leading-snug text-muted-foreground">
                      Leave blank to publish as{" "}
                      <span className="font-mono">{placeholder}</span>. Later
                      checkpoints go to this same repo.
                    </p>
                  </div>
                )}

                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <Label className="font-normal text-muted-foreground">
                      Checkpoints
                    </Label>
                    {unpublished.length > 0 && (
                      <button
                        type="button"
                        onClick={selectAll}
                        className="text-[11px] text-primary hover:underline"
                      >
                        {allSelected
                          ? "Clear all"
                          : `Select all (${unpublished.length})`}
                      </button>
                    )}
                  </div>
                  {hubUnknown && (
                    <p className="leading-snug text-muted-foreground">
                      Couldn't reach the Hub to check which checkpoints are
                      already published — the badges below may be incomplete.
                    </p>
                  )}
                  <div className="max-h-44 space-y-0.5 overflow-y-auto rounded border border-border p-1">
                    {checkpoints.map((c) => (
                      <label
                        key={c.step}
                        className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 hover:bg-muted"
                      >
                        <Checkbox
                          checked={selected.includes(c.step)}
                          onCheckedChange={() => toggle(c.step)}
                          className="h-3.5 w-3.5"
                        />
                        <span className="flex-1">{stepLabel(c.step)}</span>
                        {c.published && (
                          <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                            published
                          </span>
                        )}
                      </label>
                    ))}
                  </div>
                  {selected.length > 1 && (
                    <p className="leading-snug text-muted-foreground">
                      {selected.length} checkpoints upload one after another —
                      each is a full copy of the policy weights.
                    </p>
                  )}
                  {selected.some(
                    (s) => checkpoints.find((c) => c.step === s)?.published,
                  ) && (
                    <p className="leading-snug text-muted-foreground">
                      Re-selecting a published checkpoint overwrites it in place.
                    </p>
                  )}
                </div>

                <Button
                  size="sm"
                  onClick={onConfirm}
                  disabled={selected.length === 0}
                  className="h-7 w-full gap-1 text-xs"
                >
                  <UploadIcon className="h-3 w-3" />
                  {selected.length === 0
                    ? "Select a checkpoint"
                    : `Upload ${selected.length} checkpoint${selected.length === 1 ? "" : "s"}`}
                </Button>
              </div>
            </PopoverContent>
          </Popover>
        )}
      </div>

      {data?.legacy_root_checkpoint && (
        <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
          This repo also holds a checkpoint at its root from an earlier upload.
          It stays readable, but the step-addressed copies above are what tools
          load.
        </p>
      )}
    </div>
  );
};

export default PublishToHubRow;
