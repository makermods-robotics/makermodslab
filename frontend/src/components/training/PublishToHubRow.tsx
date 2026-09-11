import React, { useCallback, useEffect, useRef, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useCanUpload } from "@/hooks/useCanUpload";
import { useEyebrowClass } from "@/hooks/useEyebrowClass";
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

// step 0 is the sentinel an imported single-model checkpoint carries (lerobot
// never saves at step 0) — same label rule, and the SAME catalog keys, as
// CheckpointDropdown (resolved in-component; see stepLabel below).

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
  const { t } = useTranslation();
  const eyebrowClass = useEyebrowClass();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { auth } = useHfAuth();
  const canUpload = useCanUpload();
  const username = auth.status === "authenticated" ? auth.username : null;
  const writableNamespaces =
    auth.status === "authenticated" ? auth.writableNamespaces : [];

  const stepLabel = (step: number) =>
    step === 0
      ? t("jobs.checkpointDropdown.latest")
      : t("jobs.checkpointDropdown.step", { step: String(step) });

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
      toast({
        title: t("training.publish.toast.publishedTitle", {
          count: s.done_steps.length,
        }),
        description: (
          <span>
            <Trans
              i18nKey="training.publish.toast.publishedBody"
              values={{ repoId: s.repo_id ?? "" }}
              components={[
                // A missing url (never happens on the done path today) falls
                // back to plain text rather than an underlined dead link.
                s.url ? (
                  <a
                    key="0"
                    href={s.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline font-medium"
                  />
                ) : (
                  <span key="0" />
                ),
              ]}
            />
          </span>
        ),
      });
      setSelected([]);
      void refresh();
      onPublished();
    },
    onError: (message, s) => {
      const landed = s.done_steps.length;
      // `message` is the backend's own prose, surfaced verbatim; only the
      // retry hint appended after it is ours to translate.
      toast({
        title: t("training.publish.toast.failedTitle"),
        description: landed
          ? `${message} ${t("training.publish.toast.failedLanded", { count: landed })}`
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

  // Pre-flight check of the typed repo id, so an unwritable namespace is an
  // inline message at the input instead of a 403 surfacing minutes later
  // through publish-status. The VALUE submitted is untouched — this only
  // gates the button. A bare name (no slash) lands in the user's own
  // namespace and is always fine; the full grammar check stays server-side
  // (HFValidationError → 400).
  const typedRepoId = repoId.trim();
  const repoIdError = (() => {
    if (pinnedRepo || !typedRepoId) return null;
    const parts = typedRepoId.split("/");
    if (parts.length > 2 || parts.some((p) => p.length === 0)) {
      return t("training.publish.repoInvalid");
    }
    // Case-insensitive, like the backend's canonical_writable_namespace and
    // every other namespace gate — a locally-typed casing that differs from
    // whoami's must not block a push the backend would accept.
    const fold = parts[0].toLowerCase();
    if (
      parts.length === 2 &&
      !writableNamespaces.some((n) => n.toLowerCase() === fold)
    ) {
      return t("training.publish.repoNotWritable", { namespace: parts[0] });
    }
    return null;
  })();

  const onConfirm = async () => {
    // `publishing` only flips once the POST resolves, so a fast second click
    // would fire a second POST and collect a 409 toast. This guard is
    // synchronous.
    if (selected.length === 0 || repoIdError != null || submittingRef.current)
      return;
    submittingRef.current = true;
    setOpen(false);
    try {
      const err = await publish(
        pinnedRepo ?? (typedRepoId || undefined),
        [...selected].sort((a, b) => a - b),
      );
      if (err) {
        toast({
          title: t("training.publish.toast.failedTitle"),
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
  // "label · step" composition: the two halves are each a complete catalog
  // phrase; the "·" between them is a separator, not grammar.
  const progressLabel =
    (total > 1
      ? t("training.publish.uploadingOf", {
          current: Math.min((status?.done ?? 0) + 1, total),
          total,
        })
      : t("training.publish.uploading")) +
    (status?.current_step != null ? ` · ${stepLabel(status.current_step)}` : "");

  return (
    <div className="rounded-md border border-border bg-card p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <span className={eyebrowClass}>{t("training.publish.title")}</span>
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
              {"· "}
              {hubUnknown
                ? t("training.publish.hubUnknownShort")
                : t("training.publish.publishedOf", {
                    published: publishedCount,
                    total: checkpoints.length,
                  })}
            </p>
          ) : (
            <p className="mt-0.5 text-xs text-muted-foreground">
              {t("training.publish.intro")}
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
              aria-label={t("training.publish.publishingAria")}
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
                {pinnedRepo
                  ? t("training.publish.addCheckpoints")
                  : t("training.publish.uploadToHub")}
              </Button>
            </PopoverTrigger>
            <PopoverContent align="end" className="w-80 text-xs">
              <div className="space-y-3">
                {pinnedRepo ? (
                  <p className="leading-snug text-muted-foreground">
                    <Trans
                      i18nKey="training.publish.addingTo"
                      values={{ repo: pinnedRepo }}
                      components={[
                        <span key="0" className="font-mono text-foreground" />,
                      ]}
                    />
                  </p>
                ) : (
                  <div className="space-y-1">
                    <Label
                      htmlFor={`publish-repo-id-${jobId}`}
                      className="font-normal text-muted-foreground"
                    >
                      {t("training.publish.repoNameLabel")}
                    </Label>
                    <Input
                      id={`publish-repo-id-${jobId}`}
                      value={repoId}
                      onChange={(e) => setRepoId(e.target.value)}
                      placeholder={placeholder}
                      className="h-7 text-xs"
                    />
                    {repoIdError ? (
                      <p className="leading-snug text-destructive">
                        {repoIdError}
                      </p>
                    ) : (
                      <p className="leading-snug text-muted-foreground">
                        <Trans
                          i18nKey="training.publish.leaveBlank"
                          values={{ placeholder }}
                          components={[<span key="0" className="font-mono" />]}
                        />
                      </p>
                    )}
                  </div>
                )}

                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <Label className="font-normal text-muted-foreground">
                      {t("training.publish.checkpointsLabel")}
                    </Label>
                    {unpublished.length > 0 && (
                      <button
                        type="button"
                        onClick={selectAll}
                        className="text-[11px] text-primary hover:underline"
                      >
                        {allSelected
                          ? t("training.publish.clearAll")
                          : t("training.publish.selectAllCount", {
                              total: unpublished.length,
                            })}
                      </button>
                    )}
                  </div>
                  {hubUnknown && (
                    <p className="leading-snug text-muted-foreground">
                      {t("training.publish.hubUnknownDetail")}
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
                            {t("training.publish.publishedBadge")}
                          </span>
                        )}
                      </label>
                    ))}
                  </div>
                  {selected.length > 1 && (
                    <p className="leading-snug text-muted-foreground">
                      {t("training.publish.multiNote", {
                        count: selected.length,
                      })}
                    </p>
                  )}
                  {selected.some(
                    (s) => checkpoints.find((c) => c.step === s)?.published,
                  ) && (
                    <p className="leading-snug text-muted-foreground">
                      {t("training.publish.overwriteNote")}
                    </p>
                  )}
                </div>

                <Button
                  size="sm"
                  onClick={onConfirm}
                  disabled={selected.length === 0 || repoIdError != null}
                  className="h-7 w-full gap-1 text-xs"
                >
                  <UploadIcon className="h-3 w-3" />
                  {selected.length === 0
                    ? t("training.publish.selectPrompt")
                    : t("training.publish.uploadCount", {
                        count: selected.length,
                      })}
                </Button>
              </div>
            </PopoverContent>
          </Popover>
        )}
      </div>

      {data?.legacy_root_checkpoint && (
        <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
          {t("training.publish.legacyRootNote")}
        </p>
      )}
    </div>
  );
};

export default PublishToHubRow;
