import React, { useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Loader2, Trash2, HardDrive, AlertTriangle, GitMerge } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import {
  DatasetItem,
  MergeCleanupResult,
  cleanupTemporaryMerges,
  deleteDataset,
  getDatasetHubStatus,
  getDatasetInfo,
} from "@/lib/replayApi";
import { listRunnerHardware } from "@/lib/jobsApi";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasets: DatasetItem[];
  /** Called after one or more caches are cleared so the parent can refresh the
   * list (flips the cleared rows source both -> hub everywhere). */
  onCleared: () => void;
}

const formatBytes = (bytes: number): string => {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
};

const ManageCachesDialog: React.FC<Props> = ({
  open,
  onOpenChange,
  datasets,
  onCleared,
}) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

  // Datasets whose local cache can be cleared = cached AND on the Hub.
  const cached = datasets.filter((d) => d.source === "both");

  // Throwaway merges minted for a combine-and-train launch. Filtered
  // independently of `cached`: a temp merge that was never pushed to the Hub is
  // `source: "local"`, so it would miss the "both" list entirely.
  const tempMerges = datasets.filter((d) => d.merge?.temporary);
  const [cleanupConfirm, setCleanupConfirm] = useState(false);
  const [cleaningUp, setCleaningUp] = useState(false);

  // Per-row on-disk size, fetched lazily from the info endpoint. A row with no
  // entry (fetch pending or failed) simply omits its size.
  const [sizes, setSizes] = useState<Record<string, number>>({});
  // Repo ids currently being cleared (per-row spinner + disabled buttons).
  const [clearing, setClearing] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  // HF_HUB_OFFLINE on the backend: a cleared cache can't be re-downloaded.
  const [offline, setOffline] = useState(false);
  // Rows whose Hub repo EXISTS but holds no dataset (hub_has_data === false):
  // an upload that died after creating the repo. This dialog's whole premise —
  // "the Hub copy stays" — is false for them: clearing would delete the only
  // real copy. They stay listed (so the state is visible) but aren't clearable
  // until a re-upload fills the repo. No claim (null) leaves a row clearable.
  const [notBackedUp, setNotBackedUp] = useState<Set<string>>(new Set());
  // Rows whose hub-status fetch has SETTLED (answered or failed). Until then a
  // row is not clearable: the guard above only holds once the answer is in,
  // and the slow-network window where the fetch is still in flight is exactly
  // when a half-uploaded repo would otherwise be one click from deletion.
  const [statusSettled, setStatusSettled] = useState<Set<string>>(new Set());

  // On open: reset transient state and fetch sizes + the offline signal.
  useEffect(() => {
    if (!open) return;
    setError(null);
    setClearing(new Set());
    setSizes({});
    setNotBackedUp(new Set());
    setStatusSettled(new Set());

    let cancelled = false;
    listRunnerHardware(baseUrl, fetchWithHeaders)
      .then((h) => {
        if (!cancelled) setOffline(!!h.offline);
      })
      .catch(() => {
        if (!cancelled) setOffline(false);
      });

    for (const d of cached) {
      getDatasetInfo(baseUrl, fetchWithHeaders, d.repo_id)
        .then((info) => {
          // size_bytes is null for a hub-summary response (shouldn't happen
          // here — these rows are "both", so local info wins — but guard).
          const size = info.size_bytes;
          if (!cancelled && size != null)
            setSizes((prev) => ({ ...prev, [d.repo_id]: size }));
        })
        .catch(() => {
          // Size unavailable — the row just shows no size.
        });
      getDatasetHubStatus(baseUrl, fetchWithHeaders, d.repo_id)
        .then((s) => {
          if (!cancelled && s.hub_has_data === false)
            setNotBackedUp((prev) => new Set(prev).add(d.repo_id));
        })
        .catch(() => {
          // No claim — the row becomes clearable, like hub_has_data === null.
        })
        .finally(() => {
          if (!cancelled)
            setStatusSettled((prev) => new Set(prev).add(d.repo_id));
        });
    }

    return () => {
      cancelled = true;
    };
    // Re-run only when the dialog opens; `cached` is derived from `datasets`,
    // which is stable while the dialog is open.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, baseUrl, fetchWithHeaders]);

  const clearOne = async (repoId: string) => {
    setError(null);
    setClearing((prev) => new Set(prev).add(repoId));
    try {
      const res = await deleteDataset(baseUrl, fetchWithHeaders, repoId);
      if (!res.success) {
        // `res.message` is the backend's own explanation — English server
        // prose, surfaced verbatim. Only our fallback is translated.
        setError(
          res.message ?? t("landing.manageCaches.clearFailed", { repoId }),
        );
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setClearing((prev) => {
        const next = new Set(prev);
        next.delete(repoId);
        return next;
      });
      // Refresh the parent list so cleared rows flip both -> hub.
      onCleared();
    }
  };

  // Rows "Clear all" may touch: never one whose Hub repo is known-empty, and
  // never one whose status is still in flight (it could turn out to be).
  const clearable = cached.filter(
    (d) => statusSettled.has(d.repo_id) && !notBackedUp.has(d.repo_id),
  );

  const clearAll = async () => {
    setError(null);
    for (const d of clearable) {
      // Sequential so failures surface one at a time and the backend isn't
      // hammered with concurrent deletes.
      await clearOne(d.repo_id);
    }
  };

  const runCleanup = async () => {
    setCleaningUp(true);
    setError(null);
    try {
      const res: MergeCleanupResult = await cleanupTemporaryMerges(
        baseUrl,
        fetchWithHeaders,
        tempMerges.map((d) => d.repo_id),
      );
      const parts: string[] = [
        t("landing.manageCaches.cleanedUp", { count: res.deleted.length }),
      ];
      if (res.skipped.length > 0)
        parts.push(
          t("landing.manageCaches.cleanupSkipped", { count: res.skipped.length }),
        );
      if (res.hub_failed.length > 0)
        parts.push(
          t("landing.manageCaches.cleanupHubFailed", {
            count: res.hub_failed.length,
          }),
        );
      toast({
        title: t("landing.manageCaches.temporaryMergesTitle"),
        description: parts.join(" · "),
      });
      // Refresh the parent list so the removed merges drop out.
      onCleared();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setCleaningUp(false);
      setCleanupConfirm(false);
    }
  };

  const busy = clearing.size > 0;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <HardDrive className="w-5 h-5" />{" "}
            {t("landing.manageCaches.title")}
          </DialogTitle>
          <DialogDescription className="text-muted-foreground">
            {t("landing.manageCaches.description")}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {offline && (
            <div className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 p-2 text-xs text-amber-700 dark:text-amber-200">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>
                {/* The env var name is a literal; <Trans> keeps it inside the
                    sentence instead of splitting the copy around it. */}
                <Trans
                  i18nKey="landing.manageCaches.offlineNote"
                  components={[
                    <code
                      key="0"
                      className="text-amber-800 dark:text-amber-100"
                    />,
                  ]}
                />
              </span>
            </div>
          )}

          {cached.length === 0 ? (
            <p className="rounded-md border border-border p-3 text-sm text-muted-foreground">
              {t("landing.manageCaches.empty")}
            </p>
          ) : (
            <div className="max-h-72 overflow-auto rounded-md border border-border divide-y divide-border">
              {cached.map((d) => {
                const size = sizes[d.repo_id];
                const isClearing = clearing.has(d.repo_id);
                const unsafe =
                  notBackedUp.has(d.repo_id) || !statusSettled.has(d.repo_id);
                return (
                  <div
                    key={d.repo_id}
                    className="flex items-start gap-2 p-2 text-sm"
                  >
                    <div className="min-w-0 flex-1">
                      <span className="break-all">{d.repo_id}</span>
                      {notBackedUp.has(d.repo_id) && (
                        <p className="mt-0.5 flex items-center gap-1 text-xs text-amber-700 dark:text-amber-400">
                          <AlertTriangle className="h-3 w-3 shrink-0" />
                          {t("landing.manageCaches.notBackedUp")}
                        </p>
                      )}
                    </div>
                    {size != null && (
                      <span className="shrink-0 text-xs text-muted-foreground">
                        {formatBytes(size)}
                      </span>
                    )}
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={isClearing || busy || unsafe}
                      onClick={() => clearOne(d.repo_id)}
                      className="h-7 shrink-0"
                    >
                      {isClearing ? (
                        <>
                          <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                          {t("landing.manageCaches.clearing")}
                        </>
                      ) : (
                        <>
                          <Trash2 className="mr-1.5 h-3.5 w-3.5" />{" "}
                          {t("landing.manageCaches.clear")}
                        </>
                      )}
                    </Button>
                  </div>
                );
              })}
            </div>
          )}

          {tempMerges.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <GitMerge className="h-4 w-4 shrink-0" />
                <h3 className="text-sm font-medium">
                  {t("landing.manageCaches.temporaryMergesTitle")}
                </h3>
              </div>
              <p className="text-xs text-muted-foreground">
                {t("landing.manageCaches.temporaryMergesHint")}
              </p>
              <div className="max-h-48 overflow-auto rounded-md border border-border divide-y divide-border">
                {tempMerges.map((d) => (
                  <div
                    key={d.repo_id}
                    className="flex items-center gap-2 p-2 text-sm"
                  >
                    <span className="min-w-0 flex-1 break-all">{d.repo_id}</span>
                    {d.merge?.weighted && (
                      <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
                        {t("landing.manageCaches.weightedChip")}
                      </span>
                    )}
                    {d.merge && d.merge.source_count > 0 && (
                      <span className="shrink-0 text-xs text-muted-foreground">
                        {t("landing.manageCaches.sourceCount", {
                          n: d.merge.source_count,
                        })}
                      </span>
                    )}
                  </div>
                ))}
              </div>
              <Button
                variant="outline"
                size="sm"
                disabled={cleaningUp}
                onClick={() => setCleanupConfirm(true)}
                className="h-7"
              >
                {cleaningUp ? (
                  <>
                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                    {t("landing.manageCaches.cleaningUp")}
                  </>
                ) : (
                  <>
                    <Trash2 className="mr-1.5 h-3.5 w-3.5" />
                    {t("landing.manageCaches.cleanUp")}
                  </>
                )}
              </Button>
            </div>
          )}

          {error ? <p className="text-sm text-destructive">{error}</p> : null}

          <div className="flex items-center justify-between">
            <Button
              variant="outline"
              onClick={() => onOpenChange(false)}
            >
              {t("common.close")}
            </Button>
            {clearable.length > 0 && (
              <Button
                onClick={clearAll}
                disabled={busy}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                {busy ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />{" "}
                    {t("landing.manageCaches.clearing")}
                  </>
                ) : (
                  <>
                    <Trash2 className="mr-2 h-4 w-4" />{" "}
                    {t("landing.manageCaches.clearAll", {
                      n: clearable.length,
                    })}
                  </>
                )}
              </Button>
            )}
          </div>
        </div>
      </DialogContent>

      <AlertDialog open={cleanupConfirm} onOpenChange={setCleanupConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("landing.manageCaches.cleanUpConfirmTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t("landing.manageCaches.cleanUpConfirmBody")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={cleaningUp}>
              {t("common.cancel")}
            </AlertDialogCancel>
            <AlertDialogAction
              disabled={cleaningUp}
              onClick={(e) => {
                // Keep the dialog mounted until runCleanup resolves; it clears
                // cleanupConfirm itself in its finally block.
                e.preventDefault();
                void runCleanup();
              }}
            >
              {t("landing.manageCaches.cleanUp")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Dialog>
  );
};

export default ManageCachesDialog;
