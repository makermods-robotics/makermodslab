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
import { Loader2, Trash2, HardDrive, AlertTriangle } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import {
  DatasetItem,
  deleteDataset,
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

  // Datasets whose local cache can be cleared = cached AND on the Hub.
  const cached = datasets.filter((d) => d.source === "both");

  // Per-row on-disk size, fetched lazily from the info endpoint. A row with no
  // entry (fetch pending or failed) simply omits its size.
  const [sizes, setSizes] = useState<Record<string, number>>({});
  // Repo ids currently being cleared (per-row spinner + disabled buttons).
  const [clearing, setClearing] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  // HF_HUB_OFFLINE on the backend: a cleared cache can't be re-downloaded.
  const [offline, setOffline] = useState(false);

  // On open: reset transient state and fetch sizes + the offline signal.
  useEffect(() => {
    if (!open) return;
    setError(null);
    setClearing(new Set());
    setSizes({});

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

  const clearAll = async () => {
    setError(null);
    for (const d of cached) {
      // Sequential so failures surface one at a time and the backend isn't
      // hammered with concurrent deletes.
      await clearOne(d.repo_id);
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
                return (
                  <div
                    key={d.repo_id}
                    className="flex items-start gap-2 p-2 text-sm"
                  >
                    <span className="min-w-0 flex-1 break-all">{d.repo_id}</span>
                    {size != null && (
                      <span className="shrink-0 text-xs text-muted-foreground">
                        {formatBytes(size)}
                      </span>
                    )}
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={isClearing || busy}
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

          {error ? <p className="text-sm text-destructive">{error}</p> : null}

          <div className="flex items-center justify-between">
            <Button
              variant="outline"
              onClick={() => onOpenChange(false)}
            >
              {t("common.close")}
            </Button>
            {cached.length > 0 && (
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
                      n: cached.length,
                    })}
                  </>
                )}
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ManageCachesDialog;
