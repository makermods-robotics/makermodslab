import React, { useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import {
  Download as DownloadIcon,
  ExternalLink,
  Loader2,
  Trash2,
  Upload as UploadIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { ApiError } from "@/lib/apiClient";
import { policyTypeDisplayName } from "@/components/training/types";
import { ModelInfo, getModelInfo, uploadModel } from "@/lib/modelsApi";
import { useModelDownload } from "@/hooks/useModelDownload";

/** 16000 -> "16k", 950 -> "950". Steps get a compact form like the dataset
 * card's frame counts. */
const formatCount = (n: number): string => {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(n);
};

const formatBytes = (bytes: number | null | undefined): string => {
  // Null-safe: an unknown size renders nothing rather than "null B". Callers
  // still gate the whole Size row on presence, so this is belt-and-suspenders.
  if (bytes == null) return "";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
};

const Row: React.FC<{ label: string; children: React.ReactNode }> = ({
  label,
  children,
}) => (
  <div className="flex items-baseline gap-2">
    <span className="w-14 shrink-0 text-muted-foreground">{label}</span>
    <span className="min-w-0 flex-1 break-all text-foreground">{children}</span>
  </div>
);

/** True when the logged-in user can push to their own namespace — the gate for
 * offering Upload on a local model. Mirrors DatasetInfoCard's useCanEditHub for
 * a bare (own-namespace) target: false while loading / unauthenticated. */
const useCanUpload = (): boolean => {
  const { auth } = useHfAuth();
  if (auth.status !== "authenticated") return false;
  return auth.username != null && auth.writableNamespaces.length > 0;
};

/**
 * "Download to this machine" affordance for a model with no local checkpoint
 * (the models mirror of DatasetInfoCard's HubDownloadRow). Starts a background
 * download into the local models dir and, while it runs, shows a "Downloading…"
 * state that survives navigation (useModelDownload re-attaches by polling on
 * mount). On completion it fires onDownloaded so the parent re-reads
 * /models/info (now local) and refreshes the listing (source flips to "both").
 */
const ModelDownloadRow: React.FC<{
  repoId: string;
  onDownloaded: () => void;
}> = ({ repoId, onDownloaded }) => {
  const { t } = useTranslation();
  const { toast } = useToast();
  const { downloading, start } = useModelDownload({
    repoId,
    onDone: () => {
      toast({
        title: t("landing.modelInfo.download.doneTitle"),
        description: t("landing.modelInfo.download.doneBody", { repoId }),
      });
      onDownloaded();
    },
    onError: (message) => {
      toast({
        // `message` comes from the backend / the hook's own fallback.
        title: t("landing.modelInfo.download.failedTitle"),
        description: message,
        variant: "destructive",
      });
    },
  });

  if (downloading) {
    return (
      <div className="flex items-center gap-1.5 text-muted-foreground">
        <Loader2 className="h-3 w-3 animate-spin" />
        <span>{t("landing.modelInfo.download.inProgress")}</span>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-muted-foreground">
        {t("landing.modelInfo.download.notDownloaded")}
      </span>
      <Button
        size="sm"
        variant="outline"
        onClick={async () => {
          const err = await start();
          if (err) {
            toast({
              title: t("landing.modelInfo.download.startFailedTitle"),
              description: err,
              variant: "destructive",
            });
          }
        }}
        className="h-6 gap-1 border-blue-500/50 px-2 text-xs text-blue-700 dark:text-blue-300 hover:bg-blue-500/10"
      >
        <DownloadIcon className="h-3 w-3" />
        {t("landing.modelInfo.download.button")}
      </Button>
    </div>
  );
};

interface ModelInfoCardProps {
  /** The selected model id (local run id or Hub repo id). */
  id: string;
  /** True when the selection is a local-only model — gates the Upload
   * affordance (LOCAL-only, mirroring the dataset card). */
  isLocal?: boolean;
  /** When true, show the trash affordance. Local models (deleting the run
   * output) and pinned custom Hub models (where it just unpins) qualify —
   * mirroring the dataset card's canDelete. Defaults to `isLocal`. */
  canDelete?: boolean;
  /** Invoked when the user clicks Delete. Routed through the parent's confirm
   * dialog; nothing is deleted inline. */
  onDelete?: () => void;
  /** Called after a successful upload so the parent can refresh the listing
   * (the model flips local -> both). */
  onUploaded?: () => void;
  /** Called after a Hub model's checkpoint finishes downloading to the local
   * models dir, so the parent can refresh the listing (source flips to
   * "both"). The card also re-reads its own /models/info to pick up the local
   * path/size. */
  onDownloaded?: () => void;
}

/**
 * Compact always-visible summary of the model selected on the home page:
 * policy type, base dataset, step count, on-disk size, and the local checkpoint
 * path (local) or Hub repo link (hub). Data comes from /models/info.
 *
 * Upload (local-only): pushes the checkpoint to the Hub as a public,
 * makermods/openbooth/MakerModsLab-tagged repo — the backend does the tagging. Delete
 * (local-only): removes the run's output dir. Both mirror the dataset card's
 * upload/delete gating (offered only for a local model whose namespace the user
 * can write to / whose local copy they own).
 */
const ModelInfoCard: React.FC<ModelInfoCardProps> = ({
  id,
  isLocal = false,
  canDelete,
  onDelete,
  onUploaded,
  onDownloaded,
}) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const canUpload = useCanUpload();
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<{ notFound: boolean } | null>(null);
  const [uploading, setUploading] = useState(false);
  // Bumped when a Hub model's checkpoint finishes downloading, to re-run the
  // info fetch (now local: /models/info picks up the path + on-disk size).
  const [infoRefreshKey, setInfoRefreshKey] = useState(0);
  const showDelete = canDelete ?? isLocal;

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setInfo(null);
    setError(null);
    getModelInfo(baseUrl, fetchWithHeaders, id, controller.signal)
      .then((data) => {
        setInfo(data);
        setLoading(false);
      })
      .catch((e) => {
        if (controller.signal.aborted) return;
        setError({ notFound: e instanceof ApiError && e.status === 404 });
        setLoading(false);
      });
    return () => controller.abort();
  }, [baseUrl, fetchWithHeaders, id, infoRefreshKey]);

  const doUpload = async () => {
    setUploading(true);
    try {
      const res = await uploadModel(baseUrl, fetchWithHeaders, id);
      toast({
        title: t("landing.modelInfo.uploadedTitle"),
        description: (
          <span>
            <Trans
              i18nKey="landing.modelInfo.uploadedBody"
              values={{ repoId: res.repo_id }}
              components={[
                <a
                  key="0"
                  href={res.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline font-medium"
                />,
              ]}
            />
          </span>
        ),
      });
      onUploaded?.();
    } catch (e) {
      toast({
        // The description is the backend detail / raw error — English server
        // prose, surfaced verbatim.
        title: t("landing.modelInfo.uploadFailedTitle"),
        description:
          e instanceof ApiError && e.detail
            ? e.detail
            : e instanceof Error
              ? e.message
              : String(e),
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="rounded-md border border-border bg-muted/40 px-3 py-2 text-xs">
      {loading && (
        <div
          className="animate-pulse space-y-2 py-0.5"
          aria-label={t("landing.modelInfo.loadingAria")}
        >
          <div className="h-3 w-3/4 rounded bg-muted" />
          <div className="h-3 w-1/2 rounded bg-muted" />
          <div className="h-3 w-2/3 rounded bg-muted" />
        </div>
      )}

      {!loading && error && (
        <p className="text-muted-foreground">
          {error.notFound
            ? t("landing.modelInfo.notFound")
            : t("landing.modelInfo.loadError")}
        </p>
      )}

      {!loading && info && (
        <div className="space-y-1.5">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0 flex-1 font-medium text-foreground">
              {/* Policy type when known; otherwise fall back to something real
                  (display name / repo id / run id) — never the string "unknown"
                  as the card's title. */}
              {info.policy_type
                ? policyTypeDisplayName(info.policy_type)
                : info.name || info.hf_repo_id || info.id}
              {info.steps != null && (
                <span className="text-muted-foreground">
                  {" · "}
                  {t("landing.modelInfo.steps", {
                    steps: formatCount(info.steps),
                  })}
                </span>
              )}
              {info.private && (
                <span className="ml-2 align-middle text-xs font-normal text-amber-600 dark:text-amber-400">
                  {t("landing.modelInfo.private")}
                </span>
              )}
            </div>
            {/* Delete — local models (removing the run output) or pinned
                custom Hub models (just unpins). Routed through the parent's
                confirm dialog; nothing inline. */}
            {showDelete && onDelete && (
              <button
                type="button"
                onClick={onDelete}
                aria-label={t("landing.modelInfo.deleteAria")}
                title={t("landing.modelInfo.deleteAria")}
                className="-mr-1 -mt-0.5 shrink-0 rounded p-1 text-muted-foreground hover:text-destructive"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            )}
          </div>

          {/* Base dataset: omit when unknown rather than render "unknown". */}
          {info.dataset && (
            <Row label={t("landing.modelInfo.rowDataset")}>{info.dataset}</Row>
          )}

          {info.size_bytes != null && (
            <Row label={t("landing.modelInfo.rowSize")}>
              {formatBytes(info.size_bytes)}
            </Row>
          )}

          {/* Hub-only view: when the repo was last pushed to (the local view's
              timestamp is the run end, already implicit in the run name). */}
          {!info.path && info.last_modified && (
            <Row label={t("landing.modelInfo.rowUpdated")}>
              {/* Date formatting is deliberately untouched — the value keeps
                  rendering exactly as it does today. */}
              {new Date(info.last_modified).toLocaleDateString()}
            </Row>
          )}

          {/* Location: the Hub repo (link) for a hub/both model, else the local
              checkpoint path. */}
          {info.hf_repo_id ? (
            <Row label={t("landing.modelInfo.rowHub")}>
              <a
                href={`https://huggingface.co/${info.hf_repo_id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-foreground hover:text-foreground"
              >
                {info.hf_repo_id}
                <ExternalLink className="h-3 w-3 shrink-0" />
              </a>
            </Row>
          ) : info.path ? (
            <Row label={t("landing.modelInfo.rowPath")}>
              <span className="font-mono">{info.path}</span>
            </Row>
          ) : null}

          {/* Download — a Hub model with no local checkpoint (path is null)
              can be fetched into the local models dir so inference works
              offline. Mirrors the dataset card's hub-only download row. */}
          {!info.path && info.hf_repo_id && (
            <div className="mt-1.5 border-t border-border pt-1.5">
              <ModelDownloadRow
                repoId={info.hf_repo_id}
                onDownloaded={() => {
                  setInfoRefreshKey((k) => k + 1);
                  onDownloaded?.();
                }}
              />
            </div>
          )}

          {/* Upload — local-only, gated on Hub write access, same as the
              dataset card. A "both" model is already on the Hub, so no upload. */}
          {isLocal && canUpload && (
            <div className="mt-1.5 flex items-center justify-end border-t border-border pt-1.5">
              <Button
                size="sm"
                variant="outline"
                onClick={doUpload}
                disabled={uploading}
                className="h-6 gap-1 border-teal-500/50 px-2 text-xs text-teal-700 dark:text-teal-300 hover:bg-teal-500/10"
              >
                {uploading ? (
                  <>
                    <Loader2 className="h-3 w-3 animate-spin" />
                    {t("landing.modelInfo.uploading")}
                  </>
                ) : (
                  <>
                    <UploadIcon className="h-3 w-3" />
                    {t("landing.modelInfo.upload")}
                  </>
                )}
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ModelInfoCard;
