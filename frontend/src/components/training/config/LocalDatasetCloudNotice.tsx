import React from "react";
import { Trans, useTranslation } from "react-i18next";
import { AlertTriangle, Loader2, UploadCloud, WifiOff } from "lucide-react";

interface LocalDatasetCloudNoticeProps {
  /** The local-only dataset the cloud run would train on. */
  repoId: string;
  /** Approximate on-disk size in bytes, if the info endpoint had it cheaply.
   * null when unknown (Hub-only detail, or the fetch hasn't resolved). */
  sizeBytes: number | null;
  /** Backend is in HF_HUB_OFFLINE mode: uploads are disabled, so training
   * can't proceed for this dataset at all. */
  offline: boolean;
  /** True while this dataset's upload is in flight (drives the progress line). */
  uploading: boolean;
  /** Last upload error, shown in-place so the user doesn't lose it to a toast. */
  errorMessage?: string | null;
}

// Byte units are the conventional binary abbreviations, not copy — left as is.
const formatSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(value >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
};

/**
 * Amber notice shown in the training config when a Hugging Face Cloud run is
 * targeted at a dataset that exists only on this machine. HF Jobs trains from
 * the Hub, so the dataset must be uploaded first — the Start button becomes
 * "Upload & start training" and the upload is chained before the job launches
 * (see Training.tsx). In offline mode the upload is impossible, so this turns
 * into a hard block instead.
 */
const LocalDatasetCloudNotice: React.FC<LocalDatasetCloudNoticeProps> = ({
  repoId,
  sizeBytes,
  offline,
  uploading,
  errorMessage,
}) => {
  const { t } = useTranslation();
  const sizeLabel = sizeBytes != null ? formatSize(sizeBytes) : null;

  if (offline) {
    return (
      <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-4 text-sm text-amber-700 dark:text-amber-100">
        <div className="flex items-start gap-2">
          <WifiOff className="w-4 h-4 mt-0.5 shrink-0 text-amber-600 dark:text-amber-300" />
          <div>
            <div className="font-semibold">
              {t("training.datasetNotice.title")}
            </div>
            <p className="mt-1 text-amber-700/80 dark:text-amber-200/80">
              <Trans
                i18nKey="training.datasetNotice.offline"
                values={{ repoId }}
                components={[
                  <code key="0" className="text-amber-700 dark:text-amber-100" />,
                  <span key="1" className="font-medium" />,
                ]}
              />
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-4 text-sm text-amber-700 dark:text-amber-100">
      <div className="flex items-start gap-2">
        <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0 text-amber-600 dark:text-amber-300" />
        <div className="w-full">
          <div className="font-semibold">
            {t("training.datasetNotice.title")}
          </div>
          {/* Two complete sentences rather than an inline "(~120 MB)" fragment
              spliced into one — the size sits in different places per language.
              {{size}} arrives pre-formatted from formatSize. */}
          <p className="mt-1 text-amber-700/80 dark:text-amber-200/80">
            <Trans
              i18nKey={
                sizeLabel
                  ? "training.datasetNotice.bodyWithSize"
                  : "training.datasetNotice.body"
              }
              values={{ repoId, size: sizeLabel ?? "" }}
              components={[
                <span key="0" className="font-medium" />,
                <span key="1" className="font-medium" />,
              ]}
            />
          </p>
          {uploading ? (
            <p className="mt-2 flex items-center gap-2 text-amber-700 dark:text-amber-100">
              <Loader2 className="w-4 h-4 animate-spin" />
              {t("training.datasetNotice.uploading")}
            </p>
          ) : errorMessage ? (
            // Backend text — shown exactly as it arrived.
            <p className="mt-2 text-red-300">{errorMessage}</p>
          ) : (
            <p className="mt-2 flex items-center gap-2 text-amber-700/70 dark:text-amber-200/70">
              <UploadCloud className="w-4 h-4" />
              {t("training.cloudNotice.uploadHint", {
                action: t("training.configurator.button.uploadAndStart"),
              })}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default LocalDatasetCloudNotice;
