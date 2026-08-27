import React from "react";
import { Trans, useTranslation } from "react-i18next";
import { AlertTriangle, Lock, UploadCloud, WifiOff } from "lucide-react";

interface LocalCheckpointCloudNoticeProps {
  /** Which crossing this is: a local run being CONTINUED on the cloud (the
   * whole checkpoint moves, weights and optimizer state), or a local base being
   * FINE-TUNED there (only its weights move — a fine-tune starts a fresh
   * optimizer and never reads the rest). */
  mode: "resume" | "finetune";
  /** The run being continued / the base being fine-tuned, for a notice that
   * names what moves. */
  runName: string;
  /** The checkpoint step involved; null ⇒ the latest. */
  step: number | null;
  /** Backend is in HF_HUB_OFFLINE mode: nothing can be uploaded, so this
   * launch can't run on the cloud at all. */
  offline: boolean;
}

/**
 * Amber notice shown when a run targeting Hugging Face Cloud depends on a
 * checkpoint that exists only on this machine. The pod reads its checkpoints
 * from the Hub, so those bytes have to be uploaded first — F7's local→cloud
 * direction, in both of its modes (see `mode`).
 *
 * The twin of LocalDatasetCloudNotice, and deliberately louder about privacy
 * than its dataset sibling: a dataset upload is public by default because a
 * dataset is a thing people share, while this is an artifact of someone's own
 * run that the user never asked to publish. It goes to a private repo, and the
 * notice says so before the click rather than after — an upload is a
 * disclosure, so it is never a silent side effect of launching. The backend
 * enforces the same rule: it refuses the launch unless the request carries the
 * consent this notice is asking for.
 *
 * Each mode owns COMPLETE sentences in the catalog rather than sharing a
 * template with a spliced-in clause: what moves, and what to do instead when
 * the Hub is unreachable, differ per mode in more than one place.
 */
const LocalCheckpointCloudNotice: React.FC<LocalCheckpointCloudNoticeProps> = ({
  mode,
  runName,
  step,
  offline,
}) => {
  const { t } = useTranslation();
  // A noun phrase naming which checkpoint moves. {{step}} keeps its existing
  // (non-locale-aware) formatting and is passed in pre-formatted.
  const stepLabel =
    step != null
      ? t("training.checkpointNotice.stepLabel", {
          step: step.toLocaleString(),
        })
      : t("training.checkpointNotice.latestLabel");
  const isFinetune = mode === "finetune";

  if (offline) {
    return (
      <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-4 text-sm text-amber-700 dark:text-amber-100">
        <div className="flex items-start gap-2">
          <WifiOff className="w-4 h-4 mt-0.5 shrink-0 text-amber-600 dark:text-amber-300" />
          <div>
            <div className="font-semibold">
              {t("training.checkpointNotice.title")}
            </div>
            <p className="mt-1 text-amber-700/80 dark:text-amber-200/80">
              <Trans
                i18nKey={
                  isFinetune
                    ? "training.checkpointNotice.offlineFinetune"
                    : "training.checkpointNotice.offlineResume"
                }
                values={{ stepLabel, runName }}
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
            {t("training.checkpointNotice.title")}
          </div>
          <p className="mt-1 text-amber-700/80 dark:text-amber-200/80">
            <Trans
              i18nKey={
                isFinetune
                  ? "training.checkpointNotice.bodyFinetune"
                  : "training.checkpointNotice.bodyResume"
              }
              values={{ stepLabel, runName }}
              components={[
                <span key="0" className="font-medium" />,
                <span key="1" className="font-medium" />,
              ]}
            />
          </p>
          <p className="mt-2 flex items-center gap-2 text-amber-700/70 dark:text-amber-200/70">
            <Lock className="w-4 h-4" />
            {t("training.checkpointNotice.privacy")}
          </p>
          <p className="mt-1 flex items-center gap-2 text-amber-700/70 dark:text-amber-200/70">
            <UploadCloud className="w-4 h-4" />
            {/* Quotes the Start button's own label, so the two can't drift. */}
            {t("training.cloudNotice.uploadHint", {
              action: isFinetune
                ? t("training.configurator.button.uploadAndStart")
                : t("training.configurator.button.uploadAndContinue"),
            })}
          </p>
        </div>
      </div>
    </div>
  );
};

export default LocalCheckpointCloudNotice;
