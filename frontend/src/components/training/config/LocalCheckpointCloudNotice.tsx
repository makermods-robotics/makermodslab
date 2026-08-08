import React from "react";
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
 */
const LocalCheckpointCloudNotice: React.FC<LocalCheckpointCloudNoticeProps> = ({
  mode,
  runName,
  step,
  offline,
}) => {
  const stepLabel =
    step != null ? `step ${step.toLocaleString()}` : "its latest checkpoint";
  const isFinetune = mode === "finetune";

  if (offline) {
    return (
      <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-4 text-sm text-amber-700 dark:text-amber-100">
        <div className="flex items-start gap-2">
          <WifiOff className="w-4 h-4 mt-0.5 shrink-0 text-amber-600 dark:text-amber-300" />
          <div>
            <div className="font-semibold">
              This checkpoint is only on this machine
            </div>
            <p className="mt-1 text-amber-700/80 dark:text-amber-200/80">
              {isFinetune
                ? "Hugging Face Cloud loads base weights from the Hub"
                : "Hugging Face Cloud continues from the Hub"}
              , but the server is in offline mode (
              <code className="text-amber-700 dark:text-amber-100">
                HF_HUB_OFFLINE
              </code>
              ), so {stepLabel} of{" "}
              <span className="font-medium">{runName}</span> can't be uploaded.
              Switch off offline mode, or{" "}
              {isFinetune
                ? "run this fine-tune locally"
                : "continue this run locally"}
              .
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
            This checkpoint is only on this machine
          </div>
          {isFinetune ? (
            <p className="mt-1 text-amber-700/80 dark:text-amber-200/80">
              Hugging Face Cloud loads base weights from the Hub, so {stepLabel}{" "}
              of <span className="font-medium">{runName}</span> — its weights,
              not its optimizer state — will be uploaded to a{" "}
              <span className="font-medium">private</span> repo in your account
              before the job starts. Fine-tuning the same checkpoint again
              reuses that upload.
            </p>
          ) : (
            <p className="mt-1 text-amber-700/80 dark:text-amber-200/80">
              Hugging Face Cloud continues from the Hub, so {stepLabel} of{" "}
              <span className="font-medium">{runName}</span> — its weights and
              optimizer state — will be uploaded to a{" "}
              <span className="font-medium">private</span> repo in your account
              before the job starts. Continuing the same checkpoint again reuses
              that upload.
            </p>
          )}
          <p className="mt-2 flex items-center gap-2 text-amber-700/70 dark:text-amber-200/70">
            <Lock className="w-4 h-4" />
            Private to your account — nothing is published.
          </p>
          <p className="mt-1 flex items-center gap-2 text-amber-700/70 dark:text-amber-200/70">
            <UploadCloud className="w-4 h-4" />
            Use “
            {isFinetune ? "Upload & start training" : "Upload & continue training"}
            ” below to upload, then launch.
          </p>
        </div>
      </div>
    </div>
  );
};

export default LocalCheckpointCloudNotice;
