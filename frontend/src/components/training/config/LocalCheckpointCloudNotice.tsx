import React from "react";
import { AlertTriangle, Lock, UploadCloud } from "lucide-react";

interface LocalCheckpointCloudNoticeProps {
  /** The run being continued, for a notice that names what moves. */
  runName: string;
  /** The checkpoint step the continuation resumes from; null ⇒ the latest. */
  step: number | null;
}

/**
 * Amber notice shown when a run that trained LOCALLY is being continued on
 * Hugging Face Cloud. The pod resumes from the Hub, so the parent's checkpoint
 * — weights AND optimizer state — has to be uploaded first (F7's local→cloud
 * direction).
 *
 * The twin of LocalDatasetCloudNotice, and deliberately louder about privacy
 * than its dataset sibling: a dataset upload is public by default because a
 * dataset is a thing people share, while this is an intermediate artifact of
 * someone's own run that the user never asked to publish. It goes to a private
 * repo, and the notice says so before the click rather than after — an upload
 * is a disclosure, so it is never a silent side effect of Continue. The backend
 * enforces the same rule: it refuses this launch unless the request carries the
 * consent this notice is asking for.
 */
const LocalCheckpointCloudNotice: React.FC<LocalCheckpointCloudNoticeProps> = ({
  runName,
  step,
}) => {
  const stepLabel =
    step != null ? `step ${step.toLocaleString()}` : "its latest checkpoint";

  return (
    <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-4 text-sm text-amber-700 dark:text-amber-100">
      <div className="flex items-start gap-2">
        <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0 text-amber-600 dark:text-amber-300" />
        <div className="w-full">
          <div className="font-semibold">
            This checkpoint is only on this machine
          </div>
          <p className="mt-1 text-amber-700/80 dark:text-amber-200/80">
            Hugging Face Cloud continues from the Hub, so {stepLabel} of{" "}
            <span className="font-medium">{runName}</span> — its weights and
            optimizer state — will be uploaded to a{" "}
            <span className="font-medium">private</span> repo in your account
            before the job starts. Continuing the same checkpoint again reuses
            that upload.
          </p>
          <p className="mt-2 flex items-center gap-2 text-amber-700/70 dark:text-amber-200/70">
            <Lock className="w-4 h-4" />
            Private to your account — nothing is published.
          </p>
          <p className="mt-1 flex items-center gap-2 text-amber-700/70 dark:text-amber-200/70">
            <UploadCloud className="w-4 h-4" />
            Use “Upload &amp; continue training” below to upload, then launch.
          </p>
        </div>
      </div>
    </div>
  );
};

export default LocalCheckpointCloudNotice;
