import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { CheckCircle, Loader2, Trash2, Upload as UploadIcon, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { useStudio } from "@/contexts/StudioContext";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";
import { useDatasetUpload } from "@/hooks/useDatasetUpload";
import UploadDatasetDialog from "@/components/landing/UploadDatasetDialog";
import MilestoneReveal from "@/components/onboarding/MilestoneReveal";
import { useOnceFlag } from "@/lib/onboarding/storage";

/** Router-state payload left by the Recording page when a session ends (see
 * Recording.tsx). Replaces the old /upload page hop. */
interface RecordedInfo {
  repo_id: string;
  saved_episodes?: number;
  // Set when the session saved zero episodes and the backend discarded the
  // (empty) dataset directory — nothing is on disk.
  discarded_empty?: boolean;
}

/**
 * Post-recording handoff banner on the Launchpad. Reads the `recorded` payload
 * the Recording page leaves in router state and offers the two next steps that
 * used to live on the /upload page + home info card: train on the just-recorded
 * dataset, or upload it to the Hub. Dismissible; clears the router state so it
 * doesn't resurrect on re-render.
 */
const CollectHandoff: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { setSelectedDataset } = useSelectedDataset();
  const { openStudio, collectForm } = useStudio();

  const recorded = location.state?.recorded as RecordedInfo | undefined;
  const [dismissed, setDismissed] = useState(false);

  const discardedEmpty = recorded?.discarded_empty ?? false;
  // A discarded (empty) session left nothing on disk, so there's no repo id to
  // link, preselect, or upload.
  const repoId = discardedEmpty ? null : (recorded?.repo_id ?? null);

  const { seen: hasSeenRecordingMilestone, markSeen: markRecordingMilestoneSeen } =
    useOnceFlag("makerlab:milestone-first-recording");
  const { seen: hasSeenHubUploadMilestone, markSeen: markHubUploadMilestoneSeen } =
    useOnceFlag("makerlab:milestone-first-hub-upload");
  const [showRecordingMilestone, setShowRecordingMilestone] = useState(false);
  const [showHubMilestone, setShowHubMilestone] = useState(false);

  // Preserve the old Upload page's effect: preselect the just-recorded dataset
  // so the Train panel (useSelectedDataset) opens straight onto it.
  useEffect(() => {
    if (repoId) setSelectedDataset(repoId);
  }, [repoId, setSelectedDataset]);

  useEffect(() => {
    if (
      !discardedEmpty &&
      (recorded?.saved_episodes ?? 0) > 0 &&
      !hasSeenRecordingMilestone
    ) {
      setShowRecordingMilestone(true);
      markRecordingMilestoneSeen();
    }
    // Runs once for this recorded payload — hasSeenRecordingMilestone and
    // markRecordingMilestoneSeen are stable for the component's lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!recorded || dismissed) return null;

  const dismiss = () => {
    setDismissed(true);
    // Clear the router state so a reload / re-render doesn't bring the banner
    // back for a session already handled.
    navigate(".", { replace: true, state: null });
  };

  const trainOnThis = () => {
    if (!repoId) return;
    setSelectedDataset(repoId);
    openStudio("train", { train: { datasetRepoId: repoId } });
  };

  return (
    <div className="w-full space-y-3">
      <div className="w-full rounded-lg border border-border bg-card p-4 shadow-1">
        <div className="flex items-start gap-3">
          {discardedEmpty ? (
            <Trash2 className="mt-0.5 h-5 w-5 shrink-0 text-muted-foreground" />
          ) : (
            <CheckCircle className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600 dark:text-emerald-400" />
          )}
          <div className="min-w-0 flex-1">
            {discardedEmpty ? (
              <>
                <p className="font-medium text-foreground">
                  No episodes were recorded
                </p>
                <p className="mt-0.5 text-sm text-muted-foreground">
                  Nothing was saved — the empty dataset was discarded so it
                  doesn't take up disk space.
                </p>
              </>
            ) : (
              <>
                <p className="font-medium text-foreground">
                  Dataset{" "}
                  <span className="break-all font-mono text-foreground">
                    {repoId}
                  </span>{" "}
                  saved
                  {recorded.saved_episodes != null && (
                    <span className="text-muted-foreground">
                      {" · "}
                      {recorded.saved_episodes} episode
                      {recorded.saved_episodes === 1 ? "" : "s"}
                    </span>
                  )}
                </p>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <Button size="sm" onClick={trainOnThis}>
                    Train on this dataset
                  </Button>
                  {repoId && (
                    <UploadToHubAction
                      repoId={repoId}
                      // Kick off the Hub push automatically when the Collect
                      // form's advanced toggle (default on) says so. A repo id
                      // without a namespace means the user wasn't logged in at
                      // record time — the push would only 401, so stay manual.
                      autoStart={collectForm.pushToHub && repoId.includes("/")}
                      onUploaded={() => {
                        if (!hasSeenHubUploadMilestone) {
                          setShowHubMilestone(true);
                          markHubUploadMilestoneSeen();
                        }
                      }}
                    />
                  )}
                </div>
              </>
            )}
          </div>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Dismiss"
            onClick={dismiss}
            className="h-7 w-7 shrink-0 text-muted-foreground"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
      {showRecordingMilestone && (
        <MilestoneReveal
          title="Nice, your first episodes are recorded!"
          description="Train a policy on this dataset from the Train panel, or upload it to the Hub to share it or train in the cloud."
          onDismiss={() => setShowRecordingMilestone(false)}
        />
      )}
      {showHubMilestone && (
        <MilestoneReveal
          title="Uploaded to the Hub!"
          description="Your dataset is public and shareable — reference its repo id anywhere in MakerLab, or fine-tune a skill on it from the Train panel."
          onDismiss={() => setShowHubMilestone(false)}
        />
      )}
    </div>
  );
};

/** Repo ids already auto-pushed this app session — module-level so a banner
 * remount (e.g. browser-back onto the history entry that carries the
 * `recorded` state) doesn't fire a second, redundant upload. */
const autoPushed = new Set<string>();

/** "Upload to Hub" affordance — reuses the existing UploadDatasetDialog +
 * useDatasetUpload flow (identical to DatasetInfoCard's HubSyncRow). Rendered
 * only when there's a local dataset to upload, so the hook has a real repoId.
 * With `autoStart`, the upload kicks off on mount (no tags, public — the
 * dialog's own defaults) instead of waiting for a click. */
const UploadToHubAction: React.FC<{
  repoId: string;
  autoStart?: boolean;
  onUploaded?: () => void;
}> = ({ repoId, autoStart = false, onUploaded }) => {
  const { toast } = useToast();
  const { uploading, start } = useDatasetUpload({
    repoId,
    onDone: (url) => {
      toast({
        title: "Uploaded to Hub",
        description: (
          <span>
            {repoId} is now on the Hub.{" "}
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium underline"
            >
              View dataset
            </a>
          </span>
        ),
      });
      onUploaded?.();
    },
    onError: (message, docsUrl) => {
      toast({
        title: "Upload failed",
        description: docsUrl ? (
          <span>
            {message}{" "}
            <a
              href={docsUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium underline"
            >
              Open setup guide
            </a>
          </span>
        ) : (
          message
        ),
        variant: "destructive",
      });
    },
  });

  // Auto-push: fire once per repo per app session (the Set guards remounts).
  // A refused start (another upload running / dataset busy) is surfaced so the
  // user knows to fall back to the manual button.
  useEffect(() => {
    if (!autoStart || autoPushed.has(repoId)) return;
    autoPushed.add(repoId);
    start([], false).then((error) => {
      if (error) {
        toast({
          title: "Automatic Hub upload not started",
          description: error,
        });
      }
    });
  }, [autoStart, repoId, start, toast]);

  if (uploading) {
    return (
      <span className="inline-flex items-center gap-1.5 text-sm text-muted-foreground">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Uploading to Hub…
      </span>
    );
  }

  return (
    <UploadDatasetDialog repoId={repoId} start={start}>
      <Button size="sm" variant="outline" className="gap-1.5">
        <UploadIcon className="h-3.5 w-3.5" />
        Upload to Hub
      </Button>
    </UploadDatasetDialog>
  );
};

export default CollectHandoff;
