import React, { useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { CheckCircle, Loader2, Trash2, Upload as UploadIcon, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { useStudio, type RecordedInfo } from "@/contexts/StudioContext";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";
import { useDatasetUpload } from "@/hooks/useDatasetUpload";
import UploadDatasetDialog from "@/components/landing/UploadDatasetDialog";
import MilestoneReveal from "@/components/onboarding/MilestoneReveal";
import { useOnceFlag } from "@/lib/onboarding/storage";

/**
 * Post-recording handoff banner, rendered at the top of the studio's Collect
 * panel. Offers the two next steps that used to live on the /upload page + home
 * info card: train on the just-recorded dataset, or upload it to the Hub. It
 * also carries two side effects that are the real payload — preselecting the
 * dataset for Train, and kicking off the automatic Hub push — so it has to
 * mount after every saved session, not just when someone is looking at it.
 *
 * The `recorded` payload comes from StudioContext. It used to arrive in router
 * state, stamped by a navigate("/") that also closed the studio; a session now
 * leaves the user in the studio, so there is no navigation to hang it on.
 */
const CollectHandoff: React.FC<{
  recorded: RecordedInfo | null;
  onDismiss: () => void;
}> = ({ recorded, onDismiss }) => {
  const { t } = useTranslation();
  const { setSelectedDataset } = useSelectedDataset();
  const { openStudio, collectForm } = useStudio();

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

  // Keyed on the payload, NOT on mount. This banner used to remount for each
  // session (it was driven by router state on a page the user navigated back
  // to), so an empty dep array fired exactly once per recording. It now lives
  // in the always-mounted Collect panel, where a mount-only effect would run
  // once with no payload and never again — the milestone would never show.
  // markSeen flips `seen` synchronously, so this settles after one pass.
  useEffect(() => {
    if (
      recorded &&
      !discardedEmpty &&
      (recorded.saved_episodes ?? 0) > 0 &&
      !hasSeenRecordingMilestone
    ) {
      setShowRecordingMilestone(true);
      markRecordingMilestoneSeen();
    }
  }, [
    recorded,
    discardedEmpty,
    hasSeenRecordingMilestone,
    markRecordingMilestoneSeen,
  ]);

  if (!recorded) return null;

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
                  {t("studio.handoff.emptyTitle")}
                </p>
                <p className="mt-0.5 text-sm text-muted-foreground">
                  {t("studio.handoff.emptyBody")}
                </p>
              </>
            ) : (
              <>
                <p className="font-medium text-foreground">
                  {/* The repo id is DATA: interpolated whole into the <0> slot
                      rather than concatenated around translated fragments. */}
                  <Trans
                    i18nKey="studio.handoff.savedTitle"
                    values={{ repoId: repoId ?? "" }}
                    components={[
                      <span
                        key="0"
                        className="break-all font-mono text-foreground"
                      />,
                    ]}
                  />
                  {recorded.saved_episodes != null && (
                    <span className="text-muted-foreground">
                      {" · "}
                      {t("studio.handoff.episodes", {
                        count: recorded.saved_episodes,
                      })}
                    </span>
                  )}
                </p>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <Button size="sm" onClick={trainOnThis}>
                    {t("studio.handoff.trainOnThis")}
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
            aria-label={t("studio.common.dismiss")}
            onClick={onDismiss}
            className="h-7 w-7 shrink-0 text-muted-foreground"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
      {showRecordingMilestone && (
        <MilestoneReveal
          title={t("studio.handoff.milestone.recording.title")}
          description={t("studio.handoff.milestone.recording.description")}
          onDismiss={() => setShowRecordingMilestone(false)}
        />
      )}
      {showHubMilestone && (
        <MilestoneReveal
          title={t("studio.handoff.milestone.hubUpload.title")}
          description={t("studio.handoff.milestone.hubUpload.description")}
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
  const { t } = useTranslation();
  const { toast } = useToast();
  const { uploading, start } = useDatasetUpload({
    repoId,
    onDone: (url) => {
      toast({
        title: t("studio.handoff.upload.doneTitle"),
        description: (
          <span>
            <Trans
              i18nKey="studio.handoff.upload.doneBody"
              values={{ repoId }}
              components={[
                <a
                  key="0"
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium underline"
                />,
              ]}
            />
          </span>
        ),
      });
      onUploaded?.();
    },
    onError: (message, docsUrl) => {
      toast({
        title: t("studio.handoff.upload.failedTitle"),
        // `message` is the backend's own failure text and stays as sent —
        // only the link label beside it is translated.
        description: docsUrl ? (
          <span>
            {message}{" "}
            <a
              href={docsUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium underline"
            >
              {t("studio.handoff.upload.setupGuide")}
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
          title: t("studio.handoff.upload.autoFailedTitle"),
          // `error` comes from useDatasetUpload / the backend — shown as sent.
          description: error,
        });
      }
    });
  }, [autoStart, repoId, start, toast, t]);

  if (uploading) {
    return (
      <span className="inline-flex items-center gap-1.5 text-sm text-muted-foreground">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        {t("studio.handoff.upload.uploading")}
      </span>
    );
  }

  return (
    <UploadDatasetDialog repoId={repoId} start={start}>
      <Button size="sm" variant="outline" className="gap-1.5">
        <UploadIcon className="h-3.5 w-3.5" />
        {t("studio.handoff.upload.button")}
      </Button>
    </UploadDatasetDialog>
  );
};

export default CollectHandoff;
