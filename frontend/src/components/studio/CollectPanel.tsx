import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Check, GitMerge, RefreshCw, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useRobots } from "@/hooks/useRobots";
import { useDatasets } from "@/hooks/useDatasets";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";
import {
  datasetNameIssue,
  formatDatasetNameIssue,
} from "@/lib/datasetName";
import { useStudio } from "@/contexts/StudioContext";
import MergeDatasetsDialog from "@/components/landing/MergeDatasetsDialog";
import type { MergeStatus } from "@/lib/replayApi";
import RecordingForm from "@/components/studio/RecordingForm";
import CollectHandoff from "@/components/studio/CollectHandoff";
import RecordingSessionDialog, {
  RecordedInfo,
  RecordingConfig,
} from "@/components/recording/RecordingSessionDialog";
import LibraryHeader from "@/components/library/LibraryHeader";
import {
  DatasetLibraryList,
  clearDatasetInfoCache,
} from "@/components/library/DatasetLibrary";
import {
  LibrarySection,
  PanelEntryControl,
  PanelHeader,
  SLIDE,
} from "@/components/studio/panel/primitives";
import DatasetDetailDialog from "@/components/dialogs/DatasetDetailDialog";
import type { DatasetItem } from "@/lib/replayApi";

/**
 * Studio panel 1 · Collect. Stacked sections (the shared studio anatomy):
 *
 * 1. "Record new dataset" — a select-style entry control that slides the full
 *    recording form open in place (no dialog). Opening it folds the dataset
 *    library down to its header, which stays clickable to re-expand.
 * 2. Dataset library, pinned to the panel foot — every dataset with a local
 *    copy (whoever recorded it) plus the user's own Hub datasets, each a card
 *    carrying the dataset's metadata. The shared selected dataset (stamped by
 *    CollectHandoff after a session, consumed by Train) shows as a chip in
 *    the library's header row; selecting a card feeds it. A merge button
 *    covers growing a dataset.
 *
 * Every recording session creates a NEW dataset — recording on top of an
 * existing one was removed; merging datasets covers growing one.
 */
const CollectPanel: React.FC = () => {
  const { auth } = useHfAuth();
  const { t } = useTranslation();
  const { selectedRecord } = useRobots();
  const { datasets, loading: datasetsLoading, refresh } = useDatasets();
  const { selectedDataset, setSelectedDataset } = useSelectedDataset();
  const { toast } = useToast();

  // The recording-form draft lives in StudioContext so filled-in parameters
  // survive route changes (this panel unmounts with the Launchpad route).
  const {
    collectForm,
    updateCollectForm,
    mergePrefill,
    clearMergePrefill,
    openStudio,
    lastRecorded,
    setLastRecorded,
  } = useStudio();
  const {
    formOpen,
    datasetName,
    singleTask,
    numEpisodes,
    episodeTimeS,
    resetTimeS,
    streamingEncoding,
    pushToHub,
  } = collectForm;

  // The session's cameras ARE the selected robot's cameras — the backend
  // resolves them from the robot record and the start request carries none.
  // Read straight off the record so the panel can't show (or hold on to) a set
  // the server wouldn't use; edits live in the robot settings dialog.
  const cameras = selectedRecord?.cameras ?? [];

  // The record-new form slides open in place; the library folds to its header
  // while the form is open (still expandable by hand).
  const [libraryOpen, setLibraryOpen] = useState(!formOpen);
  const [mergeOpen, setMergeOpen] = useState(false);
  // Mirrors the merge dialog's own state so closing it mid-run does not throw
  // away the fine-tune half of a coaching handoff.
  const [mergeStatus, setMergeStatus] = useState<MergeStatus | null>(null);

  // A coaching session that just ended asked for this merge, with both halves
  // already chosen. Opening it here rather than at the call site keeps the
  // dialog owned by the panel that owns the dataset list it needs.
  useEffect(() => {
    if (mergePrefill) setMergeOpen(true);
  }, [mergePrefill]);

  // The episode viewer — opened by a dataset card's "view" button, separate
  // from selecting the card for recording.
  const [viewRepo, setViewRepo] = useState<string | null>(null);
  const [viewOpen, setViewOpen] = useState(false);
  const openDatasetDetail = (item: DatasetItem) => {
    setViewRepo(item.repo_id);
    setViewOpen(true);
  };

  // A live session renders as a modal dialog over the studio (the old
  // /recording page). While it runs, the form below stays mounted with its
  // camera previews released; `sessionCount` keys the form so a finished
  // session remounts it and the previews come back.
  const [activeRecording, setActiveRecording] =
    useState<RecordingConfig | null>(null);
  const [sessionCount, setSessionCount] = useState(0);

  const toggleForm = (open: boolean) => {
    updateCollectForm({ formOpen: open });
    setLibraryOpen(!open);
  };

  const releaseStreamsRef = useRef<(() => void) | null>(null);

  // Release camera streams when the panel unmounts (e.g. navigating to the
  // recording session), so cv2 can grab the devices exclusively.
  useEffect(() => {
    return () => {
      releaseStreamsRef.current?.();
    };
  }, []);

  // The library: every dataset with a local copy — whoever recorded it, a
  // local copy is workable — plus the user's own Hub datasets (namespace is
  // the account or one of its writable orgs). Hub-only datasets from other
  // authors stay out; they live in the community listings. The backend's
  // newest-first (last_modified) order is kept — every studio library sorts
  // by recently added.
  const libraryDatasets = useMemo(() => {
    const ownedNamespaces =
      auth.status === "authenticated"
        ? new Set(
            [auth.username, ...auth.writableNamespaces].map((n) =>
              n.toLowerCase(),
            ),
          )
        : null;
    return datasets.filter((d) => {
      if (d.source !== "hub") return true;
      if (!ownedNamespaces) return false;
      const namespace = d.repo_id.split("/")[0]?.toLowerCase() ?? "";
      return ownedNamespaces.has(namespace);
    });
  }, [datasets, auth]);

  // Ported from Landing.tsx handleStartRecording (resume path removed —
  // sessions always create a new dataset).
  const handleStartRecording = async () => {
    if (!selectedRecord) {
      toast({
        title: t("studio.collect.toast.noRobotTitle"),
        description: t("studio.collect.toast.noRobotBody"),
        variant: "destructive",
      });
      return;
    }
    const robot = selectedRecord;
    // No is_clean gate here any more: record readiness is the SERVER's check
    // now (400 robot.not_ready from POST /api/v1/sessions, rendered by the
    // session dialog's start-failure toast). The Start button below still
    // disables on the same condition as a courtesy.
    if (!datasetName || !singleTask) {
      toast({
        title: t("studio.collect.toast.missingDetailsTitle"),
        description: t("studio.collect.toast.missingDetailsBody"),
        variant: "destructive",
      });
      return;
    }
    const nameIssue = datasetNameIssue(datasetName);
    const nameError = nameIssue ? formatDatasetNameIssue(t, nameIssue) : null;
    if (nameError) {
      toast({
        title: t("studio.collect.toast.invalidNameTitle"),
        // validateDatasetName's own message — client-side, but owned by
        // lib/datasetName.ts, so it is shown exactly as returned.
        description: nameError,
        variant: "destructive",
      });
      return;
    }

    const datasetRepoId =
      auth.status === "authenticated"
        ? `${auth.username}/${datasetName}`
        : datasetName;

    if (cameras.length > 0 && releaseStreamsRef.current) {
      toast({
        title: t("studio.collect.toast.preparingCamerasTitle"),
        description: t("studio.collect.toast.releasingStreams", {
          count: cameras.length,
        }),
      });
      releaseStreamsRef.current();
      await new Promise((resolve) => setTimeout(resolve, 500));
      toast({
        title: t("studio.collect.toast.camerasReadyTitle"),
        description: t("studio.collect.toast.camerasReadyBody"),
      });
    }

    // Robot NAME + dataset-shaped options only: the session dialog POSTs this
    // to /api/v1/sessions and the server resolves everything hardware-shaped
    // (ports, configs, mode, right-arm fields, cameras) from the saved record.
    const recordingConfig = {
      robot: robot.name,
      dataset_repo_id: datasetRepoId,
      single_task: singleTask,
      num_episodes: numEpisodes,
      episode_time_s: episodeTimeS,
      reset_time_s: resetTimeS,
      fps: 30,
      video: true,
      push_to_hub: false,
      resume: false,
      streaming_encoding: streamingEncoding,
    };

    setActiveRecording(recordingConfig);
  };

  // Every exit path of the session dialog lands here. A `recorded` payload
  // (clean finish / "keep episodes") hands the session off to the banner at the
  // top of this panel.
  //
  // The studio deliberately stays OPEN. This used to closeStudio() and
  // navigate("/") — the contract the old /recording page fulfilled by going
  // home — which dropped the user on the Launchpad after every session, away
  // from the library and Train panel that are the actual next steps. The
  // payload moved to StudioContext at the same time, because with no
  // navigation there is no router state to stamp.
  const handleRecordingExit = useCallback(
    (recorded?: RecordedInfo) => {
      setActiveRecording(null);
      setSessionCount((n) => n + 1);
      if (recorded) {
        // A dataset was saved: fold the record-new form so the panel opens onto
        // the library (with the fresh dataset preselected by CollectHandoff). A
        // discarded (empty) session keeps the form + draft open for a retry.
        if (!recorded.discarded_empty) {
          updateCollectForm({ formOpen: false });
          setLibraryOpen(true);
        }
        setLastRecorded(recorded);
      }
    },
    [setLastRecorded, updateCollectForm],
  );

  // Gate for the pinned Start button: robot ready + every required parameter
  // filled in (name valid per the backend's rules, task described).
  const canStart =
    !!selectedRecord &&
    selectedRecord.is_clean &&
    datasetNameIssue(datasetName) === null &&
    singleTask.trim().length > 0;

  return (
    <div className="flex flex-1 flex-col gap-5 p-5">
      <PanelHeader
        step="1"
        title={t("studio.collect.title")}
        dataTour="studio-collect"
      />

      {/* Post-session handoff. Renders nothing until a session saves something,
          but stays mounted so its Hub auto-push and dataset preselection are
          not conditional on the user looking at this panel. */}
      <CollectHandoff
        recorded={lastRecorded}
        onDismiss={() => setLastRecorded(null)}
      />

      {/* Record new dataset — the form slides open in place (no dialog). */}
      <Collapsible open={formOpen} onOpenChange={toggleForm} className="space-y-5">
        <CollapsibleTrigger asChild>
          <PanelEntryControl open={formOpen} dotClassName="bg-red-500">
            {t("studio.collect.entry")}
          </PanelEntryControl>
        </CollapsibleTrigger>
        <CollapsibleContent className={SLIDE}>
          <RecordingForm
            key={sessionCount}
            robot={selectedRecord}
            datasetName={datasetName}
            setDatasetName={(v) => updateCollectForm({ datasetName: v })}
            singleTask={singleTask}
            setSingleTask={(v) => updateCollectForm({ singleTask: v })}
            numEpisodes={numEpisodes}
            setNumEpisodes={(v) => updateCollectForm({ numEpisodes: v })}
            episodeTimeS={episodeTimeS}
            setEpisodeTimeS={(v) => updateCollectForm({ episodeTimeS: v })}
            resetTimeS={resetTimeS}
            setResetTimeS={(v) => updateCollectForm({ resetTimeS: v })}
            streamingEncoding={streamingEncoding}
            setStreamingEncoding={(v) =>
              updateCollectForm({ streamingEncoding: v })
            }
            pushToHub={pushToHub}
            setPushToHub={(v) => updateCollectForm({ pushToHub: v })}
            releaseStreamsRef={releaseStreamsRef}
          />
        </CollapsibleContent>
      </Collapsible>

      {/* Start recording — pinned directly above the dataset library so the
          panel's primary action sits at the same level as Train's Start and
          Deploy's Start/Stop. Disabled until the robot is ready and the
          required parameters are filled in. */}
      <div className="mt-auto pt-2">
        <Button
          onClick={handleStartRecording}
          disabled={!canStart}
          className="w-full gap-2"
        >
          <span className="h-2.5 w-2.5 rounded-full bg-red-500" />
          {t("studio.collect.start")}
        </Button>
      </div>

      {/* Dataset library — the user's own datasets, pinned to the panel foot
          like Train's jobs and Deploy's models. The selected-dataset chip
          lives in the header row, beside Merge. */}
      <LibrarySection className="mt-0">
        <Collapsible
          open={libraryOpen}
          onOpenChange={setLibraryOpen}
          className="space-y-3"
        >
          <LibraryHeader
            title={t("studio.collect.library.title")}
            count={libraryDatasets.length}
            open={libraryOpen}
            actions={
              <>
                {selectedDataset ? (
                  <span
                    className="flex min-w-0 items-center gap-1 rounded-md border border-border bg-muted/40 py-0.5 pl-1.5 pr-0.5 font-mono text-[11px] text-foreground"
                    title={selectedDataset}
                  >
                    <Check className="h-3 w-3 shrink-0 text-primary" />
                    <span className="truncate">{selectedDataset}</span>
                    <button
                      type="button"
                      onClick={() => setSelectedDataset(null)}
                      aria-label={t("studio.collect.library.clearSelected")}
                      title={t("studio.collect.library.clearSelected")}
                      className="shrink-0 rounded p-0.5 text-muted-foreground hover:text-foreground"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </span>
                ) : null}
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setMergeOpen(true)}
                  className="h-7 shrink-0 gap-1.5 px-2 text-xs"
                >
                  <GitMerge className="h-3.5 w-3.5" />
                  {t("studio.collect.library.merge")}
                </Button>
                <button
                  type="button"
                  onClick={() => {
                    clearDatasetInfoCache();
                    refresh();
                  }}
                  aria-label={t("studio.collect.library.refresh")}
                  title={t("studio.collect.library.refresh")}
                  className="shrink-0 rounded p-1 text-muted-foreground hover:text-foreground"
                >
                  <RefreshCw
                    className={cn(
                      "h-3.5 w-3.5",
                      datasetsLoading && "animate-spin",
                    )}
                  />
                </button>
              </>
            }
          />
          <CollapsibleContent className={SLIDE}>
            <DatasetLibraryList
              datasets={libraryDatasets}
              loading={datasetsLoading}
              selectedRepoId={selectedDataset}
              onSelect={(item) =>
                setSelectedDataset(
                  item.repo_id === selectedDataset ? null : item.repo_id,
                )
              }
              onView={openDatasetDetail}
            />
          </CollapsibleContent>
        </Collapsible>
      </LibrarySection>

      <MergeDatasetsDialog
        open={mergeOpen}
        onOpenChange={(next) => {
          setMergeOpen(next);
          // Closing consumes the prefill — UNLESS a merge it started is still
          // running. A real merge takes minutes, and Radix fires this for
          // Escape and outside-click as well as the X, so an operator who
          // stepped away used to come back to a merged dataset and no
          // fine-tune, with nothing saying why the one button they pressed
          // had not finished.
          if (!next && mergeStatus?.state !== "running") clearMergePrefill();
        }}
        onStatusChange={setMergeStatus}
        datasets={libraryDatasets}
        initialSources={mergePrefill?.sources}
        initialOutput={mergePrefill?.suggestedOutput}
        onMerged={(outputRepoId) => {
          clearDatasetInfoCache();
          refresh();
          // The second half of the coaching loop. Corrections merged into the
          // training set are still only a dataset; what the operator actually
          // wanted was a better policy, and leaving them to find the training
          // panel and re-pick both the base checkpoint and the dataset they
          // just built is where the old prose-only handoff lost people.
          const base = mergePrefill?.finetuneBaseJobId;
          if (base && outputRepoId) {
            clearMergePrefill();
            setMergeOpen(false);
            openStudio("train", {
              train: {
                baseJobId: base,
                baseName: mergePrefill?.finetuneBaseName,
                datasetRepoId: outputRepoId,
              },
            });
          }
        }}
      />

      <DatasetDetailDialog
        repoId={viewRepo}
        open={viewOpen}
        onOpenChange={setViewOpen}
      />

      {/* The live recording session — a modal dialog over the studio instead
          of a route hop, so the panel (and the filled-in form) stays put. */}
      {activeRecording ? (
        <RecordingSessionDialog
          config={activeRecording}
          onExit={handleRecordingExit}
        />
      ) : null}
    </div>
  );
};

export default CollectPanel;
