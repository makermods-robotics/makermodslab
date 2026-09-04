import React, {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import type { ResumeSeed } from "@/components/training/TrainingConfigurator";

export type StudioPanel = "collect" | "train" | "deploy";

/** What a finished recording session leaves behind for the Collect panel's
 * handoff banner. Mirrors RecordingSessionDialog's RecordedInfo; declared here
 * rather than imported so the provider doesn't depend on a dialog module. */
export interface RecordedInfo {
  repo_id: string;
  saved_episodes?: number;
  /** The session saved zero episodes and the backend discarded the (empty)
   * dataset directory — nothing is on disk to train on or upload. */
  discarded_empty?: boolean;
}

/** The Collect panel's recording-form draft. Lives in this provider (mounted
 * above the router) so filled-in parameters survive navigating to /recording
 * and back — the panel itself unmounts with the Launchpad route. */
export interface CollectFormState {
  formOpen: boolean;
  datasetName: string;
  singleTask: string;
  numEpisodes: number;
  episodeTimeS: number;
  resetTimeS: number;
  streamingEncoding: boolean;
  /** Push the finished dataset to the Hugging Face Hub automatically when the
   * session ends (via the background UploadManager, not the recorder's
   * blocking in-session push). Consumed by CollectHandoff. */
  pushToHub: boolean;
  // Cameras are deliberately NOT part of this draft. A session records the
  // selected robot's cameras, resolved server-side from the robot record, so a
  // session-local editable copy could only diverge from what actually runs —
  // its edits were never written back and never sent. The Collect panel now
  // shows the record's cameras read-only; they're edited in Robot settings.
}

const DEFAULT_COLLECT_FORM: CollectFormState = {
  formOpen: false,
  datasetName: "",
  singleTask: "",
  numEpisodes: 5,
  episodeTimeS: 60,
  resetTimeS: 15,
  streamingEncoding: true,
  pushToHub: true,
};

/** Pre-fills the Deploy panel when a policy card / job row says "Run on robot".
 * `job` sources resolve through the local job registry (id + optional step);
 * `hub` sources are Hub model repo ids that DeployPanel lazy-imports. */
export interface DeployPrefill {
  source: "job" | "hub";
  id: string;
  step?: number;
  /** Which run mode to open on. Carried so a bad result can hand the user
   * straight into coaching ("Policy failing? Coach it") instead of leaving
   * them to find a control they may not know exists. Omitted ⇒ the panel
   * keeps whatever mode it was already on. */
  mode?: "single" | "eval" | "coach";
}

/** Pre-fills the Train panel: fine-tune base, resume seed, and/or a
 * preselected dataset.
 *
 * A local policy's fine-tune base is a job registry id (`baseJobId`); a Hub
 * policy's is a repo id (`baseModelRepoId`) that the panel lazy-imports. Set
 * exactly one of the two. `baseStep` optionally pins the checkpoint to
 * fine-tune from (the card's dropdown choice); omitted ⇒ latest.
 *
 * `resume` is the sibling of that pair for Continue / Resume-cloud, and is
 * mutually exclusive with them — a run is either continued or used as a
 * fine-tune base, never both. It differs in kind from the base fields on
 * purpose: a fine-tune base is a *reference* the panel still has to resolve
 * (import the Hub repo, read the policy type, list checkpoints), whereas a
 * resume seed is already complete at the call site, which holds the parent's
 * persisted `config`. So this carries the finished ResumeSeed rather than a
 * job id for the panel to look up. */
export interface TrainPrefill {
  baseModelRepoId?: string;
  baseJobId?: string;
  baseStep?: number;
  /** Display name for the base policy, shown in the picker while (or in case)
   * the models listing doesn't carry this id. */
  baseName?: string;
  datasetRepoId?: string;
  /** Episode subset for datasetRepoId, e.g. seeded from the dataset viewer's
   * exclude-from-training checkboxes. Only meaningful alongside datasetRepoId
   * — TrainPanel drops it if the dataset selection later diverges. */
  episodeIndices?: number[];
  /** Built by buildResumeSeed — see components/jobs/resumeSeed.ts. */
  resume?: ResumeSeed;
}

/** Opens the Collect panel's merge dialog with both halves already chosen,
 * and remembers what to do once the merge finishes.
 *
 * Exists for the coaching handoff. A coaching session produces corrections that
 * are worthless alone: they have to be merged with the dataset the coached
 * checkpoint was last trained on, and the checkpoint fine-tuned on the result.
 * `TrainPanel` takes exactly one dataset, so the merge is mandatory rather than
 * an optimisation — which left the whole payoff of coaching behind a manual
 * chore, described in prose, with no buttons.
 *
 * `finetuneBaseJobId` is what makes this a loop rather than a shortcut: when
 * the merge completes, the Train panel opens with that skill as the fine-tune
 * base and the merged dataset selected. Omitted ⇒ the merge stands alone. */
export interface MergePrefill {
  sources: string[];
  suggestedOutput?: string;
  finetuneBaseJobId?: string;
  finetuneBaseName?: string;
}

interface StudioContextValue {
  open: boolean;
  activePanel: StudioPanel;
  deployPrefill: DeployPrefill | null;
  trainPrefill: TrainPrefill | null;
  mergePrefill: MergePrefill | null;
  /** Open the studio overlay, optionally focusing a panel and seeding
   * prefills. The overlay lives on the Launchpad route — callers on other
   * routes must also navigate("/") after calling this. */
  openStudio: (
    panel?: StudioPanel,
    opts?: { deploy?: DeployPrefill; train?: TrainPrefill; merge?: MergePrefill },
  ) => void;
  closeStudio: () => void;
  /** Training job whose monitor dialog is open over the studio (null = none).
   * The dialog renders in the Train panel — see TrainingJobDialog. */
  monitorJobId: string | null;
  /** Open a job's monitor dialog. Also opens the studio on the Train panel so
   * closing the dialog lands back in the studio. Like openStudio, callers on
   * other routes must also navigate("/") — the overlay lives on Launchpad. */
  openJobMonitor: (jobId: string) => void;
  closeJobMonitor: () => void;
  setActivePanel: (panel: StudioPanel) => void;
  clearDeployPrefill: () => void;
  clearTrainPrefill: () => void;
  clearMergePrefill: () => void;
  /** Collect's recording-form draft — see CollectFormState. */
  collectForm: CollectFormState;
  updateCollectForm: (patch: Partial<CollectFormState>) => void;
  /** The most recent finished recording session, or null once handled. Drives
   * the Collect panel's handoff banner. Lives here rather than in router state
   * because a session no longer sends the user back to the Launchpad — the
   * studio stays open, so there is no navigation to hang the payload on. */
  lastRecorded: RecordedInfo | null;
  setLastRecorded: (recorded: RecordedInfo | null) => void;
}

const StudioContext = createContext<StudioContextValue | null>(null);

export const StudioProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [open, setOpen] = useState(false);
  const [activePanel, setActivePanel] = useState<StudioPanel>("collect");
  const [deployPrefill, setDeployPrefill] = useState<DeployPrefill | null>(
    null,
  );
  const [trainPrefill, setTrainPrefill] = useState<TrainPrefill | null>(null);
  const [mergePrefill, setMergePrefill] = useState<MergePrefill | null>(null);
  const [collectForm, setCollectForm] =
    useState<CollectFormState>(DEFAULT_COLLECT_FORM);
  const [lastRecorded, setLastRecorded] = useState<RecordedInfo | null>(null);

  const updateCollectForm = useCallback(
    (patch: Partial<CollectFormState>) =>
      setCollectForm((prev) => ({ ...prev, ...patch })),
    [],
  );

  const openStudio = useCallback(
    (
      panel: StudioPanel = "collect",
      opts?: { deploy?: DeployPrefill; train?: TrainPrefill; merge?: MergePrefill },
    ) => {
      if (opts?.deploy) setDeployPrefill(opts.deploy);
      if (opts?.train) setTrainPrefill(opts.train);
      if (opts?.merge) setMergePrefill(opts.merge);
      setActivePanel(panel);
      setOpen(true);
    },
    [],
  );

  const closeStudio = useCallback(() => setOpen(false), []);

  const [monitorJobId, setMonitorJobId] = useState<string | null>(null);
  const openJobMonitor = useCallback((jobId: string) => {
    setMonitorJobId(jobId);
    setActivePanel("train");
    setOpen(true);
  }, []);
  const closeJobMonitor = useCallback(() => setMonitorJobId(null), []);

  const clearDeployPrefill = useCallback(() => setDeployPrefill(null), []);
  const clearTrainPrefill = useCallback(() => setTrainPrefill(null), []);
  const clearMergePrefill = useCallback(() => setMergePrefill(null), []);

  const value = useMemo(
    () => ({
      open,
      activePanel,
      deployPrefill,
      trainPrefill,
      mergePrefill,
      openStudio,
      closeStudio,
      monitorJobId,
      openJobMonitor,
      closeJobMonitor,
      setActivePanel,
      clearDeployPrefill,
      clearTrainPrefill,
      clearMergePrefill,
      collectForm,
      updateCollectForm,
      lastRecorded,
      setLastRecorded,
    }),
    [
      open,
      activePanel,
      deployPrefill,
      trainPrefill,
      mergePrefill,
      openStudio,
      closeStudio,
      monitorJobId,
      openJobMonitor,
      closeJobMonitor,
      clearDeployPrefill,
      clearTrainPrefill,
      clearMergePrefill,
      collectForm,
      updateCollectForm,
      lastRecorded,
    ],
  );

  return (
    <StudioContext.Provider value={value}>{children}</StudioContext.Provider>
  );
};

export function useStudio(): StudioContextValue {
  const ctx = useContext(StudioContext);
  if (!ctx) throw new Error("useStudio must be used within StudioProvider");
  return ctx;
}
