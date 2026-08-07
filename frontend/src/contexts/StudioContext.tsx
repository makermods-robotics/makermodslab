import React, {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";

export type StudioPanel = "collect" | "train" | "deploy";

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

/** Pre-fills the Deploy panel when a skill card / job row says "Run on robot".
 * `job` sources resolve through the local job registry (id + optional step);
 * `hub` sources are Hub model repo ids that DeployPanel lazy-imports. */
export interface DeployPrefill {
  source: "job" | "hub";
  id: string;
  step?: number;
}

/** Pre-fills the Train panel: fine-tune base and/or a preselected dataset.
 * A local skill's fine-tune base is a job registry id (`baseJobId`); a Hub
 * skill's is a repo id (`baseModelRepoId`) that the panel lazy-imports. Set
 * exactly one of the two. `baseStep` optionally pins the checkpoint to
 * fine-tune from (the card's dropdown choice); omitted ⇒ latest. */
export interface TrainPrefill {
  baseModelRepoId?: string;
  baseJobId?: string;
  baseStep?: number;
  /** Display name for the base skill, shown in the picker while (or in case)
   * the models listing doesn't carry this id. */
  baseName?: string;
  datasetRepoId?: string;
}

interface StudioContextValue {
  open: boolean;
  activePanel: StudioPanel;
  deployPrefill: DeployPrefill | null;
  trainPrefill: TrainPrefill | null;
  /** Open the studio overlay, optionally focusing a panel and seeding
   * prefills. The overlay lives on the Launchpad route — callers on other
   * routes must also navigate("/") after calling this. */
  openStudio: (
    panel?: StudioPanel,
    opts?: { deploy?: DeployPrefill; train?: TrainPrefill },
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
  /** Collect's recording-form draft — see CollectFormState. */
  collectForm: CollectFormState;
  updateCollectForm: (patch: Partial<CollectFormState>) => void;
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
  const [collectForm, setCollectForm] =
    useState<CollectFormState>(DEFAULT_COLLECT_FORM);

  const updateCollectForm = useCallback(
    (patch: Partial<CollectFormState>) =>
      setCollectForm((prev) => ({ ...prev, ...patch })),
    [],
  );

  const openStudio = useCallback(
    (
      panel: StudioPanel = "collect",
      opts?: { deploy?: DeployPrefill; train?: TrainPrefill },
    ) => {
      if (opts?.deploy) setDeployPrefill(opts.deploy);
      if (opts?.train) setTrainPrefill(opts.train);
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

  const value = useMemo(
    () => ({
      open,
      activePanel,
      deployPrefill,
      trainPrefill,
      openStudio,
      closeStudio,
      monitorJobId,
      openJobMonitor,
      closeJobMonitor,
      setActivePanel,
      clearDeployPrefill,
      clearTrainPrefill,
      collectForm,
      updateCollectForm,
    }),
    [
      open,
      activePanel,
      deployPrefill,
      trainPrefill,
      openStudio,
      closeStudio,
      monitorJobId,
      openJobMonitor,
      closeJobMonitor,
      clearDeployPrefill,
      clearTrainPrefill,
      collectForm,
      updateCollectForm,
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
