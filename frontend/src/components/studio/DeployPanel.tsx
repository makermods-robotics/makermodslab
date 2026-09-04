import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Trans, useTranslation } from "react-i18next";
import {
  AlertTriangle,
  ChevronsUpDown,
  Loader2,
  Play,
  // No VideoOff (the rework dropped CameraThumbnail for SessionCameraList) and
  // no Square: a live run's Stop lives in the session dialog (local) or in the
  // Remote tab's status panel, never beside Start.
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useStudio } from "@/contexts/StudioContext";
import { useInferenceSession } from "@/contexts/InferenceSessionContext";
import { useRobots, jointsPerArm } from "@/hooks/useRobots";
import { formatRobotSetupGap } from "@/lib/robotSetupGap";
import { useInferenceLaunch } from "@/hooks/useInferenceLaunch";
import {
  JobCheckpoint,
  PolicyConfigSummary,
  getCheckpointPolicyConfig,
  listJobCheckpoints,
} from "@/lib/checkpointsApi";
import { InferenceStatus, getInferenceStatus } from "@/lib/inferenceApi";
import { startSession, formatSessionHeld } from "@/lib/sessionApi";
import { tabOwnerId } from "@/lib/sessionOwner";
import { JobRecord, getJob, jobDisplayName } from "@/lib/jobsApi";
import { getDatasetInfo } from "@/lib/replayApi";
import { SkillItem } from "@/lib/modelsApi";
import { useSkills } from "@/hooks/useSkills";
import { importSourceForModel } from "@/lib/inferenceLaunch";
import { deployBlockedReason } from "./deployGuards";
import type { DeployRunMode } from "./deployGuards";
import { useSessionHeartbeat } from "@/hooks/useSessionHeartbeat";
import { useRemoteInferenceStatus } from "@/hooks/useRemoteInferenceStatus";
import {
  transportIsReady,
  useRemoteInferenceTransport,
} from "@/hooks/useRemoteInferenceTransport";
import {
  useGpuKnobs,
  useGpuLauncher,
  useGpuTargets,
} from "@/hooks/useGpuLauncher";
import CameraRoleBindings, {
  type CameraRoleOption,
  type CameraRoleSlot,
} from "@/components/remote-inference/CameraRoleBindings";
import GpuLaunchSection from "@/components/remote-inference/GpuLaunchSection";
import RemoteManualSection from "@/components/remote-inference/RemoteManualSection";
import RemoteAdvancedSection from "@/components/remote-inference/RemoteAdvancedSection";
import {
  SFU_OFF_SUMMARY_KEY,
  summarizeTransport,
} from "@/components/remote-inference/transportSummary";
import { POLICY_PATH_PLACEHOLDER } from "@/components/remote-inference/modalCommand";
import {
  horizonForEngine,
  defaultEngineForPolicy,
  policySupportsRtc,
  armSupportsRemoteInference,
  DEFAULT_REMOTE_RUN_CONFIG,
  type RemoteEngine,
  type RemoteRunConfig,
} from "@/components/remote-inference/remoteRunConfig";
import {
  remoteCameraRoleKey,
  useRemoteCameraRoles,
} from "@/hooks/useRemoteCameraRoles";
import DisplayName from "@/components/library/DisplayName";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import ModelsLibrary from "@/components/jobs/ModelsLibrary";
import ModelPicker from "@/components/landing/ModelPicker";
import PolicyExtraDialog from "@/components/training/PolicyExtraDialog";
import {
  AdvancedSection,
  LibrarySection,
  PanelEntryControl,
  PanelHeader,
  RobotStatus,
  SLIDE,
  useEyebrowClass,
} from "@/components/studio/panel/primitives";
import { cn } from "@/lib/utils";
import { useAvailableCameras } from "@/hooks/useAvailableCameras";
// Collect's read-only camera list, rendered here verbatim: both panels are
// answering the same question (which of this robot's cameras will be used), so
// they are the same component rather than two views of one robot record.
import {
  SessionCameraList,
  type CameraConfig,
} from "@/components/recording/CameraConfiguration";
import { isCameraConnected } from "@/lib/cameraResolve";
import MilestoneReveal from "@/components/onboarding/MilestoneReveal";
import { useOnceFlag } from "@/lib/onboarding/storage";

/**
 * Studio panel 3 · Deploy — run a skill (local trained checkpoint or an
 * imported Hub model) on the corner robot. Every "Run on robot" action lands
 * here via `useStudio().deployPrefill`.
 *
 * This is a PARALLEL surface to the legacy `InferenceModal` (still used by
 * JobsSection + the Landing Models panel through `useInferenceLaunch`). To keep
 * those consumers untouched and avoid drift, the checkpoint/policy-config
 * fetch, the bimanual `left_` camera-prefix round-trip, the state_dim 6-vs-12
 * arm-count guard and the start flow are ported VERBATIM from
 * `components/landing/InferenceModal.tsx` (only the palette becomes token
 * classes). The Hub lazy-import reuses `useInferenceLaunch().importSource` so
 * the husk-repo messaging is identical, not re-implemented.
 *
 * Cameras are the one place this panel deliberately left the modal behind: the
 * ported per-feature binding dropdowns are gone, replaced by Collect's
 * read-only SessionCameraList plus name-based binding (see `cameraBindings`).
 */

// Mirrors rollout.MAX_EVAL_EPISODES — the server clamps to the same bound, this
// just stops the stepper from offering a number that would be silently reduced.
const MAX_EVAL_EPISODES = 200;

// Mirrors rollout.MAX_COACHING_CORRECTIONS. Far lower than the eval bound
// because every correction is a hands-on takeover — the operator is standing at
// the arm for all of them.
const MAX_COACHING_CORRECTIONS = 100;

// The shapes a Deploy run can take, as ONE value the guards and the launch
// still read. Since S3.9 it is DERIVED rather than chosen: the panel asks two
// independent questions instead — where the policy runs, and what the operator
// does — and this is the pair collapsed into the vocabulary the backend and
// `deployGuards` already speak.
//
// "remote" is the DRTC run: the same checkpoint, the same robot, the same
// cameras — but the policy runs on a remote GPU and the two meet in a LiveKit
// room. It is a separate SESSION KIND server-side (remote_inference), not a
// flag on inference.
type RunMode = DeployRunMode;

/** WHERE the policy runs. The arm is always here. */
type RunsOn = "local" | "remote";

/** WHAT the operator does while it runs. */
type OperatorMode = "single" | "coach";

/** How long a remote run may be, expressed as "no limit". The backend's own
 * unbounded contract for `duration_s`; a LOCAL rollout is handed the number
 * verbatim and would stop the instant it started, which is why the two read it
 * differently and `deployGuards` asks per mode. */
const UNBOUNDED_DURATION_S = 0;

/** How often the transport re-probes itself while the remote half of the form
 * is open and nothing is running.
 *
 * Slow on purpose: each probe opens a real (short) `list_participants` call
 * against the SFU, and the thing it is watching for — a GPU joining the room —
 * takes 1-3 minutes to happen. This is the fallback for a GPU this Lab did not
 * launch; one launched from the card here re-probes on its own transitions the
 * moment it lands. */
const TRANSPORT_REPROBE_MS = 15_000;

/** Stable empty list, so "no checkpoints for the current skill" never hands
 * children a fresh array identity on every render. */
const NO_CHECKPOINTS: JobCheckpoint[] = [];

/** Coefficient of the original ACT paper's exponential weighting (see
 * lerobot's ACTTemporalEnsembler). Offered as the starting point when the user
 * switches temporal ensembling on. */
const DEFAULT_TEMPORAL_ENSEMBLE_COEFF = 0.01;

/** The studio's form-field trigger size — what a bare shadcn <SelectTrigger>
 * and <Input> already are (h-10, full width, text-sm), spelled out for the one
 * control that ships a card-sized trigger of its own. `cn` is tailwind-merge,
 * so these win over the component's defaults rather than fighting them. */
const FIELD_TRIGGER = "h-10 w-full text-sm";

/**
 * One camera as the panel sees it. The BiSO prefix round-trip lives here so the
 * future per-arm routing work has a single obvious place to extend.
 *
 * (Verbatim port of InferenceModal's CameraMapping / cameraMappings — see that
 * file's doc comment for the full BiSO `left_` prefix rationale. Kept here so
 * the legacy modal stays untouched for its existing consumers.)
 */
interface CameraMapping {
  /** Checkpoint feature key — the key into `policyConfig.image_features`. */
  feature: string;
  /** Name shown in the UI. Stripped bare name (bimanual) or `feature`. */
  display: string;
  /** Key used in the start-inference camera dict. Equals `display`. */
  requestKey: string;
}

const ARM_PREFIX_RE = /^(left|right)_/;

/**
 * Build the display/request mapping for the checkpoint's camera features.
 *
 * Single-arm: identity — bare names pass through untouched.
 *
 * Bimanual: strip the `left_`/`right_` prefix that BiSO added at record time,
 * so the user sees the name they chose and the rollout re-prefixes it correctly.
 * Guard against collisions: if two features would strip to the same bare name,
 * fall back to the FULL feature name for every colliding entry — correctness
 * over cosmetics.
 */
function cameraMappings(
  features: string[],
  isBimanual: boolean,
): CameraMapping[] {
  if (!isBimanual) {
    return features.map((f) => ({ feature: f, display: f, requestKey: f }));
  }
  const strippedCounts = new Map<string, number>();
  for (const f of features) {
    const bare = f.replace(ARM_PREFIX_RE, "");
    strippedCounts.set(bare, (strippedCounts.get(bare) ?? 0) + 1);
  }
  return features.map((f) => {
    const bare = f.replace(ARM_PREFIX_RE, "");
    const name = strippedCounts.get(bare) === 1 ? bare : f;
    return { feature: f, display: name, requestKey: name };
  });
}

/**
 * The operator axis, as tabs: what the PERSON does while the policy runs.
 *
 * Two axes rather than one strip of three. "Run it remotely" was never a
 * sibling of "Run" and "Human in the loop" — it answers a different question
 * (where the weights are loaded), and putting it beside them made every field
 * that belongs to BOTH questions (the engine, the duration, the cameras) live
 * inside one of the three and vanish from the others. So WHERE moved to its own
 * segmented control above this strip, and this strip is only the operator's own
 * involvement: hands off, or hands on the leader.
 *
 * Eval ("Score it") is deliberately not a tab: the scored-evaluation engine
 * still exists and `eval` is still a valid RunMode a prefill can name, and it
 * renders inside the Run tab (its episode count shows there) rather than as a
 * third thing to choose between.
 */
const OPERATOR_TABS: {
  value: OperatorMode;
  // Key stem under `studio.deploy.runMode`; the strip resolves `${stem}.title`
  // for the tab label and `.what` / `.commitment` for the line beneath it.
  stem: string;
  handsOn?: boolean;
}[] = [
  { value: "single", stem: "single" },
  { value: "coach", stem: "coach", handsOn: true },
];

const DeployPanel: React.FC = () => {
  const { t } = useTranslation();
  const eyebrow = useEyebrowClass();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { open, deployPrefill, clearDeployPrefill } = useStudio();
  const { openInferenceSession, sessionOpen } = useInferenceSession();
  const { selectedRecord: robot } = useRobots();
  // Reuse the shared lazy-import (husk-repo messaging + idempotent registration)
  // so a Hub skill resolves to a pseudo-job exactly as the Jobs cards do.
  const { importSource } = useInferenceLaunch();
  // --- Skill picker state ------------------------------------------------
  // The listing is NOT owned here. It is one app-wide fetch behind the
  // `jobs_changed` push (see ModelsDataContext), so a run that finishes while
  // this panel is open appears in the picker without waiting for a reopen —
  // and the launchpad's slider, the Train panel's Starting point and this
  // picker can no longer be looking at three different snapshots.
  const {
    skills,
    hub: hubStatus,
    loading: modelsLoading,
    error: modelsError,
    refresh: refreshModels,
  } = useSkills();
  // The picker offers what can actually run. A row that is not deployable is
  // still in the listing — carrying WHY, so the library can explain it — but it
  // has no business being selectable here.
  const models: SkillItem[] = useMemo(
    () => skills.filter((s) => s.deployable),
    [skills],
  );
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<JobRecord | null>(null);
  const [resolving, setResolving] = useState(false);

  // (No client-side deny-list here any more. The rework derived one from the
  // job registry's `checkpoint_count` to keep a cloud run that died before its
  // first checkpoint out of the picker; `/skills` now answers that question
  // server-side and better — `deployable` means the weights are loadable AND
  // nothing supersedes the row — so `models` above is already the filtered
  // set and a second, weaker filter over it would only be drift waiting to
  // happen.)

  // The run form slides open in place under the panel's entry control, same as
  // Collect's "Record new dataset" and Train's "Start a new training"; the
  // skills library folds to its header while it is open (still expandable by
  // hand). Everything that configures a run lives inside it, the skill picker
  // first.
  const [formOpen, setFormOpen] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(true);

  const toggleForm = useCallback((open: boolean) => {
    setFormOpen(open);
    setLibraryOpen(!open);
  }, []);

  // --- Inference config state (ported from InferenceModal) ---------------
  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  // Keyed on `ref`, NOT step. The picker lists a whole resume lineage, and a
  // rewind (resuming from an ancestor's checkpoint) legitimately produces two
  // DIFFERENT checkpoints at the same step. Keying on step made the second one
  // unselectable — `find(c => c.step === selectedStep)` always returns the
  // first — so picking it silently deployed the other one's weights.
  const [selectedRef, setSelectedRef] = useState<string | null>(null);
  // Callers outside this panel (a deploy deep link, a ModelsLibrary card) can
  // only name a STEP — they have no ref. Park it until the checkpoint list
  // arrives, then resolve it to a ref. A step-only request is inherently
  // ambiguous on a rewound lineage; it resolves to the first match, which is
  // what these callers got before. Selection made INSIDE the panel is always
  // ref-exact.
  // Resolved by the effect below, which keys on the checkpoint LIST rather than
  // on the job — resolving it inside the fetch coupled it to `jobId` changing,
  // so picking a step on the ALREADY-selected model stranded it and dropped the
  // dropdown to its placeholder.
  const [pendingStep, setPendingStep] = useState<number | null>(null);

  // Attribution for the dropdown, built from the server's own owner fields. The
  // dropdown renders it only when the list actually spans more than one run, so
  // a plain (unresumed) skill shows nothing extra.
  const checkpointOwnerMap = useMemo(() => {
    const out: Record<
      string,
      { name: string; number: number; detail: string }
    > = {};
    for (const c of checkpoints) {
      if (!c.owner_job_id) continue;
      out[c.ref] = {
        name: c.owner_name ?? c.owner_job_id,
        number: c.owner_job_number ?? 0,
        detail: c.owner_job_id,
      };
    }
    return out;
  }, [checkpoints]);

  const selectedCheckpoint =
    selectedRef != null
      ? (checkpoints.find((c) => c.ref === selectedRef) ?? null)
      : null;
  // The step is now DERIVED from the selected checkpoint, never the other way
  // round — it is a label, not an identity.
  const selectedStep = selectedCheckpoint?.step ?? null;
  const [task, setTask] = useState("");
  const [durationS, setDurationS] = useState(60);
  // Multi-episode evaluation. 1 (the default) is the plain single rollout; >1
  // switches the session dialog into eval mode — N scored episodes with a reset
  // between each and an accuracy at the end. Clamped again server-side.
  const [evalEpisodes, setEvalEpisodes] = useState(1);
  // The two axes. WHERE the policy runs and WHAT the operator does are
  // independent questions, so they are two controls; `runMode` below collapses
  // them back into the one word the guards and the launch speak.
  const [runsOn, setRunsOn] = useState<RunsOn>("local");
  const [operatorMode, setOperatorMode] = useState<OperatorMode>("single");
  // Scored evaluation, which no control on this panel offers any more: it is
  // reachable only through a prefill (the session dialog's own "score it"
  // handoff), and it renders inside the Run tab as an episode count.
  const [scoring, setScoring] = useState(false);
  // Coaching (DAgger): run the policy, take over when it's about to fail, and
  // record each takeover as training data. See StartInferenceRequest.coaching.
  const [targetCorrections, setTargetCorrections] = useState(10);
  const [coachDatasetName, setCoachDatasetName] = useState("");
  // Task strings from the dataset this checkpoint was trained on. A
  // language-conditioned policy is steered by this string, and a wrong one
  // doesn't fail loudly — it just makes the policy worse in ways that look
  // like the policy being bad. So a single unambiguous task is filled in, and
  // several are offered as choices rather than guessed between.
  const [datasetTasks, setDatasetTasks] = useState<string[]>([]);
  // (The engine is no longer a state of its own. It is ONE field for both
  // places a run can happen — see `engine` below, which reads it off
  // `remoteConfig` so the GPU card and the generated `modal run` line are built
  // from the same value the local rollout would be started with.)
  const [submitting, setSubmitting] = useState(false);
  // ACT temporal ensembling. Held as (on, coeff) rather than `number | null`
  // so clearing the number field mid-edit doesn't silently switch the feature
  // off; the request sends the coeff only while `on`.
  const [temporalEnsemble, setTemporalEnsemble] = useState(false);
  const [temporalEnsembleCoeff, setTemporalEnsembleCoeff] = useState<
    number | undefined
  >(DEFAULT_TEMPORAL_ENSEMBLE_COEFF);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // The REMOTE Advanced block (the transport knobs) opens independently of the
  // ACT one above: they are never both on screen, but sharing one flag would
  // make opening one silently open the other on the next switch.
  const [transportAdvancedOpen, setTransportAdvancedOpen] = useState(false);

  const [policyConfig, setPolicyConfig] = useState<PolicyConfigSummary | null>(
    null,
  );
  const [policyConfigLoading, setPolicyConfigLoading] = useState(false);
  const [policyConfigError, setPolicyConfigError] = useState<string | null>(
    null,
  );
  const [policyExtra, setPolicyExtra] = useState<{
    policyType: string;
    packageName: string;
    installTarget: string;
    installHint: string;
  } | null>(null);
  const [checkingExtra, setCheckingExtra] = useState(false);

  const { cameras: availableCameras } = useAvailableCameras({ enabled: open });

  // Light status poll while the panel is visible so the launch guards know
  // whether a rollout is already running, and so SessionCameraList keeps its
  // previews released for as long as that rollout holds the devices.
  const [status, setStatus] = useState<InferenceStatus | null>(null);

  // --- Remote inference (DRTC) -------------------------------------------
  // Everything about this mode lives in components/remote-inference/; the
  // panel holds only what the START request and the guards need.
  const [remoteConfig, setRemoteConfig] = useState<RemoteRunConfig>(
    DEFAULT_REMOTE_RUN_CONFIG,
  );
  // 1 Hz while a remote run is live, a slow tick while this panel is open, and
  // an eager refetch on every `session_changed` hint. Read here only for the
  // guards and the camera previews — the RUN itself is shown in the session
  // dialog now, which polls this same status for itself.
  const { status: remoteStatus } = useRemoteInferenceStatus(open);
  const remoteActive = remoteStatus?.remote_inference_active === true;
  const remote = runsOn === "remote";
  // The probe opens a real (short) `list_participants` call against the SFU, so
  // it is read while the remote half of the form is on screen rather than
  // app-wide. It re-probes on the GPU's own transitions and on a slow timer —
  // see the two effects below, which replaced the hand-driven Re-check button
  // the retired Transport section carried.
  const remoteTransport = useRemoteInferenceTransport(open && remote);
  // The GPU is a LAB-LEVEL resource, not part of the session: it holds no
  // hardware, stopping it is not a safety action, and it outlives the run. It
  // is owned HERE, and only here — the session dialog reads its status for one
  // billing line and offers no control over it.
  const gpu = useGpuLauncher(open && remote);
  // WHICH WORKSPACE PAYS. Its own hook beside the launcher rather than inside
  // it: the listing is a read of this MACHINE (two `modal … list --json`
  // calls), not of the launch, and it must keep answering — and keep being
  // pickable — while a GPU is up.
  const gpuTargets = useGpuTargets(open && remote);
  // WHAT IT RUNS AS and WHAT IT RUNS ON (S3.8e). Remembered per Lab like the
  // target above, and owned here for the same reason: the picker lives under
  // Advanced, the launch reads it from the GPU card, and the generated `modal
  // run` line has to say the same thing as both.
  const gpuKnobs = useGpuKnobs();

  // Human in the loop is not startable on a remote run: the GPU child has no
  // takeover protocol, so there is no way to hand the arm to the leader
  // mid-chunk. The tab is DISABLED rather than hidden (a tab that disappears
  // reads as a bug, and the reason is worth saying), and this is the belt to
  // that brace — a coach selection standing while Remote GPU is picked can
  // never reach the launch.
  const effectiveOperator: OperatorMode = remote ? "single" : operatorMode;
  const coaching = effectiveOperator === "coach";
  // The two axes collapsed into the vocabulary the guards and the backend use.
  const runMode: RunMode = remote
    ? "remote"
    : coaching
      ? "coach"
      : scoring
        ? "eval"
        : "single";

  /** The checkpoint's own chunk width — the CEILING on the horizon. */
  const checkpointHorizon = policyConfig?.n_action_steps ?? null;

  // ONE engine for both places a run can happen. It lives on `remoteConfig`
  // because the GPU card and the generated `modal run` line are built from that
  // object and MUST agree with it; the local rollout reads the same value as
  // `inference_engine`. Coaching pins sync — server-side too — but pins it
  // without writing to state, so switching back off the coach tab restores
  // whatever was chosen.
  const engine = remoteConfig.engine;
  const setEngine = useCallback(
    (next: RemoteEngine) =>
      setRemoteConfig((prev) => ({
        ...prev,
        engine: next,
        // Switching engines re-seeds the horizon, because the two regimes want
        // different ones (one open-loop ACT block vs the flow families' full
        // chunk_size) and a horizon carried over from the other engine is the
        // mismatch Portal drops packets over. An operator who has already typed
        // their own keeps it.
        //
        // Both sides of that comparison go through `horizonForEngine`, so the
        // checkpoint's ceiling holds across the switch: without it, a 30-step
        // checkpoint seeded to 16 for sync would read as "the sync default,
        // untouched" and be re-seeded to the rtc default of 50 — straight past
        // the ceiling, into a silently dropped run.
        horizon:
          prev.horizon !==
          horizonForEngine(prev.engine, checkpointHorizon)
            ? prev.horizon
            : horizonForEngine(next, checkpointHorizon),
      })),
    [checkpointHorizon],
  );

  // Whether this checkpoint can be in-painted at all — the rtc engine's whole
  // premise. The whole policy config goes in, not just its type: the SERVER's
  // `supports_rtc` is the answer whenever it has one, and the frontend's own
  // family list only decides for a policy type newer than the server's table.
  // Unknown on both sides counts as "no": guessing rtc would pair the arm with
  // a GPU server the operator was never told to start.
  const rtcSupported = policySupportsRtc(policyConfig);

  // Preselect the engine from the checkpoint's policy family, and the horizon
  // from the engine. A flow policy defaults to rtc because that is the whole
  // reason the engine exists: at ~400 ms round trip the sync player re-plans
  // about once a second, and two flow-policy plans made 400 ms apart disagree
  // at every seam — a visible ~1 Hz twitch with a perfectly healthy transport.
  //
  // Keyed on everything the seed READS, not on the policy type alone — since
  // S3.7b the horizon also follows the checkpoint's own `n_action_steps`, and
  // two checkpoints of the same family can declare different ones (MolmoAct2's
  // published checkpoint returns 30 against an rtc default of 50). Keying on
  // the type alone would leave the first checkpoint's horizon standing over the
  // second, which is the exact mismatch Portal drops every packet over.
  //
  // Never while a run is live (the fields are disabled then, and re-seeding
  // under a live run would make the generated command disagree with the arm).
  // It seeds a DEFAULT, so it deliberately overwrites: an engine left on rtc
  // from the previous checkpoint is exactly the state this exists to correct.
  //
  // Since S3.9 this seeds the ONE shared engine, so a flow checkpoint now
  // arrives on rtc for a local rollout too — previously the local picker always
  // opened on sync. That is the same informed default, applied to the same
  // question — and it is the SAME fact the picker gates on, so the seed can
  // never land on an engine the field would then refuse to offer.
  const seededEngineFor = useRef<string | null>(null);
  useEffect(() => {
    const policyType = policyConfig?.policy_type ?? null;
    if (!policyType || remoteActive) return;
    const seedKey = [
      policyType,
      policyConfig?.supports_rtc ?? "?",
      policyConfig?.n_action_steps ?? "?",
    ].join("|");
    if (seededEngineFor.current === seedKey) return;
    seededEngineFor.current = seedKey;
    const engine = defaultEngineForPolicy(policyConfig);
    setRemoteConfig((prev) => ({
      ...prev,
      engine,
      horizon: horizonForEngine(engine, policyConfig?.n_action_steps),
    }));
  }, [policyConfig, remoteActive]);

  // Edge-triggered "consume once": handleStart sets the pending flag, and the
  // effect below latches it into showDeployMilestone the first time the live
  // InferenceSessionDialog closes, then clears the pending flag so it can't
  // re-trigger — sessionOpen cycling true→false again later (a normal
  // redeploy from this same panel) must not resurrect this banner.
  const { seen: hasSeenDeployMilestone, markSeen: markDeployMilestoneSeen } =
    useOnceFlag("makerlab:milestone-first-deploy");
  const [deployMilestonePending, setDeployMilestonePending] = useState(false);
  const [showDeployMilestone, setShowDeployMilestone] = useState(false);

  useEffect(() => {
    if (!sessionOpen && deployMilestonePending) {
      setShowDeployMilestone(true);
      setDeployMilestonePending(false);
    }
  }, [sessionOpen, deployMilestonePending]);

  const jobId = selectedJob?.id ?? null;
  // Address the policy-config endpoint by the checkpoint's OWNER. `(owner, step)`
  // is unique even on a rewound lineage, where `(tip, step)` is not. Falls back
  // to the tip when there is no owner — a single-run listing, where the step is
  // unique anyway.
  const policyConfigJobId = selectedCheckpoint?.owner_job_id ?? jobId;
  const isBimanual = robot?.mode === "bimanual";

  const cameraMap = useMemo(
    () =>
      cameraMappings(
        Object.keys(policyConfig?.image_features ?? {}),
        isBimanual,
      ),
    [policyConfig, isBimanual],
  );

  const robotCameras: CameraConfig[] = useMemo(
    () => robot?.cameras ?? [],
    [robot],
  );

  // Per-role camera picks for a REMOTE run, remembered per (checkpoint, robot).
  // The checkpoint half is the pair that addresses its policy config — the
  // owning job id and the checkpoint ref — because that is what "this
  // checkpoint" means everywhere else in this panel. A remembered pick naming a
  // camera the record no longer holds is dropped on read, inside the hook.
  const robotCameraNames = useMemo(
    () => robotCameras.map((c) => c.name),
    [robotCameras],
  );
  const { roles: remoteCameraRoles, setRole: setRemoteCameraRole } =
    useRemoteCameraRoles(
      remoteCameraRoleKey(policyConfigJobId, selectedRef),
      robot?.name ?? null,
      robotCameraNames,
    );

  /**
   * The camera bindings, DERIVED BY NAME rather than chosen: each camera the
   * checkpoint was trained with takes the robot camera of the same name.
   *
   * There is no picker any more — the panel shows the robot's cameras exactly
   * as Collect does (SessionCameraList), and a mismatch is reported rather than
   * repaired here, because a camera's name is a property of the robot and
   * Robot settings is the one place it is edited. Name matching is
   * case-insensitive, but the PAYLOAD carries the robot record's own spelling.
   *
   * `display` is the bare name in bimanual mode (cameraMappings strips BiSO's
   * `left_`/`right_` prefix), which is exactly what the robot record stores, so
   * the round-trip still works: the rollout re-prefixes it back into the
   * checkpoint's `left_<name>` feature.
   */
  const cameraBindings = useMemo(
    () =>
      cameraMap.map((mapping) => {
        const camera =
          robotCameras.find(
            (c) => c.name.toLowerCase() === mapping.display.toLowerCase(),
          ) ?? null;
        const dims = policyConfig?.image_features[mapping.feature];
        return {
          mapping,
          camera,
          dims,
          // A stored camera_index goes stale on replug, so presence is judged
          // by unique_id against the live enumeration — the same check the
          // preview cards use.
          connected:
            camera != null && isCameraConnected(camera, availableCameras),
          resolutionDiffers:
            camera != null &&
            dims != null &&
            (camera.width !== dims.width || camera.height !== dims.height),
        };
      }),
    [cameraMap, robotCameras, policyConfig, availableCameras],
  );

  /**
   * The same bindings, with the operator's per-role picks filled in where
   * nothing matched by name.
   *
   * A checkpoint's camera name is a ROLE, not a claim about this robot —
   * `lerobot/MolmoAct2-SO100_101-LeRobot` names `cam0` / `cam1`, which no robot
   * record has ever been called — and a role nothing matches is a question the
   * operator has to answer, not a defect in the record. A robot camera's name
   * stays its identity: nothing here renames anything, and the pick is
   * remembered per (checkpoint, robot) rather than written to the record.
   *
   * Layered over the name-derived list rather than replacing it, so a name
   * match ALWAYS wins. A pick is re-checked against the record on every render
   * (a camera deleted in Robot settings simply stops resolving) and its
   * connectedness is judged the same way a name match's is — a pick of an
   * unplugged camera is a pick, not a binding, and still blocks Start.
   *
   * Every mode reads this since S3.9. The picker was remote-only when it landed
   * (S3.7b) purely because the remote run was the first thing anyone pointed at
   * a checkpoint whose cameras are called `cam0`/`cam1` — but the question it
   * answers ("which camera plays this role?") is nothing to do with where the
   * policy runs, and a local run had no answer to it at all.
   */
  const boundCameraBindings = useMemo(
    () =>
      cameraBindings.map((b) => {
        if (b.camera != null) return b;
        const picked = remoteCameraRoles[b.mapping.requestKey];
        const camera = picked
          ? (robotCameras.find((c) => c.name === picked) ?? null)
          : null;
        if (camera == null) return b;
        return {
          ...b,
          camera,
          connected: isCameraConnected(camera, availableCameras),
          resolutionDiffers:
            b.dims != null &&
            (camera.width !== b.dims.width || camera.height !== b.dims.height),
        };
      }),
    [cameraBindings, remoteCameraRoles, robotCameras, availableCameras],
  );

  /** Cameras the policy needs that this robot has nothing named for AND nothing
   * picked for. Start is blocked on these: the rollout cannot invent the feed. */
  const unmatchedCameras = boundCameraBindings.filter((b) => b.camera == null);
  /** Bound, but not enumerated right now (unplugged since the record was
   * saved). Also blocks Start — the same strictness the picker had. */
  const disconnectedCameras = boundCameraBindings.filter(
    (b) => b.camera != null && !b.connected,
  );
  /** Bound and present, but the robot captures at a different size than the
   * policy trained on. A warning only: the rollout forwards the checkpoint's
   * dims as `camera_dims` and the camera is opened at those. */
  const mismatchedCameras = boundCameraBindings.filter(
    (b) => b.resolutionDiffers,
  );

  /** The picker's slots: the roles with NO name match, in checkpoint order.
   * A matched role gets no control — there is no decision to make. */
  const cameraRoleSlots: CameraRoleSlot[] = useMemo(
    () =>
      cameraBindings
        .filter((b) => b.camera == null)
        .map((b) => ({
          requestKey: b.mapping.requestKey,
          display: b.mapping.display,
          dims: b.dims,
          selected: remoteCameraRoles[b.mapping.requestKey] ?? null,
        })),
    [cameraBindings, remoteCameraRoles],
  );

  /** The robot's cameras as options. Names are the record's own spelling —
   * the value the start request carries. */
  const cameraRoleOptions: CameraRoleOption[] = useMemo(
    () =>
      robotCameras.map((c) => ({
        name: c.name,
        width: c.width,
        height: c.height,
        connected: isCameraConnected(c, availableCameras),
      })),
    [robotCameras, availableCameras],
  );

  // Opening the studio is a freshness gesture, so it still re-pulls — but it is
  // no longer the ONLY thing that does, which is what made a run completing
  // behind an open panel invisible until it was closed and reopened.
  useEffect(() => {
    if (open) refreshModels();
  }, [open, refreshModels]);

  // (Importing lives in the skills library's own header, which owns its modal
  // and its own refetch — this panel no longer duplicates that entry point.
  // Nothing is lost by dropping the local `handleImported`: the listing is the
  // app-wide ModelsDataContext one, so the library's own refresh repopulates
  // this picker as well.)

  // Apply a "Run on robot" prefill: source "job" selects that job (+ optional
  // step); source "hub" lazy-imports the repo, then selects the pseudo-job.
  // Cleared only by the run that actually finished resolving THIS prefill — a
  // cancelled (superseded) run must not clear a newer prefill out from under
  // the run that is handling it.
  useEffect(() => {
    if (!deployPrefill) return;
    let cancelled = false;
    (async () => {
      setResolving(true);
      // A prefill is an intent to configure a run, so it slides the form open
      // too — the same move Train's prefill effect makes. Without it the skill
      // resolved below would land inside a form the user still has to open by
      // hand, and the handoff would look like nothing happened.
      toggleForm(true);
      try {
        if (deployPrefill.source === "job") {
          const job = await getJob(baseUrl, fetchWithHeaders, deployPrefill.id);
          if (cancelled) return;
          setPendingStep(deployPrefill.step ?? null);
          setSelectedJob(job);
          setSelectedModelId(job.id);
        } else {
          const imported = await importSource(deployPrefill.id);
          if (cancelled || !imported) return;
          setPendingStep(deployPrefill.step ?? null);
          setSelectedJob(imported);
          setSelectedModelId(imported.id);
        }
      } catch (e) {
        if (!cancelled) {
          toast({
            title: t("studio.deploy.toast.loadPolicyFailed"),
            // The thrown error's own text — shown exactly as raised.
            description: e instanceof Error ? e.message : String(e),
            variant: "destructive",
          });
        }
      } finally {
        if (!cancelled) {
          setResolving(false);
          clearDeployPrefill();
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [
    deployPrefill,
    baseUrl,
    fetchWithHeaders,
    importSource,
    clearDeployPrefill,
    toggleForm,
    toast,
    t,
  ]);

  // Manual skill pick: resolve the chosen model to a launchable job (its own
  // registry id, an already-imported repo, or a fresh lazy import).
  //
  // NOTHING is committed until that resolution succeeds. `selectedModelId`
  // drives the picker's checkmark and `selectedJob` drives its label, the
  // checkpoint list and the launch — so committing the id up front made a
  // failed resolve leave the two disagreeing: the tick sat on the skill the
  // user had just clicked while every other control, Start included, still
  // belonged to the previous one. A failed pick now changes nothing at all,
  // which is what the "leave the prior selection" catch below always meant.
  const handlePickSkill = useCallback(
    async (modelId: string) => {
      const model = models.find((m) => m.id === modelId);
      if (!model) return;
      setResolving(true);
      try {
        // `job_id` is stamped by the server, which already ranks the runs
        // sharing an output repo (`_job_outranks`). This used to re-list up to
        // 200 jobs and re-implement that ranking in TypeScript — a second copy
        // of the definition, kept in sync by hand, and blind to any run past
        // the scan limit.
        let resolved: JobRecord | null = null;
        if (model.job_id) {
          try {
            resolved = await getJob(baseUrl, fetchWithHeaders, model.job_id);
          } catch {
            // The record went away between the listing and this click (deleted
            // in another tab, or from the Train panel). The weights may still be
            // on the Hub, so fall through to the import rather than dead-end —
            // the path the old lookup took whenever it found no job at all.
          }
        }
        if (!resolved) {
          // No run tracks it (a bare Hub repo, a scanned directory), or the one
          // that did is gone — the lazy import registers one, as before.
          resolved = (await importSource(importSourceForModel(model))) ?? null;
        }
        if (!resolved) return;
        // New skill → drop the prior checkpoint selection so the load effect
        // takes the new job's latest.
        setSelectedRef(null);
        setPendingStep(null);
        setSelectedJob(resolved);
        setSelectedModelId(modelId);
      } catch {
        // Resolution failed → leave the prior selection; a toast already fired
        // for the import path.
      } finally {
        setResolving(false);
      }
    },
    [models, baseUrl, fetchWithHeaders, importSource],
  );

  // Load checkpoints when the selected job changes.
  useEffect(() => {
    if (!open || !jobId) {
      setCheckpoints([]);
      return;
    }
    let cancelled = false;
    // Lineage-wide: a resumed run and the run it resumed are ONE skill, and
    // the picker must offer every step the chain trained through — not just
    // the ones this link happened to save.
    listJobCheckpoints(baseUrl, fetchWithHeaders, jobId, undefined, true)
      .then((cks) => {
        if (cancelled) return;
        setCheckpoints(cks);
        if (cks.length > 0) {
          // Ascending by step, so the last entry is the newest. A ref that is
          // no longer in the list (the run was deleted, or a rewind rewrote
          // the chain) falls back to the latest rather than leaving a
          // selection that resolves to nothing.
          const latest = cks[cks.length - 1].ref;
          setSelectedRef((prev) =>
            prev != null && cks.some((c) => c.ref === prev) ? prev : latest,
          );
        }
      })
      .catch(() => {
        if (cancelled) return;
        setCheckpoints([]);
        // Never leave a parked step behind: it would resolve against whichever
        // list arrives next.
        setPendingStep(null);
      });
    return () => {
      cancelled = true;
    };
  }, [open, baseUrl, fetchWithHeaders, jobId]);

  // Resolve a step parked by a caller that had no ref (a deploy deep link, a
  // ModelsLibrary row). Keyed on the checkpoint LIST, not on the job, so it
  // fires both when the list has just arrived AND when it was already loaded —
  // the latter is picking a step on the model that is already selected, which
  // resolving inside the fetch could never handle.
  //
  // A step-only request is ambiguous on a rewound lineage; `find` takes the
  // first match, which is the tip's checkpoint (list_chain_checkpoints appends
  // the tip first and the sort is stable). That is the same one the backend's
  // own-first policy-config lookup returns, so the two agree.
  useEffect(() => {
    if (pendingStep == null || checkpoints.length === 0) return;
    const match = checkpoints.find((c) => c.step === pendingStep);
    setPendingStep(null);
    if (match) setSelectedRef(match.ref);
  }, [pendingStep, checkpoints]);

  // Load policy config when step changes.
  useEffect(() => {
    if (!open || !policyConfigJobId || selectedStep == null) {
      setPolicyConfig(null);
      setPolicyConfigError(null);
      return;
    }
    let cancelled = false;
    setPolicyConfigLoading(true);
    setPolicyConfigError(null);
    getCheckpointPolicyConfig(
      baseUrl,
      fetchWithHeaders,
      // The OWNER's job id, not the chain tip's: `(owner_job_id, step)` is
      // unique, so a rewind's two same-step checkpoints resolve to the right
      // one. Falls back to the tip for a single-run listing, which carries no
      // owner and where the step is unique anyway.
      policyConfigJobId,
      selectedStep,
    )
      .then((cfg) => {
        if (cancelled) return;
        setPolicyConfig(cfg);
      })
      .catch((e) => {
        if (cancelled) return;
        setPolicyConfig(null);
        setPolicyConfigError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setPolicyConfigLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // `policyConfigJobId` (staging's owner-addressed lookup), and no
    // `isBimanual`: the effect no longer re-seeds camera bindings, so the arm
    // layout is not an input to it any more.
  }, [open, baseUrl, fetchWithHeaders, policyConfigJobId, selectedStep]);

  // (No binding effects: the pairing is derived by name in `cameraBindings`
  // above, so there is no stored selection to seed, prune, or keep in step
  // with the robot record.)

  // ONE engine field, ONE rule: rtc is offered only for a checkpoint whose
  // architecture affirmatively supports it (`rtcSupported` above), on BOTH
  // paths.
  //
  // It used to be two rules. The field asked "will the server refuse this
  // request?", which is fail-OPEN — the server refuses `inference_engine:
  // "rtc"` with a 400 only for a policy type it KNOWS can't run guided chunk
  // prediction, and `supports_rtc: null` (a policy newer than its table) left
  // the option selectable for the subprocess to decide. The remote path asked
  // the stricter question and answered it with a warning plus a launch guard.
  // So the same unclassified checkpoint was a silent yes locally and a refusal
  // remotely, from one control.
  //
  // Fail-closed on both is the merge, because the fail-open half was never
  // buying anything: rtc on a checkpoint nobody has classified is a guess, and
  // the cost of guessing wrong is not an error message. Locally it degrades the
  // run against the contract the checkpoint was evaluated in; remotely it pairs
  // the arm with a GPU server the operator was never told to start. Sync is
  // correct for ANY policy, so the fallback costs nothing, and the one hint
  // under the picker says why the option is greyed out.
  //
  // Picking a checkpoint that can't run RTC also drops a stale "rtc" selection
  // back to sync. Runs on the config that just landed (the fetch above swaps
  // policyConfig in one setState), so the reset is a single render behind the
  // checkpoint change and the launch below can't carry "rtc" for it.
  useEffect(() => {
    if (!rtcSupported) setEngine("sync");
  }, [rtcSupported, setEngine]);

  // The transport probe, re-read on its own rather than by a Re-check button.
  //
  // The retired Transport section was a row-by-row read-out with a manual
  // refresh, which meant the one live fact in it (`operator_present`, the gate
  // on Start) was as stale as the last time somebody pressed it — and the
  // moment it changes is precisely the moment the GPU joins the room, which the
  // operator is watching the GPU card for, not that button. So: the hook probes
  // when the remote half opens, this re-probes the instant the launcher says
  // the container reached the room, and…
  const { refresh: refreshTransport } = remoteTransport;
  const gpuState = gpu.status?.state ?? null;
  const gpuPhase = gpu.status?.phase ?? null;
  useEffect(() => {
    if (!open || !remote) return;
    if (gpuState === "ready" || gpuPhase === "connected" || gpuPhase === "claimed")
      refreshTransport();
  }, [open, remote, gpuState, gpuPhase, refreshTransport]);

  // …a slow tick covers a GPU this Lab did not launch (the hand-typed command,
  // another machine). Never while a run is live — the probe is about whether one
  // COULD start — and never while the panel is shut.
  //
  // Either inference kind holds the robot; they are mutually exclusive
  // server-side, so every guard and every camera preview treats them alike.
  const runActive = remoteActive || status?.inference_active === true;
  useEffect(() => {
    if (!open || !remote || runActive) return;
    const id = setInterval(refreshTransport, TRANSPORT_REPROBE_MS);
    return () => clearInterval(id);
  }, [open, remote, runActive, refreshTransport]);

  // Poll inference status while visible so the guards reflect a live rollout.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const s = await getInferenceStatus(baseUrl, fetchWithHeaders);
        if (!cancelled) setStatus(s);
      } catch {
        // Transient; the next tick retries.
      }
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [open, baseUrl, fetchWithHeaders]);

  // Arm-count mismatch between CHECKPOINT and ROBOT — client mirror of the
  // server's `_arm_count_mismatch` 409 guard. (Ported verbatim.)
  //
  // Per-arm DOF is a property of the ROBOT, not a constant: an SO-101 arm is
  // 6-DOF and a CAN arm (Maker, Metal) 7 (six joints plus its permanent
  // gripper). Measured against 6, a 7-dim CAN checkpoint is not a clean
  // multiple, so checkpointArms would resolve to null and this guard would
  // silently go quiet on exactly the mismatch it exists to catch. Mirrors the
  // server's `_ARM_STATE_DIMS` in rollout.py — change both together.
  const armDof = jointsPerArm(robot?.arm_type);
  const checkpointDim =
    policyConfig?.state_dim ?? policyConfig?.action_dim ?? null;
  const checkpointArms =
    checkpointDim != null && checkpointDim % armDof === 0
      ? checkpointDim / armDof
      : null;
  const checkpointIsBimanual = checkpointArms != null && checkpointArms >= 2;
  const robotCheckpointArmMismatch =
    !!robot &&
    !!policyConfig &&
    checkpointArms != null &&
    checkpointIsBimanual !== isBimanual;

  // Every camera the policy needs is bound — by name or by a role pick — AND
  // plugged in. One fact for the whole panel again, now that the picks apply to
  // every mode: a role bound by hand is never LESS ready than a name match,
  // because picks only ever add bindings.
  const allCamerasReady =
    unmatchedCameras.length === 0 && disconnectedCameras.length === 0;

  // Temporal ensembling is an ACT config field — no other policy type has it,
  // and passing --policy.temporal_ensemble_coeff to one would fail the
  // rollout's config parse. Show the control only for ACT checkpoints.
  const isAct = policyConfig?.policy_type === "act";
  // Empty field or a non-positive number: the backend rejects it (weights are
  // exp(-coeff * i)), so block Start rather than round-trip a 400.
  // Not in remote mode: the control is hidden there (no local rollout to
  // configure) and the coeff is never sent, so blocking on it would be a dead
  // end — a refusal naming a field the operator cannot make appear.
  const temporalEnsembleInvalid =
    isAct &&
    runMode !== "remote" &&
    temporalEnsemble &&
    (temporalEnsembleCoeff === undefined || temporalEnsembleCoeff <= 0);

  // Inference drives the follower(s) only — gate on follower_ready, not
  // is_clean, so a robot with no leader port/calibration (which inference
  // never touches) can still deploy.
  // Coaching drives the leader as well — and drives it under torque during the
  // handover — so "follower is ready" is not enough. Without this the panel
  // showed green lights, Start was enabled, and the leader gap only surfaced
  // as a 400 from the server after the user had committed to a launch.
  const leaderMissing =
    !robot?.leader_port ||
    !robot?.leader_config ||
    (robot?.mode === "bimanual" &&
      (!robot?.right_leader_port || !robot?.right_leader_config));
  const coachLeaderMissing = coaching && leaderMissing;
  // Coaching writes the task string into every recorded frame, so the server
  // refuses an empty one for ANY policy — including one that doesn't condition
  // on language and therefore never showed the field. That combination gave a
  // green panel, an enabled Start, and a 400 naming a control the operator
  // could not make appear.
  const coachTaskMissing = task.trim() === "";

  // A max duration a LOCAL rollout can use. 0 is the remote run's unbounded
  // contract and a local run that stops the instant it starts, which is the one
  // place the shared field's two meanings collide — see deployGuards.
  const durationValid = durationS > UNBOUNDED_DURATION_S;

  // Everything a launch needs, whichever way the two axes are set.
  const canStartAnyMode =
    !!robot &&
    robot.follower_ready &&
    !robotCheckpointArmMismatch &&
    selectedRef != null &&
    !!policyConfig &&
    allCamerasReady &&
    !temporalEnsembleInvalid &&
    !submitting &&
    !checkingExtra &&
    !runActive;

  /** Why `mode` cannot be launched right now, or null when it can — as a
   * translation KEY, so the caller can tell WHICH refusal fired (the transport
   * one is answered with the probe's own sentence rather than this generic
   * line). Still per-mode: the derived mode is what the guards speak. */
  const blockedReasonKey = (mode: RunMode): string | null =>
    deployBlockedReason(mode, {
      hasRobot: !!robot,
      followerReady: !!robot?.follower_ready,
      hasCheckpoint: selectedRef != null && !!policyConfig,
      armMismatch: robotCheckpointArmMismatch,
      // "Bound" means matched-by-name-or-picked AND plugged in. There is no
      // other way to run a checkpoint whose cameras are named `cam0`/`cam1`,
      // and since S3.9 that escape hatch is open to every mode.
      allCamerasBound: allCamerasReady,
      temporalEnsembleInvalid,
      durationValid,
      inferenceActive: runActive,
      leaderMissing,
      // Remote inference: both flags are client mirrors of refusals the
      // backend makes anyway — this only moves them to before the launch.
      transportReady: transportIsReady(remoteTransport.transport),
      armSupportsRemote: armSupportsRemoteInference(robot),
      // This one has NO backend twin and cannot have: the server never loads
      // the checkpoint, so it cannot tell a flow policy from an ACT one and
      // accepts whichever engine it is handed. Sync suits any policy, so only
      // the rtc choice can be wrong here.
      //
      // Belt and braces since the engine rule went fail-closed: the picker
      // can no longer OFFER rtc for a checkpoint this reads as unsupported, so
      // the guard fires only in the one render between a new checkpoint landing
      // and the effect above snapping the engine back to sync. It is derived
      // from the same `rtcSupported` fact precisely so the two cannot drift —
      // and it is the last thing standing between a stale selection and a
      // launched run, which is not a job to leave to render ordering.
      remoteEngineSupported: engine !== "rtc" || rtcSupported,
      requiresTask: !!policyConfig?.requires_task,
      // The effective value: an empty box that falls back to a real default is
      // not a missing task, and blocking on it would be a dead end.
      task: effectiveTask,
    });

  // Prefill the corrections dataset name from the model being coached, so the
  // pair reads as one thing in the library: `correction_<model>`. Naming it by
  // hand produced datasets nobody could match back to a policy a week later.
  //
  // `correction_` leads so the kind of dataset is the first thing read, but it
  // cannot be the WHOLE name: lerobot refuses a rollout dataset whose repo name
  // does not start with `rollout_` (lerobot/rollout/context.py), and merge.py's
  // `_looks_like_our_coaching_dataset` keys the lossless `intervention`-column
  // drop on that same prefix — so dropping it would break merging corrections
  // back into the demos they were collected against. The final name is
  // `rollout_correction_<model>_<timestamp>`.
  //
  // Held as a DEFAULT, never written into the field.
  //
  // Prefilling meant the operator could not tell a name the app had guessed
  // from one they had chosen, and clearing the box left them staring at an
  // empty field with no hint of what would happen. As a placeholder the default
  // is visibly not-their-input, survives clearing the box, and needs no ref to
  // track whether it is still "the auto value".
  //
  // Named after the DATASET the checkpoint was trained on, falling back to the
  // model. That is the thing the corrections will be merged back into, so it is
  // the name that makes the pair findable a week later.
  const defaultCoachName = useMemo(() => {
    if (!selectedJob) return "";
    const trainedOn = selectedJob.config?.dataset_repo_id;
    const base =
      trainedOn && trainedOn !== "(imported)"
        ? trainedOn.split("/").pop()
        : jobDisplayName(selectedJob);
    // Same character class the backend accepts in a repo id; collapse anything
    // else so a name with spaces or slashes cannot produce a bad path.
    const slug = (base ?? "")
      .trim()
      .replace(/[^a-zA-Z0-9._-]+/g, "_")
      .replace(/^_+|_+$/g, "");
    return slug ? `correction_${slug}` : "";
  }, [selectedJob]);

  // What actually gets sent: whatever they typed, else the default.
  const effectiveCoachName = coachDatasetName.trim() || defaultCoachName;

  // The task the checkpoint was trained on, most-represented first. Same
  // placeholder contract as the name above: shown greyed, sent when the field
  // is left empty, and restored the moment the operator clears what they typed.
  const defaultTask = datasetTasks[0] ?? "";
  const effectiveTask = task.trim() || defaultTask;

  // Prefill the task from the dataset the selected checkpoint was trained on.
  // Typing it by hand means retyping a sentence that already exists, and a
  // typo'd task is invisible until the policy underperforms.
  useEffect(() => {
    const repoId = selectedJob?.config?.dataset_repo_id;
    if (!repoId || repoId === "(imported)") {
      setDatasetTasks([]);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const info = await getDatasetInfo(baseUrl, fetchWithHeaders, repoId);
        if (cancelled) return;
        // Most-represented task first: if the operator has to choose, the one
        // the policy saw most is the likeliest answer and should be nearest.
        const tasks = [...(info.tasks ?? [])]
          .sort((a, b) => b.num_episodes - a.num_episodes)
          .map((t) => t.task)
          .filter(Boolean);
        // Offered as a PLACEHOLDER default (see `defaultTask`), not written
        // into the field: the operator should be able to tell the app's guess
        // from their own sentence, and clearing the box should fall back to the
        // guess rather than to nothing.
        setDatasetTasks(tasks);
      } catch {
        // A Hub-only or missing dataset simply has no tasks to offer; the
        // field stays as the user left it.
        if (!cancelled) setDatasetTasks([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedJob, baseUrl, fetchWithHeaders]);

  // A prefill may name the run mode — that is how "Policy failing? Coach it"
  // lands the user in coaching without them having to know the control exists.
  // It speaks the one-word vocabulary, so it is unpacked onto the two axes
  // here. Every mode a prefill can name is a LOCAL one (DeployPrefill.mode is
  // single | eval | coach — nothing hands the operator into a remote run), so
  // it always lands on This machine, exactly as `setRunMode(mode)` did.
  useEffect(() => {
    const mode = deployPrefill?.mode;
    if (!mode) return;
    setRunsOn("local");
    setOperatorMode(mode === "coach" ? "coach" : "single");
    setScoring(mode === "eval");
  }, [deployPrefill]);

  const handleStart = async (mode: RunMode = runMode) => {
    if (
      !robot ||
      robotCheckpointArmMismatch ||
      selectedRef == null ||
      !policyConfig
    )
      return;

    // Pre-flight: pi0/pi0_fast/pi05/smolvla/diffusion need an optional
    // package installed locally. The rollout subprocess runs against this
    // machine's own environment (it drives the physically-connected robot),
    // so catch a missing extra here with a one-click installer instead of a
    // buried ImportError after the rollout has already claimed the cameras.
    if (policyConfig.policy_type) {
      setCheckingExtra(true);
      try {
        const r = await fetchWithHeaders(
          `${baseUrl}/api/v1/system/policy-extra/${policyConfig.policy_type}`,
        );
        if (r.ok) {
          const extra = await r.json();
          if (extra.needs_extra && !extra.available) {
            setPolicyExtra({
              policyType: policyConfig.policy_type,
              packageName: extra.package,
              installTarget: extra.install_target,
              installHint: extra.install_hint,
            });
            return;
          }
        }
      } catch {
        // Check failed (offline / older backend) — fall through and let the
        // rollout report any problem itself.
      } finally {
        setCheckingExtra(false);
      }
    }

    // Drops every camera card's preview stream so the rollout subprocess can
    // open the same camera index via OpenCV without colliding on the device.
    setSubmitting(true);
    await new Promise((r) => setTimeout(r, 300));
    // Emit binding keys under the DISPLAY/request name (bare in bimanual
    // mode). The rollout hands the resolved cameras to the BiSO
    // left_arm_config, which re-prefixes with `left_` — reconstructing the
    // checkpoint's `left_<name>` feature. The VALUES are robot-record camera
    // names; every setting behind them is read server-side from the record.
    const cameraBindingPayload: Record<string, string> = {};
    // ...except capture RESOLUTION, which comes from the checkpoint: lerobot's
    // rollout doesn't resize frames to the policy's input shape, so the camera
    // must capture at the size the policy was trained on. Forwarded the same
    // way `checkpoint_state_dim` is — the client already read these dims off
    // /policy-config, the server applies them.
    const cameraDimsPayload: Record<string, { width: number; height: number }> =
      {};
    // The bindings, name matches and role picks alike — one list for every
    // mode since S3.9. A role pick changes nothing else about the request: the
    // VALUES are still robot-record camera names the server resolves to devices
    // itself, and `camera_dims` still comes from the CHECKPOINT's own
    // image_features, the same for a picked role as for a name-matched one.
    for (const { mapping, camera, dims } of boundCameraBindings) {
      if (camera == null) continue;
      // The robot record's own spelling, not the checkpoint's — the name
      // matched case-insensitively and the server looks it up verbatim.
      cameraBindingPayload[mapping.requestKey] = camera.name;
      if (dims?.width && dims?.height) {
        cameraDimsPayload[mapping.requestKey] = {
          width: dims.width,
          height: dims.height,
        };
      }
    }
    // Remote inference is its own session KIND, with its own options model and
    // its own status surface — so it forks here rather than adding a fourth
    // conditional to the inference options below. Same robot, same checkpoint,
    // same camera derivation; only the policy is somewhere else.
    if (mode === "remote") {
      try {
        const { session } = await startSession(baseUrl, fetchWithHeaders, {
          kind: "remote_inference",
          robot: robot.name,
          owner: tabOwnerId(),
          options: {
            policy_ref: selectedRef,
            // Advisory to the backend in this slice — it is the GPU side's
            // --policy-path, and keeping it on the request is what lets the
            // panel generate that command from this same object.
            policy_hub_id:
              remoteConfig.policyHubId.trim() || selectedJob?.hf_repo_id || "",
            task: effectiveTask,
            camera_bindings: cameraBindingPayload,
            camera_dims: cameraDimsPayload,
            checkpoint_state_dim: policyConfig.state_dim ?? undefined,
            // The shared field. 0 is this kind's own unbounded contract.
            duration_s: durationS,
            // The transport triple. It MUST match the `modal run` line above
            // it: Portal fingerprints the wire schema and silently drops
            // mismatched packets, so a disagreement here is a healthy-looking
            // session that never receives a chunk.
            horizon: remoteConfig.horizon,
            fps: remoteConfig.fps,
            video_codec: remoteConfig.videoCodec,
            // The engine picks which chunk player the server spawns; s_min is
            // half a contract with the GPU side and is only read for rtc.
            engine: remoteConfig.engine,
            s_min: remoteConfig.sMin,
          },
        });
        // The run now surfaces in the SAME session dialog a local run opens —
        // pill, timer, policy line, one full-width Stop — with the GPU's
        // telemetry where a coaching run shows its tally. Before S3.9 it stayed
        // inline on this panel, which forced the form open for the whole run
        // just to keep a Stop on screen.
        openInferenceSession(session.id, null, "remote_inference");
        if (!hasSeenDeployMilestone) {
          setDeployMilestonePending(true);
          markDeployMilestoneSeen();
        }
      } catch (e) {
        toast({
          title: t("remoteInference.toast.startFailed"),
          description:
            formatSessionHeld(t, e) ??
            (e instanceof Error ? e.message : String(e)),
          variant: "destructive",
        });
      } finally {
        setSubmitting(false);
      }
      return;
    }

    try {
      // Robot NAME + policy-shaped options only — ports, configs, mode and
      // the camera devices behind the bindings resolve server-side from the
      // saved record. The owner attaches the lease the session dialog renews.
      // Coaching's LEADER arms resolve there too, off the same record: no
      // separate picker, because the record already pairs leader with follower
      // and re-asking would only be a chance to get it wrong.
      const { session } = await startSession(baseUrl, fetchWithHeaders, {
        kind: "inference",
        robot: robot.name,
        owner: tabOwnerId(),
        options: {
          policy_ref: selectedRef,
          task: effectiveTask,
          camera_bindings: cameraBindingPayload,
          camera_dims: cameraDimsPayload,
          duration_s: durationS,
          checkpoint_state_dim: policyConfig.state_dim ?? undefined,
          // Only ever >1 in eval mode — the count field is hidden otherwise,
          // but pinning it here means a stale value left over from switching
          // modes can't quietly turn a single run into a 20-episode evaluation.
          eval_episodes: mode === "eval" ? evalEpisodes : 1,
          // Coaching is pinned to sync server-side too (RTC snaps the arm back
          // toward its pre-correction pose on hand-back); sending it correctly
          // from here keeps the request honest rather than relying on that.
          inference_engine: mode === "coach" ? "sync" : engine,
          ...(mode === "coach"
            ? {
                coaching: true,
                target_corrections: targetCorrections,
                coaching_dataset_name: effectiveCoachName,
              }
            : {}),
          // ACT-only, and only while the switch is on — otherwise omitted so
          // the checkpoint's own (ensembling-off) config stands.
          temporal_ensemble_coeff:
            isAct && temporalEnsemble ? temporalEnsembleCoeff : undefined,
        },
      });
      // The run surfaces as the InferenceSessionDialog over this panel —
      // closing it lands back here (the studio stays open underneath).
      // Coaching only: hand the session what it needs to offer the merge +
      // fine-tune when it ends. A plain run or an eval produces no dataset and
      // has nothing to hand on. See CoachingLineage.
      openInferenceSession(
        session.id,
        mode === "coach" && jobId
          ? {
              jobId,
              jobName: selectedJob?.name ?? undefined,
              trainingDatasetRepoId:
                selectedJob?.config?.dataset_repo_id &&
                selectedJob.config.dataset_repo_id !== "(imported)"
                  ? selectedJob.config.dataset_repo_id
                  : undefined,
            }
          : null,
      );
      if (!hasSeenDeployMilestone) {
        setDeployMilestonePending(true);
        markDeployMilestoneSeen();
      }
      // The POST claims the inference slot synchronously, so a status fetch
      // issued now reflects THIS run — hand the released-previews / disabled-
      // Start duty from `submitting` to `inferenceActive` (kept fresh by the
      // poll). Unlike the modal this panel never unmounts, so `submitting`
      // must be cleared here or Start stays stuck on "Starting…" forever.
      try {
        setStatus(await getInferenceStatus(baseUrl, fetchWithHeaders));
      } catch {
        // The 2s poll catches up on its next tick.
      }
      setSubmitting(false);
    } catch (e) {
      toast({
        title: t("studio.deploy.toast.startFailed"),
        // 409 session.held renders as the shared localized "robot is busy"
        // line; everything else is the server's raw error text.
        description:
          formatSessionHeld(t, e) ??
          (e instanceof Error ? e.message : String(e)),
        variant: "destructive",
      });
      // Failure: bring the previews back so the user can adjust.
      setSubmitting(false);
    }
  };

  // No `onCameraBindingChange`: the pairing is derived by name in
  // `cameraBindings` above, so there is no stored selection left to write.
  //
  // No `handleStop` either, and that one is a decision worth recording,
  // because the two branches this file was merged from argued it opposite ways.
  //
  //   · The rework's note (b21bb1f7's Start ⇄ "View running session" toggle
  //     could not be landed — `openInferenceSession` needs the session id the
  //     POST returned, and a rollout this panel did not start has none) went on
  //     to argue the panel should therefore KEEP its own stop, as the only way
  //     to wind a run down after the session dialog has been closed.
  //   · The coaching work removed it: a live rollout owns the modal
  //     InferenceSessionDialog, which carries its own Stop, so a second stop
  //     control on a panel the operator cannot see during a run is dead weight
  //     the rest of the time and enabled only in the one state where it is
  //     unreachable.
  //
  // The second is the later reading and the one the Start button is built
  // around (it is the whole action surface now), so it stands. Since S3.9 it
  // covers the remote run as well: that run opens the same dialog, which
  // carries the same Stop, so the panel no longer has to force its own form
  // open to keep one on screen.
  const activeTabDef =
    OPERATOR_TABS.find((m) => m.value === effectiveOperator) ?? OPERATOR_TABS[0];
  // A run in flight (either kind, or one being started) freezes the controls:
  // switching mid-run would only rewrite a form the run no longer reads.
  const controlsLocked = runActive || submitting;
  const startBlockedKey = blockedReasonKey(runMode);

  // The transport read-out as ONE sentence, chosen by the first thing that is
  // wrong. It REPLACES the generic "the transport isn't ready" line, because
  // that line named a section that no longer exists and never said which of the
  // five things between here and a running GPU was the one to fix.
  const transportSummary = summarizeTransport(
    remoteTransport.transport,
    remoteTransport.loading,
    remoteTransport.error,
  );
  const transportTone = {
    ok: "text-ok",
    warn: "text-warn",
    error: "text-destructive",
    muted: "text-muted-foreground",
  }[transportSummary.tone];
  // The probe's own coded failure, when it has one. Backend data, second line,
  // verbatim — the summary says what to do, this says what the server said.
  const transportDetail =
    remote && remoteTransport.transport?.error_code
      ? `${remoteTransport.transport.error_code}${
          remoteTransport.transport.message
            ? ` — ${remoteTransport.transport.message}`
            : ""
        }`
      : null;

  const selectedSkillLabel = selectedJob ? jobDisplayName(selectedJob) : null;
  // The Hub repo the GPU container would load if the operator names none.
  const hubIdDefault = selectedJob?.hf_repo_id ?? "";

  return (
    <div className="flex flex-1 flex-col gap-5 p-5">
      <PanelHeader
        step="3"
        title={t("studio.deploy.title")}
        dataTour="studio-deploy"
      >
        {resolving ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
        ) : null}
      </PanelHeader>

      {/* Run a skill — the panel's entry control, the same opener Collect
          ("Record new dataset") and Train ("Start a new training") wear: it
          slides the run's form open in place and folds the skills library to
          its header while it is open. */}
      <Collapsible
        // No forcing open any more. A live remote run used to keep its
        // telemetry and its Stop INLINE here, so collapsing the form would have
        // left an energized arm with no Stop on screen; since S3.9 that run
        // opens the session dialog like every other, and the form is free to
        // fold exactly when the operator says so.
        open={formOpen}
        onOpenChange={toggleForm}
        className="space-y-5"
      >
        <CollapsibleTrigger asChild>
          <PanelEntryControl open={formOpen} dotClassName="bg-sky-500">
            {t("studio.deploy.entry")}
          </PanelEntryControl>
        </CollapsibleTrigger>
        <CollapsibleContent className={SLIDE}>
          <div className="space-y-6">
            {/* The form's one-line brief, in the slot and voice Train uses
                ("Choose what to train on…"): under the opener, above the first
                field label. It also covers what the Skill field's own helper
                used to say, so that line is gone. */}
            <p className="text-sm leading-relaxed text-muted-foreground">
              {t("studio.deploy.intro")}
            </p>

            {/* Robot readiness — a warning, not a parameter, so no eyebrow. A
                ready robot renders nothing: the robot menu already names the
                selection and its arm layout. */}
            <RobotStatus ready={!!robot && robot.follower_ready}>
              {!robot ? (
                t("studio.deploy.noRobot")
              ) : (
                /* The plural is on the follower ARM count — 2 for a bimanual
                   robot, 1 otherwise. The number itself is never printed; it
                   only picks the variant, replacing the old `{s}` splice.
                   Coaching gets its own key AND its own gap scope: it
                   teleoperates through the leader too, so a missing leader port
                   is a real blocker there and noise in every other mode. */
                <Trans
                  i18nKey={
                    coaching
                      ? "studio.deploy.robotNotReadyCoach"
                      : "studio.deploy.robotNotReady"
                  }
                  count={isBimanual ? 2 : 1}
                  values={{
                    name: robot.name,
                    gap: formatRobotSetupGap(
                      t,
                      robot,
                      coaching ? "all" : "follower",
                    ),
                  }}
                  components={[<strong key="0" />]}
                />
              )}
            </RobotStatus>

            {/* Skill and Checkpoint share one row — the same two-column
                grid Collect pairs Episode duration / Reset duration with.
                They are one decision in two halves (which weights, which
                step of them), and the second is meaningless without the
                first, so the row is always two-up: with no skill picked the
                Checkpoint column holds a disabled "Pick a skill first"
                instead of collapsing. min-w-0 on both columns so a long
                name truncates inside its half instead of widening it. */}
            <div className="grid grid-cols-2 gap-4">
              {/* Skill — the form's one mandatory field, built as Train's
                  dataset field is: the current choice as a chip, otherwise the
                  full-list picker docked in the same box.

                  One way in, not two: the whole control is the popover's
                  trigger, and ModelPicker's own CommandInput is the only
                  search box (it also owns the loading / "no models yet" /
                  "no match" states). The trigger wears SelectTrigger's own
                  classes so it reads as the same kind of control as the
                  Checkpoint dropdown beside it and the engine Select below —
                  it can't BE a SelectTrigger, because what it opens is a
                  Popover. Picking replaces the selection outright; there is
                  no clear ✕, since a run always needs a skill and "none" is
                  only ever the pre-selection state. */}
              <div className="min-w-0 space-y-2">
                <Label htmlFor="deploy-skill">
                  {t("studio.deploy.policy.label")}
                </Label>
                {/* `models` is already the DEPLOYABLE projection of /skills —
                    the server decides it (weights loadable AND nothing
                    supersedes the row), which is a stricter and better-informed
                    answer than the checkpoint_count deny-list this panel used
                    to derive from the job registry. */}
                <ModelPicker
                  models={models}
                  loading={modelsLoading}
                  onPickExisting={(m) => handlePickSkill(m.id)}
                >
                  <button
                    id="deploy-skill"
                    type="button"
                    disabled={resolving}
                    className="flex h-10 w-full items-center justify-between gap-2 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {selectedSkillLabel ? (
                      <DisplayName
                        name={selectedSkillLabel}
                        className="min-w-0"
                      />
                    ) : (
                      <span className="truncate text-muted-foreground">
                        {t("studio.deploy.picker.placeholder")}
                      </span>
                    )}
                    <ChevronsUpDown className="h-4 w-4 shrink-0 opacity-50" />
                  </button>
                </ModelPicker>
                {/* Listing health, kept from the coaching branch and moved OUT
                    of the dropdown now that the picker owns its own empty
                    state. "We could not ask" and "you own nothing" are
                    different sentences, and an unreachable Hub used to look
                    exactly like an empty shelf — the rows simply were not
                    there. The picker still serves the last complete Hub result
                    underneath. */}
                {modelsError ? (
                  <p className="text-xs text-destructive">
                    {t("studio.deploy.picker.error")}
                  </p>
                ) : hubStatus && !hubStatus.ok ? (
                  <p className="text-[11px] text-amber-600 dark:text-amber-500">
                    {t("studio.deploy.picker.hubDegraded")}
                  </p>
                ) : null}
              </div>
              {/* Checkpoint — the one control with nothing to offer until a skill is
                  chosen, so it renders disabled saying so rather than vanishing.

                  CheckpointDropdown's own trigger is sized for a card's action row
                  (h-8, text-xs, w-auto); FIELD_TRIGGER puts it back on the studio's
                  form-field size. Passed from here rather than changed in the
                  component, because its other callers (JobCard's action line,
                  ModelCard's w-36 slot) want the compact one. */}
              <div className="min-w-0 space-y-2">
                <Label htmlFor="deploy-checkpoint">
                  {t("studio.deploy.checkpoint.label")}
                </Label>
                {!selectedJob ? (
                  <CheckpointDropdown
                    id="deploy-checkpoint"
                    checkpoints={NO_CHECKPOINTS}
                    selectedRef={null}
                    onChange={() => {}}
                    disabled
                    placeholder={t("studio.deploy.checkpoint.pickPolicyFirst")}
                    className={FIELD_TRIGGER}
                  />
                ) : checkpoints.length === 0 ? (
                  <Alert className="border-warn/40 text-warn [&>svg]:text-warn">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      {t("studio.deploy.checkpoint.none")}
                    </AlertDescription>
                  </Alert>
                ) : (
                  <CheckpointDropdown
                    id="deploy-checkpoint"
                    checkpoints={checkpoints}
                    // Lineage-wide list: two entries can share a step, so the
                    // ref is the only safe identity to select by.
                    selectedRef={selectedRef}
                    onChange={(c) => setSelectedRef(c.ref)}
                    owners={checkpointOwnerMap}
                    className={FIELD_TRIGGER}
                  />
                )}
              </div>
            </div>

            {/* Reading the checkpoint's config is what tells this panel the
                policy's cameras, its task requirement and its arm count, so the
                progress and the failure both belong under the checkpoint rather
                than beside the fields they would have filled in. */}
            {policyConfigLoading ? (
              <p className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" />
                {t("studio.deploy.cameras.loading")}
              </p>
            ) : null}
            {policyConfigError ? (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  {/* The error text is the backend's own — passed through. */}
                  {t("studio.deploy.cameras.configError", {
                    error: policyConfigError,
                  })}
                </AlertDescription>
              </Alert>
            ) : null}
            {robotCheckpointArmMismatch ? (
              <Alert className="border-warn/40 text-warn [&>svg]:text-warn">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  {/* Each branch is one complete key rather than shared
                      fragments, so a translator owns the whole sentence. */}
                  <Trans
                    i18nKey={
                      checkpointIsBimanual
                        ? "studio.deploy.armMismatch.bimanualCheckpoint"
                        : "studio.deploy.armMismatch.singleCheckpoint"
                    }
                    values={{
                      dim: checkpointDim,
                      arms: checkpointArms,
                      name: robot?.name,
                    }}
                    components={[<strong key="0" />, <strong key="1" />]}
                  />
                </AlertDescription>
              </Alert>
            ) : null}

            {/* Hub policy id — REMOTE only, and directly under the checkpoint
                row rather than down with the GPU card, because it is the other
                half of the same question: which weights run. The row above says
                which checkpoint this machine measures the run against; this
                says which repo the container actually loads. --------------- */}
            {remote ? (
              <div className="space-y-2">
                <Label htmlFor="remote-hub-id">
                  {t("remoteInference.form.hubIdLabel")}
                </Label>
                <Input
                  id="remote-hub-id"
                  value={remoteConfig.policyHubId}
                  disabled={controlsLocked}
                  onChange={(e) =>
                    setRemoteConfig((prev) => ({
                      ...prev,
                      policyHubId: e.target.value,
                    }))
                  }
                  // A repo id shape, not prose — the literal the operator must
                  // match. Offered as a PLACEHOLDER default the same way the
                  // task and coaching-dataset fields offer theirs: visibly not
                  // the operator's input, and restored the moment they clear
                  // the box.
                  placeholder={hubIdDefault || POLICY_PATH_PLACEHOLDER}
                  className="font-mono"
                />
                <p className="text-xs text-muted-foreground">
                  {t("remoteInference.form.hubIdHint")}
                  {remoteConfig.policyHubId.trim() === "" && hubIdDefault
                    ? ` ${t("remoteInference.form.hubIdInherited")}`
                    : ""}
                </p>
              </div>
            ) : null}

            {/* Run parameters — flat, each with its own <Label>; the old "Run
                parameters" eyebrow sat above two fields that already say what
                they are. The block is no longer gated on the policy config
                having loaded — that gate made half the form appear and vanish
                with the skill. What IS gated is which fields a given run mode
                actually uses. ---------------------------------------------- */}
            <div className="space-y-2">
              <Label htmlFor="deploy-task">
                {t("studio.deploy.task.label")}
              </Label>
              <Input
                id="deploy-task"
                value={task}
                onChange={(e) => setTask(e.target.value)}
                // The trained-on sentence, shown greyed rather than typed in.
                // Leaving the box empty sends it; clearing what you typed brings
                // it back. No invented example: a fake task shown greyed in the
                // same slot the REAL inherited task uses is indistinguishable
                // from one. When the lineage yields nothing, say so instead.
                placeholder={
                  defaultTask || t("studio.deploy.task.placeholderNone")
                }
              />
              {/* Whether the field is even read is a property of the checkpoint,
                  so the helper answers that question in all three states rather
                  than the field appearing and disappearing with the skill — plus
                  a fourth: coaching writes the string into every recorded frame,
                  so it is read even by a policy that ignores it.
                  The policy type is an identifier — rendered verbatim. */}
              <p className="text-xs text-muted-foreground">
                {!policyConfig
                  ? t("studio.deploy.task.hintUnknown")
                  : policyConfig.requires_task
                    ? t("studio.deploy.task.hint", {
                        policyType: policyConfig.policy_type ?? "",
                      })
                    : coaching
                      ? t("studio.deploy.task.hintCoach")
                      : t("studio.deploy.task.hintNotConditioned", {
                          policyType: policyConfig.policy_type ?? "",
                        })}
                {/* Says where the greyed sentence comes from. Only while the box
                    is EMPTY — once the operator types, the default is not what
                    will be sent and claiming otherwise would be a lie. */}
                {task.trim() === "" && defaultTask
                  ? ` ${t("studio.deploy.task.leaveEmpty")}`
                  : ""}
              </p>
              {datasetTasks.length > 1 && (
                <div className="space-y-1">
                  <p className="text-xs text-muted-foreground">
                    {t("studio.deploy.task.multiTaskHint", {
                      count: datasetTasks.length,
                    })}
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {datasetTasks.map((candidate) => (
                      <button
                        key={candidate}
                        type="button"
                        onClick={() => setTask(candidate)}
                        className={cn(
                          "rounded border px-2 py-0.5 text-xs transition-colors",
                          task === candidate
                            ? "border-primary bg-primary/10"
                            : "border-border hover:bg-muted",
                        )}
                      >
                        {candidate}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
            {/* Camera roles — for EVERY combination, and only when there is a
                decision to make. A checkpoint's camera name is a role, not a
                claim about this robot (`lerobot/MolmoAct2-SO100_101-LeRobot`
                names cam0/cam1, which no robot record has ever been called),
                and a role nothing matches is a question only the operator can
                answer. Renders nothing at all when every role matched by name,
                which is the ordinary case. --------------------------------- */}
            <CameraRoleBindings
              slots={cameraRoleSlots}
              cameras={cameraRoleOptions}
              nameMatchedCount={cameraBindings.length - cameraRoleSlots.length}
              onChange={setRemoteCameraRole}
              disabled={controlsLocked}
            />

            {/* Inference engine — ONE field, not two. It used to be asked twice
                in two vocabularies ("Inference engine: Sync / RTC" for the
                local rollout, "Chunk engine: Adaptive sync / Real-time
                chunking" for the remote one), which read as two settings and
                was one: which chunk player drives the arm. -------------- */}
            <div className="space-y-2">
              <Label htmlFor="deploy-engine">
                {t("studio.deploy.engine.label")}
              </Label>
              <Select
                // Coaching PINS sync (the server does too) without writing to
                // state, so leaving the coach tab restores what was chosen.
                value={coaching ? "sync" : engine}
                disabled={coaching || controlsLocked}
                onValueChange={(v) => setEngine(v as RemoteEngine)}
              >
                <SelectTrigger id="deploy-engine">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {/* Option VALUES ("sync" / "rtc") are what both the backend
                      and the GPU side parse — only the labels are translated. */}
                  <SelectItem value="sync">
                    {t("studio.deploy.engine.sync")}
                  </SelectItem>
                  {/* Disabled rather than hidden: a checkpoint that can't be
                      in-painted should SAY so, not silently offer one engine.
                      One rule on both paths, fail-closed — an unclassified
                      checkpoint is greyed out too. See the note above. */}
                  <SelectItem value="rtc" disabled={!rtcSupported}>
                    {t("studio.deploy.engine.rtc")}
                  </SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs leading-relaxed text-muted-foreground">
                {coaching
                  ? t("studio.deploy.engine.coachingNote")
                  : engine === "rtc"
                    ? t("remoteInference.form.engine.rtcHint")
                    : t("remoteInference.form.engine.syncHint")}
              </p>
              {/* ONE hint for the disabled case, on both paths — the greying
                  out is the refusal, this is why. The remote-only warning that
                  used to sit under it is gone with the fail-open rule that
                  made it reachable: rtc can no longer be SELECTED for a
                  checkpoint that cannot be in-painted, so there is nothing left
                  to warn about after the fact. */}
              {!coaching && !rtcSupported ? (
                <p className="text-xs text-muted-foreground">
                  {t("studio.deploy.engine.rtcUnavailable")}
                </p>
              ) : null}
            </div>

            {/* Max duration — ONE field, whose 0 means two different things.
                Unbounded for a remote run (the backend's own contract), and
                impossible for a local one, which is a blocked reason rather
                than a min= on the input: the same control serves both. ---- */}
            <div className="space-y-2">
              <Label htmlFor="deploy-duration">
                {t("studio.deploy.duration.label")}
              </Label>
              <NumberInput
                id="deploy-duration"
                min={remote ? 0 : 1}
                value={durationS}
                disabled={controlsLocked}
                onChange={(v) => {
                  if (v !== undefined) setDurationS(v);
                }}
              />
              <p className="text-xs text-muted-foreground">
                {coaching
                  ? t("studio.deploy.duration.coachHint")
                  : remote
                    ? durationS === UNBOUNDED_DURATION_S
                      ? t("studio.deploy.duration.remoteUnbounded")
                      : t("studio.deploy.duration.remoteHint")
                    : scoring
                      ? t("studio.deploy.duration.hint")
                      : t("studio.deploy.duration.singleHint")}
              </p>
            </div>

            {/* Runs on — WHERE the policy runs. Directly above the operator
                strip because the two are read together, and a segmented
                control rather than a second tab strip in spirit: it selects a
                value, it does not reveal a pane of its own. (It wears the same
                clothes, which is what makes the pair read as one question in
                two halves.) -------------------------------------------- */}
            <div className="space-y-2">
              <Label id="deploy-runs-on-label">
                {t("studio.deploy.runsOn.label")}
              </Label>
              <div
                role="radiogroup"
                aria-labelledby="deploy-runs-on-label"
                className="grid grid-cols-2 gap-1 rounded-md bg-muted p-1 text-muted-foreground"
              >
                {(["local", "remote"] as RunsOn[]).map((where) => (
                  <button
                    key={where}
                    type="button"
                    role="radio"
                    aria-checked={runsOn === where}
                    disabled={controlsLocked && runsOn !== where}
                    onClick={() => setRunsOn(where)}
                    className={cn(
                      "inline-flex items-center justify-center rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50",
                      runsOn === where && "bg-background text-foreground shadow-sm",
                    )}
                  >
                    {t(`studio.deploy.runsOn.${where}` as never)}
                  </button>
                ))}
              </div>
              <p className="text-xs leading-relaxed text-muted-foreground">
                {remote
                  ? t("studio.deploy.runsOn.remoteHint")
                  : t("studio.deploy.runsOn.localHint")}
              </p>
            </div>

            {/* Everything that only exists because the policy is somewhere
                else: the GPU the Lab launches, the command for launching it by
                hand, and the wire parameters the two sides must agree on. --- */}
            {remote ? (
              <>
                <GpuLaunchSection
                  launcher={gpu}
                  targets={gpuTargets}
                  knobs={gpuKnobs}
                  config={remoteConfig}
                  hubIdDefault={hubIdDefault}
                  // The SAME string the start request sends, so the GPU side
                  // and the robot side steer the policy identically.
                  task={effectiveTask}
                  // The GPU launch has no server-side twin of deployGuards'
                  // task check (the launcher knows a Hub id, not a policy
                  // type), so the panel gates Start GPU on the same fact.
                  taskRequired={!!policyConfig?.requires_task}
                />
                <RemoteManualSection
                  config={remoteConfig}
                  transport={remoteTransport.transport}
                  hubIdDefault={hubIdDefault}
                  task={effectiveTask}
                  // So the pasted line bills the same workspace Start GPU would.
                  profile={gpuTargets.profile}
                  environment={gpuTargets.environment}
                  knobs={gpuKnobs}
                />
                <RemoteAdvancedSection
                  config={remoteConfig}
                  onChange={setRemoteConfig}
                  knobs={gpuKnobs}
                  checkpointHorizon={checkpointHorizon}
                  open={transportAdvancedOpen}
                  onOpenChange={setTransportAdvancedOpen}
                  disabled={controlsLocked}
                />
              </>
            ) : null}

            {/* What the OPERATOR does. Two tabs, not three: where the policy
                runs is the control above, and folding it in here made every
                field that belongs to both questions live inside one tab and
                vanish from the others. ---------------------------------- */}
            <Tabs
              value={effectiveOperator}
              onValueChange={(v) => setOperatorMode(v as OperatorMode)}
              className="space-y-4"
            >
              <TabsList
                aria-label={t("studio.deploy.tabs.groupLabel")}
                className="grid h-auto w-full grid-cols-2"
              >
                {OPERATOR_TABS.map((m) => (
                  <TabsTrigger
                    key={m.value}
                    value={m.value}
                    // Human in the loop needs the policy on this machine: the
                    // remote child has no takeover protocol, so there is
                    // nothing to hand the arm over to. Disabled, not hidden —
                    // the reason is under the strip.
                    disabled={
                      (m.value === "coach" && remote) ||
                      (controlsLocked && m.value !== effectiveOperator)
                    }
                    title={t(`studio.deploy.runMode.${m.stem}.what` as never)}
                    className="min-w-0 px-2 py-1.5 leading-tight whitespace-normal"
                  >
                    {t(`studio.deploy.runMode.${m.stem}.title` as never)}
                  </TabsTrigger>
                ))}
              </TabsList>
              {/* What the selected tab does and what it asks of the operator,
                  read before Start rather than discovered at the arm. The
                  hands-on commitment is weighted as well as coloured: weight
                  survives greyscale and peripheral vision, colour alone does
                  not. */}
              <p className="text-xs leading-relaxed text-muted-foreground">
                {t(`studio.deploy.runMode.${activeTabDef.stem}.what` as never)}{" "}
                <span
                  className={
                    activeTabDef.handsOn ? "font-semibold text-warn" : undefined
                  }
                >
                  {t(
                    `studio.deploy.runMode.${activeTabDef.stem}.commitment` as never,
                  )}
                </span>
              </p>
              {remote ? (
                <p className="text-xs leading-relaxed text-warn">
                  {t("studio.deploy.tabs.coachNeedsLocal")}
                </p>
              ) : null}

              {/* Run — hands off. The scored-evaluation count is the only
                  thing this tab adds, and only when a prefill asked for one:
                  everything else a plain run needs is shared above. ------ */}
              <TabsContent value="single" className="mt-0 space-y-6">
                {scoring ? (
                  <div className="space-y-2">
                    <Label htmlFor="deploy-episodes">
                      {t("studio.deploy.episodes.label")}
                    </Label>
                    <NumberInput
                      id="deploy-episodes"
                      min={1}
                      max={MAX_EVAL_EPISODES}
                      value={evalEpisodes}
                      disabled={controlsLocked}
                      onChange={(v) => {
                        if (v !== undefined) setEvalEpisodes(v);
                      }}
                    />
                    <p className="text-xs text-muted-foreground">
                      {t("studio.deploy.episodes.scoreHint")}
                    </p>
                  </div>
                ) : null}
              </TabsContent>

              {/* Human in the loop — coaching (DAgger). ------------------ */}
              <TabsContent value="coach" className="mt-0 space-y-6">
                {/* Readiness, stated before the operator commits an hour to it.
                    Coaching corrects a policy's OWN failures, so it has nothing
                    to work with until the policy sometimes succeeds: CR-DAgger
                    (arXiv:2506.16685) recommends starting only once the base
                    policy is at 10-20%, and below that the honest answer is more
                    demonstrations, not more corrections. This is the cheapest
                    possible place to say so — the alternative is discovering it
                    after a session spent rescuing an arm that never got close. */}
                <div className="rounded-lg border border-border bg-muted/40 p-3">
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    <span className="font-semibold text-foreground">
                      Coaching pays off once the policy already works sometimes.
                    </span>{" "}
                    It learns from rescuing the policy's own mistakes, so it
                    needs the policy to get far enough to make interesting ones
                    — roughly a 1-in-10 success rate. If it fails immediately
                    every time, record more demonstrations first; that's faster
                    than correcting your way there.
                  </p>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="deploy-corrections">
                    {t("studio.deploy.coaching.correctionsLabel")}
                  </Label>
                  <NumberInput
                    id="deploy-corrections"
                    min={1}
                    max={MAX_COACHING_CORRECTIONS}
                    value={targetCorrections}
                    onChange={(v) => {
                      if (v !== undefined) setTargetCorrections(v);
                    }}
                  />
                  <p className="text-xs text-muted-foreground">
                    {t("studio.deploy.coaching.correctionsHint")}
                  </p>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="deploy-coach-dataset">
                    {t("studio.deploy.coaching.datasetLabel")}
                  </Label>
                  {/* `rollout_` is rendered as a fixed, unfocusable part of the
                      field rather than left in the operator's text. It is not
                      optional — lerobot refuses a rollout dataset whose repo
                      name lacks it (rollout/context.py), and merge.py keys the
                      lossless `intervention`-column drop on the same prefix — so
                      it must never be something a person can delete or forget.
                      What they type follows it. */}
                  <div className="flex items-center rounded-md border border-input bg-background focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2 ring-offset-background">
                    <span
                      aria-hidden
                      className="select-none pl-3 pr-0.5 font-mono text-sm text-muted-foreground"
                    >
                      rollout_
                    </span>
                    <Input
                      id="deploy-coach-dataset"
                      value={coachDatasetName}
                      onChange={(e) => setCoachDatasetName(e.target.value)}
                      placeholder={
                        defaultCoachName ||
                        t("studio.deploy.coaching.datasetFallback")
                      }
                      className="border-0 bg-transparent pl-0 font-mono shadow-none focus-visible:ring-0 focus-visible:ring-offset-0"
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {/* <0> wraps the literal on-disk prefix, which is an
                        identifier and stays in the Latin script. */}
                    <Trans
                      i18nKey="studio.deploy.coaching.datasetHint"
                      values={{
                        prefix: `rollout_${
                          effectiveCoachName ||
                          t("studio.deploy.coaching.datasetFallback")
                        }_`,
                      }}
                      components={[<span key="0" className="font-mono" />]}
                    />
                  </p>
                </div>
                <div className="space-y-2">
                  <Label>{t("studio.deploy.coaching.leaderLabel")}</Label>
                  <p
                    className={cn(
                      "text-xs",
                      coachLeaderMissing
                        ? "text-destructive"
                        : "text-muted-foreground",
                    )}
                  >
                    {!robot
                      ? t("studio.deploy.coaching.leaderNoRobot")
                      : coachLeaderMissing
                        ? t("studio.deploy.coaching.leaderMissing")
                        : t("studio.deploy.coaching.leaderFrom", {
                            name: robot.name,
                            // Calibration file names — data, never translated.
                            configs: isBimanual
                              ? `${robot.leader_config} + ${robot.right_leader_config}`
                              : robot.leader_config,
                          })}
                  </p>
                  {isBimanual ? (
                    <p className="text-xs text-warn">
                      {t("studio.deploy.coaching.bimanualWarning")}
                    </p>
                  ) : null}
                </div>
              </TabsContent>

            </Tabs>

            {/* Cameras — literally Collect's list: the same component, the same
                heading and sentence, the same cards, fed from the same place
                (the selected robot's record). At the FOOT of the form, because
                whichever way the run is set up it sees the same robot cameras,
                and this is the last thing to check before pressing Start —
                literally the thing an operator looks at while reaching for the
                button. Nothing is picked here: each camera the checkpoint was
                trained with takes the robot camera of the SAME NAME (see
                cameraBindings), a role that matches nothing is answered by the
                picker above, and the two remaining ways it can go wrong are
                reported once, below. ---------------------------------- */}
            <div className="space-y-4">
              <SessionCameraList
                cameras={robotCameras}
                paused={submitting || runActive}
                emptyLabel={
                  robot
                    ? t("studio.deploy.cameras.robotHasNone")
                    : t("studio.deploy.cameras.noRobot")
                }
              />

              {/* One alert for both failure modes, and only when there is one:
                  a camera the policy names that nothing on the robot answers to
                  (blocks Start — the rollout cannot invent the feed), and a
                  binding whose resolution differs from the checkpoint's (a
                  warning: the run captures at the policy's size regardless). */}
              {unmatchedCameras.length > 0 || mismatchedCameras.length > 0 ? (
                <Alert
                  variant={
                    unmatchedCameras.length > 0 ? "destructive" : undefined
                  }
                  className={
                    unmatchedCameras.length > 0
                      ? undefined
                      : "border-warn/40 text-warn [&>svg]:text-warn"
                  }
                >
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription className="space-y-1">
                    {unmatchedCameras.map((b) => (
                      <p key={b.mapping.requestKey}>
                        {/* The camera NAME is data (it is the robot record's
                            own key), so it rides in as a value and <0> only
                            makes it bold.

                            Remote runs keep their own sentence: renaming a
                            robot camera is the WRONG remedy there — the name on
                            the record is that camera's identity, and the role
                            picker above is the answer. */}
                        <Trans
                          i18nKey={
                            remote
                              ? "studio.deploy.cameras.unmatchedRemote"
                              : "studio.deploy.cameras.unmatched"
                          }
                          values={{ name: b.mapping.display }}
                          components={[<strong key="0" />]}
                        />
                      </p>
                    ))}
                    {/* `resolutionDiffers` is only ever true with both sides
                        present; the guard is what tells the compiler so. */}
                    {mismatchedCameras.map(({ mapping, camera, dims }) =>
                      camera && dims ? (
                        <p key={mapping.requestKey}>
                          <Trans
                            i18nKey="studio.deploy.cameras.resolutionMismatch"
                            values={{
                              name: camera.name,
                              robotWidth: camera.width,
                              robotHeight: camera.height,
                              policyWidth: dims.width,
                              policyHeight: dims.height,
                            }}
                            components={[<strong key="0" />]}
                          />
                        </p>
                      ) : null,
                    )}
                  </AlertDescription>
                </Alert>
              ) : null}
            </div>

            {/* Advanced parameters — same AdvancedSection trigger and inner
                eyebrow/label/help-text rhythm as the Train form's AdvancedCard,
                so the two panels read as one form. ACT-only for now: temporal
                ensembling is an ACT config field, so for every other policy type
                the block has nothing to hold and stays hidden. ------------- */}
            {/* Hidden for a remote run for the same reason it is hidden for a
                non-ACT policy: there is no local rollout whose action selection
                this could configure. */}
            {isAct && !remote ? (
              <AdvancedSection
                open={advancedOpen}
                onOpenChange={setAdvancedOpen}
                summary={t("studio.deploy.advanced.summary")}
              >
                <div className="space-y-6">
                  <section className="space-y-3">
                    <h4 className={eyebrow}>
                      {t("studio.deploy.advanced.actionSelection")}
                    </h4>
                    <div className="flex items-center gap-3">
                      <Switch
                        id="deploy-temporal-ensemble"
                        checked={temporalEnsemble}
                        onCheckedChange={setTemporalEnsemble}
                        className="data-[state=checked]:bg-primary"
                      />
                      <Label htmlFor="deploy-temporal-ensemble">
                        {t("studio.deploy.advanced.temporalEnsemble")}
                      </Label>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {t("studio.deploy.advanced.temporalEnsembleHint")}
                    </p>
                    {temporalEnsemble ? (
                      <div className="space-y-2">
                        <Label htmlFor="deploy-temporal-ensemble-coeff">
                          {t("studio.deploy.advanced.coeffLabel")}
                        </Label>
                        <NumberInput
                          id="deploy-temporal-ensemble-coeff"
                          integer={false}
                          step="0.001"
                          min={0}
                          value={temporalEnsembleCoeff}
                          onChange={setTemporalEnsembleCoeff}
                          placeholder={t(
                            "studio.deploy.advanced.coeffPlaceholder",
                            { value: DEFAULT_TEMPORAL_ENSEMBLE_COEFF },
                          )}
                          aria-invalid={temporalEnsembleInvalid}
                          className={cn(
                            "w-40",
                            temporalEnsembleInvalid && "border-destructive",
                          )}
                        />
                        {temporalEnsembleInvalid ? (
                          <p className="text-xs text-destructive">
                            {t("studio.deploy.advanced.coeffInvalid")}
                          </p>
                        ) : (
                          <p className="text-xs text-muted-foreground">
                            {t("studio.deploy.advanced.coeffHint", {
                              value: DEFAULT_TEMPORAL_ENSEMBLE_COEFF,
                            })}
                          </p>
                        )}
                      </div>
                    ) : null}
                  </section>
                </div>
              </AdvancedSection>
            ) : null}
          </div>
        </CollapsibleContent>
      </Collapsible>

      {/* Deploy-started milestone — the effect above latches this true the
          first time the live InferenceSessionDialog closes after handleStart
          sets deployMilestonePending. Gated on the latched flag alone (not
          live on !sessionOpen) so a later, unrelated session close (a normal
          redeploy from this same panel — the banner's own copy invites
          exactly that) can't resurrect an already-dismissed-or-shown
          banner. */}
      {showDeployMilestone && (
        <MilestoneReveal
          title={t("studio.deploy.milestone.title")}
          description={t("studio.deploy.milestone.description")}
          onDismiss={() => setShowDeployMilestone(false)}
        />
      )}

      {/* Actions — directly under the form at the panel's normal gap-5 rhythm;
          the library below carries the column's stretch, so this row no longer
          needs mt-auto.

          One Start, for whichever tab is selected: the run shape was chosen
          in the strip above, so the button has nothing left to say but go,
          and the reason it refuses (when it refuses) is the selected tab's.
          No Stop beside it — see the note at `selectedSkillLabel` for why the
          panel's own stop was dropped rather than kept. ------------------ */}
      <div className="flex flex-col gap-2">
        <Button
          onClick={() => void handleStart(runMode)}
          disabled={!canStartAnyMode || startBlockedKey !== null}
          className="w-full"
        >
          {submitting || checkingExtra ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Play className="h-4 w-4" />
          )}
          {checkingExtra
            ? t("studio.deploy.actions.checking")
            : submitting
              ? t("studio.deploy.actions.starting")
              : coaching
                ? t("studio.deploy.actions.startCoach", {
                    corrections: targetCorrections,
                  })
                : scoring
                  ? t("studio.deploy.actions.startEval", {
                      episodes: evalEpisodes,
                    })
                  : t("studio.deploy.actions.start")}
        </Button>
        {/* The refusal. When it is the transport's, the PROBE's own sentence
            stands in for the generic one: it names which of the five things
            between here and a running GPU is the one to fix, which the generic
            line never could — and it carries the probe's tone, so "still
            checking" does not read as a failure. */}
        {startBlockedKey === "studio.deploy.blocked.transportNotReady" ? (
          <>
            <p className={cn("text-xs leading-relaxed", transportTone)}>
              {t(transportSummary.key as never, transportSummary.values)}
            </p>
            {/* The one verdict whose remedy is a command rather than a
                setting, and the only thing the retired Transport section
                rendered that had nowhere else to go. Shown only for that
                case, so the Start row stays one sentence the rest of the
                time: the sentence, the command, and — when the server sent
                one — how to get `livekit-server` in the first place. */}
            {transportSummary.key === SFU_OFF_SUMMARY_KEY ? (
              <>
                {/* A shell line: DATA, verbatim, never translated. */}
                <pre className="overflow-x-auto rounded bg-muted/60 p-2 font-mono text-[11px] break-words whitespace-pre-wrap">
                  makermodslab --sfu --sfu-external-ip
                </pre>
                {remoteTransport.transport?.sfu_install_hint ? (
                  // The backend's own per-OS install line, shown as raised.
                  <p className="text-[11px] leading-relaxed text-warn">
                    {remoteTransport.transport.sfu_install_hint}
                  </p>
                ) : null}
              </>
            ) : null}
            {transportDetail ? (
              // The backend's code and message — data, verbatim.
              <p className="font-mono text-xs leading-relaxed break-all text-destructive">
                {transportDetail}
              </p>
            ) : null}
          </>
        ) : startBlockedKey ? (
          <p className="text-xs leading-relaxed text-warn">
            {t(startBlockedKey as never)}
          </p>
        ) : null}
      </div>

      {/* Model / policy library — imported models + uploaded Hub repos.
          Picking a card selects it as the skill in the form above (step null →
          the checkpoint loader falls back to the latest). LibrarySection's own
          stretch now stands (no mt-0 override): the opener and Start row
          top-pack, the free space falls between them and this, and the library
          sits at the column foot so its "Show all" footer lines up with
          Collect's and Train's. Its body is a scrolling viewport, so expanding
          scrolls inside it and the footer never moves. */}
      <LibrarySection>
        <ModelsLibrary
          open={libraryOpen}
          onOpenChange={setLibraryOpen}
          onPick={(job, step) => {
            setPendingStep(step);
            setSelectedJob(job);
            setSelectedModelId(job.id);
            // …and, like a prefill, slide the form open onto it: the skill a
            // card selects is only configurable in there, so leaving the form
            // shut would make the click look like it did nothing.
            toggleForm(true);
          }}
        />
      </LibrarySection>

      {policyExtra && (
        <PolicyExtraDialog
          open={!!policyExtra}
          onOpenChange={(o) => !o && setPolicyExtra(null)}
          policyType={policyExtra.policyType}
          packageName={policyExtra.packageName}
          installTarget={policyExtra.installTarget}
          installHint={policyExtra.installHint}
          purpose="inference"
        />
      )}
    </div>
  );
};

export default DeployPanel;
