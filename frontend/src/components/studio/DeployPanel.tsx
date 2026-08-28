import React, {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { Trans, useTranslation } from "react-i18next";
import {
  AlertTriangle,
  Download,
  Loader2,
  Play,
  Square,
  VideoOff,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Alert, AlertDescription } from "@/components/ui/alert";
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
import { useRobots } from "@/hooks/useRobots";
import { formatRobotSetupGap } from "@/lib/robotSetupGap";
import { useInferenceLaunch } from "@/hooks/useInferenceLaunch";
import {
  JobCheckpoint,
  PolicyConfigSummary,
  getCheckpointPolicyConfig,
  listJobCheckpoints,
} from "@/lib/checkpointsApi";
import {
  InferenceStatus,
  getInferenceStatus,
  stopInference,
} from "@/lib/inferenceApi";
import { startSession, formatSessionHeld } from "@/lib/sessionApi";
import { tabOwnerId } from "@/lib/sessionOwner";
import { JobRecord, getJob, jobDisplayName } from "@/lib/jobsApi";
import { SkillItem } from "@/lib/modelsApi";
import { useSkills } from "@/hooks/useSkills";
import { importSourceForModel } from "@/lib/inferenceLaunch";
import DisplayName from "@/components/library/DisplayName";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import ModelsLibrary from "@/components/jobs/ModelsLibrary";
import ImportModelModal from "@/components/jobs/ImportModelModal";
import PolicyExtraDialog from "@/components/training/PolicyExtraDialog";
import {
  AdvancedSection,
  FormSection,
  LibrarySection,
  PANEL_ENTRY_CLASS,
  PanelEntryDot,
  PanelHeader,
  RobotStatus,
  useEyebrowClass,
} from "@/components/studio/panel/primitives";
import { useLanguage } from "@/contexts/LanguageContext";
import { isCaselessScript } from "@/i18n/config";
import { cn } from "@/lib/utils";
import { useAvailableCameras } from "@/hooks/useAvailableCameras";
import BackendCameraStream from "@/components/BackendCameraStream";
import type { CameraConfig } from "@/components/recording/CameraConfiguration";
import { isCameraConnected, resolveCameraIndex } from "@/lib/cameraResolve";
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
 * arm-count guard, the camera thumbnails and the start flow are ported VERBATIM
 * from `components/landing/InferenceModal.tsx` (only the palette becomes token
 * classes). The Hub lazy-import reuses `useInferenceLaunch().importSource` so
 * the husk-repo messaging is identical, not re-implemented.
 */

// Mirrors rollout.MAX_EVAL_EPISODES — the server clamps to the same bound, this
// just stops the stepper from offering a number that would be silently reduced.
const MAX_EVAL_EPISODES = 200;

// Mirrors rollout.MAX_COACHING_CORRECTIONS. Far lower than the eval bound
// because every correction is a hands-on takeover — the operator is standing at
// the arm for all of them.
const MAX_COACHING_CORRECTIONS = 100;

// The three shapes a Deploy run can take. One control instead of inferring the
// mode from an episode count, which was already a little cryptic at 1-vs-many
// and would be worse with a third option folded in.
type RunMode = "single" | "eval" | "coach";

/** Coefficient of the original ACT paper's exponential weighting (see
 * lerobot's ACTTemporalEnsembler). Offered as the starting point when the user
 * switches temporal ensembling on. */
const DEFAULT_TEMPORAL_ENSEMBLE_COEFF = 0.01;

/** Small preview for verifying which physical camera a role binds to.
 *
 * Streams from the backend by cv2 index — the live feed at exactly the index
 * the rollout will open, independent of any browser deviceId match. That match
 * was by localizedName, so twin cameras ("KD-USB Cameras" x2) paired
 * arbitrarily and the tiles swapped footage between refreshes.
 * `paused` unmounts the stream so the rollout subprocess can claim the device.
 * (Ported from InferenceModal.) */
const CameraThumbnail: React.FC<{
  cameraIndex?: number;
  uniqueId?: string;
  paused: boolean;
}> = ({ cameraIndex, uniqueId, paused }) => {
  const { t } = useTranslation();
  if (paused || cameraIndex === undefined) {
    return (
      <div className="flex h-24 w-32 flex-col items-center justify-center rounded border border-border bg-muted">
        <VideoOff className="mb-1 h-5 w-5 text-muted-foreground" />
        <span className="text-[10px] text-muted-foreground">
          {paused
            ? t("studio.deploy.thumbnail.released")
            : t("studio.deploy.thumbnail.noPreview")}
        </span>
      </div>
    );
  }
  // BackendCameraStream owns its own failure/retry UI.
  return (
    <BackendCameraStream
      cameraIndex={cameraIndex}
      uniqueId={uniqueId}
      className="h-24 w-32 rounded border border-border bg-muted object-cover"
    />
  );
};

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

const DeployPanel: React.FC = () => {
  const { t } = useTranslation();
  const { language } = useLanguage();
  const isCJK = isCaselessScript(language);
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
  // Duplicate of ModelsLibrary's "Import skill" entry point, surfaced right
  // on the skill picker itself so importing doesn't require scrolling down
  // to the library section below.
  const [importModalOpen, setImportModalOpen] = useState(false);

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
      ? checkpoints.find((c) => c.ref === selectedRef) ?? null
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
  // Which of the three run shapes this launch is. `evalEpisodes` still carries
  // the count, but the MODE is explicit now rather than implied by it being >1.
  const [runMode, setRunMode] = useState<RunMode>("single");
  // Coaching (DAgger): run the policy, take over when it's about to fail, and
  // record each takeover as training data. See StartInferenceRequest.coaching.
  const [targetCorrections, setTargetCorrections] = useState(10);
  const [coachDatasetName, setCoachDatasetName] = useState("");
  // Inference engine A/B. "sync" is the server default and the historical
  // behaviour; "rtc" is experimental (see InferenceSessionOptions).
  const [inferenceEngine, setInferenceEngine] = useState<"sync" | "rtc">("sync");
  const [submitting, setSubmitting] = useState(false);
  // ACT temporal ensembling. Held as (on, coeff) rather than `number | null`
  // so clearing the number field mid-edit doesn't silently switch the feature
  // off; the request sends the coeff only while `on`.
  const [temporalEnsemble, setTemporalEnsemble] = useState(false);
  const [temporalEnsembleCoeff, setTemporalEnsembleCoeff] = useState<
    number | undefined
  >(DEFAULT_TEMPORAL_ENSEMBLE_COEFF);
  const [advancedOpen, setAdvancedOpen] = useState(false);

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

  // Per camera DISPLAY name → the NAME of one of the selected robot's cameras.
  // Keyed by the stripped display name (== requestKey), and sent verbatim as
  // the request's `camera_bindings`. The binding is a name pairing only: the
  // server reads which device and how to open it (index, unique_id, fps,
  // fourcc, backend) out of the robot record, so a run can never open a camera
  // set the saved robot doesn't have. Capture resolution is the exception —
  // it's forwarded from the checkpoint as `camera_dims`, because the rollout
  // doesn't resize frames. Cameras are edited in Robot settings.
  const [cameraBindings, setCameraBindings] = useState<
    Record<string, string | null>
  >({});
  const { cameras: availableCameras } = useAvailableCameras({ enabled: open });

  // Light status poll while the panel is visible so ⏹ Stop enables only when a
  // rollout is actually active.
  const [status, setStatus] = useState<InferenceStatus | null>(null);
  const [stopping, setStopping] = useState(false);

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

  // The settings block (robot, checkpoint, run parameters, cameras) collapses
  // as one so a configured deploy can be folded down to picker + actions.

  const jobId = selectedJob?.id ?? null;
  // Address the policy-config endpoint by the checkpoint's OWNER. `(owner, step)`
  // is unique even on a rewound lineage, where `(tip, step)` is not. Falls back
  // to the tip when there is no owner — a single-run listing, where the step is
  // unique anyway.
  const policyConfigJobId = selectedCheckpoint?.owner_job_id ?? jobId;
  const isBimanual = robot?.mode === "bimanual";

  const cameraMap = useMemo(
    () =>
      cameraMappings(Object.keys(policyConfig?.image_features ?? {}), isBimanual),
    [policyConfig, isBimanual],
  );

  const robotCameras: CameraConfig[] = useMemo(
    () => robot?.cameras ?? [],
    [robot],
  );

  /** The robot's camera a binding names, or undefined once the record no
   * longer has it (camera removed in Robot settings, or another robot
   * selected). */
  const recordCameraByName = useCallback(
    (name: string | null | undefined) =>
      name == null ? undefined : robotCameras.find((cam) => cam.name === name),
    [robotCameras],
  );

  /** Bound AND physically present. A stored camera_index goes stale on replug,
   * so presence is judged by unique_id against the live enumeration — the same
   * check the preview tiles use, and the same strictness Start had when
   * bindings pointed straight at an enumerated device. */
  const cameraIsReady = useCallback(
    (name: string | null | undefined) => {
      const cam = recordCameraByName(name);
      return cam != null && isCameraConnected(cam, availableCameras);
    },
    [recordCameraByName, availableCameras],
  );

  // Opening the studio is a freshness gesture, so it still re-pulls — but it is
  // no longer the ONLY thing that does, which is what made a run completing
  // behind an open panel invisible until it was closed and reopened.
  useEffect(() => {
    if (open) refreshModels();
  }, [open, refreshModels]);

  // Re-pull after a successful import so the new skill shows up right away —
  // mirrors ModelsLibrary's onImported. Still explicit: the import posts to
  // /jobs, so `jobs_changed` covers it, but a fire-and-forget broadcast the
  // server drops when no socket is registered is not something the panel that
  // just did the import should be relying on.
  const handleImported = useCallback(() => {
    refreshModels();
  }, [refreshModels]);

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
      // The settings (robot, checkpoint, cameras) are no longer collapsible —
      // they render as soon as a skill is selected, which the prefill does
      // below, so there is nothing to re-open here.
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
            title: t("studio.deploy.toast.loadSkillFailed"),
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
    toast,
    t,
  ]);

  // Manual skill pick: resolve the chosen model to a launchable job (its own
  // registry id, an already-imported repo, or a fresh lazy import).
  const handlePickSkill = useCallback(
    async (modelId: string) => {
      setSelectedModelId(modelId);
      const model = models.find((m) => m.id === modelId);
      if (!model) return;
      // New skill → drop the prior selection so the load effect picks the new
      // job's latest checkpoint.
      setSelectedRef(null);
      setPendingStep(null);
      setResolving(true);
      try {
        // `job_id` is stamped by the server, which already ranks the runs
        // sharing an output repo (`_job_outranks`). This used to re-list up to
        // 200 jobs and re-implement that ranking in TypeScript — a second copy
        // of the definition, kept in sync by hand, and blind to any run past
        // the scan limit.
        if (model.job_id) {
          try {
            const job = await getJob(baseUrl, fetchWithHeaders, model.job_id);
            setSelectedJob(job);
            return;
          } catch {
            // The record went away between the listing and this click (deleted
            // in another tab, or from the Train panel). The weights may still be
            // on the Hub, so fall through to the import rather than dead-end —
            // the path the old lookup took whenever it found no job at all.
          }
        }
        // No run tracks it (a bare Hub repo, a scanned directory), or the one
        // that did is gone — the lazy import registers one, as before.
        const imported = await importSource(importSourceForModel(model));
        if (imported) setSelectedJob(imported);
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
        const mappings = cameraMappings(
          Object.keys(cfg.image_features),
          isBimanual,
        );
        setCameraBindings((prev) => {
          const next: Record<string, string | null> = {};
          for (const m of mappings) {
            next[m.requestKey] = prev[m.requestKey] ?? null;
          }
          return next;
        });
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
  }, [open, baseUrl, fetchWithHeaders, policyConfigJobId, selectedStep, isBimanual]);

  // Auto-bind robot cameras whose names match a policy-expected camera, by
  // name against the DISPLAY name (the bare name the user chose at record
  // time — that's what the robot record stores). No device enumeration is
  // involved any more: the binding names a RECORD camera, and the record is
  // what the server resolves against.
  useEffect(() => {
    if (!policyConfig || robotCameras.length === 0) return;
    setCameraBindings((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const m of cameraMap) {
        if (next[m.requestKey] != null) continue;
        const robotCam = robotCameras.find(
          (c) => c.name.toLowerCase() === m.display.toLowerCase(),
        );
        if (robotCam) {
          next[m.requestKey] = robotCam.name;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [policyConfig, robotCameras, cameraMap]);

  // Drop a binding the robot record no longer backs — the camera was removed
  // in Robot settings, or another robot was selected. (A binding to a camera
  // that is merely UNPLUGGED is kept: the tile says "disconnected" and Start
  // stays disabled, so reconnecting it doesn't cost the user the binding.)
  useEffect(() => {
    if (!policyConfig) return;
    setCameraBindings((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const [name, boundTo] of Object.entries(prev)) {
        if (boundTo != null && !recordCameraByName(boundTo)) {
          next[name] = null;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [policyConfig, recordCameraByName]);

  // Poll inference status while visible so ⏹ Stop reflects a live rollout.
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
  const SO101_DOF = 6;
  const checkpointDim =
    policyConfig?.state_dim ?? policyConfig?.action_dim ?? null;
  const checkpointArms =
    checkpointDim != null && checkpointDim % SO101_DOF === 0
      ? checkpointDim / SO101_DOF
      : null;
  const checkpointIsBimanual = checkpointArms != null && checkpointArms >= 2;
  const robotCheckpointArmMismatch =
    !!robot &&
    !!policyConfig &&
    checkpointArms != null &&
    checkpointIsBimanual !== isBimanual;

  const allCamerasBound = cameraMap.every((m) =>
    cameraIsReady(cameraBindings[m.requestKey]),
  );

  const inferenceActive = status?.inference_active === true;

  // Temporal ensembling is an ACT config field — no other policy type has it,
  // and passing --policy.temporal_ensemble_coeff to one would fail the
  // rollout's config parse. Show the control only for ACT checkpoints.
  const isAct = policyConfig?.policy_type === "act";
  // Empty field or a non-positive number: the backend rejects it (weights are
  // exp(-coeff * i)), so block Start rather than round-trip a 400.
  const temporalEnsembleInvalid =
    isAct &&
    temporalEnsemble &&
    (temporalEnsembleCoeff === undefined || temporalEnsembleCoeff <= 0);

  // Inference drives the follower(s) only — gate on follower_ready, not
  // is_clean, so a robot with no leader port/calibration (which inference
  // never touches) can still deploy.
  // Coaching drives the leader as well — and drives it under torque during the
  // handover — so "follower is ready" is not enough. Without this the panel
  // showed green lights, Start was enabled, and the leader gap only surfaced
  // as a 400 from the server after the user had committed to a launch.
  const coachLeaderMissing =
    runMode === "coach" &&
    (!robot?.leader_port ||
      !robot?.leader_config ||
      (robot?.mode === "bimanual" &&
        (!robot?.right_leader_port || !robot?.right_leader_config)));

  const canStart =
    !!robot &&
    robot.follower_ready &&
    !coachLeaderMissing &&
    !robotCheckpointArmMismatch &&
    selectedRef != null &&
    !!policyConfig &&
    allCamerasBound &&
    !temporalEnsembleInvalid &&
    !submitting &&
    !checkingExtra &&
    !inferenceActive;

  const handleStart = async () => {
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

    // Drops every CameraThumbnail's browser stream so the rollout subprocess can
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
    for (const m of cameraMap) {
      const boundTo = cameraBindings[m.requestKey];
      if (boundTo == null || !recordCameraByName(boundTo)) continue;
      cameraBindingPayload[m.requestKey] = boundTo;
      const dims = policyConfig.image_features[m.feature];
      if (dims?.width && dims?.height) {
        cameraDimsPayload[m.requestKey] = {
          width: dims.width,
          height: dims.height,
        };
      }
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
          task,
          camera_bindings: cameraBindingPayload,
          camera_dims: cameraDimsPayload,
          duration_s: durationS,
          checkpoint_state_dim: policyConfig.state_dim ?? undefined,
          // Only ever >1 in eval mode — the count field is hidden otherwise,
          // but pinning it here means a stale value left over from switching
          // modes can't quietly turn a single run into a 20-episode evaluation.
          eval_episodes: runMode === "eval" ? evalEpisodes : 1,
          // Coaching is pinned to sync server-side too (RTC snaps the arm back
          // toward its pre-correction pose on hand-back); sending it correctly
          // from here keeps the request honest rather than relying on that.
          inference_engine: runMode === "coach" ? "sync" : inferenceEngine,
          ...(runMode === "coach"
            ? {
                coaching: true,
                target_corrections: targetCorrections,
                coaching_dataset_name: coachDatasetName,
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
      openInferenceSession(session.id);
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

  const handleStop = async () => {
    setStopping(true);
    try {
      await stopInference(baseUrl, fetchWithHeaders);
      toast({
        title: t("studio.deploy.toast.stoppingTitle"),
        description: t("studio.deploy.toast.stoppingBody"),
      });
    } catch (e) {
      toast({
        title: t("studio.deploy.toast.stopFailed"),
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setStopping(false);
    }
  };

  const onCameraBindingChange = (name: string, value: string) => {
    setCameraBindings((prev) => ({ ...prev, [name]: value }));
  };

  const selectedSkillLabel = selectedJob ? jobDisplayName(selectedJob) : null;

  return (
    <div className="flex flex-1 flex-col gap-5 p-5">
      <PanelHeader step="3" title={t("studio.deploy.title")} dataTour="studio-deploy">
        {resolving ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
        ) : null}
      </PanelHeader>

      {/* Skill picker — the panel's entry control. A real <Select> rather than
          a PanelEntryControl because picking a skill IS the value here, not a
          trigger that opens a form; it wears PANEL_ENTRY_CLASS and a dot so it
          still reads as the same control as Collect's and Train's openers. */}
      <div className="space-y-2">
        <div className="relative">
          <Select
            value={selectedModelId ?? undefined}
            onValueChange={handlePickSkill}
            disabled={resolving}
          >
            {/* justify-start + ml-auto on the chevron: SelectTrigger defaults
                to justify-between, which would shove the dot away from the
                label once a third child is added. pr-9 reserves room on the
                right for the Import button overlaid below, so the chevron
                and the button's own gutter don't collide. */}
            <SelectTrigger
              className={cn(
                PANEL_ENTRY_CLASS,
                "justify-start pr-9 [&>svg]:ml-auto [&>svg]:shrink-0",
              )}
            >
              <PanelEntryDot className="bg-sky-500" />
              {selectedSkillLabel ? (
                <DisplayName name={selectedSkillLabel} className="min-w-0" />
              ) : (
                <SelectValue placeholder={t("studio.deploy.picker.placeholder")} />
              )}
            </SelectTrigger>
            <SelectContent>
              {modelsLoading ? (
                <div className="px-2 py-1.5 text-xs text-muted-foreground">
                  {t("studio.deploy.picker.loading")}
                </div>
              ) : models.length === 0 ? (
                // "We could not ask" and "you own nothing" are different
                // sentences, and the picker used to render the second for the
                // first: a `/models` failure was caught into an empty array, so
                // a backend hiccup or a dropped Hub listing read on screen as
                // the user's skills having been deleted. The listing is only
                // empty when the fetch actually succeeded and returned nothing.
                <div
                  className={cn(
                    "px-2 py-1.5 text-xs",
                    modelsError ? "text-destructive" : "text-muted-foreground",
                  )}
                >
                  {modelsError
                    ? t("studio.deploy.picker.error")
                    : t("studio.deploy.picker.empty")}
                </div>
              ) : (
                models.map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    <DisplayName name={m.name} className="min-w-0" />
                    {/* `uppercase` is a no-op on Chinese but the tracking is
                        not — drop both together on a caseless script. */}
                    <span
                      className={cn(
                        "ml-2 text-[10px] text-muted-foreground",
                        isCJK ? "" : "uppercase tracking-wide",
                      )}
                    >
                      {m.source === "hub"
                        ? t("studio.deploy.source.hub")
                        : m.source === "both"
                          ? t("studio.deploy.source.both")
                          : t("studio.deploy.source.local")}
                    </span>
                    {/* A failed run that saved weights IS runnable, and the
                        Train panel's card has always run one. It is offered
                        here rather than silently withheld — but it says so,
                        because a non-zero exit is a fact about the run the
                        user should weigh before deploying it. */}
                    {m.state === "failed" && (
                      <span
                        className={cn(
                          "ml-2 text-[10px] text-amber-600 dark:text-amber-500",
                          isCJK ? "" : "uppercase tracking-wide",
                        )}
                      >
                        {t("studio.deploy.picker.failedBadge")}
                      </span>
                    )}
                  </SelectItem>
                ))
              )}
            </SelectContent>
          </Select>
          {/* An unreachable Hub used to look exactly like an empty shelf: the
              rows simply were not there. Now the listing says which it is, and
              keeps serving the last complete Hub result underneath. */}
          {hubStatus && !hubStatus.ok && (
            <p className="mt-1 px-1 text-[11px] text-amber-600 dark:text-amber-500">
              {t("studio.deploy.picker.hubDegraded")}
            </p>
          )}
          {/* Duplicate of ModelsLibrary's "Import skill" button, docked
              inside the picker's own box (right edge) so it's visible
              without opening the dropdown. A sibling overlay, not a child of
              SelectTrigger — SelectTrigger is itself a <button>, and Radix
              opens on pointerdown, so nesting would either be invalid HTML or
              also trigger the dropdown. Sitting on top as an absolutely
              positioned sibling means it alone receives the click. */}
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={(e) => {
              e.stopPropagation();
              setImportModalOpen(true);
            }}
            disabled={resolving}
            className="absolute right-1 top-1/2 h-7 w-7 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            title={t("studio.deploy.picker.import")}
            aria-label={t("studio.deploy.picker.import")}
          >
            <Download className="h-3.5 w-3.5" />
          </Button>
        </div>
        {!selectedJob ? (
          <p className="text-xs text-muted-foreground">
            {t("studio.deploy.picker.hint")}
          </p>
        ) : null}
      </div>
      <ImportModelModal
        open={importModalOpen}
        onOpenChange={setImportModalOpen}
        onImported={handleImported}
      />

      {/* Everything below is flat and appears as soon as a skill is picked —
          disclosure comes from the selection, not from a second click. The old
          "Settings & configuration" collapsible was an extra step neither
          Collect nor Train has. ------------------------------------------- */}
      {selectedJob ? (
        <div className="space-y-5">
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
                 Coaching gets its own key AND its own gap scope: it teleoperates
                 through the leader too, so a missing leader port is a real
                 blocker there and noise in every other mode. */
              <Trans
                i18nKey={
                  runMode === "coach"
                    ? "studio.deploy.robotNotReadyCoach"
                    : "studio.deploy.robotNotReady"
                }
                count={isBimanual ? 2 : 1}
                values={{
                  name: robot.name,
                  gap: formatRobotSetupGap(
                    t,
                    robot,
                    runMode === "coach" ? "all" : "follower",
                  ),
                }}
                components={[<strong key="0" />]}
              />
            )}
          </RobotStatus>

          {/* Checkpoint ------------------------------------------------------- */}
          <div className="space-y-2">
            <Label htmlFor="deploy-checkpoint">
              {t("studio.deploy.checkpoint.label")}
            </Label>
            {checkpoints.length === 0 ? (
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
                  // Lineage-wide list: two entries can share a step, so the ref
                  // is the only safe identity to select by.
                  selectedRef={selectedRef}
                  onChange={(c) => setSelectedRef(c.ref)}
                  owners={checkpointOwnerMap}
                />
              )}
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
          </div>

          {/* Run parameters — flat, each with its own <Label>; the old "Run
              parameters" eyebrow sat above two fields that already say what
              they are. --------------------------------------------------- */}
          {policyConfig ? (
            <>
              {policyConfig.requires_task ? (
                <div className="space-y-2">
                  <Label htmlFor="deploy-task">
                    {t("studio.deploy.task.label")}
                  </Label>
                  <Input
                    id="deploy-task"
                    value={task}
                    onChange={(e) => setTask(e.target.value)}
                    placeholder={t("studio.deploy.task.placeholder")}
                  />
                  <p className="text-xs text-muted-foreground">
                    {/* The policy type is an identifier — rendered verbatim. */}
                    {t("studio.deploy.task.hint", {
                      policyType: policyConfig.policy_type ?? "",
                    })}
                  </p>
                </div>
              ) : null}
              <div className="space-y-2">
                <Label htmlFor="deploy-run-mode">
                  {t("studio.deploy.runMode.label")}
                </Label>
                <Select
                  value={runMode}
                  onValueChange={(v) => setRunMode(v as RunMode)}
                >
                  <SelectTrigger id="deploy-run-mode">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {/* Option VALUES ("single" / "eval" / "coach") are what the
                        backend parses — only the labels are translated. */}
                    <SelectItem value="single">
                      {t("studio.deploy.runMode.single")}
                    </SelectItem>
                    <SelectItem value="eval">
                      {t("studio.deploy.runMode.eval")}
                    </SelectItem>
                    <SelectItem value="coach">
                      {t("studio.deploy.runMode.coach")}
                    </SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {runMode === "coach"
                    ? t("studio.deploy.runMode.coachHint")
                    : runMode === "eval"
                      ? t("studio.deploy.runMode.evalHint")
                      : t("studio.deploy.runMode.singleHint")}
                </p>
              </div>
              {runMode !== "coach" ? (
                <div className="space-y-2">
                  <Label htmlFor="deploy-duration">
                    {t("studio.deploy.duration.label")}
                  </Label>
                  <NumberInput
                    id="deploy-duration"
                    min={1}
                    value={durationS}
                    onChange={(v) => {
                      if (v !== undefined) setDurationS(v);
                    }}
                  />
                  <p className="text-xs text-muted-foreground">
                    {runMode === "eval"
                      ? t("studio.deploy.duration.hint")
                      : t("studio.deploy.duration.singleHint")}
                  </p>
                </div>
              ) : null}
              {runMode === "eval" ? (
                <div className="space-y-2">
                  <Label htmlFor="deploy-episodes">
                    {t("studio.deploy.episodes.label")}
                  </Label>
                  <NumberInput
                    id="deploy-episodes"
                    min={1}
                    max={MAX_EVAL_EPISODES}
                    value={evalEpisodes}
                    onChange={(v) => {
                      if (v !== undefined) setEvalEpisodes(v);
                    }}
                  />
                  <p className="text-xs text-muted-foreground">
                    {t("studio.deploy.episodes.scoreHint")}
                  </p>
                </div>
              ) : null}
              {runMode === "coach" ? (
                <>
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
                    <Input
                      id="deploy-coach-dataset"
                      value={coachDatasetName}
                      onChange={(e) => setCoachDatasetName(e.target.value)}
                      placeholder={t("studio.deploy.coaching.datasetPlaceholder")}
                    />
                    <p className="text-xs text-muted-foreground">
                      {/* <0> wraps the literal on-disk prefix, which is an
                          identifier and stays in the Latin script. */}
                      <Trans
                        i18nKey="studio.deploy.coaching.datasetHint"
                        values={{
                          prefix: `rollout_${coachDatasetName || "corrections"}_`,
                        }}
                        components={[<span key="0" className="font-mono" />]}
                      />
                    </p>
                  </div>
                  <div className="space-y-2">
                    <Label>{t("studio.deploy.coaching.leaderLabel")}</Label>
                    <p
                      className={`text-xs ${
                        coachLeaderMissing ? "text-destructive" : "text-muted-foreground"
                      }`}
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
                </>
              ) : null}
              {runMode !== "coach" ? (
                <div className="space-y-2">
                  <Label htmlFor="deploy-engine">
                    {t("studio.deploy.engine.label")}
                  </Label>
                  <Select
                    value={inferenceEngine}
                    onValueChange={(v) => setInferenceEngine(v as "sync" | "rtc")}
                  >
                    <SelectTrigger id="deploy-engine">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {/* Option VALUES ("sync" / "rtc") are what the backend
                          parses — only the labels are translated. */}
                      <SelectItem value="sync">
                        {t("studio.deploy.engine.sync")}
                      </SelectItem>
                      <SelectItem value="rtc">
                        {t("studio.deploy.engine.rtc")}
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    {inferenceEngine === "rtc"
                      ? t("studio.deploy.engine.rtcHint")
                      : t("studio.deploy.engine.syncHint")}
                  </p>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">
                  {t("studio.deploy.engine.coachingNote")}
                </p>
              )}
            </>
          ) : null}

          {/* Cameras — a repeater, so it keeps its eyebrow. ------------------ */}
          <FormSection title={t("studio.deploy.cameras.title")}>
              {policyConfigLoading ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {t("studio.deploy.cameras.loading")}
                </div>
              ) : policyConfigError ? (
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    {/* The error text is the backend's own — passed through. */}
                    {t("studio.deploy.cameras.configError", {
                      error: policyConfigError,
                    })}
                  </AlertDescription>
                </Alert>
              ) : !policyConfig ? null : cameraMap.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  {t("studio.deploy.cameras.none")}
                </p>
              ) : (
                <div className="space-y-3">
                  <p className="text-xs text-muted-foreground">
                    {t("studio.deploy.cameras.intro")}
                  </p>
                  {cameraMap.map((m) => {
                    const dims = policyConfig.image_features[m.feature];
                    const value = cameraBindings[m.requestKey];
                    const boundCamera = recordCameraByName(value);
                    const connected =
                      boundCamera != null &&
                      isCameraConnected(boundCamera, availableCameras);
                    return (
                      <div key={m.requestKey} className="flex items-center gap-3">
                        <div className="flex-1">
                          <Label className="text-sm font-medium">{m.display}</Label>
                          <p className="text-xs text-muted-foreground">
                            {t("studio.deploy.cameras.captures", {
                              width: dims.width,
                              height: dims.height,
                            })}
                          </p>
                          {boundCamera &&
                          (boundCamera.width !== dims.width ||
                            boundCamera.height !== dims.height) ? (
                            <p className="text-xs text-muted-foreground">
                              {t("studio.deploy.cameras.mismatch", {
                                name: boundCamera.name,
                                width: boundCamera.width,
                                height: boundCamera.height,
                              })}
                            </p>
                          ) : null}
                          {boundCamera && !connected ? (
                            <p className="text-xs text-destructive">
                              {t("studio.deploy.cameras.disconnected")}
                            </p>
                          ) : null}
                        </div>
                        <Select
                          value={value ?? undefined}
                          onValueChange={(v) => onCameraBindingChange(m.requestKey, v)}
                        >
                          <SelectTrigger className="w-52">
                            <SelectValue
                              placeholder={t("studio.deploy.cameras.select")}
                            />
                          </SelectTrigger>
                          <SelectContent>
                            {robotCameras.length === 0 ? (
                              <div className="px-2 py-1.5 text-xs text-muted-foreground">
                                {t("studio.deploy.cameras.robotHasNone")}
                              </div>
                            ) : (
                              robotCameras.map((cam) => (
                                <SelectItem key={cam.name} value={cam.name}>
                                  {cam.name} — {cam.width}×{cam.height}
                                </SelectItem>
                              ))
                            )}
                          </SelectContent>
                        </Select>
                        <CameraThumbnail
                          cameraIndex={
                            boundCamera && connected
                              ? resolveCameraIndex(boundCamera, availableCameras)
                              : undefined
                          }
                          uniqueId={boundCamera?.unique_id}
                          paused={submitting || inferenceActive}
                        />
                      </div>
                    );
                  })}
                </div>
              )}
          </FormSection>

          {/* Advanced parameters — same AdvancedSection trigger and inner
              eyebrow/label/help-text rhythm as the Train form's AdvancedCard,
              so the two panels read as one form. ACT-only for now: temporal
              ensembling is an ACT config field, so for every other policy type
              the block has nothing to hold and stays hidden. ------------- */}
          {isAct ? (
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
      ) : null}

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

      {/* Actions — pinned directly above the skill library. Side by side so
          the row sits level with Collect's and Train's single Start. -------- */}
      <div className="mt-auto flex gap-2 pt-2">
        <Button
          onClick={handleStart}
          disabled={!canStart}
          className="flex-1 gap-2"
        >
          <Play className="h-4 w-4" />
          {submitting
            ? t("studio.deploy.actions.starting")
            : checkingExtra
              ? t("studio.deploy.actions.checking")
              : runMode === "coach"
                ? t("studio.deploy.actions.startCoach", {
                    corrections: targetCorrections,
                  })
                : runMode === "eval"
                  ? t("studio.deploy.actions.startEval", {
                      episodes: evalEpisodes,
                    })
                  : t("studio.deploy.actions.start")}
        </Button>
        <Button
          onClick={handleStop}
          variant="outline"
          disabled={!inferenceActive || stopping}
          className="flex-1 gap-2"
        >
          <Square className="h-4 w-4" />
          {stopping
            ? t("studio.deploy.actions.stopping")
            : t("studio.deploy.actions.stop")}
        </Button>
      </div>

      {/* Model / policy library — imported models + uploaded Hub repos.
          Picking a card selects it as the skill above (step null → the
          checkpoint loader falls back to the latest). mt-0 keeps it glued to
          the actions block above, which carries the panel's mt-auto. */}
      <LibrarySection className="mt-0">
        <ModelsLibrary
          onPick={(job, step) => {
            setPendingStep(step);
            setSelectedJob(job);
            setSelectedModelId(job.id);
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
