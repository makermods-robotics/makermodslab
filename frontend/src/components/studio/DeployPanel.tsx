import React, { useCallback, useEffect, useMemo, useState } from "react";
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
import { useRobots, robotSetupGap } from "@/hooks/useRobots";
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
  startInference,
  stopInference,
} from "@/lib/inferenceApi";
import { JobRecord, getJob, jobDisplayName, listJobs } from "@/lib/jobsApi";
import { ModelItem, getModels } from "@/lib/modelsApi";
import { findJobForModel, importSourceForModel } from "@/lib/inferenceLaunch";
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
} from "@/components/studio/panel/primitives";
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

const JOB_SCAN_LIMIT = 200;
// Mirrors rollout.MAX_EVAL_EPISODES — the server clamps to the same bound, this
// just stops the stepper from offering a number that would be silently reduced.
const MAX_EVAL_EPISODES = 200;

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
  if (paused || cameraIndex === undefined) {
    return (
      <div className="flex h-24 w-32 flex-col items-center justify-center rounded border border-border bg-muted">
        <VideoOff className="mb-1 h-5 w-5 text-muted-foreground" />
        <span className="text-[10px] text-muted-foreground">
          {paused ? "Released" : "No preview"}
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
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { open, deployPrefill, clearDeployPrefill } = useStudio();
  const { openInferenceSession, sessionOpen } = useInferenceSession();
  const { selectedRecord: robot } = useRobots();
  // Reuse the shared lazy-import (husk-repo messaging + idempotent registration)
  // so a Hub skill resolves to a pseudo-job exactly as the Jobs cards do.
  const { importSource } = useInferenceLaunch();

  // --- Skill picker state ------------------------------------------------
  const [models, setModels] = useState<ModelItem[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<JobRecord | null>(null);
  const [resolving, setResolving] = useState(false);
  // Duplicate of ModelsLibrary's "Import skill" entry point, surfaced right
  // on the skill picker itself so importing doesn't require scrolling down
  // to the library section below.
  const [importModalOpen, setImportModalOpen] = useState(false);

  // --- Inference config state (ported from InferenceModal) ---------------
  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [task, setTask] = useState("");
  const [durationS, setDurationS] = useState(60);
  // Multi-episode evaluation. 1 (the default) is the plain single rollout; >1
  // switches the session dialog into eval mode — N scored episodes with a reset
  // between each and an accuracy at the end. Clamped again server-side.
  const [evalEpisodes, setEvalEpisodes] = useState(1);
  // Inference engine A/B. "sync" is the server default and the historical
  // behaviour; "rtc" is experimental (see StartInferenceRequest).
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

  const { seen: hasSeenDeployMilestone, markSeen: markDeployMilestoneSeen } =
    useOnceFlag("makerlab:milestone-first-deploy");
  const [showDeployMilestone, setShowDeployMilestone] = useState(false);

  // The settings block (robot, checkpoint, run parameters, cameras) collapses
  // as one so a configured deploy can be folded down to picker + actions.

  const jobId = selectedJob?.id ?? null;
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

  // Load the skill listing (local runs + Hub models) when the studio opens.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setModelsLoading(true);
    getModels(baseUrl, fetchWithHeaders)
      .then((m) => {
        if (!cancelled) setModels(m);
      })
      .catch(() => {
        if (!cancelled) setModels([]);
      })
      .finally(() => {
        if (!cancelled) setModelsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, baseUrl, fetchWithHeaders]);

  // Re-fetch the skill listing after a successful import so the new skill
  // shows up in the picker right away — mirrors ModelsLibrary's onImported.
  const handleImported = useCallback(() => {
    setModelsLoading(true);
    getModels(baseUrl, fetchWithHeaders)
      .then((m) => setModels(m))
      .catch(() => setModels([]))
      .finally(() => setModelsLoading(false));
  }, [baseUrl, fetchWithHeaders]);

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
          setSelectedStep(deployPrefill.step ?? null);
          setSelectedJob(job);
          setSelectedModelId(job.id);
        } else {
          const imported = await importSource(deployPrefill.id);
          if (cancelled || !imported) return;
          setSelectedStep(deployPrefill.step ?? null);
          setSelectedJob(imported);
          setSelectedModelId(imported.id);
        }
      } catch (e) {
        if (!cancelled) {
          toast({
            title: "Couldn't load the skill",
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
  ]);

  // Manual skill pick: resolve the chosen model to a launchable job (its own
  // registry id, an already-imported repo, or a fresh lazy import).
  const handlePickSkill = useCallback(
    async (modelId: string) => {
      setSelectedModelId(modelId);
      const model = models.find((m) => m.id === modelId);
      if (!model) return;
      // New skill → drop the prior step so the load effect picks the new job's
      // latest checkpoint.
      setSelectedStep(null);
      setResolving(true);
      try {
        const jobs = await listJobs(
          baseUrl,
          fetchWithHeaders,
          JOB_SCAN_LIMIT,
        );
        const hit = findJobForModel(model, jobs);
        if (hit) {
          setSelectedJob(hit);
          return;
        }
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
    listJobCheckpoints(baseUrl, fetchWithHeaders, jobId)
      .then((cks) => {
        if (cancelled) return;
        setCheckpoints(cks);
        if (cks.length > 0) {
          const latest = cks[cks.length - 1].step;
          setSelectedStep((prev) => (prev != null ? prev : latest));
        }
      })
      .catch(() => {
        if (cancelled) return;
        setCheckpoints([]);
      });
    return () => {
      cancelled = true;
    };
  }, [open, baseUrl, fetchWithHeaders, jobId]);

  // Load policy config when step changes.
  useEffect(() => {
    if (!open || !jobId || selectedStep == null) {
      setPolicyConfig(null);
      setPolicyConfigError(null);
      return;
    }
    let cancelled = false;
    setPolicyConfigLoading(true);
    setPolicyConfigError(null);
    getCheckpointPolicyConfig(baseUrl, fetchWithHeaders, jobId, selectedStep)
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
  }, [open, baseUrl, fetchWithHeaders, jobId, selectedStep, isBimanual]);

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

  const selectedRef =
    selectedStep != null
      ? checkpoints.find((c) => c.step === selectedStep)?.ref ?? null
      : null;

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
  const canStart =
    !!robot &&
    robot.follower_ready &&
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
          `${baseUrl}/system/policy-extra/${policyConfig.policy_type}`,
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
      await startInference(baseUrl, fetchWithHeaders, {
        follower_port: robot.follower_port,
        follower_config: robot.follower_config,
        policy_ref: selectedRef,
        task,
        camera_bindings: cameraBindingPayload,
        camera_dims: cameraDimsPayload,
        duration_s: durationS,
        mode: robot.mode,
        right_follower_port: robot.right_follower_port,
        right_follower_config: robot.right_follower_config,
        robot_name: robot.name,
        checkpoint_state_dim: policyConfig.state_dim ?? undefined,
        eval_episodes: evalEpisodes,
        inference_engine: inferenceEngine,
        // ACT-only, and only while the switch is on — otherwise omitted so the
        // checkpoint's own (ensembling-off) config stands.
        temporal_ensemble_coeff:
          isAct && temporalEnsemble ? temporalEnsembleCoeff : undefined,
      });
      // The run surfaces as the InferenceSessionDialog over this panel —
      // closing it lands back here (the studio stays open underneath).
      openInferenceSession();
      if (!hasSeenDeployMilestone) {
        setShowDeployMilestone(true);
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
        title: "Couldn't start inference",
        description: e instanceof Error ? e.message : String(e),
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
      toast({ title: "Stopping inference", description: "The rollout is winding down." });
    } catch (e) {
      toast({
        title: "Stop failed",
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
      <PanelHeader step="3" title="Run">
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
                <SelectValue placeholder="Pick a skill" />
              )}
            </SelectTrigger>
            <SelectContent>
              {modelsLoading ? (
                <div className="px-2 py-1.5 text-xs text-muted-foreground">
                  Loading skills…
                </div>
              ) : models.length === 0 ? (
                <div className="px-2 py-1.5 text-xs text-muted-foreground">
                  No trained or imported skills yet
                </div>
              ) : (
                models.map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    <DisplayName name={m.name} className="min-w-0" />
                    <span className="ml-2 text-[10px] uppercase tracking-wide text-muted-foreground">
                      {m.source === "hub"
                        ? "hub"
                        : m.source === "both"
                          ? "local · hub"
                          : "local"}
                    </span>
                  </SelectItem>
                ))
              )}
            </SelectContent>
          </Select>
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
            title="Import skill"
            aria-label="Import skill"
          >
            <Download className="h-3.5 w-3.5" />
          </Button>
        </div>
        {!selectedJob ? (
          <p className="text-xs text-muted-foreground">
            Pick a trained checkpoint or an imported Hub skill to run on your
            robot.
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
            Run this skill on your robot, then start inference.
          </p>

          {/* Robot readiness — a warning, not a parameter, so no eyebrow. A
              ready robot renders nothing: the robot menu already names the
              selection and its arm layout. */}
          <RobotStatus ready={!!robot && robot.follower_ready}>
            {!robot ? (
              <>
                Select a robot to run on — use the robot menu in the top-right
                corner of this window.
              </>
            ) : (
              <>
                <strong>{robot.name}</strong> {robotSetupGap(robot, "follower")}.
                Open Robot settings before running inference. (Inference only
                uses the follower arm{isBimanual ? "s" : ""} — leader setup
                isn't needed.)
              </>
            )}
          </RobotStatus>

          {/* Checkpoint ------------------------------------------------------- */}
          <div className="space-y-2">
            <Label htmlFor="deploy-checkpoint">Checkpoint</Label>
            {checkpoints.length === 0 ? (
                <Alert className="border-warn/40 text-warn [&>svg]:text-warn">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    No checkpoints available for this skill yet.
                  </AlertDescription>
                </Alert>
              ) : (
                <CheckpointDropdown
                  id="deploy-checkpoint"
                  checkpoints={checkpoints}
                  // Single-job list: steps are unique here, so the step maps
                  // 1:1 onto the checkpoint's identifying ref.
                  selectedRef={
                    checkpoints.find((c) => c.step === selectedStep)?.ref ??
                    null
                  }
                  onChange={(c) => setSelectedStep(c.step)}
                />
              )}
              {robotCheckpointArmMismatch ? (
                <Alert className="border-warn/40 text-warn [&>svg]:text-warn">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    {checkpointIsBimanual ? (
                      <>
                        This checkpoint was trained on a{" "}
                        <strong>bimanual robot</strong> ({checkpointDim}-dim state,{" "}
                        {checkpointArms} arms), but <strong>{robot?.name}</strong> is
                        a single-arm robot. Pick a single-arm checkpoint, or select a
                        bimanual robot from the top-right robot menu.
                      </>
                    ) : (
                      <>
                        This checkpoint was trained on a{" "}
                        <strong>single-arm robot</strong> ({checkpointDim}-dim
                        state), but <strong>{robot?.name}</strong> is a bimanual
                        robot. Pick a bimanual checkpoint, or select a single-arm
                        robot from the top-right robot menu.
                      </>
                    )}
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
                  <Label htmlFor="deploy-task">Task description</Label>
                  <Input
                    id="deploy-task"
                    value={task}
                    onChange={(e) => setTask(e.target.value)}
                    placeholder="e.g., pick up the red block"
                  />
                  <p className="text-xs text-muted-foreground">
                    This policy is language-conditioned ({policyConfig.policy_type}).
                  </p>
                </div>
              ) : null}
              <div className="space-y-2">
                <Label htmlFor="deploy-duration">Max duration (s)</Label>
                <NumberInput
                  id="deploy-duration"
                  min={1}
                  value={durationS}
                  onChange={(v) => {
                    if (v !== undefined) setDurationS(v);
                  }}
                />
                <p className="text-xs text-muted-foreground">
                  Per episode. An episode that runs this long without you
                  calling it a success counts as a failure.
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="deploy-episodes">Episodes</Label>
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
                  {evalEpisodes > 1
                    ? `Evaluation run: ${evalEpisodes} episodes with a reset between each, scored into an accuracy.`
                    : "Leave at 1 for a single run. More than 1 starts a scored evaluation."}
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="deploy-engine">Inference engine</Label>
                <Select
                  value={inferenceEngine}
                  onValueChange={(v) => setInferenceEngine(v as "sync" | "rtc")}
                >
                  <SelectTrigger id="deploy-engine">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="sync">Sync (default)</SelectItem>
                    <SelectItem value="rtc">
                      RTC — experimental, smoother control
                    </SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {inferenceEngine === "rtc"
                    ? "Real-Time Chunking overlaps inference with motion, removing the pause between action chunks. It also changes how actions are generated — compare against Sync before trusting a result."
                    : "One policy forward per control step. The arm pauses briefly between action chunks."}
                </p>
              </div>
            </>
          ) : null}

          {/* Cameras — a repeater, so it keeps its eyebrow. ------------------ */}
          <FormSection title="Cameras">
              {policyConfigLoading ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Reading policy config…
                </div>
              ) : policyConfigError ? (
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    Couldn't load policy config: {policyConfigError}
                  </AlertDescription>
                </Alert>
              ) : !policyConfig ? null : cameraMap.length === 0 ? (
                <p className="text-xs text-muted-foreground">
                  This policy doesn't use cameras.
                </p>
              ) : (
                <div className="space-y-3">
                  <p className="text-xs text-muted-foreground">
                    Bind one of this robot's cameras to each name the policy was
                    trained with. Which camera and how it's opened come from the
                    robot (edit in Robot settings); the capture resolution comes
                    from the checkpoint.
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
                            Captures at {dims.width}×{dims.height} — the
                            policy's resolution
                          </p>
                          {boundCamera &&
                          (boundCamera.width !== dims.width ||
                            boundCamera.height !== dims.height) ? (
                            <p className="text-xs text-muted-foreground">
                              ({boundCamera.name} is set to {boundCamera.width}×
                              {boundCamera.height} in Robot settings)
                            </p>
                          ) : null}
                          {boundCamera && !connected ? (
                            <p className="text-xs text-destructive">
                              Disconnected — reconnect it before starting
                            </p>
                          ) : null}
                        </div>
                        <Select
                          value={value ?? undefined}
                          onValueChange={(v) => onCameraBindingChange(m.requestKey, v)}
                        >
                          <SelectTrigger className="w-52">
                            <SelectValue placeholder="Select a camera" />
                          </SelectTrigger>
                          <SelectContent>
                            {robotCameras.length === 0 ? (
                              <div className="px-2 py-1.5 text-xs text-muted-foreground">
                                This robot has no cameras — add them in Robot
                                settings
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
              summary="Temporal ensembling for ACT"
            >
              <div className="space-y-6">
                <section className="space-y-3">
                  <h4 className="eyebrow">Action selection</h4>
                  <div className="flex items-center gap-3">
                    <Switch
                      id="deploy-temporal-ensemble"
                      checked={temporalEnsemble}
                      onCheckedChange={setTemporalEnsemble}
                      className="data-[state=checked]:bg-primary"
                    />
                    <Label htmlFor="deploy-temporal-ensemble">
                      Temporal ensembling
                    </Label>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Averages the overlapping action chunks the policy predicts
                    at each step instead of executing one chunk open-loop —
                    smoother motion, but the policy runs every control step, so
                    it is slower.
                  </p>
                  {temporalEnsemble ? (
                    <div className="space-y-2">
                      <Label htmlFor="deploy-temporal-ensemble-coeff">
                        Ensemble coefficient
                      </Label>
                      <NumberInput
                        id="deploy-temporal-ensemble-coeff"
                        integer={false}
                        step="0.001"
                        min={0}
                        value={temporalEnsembleCoeff}
                        onChange={setTemporalEnsembleCoeff}
                        placeholder={`${DEFAULT_TEMPORAL_ENSEMBLE_COEFF} (ACT paper default)`}
                        aria-invalid={temporalEnsembleInvalid}
                        className={cn(
                          "w-40",
                          temporalEnsembleInvalid && "border-destructive",
                        )}
                      />
                      {temporalEnsembleInvalid ? (
                        <p className="text-xs text-destructive">
                          Enter a number greater than 0.
                        </p>
                      ) : (
                        <p className="text-xs text-muted-foreground">
                          Weights are exp(-coeff × age): higher favours the
                          newest prediction, lower averages more evenly. The
                          ACT paper uses {DEFAULT_TEMPORAL_ENSEMBLE_COEFF}.
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

      {/* Deploy-started milestone — gated on the live InferenceSessionDialog
          actually being closed (!sessionOpen) so it appears once the user
          exits that session, mirroring the training milestone's
          !monitorJobId guard. */}
      {showDeployMilestone && !sessionOpen && (
        <MilestoneReveal
          title="First skill deployed!"
          description="Your robot just ran a trained policy. Come back here anytime to redeploy it, swap checkpoints, or run a different skill."
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
            ? "Starting…"
            : checkingExtra
              ? "Checking…"
              : evalEpisodes > 1
                ? `Start evaluation (${evalEpisodes})`
                : "Start inference"}
        </Button>
        <Button
          onClick={handleStop}
          variant="outline"
          disabled={!inferenceActive || stopping}
          className="flex-1 gap-2"
        >
          <Square className="h-4 w-4" />
          {stopping ? "Stopping…" : "Stop inference"}
        </Button>
      </div>

      {/* Model / policy library — imported models + uploaded Hub repos.
          Picking a card selects it as the skill above (step null → the
          checkpoint loader falls back to the latest). mt-0 keeps it glued to
          the actions block above, which carries the panel's mt-auto. */}
      <LibrarySection className="mt-0">
        <ModelsLibrary
          onPick={(job, step) => {
            setSelectedStep(step);
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
