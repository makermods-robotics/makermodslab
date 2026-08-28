import React, { useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { AdvancedSection } from "@/components/studio/panel/primitives";
import { cn } from "@/lib/utils";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AlertTriangle, CheckCircle, Loader2, Play, VideoOff } from "lucide-react";
import { RobotRecord } from "@/hooks/useRobots";
import { formatRobotSetupGap } from "@/lib/robotSetupGap";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useInferenceSession } from "@/contexts/InferenceSessionContext";
import {
  JobCheckpoint,
  PolicyConfigSummary,
  getCheckpointPolicyConfig,
  listJobCheckpoints,
} from "@/lib/checkpointsApi";
import { startSession, formatSessionHeld } from "@/lib/sessionApi";
import { tabOwnerId } from "@/lib/sessionOwner";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import { useAvailableCameras } from "@/hooks/useAvailableCameras";
import BackendCameraStream from "@/components/BackendCameraStream";
import type { CameraConfig } from "@/components/recording/CameraConfiguration";
import { isCameraConnected, resolveCameraIndex } from "@/lib/cameraResolve";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  robot: RobotRecord | null;
  jobId: string;
  initialStep: number | null;
}

/** Coefficient of the original ACT paper's exponential weighting (see
 * lerobot's ACTTemporalEnsembler). Offered as the starting point when the user
 * switches temporal ensembling on. */
const DEFAULT_TEMPORAL_ENSEMBLE_COEFF = 0.01;


/** Small preview for verifying which physical camera a role binds to.
 *
 * Streams from the backend by cv2 index — the live feed at exactly the index
 * the rollout will open, independent of any browser deviceId match. That match
 * was by localizedName, so twin cameras ("KD-USB Cameras" x2) paired
 * arbitrarily and the front/wrist tiles swapped footage between refreshes.
 * `paused` unmounts the stream so the rollout subprocess can claim the device. */
const CameraThumbnail: React.FC<{
  cameraIndex?: number;
  uniqueId?: string;
  paused: boolean;
}> = ({ cameraIndex, uniqueId, paused }) => {
  const { t } = useTranslation();
  if (paused || cameraIndex === undefined) {
    return (
      <div className="w-32 h-24 bg-muted rounded border border-border flex flex-col items-center justify-center">
        <VideoOff className="w-5 h-5 text-muted-foreground mb-1" />
        <span className="text-[10px] text-muted-foreground">
          {paused
            ? t("landing.inference.thumbnailReleased")
            : t("landing.inference.thumbnailNoPreview")}
        </span>
      </div>
    );
  }
  // BackendCameraStream owns its own failure/retry UI.
  return (
    <BackendCameraStream
      cameraIndex={cameraIndex}
      uniqueId={uniqueId}
      className="w-32 h-24 object-cover rounded border border-border bg-muted"
    />
  );
};

/**
 * One camera as the modal sees it. The BiSO prefix round-trip lives here so the
 * future per-arm routing work has a single obvious place to extend.
 *
 * `feature` is the checkpoint's camera key exactly as it comes back from
 * `get_policy_config_summary` (the suffix after `observation.images.`). For a
 * bimanual checkpoint recorded through MakerMods Lab, lerobot's BiSOFollower parks every
 * camera on the LEFT arm and auto-prefixes each feature with `left_` when it
 * writes the dataset — so a camera the user named `front` at record time becomes
 * the feature `left_front`.
 *
 * Inference mirrors that: the rollout hands the request's camera dict to
 * `--robot.left_arm_config.cameras`, and BiSO re-prefixes with `left_` at
 * runtime. So the modal must bind + send under the BARE name (`front`), which
 * the rollout re-prefixes back to `left_front` — matching the checkpoint. If we
 * bound/sent under the literal `left_front` we'd emit `left_left_front` (double
 * prefix → policy mismatch).
 *
 * `display` / `requestKey` are therefore the stripped bare name in bimanual
 * mode, and identical to `feature` in single-arm mode (single-arm checkpoints
 * already carry bare names — a camera legitimately named e.g. `left_side` must
 * never be mangled). We only strip when it's unambiguous: see `cameraMappings`.
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
 * Guard against collisions: if two features would strip to the same bare name
 * (a checkpoint carrying BOTH `left_x` and `right_x`, or a bare `x` alongside
 * `left_x`), fall back to the FULL feature name for every colliding entry —
 * correctness over cosmetics. MakerMods Lab can't produce `right_*` checkpoints today,
 * so this is a defensive branch for externally-recorded bimanual checkpoints;
 * it must not silently mis-bind them.
 */
function cameraMappings(
  features: string[],
  isBimanual: boolean,
): CameraMapping[] {
  if (!isBimanual) {
    return features.map((f) => ({ feature: f, display: f, requestKey: f }));
  }
  // Count how many features want each stripped bare name so we can detect
  // collisions before committing to the shortened form.
  const strippedCounts = new Map<string, number>();
  for (const f of features) {
    const bare = f.replace(ARM_PREFIX_RE, "");
    strippedCounts.set(bare, (strippedCounts.get(bare) ?? 0) + 1);
  }
  return features.map((f) => {
    const bare = f.replace(ARM_PREFIX_RE, "");
    // Strip only when the bare name is unique across all features; otherwise
    // keep the full feature name so the two colliding cameras stay distinct.
    const name = strippedCounts.get(bare) === 1 ? bare : f;
    return { feature: f, display: name, requestKey: name };
  });
}

const InferenceModal: React.FC<Props> = ({
  open,
  onOpenChange,
  robot,
  jobId,
  initialStep,
}) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { openInferenceSession } = useInferenceSession();

  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(initialStep);
  const [task, setTask] = useState("");
  const [durationS, setDurationS] = useState(60);
  // Inference engine A/B. "sync" is the server default and the historical
  // behaviour; "rtc" is experimental (see InferenceSessionOptions).
  const [inferenceEngine, setInferenceEngine] = useState<"sync" | "rtc">("sync");
  const [submitting, setSubmitting] = useState(false);
  // ACT temporal ensembling — see DeployPanel, which carries the same pair.
  // (on, coeff) rather than `number | null` so clearing the field mid-edit
  // doesn't silently switch the feature off.
  const [temporalEnsemble, setTemporalEnsemble] = useState(false);
  const [temporalEnsembleCoeff, setTemporalEnsembleCoeff] = useState<
    number | undefined
  >(DEFAULT_TEMPORAL_ENSEMBLE_COEFF);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const [policyConfig, setPolicyConfig] = useState<PolicyConfigSummary | null>(null);
  const [policyConfigLoading, setPolicyConfigLoading] = useState(false);
  const [policyConfigError, setPolicyConfigError] = useState<string | null>(null);

  // Per camera DISPLAY name → the NAME of one of the selected robot's cameras.
  // Keyed by the stripped display name (== requestKey), not the checkpoint
  // feature key — see `cameraMappings` / the CameraMapping doc for the
  // round-trip. Sent verbatim as the request's `camera_bindings`: only the name
  // pairing travels, and the server reads which device and how to open it out
  // of the robot record (see makermodslab/utils/config.py bind_robot_cameras).
  // Capture resolution is the exception — forwarded from the checkpoint as
  // `camera_dims`, because the rollout doesn't resize frames.
  const [cameraBindings, setCameraBindings] = useState<Record<string, string | null>>({});
  const { cameras: availableCameras } = useAvailableCameras({ enabled: open });

  // `lerobot-rollout` drives any Robot generically, including `bi_so_follower`,
  // so a bimanual record now runs inference on BOTH followers — the server
  // stages the two follower calibrations and builds a `bi_so_follower` command.
  // We no longer block bimanual robots here.
  const isBimanual = robot?.mode === "bimanual";

  // Checkpoint feature ↔ display/request-key mapping. Bimanual checkpoints carry
  // BiSO's `left_`-prefixed camera features; the modal shows + binds + sends the
  // bare name so the rollout re-prefixes it back to match the checkpoint.
  const cameraMap = React.useMemo(
    () => cameraMappings(Object.keys(policyConfig?.image_features ?? {}), isBimanual),
    [policyConfig, isBimanual],
  );

  const robotCameras: CameraConfig[] = React.useMemo(
    () => robot?.cameras ?? [],
    [robot],
  );

  /** The robot's camera a binding names, or undefined once the record no
   * longer has it (camera removed in Robot settings, or another robot). */
  const recordCameraByName = React.useCallback(
    (name: string | null | undefined) =>
      name == null ? undefined : robotCameras.find((cam) => cam.name === name),
    [robotCameras],
  );

  /** Bound AND physically present — a stored camera_index goes stale on
   * replug, so presence is judged by unique_id against the live enumeration. */
  const cameraIsReady = React.useCallback(
    (name: string | null | undefined) => {
      const cam = recordCameraByName(name);
      return cam != null && isCameraConnected(cam, availableCameras);
    },
    [recordCameraByName, availableCameras],
  );

  // Re-arm Start on reopen. The success path closes the modal with
  // `submitting` still true (previews stay released while the rollout owns the
  // cameras), and the component stays MOUNTED in its consumers — so without
  // this reset a reopened modal would be stuck on a disabled "Starting…".
  useEffect(() => {
    if (open) setSubmitting(false);
  }, [open]);

  // Load checkpoints when modal opens.
  useEffect(() => {
    if (!open) return;
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
    if (!open || selectedStep == null) {
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
        // Reset camera bindings to one entry per DISPLAY name (bare name in
        // bimanual mode). Preserve any prior selection that's still relevant.
        const mappings = cameraMappings(Object.keys(cfg.image_features), isBimanual);
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

  // If the selected robot has cameras whose names match a policy-expected
  // camera, auto-bind them. Match against the DISPLAY name (the bare name the
  // user chose at record time — that's what the robot record stores), not the
  // `left_`-prefixed checkpoint feature. No device enumeration is involved: the
  // binding names a RECORD camera, and the record is what the server resolves.
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

  // Drop a binding the robot record no longer backs (camera removed in Robot
  // settings, or another robot selected). A merely UNPLUGGED camera keeps its
  // binding — the tile says disconnected and Start stays disabled.
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

  const selectedRef =
    selectedStep != null
      ? checkpoints.find((c) => c.step === selectedStep)?.ref ?? null
      : null;

  // Arm-count mismatch between the CHECKPOINT and the selected ROBOT. A
  // bimanual-trained checkpoint carries a two-arm-wide state/action (12 dims
  // for SO-101, 14 for a Maker arm) and left_/right_-prefixed camera names; a
  // single-arm checkpoint is one arm wide. Running a policy on the wrong arm count crashes on a shape mismatch
  // deep in the rollout subprocess. Detect it here from the checkpoint's state
  // dim (fall back to action dim) and explain it before Start. This is the
  // client mirror of the server's `_arm_count_mismatch` 409 guard — we forward
  // `checkpoint_state_dim` so the server enforces the same rule authoritatively.
  // Per-arm DOF is a property of the ROBOT, not a constant: an SO-101 arm is
  // 6-DOF and a Maker arm 7 (six joints plus its permanent gripper). Measured
  // against 6, a 7-dim Maker checkpoint is not a clean multiple, so
  // checkpointArms would resolve to null and this guard would silently go
  // quiet on exactly the mismatch it exists to catch. Mirrors the server's
  // `_ARM_STATE_DIMS` in rollout.py — change both together.
  const armDof = robot?.arm_type === "maker" ? 7 : 6;
  const checkpointDim = policyConfig?.state_dim ?? policyConfig?.action_dim ?? null;
  const checkpointArms =
    checkpointDim != null && checkpointDim % armDof === 0
      ? checkpointDim / armDof
      : null;
  const checkpointIsBimanual = checkpointArms != null && checkpointArms >= 2;
  // Flag both directions: a bimanual checkpoint on a single-arm robot, AND a
  // single-arm checkpoint on a bimanual robot. Only assert a mismatch when the
  // checkpoint exposes a recognisable arm count (checkpointArms != null) — a
  // vision-only checkpoint with no state dim can't be judged here, so we let
  // the server's post-mortem shape check speak instead of guessing.
  const robotCheckpointArmMismatch =
    !!robot &&
    !!policyConfig &&
    checkpointArms != null &&
    checkpointIsBimanual !== isBimanual;

  const allCamerasBound = cameraMap.every((m) =>
    cameraIsReady(cameraBindings[m.requestKey]),
  );

  // Inference drives the follower(s) only — gate on follower_ready, not
  // is_clean, so a robot with no leader setup can still run a policy.
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

  const canStart =
    !!robot &&
    robot.follower_ready &&
    !robotCheckpointArmMismatch &&
    selectedRef != null &&
    !!policyConfig &&
    allCamerasBound &&
    !temporalEnsembleInvalid &&
    !submitting;

  const handleStart = async () => {
    if (
      !robot ||
      robotCheckpointArmMismatch ||
      selectedRef == null ||
      !policyConfig
    )
      return;
    // Setting submitting=true makes every CameraPreview drop its
    // browser stream — required so the rollout subprocess can open the
    // same camera index via OpenCV without colliding on the device.
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
    // way `checkpoint_state_dim` is.
    const cameraDimsPayload: Record<string, { width: number; height: number }> = {};
    for (const m of cameraMap) {
      const boundTo = cameraBindings[m.requestKey];
      if (boundTo == null || !recordCameraByName(boundTo)) continue;
      cameraBindingPayload[m.requestKey] = boundTo;
      const dims = policyConfig.image_features[m.feature];
      if (dims?.width && dims?.height) {
        cameraDimsPayload[m.requestKey] = { width: dims.width, height: dims.height };
      }
    }
    try {
      // POST /api/v1/sessions returns as soon as the run is claimed (the
      // server downloads the model + preflights the arm in the background), so
      // this opens the inference dialog right away — the download and its
      // progress, any warn-but-allow arm finding, and any failure all surface
      // there via /inference-status polling. The request carries the robot
      // NAME plus policy-shaped options only — ports, configs, mode and the
      // camera devices behind the bindings all resolve server-side from the
      // saved record. The owner attaches the lease the dialog keeps renewed.
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
          // Forward the checkpoint's flat state width so the server enforces
          // the same arm-count guard authoritatively (omitted when the
          // checkpoint lacks observation.state — the server then defers to
          // its shape check).
          checkpoint_state_dim: policyConfig.state_dim ?? undefined,
          inference_engine: inferenceEngine,
          // ACT-only, and only while the switch is on — otherwise omitted so
          // the checkpoint's own (ensembling-off) config stands.
          temporal_ensemble_coeff:
            isAct && temporalEnsemble ? temporalEnsembleCoeff : undefined,
        },
      });
      onOpenChange(false);
      openInferenceSession(session.id);
    } catch (e) {
      toast({
        title: t("landing.inference.startFailedTitle"),
        // 409 session.held renders as the shared localized "robot is busy"
        // line; everything else is the server's raw error text — not ours to
        // translate.
        description:
          formatSessionHeld(t, e) ??
          (e instanceof Error ? e.message : String(e)),
        variant: "destructive",
      });
      // Failure: bring the previews back so the user can adjust.
      setSubmitting(false);
    }
  };

  const onCameraBindingChange = (name: string, value: string) => {
    setCameraBindings((prev) => ({ ...prev, [name]: value }));
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px] p-8 max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex justify-center items-center mb-4">
            <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
              <Play className="w-4 h-4 text-primary-foreground" />
            </div>
          </div>
          <DialogTitle className="text-center text-2xl font-bold">
            {t("landing.inference.title")}
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6 py-4">
          <DialogDescription className="text-base leading-relaxed text-center">
            {t("landing.inference.description")}
          </DialogDescription>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold border-b border-border pb-2">
              {t("landing.inference.robotSection")}
            </h3>
            {!robot ? (
              <Alert className="border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-200">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  {t("landing.inference.noRobot")}
                </AlertDescription>
              </Alert>
            ) : !robot.follower_ready ? (
              <Alert className="border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-200">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  {/* The robot name is DATA; {{gap}} is the localized
                      setup-gap predicate (robot.setupGap.* in the catalog). */}
                  <Trans
                    i18nKey="landing.inference.followerNotReady"
                    values={{
                      name: robot.name,
                      gap: formatRobotSetupGap(t, robot, "follower"),
                    }}
                    components={[<strong key="0" />]}
                  />
                </AlertDescription>
              </Alert>
            ) : (
              <div className="flex items-center gap-2 text-sm">
                <CheckCircle className="w-4 h-4 text-ok" />
                <span className="text-foreground">
                  {/* Two whole sentences rather than a translated suffix
                      concatenated onto a translated stem. */}
                  <Trans
                    i18nKey={
                      isBimanual
                        ? "landing.inference.runningOnBimanual"
                        : "landing.inference.runningOn"
                    }
                    values={{ name: robot.name }}
                    components={[<strong key="0" />]}
                  />
                </span>
              </div>
            )}
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold border-b border-border pb-2">
              {t("landing.inference.checkpointSection")}
            </h3>
            {checkpoints.length === 0 ? (
              <Alert className="border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-200">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  {t("landing.inference.noCheckpoints")}
                </AlertDescription>
              </Alert>
            ) : (
              <CheckpointDropdown
                checkpoints={checkpoints}
                // Single-job list: steps are unique here, so the step maps
                // 1:1 onto the checkpoint's identifying ref.
                selectedRef={
                  checkpoints.find((c) => c.step === selectedStep)?.ref ?? null
                }
                onChange={(c) => setSelectedStep(c.step)}
              />
            )}
            {robotCheckpointArmMismatch ? (
              <Alert className="border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-200">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  <Trans
                    i18nKey={
                      checkpointIsBimanual
                        ? "landing.inference.mismatchBimanual"
                        : "landing.inference.mismatchSingle"
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

          <div className="space-y-4">
            <h3 className="text-lg font-semibold border-b border-border pb-2">
              {t("landing.inference.paramsSection")}
            </h3>
            {policyConfig?.requires_task ? (
              <div className="space-y-2">
                <Label htmlFor="task" className="text-sm font-medium text-muted-foreground">
                  {t("landing.inference.taskLabel")}
                </Label>
                <Input
                  id="task"
                  value={task}
                  onChange={(e) => setTask(e.target.value)}
                  placeholder={t("landing.inference.taskPlaceholder")}
                  className=""
                />
                <p className="text-xs text-muted-foreground">
                  {/* The policy type is the checkpoint's own id — data. */}
                  {t("landing.inference.languageConditioned", {
                    // Renders empty when the checkpoint omits the type —
                    // exactly what the bare JSX interpolation did.
                    policyType: policyConfig.policy_type ?? "",
                  })}
                </p>
              </div>
            ) : null}
            <div className="space-y-2">
              <Label htmlFor="durationS" className="text-sm font-medium text-muted-foreground">
                {t("landing.inference.durationLabel")}
              </Label>
              <NumberInput
                id="durationS"
                min={1}
                value={durationS}
                onChange={(v) => {
                  if (v !== undefined) setDurationS(v);
                }}
                className=""
              />
            </div>
            <div className="space-y-2">
              <Label
                htmlFor="inference-engine"
                className="text-sm font-medium text-muted-foreground"
              >
                {t("landing.inference.engineLabel")}
              </Label>
              <Select
                value={inferenceEngine}
                onValueChange={(v) => setInferenceEngine(v as "sync" | "rtc")}
              >
                <SelectTrigger id="inference-engine">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {/* The submitted values stay "sync"/"rtc" — only the
                      labels are display text. */}
                  <SelectItem value="sync">
                    {t("landing.inference.engineSync")}
                  </SelectItem>
                  <SelectItem value="rtc">
                    {t("landing.inference.engineRtc")}
                  </SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {inferenceEngine === "rtc"
                  ? t("landing.inference.engineRtcHint")
                  : t("landing.inference.engineSyncHint")}
              </p>
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold border-b border-border pb-2">
              {t("landing.inference.camerasSection")}
            </h3>
            {policyConfigLoading ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="w-4 h-4 animate-spin" />
                {t("landing.inference.policyConfigLoading")}
              </div>
            ) : policyConfigError ? (
              <Alert className="border-destructive/40 bg-destructive/10 text-destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  {/* The message half is the raw error — kept verbatim. */}
                  {t("landing.inference.policyConfigError", {
                    message: policyConfigError,
                  })}
                </AlertDescription>
              </Alert>
            ) : !policyConfig ? null : cameraMap.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                {t("landing.inference.noCameras")}
              </p>
            ) : (
              <div className="space-y-3">
                <p className="text-xs text-muted-foreground">
                  {t("landing.inference.bindHint")}
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
                        <Label className="text-sm font-medium text-foreground">
                          {m.display}
                        </Label>
                        <p className="text-xs text-muted-foreground">
                          {t("landing.inference.capturesAt", {
                            width: dims.width,
                            height: dims.height,
                          })}
                        </p>
                        {boundCamera &&
                        (boundCamera.width !== dims.width ||
                          boundCamera.height !== dims.height) ? (
                          <p className="text-xs text-muted-foreground">
                            {t("landing.inference.robotCameraResolution", {
                              name: boundCamera.name,
                              width: boundCamera.width,
                              height: boundCamera.height,
                            })}
                          </p>
                        ) : null}
                        {boundCamera && !connected ? (
                          <p className="text-xs text-destructive">
                            {t("landing.inference.disconnected")}
                          </p>
                        ) : null}
                      </div>
                      <Select
                        value={value ?? undefined}
                        onValueChange={(v) => onCameraBindingChange(m.requestKey, v)}
                      >
                        <SelectTrigger className="w-56">
                          <SelectValue
                            placeholder={t("landing.inference.selectCamera")}
                          />
                        </SelectTrigger>
                        <SelectContent>
                          {robotCameras.length === 0 ? (
                            <div className="px-2 py-1.5 text-xs text-muted-foreground">
                              {t("landing.inference.noRobotCameras")}
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
                      {/* Backend preview at the exact cv2 index this role
                          binds to, so the tile can't disagree with the run. */}
                      <CameraThumbnail
                        cameraIndex={
                          boundCamera && connected
                            ? resolveCameraIndex(boundCamera, availableCameras)
                            : undefined
                        }
                        uniqueId={boundCamera?.unique_id}
                        paused={submitting}
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Advanced parameters — the shared AdvancedSection, same trigger and
              inner eyebrow/label/help-text rhythm as the Train form's
              AdvancedCard and the Deploy panel. ACT-only: temporal ensembling
              is an ACT config field, so for every other policy type the block
              has nothing to hold and stays hidden. ----------------------- */}
          {isAct ? (
            <AdvancedSection
              open={advancedOpen}
              onOpenChange={setAdvancedOpen}
              summary={t("landing.inference.advancedSummary")}
            >
              <div className="space-y-6">
                <section className="space-y-3">
                  <h4 className="eyebrow">
                    {t("landing.inference.actionSelection")}
                  </h4>
                  <div className="flex items-center gap-3">
                    <Switch
                      id="temporal-ensemble"
                      checked={temporalEnsemble}
                      onCheckedChange={setTemporalEnsemble}
                      className="data-[state=checked]:bg-primary"
                    />
                    <Label htmlFor="temporal-ensemble">
                      {t("landing.inference.temporalEnsemble")}
                    </Label>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {t("landing.inference.temporalEnsembleHint")}
                  </p>
                  {temporalEnsemble ? (
                    <div className="space-y-2">
                      <Label htmlFor="temporal-ensemble-coeff">
                        {t("landing.inference.coeffLabel")}
                      </Label>
                      <NumberInput
                        id="temporal-ensemble-coeff"
                        integer={false}
                        step="0.001"
                        min={0}
                        value={temporalEnsembleCoeff}
                        onChange={setTemporalEnsembleCoeff}
                        placeholder={t("landing.inference.coeffPlaceholder", {
                          coeff: DEFAULT_TEMPORAL_ENSEMBLE_COEFF,
                        })}
                        aria-invalid={temporalEnsembleInvalid}
                        className={cn(
                          "w-40",
                          temporalEnsembleInvalid && "border-destructive",
                        )}
                      />
                      {temporalEnsembleInvalid ? (
                        <p className="text-xs text-destructive">
                          {t("landing.inference.coeffInvalid")}
                        </p>
                      ) : (
                        <p className="text-xs text-muted-foreground">
                          {t("landing.inference.coeffHint", {
                            coeff: DEFAULT_TEMPORAL_ENSEMBLE_COEFF,
                          })}
                        </p>
                      )}
                    </div>
                  ) : null}
                </section>
              </div>
            </AdvancedSection>
          ) : null}

          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <Button
              onClick={handleStart}
              disabled={!canStart}
              className="w-full sm:w-auto px-10 py-6 text-lg disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Play className="w-5 h-5 mr-2" />
              {submitting
                ? t("landing.inference.starting")
                : t("landing.inference.start")}
            </Button>
            <Button
              onClick={() => onOpenChange(false)}
              variant="outline"
              className="w-full sm:w-auto px-10 py-6 text-lg"
            >
              {t("common.cancel")}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default InferenceModal;
