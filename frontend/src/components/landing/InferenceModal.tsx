import React, { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
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
import { RobotRecord, robotSetupGap } from "@/hooks/useRobots";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useInferenceSession } from "@/contexts/InferenceSessionContext";
import {
  JobCheckpoint,
  PolicyConfigSummary,
  getCheckpointPolicyConfig,
  listJobCheckpoints,
} from "@/lib/checkpointsApi";
import { startInference } from "@/lib/inferenceApi";
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
  if (paused || cameraIndex === undefined) {
    return (
      <div className="w-32 h-24 bg-muted rounded border border-border flex flex-col items-center justify-center">
        <VideoOff className="w-5 h-5 text-muted-foreground mb-1" />
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
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { openInferenceSession } = useInferenceSession();

  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(initialStep);
  const [task, setTask] = useState("");
  const [durationS, setDurationS] = useState(60);
  // Inference engine A/B. "sync" is the server default and the historical
  // behaviour; "rtc" is experimental (see StartInferenceRequest).
  const [inferenceEngine, setInferenceEngine] = useState<"sync" | "rtc">("sync");
  const [submitting, setSubmitting] = useState(false);

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
  // bimanual-trained SO-101 checkpoint carries a 12-dim state/action (two 6-DOF
  // arms) and left_/right_-prefixed camera names; a single-arm checkpoint is
  // 6-dim. Running a policy on the wrong arm count crashes on a shape mismatch
  // deep in the rollout subprocess. Detect it here from the checkpoint's state
  // dim (fall back to action dim) and explain it before Start. This is the
  // client mirror of the server's `_arm_count_mismatch` 409 guard — we forward
  // `checkpoint_state_dim` so the server enforces the same rule authoritatively.
  const SO101_DOF = 6;
  const checkpointDim = policyConfig?.state_dim ?? policyConfig?.action_dim ?? null;
  const checkpointArms =
    checkpointDim != null && checkpointDim % SO101_DOF === 0
      ? checkpointDim / SO101_DOF
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
  const canStart =
    !!robot &&
    robot.follower_ready &&
    !robotCheckpointArmMismatch &&
    selectedRef != null &&
    !!policyConfig &&
    allCamerasBound &&
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
      // The POST now returns immediately (it only validates cheaply, then the
      // server downloads the model + preflights the arm in the background), so
      // this opens the inference dialog right away — the download and its
      // progress, any warn-but-allow arm finding, and any failure all surface
      // there via /inference-status polling.
      await startInference(baseUrl, fetchWithHeaders, {
        follower_port: robot.follower_port,
        follower_config: robot.follower_config,
        policy_ref: selectedRef,
        task,
        camera_bindings: cameraBindingPayload,
        camera_dims: cameraDimsPayload,
        duration_s: durationS,
        // Bimanual: forward the mode + right-arm follower so the server builds a
        // `bi_so_follower` command staging both follower calibrations. In single
        // mode the right_* fields are inert (mode defaults to "single"
        // server-side). robot_name is the BiSO staging base id.
        mode: robot.mode,
        right_follower_port: robot.right_follower_port,
        right_follower_config: robot.right_follower_config,
        robot_name: robot.name,
        // Forward the checkpoint's flat state width so the server enforces the
        // same arm-count guard authoritatively (null when the checkpoint omits
        // observation.state — the server then defers to its shape check).
        checkpoint_state_dim: policyConfig.state_dim ?? undefined,
        inference_engine: inferenceEngine,
      });
      onOpenChange(false);
      openInferenceSession();
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
            Configure Inference
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6 py-4">
          <DialogDescription className="text-base leading-relaxed text-center">
            Pick a checkpoint and confirm hardware. The selected policy will
            drive the follower autonomously for the configured duration.
          </DialogDescription>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold border-b border-border pb-2">
              Robot Configuration
            </h3>
            {!robot ? (
              <Alert className="border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-200">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  Select and configure a robot on the Landing page first.
                </AlertDescription>
              </Alert>
            ) : !robot.follower_ready ? (
              <Alert className="border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-200">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  <strong>{robot.name}</strong> {robotSetupGap(robot, "follower")}.
                  Open Robot settings before running inference. (Inference only
                  uses the follower arm — leader setup isn't needed.)
                </AlertDescription>
              </Alert>
            ) : (
              <div className="flex items-center gap-2 text-sm">
                <CheckCircle className="w-4 h-4 text-ok" />
                <span className="text-foreground">
                  Running on <strong>{robot.name}</strong>
                  {isBimanual ? " (bimanual — both followers)" : ""}
                </span>
              </div>
            )}
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold border-b border-border pb-2">
              Checkpoint
            </h3>
            {checkpoints.length === 0 ? (
              <Alert className="border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-200">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  No checkpoints available for this job yet.
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
                  {checkpointIsBimanual ? (
                    <>
                      This checkpoint was trained on a{" "}
                      <strong>bimanual robot</strong> ({checkpointDim}-dim state,{" "}
                      {checkpointArms} arms), but <strong>{robot?.name}</strong>{" "}
                      is a single-arm robot. Pick a single-arm checkpoint, or
                      select a bimanual robot on the Landing page.
                    </>
                  ) : (
                    <>
                      This checkpoint was trained on a{" "}
                      <strong>single-arm robot</strong> ({checkpointDim}-dim
                      state), but <strong>{robot?.name}</strong> is a bimanual
                      robot. Pick a bimanual checkpoint, or select a single-arm
                      robot on the Landing page.
                    </>
                  )}
                </AlertDescription>
              </Alert>
            ) : null}
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold border-b border-border pb-2">
              Run parameters
            </h3>
            {policyConfig?.requires_task ? (
              <div className="space-y-2">
                <Label htmlFor="task" className="text-sm font-medium text-muted-foreground">
                  Task description
                </Label>
                <Input
                  id="task"
                  value={task}
                  onChange={(e) => setTask(e.target.value)}
                  placeholder="e.g., pick up the red block"
                  className=""
                />
                <p className="text-xs text-muted-foreground">
                  This policy is language-conditioned ({policyConfig.policy_type}).
                </p>
              </div>
            ) : null}
            <div className="space-y-2">
              <Label htmlFor="durationS" className="text-sm font-medium text-muted-foreground">
                Max duration (seconds)
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
                Inference engine
              </Label>
              <Select
                value={inferenceEngine}
                onValueChange={(v) => setInferenceEngine(v as "sync" | "rtc")}
              >
                <SelectTrigger id="inference-engine">
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
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold border-b border-border pb-2">
              Cameras
            </h3>
            {policyConfigLoading ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="w-4 h-4 animate-spin" />
                Reading policy config…
              </div>
            ) : policyConfigError ? (
              <Alert className="border-destructive/40 bg-destructive/10 text-destructive">
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
                        <Label className="text-sm font-medium text-foreground">
                          {m.display}
                        </Label>
                        <p className="text-xs text-muted-foreground">
                          Captures at {dims.width}×{dims.height} — the policy's
                          resolution
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
                        <SelectTrigger className="w-56">
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

          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <Button
              onClick={handleStart}
              disabled={!canStart}
              className="w-full sm:w-auto px-10 py-6 text-lg disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Play className="w-5 h-5 mr-2" />
              {submitting ? "Starting…" : "Start Inference"}
            </Button>
            <Button
              onClick={() => onOpenChange(false)}
              variant="outline"
              className="w-full sm:w-auto px-10 py-6 text-lg"
            >
              Cancel
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default InferenceModal;
