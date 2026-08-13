import React, { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Camera, Plus, Trash2, VideoOff, RefreshCw, ChevronRight } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useToast } from "@/hooks/use-toast";
import { useAvailableCameras } from "@/hooks/useAvailableCameras";
import BackendCameraStream from "@/components/BackendCameraStream";
import { isCameraConnected, resolveCameraIndex } from "@/lib/cameraResolve";

// Sentinels distinguish "leave unset" (auto-detect / platform default) from an
// explicit choice. Radix Select disallows an empty-string value, so we map these
// to `undefined` on the CameraConfig.
const FOURCC_AUTO = "__auto__";
const BACKEND_DEFAULT = "__default__";
const FOURCC_OPTIONS = ["MJPG", "YUYV", "I420", "NV12", "H264", "MP4V"];
// Mirrors lerobot's Cv2Backends enum names.
const BACKEND_OPTIONS = [
  "ANY",
  "V4L2",
  "DSHOW",
  "PVAPI",
  "ANDROID",
  "AVFOUNDATION",
  "MSMF",
];
// Common SO-101 rig placements, offered as one-click camera names before
// falling back to a free-text name via the CAMERA_NAME_CUSTOM sentinel.
const CAMERA_NAME_PRESETS = ["wrist", "top", "front", "side"];
const CAMERA_NAME_CUSTOM = "__custom_name__";

export interface CameraConfig {
  id: string;
  name: string;
  type: string;
  camera_index?: number; // cv2 index — what the recorder opens
  device_id: string; // Browser deviceId matched to the cv2 index by AVFoundation localizedName
  // Stable OS device identity (AVFoundation uniqueID). The authoritative link
  // to the physical camera: cv2 indices shift on replug and device names can
  // collide (two "KD-USB Cameras"), but this survives both.
  unique_id?: string;
  width: number;
  height: number;
  fps?: number;
  fourcc?: string; // 4-char OpenCV pixel format (e.g. "MJPG"); undefined = auto-detect
  backend?: string; // Cv2Backends name (e.g. "AVFOUNDATION"); undefined = platform default
}

interface CameraConfigurationProps {
  cameras: CameraConfig[];
  onCamerasChange: (cameras: CameraConfig[]) => void;
  releaseStreamsRef?: React.MutableRefObject<(() => void) | null>; // Ref to expose stream release function
}

const CameraConfiguration: React.FC<CameraConfigurationProps> = ({
  cameras,
  onCamerasChange,
  releaseStreamsRef,
}) => {
  const { toast } = useToast();

  // Recording start pauses the previews via releaseStreamsRef; gate camera
  // enumeration on the same flag so the getUserMedia/devicechange probing fully
  // stops before cv2 opens the devices. Otherwise the enumeration probe can
  // keep index 0 open and starve the recorder (OpenCVCamera(0) actual_fps=5.0).
  const [streamsPaused, setStreamsPaused] = useState(false);

  const {
    cameras: availableCameras,
    isLoading: isLoadingCameras,
    refresh: refreshCameras,
  } = useAvailableCameras({ enabled: !streamsPaused });
  const [selectedCameraIndex, setSelectedCameraIndex] = useState<string>("");
  const [cameraName, setCameraName] = useState("");
  // Tracks which name-picker option is active: "" (none yet), one of
  // CAMERA_NAME_PRESETS, or CAMERA_NAME_CUSTOM (free-text `cameraName`
  // input revealed below). Separate from `cameraName` itself so switching
  // back to a preset after typing a custom name doesn't require clearing it.
  const [nameChoice, setNameChoice] = useState<string>("");
  const handleNameChoice = (choice: string) => {
    setNameChoice(choice);
    if (choice !== CAMERA_NAME_CUSTOM) setCameraName(choice);
  };

  // The camera currently picked in the dropdown (not yet added). Drives the
  // immediate live preview shown before the camera is named.
  const selectedCamera = selectedCameraIndex
    ? availableCameras.find(
        (cam) => cam.index === parseInt(selectedCameraIndex)
      )
    : undefined;

  // cv2's AVFoundation order is uniqueID-sorted, so plugging/unplugging a
  // device between sessions shifts indices. Refresh each seeded camera's
  // camera_index by unique_id (exact physical identity) when the record has
  // one, falling back to the browser device_id for older records — otherwise
  // the recorder opens the wrong physical device and the dropdown's "already
  // added" check guards a stale index.
  //
  // device_id alone is a COIN FLIP when two cameras share a name (twin
  // "KD-USB Cameras"): the deviceId↔index pairing is decided by
  // enumerateDevices() order, which is unrelated to the uniqueID sort and not
  // stable across refreshes. Anchoring on unique_id is what stops this effect
  // from silently rewriting the recorder's index to the other camera.
  useEffect(() => {
    if (availableCameras.length === 0 || cameras.length === 0) return;
    let changed = false;
    const refreshed = cameras.map((cam) => {
      const match =
        (cam.unique_id
          ? availableCameras.find((m) => m.uniqueId === cam.unique_id)
          : undefined) ??
        (cam.device_id
          ? availableCameras.find((m) => m.deviceId === cam.device_id)
          : undefined);
      if (match && match.index !== cam.camera_index) {
        changed = true;
        return { ...cam, camera_index: match.index };
      }
      return cam;
    });
    if (changed) onCamerasChange(refreshed);
    // We deliberately don't depend on `cameras`/`onCamerasChange` to avoid
    // re-running every keystroke in the camera-name input — re-syncing only
    // when the available-cameras list itself changes is sufficient.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableCameras]);

  const addCamera = () => {
    if (!selectedCameraIndex || !cameraName.trim()) {
      toast({
        title: "Missing Information",
        description: !selectedCameraIndex
          ? "Select a camera first."
          : "Give this camera a name before adding it (e.g. workspace_cam).",
        variant: "destructive",
      });
      return;
    }

    const cameraIndex = parseInt(selectedCameraIndex);
    const selectedCamera = availableCameras.find(
      (cam) => cam.index === cameraIndex
    );

    if (!selectedCamera) {
      toast({
        title: "Invalid Camera",
        description: "Selected camera is not available.",
        variant: "destructive",
      });
      return;
    }

    // Block duplicates by unique_id, cv2 index, or browser deviceId — a stale
    // camera_index in a seeded camera can otherwise let the same physical
    // device sneak in under a different index. unique_id is checked first
    // because it's the only one that can't alias between twin cameras.
    const isDuplicate = cameras.some(
      (cam) =>
        (selectedCamera.uniqueId && cam.unique_id === selectedCamera.uniqueId) ||
        cam.camera_index === selectedCamera.index ||
        (selectedCamera.deviceId && cam.device_id === selectedCamera.deviceId),
    );
    if (isDuplicate) {
      toast({
        title: "Camera Already Added",
        description: "This camera is already in the configuration.",
        variant: "destructive",
      });
      return;
    }

    // Names must be unique within a robot: they key the session camera dict
    // server-side (dataset feature keys when recording, binding targets when
    // deploying), so two cameras sharing one would be ambiguous — the backend
    // now refuses to start such a record rather than silently dropping one of
    // them. Block it here, at the only place a name is ever assigned, so the
    // user can't save a robot that fails at start. Compared case-insensitively
    // because the inference auto-binder matches names that way.
    const nameTaken = cameras.some(
      (cam) =>
        cam.name.trim().toLowerCase() === cameraName.trim().toLowerCase(),
    );
    if (nameTaken) {
      toast({
        title: "Name Already Used",
        description: `Another camera on this robot is already named "${cameraName.trim()}". Pick a different name.`,
        variant: "destructive",
      });
      return;
    }

    const newCamera: CameraConfig = {
      id: `camera_${Date.now()}`,
      name: cameraName.trim(),
      type: "opencv",
      camera_index: selectedCamera.index,
      device_id: selectedCamera.deviceId,
      unique_id: selectedCamera.uniqueId,
      width: 640,
      height: 480,
      fps: 30,
    };

    onCamerasChange([...cameras, newCamera]);

    setSelectedCameraIndex("");
    setCameraName("");
    setNameChoice("");

    toast({
      title: "Camera Added",
      description: `${newCamera.name} has been added to the configuration.`,
    });
  };

  const removeCamera = (cameraId: string) => {
    onCamerasChange(cameras.filter((cam) => cam.id !== cameraId));
    toast({
      title: "Camera Removed",
      description: "Camera has been removed from the configuration.",
    });
  };

  const updateCamera = (cameraId: string, updates: Partial<CameraConfig>) => {
    onCamerasChange(
      cameras.map((cam) =>
        cam.id === cameraId ? { ...cam, ...updates } : cam
      )
    );
  };

  // When the recording session is starting, the parent calls
  // releaseStreamsRef.current() to make every CameraPreview drop its browser
  // stream so cv2.VideoCapture can grab the camera exclusively. Flipping
  // streamsPaused also disables useAvailableCameras above (see its comment).
  const releaseAllCameraStreams = useCallback(() => {
    setStreamsPaused(true);
  }, []);

  useEffect(() => {
    if (releaseStreamsRef) {
      releaseStreamsRef.current = releaseAllCameraStreams;
    }
  }, [releaseStreamsRef, releaseAllCameraStreams]);


  return (
    <div className="space-y-4">
      {/* Cameras is a repeater, not a single labelled control, so it keeps an
          eyebrow heading — the studio's one exception to the flat rule. */}
      <h3 className="eyebrow">Cameras</h3>

      {/* Add Camera Section */}
      <div className="bg-muted/50 rounded-lg p-4 space-y-4">
        <h4 className="text-sm font-medium text-foreground">Add a camera</h4>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-sm font-medium text-muted-foreground">
              Available cameras
            </Label>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => refreshCameras()}
              disabled={isLoadingCameras}
              className="h-6 w-6 text-muted-foreground hover:text-foreground"
              title="Rescan for cameras (e.g. after plugging in a new USB camera)"
              aria-label="Rescan for cameras"
            >
              <RefreshCw
                className={`w-3.5 h-3.5 ${isLoadingCameras ? "animate-spin" : ""}`}
              />
            </Button>
          </div>
          <Select
            value={selectedCameraIndex}
            onValueChange={setSelectedCameraIndex}
            disabled={isLoadingCameras}
          >
            <SelectTrigger className="bg-background border-border text-foreground">
              <SelectValue
                placeholder={
                  isLoadingCameras ? "Loading cameras..." : "Select camera"
                }
              />
            </SelectTrigger>
            <SelectContent className="bg-popover border-border">
              {availableCameras.map((camera) => {
                const alreadyAdded = cameras.some(
                  (cam) =>
                    cam.camera_index === camera.index ||
                    (camera.deviceId && cam.device_id === camera.deviceId),
                );
                return (
                  <SelectItem
                    key={camera.index}
                    value={camera.index.toString()}
                    className="text-foreground"
                    disabled={!camera.available || alreadyAdded}
                  >
                    <div className="flex flex-col">
                      <span className="font-medium">{camera.name}</span>
                      <span className="text-xs text-muted-foreground">
                        Index {camera.index}
                        {alreadyAdded && " · already added"}
                      </span>
                    </div>
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
        </div>

        {/* Live preview appears as soon as a camera is selected; naming +
            confirmation happens alongside it. */}
        {selectedCamera && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="bg-card rounded-lg border border-border overflow-hidden">
              <CameraStreamBox
                cameraIndex={selectedCamera.index}
                uniqueId={selectedCamera.uniqueId}
                paused={streamsPaused}
              />
            </div>

            <div className="flex flex-col justify-center gap-4">
              <div className="space-y-2">
                <Label className="text-sm font-medium text-muted-foreground">
                  Camera Name <span className="text-warn">*</span>
                </Label>
                <Select value={nameChoice} onValueChange={handleNameChoice}>
                  <SelectTrigger className="bg-background border-border text-foreground">
                    <SelectValue placeholder="Select a name" />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    {CAMERA_NAME_PRESETS.map((name) => {
                      // Same treatment the camera dropdown gives an
                      // already-added device: shown, labelled, not selectable.
                      const nameTaken = cameras.some(
                        (cam) => cam.name.trim().toLowerCase() === name,
                      );
                      return (
                        <SelectItem
                          key={name}
                          value={name}
                          className="text-foreground"
                          disabled={nameTaken}
                        >
                          {name}
                          {nameTaken && " · already used"}
                        </SelectItem>
                      );
                    })}
                    <SelectItem
                      value={CAMERA_NAME_CUSTOM}
                      className="text-foreground"
                    >
                      Custom name…
                    </SelectItem>
                  </SelectContent>
                </Select>
                {nameChoice === CAMERA_NAME_CUSTOM && (
                  <Input
                    value={cameraName}
                    onChange={(e) => setCameraName(e.target.value)}
                    placeholder="e.g., workspace_cam"
                    autoFocus
                    className="bg-background border-border text-foreground"
                  />
                )}
              </div>

              {/* Deliberately NOT disabled when the name is missing: a dead
                  button can't explain itself, so clicking runs addCamera's
                  validation and its toast says what's missing. */}
              <Button
                onClick={addCamera}
                className="bg-primary text-primary-foreground hover:bg-primary/90"
              >
                <Plus className="w-4 h-4 mr-2" />
                Add camera
              </Button>
              {!cameraName.trim() && (
                <p className="text-xs text-muted-foreground">
                  Name this camera to add it.
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Configured Cameras */}
      {cameras.length > 0 && (
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-foreground">
            Configured cameras ({cameras.length})
          </h4>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-4">
            {cameras.map((camera) => (
              <CameraPreview
                key={camera.id}
                camera={camera}
                connected={isCameraConnected(camera, availableCameras)}
                paused={streamsPaused}
                onRemove={() => removeCamera(camera.id)}
                onUpdate={(updates) => updateCamera(camera.id, updates)}
              />
            ))}
          </div>
        </div>
      )}

      {cameras.length === 0 && (
        <div className="text-center py-8 text-muted-foreground">
          <Camera className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
          <p>No cameras configured. Add a camera to get started.</p>
        </div>
      )}
    </div>
  );
};

interface CameraStreamBoxProps {
  cameraIndex?: number;
  uniqueId?: string;
  paused: boolean;
  /** Shown when there's no index to stream. Distinguishes "nothing picked yet"
   * (dropdown preview) from "this configured camera is gone" (camera card). */
  emptyLabel?: string;
}

/** Live preview for a camera. Used both for the pre-add preview (as soon as
 * a camera is picked in the dropdown) and for each configured camera's card.
 *
 * Streams from the BACKEND by cv2 index (GET /camera-preview/{index}), not via
 * getUserMedia. The browser identifies cameras by deviceId, which the frontend
 * could only map to a cv2 index by localizedName — and two cameras of the same
 * model share that name, so the mapping was a coin flip that swapped between
 * refreshes. Streaming by index shows exactly the device the recorder will
 * open, and is the only preview that works when MakerMods Lab runs on a headless
 * host. `uniqueId` re-anchors the index server-side across replugs.
 *
 * Pausing (recording start / modal close) unmounts the stream so cv2 can grab
 * the device; the backend also force-releases previews on the record path. */
const CameraStreamBox: React.FC<CameraStreamBoxProps> = ({
  cameraIndex,
  uniqueId,
  paused,
  emptyLabel = "No camera selected",
}) => {
  const showStream = !paused && cameraIndex !== undefined;
  return (
    <div className="aspect-[4/3] bg-muted relative">
      {showStream ? (
        <BackendCameraStream
          cameraIndex={cameraIndex}
          uniqueId={uniqueId}
          className="w-full h-full object-cover"
        />
      ) : (
        <div className="w-full h-full flex flex-col items-center justify-center">
          <VideoOff className="w-8 h-8 text-muted-foreground mb-2" />
          <span className="text-muted-foreground text-sm">
            {paused ? "Preview paused" : emptyLabel}
          </span>
        </div>
      )}
    </div>
  );
};

interface CameraPreviewProps {
  camera: CameraConfig;
  /** False when this camera's unique_id is verifiably absent from the current
   * enumeration — the tile must show "disconnected", never a wrong device: a
   * stale camera_index now points at whatever took its place. */
  connected: boolean;
  paused: boolean;
  onRemove: () => void;
  onUpdate: (updates: Partial<CameraConfig>) => void;
}

const CameraPreview: React.FC<CameraPreviewProps> = ({
  camera,
  connected,
  paused,
  onRemove,
  onUpdate,
}) => {
  return (
    <div className="bg-card rounded-lg border border-border overflow-hidden">
      <CameraStreamBox
        cameraIndex={connected ? camera.camera_index : undefined}
        uniqueId={camera.unique_id}
        paused={paused}
        emptyLabel="Camera disconnected — reconnect it or rescan"
      />

      {/* Camera Info */}
      <div className="p-3 space-y-2">
        <div className="flex items-center justify-between">
          <h5 className="font-medium text-foreground truncate">{camera.name}</h5>
          <Button
            onClick={onRemove}
            size="sm"
            variant="ghost"
            className="text-destructive hover:text-destructive hover:bg-destructive/10 p-1"
            aria-label="Remove camera"
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>

        <Collapsible>
          <CollapsibleTrigger className="group flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors">
            <ChevronRight className="w-3.5 h-3.5 transition-transform group-data-[state=open]:rotate-90" />
            Configuration
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-2 space-y-2">
            <div className="grid grid-cols-1 gap-2 text-xs text-muted-foreground">
              <div className="flex items-center gap-2">
                <span className="w-16">Resolution:</span>
                <div className="flex items-center gap-1">
                  <NumberInput
                    value={camera.width}
                    onChange={(v) => {
                      if (v !== undefined) onUpdate({ width: v });
                    }}
                    className="bg-background border-border text-foreground text-xs h-6 px-2 w-16"
                    min="320"
                    max="1920"
                  />
                  <span className="flex items-center">×</span>
                  <NumberInput
                    value={camera.height}
                    onChange={(v) => {
                      if (v !== undefined) onUpdate({ height: v });
                    }}
                    className="bg-background border-border text-foreground text-xs h-6 px-2 w-16"
                    min="240"
                    max="1080"
                  />
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16">FPS:</span>
                <NumberInput
                  value={camera.fps ?? 30}
                  onChange={(v) => {
                    if (v !== undefined) onUpdate({ fps: v });
                  }}
                  className="bg-background border-border text-foreground text-xs h-6 px-2 w-16"
                  min="10"
                  max="60"
                />
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16">FOURCC:</span>
                <Select
                  value={camera.fourcc ?? FOURCC_AUTO}
                  onValueChange={(v) =>
                    onUpdate({ fourcc: v === FOURCC_AUTO ? undefined : v })
                  }
                >
                  <SelectTrigger className="bg-background border-border text-foreground text-xs h-6 px-2 w-28">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    <SelectItem
                      value={FOURCC_AUTO}
                      className="text-foreground text-xs"
                    >
                      Auto
                    </SelectItem>
                    {FOURCC_OPTIONS.map((code) => (
                      <SelectItem
                        key={code}
                        value={code}
                        className="text-foreground text-xs"
                      >
                        {code}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16">Backend:</span>
                <Select
                  value={camera.backend ?? BACKEND_DEFAULT}
                  onValueChange={(v) =>
                    onUpdate({ backend: v === BACKEND_DEFAULT ? undefined : v })
                  }
                >
                  <SelectTrigger className="bg-background border-border text-foreground text-xs h-6 px-2 w-28">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-popover border-border">
                    <SelectItem
                      value={BACKEND_DEFAULT}
                      className="text-foreground text-xs"
                    >
                      Default
                    </SelectItem>
                    {BACKEND_OPTIONS.map((name) => (
                      <SelectItem
                        key={name}
                        value={name}
                        className="text-foreground text-xs"
                      >
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <p className="text-[10px] text-muted-foreground leading-tight">
                Overriding the backend can reorder camera indices on macOS.
              </p>
            </div>
            <div className="text-xs text-muted-foreground">
              Type: {camera.type} | Device:{" "}
              {camera.device_id?.substring(0, 10)}...
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>
    </div>
  );
};

interface SessionCameraListProps {
  /** The selected robot record's cameras, exactly as stored. */
  cameras: CameraConfig[];
  /** Filled with a function that drops every preview stream, so the caller can
   * hand the devices to cv2 before a session starts (same contract as
   * CameraConfiguration's prop of the same name). */
  releaseStreamsRef?: React.MutableRefObject<(() => void) | null>;
  /** Shown when the robot has no cameras. */
  emptyLabel?: string;
}

/**
 * READ-ONLY view of the cameras a session will open — name, settings, and a
 * live preview per camera, with no add/remove/edit controls.
 *
 * Sessions no longer carry their own camera set: the backend resolves the
 * cameras from the robot record named in the start request (see
 * makermodslab/utils/config.py's load_robot_cameras / bind_robot_cameras). A
 * per-session editable copy could only ever diverge from what actually runs —
 * edits made here were silently discarded, and the panel could show a set the
 * server would not use. Cameras are therefore edited in exactly one place, the
 * robot settings dialog (which still renders the full CameraConfiguration
 * above), and shown here only to confirm what is about to be recorded.
 *
 * Previews stream from the backend by cv2 index, re-anchored to each camera's
 * unique_id against the live enumeration — a stored index goes stale on
 * replug, and showing the wrong device is exactly the confusion this list
 * exists to prevent.
 */
export const SessionCameraList: React.FC<SessionCameraListProps> = ({
  cameras,
  releaseStreamsRef,
  emptyLabel = "No cameras on this robot.",
}) => {
  // Same handover as the editable component: pausing unmounts the streams AND
  // stops the enumeration probe, so cv2 can open the devices exclusively.
  const [streamsPaused, setStreamsPaused] = useState(false);
  const { cameras: availableCameras } = useAvailableCameras({
    enabled: !streamsPaused,
  });

  const releaseAllCameraStreams = useCallback(() => setStreamsPaused(true), []);
  useEffect(() => {
    if (releaseStreamsRef) {
      releaseStreamsRef.current = releaseAllCameraStreams;
    }
  }, [releaseStreamsRef, releaseAllCameraStreams]);

  return (
    <div className="space-y-4">
      {/* Cameras is a repeater, not a single labelled control, so it keeps an
          eyebrow heading — matching the editable component. */}
      <h3 className="eyebrow">Cameras</h3>
      <p className="text-xs text-muted-foreground">
        Cameras come from the selected robot. Add, remove, or adjust them in
        Robot settings.
      </p>

      {cameras.length === 0 ? (
        <div className="py-6 text-center text-muted-foreground">
          <Camera className="mx-auto mb-3 h-10 w-10 text-muted-foreground" />
          <p className="text-sm">{emptyLabel}</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {cameras.map((camera) => {
            const connected = isCameraConnected(camera, availableCameras);
            return (
              <div
                key={camera.id ?? camera.name}
                className="overflow-hidden rounded-lg border border-border bg-card"
              >
                <CameraStreamBox
                  cameraIndex={
                    connected
                      ? resolveCameraIndex(camera, availableCameras)
                      : undefined
                  }
                  uniqueId={camera.unique_id}
                  paused={streamsPaused}
                  emptyLabel="Camera disconnected — reconnect it or check Robot settings"
                />
                <div className="space-y-0.5 p-3">
                  <h5 className="truncate font-medium text-foreground">
                    {camera.name}
                  </h5>
                  <p className="text-xs text-muted-foreground">
                    {camera.width}×{camera.height}
                    {camera.fps ? ` · ${camera.fps} fps` : ""}
                    {camera.fourcc ? ` · ${camera.fourcc}` : ""}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default CameraConfiguration;
