import React, { useState, useEffect, useCallback } from "react";
import { useTranslation } from "react-i18next";
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
import { useAvailableCameras, type AvailableCamera } from "@/hooks/useAvailableCameras";
import BackendCameraStream from "@/components/BackendCameraStream";
import {
  isCameraConnected,
  isSameCamera,
  resolveCameraIndex,
} from "@/lib/cameraResolve";
import { useEyebrowClass } from "@/hooks/useEyebrowClass";

// Sentinels distinguish "leave unset" (auto-detect / platform default) from an
// explicit choice. Radix Select disallows an empty-string value, so we map these
// to `undefined` on the CameraConfig.
const FOURCC_AUTO = "__auto__";
const BACKEND_DEFAULT = "__default__";
// DATA, never translated: OpenCV FOURCC codes, submitted to the backend
// verbatim as the camera's pixel format.
const FOURCC_OPTIONS = ["MJPG", "YUYV", "I420", "NV12", "H264", "MP4V"];
// Mirrors lerobot's Cv2Backends enum names. Also DATA: the option value IS the
// enum name the backend resolves, so these labels stay untranslated.
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
//
// DELIBERATELY NOT TRANSLATED: each entry is simultaneously the button label
// AND the camera name written into the robot record and sent to the backend
// (it keys the session camera dict, and becomes a dataset feature key).
// Localizing the label would store a Chinese camera name in the config — data,
// not copy. Same reasoning for the sentinels around it.
const CAMERA_NAME_PRESETS = ["wrist", "top", "front", "side"];
const CAMERA_NAME_CUSTOM = "__custom_name__";

export interface CameraConfig {
  id: string;
  name: string;
  type: string;
  camera_index?: number; // cv2 index — what the recorder opens
  device_id: string; // Browser deviceId matched to the cv2 index by AVFoundation localizedName
  // OS device identity (AVFoundation uniqueID). The best link to the physical
  // camera we have — cv2 indices shift on replug and device names collide (two
  // "KD-USB Cameras") — but NOT a device serial, and it does NOT survive a
  // replug into a different port.
  //
  // Measured on the SO-101 rig (2026-09-01): the id is the USB **locationID**
  // with a per-model constant appended, so it encodes (model, topology
  // position), not the unit. locationID 0x132200 -> "0x1322002c7f4a60";
  // 0x1130000 -> "0x11300002c7f4a60". Move a camera to another port, or let a
  // bus-powered hub enumerate its ports in a different order across a power
  // cycle, and the id changes for the same physical device.
  //
  // A USB serial would be the stable anchor, but these cameras don't have one:
  // all three report `USB Serial Number = "KD-USB Cameras"`, so keying on it
  // would collide every unit into a single identity. Don't "fix" this by
  // switching to serials without re-checking the hardware.
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
  const { t } = useTranslation();
  const eyebrow = useEyebrowClass();

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
  // camera_index by unique_id when the record has one, falling back to the
  // browser device_id for older records — otherwise the recorder opens the
  // wrong physical device and the "already added" checks guard a stale index.
  //
  // device_id alone is a COIN FLIP when two cameras share a name (twin
  // "KD-USB Cameras"): the deviceId↔index pairing is decided by
  // enumerateDevices() order, which is unrelated to the uniqueID sort and not
  // stable across refreshes. Anchoring on unique_id is what stops this effect
  // from silently rewriting the recorder's index to the other camera.
  //
  // Whatever matched, ALL THREE identifiers are written back, not just the
  // index. Both weaker ids decay — a browser deviceId rotates when site data is
  // cleared, and unique_id tracks the USB port (see CameraConfig.unique_id) —
  // and a record only heals while something still matches. Refreshing them at
  // the moment of a confirmed match is the one chance to do it; leaving a
  // stale id behind poisons every later comparison for the life of the record.
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
      if (!match) return cam;
      // Only write ids the enumeration actually reported: off macOS uniqueId is
      // absent, and deviceId is "" when no browser device matched the label.
      // Clobbering a good saved id with an empty one would lose the anchor.
      const healed = {
        ...cam,
        camera_index: match.index,
        ...(match.uniqueId ? { unique_id: match.uniqueId } : {}),
        ...(match.deviceId ? { device_id: match.deviceId } : {}),
      };
      if (
        healed.camera_index !== cam.camera_index ||
        healed.unique_id !== cam.unique_id ||
        healed.device_id !== cam.device_id
      ) {
        changed = true;
        return healed;
      }
      return cam;
    });
    if (changed) onCamerasChange(refreshed);
    // `cameras` IS a dependency: the saved record is fetched, so it usually
    // lands a tick AFTER the enumeration has settled. Keying only on
    // `availableCameras` meant the effect had already run (and bailed on the
    // empty record) by the time the cameras arrived, and never re-ran — so the
    // dialog spent its whole life comparing against indices stale from disk.
    // Re-running is safe and converges: the body is a no-op unless an index
    // actually differs, so the state update it triggers settles on the next
    // pass. `onCamerasChange` stays out — callers pass an inline lambda, and
    // depending on it would re-fire this effect on every parent render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableCameras, cameras]);

  const addCamera = () => {
    if (!selectedCameraIndex || !cameraName.trim()) {
      toast({
        title: t("recording.cameras.toast.missingInfoTitle"),
        description: !selectedCameraIndex
          ? t("recording.cameras.toast.selectCameraFirst")
          : t("recording.cameras.toast.nameCameraFirst"),
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
        title: t("recording.cameras.toast.invalidTitle"),
        description: t("recording.cameras.toast.invalidBody"),
        variant: "destructive",
      });
      return;
    }

    const isDuplicate = cameras.some((cam) => isSameCamera(cam, selectedCamera));
    if (isDuplicate) {
      toast({
        title: t("recording.cameras.toast.duplicateTitle"),
        description: t("recording.cameras.toast.duplicateBody"),
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
        title: t("recording.cameras.toast.nameTakenTitle"),
        description: t("recording.cameras.toast.nameTakenBody", {
          name: cameraName.trim(),
        }),
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
      title: t("recording.cameras.toast.addedTitle"),
      // The camera's own name is data — echoed back verbatim.
      description: t("recording.cameras.toast.addedBody", {
        name: newCamera.name,
      }),
    });
  };

  const removeCamera = (cameraId: string) => {
    onCamerasChange(cameras.filter((cam) => cam.id !== cameraId));
    toast({
      title: t("recording.cameras.toast.removedTitle"),
      description: t("recording.cameras.toast.removedBody"),
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
      <h3 className={eyebrow}>{t("recording.cameras.heading")}</h3>

      {/* Add Camera Section */}
      <div className="bg-muted/50 rounded-lg p-4 space-y-4">
        <h4 className="text-sm font-medium text-foreground">
          {t("recording.cameras.addTitle")}
        </h4>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-sm font-medium text-muted-foreground">
              {t("recording.cameras.availableLabel")}
            </Label>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => refreshCameras()}
              disabled={isLoadingCameras}
              className="h-6 w-6 text-muted-foreground hover:text-foreground"
              title={t("recording.cameras.rescanTooltip")}
              aria-label={t("recording.cameras.rescanLabel")}
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
                  isLoadingCameras
                    ? t("recording.cameras.loadingPlaceholder")
                    : t("recording.cameras.selectPlaceholder")
                }
              />
            </SelectTrigger>
            <SelectContent className="bg-popover border-border">
              {availableCameras.map((camera) => {
                // Exactly the predicate Add enforces. These used to differ —
                // the dropdown omitted the unique_id clause — so a row could
                // pass the picker and then be refused by the Add button.
                const alreadyAdded = cameras.some((cam) =>
                  isSameCamera(cam, camera),
                );
                return (
                  <SelectItem
                    key={camera.index}
                    value={camera.index.toString()}
                    className="text-foreground"
                    disabled={!camera.available || alreadyAdded}
                  >
                    <div className="flex flex-col">
                      {/* The device name comes from the OS enumeration — data. */}
                      <span className="font-medium">{camera.name}</span>
                      <span className="text-xs text-muted-foreground">
                        {t("recording.cameras.indexLabel", {
                          index: camera.index,
                        })}
                        {alreadyAdded &&
                          t("recording.cameras.alreadyAddedSuffix")}
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
                  {t("recording.cameras.nameLabel")}{" "}
                  <span className="text-warn">*</span>
                </Label>
                <Select value={nameChoice} onValueChange={handleNameChoice}>
                  <SelectTrigger className="bg-background border-border text-foreground">
                    <SelectValue
                      placeholder={t("recording.cameras.namePlaceholder")}
                    />
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
                          {/* `name` is the stored camera name, not a label —
                              see CAMERA_NAME_PRESETS. Never translated. */}
                          {name}
                          {nameTaken && t("recording.cameras.alreadyUsedSuffix")}
                        </SelectItem>
                      );
                    })}
                    <SelectItem
                      value={CAMERA_NAME_CUSTOM}
                      className="text-foreground"
                    >
                      {t("recording.cameras.customNameOption")}
                    </SelectItem>
                  </SelectContent>
                </Select>
                {nameChoice === CAMERA_NAME_CUSTOM && (
                  <Input
                    value={cameraName}
                    onChange={(e) => setCameraName(e.target.value)}
                    placeholder={t("recording.cameras.customNamePlaceholder")}
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
                {t("recording.cameras.addButton")}
              </Button>
              {!cameraName.trim() && (
                <p className="text-xs text-muted-foreground">
                  {t("recording.cameras.nameRequiredHint")}
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
            {t("recording.cameras.configuredTitle", {
              total: cameras.length,
            })}
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
          <p>{t("recording.cameras.emptyState")}</p>
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
  emptyLabel,
}) => {
  const { t } = useTranslation();
  const showStream = !paused && cameraIndex !== undefined;
  // Default resolved here rather than as a parameter default so it tracks the
  // live language instead of freezing whatever loaded first.
  const emptyText = emptyLabel ?? t("recording.cameras.noneSelected");
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
            {paused ? t("recording.cameras.previewPaused") : emptyText}
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
  const { t } = useTranslation();
  return (
    <div className="bg-card rounded-lg border border-border overflow-hidden">
      <CameraStreamBox
        cameraIndex={connected ? camera.camera_index : undefined}
        uniqueId={camera.unique_id}
        paused={paused}
        emptyLabel={t("recording.cameras.disconnected")}
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
            aria-label={t("recording.cameras.removeLabel")}
          >
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>

        <Collapsible>
          <CollapsibleTrigger className="group flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors">
            <ChevronRight className="w-3.5 h-3.5 transition-transform group-data-[state=open]:rotate-90" />
            {t("recording.cameras.configurationToggle")}
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-2 space-y-2">
            <div className="grid grid-cols-1 gap-2 text-xs text-muted-foreground">
              <div className="flex items-center gap-2">
                <span className="w-16">
                  {t("recording.cameras.resolutionLabel")}
                </span>
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
                <span className="w-16">{t("recording.cameras.fpsLabel")}</span>
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
                <span className="w-16">
                  {t("recording.cameras.fourccLabel")}
                </span>
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
                      {t("recording.cameras.fourccAuto")}
                    </SelectItem>
                    {/* The codes themselves are submitted verbatim — data. */}
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
                <span className="w-16">
                  {t("recording.cameras.backendLabel")}
                </span>
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
                      {t("recording.cameras.backendDefault")}
                    </SelectItem>
                    {/* Cv2Backends enum names — data, never translated. */}
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
                {t("recording.cameras.backendWarning")}
              </p>
            </div>
            <div className="text-xs text-muted-foreground">
              {/* Driver id and deviceId are both data. */}
              {t("recording.cameras.deviceInfo", {
                type: camera.type,
                device: camera.device_id?.substring(0, 10),
              })}
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
  /** Drive the pause from outside instead of through releaseStreamsRef, for a
   * caller whose "hand the devices over" state is derived rather than an event
   * (the Run panel pauses while a rollout is submitting or active, and resumes
   * on its own when it ends — the ref is one-way and would need a remount).
   * Left undefined, the list keeps its own state and nothing changes. */
  paused?: boolean;
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
  emptyLabel,
  paused,
}) => {
  const { t } = useTranslation();
  const eyebrow = useEyebrowClass();
  // Resolved per render (not as a parameter default) so it follows a language
  // switch. Callers that pass their own label still win.
  const emptyText = emptyLabel ?? t("recording.cameras.sessionEmpty");
  // Same handover as the editable component: pausing unmounts the streams AND
  // stops the enumeration probe, so cv2 can open the devices exclusively.
  const [streamsPaused, setStreamsPaused] = useState(false);
  const isPaused = paused ?? streamsPaused;
  const { cameras: availableCameras } = useAvailableCameras({
    enabled: !isPaused,
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
      <h3 className={eyebrow}>{t("recording.cameras.heading")}</h3>
      <p className="text-xs text-muted-foreground">
        {t("recording.cameras.sessionHint")}
      </p>

      {cameras.length === 0 ? (
        <div className="py-6 text-center text-muted-foreground">
          <Camera className="mx-auto mb-3 h-10 w-10 text-muted-foreground" />
          <p className="text-sm">{emptyText}</p>
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
                  paused={isPaused}
                  emptyLabel={t("recording.cameras.disconnectedSettings")}
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
