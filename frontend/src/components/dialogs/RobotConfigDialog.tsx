import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Switch } from "@/components/ui/switch";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Activity,
  CheckCircle,
  AlertCircle,
  AlertTriangle,
  ChevronDown,
  Loader2,
  Play,
  Square,
  Circle,
  Camera,
  ShieldQuestion,
  Hand,
  MoveHorizontal,
  RefreshCw,
  ScanSearch,
  Wand2,
  Trash2,
  FolderOpen,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useSessionHeartbeat } from "@/hooks/useSessionHeartbeat";
import { useUnloadWarning } from "@/hooks/useUnloadWarning";
import { ApiError } from "@/lib/apiClient";
import {
  startSession,
  stopSession,
  getCurrentSession,
  formatSessionHeld,
} from "@/lib/sessionApi";
import { tabOwnerId } from "@/lib/sessionOwner";
import { isMotorRangeComplete } from "@/lib/calibrationTargets";
import { calibrationErrorMessage } from "@/lib/calibrationError";
// Followers use the same family-specific product photos as the "Create a new
// robot" arm cards. Both CAN families use the same physical Star Arm 102
// leader, so its calibration gets one dedicated, shared zero-pose reference.
import makerArmPhoto from "@/assets/arms/maker.jpg";
import metalArmPhoto from "@/assets/arms/metal.jpg";
import starArm102LeaderZeroPose from "@/assets/calibration/star-arm-102-leader-zero-pose.jpg";

/**
 * Reference photos for the zero-pose calibration, keyed by manifest id — the
 * one place a built-in arm id may appear in this file. Followers get one
 * photo per family because their arm zero poses differ. The Star Arm 102 leader is identical on both
 * built-in rigs, so their leader rows share one photographed reference
 * (`ZERO_POSE_LEADER_IMAGE`). A family without an entry (an extension's)
 * shows whatever image its manifest summary / step serves, or its text alone.
 */
const ZERO_POSE_IMAGES: Record<
  string,
  { follower: string; altKey: "poseImage" | "poseImageMetal"; leader: string }
> = {
  maker: {
    follower: makerArmPhoto,
    altKey: "poseImage",
    leader: starArm102LeaderZeroPose,
  },
  metal: {
    follower: metalArmPhoto,
    altKey: "poseImageMetal",
    leader: starArm102LeaderZeroPose,
  },
};
// The SO-101's auto-calibration start pose IS the folded resting pose, which
// is exactly what the arm card's product photo already shows — so it is the
// same file, not a second copy of the same picture.
import so101ArmPhoto from "@/assets/arms/so101.jpg";
import so101ManualStartPose from "@/assets/calibration/so101-manual-start-pose.jpg";
import CameraConfiguration, {
  CameraConfig,
} from "@/components/recording/CameraConfiguration";
import { readCamerasActive, writeCamerasActive } from "@/lib/cameraPrefs";
import CalibrationLibrary from "@/components/calibration/CalibrationLibrary";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { PanelHeader, SLIDE } from "@/components/studio/panel/primitives";
import { RobotRecord, formatRobotSetupGap } from "@/hooks/useRobots";
import { useArms } from "@/hooks/useArms";
import {
  calibrationKind,
  effectiveLeaderKind,
  leaderOption as leaderOptionOf,
  leaderOptions,
  supportsAutoCalibration,
  supportsGripperWiggle,
  supportsPortProbe,
  usesFeetechBus,
} from "@/lib/armTypes";
import type { LeaderOptionInfo } from "@/lib/armsApi";
import { servedUrl } from "@/lib/armsApi";
import type { RobotArms } from "@/hooks/useRobots";
import {
  robotLayoutReady,
  setupScopeForArms,
} from "@/hooks/useRobots";
import RobotLayoutChip from "@/components/launchpad/RobotLayoutChip";
import { useLanguage } from "@/contexts/LanguageContext";
import { isCaselessScript } from "@/i18n/config";
import { cn } from "@/lib/utils";
import { detectArmPort } from "@/lib/portDetection";
import { orderedJointEntries } from "@/lib/jointOrder";

interface CalibrationStatus {
  calibration_active: boolean;
  /**
   * SO-101 range sweep: "idle" | "connecting" | "recording" | "completed" |
   * "error" | "stopping".
   *
   * Step wizard (the CAN families' zero pose, and any family whose manifest
   * says calibration.kind "steps"): "idle" | "connecting" | "awaiting_step" |
   * "saving" | "completed" | "error" | "stopping". `/calibration-status`
   * serves whichever flow is live from one endpoint; the two payloads are
   * field-compatible where they overlap.
   */
  status: string;
  device_type: string | null;
  error: string | null;
  message: string;
  step: number;
  total_steps: number;
  current_positions: Record<string, number> | null;
  recorded_ranges: Record<
    string,
    { min: number; max: number; current: number }
  > | null;
  /**
   * Step wizard only, per published step: an image the family serves beside
   * `message` (null → the bundled photo on step 1 of a built-in, else none),
   * and whether the family asked for live joint readings under it. Both are
   * defaulted (null / false) on the SO-101 sweep payload, so one client shape
   * reads both flows; `step` counts up from 1 and `total_steps` is not known
   * ahead of time (the wizard shows "Step N", never "N of M").
   */
  image_url: string | null;
  live_positions: boolean;
}

// One selectable (device_type, arm) slot — shared by the Device step's card
// picker and the multi-arm auto-calibration picker. `key` uniquely identifies
// the slot; cfgField/portField map it to the robot record's fields so the
// slot can prefill its name + port.
interface ArmSlot {
  key: string;
  label: string;
  device: "teleop" | "robot";
  arm: "left" | "right";
  cfgField: keyof RobotRecord;
  portField: keyof RobotRecord;
}

// One card in the Device step's radio-card picker. Selecting a card sets
// both deviceType and arm together, replacing what used to be two separate
// dropdown picks. "Ready" (the check mark) mirrors robotSetupGap's
// definition of a configured arm: a port AND a calibration config assigned —
// plus that port actually being plugged in right now (`portDetected`, which
// the card can't work out itself; it lives outside the window's closure).
/**
 * Radix Select rejects "" as an item value, so the "no port" row carries a
 * sentinel that is mapped back to "" at the boundary. Private to this module:
 * no handler ever sees it.
 */
const NO_PORT = "__no_port__";

// The layout selector's rows. Keys, not text: resolved with t() at render.
// The `value` is the record's `arms` field — data, sent verbatim on Save.
const LAYOUT_OPTIONS = [
  { value: "both", labelKey: "robotConfig.layout.both" },
  { value: "follower", labelKey: "robotConfig.layout.followerOnly" },
  { value: "leader", labelKey: "robotConfig.layout.leaderOnly" },
] as const satisfies readonly { value: RobotArms; labelKey: string }[];

/**
 * One device in section 01: its label, status, port picker and actions in a
 * single cell.
 *
 * This replaces a card that only SELECTED a slot, paired with one shared port
 * control below the grid. That control was modal — what it edited depended on
 * which card was selected — so configuring a bimanual rig meant four round
 * trips between the grid and the controls. Here every slot is directly
 * editable and nothing has to be selected first.
 *
 * Detect and Wiggle are icons; the paragraphs that used to sit beside them
 * moved into their tooltips. Instructions shown WHILE a detect runs stay on
 * screen (see the section body): both of the user's hands are on the arm, so a
 * tooltip is unreachable exactly when it is needed.
 */
const DeviceSlotCell = ({
  slot,
  port,
  portDetected,
  configured,
  availablePorts,
  heldByLabel,
  busy,
  detecting,
  wiggling,
  showWiggle,
  showDetect,
  showAutoDetect,
  onPortChange,
  onDetect,
  onAutoDetect,
  onWiggle,
}: {
  slot: ArmSlot;
  port: string;
  portDetected: boolean;
  configured: boolean;
  availablePorts: string[];
  /** Label of the OTHER slot holding a port, or null when it is free. */
  heldByLabel: (port: string) => string | null;
  /** A calibration or auto-cal run holds the hardware. */
  busy: boolean;
  detecting: boolean;
  wiggling: boolean;
  showWiggle: boolean;
  showDetect: boolean;
  showAutoDetect: boolean;
  onPortChange: (port: string) => void;
  onDetect: () => void;
  onAutoDetect: () => void;
  onWiggle: () => void;
}) => {
  const { t } = useTranslation();
  // A saved port that isn't currently detected outranks "ready": the arm may
  // be unplugged (or moved, or renamed by the OS), and a green check there
  // reads as "connected, all good" when nothing is on that bus.
  const undetected = !!port && !portDetected;
  const ready = !!port && portDetected && configured;
  const detectLabel = t("robotConfig.port.detect");
  const detectTip = t("robotConfig.port.detectTip");
  return (
    // A plain container, not a control. Selecting a slot used to decide which
    // one the single shared port picker edited; every slot now carries its own,
    // so there is nothing left for a selection to mean here. Calibration still
    // has a current device, and section 02's own rows set it.
    <div className="rounded-md border border-border bg-card px-3 py-2">
      <div className="flex items-center gap-2">
        <span className="truncate text-xs font-medium text-foreground">
          {slot.label}
        </span>
        {undetected ? (
          <span
            role="img"
            aria-label={t("robotConfig.slotCard.undetectedLabel")}
            title={t("robotConfig.slotCard.undetectedTitle")}
            className="shrink-0 text-warn"
          >
            <AlertTriangle aria-hidden className="h-4 w-4" />
          </span>
        ) : ready ? (
          <CheckCircle
            className="h-4 w-4 shrink-0 text-ok"
            aria-label={t("robotConfig.slotCard.readyLabel")}
          />
        ) : null}
        <div className="ml-auto flex shrink-0 items-center gap-0.5">
          {showDetect && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={onDetect}
              disabled={busy || detecting || wiggling}
              aria-label={detectLabel}
              // Radix tooltips never open on touch, and an unlabelled icon is
              // unusable without a fallback, so the native title carries it too.
              title={`${detectLabel}. ${detectTip}`}
              className="h-7 w-7 text-muted-foreground hover:text-foreground"
            >
              {detecting ? (
                <Loader2 aria-hidden className="h-4 w-4 animate-spin" />
              ) : (
                <MoveHorizontal aria-hidden className="h-4 w-4" />
              )}
            </Button>
          )}
          {showAutoDetect && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={onAutoDetect}
              disabled={busy || detecting || wiggling}
              aria-label={t("robotConfig.port.detectAuto")}
              title={`${t("robotConfig.port.detectAuto")}. ${t("robotConfig.port.detectTipAuto")}`}
              className="h-7 w-7 text-muted-foreground hover:text-foreground"
            >
              <ScanSearch aria-hidden className="h-4 w-4" />
            </Button>
          )}
          {showWiggle && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={onWiggle}
              disabled={busy || !port || wiggling || detecting}
              aria-label={t("robotConfig.port.wiggle")}
              title={`${t("robotConfig.port.wiggle")}. ${t(
                "robotConfig.port.wiggleTip",
              )}`}
              className="h-7 w-7 text-muted-foreground hover:text-foreground"
            >
              <Hand aria-hidden className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      <Select
        value={port || NO_PORT}
        onValueChange={(v) => onPortChange(v === NO_PORT ? "" : v)}
        disabled={busy}
      >
        <SelectTrigger
          aria-label={t("robotConfig.port.forSlot", { slot: slot.label })}
          className={cn(
            "mt-1.5 h-8 font-mono text-xs",
            !port && "text-muted-foreground",
            undetected && "border-warn/50 text-warn",
          )}
        >
          {/* The label is rendered here rather than by <SelectValue> on
              purpose. A saved-but-undetected port is deliberately absent from
              the item list below, and SelectValue renders NOTHING for a value
              with no matching item — so the one case where the user most needs
              to read the path (which port went missing?) is exactly the case
              where it would disappear. */}
          <span className="truncate">
            {port || t("robotConfig.port.noneAssigned")}
          </span>
        </SelectTrigger>
        <SelectContent>
          {/* The blank row IS the clear action, which is why the separate
              trash button is gone: "which port" and "no port" are one
              question, so they belong to one control. */}
          <SelectItem value={NO_PORT} className="text-xs text-muted-foreground">
            {t("robotConfig.port.noneAssigned")}
          </SelectItem>
          {availablePorts.length > 0 && <SelectSeparator />}
          {availablePorts.map((p) => {
            // In-use ports stay selectable: picking one prompts a swap (this
            // slot's current port goes to the other arm) or, if this slot is
            // empty, a take-with-warning.
            const heldBy = heldByLabel(p);
            return (
              <SelectItem key={p} value={p}>
                <span className="flex items-center gap-2 font-mono text-xs">
                  {p.startsWith("gs_usb:") ? `gs_usb · ${p.slice(7)}` : p}
                  {/* Naming the holder beats a bare "in use": on a bimanual
                      rig there are three other slots it could be, and picking
                      this port takes it off whichever one is named. */}
                  {heldBy && (
                    <span className="rounded border border-warn/40 px-1 font-body text-[10px] text-warn">
                      {heldBy}
                    </span>
                  )}
                </span>
              </SelectItem>
            );
          })}
          {/* A saved-but-undetected port is intentionally NOT offered here: an
              unplugged bus can't be calibrated against, so it's treated as no
              port. It re-selects on its own once the arm is plugged back in
              and ports are rescanned. */}
        </SelectContent>
      </Select>
    </div>
  );
};

// Per-arm terminal/running state in a concurrent batch (from the backend).
interface BatchArmStatus {
  name: string;
  port: string;
  device_type: string;
  arm: string;
  status: string; // running | completed | failed | stopped | stopping | idle
  error: string | null;
  logs: string[];
}

interface BatchAutoCalStatus {
  active: boolean;
  arms: BatchArmStatus[];
  total: number;
  completed: number;
  failed: number;
  logs: string[];
}

// Served straight out of `public/` (see CalibrationClip). Absolute paths: the
// dialog opens from every route, so a relative one would resolve differently
// depending on where the user happened to be.
const AUTO_CAL_CLIP = "/media/calibration/autocal-so101.mp4";
const AUTO_CAL_POSTER = "/media/calibration/autocal-so101.jpg";
const MANUAL_CAL_CLIP = "/media/calibration/manualcal-so101.mp4";
const MANUAL_CAL_POSTER = "/media/calibration/manualcal-so101.jpg";

/**
 * A calibration demo clip: it starts by itself and loops like a GIF, but it is
 * an h264 MP4 with native controls so the scrub slider can be dragged back to
 * the part the user actually needs. A real GIF of a minute of 30fps footage is
 * tens of megabytes and offers no way to seek at all, which is the whole
 * reason this is a <video> and not an <img>.
 *
 * `muted` is what makes `autoPlay` legal — every browser blocks autoplay with
 * sound — so the two attributes travel together; the sources are encoded
 * without an audio track anyway. `playsInline` keeps iOS Safari from hijacking
 * the dialog into its fullscreen player the moment playback starts.
 *
 * Sources live in `public/media/calibration/` rather than `src/assets/`: Vite
 * inlines and hashes imported assets, and a multi-megabyte video has no
 * business in the module graph.
 */
const CalibrationClip = ({
  src,
  poster,
  label,
  unsupported,
  linkLabel,
  clipRef,
}: {
  src: string;
  poster: string;
  label: string;
  unsupported: string;
  linkLabel: string;
  clipRef?: React.Ref<HTMLDivElement>;
}) => (
  <div ref={clipRef} className="overflow-hidden rounded-md bg-muted">
    <video
      className="aspect-video h-auto w-full"
      poster={poster}
      aria-label={label}
      autoPlay
      loop
      muted
      playsInline
      controls
      controlsList="nodownload noplaybackrate"
      disablePictureInPicture
      preload="metadata"
    >
      <source src={src} type="video/mp4" />
      <p className="py-4 text-center text-sm text-muted-foreground">
        {unsupported}
        <br />
        <a
          href={src}
          className="underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          {linkLabel}
        </a>
      </p>
    </video>
  </div>
);

export interface RobotConfigDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The robot whose settings this window edits. Required while open. */
  robotName: string | null;
}

/**
 * Robot settings as a windowed dialog — ports, calibration, cameras, and
 * motor power for one robot, popped up over the Launchpad / studio instead of
 * a full-page route. The configuration logic is the former Calibration page's,
 * ported verbatim (draft-until-Save ports/cameras/torque, Detect/Wiggle,
 * manual + concurrent multi-arm auto-calibration, exit guard); only the
 * surface changed: a shadcn Dialog styled with the policy studio's vocabulary
 * (PanelHeader step digits, eyebrow labels, hairline dividers, default
 * control sizes).
 *
 * The whole window body mounts fresh per open and unmounts on close — that is
 * what makes the page semantics carry over: camera streams release and every
 * draft resets. Closing during a live MANUAL calibration confirms an explicit
 * abort (stop by session id); any other abandonment (route change, tab gone)
 * is covered by the session lease — missed heartbeats make the SERVER stop
 * the session. A running batch auto-calibration survives close only as long
 * as its lease: reopening resumes the panel (and the heartbeat) while it
 * lives.
 */
const RobotConfigDialog = ({
  open,
  onOpenChange,
  robotName,
}: RobotConfigDialogProps) => {
  if (!open || !robotName) return null;
  return (
    <RobotConfigWindow robotName={robotName} onOpenChange={onOpenChange} />
  );
};

const RobotConfigWindow = ({
  robotName,
  onOpenChange,
}: {
  robotName: string;
  onOpenChange: (open: boolean) => void;
}) => {
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { t } = useTranslation();
  const { language } = useLanguage();
  const { byId: armById } = useArms();

  const [deviceType, setDeviceType] = useState<string>("teleop");
  const [arm, setArm] = useState<"left" | "right">("left");
  const [port, setPort] = useState<string>("");
  // `robot` is the last-fetched SERVER baseline. Client-initiated config edits in
  // this window (ports, cameras, motor power) NEVER write straight to the record;
  // they accumulate in local draft state and are persisted only when the user
  // presses Save (a single batched POST). Dirtiness is the draft-vs-baseline diff.
  const [robot, setRobot] = useState<RobotRecord | null>(null);
  // Draft overlay for the four port slots. A field present here (including "" for
  // a cleared slot) overrides the baseline until Save. `draftPort` reads through
  // the overlay so every port-derived value (dropdown, conflict checks, batch
  // slots) reflects unsaved edits.
  const [portDraft, setPortDraft] = useState<
    Partial<Record<keyof RobotRecord, string>>
  >({});
  const draftPort = useCallback(
    (field: keyof RobotRecord): string =>
      portDraft[field] ?? ((robot?.[field] as string) || ""),
    [portDraft, robot],
  );
  const [saving, setSaving] = useState(false);
  // Transient post-save acknowledgment on the Save button itself ("Saved ✓"
  // for ~2s). Without it a successful save drops the button straight into the
  // same disabled-gray it has when there's nothing to save — which reads as
  // "you can't save", not "you're done". Cleared by the timeout; the label
  // also ignores it while new edits are pending (isDirty re-enables Save).
  const [justSaved, setJustSaved] = useState(false);
  useEffect(() => {
    if (!justSaved) return;
    const t = setTimeout(() => setJustSaved(false), 2000);
    return () => clearTimeout(t);
  }, [justSaved]);
  const [quitPromptOpen, setQuitPromptOpen] = useState(false);
  // Closing the window during a live MANUAL calibration aborts it (the exit
  // guard's unmount cleanup) — confirm that first, like the page's back-button
  // confirm did.
  const [abortPromptOpen, setAbortPromptOpen] = useState(false);

  const isBimanual = robot?.mode === "bimanual";
  // The arm layout is a draft until Save, like ports and cameras. Hidden arm
  // fields remain in the record so switching the layout back restores them.
  const [armsDraft, setArmsDraft] = useState<RobotArms | null>(null);
  const draftArms: RobotArms = armsDraft ?? robot?.arms ?? "both";
  const showLeader = draftArms !== "follower";
  const showFollower = draftArms !== "leader";
  // The hardware family this robot is — a manifest id, resolved against
  // GET /api/v1/arms. Its entry answers, separately, which calibration flow
  // the Calibrate step runs (range sweep vs zero pose), whether automatic
  // calibration exists, how Detect works (protocol probe vs hand gesture),
  // and whether the servo-register UI (wiggle, motor power) applies; the id
  // itself names which calibration library the file actions address. Records
  // written before the Maker arm existed read back as "so101", so the
  // fallback here is only for the pre-fetch render where `robot` is null.
  // Before the manifest loads every predicate answers the SO-101 shape, which
  // is exactly what this window drew before it existed.
  const armType = robot?.arm_type ?? "so101";
  const armInfo = armById(armType);
  // No installed family answers to this record's arm type (a hand-edited
  // record, or an extension that is no longer installed). The server refuses
  // to start anything for it, so this window says so once and disables the
  // actions that would try — detect, calibrate — rather than letting them
  // fail one 400 at a time.
  const armUnavailable = !!robot && robot.arm_available === false;
  // The manifest has no entry for this record — either the arm is not
  // installed (above) or the manifest has not loaded yet. Either way every
  // predicate below is answering with the SO-101 fallback, and acting on that
  // for a CAN record would post Detect to the Feetech endpoint and mint an
  // unsuffixed calibration name into the SHARED Star-leader directory. So
  // detect / wiggle / calibrate / start are all held until the entry exists.
  const armActionsBlocked = !!robot && !armInfo;
  const armsNotLoaded = armActionsBlocked && !armUnavailable;
  // Which calibration UI this record gets, off the manifest: the SO-101's
  // sweep flows, the generic step wizard, or the extension's own panel
  // (which nothing here can start yet — the rows say so and stay disabled).
  const kind = calibrationKind(armInfo);
  const stepCalibration = kind === "steps";
  const panelCalibration = kind === "panel";
  const autoCalibration = supportsAutoCalibration(armInfo);
  const portProbe = supportsPortProbe(armInfo);
  const feetechBus = usesFeetechBus(armInfo);
  const gripperWiggle = supportsGripperWiggle(armInfo);
  // Which of the family's leaders drives this robot. Only a family with a
  // choice (the Metal arm: its Star Arm 102, or a second gravity-compensated
  // Metal arm; the Maker arm: the lever- or the trigger-gripper Star Arm
  // 102) renders the picker and sends `leader_kind` with its port
  // detection and library requests; every other family's requests are what
  // they always were. An ENERGIZED leader (holds torque while the human
  // moves it) answers the follower's protocol, so the probe cannot tell the
  // two apart and the gesture is refused — the gripper wiggle is what is
  // left, and its Wiggle button appears on the leader row too.
  const leaderChoices = leaderOptions(armInfo);
  const multiLeader = leaderChoices.length > 1;
  const leaderKind = effectiveLeaderKind(armInfo, robot?.leader_kind);
  const leaderChoice = leaderOptionOf(armInfo, leaderKind);
  const leaderEnergized = !!leaderChoice?.energized;
  const leaderKindParam = multiLeader ? leaderKind : undefined;
  const [savingLeaderKind, setSavingLeaderKind] = useState(false);
  // Display name for a leader option: the catalog's per-id override for the
  // built-ins (what localizes it), else the manifest's own English label.
  const leaderOptionLabel = (option: LeaderOptionInfo): string =>
    t(`robotConfig.leaderKind.optionFor.${armType}.${option.id}` as never, {
      defaultValue: option.label,
    }) as string;
  // In single (or left) mode the primary leader/follower fields are used; in
  // bimanual mode the right arm uses the right_* fields. Maps the current
  // device_type + arm to the record's port and config field names.
  const isRight = arm === "right";
  const portField = (
    deviceType === "teleop"
      ? isRight
        ? "right_leader_port"
        : "leader_port"
      : isRight
        ? "right_follower_port"
        : "follower_port"
  ) as keyof RobotRecord;
  const configField = (
    deviceType === "teleop"
      ? isRight
        ? "right_leader_config"
        : "leader_config"
      : isRight
        ? "right_follower_config"
        : "follower_config"
  ) as keyof RobotRecord;

  const assignedConfig = robot ? (robot[configField] as string) : "";
  // Calibration names are arbitrary in every mode — bimanual no longer forces
  // "<robot>_<arm>" (lerobot's "<base>_left/right" convention is satisfied by a
  // per-session staging copy on the backend, not by the on-disk name). Default
  // to the in-use config for this slot, else a per-arm suggestion so a fresh
  // bimanual robot doesn't propose the same name for all four slots.
  //
  // The family's own suffix from the manifest ("" for the SO-101, "_maker" /
  // "_metal" for the CAN families, whose Star-leader calibrations share ONE
  // library directory while the presets' zero poses differ — an unsuffixed
  // default would let a Maker robot and a Metal robot silently share a zero
  // that is wrong for one of them). Mirrors the server's
  // default_slot_config_name() in makermodslab/utils/config.py (each family's
  // default_calibration_name); the explicit config_file this window sends at
  // calibration start WINS over the server's own default, so the two must
  // agree. Before the manifest loads the suffix is "", the SO-101 shape.
  const defaultBaseName = robotName
    ? `${robotName}${armInfo?.calibration_name_suffix ?? ""}`
    : "";
  const defaultConfigName = assignedConfig?.trim()
    ? assignedConfig
    : defaultBaseName
      ? isBimanual
        ? `${defaultBaseName}_${arm}`
        : defaultBaseName
      : "";

  // No name is chosen in the UI. Calibration always saves to the robot's own
  // default config name for this slot and silently replaces it (see overwrite
  // below). To keep an old calibration under a different name, the user renames
  // it afterward via the existing per-side rename feature.
  const calibrationConfigName = defaultConfigName;

  // Bumped when a calibration completes so the per-side CalibrationLibrary
  // dropdowns re-fetch and surface any newly-named file.
  const [calibReloadToken, setCalibReloadToken] = useState(0);

  // Which calibration-file row has its "New calibration" panel expanded — the
  // row's config field — or null when every panel is collapsed. Opening a
  // row's panel also points deviceType/arm at that slot, so the whole
  // calibration flow (port lookup, save name, start request) targets it.
  const [newCalibFor, setNewCalibFor] = useState<string | null>(null);
  // CAN arms are posed individually. Advance only after a successful save.
  const [zeroCalQueue, setZeroCalQueue] = useState<ArmSlot[]>([]);
  // Which flow the open panel is set to. Chosen by the two mode buttons on an
  // SO-101; a CAN arm has only the zero pose, so it needs no choice and this
  // stays null there. Reset whenever the panel closes so reopening starts at
  // the choice again.
  const [calibMode, setCalibMode] = useState<"auto" | "manual" | null>(null);
  // Keep the expanded panel attached to the slot the calibration flow actually
  // targets: if device/arm changes while a panel is open (e.g. via the step-01
  // selector), the panel follows to the matching calibration-file row.
  useEffect(() => {
    setNewCalibFor((prev) =>
      prev && prev !== configField ? configField : prev,
    );
  }, [configField]);

  // Toggle a row's "New calibration" panel. Opening retargets the calibration
  // flow at that row's slot; clicking the active row's + again collapses it.
  const toggleNewCalibration = (
    field: string,
    device: "teleop" | "robot",
    whichArm: "left" | "right",
  ) => {
    setZeroCalQueue([]);
    if (newCalibFor === field) {
      setNewCalibFor(null);
      setCalibMode(null);
      return;
    }
    setDeviceType(device);
    setArm(whichArm);
    setCalibMode(null);
    setNewCalibFor(field);
  };

  // Ports already assigned to the OTHER arms of this robot — each physical arm
  // needs its own serial port, so these are greyed out in the dropdown. The
  // right-arm ports only count in bimanual mode (mirrors the backend guard), so
  // a single-arm robot's stale right_* ports don't get shown as taken.
  const portFields =
    robot?.mode === "bimanual"
      ? ([
          "leader_port",
          "follower_port",
          "right_leader_port",
          "right_follower_port",
        ] as const)
      : (["leader_port", "follower_port"] as const);
  const otherArmPorts = robot
    ? portFields
        .filter((f) => f !== portField)
        .map((f) => draftPort(f))
        .filter(Boolean)
    : [];

  // Human-readable name for a port slot, matching the labels the "Calibration
  // files" checklist renders. Bimanual distinguishes left/right; single
  // mode has just Leader/Follower. Used by Detect's reassign toast to name the
  // slot whose port it just took over.
  const portFieldLabel = (field: keyof RobotRecord): string => {
    switch (field) {
      case "leader_port":
        return t(
          isBimanual ? "robotConfig.arm.leftLeader" : "robotConfig.arm.leader",
        );
      case "follower_port":
        return t(
          isBimanual
            ? "robotConfig.arm.leftFollower"
            : "robotConfig.arm.follower",
        );
      case "right_leader_port":
        return t("robotConfig.arm.rightLeader");
      case "right_follower_port":
        return t("robotConfig.arm.rightFollower");
      default:
        // Field name, not copy — a developer-facing fallback.
        return String(field);
    }
  };

  // Which half of the rig a port slot belongs to, in the backend's vocabulary.
  // Detect is invoked PER CARD (handleDetect takes the slot's field), while
  // `deviceType` tracks whichever calibration row is expanded — so the CAN
  // detect path must derive the side from the field it was handed. Reading
  // `deviceType` instead probed for the leader when the user pressed Detect on
  // the follower, and staged the leader's UART port onto the follower slot.
  const portFieldDevice = (field: keyof RobotRecord): "teleop" | "robot" =>
    field === "leader_port" || field === "right_leader_port"
      ? "teleop"
      : "robot";
  const [wiggling, setWiggling] = useState(false);
  // Touch-to-identify: watching every port for a hand-moved shoulder-pan swing.
  // Which slot's Detect is running, or null. A field rather than a boolean so
  // the spinner and the live instructions appear on the row that started it.
  const [detecting, setDetecting] = useState<keyof RobotRecord | null>(null);
  // Picking a port that's in use by another arm (via the dropdown OR Detect)
  // stages the assignment here and opens a confirmation dialog instead of
  // applying immediately. Two shapes, distinguished by `source`:
  //  - When the OTHER slot holds this port and THIS slot already had a port,
  //    confirming SWAPS: the other slot receives this slot's old port, so no
  //    slot ends up empty. `swapPort` carries the old port for the message and
  //    the patch.
  //  - When this slot had no port, the swap degenerates to a take-with-warning:
  //    the other slot is left empty. `swapPort` is null in that case.
  // `releasedField`/`releasedLabel` are null when the port isn't in use at all
  // (plain Detect assign) — then confirming is just a straight assignment.
  const [portAssignPrompt, setPortAssignPrompt] = useState<{
    source: "detect" | "manual";
    port: string;
    message: string;
    // The slot the assignment lands on, captured when the prompt is staged.
    // Section 01 now edits every slot in place, so the selected device can
    // change while this dialog is open; without pinning the field here, a
    // confirm would write whichever slot happened to be selected by then.
    targetField: keyof RobotRecord;
    targetLabel: string;
    releasedField: keyof RobotRecord | null;
    releasedLabel: string | null;
    swapPort: string | null;
  } | null>(null);
  // --- Concurrent multi-arm auto-calibration ---
  // The batch is the engine behind BOTH auto-calibration entry points. Header
  // "Calibrate all" opens this picker, where the user ticks 1-4 arm slots and
  // every one's hands-off auto-cal subprocess runs at the SAME TIME, each on
  // its own port. A calibration-file row's "Auto-calibrate" runs the same
  // batch with just that row's slot ticked and never shows the picker.
  // The manual step-by-step flow is untouched and stays available separately.
  const [batchAutoCalOpen, setBatchAutoCalOpen] = useState(false);
  const [batchAutoCalPromptOpen, setBatchAutoCalPromptOpen] = useState(false);
  // "A finished run's results are still on screen." The status box used to
  // render on (batchAutoCalOpen || batchAutoCal.active), which is fine for the
  // multi-arm path — it leaves the picker open, so the box survives the run —
  // but a row's single-arm "Auto-calibrate" deliberately closes the picker, so
  // both terms went false in the same tick the run ended and the whole box
  // unmounted: per-arm rows (the ONLY place a failure's error text surfaces,
  // via their title tooltip), the completed/failed summary, and the logs all
  // vanished, leaving nothing but a transient toast — worst exactly on
  // failure. This flag is set the moment a run is (or becomes) active, so it's
  // already true when `active` flips false, and the results stay up until the
  // user dismisses them.
  const [batchAutoCalResultsOpen, setBatchAutoCalResultsOpen] = useState(false);
  // Which arm slots are ticked. Each slot's port comes straight from its
  // assignment on the robot record; each slot's save name is the robot's own
  // default config for that slot (no per-arm name input).
  const [batchSelected, setBatchSelected] = useState<Record<string, boolean>>(
    {},
  );
  const [batchAutoCal, setBatchAutoCal] = useState<BatchAutoCalStatus>({
    active: false,
    arms: [],
    total: 0,
    completed: 0,
    failed: 0,
    logs: [],
  });
  const [availablePorts, setAvailablePorts] = useState<string[]>([]);
  const [portsLoading, setPortsLoading] = useState(false);
  // False until the first scan has come back. `portsLoading` alone can't tell
  // "no ports" from "haven't looked yet" — it's still false on the first paint,
  // before the mount effect fires — and an empty availablePorts would flash a
  // "port not detected" warning on every configured arm card.
  const [portsScanned, setPortsScanned] = useState(false);
  const [multipleArmsDetected, setMultipleArmsDetected] = useState(false);
  const [cameras, setCameras] = useState<CameraConfig[]>([]);
  const releaseStreamsRef = useRef<(() => void) | null>(null);
  // Off by default so merely opening the settings window never grabs a camera.
  // The user explicitly starts a scan, which is when cameras are turned on,
  // enumerated, and the browser permission prompt is requested.
  //
  // Once they HAVE turned it on for this robot, that answer is remembered and
  // replayed on the next open (see lib/cameraPrefs): the window mounts fresh
  // every time, so without this the switch snapped back to off and previews
  // had to be re-opened by hand after every visit. A robot the user has never
  // switched on still reads false, so the "never grabs a camera on its own"
  // property holds for the case it was written for.
  const [camerasActive, setCamerasActive] = useState(() =>
    readCamerasActive(robotName),
  );

  // No releaseStreamsRef call here, on purpose. CameraConfiguration stays
  // mounted and drops its own streams when `active` goes false — that is what
  // keeps the picked camera and its preview across an off/on cycle. Calling
  // release as well would latch its internal pause flag on and the preview
  // would never come back.
  const handleCamerasActiveChange = (active: boolean) => {
    setCamerasActive(active);
    // Persist on the user's gesture only. Writing from an effect on
    // `camerasActive` would also persist states the code sets for its own
    // reasons, which is not the same thing as what the user chose.
    writeCamerasActive(robotName, active);
  };

  useEffect(() => {
    return () => {
      releaseStreamsRef.current?.();
    };
  }, []);

  // Arm slots the multi-arm auto-cal picker can offer. Bimanual exposes all
  // four (left/right × leader/follower); single-arm exposes the leader +
  // follower pair. Each maps to the record's config/port fields for prefill.
  // Slots the layout hides are dropped here, so section 01, the calibration
  // rows and "Calibrate all" all agree on which arms this machine has.
  const armSlots: ArmSlot[] = useMemo(() => {
    const all: ArmSlot[] = isBimanual
        ? [
            {
              key: "teleop:left",
              label: t("robotConfig.arm.leftLeader"),
              device: "teleop",
              arm: "left",
              cfgField: "leader_config",
              portField: "leader_port",
            },
            {
              key: "robot:left",
              label: t("robotConfig.arm.leftFollower"),
              device: "robot",
              arm: "left",
              cfgField: "follower_config",
              portField: "follower_port",
            },
            {
              key: "teleop:right",
              label: t("robotConfig.arm.rightLeader"),
              device: "teleop",
              arm: "right",
              cfgField: "right_leader_config",
              portField: "right_leader_port",
            },
            {
              key: "robot:right",
              label: t("robotConfig.arm.rightFollower"),
              device: "robot",
              arm: "right",
              cfgField: "right_follower_config",
              portField: "right_follower_port",
            },
          ]
        : [
            {
              key: "teleop:left",
              label: t("robotConfig.arm.leader"),
              device: "teleop",
              arm: "left",
              cfgField: "leader_config",
              portField: "leader_port",
            },
            {
              key: "robot:left",
              label: t("robotConfig.arm.follower"),
              device: "robot",
              arm: "left",
              cfgField: "follower_config",
              portField: "follower_port",
            },
          ];
    return all.filter((slot) =>
      slot.device === "teleop" ? showLeader : showFollower,
    );
  }, [isBimanual, showLeader, showFollower, t]);

  const fetchRobot = useCallback(async (): Promise<RobotRecord | null> => {
    if (!robotName) return null;
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/api/v1/robots/${encodeURIComponent(robotName)}`,
      );
      if (!res.ok) return null;
      const data = await res.json();
      const r = (data.robot as RobotRecord | null) ?? null;
      setRobot(r);
      return r;
    } catch (e) {
      console.error("Failed to load robot record:", e);
      return null;
    }
  }, [robotName, baseUrl, fetchWithHeaders]);

  // Open the side's calibration folder in the OS file browser (Finder/Explorer/
  // xdg-open). A local, non-network action handled server-side; the dir is
  // created there if missing so a fresh install still opens an empty folder.
  const openCalibrationFolder = useCallback(
    async (device: "teleop" | "robot") => {
      try {
        const res = await fetchWithHeaders(
          `${baseUrl}/api/v1/open-calibration-folder`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              device_type: device,
              arm_type: armType,
              ...(device === "teleop" && leaderKindParam
                ? { leader_kind: leaderKindParam }
                : {}),
            }),
          },
        );
        const data = await res.json().catch(() => ({}));
        if (!res.ok || !data.opened) {
          toast({
            title: t("robotConfig.files.toast.openFolderFailedTitle"),
            description: data.message,
            variant: "destructive",
          });
        }
      } catch (e) {
        toast({
          title: t("robotConfig.files.toast.openFolderFailedTitle"),
          description: String(e),
          variant: "destructive",
        });
      }
    },
    [baseUrl, fetchWithHeaders, toast, t, armType, leaderKindParam],
  );

  // List the USB-serial ports for the dropdown (filtered to arm-like devices by
  // the backend). Refreshable so plugging in an arm and rescanning works.
  const fetchPorts = useCallback(async () => {
    setPortsLoading(true);
    try {
      const res = await fetchWithHeaders(`${baseUrl}/api/v1/available-ports`);
      const data = await res.json();
      setAvailablePorts(Array.isArray(data.ports) ? data.ports : []);
      setMultipleArmsDetected(false);
    } catch (e) {
      console.error("Failed to list ports:", e);
    } finally {
      setPortsLoading(false);
      setPortsScanned(true);
    }
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    fetchPorts();
  }, [fetchPorts]);

  // Initial fetch + form prefill on open.
  useEffect(() => {
    if (!robotName) return;
    let cancelled = false;
    (async () => {
      const r = await fetchRobot();
      if (!r || cancelled) return;
      // Default to the first incomplete side in the checklist (leader, then follower).
      const defaultDevice = !r.leader_config
        ? "teleop"
        : !r.follower_config
          ? "robot"
          : "teleop";
      setDeviceType(defaultDevice);
      setPort(
        defaultDevice === "teleop"
          ? r.leader_port || ""
          : r.follower_port || "",
      );
      setCameras(r.cameras ?? []);
    })();
    return () => {
      cancelled = true;
    };
  }, [robotName, fetchRobot]);

  // Camera edits (adds/removes/edits AND CameraConfiguration's automatic
  // resync corrections) update the local draft only. Nothing is written to the
  // robot record until Save. `cameras` is the draft; `robot.cameras` the baseline.
  const handleCamerasChange = (next: CameraConfig[]) => {
    setCameras(next);
  };

  const [calibrationStatus, setCalibrationStatus] = useState<CalibrationStatus>(
    {
      calibration_active: false,
      status: "idle",
      device_type: null,
      error: null,
      message: "",
      step: 0,
      total_steps: 1,
      current_positions: null,
      recorded_ranges: null,
      image_url: null,
      live_positions: false,
    },
  );
  const [isPolling, setIsPolling] = useState(false);
  const terminalCalibHandled = useRef(false);

  // One /complete-calibration-step POST in flight at a time. A double-click
  // on Next used to post twice, and on a multi-step family the second POST
  // confirms the FOLLOWING step without the user. The ref is the guard (a
  // second click can land before the state's re-render); the state disables
  // the button so the guard is visible.
  const completingStepRef = useRef(false);
  const [completingStep, setCompletingStep] = useState(false);

  // Manual (step-by-step) calibration liveness. Set optimistically at start
  // (so the abort prompt already guards a close in the sub-second before the
  // first status poll) and cleared when the session reaches a terminal
  // status.
  //
  // Scope note: this tracks the MANUAL flow ONLY. The batch auto-calibration
  // subprocess resumes its panel on remount (see the batch-status
  // resume/poll effects) and is stopped only via its explicit "Stop all"
  // button — or by its lease, once nobody renews it.
  const [manualCalibLive, setManualCalibLive] = useState(false);
  useEffect(() => {
    if (calibrationStatus.calibration_active) {
      setManualCalibLive(true);
    } else if (
      ["idle", "completed", "error"].includes(calibrationStatus.status)
    ) {
      setManualCalibLive(false);
    }
  }, [calibrationStatus.calibration_active, calibrationStatus.status]);

  // Session identities from POST /api/v1/sessions — the last browser exit
  // guard (useSessionExitGuard: beforeunload beacon + popstate sentinel +
  // unmount-stop) retired when calibration joined the sessions surface. The
  // lease is THE safety net now: while a flow is live this window renews it
  // (~20s heartbeats), and an abandoned page — tab closed, wifi died, route
  // changed away — makes the SERVER stop the session when the heartbeats
  // stop. What remains browser-side is a courtesy native confirm so an
  // accidental ⌘W isn't silent. (The manual-calibration arm is LIMP — torque
  // off — so a lease-timeout stop is a clean teardown, not a mid-motion
  // halt; a batch auto-cal stop runs the script's own graceful stop.)
  const [calibSessionId, setCalibSessionId] = useState<string | null>(null);
  const [autoCalSessionId, setAutoCalSessionId] = useState<string | null>(null);
  useSessionHeartbeat(calibSessionId, tabOwnerId(), manualCalibLive);
  useSessionHeartbeat(autoCalSessionId, tabOwnerId(), batchAutoCal.active);
  useUnloadWarning(manualCalibLive || batchAutoCal.active);

  const pollStatus = async (isCurrent = () => true) => {
    try {
      const response = await fetchWithHeaders(
        `${baseUrl}/api/v1/calibration-status?arm_type=${encodeURIComponent(armType)}`,
      );
      if (response.ok) {
        const status = await response.json();
        if (!isCurrent()) return;
        setCalibrationStatus(status);

        if (
          !status.calibration_active &&
          (status.status === "completed" ||
            status.status === "error" ||
            status.status === "idle")
        ) {
          setIsPolling(false);
        }
      }
    } catch (error) {
      console.error("Error polling status:", error);
    }
  };

  // Defaults to the selected slot's port; section 01's per-row button passes
  // its own row's port so it never depends on what is selected.
  // Saved at once, not staged with the port drafts: the calibration flow
  // resolves the leader from the SAVED record server-side, so a draft that
  // differed from it would calibrate (and start) the other leader. The
  // server blanks the leader ports and calibrations on a switch — they name
  // different hardware and a different library — and the returned record
  // is adopted as the new baseline.
  const handleLeaderKindChange = async (next: string) => {
    if (!robotName || !robot || next === leaderKind) return;
    setSavingLeaderKind(true);
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/api/v1/robots/${encodeURIComponent(robotName)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ leader_kind: next }),
        },
      );
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.robot) {
        setRobot(data.robot);
        setPortDraft({});
        toast({ title: t("robotConfig.leaderKind.toast.savedTitle") });
      } else {
        toast({
          title: t("robotConfig.leaderKind.toast.saveFailedTitle"),
          description: data.detail ?? data.message,
          variant: "destructive",
        });
      }
    } catch (e) {
      toast({
        title: t("robotConfig.leaderKind.toast.saveFailedTitle"),
        description: String(e),
        variant: "destructive",
      });
    } finally {
      setSavingLeaderKind(false);
    }
  };

  const handleWiggle = async (
    wigglePort: string = port,
    device: "teleop" | "robot" = deviceType as "teleop" | "robot",
  ) => {
    if (armActionsBlocked) return;
    if (!wigglePort) {
      toast({
        title: t("robotConfig.port.toast.missingPortTitle"),
        description: t("robotConfig.port.toast.missingPortWiggle"),
        variant: "destructive",
      });
      return;
    }
    setWiggling(true);
    try {
      // A Feetech arm wiggles through the legacy servo route; a CAN family
      // with a gripper wiggle (the Metal arm) through its own, which opens
      // the port with ONLY the gripper motor on the bus and disables it
      // again afterwards — the identification of last resort when the probe
      // and the gesture cannot tell two Damiao arms apart.
      const res = feetechBus
        ? await fetchWithHeaders(`${baseUrl}/api/v1/wiggle`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ port: wigglePort }),
          })
        : await fetchWithHeaders(`${baseUrl}/api/v1/maker/wiggle-gripper`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              arm_type: armType,
              device_type: device,
              port: wigglePort,
              ...(leaderKindParam ? { leader_kind: leaderKindParam } : {}),
            }),
          });
      const data = await res.json();
      if (data.success) {
        toast({
          title: t("robotConfig.port.toast.wiggleStartedTitle"),
          description: data.message,
        });
      } else {
        toast({
          title: t("robotConfig.port.toast.wiggleFailedTitle"),
          description: data.message,
          variant: "destructive",
        });
      }
    } catch (e) {
      toast({
        title: t("robotConfig.port.toast.wiggleFailedTitle"),
        description: String(e),
        variant: "destructive",
      });
    } finally {
      setWiggling(false);
    }
  };

  // The inverse of Wiggle: instead of driving a motor, the backend watches
  // every detected port (read-only) while the user swings the arm's base by
  // hand, then reports which port saw the motion. On success the detected
  // port is STAGED for confirmation (see handleConfirmPortAssign) — nothing
  // is selected or persisted until the user confirms in the dialog.
  //
  // Detect is physical ground truth — the user just swung THIS arm on THIS
  // port — so if the record currently assigns the detected port to a DIFFERENT
  // slot, that slot's entry is stale (typical after a cable swap). We surface
  // that in the confirmation dialog and, on confirm, SWAP: the other slot
  // receives this slot's previous port (if any) while this slot takes the
  // detected port, in a single upsert (the backend's port-conflict guard
  // evaluates the prospective merged record, so a two-slot swap passes). If
  // this slot had no port the swap degenerates to a take-with-warning that
  // leaves the other slot empty. Confirm/messaging happen in
  // handleConfirmPortAssign.
  const [detectionMethod, setDetectionMethod] = useState<"swing" | "auto">(
    "swing",
  );
  const handleDetect = async (
    field: keyof RobotRecord = portField,
    method: "swing" | "auto" = "swing",
  ) => {
    if (armActionsBlocked) return;
    setDetectionMethod(method);
    setDetecting(field);
    try {
      const data = await detectArmPort(
        fetchWithHeaders,
        baseUrl,
        armType,
        portFieldDevice(field),
        t("robotConfig.port.multipleHelp"),
        method,
        { portProbe, leaderKind: leaderKindParam, leaderEnergized },
      );
      if (data.multiple) {
        setMultipleArmsDetected(true);
        return;
      }
      if (data.success && data.port) {
        // Which OTHER slot (if any) currently holds the detected port? Reuses
        // the same portFields set the dropdown uses (right_* only in bimanual),
        // so a single-arm robot's stale right_* ports don't trigger a release.
        const conflictingField = robot
          ? portFields.find((f) => f !== field && draftPort(f) === data.port)
          : undefined;
        // The port THIS slot currently holds — handed to the other slot on a
        // swap. Null/empty means the swap degenerates to a take-with-warning.
        const currentPort = draftPort(field);

        // Stage the result and open the confirmation dialog. No assignment or
        // persist happens here — that's deferred to handleConfirmPortAssign.
        setPortAssignPrompt({
          source: "detect",
          port: data.port,
          message: data.message,
          targetField: field,
          targetLabel: portFieldLabel(field),
          releasedField: conflictingField ?? null,
          releasedLabel: conflictingField
            ? portFieldLabel(conflictingField)
            : null,
          swapPort:
            conflictingField && currentPort &&
            !(currentPort.startsWith("gs_usb:") && conflictingField.includes("leader"))
              ? currentPort
              : null,
        });
      } else {
        toast({
          title: t("robotConfig.port.toast.noArmTitle"),
          // `fallback: "wiggle"` is the server naming the identification of
          // last resort: point at the row's Wiggle button.
          description:
            data.fallback === "wiggle"
              ? `${data.message ?? ""} ${t("robotConfig.port.wiggleFallback")}`.trim()
              : data.message,
          variant: "destructive",
        });
      }
    } catch (e) {
      toast({
        title: t("robotConfig.port.toast.detectFailedTitle"),
        description: String(e),
        variant: "destructive",
      });
    } finally {
      setDetecting(null);
    }
  };

  // Apply a staged port assignment (from Detect or the manual dropdown) once
  // the user confirms. Cancel simply closes the dialog (setPortAssignPrompt(null))
  // and leaves everything as-is. Three cases:
  //  - releasedField + swapPort: SWAP — this slot takes the port, the other slot
  //    takes this slot's old port. One upsert; the backend's port-conflict guard
  //    evaluates the merged record, so a two-slot swap of distinct ports passes.
  //  - releasedField, no swapPort: take-with-warning — this slot had no port, so
  //    the other slot is left empty.
  //  - neither: straight assign (port wasn't in use anywhere).
  const handleConfirmPortAssign = async () => {
    const prompt = portAssignPrompt;
    if (!prompt) return;
    setPortAssignPrompt(null);

    if (prompt.targetField === portField) setPort(prompt.port);
    const detected = prompt.source === "detect";

    if (prompt.releasedField) {
      const nextRobot = await persistPorts({
        [prompt.releasedField]: prompt.swapPort ?? "",
        [prompt.targetField]: prompt.port,
      });
      if (nextRobot) {
        if (prompt.swapPort) {
          toast({
            title: detected
              ? t("robotConfig.port.toast.swappedDetectedTitle")
              : t("robotConfig.port.toast.swappedTitle"),
            description: `${detected ? `${prompt.message} ` : ""}${t(
              "robotConfig.port.toast.swappedDescription",
              {
                port: prompt.port,
                released: prompt.releasedLabel ?? "",
                swapPort: prompt.swapPort,
              },
            )}`,
          });
        } else {
          toast({
            title: detected
              ? t("robotConfig.port.toast.movedDetectedTitle")
              : t("robotConfig.port.toast.movedTitle"),
            description: `${detected ? `${prompt.message} ` : ""}${t(
              "robotConfig.port.toast.movedDescription",
              { port: prompt.port, released: prompt.releasedLabel ?? "" },
            )}`,
          });
        }
      }
      // persistPorts surfaces its own error toast on failure.
    } else {
      persistPort(prompt.port, prompt.targetField);
      toast({
        title: detected
          ? t("robotConfig.port.toast.identifiedTitle")
          : t("robotConfig.port.toast.assignedTitle"),
        description: detected
          ? `${prompt.message} ${t("robotConfig.port.toast.identifiedDescription")}`
          : t("robotConfig.port.toast.assignedDescription", {
              port: prompt.port,
            }),
      });
    }
  };

  // Manual dropdown pick. In-use ports are now selectable (no longer greyed
  // out): picking one that another slot holds stages a swap/take confirmation
  // (same dialog as Detect). Picking a free port assigns immediately.
  const handleSelectPort = (
    nextPort: string,
    field: keyof RobotRecord = portField,
  ) => {
    const conflictingField = robot
      ? portFields.find((f) => f !== field && draftPort(f) === nextPort)
      : undefined;
    if (conflictingField) {
      const currentPort = draftPort(field);
      setPortAssignPrompt({
        source: "manual",
        port: nextPort,
        message: "",
        targetField: field,
        targetLabel: portFieldLabel(field),
        releasedField: conflictingField,
        releasedLabel: portFieldLabel(conflictingField),
        swapPort:
          currentPort.startsWith("gs_usb:") && conflictingField.includes("leader")
            ? null
            : currentPort || null,
      });
      return;
    }
    // `port` mirrors the SELECTED slot, so only touch it when this edit is
    // the selected one. A row edit elsewhere is picked up by the sync effect.
    if (field === portField) setPort(nextPort);
    persistPort(nextPort, field);
  };

  // --- Concurrent multi-arm auto-calibration ---

  // Each arm's port as designated on the robot record (assigned in the per-arm
  // flow above). Raw value — may name a port that isn't currently plugged in.
  const slotSavedPort = useCallback(
    (slot: ArmSlot) => draftPort(slot.portField).trim(),
    [draftPort],
  );

  // The port the batch will actually use: the saved port ONLY if it's currently
  // detected. A saved-but-undetected port (arm unplugged, moved, or renamed by
  // the OS) is treated as no port at all — you can't calibrate against an absent
  // bus, and the subprocess would just fail to open it. Single source of truth
  // for batch ports; never re-entered by the user.
  const slotPort = useCallback(
    (slot: ArmSlot) => {
      const saved = slotSavedPort(slot);
      return saved && availablePorts.includes(saved) ? saved : "";
    },
    [slotSavedPort, availablePorts],
  );

  // What the Device cards show as "plugged in right now". Until the first scan
  // lands nothing is known, so every slot reads as detected — otherwise opening
  // the window would flash a warning on arms that are perfectly fine. A rescan
  // keeps the previous list until it resolves, so only the first one needs this.
  const slotPortDetected = useCallback(
    (slot: ArmSlot) => !portsScanned || !!slotPort(slot),
    [portsScanned, slotPort],
  );

  // Single-arm picker: the selected port only counts if it's actually detected.
  // A saved-but-unplugged port is treated as no port — same rule as the batch
  // flow — so calibration can't start against an absent bus. `port` stays set to
  // the saved value so it re-selects automatically once the arm is plugged back
  // in and ports are rescanned.
  const portDetected = !!port && availablePorts.includes(port);

  // Anything holding the hardware disables section 01's controls. Clearing or
  // reassigning a port mid-calibration wouldn't stop the run (the subprocess
  // holds the serial port), it would just desync the UI from the arm being
  // measured.
  const hardwareBusy =
    calibrationStatus.calibration_active || batchAutoCal.active;

  // Swing and wiggle remain available. A single Star/Metal pair also offers
  // protocol detection; multiple matches or bimanual mode disable that shortcut.
  const manualPortIdentification = isBimanual || multipleArmsDetected;

  // The slots the user ticked, in canonical order, with their inputs.
  const selectedBatchSlots = armSlots.filter((s) => batchSelected[s.key]);

  // Whether ANY slot currently has a detected port — gates the "Calibrate
  // all" shortcut below (nothing to select otherwise).
  const anyArmAvailable = armSlots.some((s) => !!slotPort(s));

  // "Calibrate all": the multi-arm entry point. Ticks every slot that has a
  // detected port and opens the batch picker so the user can review (and
  // amend) the selection before confirming — it doesn't skip that
  // confirmation, just the manual per-arm ticking. This is the ONLY path that
  // shows the picker; the per-row button below is single-arm.
  const handleCalibrateAll = () => {
    if (armActionsBlocked) return;
    if (stepCalibration) {
      const slots = armSlots.filter((slot) => !!slotPort(slot));
      if (!slots.length) return;
      setZeroCalQueue(slots);
      setDeviceType(slots[0].device);
      setArm(slots[0].arm);
      setNewCalibFor(String(slots[0].cfgField));
      setCalibrationStatus((status) => ({
        ...status,
        status: "idle",
        error: null,
      }));
      return;
    }
    const next: Record<string, boolean> = {};
    for (const slot of armSlots) {
      if (slotPort(slot)) next[slot.key] = true;
    }
    setBatchSelected(next);
    setBatchAutoCalOpen(true);
    setNewCalibFor((prev) => prev ?? (armSlots[0]?.cfgField as string) ?? null);
  };

  // A calibration-file row's "Auto-calibrate": the row stands for exactly one
  // arm slot (rows and slots are 1:1 on cfgField in both modes), so this ticks
  // that slot alone and goes straight to the same pre-start confirmation the
  // picker uses — a batch of one, reusing all of its machinery (status panel,
  // polling, stop, per-arm default save name, overwrite, motor_power). The
  // multi-arm checkbox list stays closed on this path.
  const handleAutoCalibrateSlot = (slot: ArmSlot) => {
    setBatchSelected({ [slot.key]: true });
    setBatchAutoCalOpen(false);
    setBatchAutoCalPromptOpen(true);
  };

  // Resume the batch panel if a run is in progress (e.g. window reopened).
  useEffect(() => {
    (async () => {
      try {
        const res = await fetchWithHeaders(
          `${baseUrl}/api/v1/auto-calibration-batch-status`,
        );
        const data = await res.json();
        setBatchAutoCal(data);
        if (data.active) {
          setBatchAutoCalOpen(true);
          // The batch panel lives inside a row's "New calibration" panel now —
          // expand one (any row works; the batch is multi-arm) so the running
          // batch is visible on reopen.
          setNewCalibFor((prev) => prev ?? "leader_config");
          // Recover the run's session id so this window can resume renewing
          // its lease (the previous window's heartbeats died with it) and
          // stop it by id. Only when the lease is this tab's — or absent —
          // so a resumed heartbeat can never 409 against another owner.
          try {
            const { session } = await getCurrentSession(
              baseUrl,
              fetchWithHeaders,
            );
            if (
              session?.kind === "auto_calibration" &&
              (session.lease === null || session.owner === tabOwnerId())
            ) {
              setAutoCalSessionId(session.id);
            }
          } catch {
            // identity is a nicety here — the kind-level stop still works
          }
        }
      } catch {
        // ignore
      }
    })();
  }, [baseUrl, fetchWithHeaders]);

  // Arm the "keep the results on screen" flag for the whole life of a run,
  // from whichever path started it — a row's Auto-calibrate, the multi-arm
  // picker, or the resume-on-mount effect above finding one already going.
  // Keying it off `active` rather than setting it at each call site means the
  // flag is guaranteed to be true BEFORE the poll flips `active` to false, so
  // the box never blinks out between the two renders.
  useEffect(() => {
    if (batchAutoCal.active) setBatchAutoCalResultsOpen(true);
  }, [batchAutoCal.active]);

  // Poll batch status + logs while a run is active.
  useEffect(() => {
    if (!batchAutoCal.active) return;
    const id = setInterval(async () => {
      try {
        const res = await fetchWithHeaders(
          `${baseUrl}/api/v1/auto-calibration-batch-status`,
        );
        const data: BatchAutoCalStatus = await res.json();
        setBatchAutoCal(data);
        if (!data.active) {
          setCalibReloadToken((t) => t + 1);
          fetchRobot();
          if (data.failed === 0) {
            toast({
              title: t("robotConfig.batch.toast.finishedTitle", {
                count: data.completed,
              }),
            });
            // Same collapse-on-success behaviour as the manual flow: back to
            // the resting view, nothing left open to dismiss.
            setBatchAutoCalOpen(false);
            setBatchAutoCalResultsOpen(false);
            setNewCalibFor(null);
            setCalibMode(null);
          } else {
            const failure = data.arms.find((arm) => arm.error)?.error;
            toast({
              title: t("robotConfig.batch.toast.issuesTitle"),
              description: failure
                ? calibrationErrorMessage(failure, t)
                : t("robotConfig.batch.summary", {
                    completed: data.completed,
                    failed: data.failed,
                  }),
              variant: data.completed > 0 ? "default" : "destructive",
            });
          }
        }
      } catch {
        // transient; keep polling
      }
    }, 700);
    return () => clearInterval(id);
  }, [batchAutoCal.active, baseUrl, fetchWithHeaders, fetchRobot, toast, t]);

  const startBatchAutoCalibration = async () => {
    setBatchAutoCalPromptOpen(false);
    if (!robotName) return;
    const slots = selectedBatchSlots;
    if (slots.length === 0) {
      toast({
        title: t("robotConfig.batch.toast.noArmsTitle"),
        description: t("robotConfig.batch.toast.noArmsDescription"),
        variant: "destructive",
      });
      return;
    }
    // Ports come from each arm's assignment on the robot record — the batch
    // never re-collects them. Guards mirror the backend; the missing-port case
    // is normally prevented by gating selection on an assigned port.
    const missingPort = slots.find((s) => !slotPort(s));
    if (missingPort) {
      toast({
        title: t("robotConfig.batch.toast.noPortTitle"),
        description: t("robotConfig.batch.toast.noPortDescription", {
          arm: missingPort.label,
        }),
        variant: "destructive",
      });
      return;
    }
    const ports = slots.map((s) => slotPort(s));
    if (new Set(ports).size !== ports.length) {
      toast({
        title: t("robotConfig.batch.toast.duplicatePortTitle"),
        description: t("robotConfig.batch.toast.duplicatePortDescription"),
        variant: "destructive",
      });
      return;
    }

    // Each arm saves to its own default name: the in-use config for that slot,
    // else a per-arm "<robot>_<arm>" (bimanual) / "<robot>" suggestion.
    const arms = slots.map((s) => ({
      device_type: s.device,
      port: slotPort(s),
      config_file:
        ((robot?.[s.cfgField] as string) || "").trim() ||
        (isBimanual ? `${robotName}_${s.arm}` : robotName || ""),
      arm: s.arm,
    }));

    try {
      // Start through the sessions surface: robot NAME plus the per-arm
      // slots. Each arm's port/save-name still travel explicitly — they are
      // this window's resolved values (detected ports, unsaved drafts
      // included), which is why the calibration kinds' options may carry
      // them. Each arm saves to its own default name, so replacing that
      // arm's existing calibration is the expected outcome — overwrite is
      // always on and the old name-taken confirmation is gone. motor_power
      // is the torque slider's CURRENT position (draft, not the saved
      // record) so what the user sees is what the calibration drives at.
      // The owner attaches the lease the heartbeat above renews.
      const { session } = await startSession(baseUrl, fetchWithHeaders, {
        kind: "auto_calibration",
        robot: robotName,
        owner: tabOwnerId(),
        options: {
          arms,
          overwrite: true,
          motor_power: motorPercent,
        },
      });
      setAutoCalSessionId(session.id);
      setBatchAutoCal({
        active: true,
        arms: [],
        total: arms.length,
        completed: 0,
        failed: 0,
        logs: [],
      });
      toast({
        title: t("robotConfig.batch.toast.startedTitle", {
          count: arms.length,
        }),
        description: t("robotConfig.batch.toast.startedDescription"),
      });
    } catch (e) {
      toast({
        title: t("robotConfig.batch.toast.startFailedTitle"),
        // 409 session.held renders as the shared localized "robot is busy"
        // line; every other coded refusal shows the server's own prose.
        description:
          formatSessionHeld(t, e) ??
          calibrationErrorMessage(
            e instanceof ApiError ? (e.detail ?? e.message) : String(e),
            t,
          ),
        variant: "destructive",
      });
    }
  };

  const stopBatchAutoCalibration = async () => {
    try {
      // Stop by session id when this window started (or recovered) it; a 404
      // means the run already ended. The kind-level stop covers a batch whose
      // session id we never learned.
      if (autoCalSessionId) {
        try {
          await stopSession(baseUrl, fetchWithHeaders, autoCalSessionId);
          return;
        } catch (e) {
          if (e instanceof ApiError && e.status === 404) return;
          throw e;
        }
      }
      await fetchWithHeaders(`${baseUrl}/api/v1/stop-auto-calibration-batch`, {
        method: "POST",
      });
    } catch (e) {
      console.error("Failed to stop batch auto-calibration:", e);
    }
  };

  const handleStartCalibration = async () => {
    if (armActionsBlocked) return;
    if (!robotName) {
      toast({
        title: t("robotConfig.calib.toast.noRobotTitle"),
        description: t("robotConfig.calib.toast.noRobotDescription"),
        variant: "destructive",
      });
      return;
    }
    if (!port) {
      toast({
        title: t("robotConfig.calib.toast.missingPortTitle"),
        description: t("robotConfig.calib.toast.missingPortDescription"),
        variant: "destructive",
      });
      return;
    }

    // Optimistically mark as active so the abort prompt already guards a
    // close before the backend reports calibration_active=true. Reverted
    // below if the start request fails.
    setManualCalibLive(true);
    terminalCalibHandled.current = false;
    setCalibrationStatus((status) => ({
      ...status,
      calibration_active: true,
      status: "connecting",
      error: null,
    }));

    try {
      // Start through the sessions surface: robot NAME + the slot
      // (device_type/arm) plus this window's port pick and save name — the
      // port may be an unsaved draft, which is why calibration's options
      // carry it (the backend writes it into the record on success). The
      // owner attaches the lease the heartbeat above renews.
      const { session } = await startSession(baseUrl, fetchWithHeaders, {
        kind: "calibration",
        robot: robotName,
        owner: tabOwnerId(),
        options: {
          device_type: deviceType as "robot" | "teleop",
          arm,
          port,
          config_file: calibrationConfigName,
          // The name is always the robot's own default for this slot, so
          // replacing its existing calibration is the expected outcome —
          // overwrite is always on and the old name-taken confirmation
          // prompt is gone. To keep the old calibration, rename it afterward
          // via the per-side rename feature.
          overwrite: true,
        },
      });
      setCalibSessionId(session.id);
      toast({
        title: t("robotConfig.calib.toast.startedTitle"),
        // `deviceType` is the backend enum ("teleop"/"robot") — the VALUE is
        // untouched; only its rendered label is localized, falling back to
        // the raw string for anything unmapped.
        description: t("robotConfig.calib.toast.startedDescription", {
          device: t(`robotConfig.deviceValue.${deviceType}` as never, {
            defaultValue: deviceType,
          }),
        }),
      });
      setIsPolling(true);
    } catch (error) {
      setManualCalibLive(false);
      setZeroCalQueue([]);
      setCalibrationStatus((status) => ({
        ...status,
        calibration_active: false,
        status: "idle",
      }));
      if (error instanceof ApiError) {
        // 409 session.held renders as the shared localized "robot is busy"
        // line; every other coded refusal shows the server's own prose.
        toast({
          title: t("robotConfig.calib.toast.startFailedTitle"),
          description:
            formatSessionHeld(t, error) ??
            calibrationErrorMessage(error.detail, t),
          variant: "destructive",
        });
      } else {
        console.error("Error starting calibration:", error);
        toast({
          title: t("robotConfig.calib.toast.errorTitle"),
          description: t("robotConfig.calib.toast.startError"),
          variant: "destructive",
        });
      }
    }
  };

  const handleStopCalibration = async () => {
    setZeroCalQueue([]);
    try {
      // Stop by session id (a 404 means the session already ended — fine);
      // fall back to the kind-level stop when this window never started one
      // (e.g. a calibration left running by another tab). `result` is
      // calibrate.py's own stop-handler response either way.
      let result: { success?: boolean; message?: string };
      if (calibSessionId) {
        try {
          ({ result } = (await stopSession(
            baseUrl,
            fetchWithHeaders,
            calibSessionId,
          )) as { result: { success?: boolean; message?: string } });
        } catch (e) {
          if (!(e instanceof ApiError && e.status === 404)) throw e;
          result = { success: true };
        }
      } else {
        const response = await fetchWithHeaders(
          `${baseUrl}/api/v1/stop-calibration`,
          { method: "POST" },
        );
        result = await response.json();
      }

      if (result.success) {
        // The 200ms polling interval will pick up the stopped state.
        toast({
          title: t("robotConfig.calib.toast.stoppedTitle"),
          description: t("robotConfig.calib.toast.stoppedDescription"),
        });
      } else {
        toast({
          title: t("robotConfig.calib.toast.errorTitle"),
          description:
            result.message || t("robotConfig.calib.toast.stopFailedFallback"),
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error("Error stopping calibration:", error);
      toast({
        title: t("robotConfig.calib.toast.errorTitle"),
        description: t("robotConfig.calib.toast.stopFailedFallback"),
        variant: "destructive",
      });
    }
  };

  // `step` is the wizard step this click confirms. The step wizard sends the
  // one it is showing, and the backend refuses a confirm for any other step
  // ("Step N is not the current step") — the second half of the double-click
  // guard above, for a click that lands after the step already advanced. The
  // range-sweep flow has no step to name and posts no body, as before.
  const handleCompleteStep = async (step?: number) => {
    if (!calibrationStatus.calibration_active) return;
    if (completingStepRef.current) return;
    completingStepRef.current = true;
    setCompletingStep(true);

    try {
      const response = await fetchWithHeaders(
        `${baseUrl}/api/v1/complete-calibration-step`,
        step === undefined
          ? { method: "POST" }
          : {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ step }),
            },
      );

      const data = await response.json();

      // The response acknowledges the click; the worker has not saved yet.
      // Success and failure toasts come from its terminal status instead.
      if (!response.ok || !data.success) {
        toast({
          title: t("robotConfig.calib.toast.stepFailedTitle"),
          description: calibrationErrorMessage(data.detail || data.message, t),
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error("Error completing step:", error);
      toast({
        title: t("robotConfig.calib.toast.errorTitle"),
        description: t("robotConfig.calib.toast.stepError"),
        variant: "destructive",
      });
    } finally {
      completingStepRef.current = false;
      setCompletingStep(false);
    }
  };

  useEffect(() => {
    if (calibrationStatus.status !== "error" || terminalCalibHandled.current)
      return;
    terminalCalibHandled.current = true;
    setZeroCalQueue([]);
    toast({
      title: t("robotConfig.calib.toast.startFailedTitle"),
      description: calibrationErrorMessage(
        calibrationStatus.error || calibrationStatus.message,
        t,
      ),
      variant: "destructive",
    });
  }, [
    calibrationStatus.status,
    calibrationStatus.error,
    calibrationStatus.message,
    toast,
    t,
  ]);

  useEffect(() => {
    if (!isPolling) return;
    // Serialize reads: a slow hardware read must not overwrite a newer result.
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      await pollStatus(() => !cancelled);
      if (!cancelled) timer = setTimeout(poll, 200);
    };
    void poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // pollStatus is stable enough — it only reads via fetchWithHeaders + setState.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPolling]);

  // Keep the port field in sync with the selected device_type + arm's saved
  // port whenever either changes (single uses leader/follower; bimanual right
  // uses the right_* fields). Port is a dropdown, so overwriting it is safe.
  useEffect(() => {
    if (!robot) return;
    setPort(draftPort(portField) || "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deviceType, arm, robot, portDraft]);

  // Refresh the saved slot, then open the next queued CAN arm's pose guide.
  useEffect(() => {
    if (
      calibrationStatus.status !== "completed" ||
      terminalCalibHandled.current
    )
      return;
    terminalCalibHandled.current = true;
    // A completed calibration may have written a new named file — nudge the
    // per-side libraries to re-fetch their config lists so it shows up.
    setCalibReloadToken((t) => t + 1);
    fetchRobot();
    // Success collapses the panel; failures leave the controls open for retry.
    toast({ title: t("robotConfig.calib.completed") });
    const remaining = zeroCalQueue.slice(1);
    setZeroCalQueue(remaining);
    if (remaining.length) {
      const next = remaining[0];
      setDeviceType(next.device);
      setArm(next.arm);
      setNewCalibFor(String(next.cfgField));
      setCalibrationStatus((status) => ({
        ...status,
        status: "idle",
        error: null,
      }));
      return;
    }
    setNewCalibFor(null);
    setCalibMode(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [calibrationStatus.status, fetchRobot, zeroCalQueue]);

  // Stage the current side's port into the local draft (no network write). A
  // re-detected USB port (which shuffles on reboot/reconnect) is recorded here
  // and only committed on Save. An empty string is a valid value: it CLEARS the
  // assignment (arm disconnected). The batched Save sends every dirty port slot
  // together so the backend's duplicate-port guard sees the merged record.
  // `field` defaults to the selected slot for legacy callers; section 01's
  // per-row controls pass their own slot explicitly, so an edit never depends
  // on a selection state-update having landed first.
  const persistPort = useCallback(
    (nextPort: string, field: keyof RobotRecord = portField) => {
      if (!robotName) return;
      setPortDraft((prev) => ({ ...prev, [field]: nextPort }));
    },
    [robotName, portField],
  );

  // Stage several port slots at once into the draft (used by Detect's reassign
  // path: clear the stale slot AND set the current one in one edit). Both land
  // in the same batched Save request, so the backend's duplicate-port guard —
  // which evaluates the prospective merged record — passes for a legitimate
  // swap. Returns the applied patch (truthy) so callers can gate their toast.
  const persistPorts = useCallback(
    (patch: Partial<Record<keyof RobotRecord, string>>) => {
      if (!robotName) return null;
      setPortDraft((prev) => ({ ...prev, ...patch }));
      return patch;
    },
    [robotName],
  );

  // --- Auto-calibration torque (per-robot, persisted) --------------------
  // The backend stores motor_power as a PERCENT of full torque (10-100; see
  // makermodslab/utils/config.py clamp_motor_power). It is the torque the
  // AUTO-CALIBRATION subprocess drives the arm at (threaded through as its
  // --torque-limit = percent × 10; see makermodslab/auto_calibrate.py). Regular
  // sessions (teleop/record/policy runs) run at stock LeRobot torque and
  // ignore this value. The UI below is expressed in RAW Torque_Limit register
  // units (0-1000) — the same scale as the vendored script's
  // DEFAULT_TORQUE_LIMIT = 380 — so operators can reason in one vocabulary.
  // We convert raw<->percent at the edges and persist a percent.
  const TORQUE_LIMIT_PER_PERCENT = 10; // must match makermodslab/motor_power.py
  const MOTOR_POWER_MIN_PERCENT = 10; // must match makermodslab/utils/config.py
  const MOTOR_POWER_MAX_PERCENT = 100; // must match makermodslab/utils/config.py
  const TORQUE_LIMIT_MIN = MOTOR_POWER_MIN_PERCENT * TORQUE_LIMIT_PER_PERCENT; // 100
  const TORQUE_LIMIT_MAX = MOTOR_POWER_MAX_PERCENT * TORQUE_LIMIT_PER_PERCENT; // 1000
  // The vendored script's own operating torque, shown as a reference marker.
  const DEFAULT_TORQUE_LIMIT_REF = 380; // makermodslab/vendor/.../calibration_defaults.py

  // Local slider position (in PERCENT). Held as a draft and committed to the
  // robot record only on Save; an auto-calibration START sends the current
  // draft directly, so the slider is WYSIWYG even before saving. Fallback
  // matches backend DEFAULT_MOTOR_POWER (38% = Torque_Limit 380); re-syncs
  // from the baseline whenever the saved value changes.
  const [powerDraft, setPowerDraft] = useState(38);
  const savedHoldingTorque = robot?.gripper_hold_torque_nm === undefined
    ? (robot?.gripper_current_limit_a == null ? 0.5 : null)
    : robot.gripper_hold_torque_nm;
  const [holdingDraft, setHoldingDraft] = useState<number | null>(0.5);
  useEffect(() => {
    setHoldingDraft(savedHoldingTorque);
  }, [robot?.name, savedHoldingTorque]);
  useEffect(() => {
    setPowerDraft(robot?.motor_power ?? 38);
  }, [robot?.motor_power]);

  // Slider is in raw Torque_Limit units; convert to the percent the draft holds.
  const torqueLimitDraft = Math.round(powerDraft) * TORQUE_LIMIT_PER_PERCENT;
  // The integer percent the draft would persist, clamped to the backend's 10-100.
  const motorPercent = Math.min(100, Math.max(10, Math.round(powerDraft)));

  // --- Draft dirtiness + batched Save ------------------------------------
  // A field is dirty when its draft differs from the last-fetched baseline.
  // Save is the ONLY path that writes the record; it POSTs every dirty field in
  // one request (batching matters for ports: the backend's duplicate-port guard
  // evaluates the merged record, so clearing one slot and assigning another must
  // arrive together). Server-side writes elsewhere (calibration completion,
  // config assignment via the library) are out of scope and untouched.
  const camerasDirty = useMemo(
    () =>
      !!robot &&
      JSON.stringify(cameras ?? []) !== JSON.stringify(robot.cameras ?? []),
    [cameras, robot],
  );
  const portsDirty = useMemo(
    () =>
      !!robot &&
      Object.entries(portDraft).some(
        ([f, v]) =>
          (v ?? "") !== ((robot[f as keyof RobotRecord] as string) || ""),
      ),
    [portDraft, robot],
  );
  const motorDirty = !!robot && motorPercent !== robot.motor_power;
  const holdingDirty = !!robot && armType === "metal" && holdingDraft !== savedHoldingTorque;
  const holdingValid = holdingDraft === null || (Number.isFinite(holdingDraft) && holdingDraft >= 0.1 && holdingDraft <= 2);
  const armsDirty = !!robot && draftArms !== (robot.arms ?? "both");
  const isDirty = camerasDirty || portsDirty || motorDirty || armsDirty || holdingDirty;

  const handleSave = useCallback(async () => {
    if (!robotName || !robot || !holdingValid) return;
    const patch: Record<string, unknown> = {};
    if (camerasDirty) patch.cameras = cameras;
    if (motorDirty) patch.motor_power = motorPercent;
    if (holdingDirty) {
      patch.gripper_hold_torque_nm = holdingDraft;
      if (robot.gripper_current_limit_a != null) patch.gripper_current_limit_a = null;
    }
    if (armsDirty) patch.arms = draftArms;
    if (portsDirty) {
      for (const [f, v] of Object.entries(portDraft)) {
        if ((v ?? "") !== ((robot[f as keyof RobotRecord] as string) || "")) {
          patch[f] = v ?? "";
        }
      }
    }
    if (Object.keys(patch).length === 0) return;
    setSaving(true);
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/api/v1/robots/${encodeURIComponent(robotName)}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(patch),
        },
      );
      const data = await res.json();
      if (res.ok && data.robot) {
        // Adopt the server record as the new baseline and clear the drafts.
        // powerDraft re-syncs via its effect when motor_power changes.
        setRobot(data.robot);
        setPortDraft({});
        setArmsDraft(null);
        setCameras((data.robot as RobotRecord).cameras ?? []);
        setJustSaved(true);
        toast({ title: t(data.gripper_application === "live"
          ? "robotConfig.advanced.holdingApplied"
          : data.gripper_application === "next_session"
            ? "robotConfig.advanced.holdingSaved"
            : "robotConfig.window.toast.saved") });
      } else {
        // Surface the backend guard (e.g. duplicate-port 409) and stay put.
        toast({
          title: t("robotConfig.window.toast.saveFailedTitle"),
          description:
            data.message || t("robotConfig.window.toast.saveFailedFallback"),
          variant: "destructive",
        });
      }
    } catch (e) {
      toast({
        title: t("robotConfig.window.toast.saveFailedTitle"),
        description: String(e),
        variant: "destructive",
      });
    } finally {
      setSaving(false);
    }
  }, [
    robotName,
    robot,
    camerasDirty,
    motorDirty,
    holdingDirty,
    holdingDraft,
    holdingValid,
    portsDirty,
    armsDirty,
    draftArms,
    cameras,
    motorPercent,
    portDraft,
    baseUrl,
    fetchWithHeaders,
    toast,
    t,
  ]);

  // Every close vector (Quit button, X, Esc, overlay click) funnels here.
  // A live manual calibration confirms the abort first; unsaved drafts prompt
  // for discard, so an accidental close never silently loses either.
  const requestClose = useCallback(() => {
    if (manualCalibLive) {
      setAbortPromptOpen(true);
      return;
    }
    if (isDirty) {
      setQuitPromptOpen(true);
      return;
    }
    onOpenChange(false);
  }, [manualCalibLive, isDirty, onOpenChange]);

  const confirmQuit = useCallback(() => {
    setQuitPromptOpen(false);
    onOpenChange(false);
  }, [onOpenChange]);

  // Confirmed abort-and-close: fire the stop explicitly, then close. The
  // retired exit guard used to do this from its unmount cleanup; without it
  // the lease would still safety-stop the abandoned session, but the user
  // asked for the abort NOW — the arm shouldn't sit claimed for the lease
  // timeout. Best-effort: a failure here is the lease's problem.
  const confirmAbortAndClose = useCallback(() => {
    setAbortPromptOpen(false);
    const stop = calibSessionId
      ? stopSession(baseUrl, fetchWithHeaders, calibSessionId).then(() => {})
      : fetchWithHeaders(`${baseUrl}/api/v1/stop-calibration`, {
          method: "POST",
        }).then(() => {});
    stop.catch((e) => console.error("Failed to stop calibration on close:", e));
    onOpenChange(false);
  }, [onOpenChange, calibSessionId, baseUrl, fetchWithHeaders]);

  const getStatusDisplay = () => {
    switch (calibrationStatus.status) {
      case "idle":
        return {
          color: "bg-muted-foreground",
          text: t("robotConfig.calib.status.idle"),
        };
      case "connecting":
        return {
          color: "bg-warn",
          text: t("robotConfig.calib.status.connecting"),
        };
      case "recording":
        return {
          color: "bg-info",
          text: t("robotConfig.calib.status.recording"),
        };
      // Step wizard — see CalibrationStatus.status.
      case "awaiting_step":
        return {
          color: "bg-info",
          text: t("robotConfig.calib.status.awaitingZero"),
        };
      case "saving":
        return {
          color: "bg-warn",
          text: t("robotConfig.calib.status.saving"),
        };
      case "completed":
        return {
          color: "bg-ok",
          text: t("robotConfig.calib.status.completed"),
        };
      case "error":
        return {
          color: "bg-destructive",
          text: t("robotConfig.calib.status.error"),
        };
      case "stopping":
        return {
          color: "bg-warn",
          text: t("robotConfig.calib.status.stopping"),
        };
      default:
        return {
          color: "bg-muted-foreground",
          text: t("robotConfig.calib.status.unknown"),
        };
    }
  };

  const statusDisplay = getStatusDisplay();

  // The expandable "New calibration" panel, rendered under whichever
  // calibration-file row's + button is active. The controls and status form
  // the main vertical on the left; the (large) demo video sits beside them so
  // it doesn't push the controls down. The auto-calibration torque slider is
  // tucked under an Advanced settings disclosure. `rowSlot` is the arm slot
  // the row stands for — what its "Auto-calibrate" button targets.
  const newCalibrationPanel = (rowLabel: string, rowSlot?: ArmSlot) => {
    const running = calibrationStatus.calibration_active;
    const batchBusy =
      batchAutoCal.active || batchAutoCalOpen || batchAutoCalResultsOpen;
    // The pre-start stack (mode choice, demo, pose, advanced, Start) shows
    // only while nothing is running and no batch UI is up. Everything below
    // it is one column, in the order things happen: choose, watch, pose,
    // start, follow the live data, save.
    const preStart = !running && !batchBusy;
    // The auto/manual choice exists only for the sweep flows; a step or
    // panel family never renders either branch, whatever the toggle holds.
    const mode = kind === "range_sweep" ? calibMode : null;

    // The auto-calibration preamble: demo clip, the pose to start from, the
    // safety note, and the drive torque. A batch run and a single-arm run are
    // the SAME procedure on N arms instead of one, so they show the SAME
    // preamble — it lives here so the two can never drift apart.
    // ONE arm list across all three phases of a batch. Before Start it is the
    // picker: a tick box and the port each arm will be driven on. After Start
    // the SAME rows report status where the port was. There used to be a
    // second list underneath carrying the status, which repeated every arm's
    // name directly below itself.
    const picking = batchAutoCalOpen && !batchAutoCal.active;
    const armRunStatus = (slot: ArmSlot) => {
      const port = slotPort(slot);
      return batchAutoCal.arms.find(
        (a) =>
          (!!port && a.port === port) ||
          (a.device_type === slot.device && a.arm === slot.arm),
      );
    };
    // While picking, every slot is offerable. Once a run exists, only the arms
    // actually in it — a row's own "Auto-calibrate" drives one arm, and
    // listing the other three idle beside it would read as a stalled batch.
    const listedSlots = batchAutoCalOpen
      ? armSlots
      : armSlots.filter((s) => armRunStatus(s));

    // The step wizard's pose reference and words, shared by the pre-start
    // card and the running wizard so the picture the arm was posed against
    // does not change between the two screens.
    //
    // Text: the catalog's per-id override for the built-ins (which is what
    // localizes it), else the family's own words from the manifest — backend
    // text, English in every language, the same way server messages are.
    // Before Start that is the side's `calibration.summary`; while running it
    // is the step the family published, and the override applies to step 1
    // ONLY — later steps are the family's own words (the built-ins have one).
    //
    // Image: the bundled photo for the built-ins (ZERO_POSE_IMAGES), else the
    // one the family serves (the summary's before Start, the step's while
    // running); neither → no image, the text stands alone.
    //
    // object-cover, not contain: the follower sources are 4:3 on white with the
    // arm in the middle band, so a 16:9 centre crop trims background, not
    // hardware. The dedicated leader reference is already 16:9.
    const isLeaderStep = deviceType === "teleop";
    const stepSide = isLeaderStep ? "leader" : "follower";
    const bundledAssets = ZERO_POSE_IMAGES[armType];
    // A non-default leader (the Metal arm's own leader) is this family's
    // arm: its zero pose is the FOLLOWER's, so the follower photo stands in
    // for the Star leader's reference, and the catalog's `leader_<kind>`
    // entry (if any) overrides the wording.
    const leaderIsOwnArm = isLeaderStep && leaderEnergized;
    const leaderKindSuffix =
      isLeaderStep && multiLeader && leaderKind !== armInfo?.default_leader_kind
        ? `_${leaderKind}`
        : "";
    const bundledImage = bundledAssets
      ? {
          src:
            isLeaderStep && !leaderIsOwnArm
              ? bundledAssets.leader
              : bundledAssets.follower,
          alt: leaderIsOwnArm
            ? t(
                `robotConfig.calib.zeroPose.leaderPoseImageFor.${armType}.${leaderKind}` as never,
                {
                  defaultValue: t(
                    `robotConfig.calib.zeroPose.${bundledAssets.altKey}`,
                  ),
                },
              )
            : isLeaderStep
              ? t("robotConfig.calib.zeroPose.poseImageLeader")
              : t(`robotConfig.calib.zeroPose.${bundledAssets.altKey}`),
        }
      : null;
    const servedImage = (url: string | null) => {
      const src = servedUrl(baseUrl, url);
      // No caption of our own: the step text beside it describes the pose,
      // and a bundled alt would mislabel a picture this code has never seen.
      return src ? { src, alt: "" } : null;
    };
    const stepImage = (image: { src: string; alt: string } | null) =>
      image ? (
        <figure>
          <img
            src={image.src}
            alt={image.alt}
            loading="lazy"
            className="aspect-video w-full rounded-md border border-border bg-muted object-cover"
          />
          {bundledAssets && (
            <figcaption className="mt-2 text-xs text-muted-foreground">
              {t("robotConfig.calib.zeroPose.poseCaption")}
            </figcaption>
          )}
        </figure>
      ) : null;
    const overrideText = (fallback: string): string =>
      t(
        `robotConfig.calib.zeroPose.instructionsFor.${armType}.${stepSide}${leaderKindSuffix}` as never,
        { defaultValue: fallback },
      );
    // The leader side's summary is the SELECTED leader's on a multi-leader
    // family (the manifest carries one per option); otherwise the family's.
    const summarySide =
      isLeaderStep && multiLeader
        ? (leaderChoice?.calibration_summary ?? null)
        : (armInfo?.calibration.summary?.[stepSide] ?? null);
    const preStartText = summarySide ? overrideText(summarySide.text) : "";
    const preStartImage =
      bundledImage ?? servedImage(summarySide?.image_url ?? null);
    const onFirstStep = calibrationStatus.step <= 1;
    const stepText = onFirstStep
      ? overrideText(calibrationStatus.message)
      : calibrationStatus.message;
    const stepImageNow =
      servedImage(calibrationStatus.image_url) ??
      (onFirstStep ? bundledImage : null);

    const autoPreamble = (
      <>
        <CalibrationClip
          src={AUTO_CAL_CLIP}
          poster={AUTO_CAL_POSTER}
          label={t("robotConfig.calib.videoAuto")}
          unsupported={t("robotConfig.calib.videoUnsupported")}
          linkLabel={t("robotConfig.calib.videoLink")}
        />
        {/* The auto-calibration start pose. The SO-101's is its folded
            RESTING pose, so this is the arm card's product photo — one file,
            not a second shot of the same thing. The caption says in words
            what the picture cannot: that the arm has to be put there BEFORE
            Start, because auto-calibration drives from wherever it finds the
            arm and a mid-air start swings it into the bench. */}
        <figure className="space-y-2">
          <img
            src={so101ArmPhoto}
            alt={t("robotConfig.calib.poseAutoStart")}
            loading="lazy"
            className="aspect-video w-full rounded-md border border-border bg-muted object-cover"
          />
          <figcaption className="text-xs text-muted-foreground">
            {t("robotConfig.calib.restingPoseCaption")}
          </figcaption>
        </figure>
        <Alert className="border-info/40 bg-info/10 text-info">
          <Activity className="h-4 w-4" />
          <AlertDescription>{t("robotConfig.calib.autoNote")}</AlertDescription>
        </Alert>
        {robot && autoCalibration && showFollower && (
          <Collapsible className="group space-y-3">
            <CollapsibleTrigger className="flex w-full items-start justify-between border-b border-border pb-2 text-sm font-semibold text-foreground">
              <span className="text-left">
                <span className="block">{t("robotConfig.advanced.title")}</span>
                <span className="block text-xs font-normal text-muted-foreground">
                  {t("robotConfig.advanced.subtitle")}
                </span>
              </span>
              <ChevronDown className="mt-0.5 h-4 w-4 shrink-0 transition-transform group-data-[state=open]:rotate-180" />
            </CollapsibleTrigger>
            <CollapsibleContent className={SLIDE}>
              <div className="space-y-2">
                <Label htmlFor="motorPower" className="text-sm font-medium">
                  {t("robotConfig.advanced.torqueLabel")}
                </Label>
                <div className="flex items-center gap-3">
                  <input
                    id="motorPower"
                    type="range"
                    min={TORQUE_LIMIT_MIN}
                    max={TORQUE_LIMIT_MAX}
                    step={TORQUE_LIMIT_PER_PERCENT}
                    value={torqueLimitDraft}
                    onChange={(e) => {
                      // Slider is in raw Torque_Limit units; store as percent.
                      setPowerDraft(
                        Number(e.target.value) / TORQUE_LIMIT_PER_PERCENT,
                      );
                    }}
                    list="motorTorqueTicks"
                    className="h-1.5 flex-1 cursor-pointer accent-primary"
                    aria-label={t("robotConfig.advanced.torqueSliderLabel")}
                  />
                  <datalist id="motorTorqueTicks">
                    {/* The vendored script's stock torque, as a reference tick. */}
                    <option value={DEFAULT_TORQUE_LIMIT_REF} />
                  </datalist>
                  <span className="w-12 shrink-0 text-right font-mono text-sm text-foreground">
                    {torqueLimitDraft}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground">
                  <Trans
                    i18nKey="robotConfig.advanced.torqueHint"
                    values={{
                      ref: DEFAULT_TORQUE_LIMIT_REF,
                      min: TORQUE_LIMIT_MIN,
                    }}
                    components={[<code key="0" />]}
                  />
                </p>
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}
      </>
    );

    return (
      <div className="ml-6 mt-2 space-y-3 rounded-md border border-border bg-muted/20 p-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-foreground">
            {t("robotConfig.calib.panelTitle", { row: rowLabel })}
          </span>
          <span className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
            <span
              aria-hidden
              className={`inline-block h-2 w-2 rounded-full ${statusDisplay.color}`}
            />
            {statusDisplay.text}
          </span>
        </div>

        {running ? (
          <Button
            onClick={handleStopCalibration}
            variant="destructive"
            className="w-full"
          >
            <Square className="mr-2 h-4 w-4" />
            {t("robotConfig.calib.cancel")}
          </Button>
        ) : batchAutoCal.active ? (
          <Button
            onClick={stopBatchAutoCalibration}
            variant="destructive"
            className="w-full"
          >
            <Square className="mr-2 h-4 w-4" />
            {batchAutoCal.total === 1
              ? t("robotConfig.batch.stopSingle")
              : t("robotConfig.batch.stopAll")}
          </Button>
        ) : null}

        {/* Mode first. The two flows differ in video, pose, and what happens
            after Start, so nothing renders until one is picked. A step
            family has exactly one flow and skips the question. */}
        {preStart && kind === "range_sweep" && (
          <div className="grid grid-cols-2 gap-2">
            <Button
              type="button"
              variant={mode === "auto" ? "default" : "outline"}
              onClick={() => setCalibMode(calibMode === "auto" ? null : "auto")}
            >
              <Wand2 className="mr-2 h-4 w-4" />
              {t("robotConfig.calib.auto")}
            </Button>
            <Button
              type="button"
              variant={mode === "manual" ? "default" : "outline"}
              onClick={() =>
                setCalibMode(calibMode === "manual" ? null : "manual")
              }
            >
              <Play className="mr-2 h-4 w-4" />
              {t("robotConfig.calib.manual")}
            </Button>
          </div>
        )}

        {/* Manual, before Start: demo video, the start pose (middle position),
            advanced parameters, Start. One column, in that order. */}
        {preStart && mode === "manual" && (
          <>
            <CalibrationClip
              src={MANUAL_CAL_CLIP}
              poster={MANUAL_CAL_POSTER}
              label={t("robotConfig.calib.demoTitle")}
              unsupported={t("robotConfig.calib.videoUnsupported")}
              linkLabel={t("robotConfig.calib.videoLink")}
            />
            <figure className="space-y-2">
              <img
                src={so101ManualStartPose}
                alt={t("robotConfig.calib.poseMiddle")}
                loading="lazy"
                className="aspect-video w-full rounded-md border border-border bg-muted object-cover"
              />
              <figcaption className="text-xs text-muted-foreground">
                {t("robotConfig.calib.middlePoseCaption")}
              </figcaption>
            </figure>
            {/* No Advanced parameters here on purpose. The only thing it
                holds is the auto-calibration drive torque, and that value
                is read exclusively by the auto-calibration subprocess:
                manual calibration never sends it, so offering it here
                would imply it changes something about this run. */}
            <Button
              onClick={() => handleStartCalibration()}
              disabled={
                !robotName || !deviceType || !portDetected || manualCalibLive
              }
              className="w-full"
            >
              <Play className="mr-2 h-4 w-4" />
              {t("robotConfig.calib.start")}
            </Button>
          </>
        )}

        {/* Auto, before Start: same order as manual so the two flows read as
            one design. The demo clip is not shot yet, so it is a slot. */}
        {preStart && mode === "auto" && (
          <>
            {autoPreamble}
            <Button
              onClick={() => rowSlot && handleAutoCalibrateSlot(rowSlot)}
              disabled={!robotName || !rowSlot || !slotPort(rowSlot)}
              className="w-full"
              title={
                rowSlot && slotPort(rowSlot)
                  ? t("robotConfig.calib.autoTitle", {
                      arm: rowSlot.label,
                      port: slotPort(rowSlot),
                    })
                  : t("robotConfig.calib.autoDisabledTitle")
              }
            >
              <Wand2 className="mr-2 h-4 w-4" />
              {t("robotConfig.calib.start")}
            </Button>
          </>
        )}

        {/* Steps, before Start: one flow, same shape. */}
        {preStart && stepCalibration && (
          <>
            {zeroCalQueue.length > 0 && (
              <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                <span>
                  {t("robotConfig.calib.zeroPose.sequence", {
                    count: zeroCalQueue.length,
                  })}
                </span>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setZeroCalQueue([])}
                >
                  {t("robotConfig.calib.zeroPose.cancelAll")}
                </Button>
              </div>
            )}
            {/* No demo clip slot here. A step is one act — pose the arm by
                hand and press the button — so there is nothing to
                demonstrate that the photo below does not already show. The
                sweep flows keep their video because the MOTION is the thing
                being taught there; a static pose is not. */}
            {stepImage(preStartImage)}
            <Alert className="border-info/40 bg-info/10 text-info">
              <Activity className="h-4 w-4" />
              <AlertDescription>
                {/* The side's summary — what to have the arm in before
                    Start. A family that answered nothing for this side gets
                    the generic note. */}
                {preStartText || t("robotConfig.calib.zeroNote")}
              </AlertDescription>
            </Alert>
            <Button
              onClick={() => handleStartCalibration()}
              disabled={
                !robotName || !deviceType || !portDetected || armActionsBlocked
              }
              className="w-full"
            >
              <Play className="mr-2 h-4 w-4" />
              {t("robotConfig.calib.zeroPose.start")}
            </Button>
          </>
        )}

        {calibrationStatus.status === "connecting" && (
          <Alert className="border-warn/40 bg-warn/10 text-warn">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              {t("robotConfig.calib.connecting")}
            </AlertDescription>
          </Alert>
        )}

        {calibrationStatus.status === "awaiting_step" && (
          <div className="space-y-3">
            {/* Reference pose. The words alone have never been enough here:
                "folded against the base, gripper fully closed" is a shape, and
                a picture of the shape is what the user actually matches the
                arm against. It sits ABOVE the instructions because it is the
                thing being described, and above the Next button because the
                arm has to be in this position before the step is confirmed.

                On step 1 of a built-in this is the same picture the pre-start
                card showed, so the pose the arm was matched against does not
                change between the two screens; a family that serves a step
                image gets that one instead. */}
            {stepImage(stepImageNow)}
            <Alert className="border-info/40 bg-info/10 text-info">
              <Activity className="h-4 w-4" />
              <AlertDescription>{stepText}</AlertDescription>
            </Alert>

            {/* Live readings only where the family asked for them: a step
                that is not about joint positions (a button press, a cable)
                has nothing to show here, and the backend reads the bus only
                while `live_positions` is set. */}
            {calibrationStatus.live_positions &&
              calibrationStatus.current_positions &&
              Object.keys(calibrationStatus.current_positions).length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Activity className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm font-medium text-foreground">
                      {t("robotConfig.calib.zeroPose.liveAngles")}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                    {orderedJointEntries(
                      calibrationStatus.current_positions,
                    ).map(([motor, angle]) => (
                      <div
                        key={motor}
                        className="flex items-baseline justify-between gap-2 border-b border-border/50 py-0.5"
                      >
                        {/* Motor names are DATA (they key the calibration
                              file and the dataset's feature columns), so they
                              render verbatim in every language. */}
                        <span className="truncate font-mono text-xs text-muted-foreground">
                          {motor}
                        </span>
                        <span className="shrink-0 font-mono text-xs tabular-nums text-foreground">
                          {angle.toFixed(1)}&deg;
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            <Button
              onClick={() => handleCompleteStep(calibrationStatus.step)}
              disabled={!calibrationStatus.calibration_active || completingStep}
              className="w-full bg-ok text-primary-foreground hover:bg-ok/90"
            >
              <CheckCircle className="mr-2 h-4 w-4" />
              {t("robotConfig.calib.zeroPose.confirm")}
            </Button>
          </div>
        )}

        {calibrationStatus.status === "saving" && (
          <Alert className="border-warn/40 bg-warn/10 text-warn">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              {t("robotConfig.calib.zeroPose.saving")}
            </AlertDescription>
          </Alert>
        )}

        {/* Manual sweep, after Start: one short combined note (what to do AND
            the torque warning, one alert instead of two), the live data kept
            compact, and Save at the end where the flow ends. */}
        {calibrationStatus.status === "recording" &&
          (() => {
            const ranges = calibrationStatus.recorded_ranges ?? {};
            const motors = orderedJointEntries(ranges);
            const allComplete =
              motors.length > 0 &&
              motors.every(([motor, range]) =>
                isMotorRangeComplete(
                  calibrationStatus.device_type,
                  motor,
                  range.max - range.min,
                ),
              );
            return (
              <div className="space-y-3">
                <Alert className="border-warn/40 bg-warn/10 text-warn">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    {t("robotConfig.calib.sweepNote")}
                  </AlertDescription>
                </Alert>
                {motors.length > 0 && (
                  <div className="rounded-md border border-border bg-muted/30 p-3">
                    <div className="grid grid-cols-2 gap-x-5 gap-y-2">
                      {motors.map(([motor, range]) => {
                        const totalRange = range.max - range.min;
                        const currentOffset = range.current - range.min;
                        const progressPercent =
                          totalRange > 0
                            ? (currentOffset / totalRange) * 100
                            : 50;
                        const rangeComplete = isMotorRangeComplete(
                          calibrationStatus.device_type,
                          motor,
                          totalRange,
                        );
                        return (
                          <div key={motor} className="space-y-1">
                            <div className="flex items-baseline justify-between gap-2">
                              {/* Motor names are DATA (calibration file keys),
                                  rendered verbatim in every language. */}
                              <span className="flex items-center gap-1 truncate font-mono text-xs text-muted-foreground">
                                {motor}
                                {rangeComplete && (
                                  <CheckCircle
                                    className="h-3 w-3 shrink-0 text-ok"
                                    aria-label={t(
                                      "robotConfig.calib.rangeComplete",
                                    )}
                                  />
                                )}
                              </span>
                              <span className="shrink-0 font-mono text-xs tabular-nums text-foreground">
                                {range.current}
                              </span>
                            </div>
                            <div className="relative h-1.5 w-full rounded-full bg-secondary">
                              <div
                                className={`absolute top-0 h-1.5 w-1 rounded-full transition-all duration-100 ${
                                  rangeComplete ? "bg-ok" : "bg-warn"
                                }`}
                                style={{
                                  left: `${Math.max(
                                    0,
                                    Math.min(100, progressPercent),
                                  )}%`,
                                  transform: "translateX(-50%)",
                                }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
                <Button
                  onClick={() => handleCompleteStep()}
                  disabled={
                    !calibrationStatus.calibration_active || completingStep
                  }
                  className={`w-full text-primary-foreground ${
                    allComplete
                      ? "bg-ok hover:bg-ok/90"
                      : "bg-warn hover:bg-warn/90"
                  }`}
                >
                  {allComplete ? (
                    <CheckCircle className="mr-2 h-4 w-4" />
                  ) : (
                    <AlertCircle className="mr-2 h-4 w-4" />
                  )}
                  {t("robotConfig.calib.save")}
                </Button>
              </div>
            );
          })()}

        {/* Auto run: the batch picker, progress, per-arm rows and the log
            terminal, at the bottom of the same single column. On success the
            whole panel collapses; failures keep this up to be read. */}
        {(batchAutoCalOpen ||
          batchAutoCal.active ||
          batchAutoCalResultsOpen) && (
          <div className="space-y-3 rounded-md border border-border bg-muted/30 p-3">
            <div className="flex items-center gap-2 text-sm font-medium text-foreground">
              <Wand2 className="h-4 w-4" />
              {!batchAutoCalOpen && batchAutoCal.total === 1
                ? t("robotConfig.batch.titleSingle")
                : t("robotConfig.batch.titleMulti")}
            </div>
            {batchAutoCalOpen && !batchAutoCal.active ? (
              <>
                {/* Identical to a single-arm auto run — same clip, same start
                    pose, same torque control. The only thing a batch adds is
                    the list below, so that is the only thing that looks new. */}
                {autoPreamble}
                <p className="text-xs text-muted-foreground">
                  {t("robotConfig.batch.pickerHint")}
                </p>
              </>
            ) : batchAutoCal.active ? (
              <p className="text-xs text-muted-foreground">
                {t("robotConfig.batch.progress", {
                  count: batchAutoCal.total,
                  done: batchAutoCal.completed + batchAutoCal.failed,
                  total: batchAutoCal.total,
                })}
              </p>
            ) : null}

            {/* The list, in every phase. Right-hand column is the port until
                the run starts and the arm's status afterwards. */}
            {listedSlots.length > 0 && (
              <div className="space-y-2">
                {listedSlots.map((slot) => {
                  const run = armRunStatus(slot);
                  // Ticked means "in this run" once one exists.
                  const selected = picking ? !!batchSelected[slot.key] : !!run;
                  const assignedPort = slotPort(slot);
                  const hasPort = !!assignedPort;
                  // Distinguish "never assigned" from "assigned but
                  // not currently detected" so the hint is actionable.
                  const savedButUndetected = !hasPort && !!slotSavedPort(slot);
                  return (
                    <label
                      key={slot.key}
                      className={`flex items-center gap-2 rounded-md border p-2 ${
                        selected
                          ? "border-ring bg-accent"
                          : "border-border bg-background"
                      } ${
                        picking && hasPort
                          ? "cursor-pointer"
                          : picking
                            ? "cursor-not-allowed opacity-60"
                            : ""
                      }`}
                    >
                      <Checkbox
                        checked={selected}
                        disabled={!picking || !hasPort}
                        onCheckedChange={(checked) =>
                          setBatchSelected((prev) => ({
                            ...prev,
                            [slot.key]: checked === true,
                          }))
                        }
                      />
                      <span className="text-sm text-foreground">
                        {slot.label}
                      </span>
                      {run ? (
                        <span
                          className={`ml-auto text-xs ${
                            run.status === "completed"
                              ? "text-ok"
                              : run.status === "failed"
                                ? "text-destructive"
                                : run.status === "stopped"
                                  ? "text-warn"
                                  : "text-info"
                          }`}
                          title={run.error ?? undefined}
                        >
                          {run.status === "completed"
                            ? t("robotConfig.batch.armStatus.completed")
                            : run.status === "failed"
                              ? t("robotConfig.batch.armStatus.failed")
                              : run.status === "stopped"
                                ? t("robotConfig.batch.armStatus.stopped")
                                : t("robotConfig.batch.armStatus.running")}
                        </span>
                      ) : (
                        <span
                          className={`ml-auto font-mono text-xs ${
                            hasPort ? "text-muted-foreground" : "text-warn/80"
                          }`}
                        >
                          {hasPort
                            ? assignedPort
                            : savedButUndetected
                              ? t("robotConfig.batch.portUndetected")
                              : t("robotConfig.batch.portMissing")}
                        </span>
                      )}
                    </label>
                  );
                })}
                {!batchAutoCal.active && batchAutoCal.total > 0 && (
                  <p className="text-xs text-muted-foreground">
                    {t("robotConfig.batch.summary", {
                      completed: batchAutoCal.completed,
                      failed: batchAutoCal.failed,
                    })}
                  </p>
                )}
              </div>
            )}

            {picking && (
              <>
                <div className="flex gap-2">
                  <Button
                    onClick={() => setBatchAutoCalPromptOpen(true)}
                    disabled={selectedBatchSlots.length === 0}
                    className="flex-1"
                  >
                    <Wand2 className="mr-2 h-4 w-4" />
                    {t("robotConfig.batch.start", {
                      count: selectedBatchSlots.length || 0,
                    })}
                  </Button>
                  <Button
                    onClick={() => {
                      setBatchAutoCalOpen(false);
                      // Also drop any finished run's results, or the box
                      // would stay up in results-only mode and Cancel
                      // would look like it did nothing.
                      setBatchAutoCalResultsOpen(false);
                    }}
                    variant="outline"
                    className="shrink-0"
                  >
                    {t("common.cancel")}
                  </Button>
                </div>
              </>
            )}

            {batchAutoCal.logs.length > 0 && (
              <div className="max-h-40 overflow-auto whitespace-pre-wrap rounded border border-border bg-muted p-2 font-mono text-xs text-foreground">
                {batchAutoCal.logs.slice(-120).map((line, i) => (
                  <div key={i}>{line}</div>
                ))}
              </div>
            )}

            {/* Results-only view (run finished, picker closed — the row
                path): nothing else here can close the box, so this is the
                way out. The multi-arm finished view reopens the picker
                instead and uses its Cancel, which clears the same flag. */}
            {!batchAutoCal.active && !batchAutoCalOpen && (
              <div className="flex justify-end">
                <Button
                  onClick={() => setBatchAutoCalResultsOpen(false)}
                  variant="outline"
                  size="sm"
                  className="shrink-0"
                >
                  {t("robotConfig.batch.dismiss")}
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <Dialog
      open
      onOpenChange={(next) => {
        if (!next) requestClose();
      }}
    >
      <DialogContent className="flex h-[85vh] max-w-3xl flex-col gap-0 overflow-hidden p-0">
        {/* Window title bar */}
        <DialogHeader className="shrink-0 space-y-0 border-b border-border px-6 py-4 text-left">
          <p className="eyebrow">{t("robotConfig.window.eyebrow")}</p>
          <DialogTitle className="flex items-center gap-2 pt-1 text-base font-semibold">
            {t("robotConfig.window.title", { name: robotName })}
            <RobotLayoutChip arms={robot?.arms} />
          </DialogTitle>
          <DialogDescription className="sr-only">
            {t("robotConfig.window.srDescription", { name: robotName })}
          </DialogDescription>
        </DialogHeader>

        {/* Scrollable window body */}
        <div className="flex-1 divide-y divide-border overflow-y-auto px-6">
          {/* Said once, above everything: no installed arm family answers to
              this record's arm type, so nothing below can be started. The
              sections still render (ports and files are readable) with their
              detect / calibrate actions disabled. */}
          {armUnavailable && (
            <Alert
              variant="destructive"
              className="my-4 border-destructive/40 bg-destructive/10"
            >
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                {t("robotConfig.window.armUnavailable", { armType })}
              </AlertDescription>
            </Alert>
          )}
          {/* Softer: the arm may well be installed, the manifest just has
              not answered yet (ArmsProvider is retrying). Actions are held
              the same way until it does. */}
          {armsNotLoaded && (
            <Alert className="my-4 border-warn/40 bg-warn/10 text-warn">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                {t("robotConfig.window.armsNotLoaded")}
              </AlertDescription>
            </Alert>
          )}
          {/* 01 · Device */}
          <section className="space-y-3 py-5">
            <div className="flex items-center gap-2">
              <PanelHeader step="01" title={t("robotConfig.device.step")} />
              {/* Rescan is global and always was; it just used to sit inside a
                  per-slot row and read as though it applied to that slot. */}
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={fetchPorts}
                disabled={
                  portsLoading || hardwareBusy || !!detecting || wiggling
                }
                aria-label={t("robotConfig.port.rescan")}
                className="ml-auto h-6 gap-1.5 px-2 text-xs text-muted-foreground hover:text-foreground"
              >
                <RefreshCw
                  aria-hidden
                  className={cn("h-3.5 w-3.5", portsLoading && "animate-spin")}
                />
                {t("robotConfig.port.rescan")}
              </Button>
            </div>

            {/* Which leader drives this robot — only a family with a choice
                (the Metal arm) renders it. Saved at once (see
                handleLeaderKindChange); an option this install cannot drive
                is greyed with the server's own remedy beneath. */}
            {multiLeader && (
              <div className="flex flex-wrap items-center gap-2">
                <Label
                  htmlFor="leader-kind"
                  className="text-xs text-muted-foreground"
                >
                  {t("robotConfig.leaderKind.label")}
                </Label>
                <Select
                  value={leaderKind}
                  onValueChange={handleLeaderKindChange}
                  disabled={
                    hardwareBusy || armActionsBlocked || savingLeaderKind
                  }
                >
                  <SelectTrigger
                    id="leader-kind"
                    aria-label={t("robotConfig.leaderKind.label")}
                    className="h-8 w-auto min-w-[16rem] text-xs"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {leaderChoices.map((option) => (
                      <SelectItem
                        key={option.id}
                        value={option.id}
                        disabled={!option.available}
                      >
                        {leaderOptionLabel(option)}
                        {option.available
                          ? ""
                          : ` (${t("robotConfig.leaderKind.unavailable")})`}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {leaderChoice && !leaderChoice.available && (
                  // Server prose (which extra to install): English in every
                  // language, like every other backend message.
                  <p className="basis-full text-xs text-warn">
                    {leaderChoice.unavailable_reason}
                  </p>
                )}
                {leaderKind === "star_vertical" && (
                  <p className="basis-full text-xs text-muted-foreground">
                    {t("robotConfig.leaderKind.verticalHint")}
                  </p>
                )}
                {leaderEnergized && (
                  <p className="basis-full text-xs text-muted-foreground">
                    {t("robotConfig.leaderKind.energizedHint")}
                  </p>
                )}
              </div>
            )}

            {/* The arm layout. Three radio rows rather than a select: the
                options are sentences, and which one is picked changes what
                the rest of this window shows. The VALUE is the record's
                `arms` field — data; only the labels localize. */}
            <div
              role="radiogroup"
              aria-label={t("robotConfig.layout.question")}
              className="space-y-1.5"
            >
              <p className="text-sm text-muted-foreground">
                {t("robotConfig.layout.question")}
              </p>
              <div className="divide-y divide-border overflow-hidden rounded-md border border-border">
                {LAYOUT_OPTIONS.map((option) => {
                  const checked = draftArms === option.value;
                  return (
                    <button
                      key={option.value}
                      type="button"
                      role="radio"
                      aria-checked={checked}
                      disabled={hardwareBusy}
                      onClick={() => setArmsDraft(option.value)}
                      className={cn(
                        "flex w-full items-center gap-2.5 bg-background px-3 py-2 text-left text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-60",
                        checked
                          ? "bg-accent/60 text-foreground"
                          : "text-muted-foreground hover:text-foreground",
                      )}
                    >
                      <span
                        aria-hidden
                        className={cn(
                          "relative h-3.5 w-3.5 shrink-0 rounded-full border",
                          checked
                            ? "border-primary"
                            : "border-muted-foreground/60",
                        )}
                      >
                        {checked ? (
                          <span className="absolute inset-[2.5px] rounded-full bg-primary" />
                        ) : null}
                      </span>
                      {t(option.labelKey)}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* One row for a single robot, one row per side when bimanual. */}
            {(isBimanual
              ? (["left", "right"] as const)
              : (["left"] as const)
            ).map((side) => {
              const rowSlots = armSlots.filter((s) => s.arm === side);
              if (rowSlots.length === 0) return null;
              return (
                <div key={side} className="space-y-1.5">
                  {isBimanual && (
                    <p className="eyebrow">
                      {t(
                        side === "left"
                          ? "robotConfig.device.left"
                          : "robotConfig.device.right",
                      )}
                    </p>
                  )}
                  <div
                    role="group"
                    aria-label={t(
                      isBimanual
                        ? "robotConfig.device.groupBimanual"
                        : "robotConfig.device.groupSingle",
                    )}
                    className={cn(
                      "grid gap-3",
                      rowSlots.length > 1 ? "grid-cols-2" : "grid-cols-1",
                    )}
                  >
                    {rowSlots.map((slot) => (
                      <DeviceSlotCell
                        key={slot.key}
                        slot={slot}
                        port={draftPort(slot.portField)}
                        portDetected={slotPortDetected(slot)}
                        configured={!!(robot?.[slot.cfgField] as string)}
                        availablePorts={availablePorts.filter(
                          (p) => !p.startsWith("gs_usb:") ||
                            (armType === "maker" && slot.device === "robot"),
                        )}
                        heldByLabel={(p) => {
                          const holder = portFields.find(
                            (f) => f !== slot.portField && draftPort(f) === p,
                          );
                          return holder ? portFieldLabel(holder) : null;
                        }}
                        busy={
                          hardwareBusy ||
                          armActionsBlocked ||
                          !!detecting ||
                          wiggling
                        }
                        detecting={detecting === slot.portField}
                        wiggling={wiggling}
                        // Feetech arms always; a CAN family with a gripper
                        // wiggle on its follower rows, and on its leader rows
                        // only when the leader has a gripper to move (an
                        // energized leader — the Star leader has no motors).
                        showWiggle={
                          !!armInfo &&
                          (feetechBus ||
                            (gripperWiggle &&
                              (slot.device === "robot" || leaderEnergized)))
                        }
                        showDetect={
                          (!portProbe || slot.device === "teleop") &&
                          !leaderEnergized
                        }
                        showAutoDetect={
                          !manualPortIdentification &&
                          portProbe &&
                          !leaderEnergized
                        }
                        // No selection side effects: every action names its own
                        // slot, so none of them depend on what is selected.
                        onPortChange={(next) =>
                          handleSelectPort(next, slot.portField)
                        }
                        onDetect={() => handleDetect(slot.portField)}
                        onAutoDetect={() =>
                          handleDetect(slot.portField, "auto")
                        }
                        onWiggle={() =>
                          handleWiggle(draftPort(slot.portField), slot.device)
                        }
                      />
                    ))}
                  </div>
                </div>
              );
            })}

            {(manualPortIdentification || !portProbe) && (
              <p className="text-xs text-muted-foreground">
                {t("robotConfig.port.multipleHelp")}
              </p>
            )}

            {/* No ports at all is a whole-section condition, not a per-slot
                one, so it is said once rather than inside four dropdowns. */}
            {portsScanned && availablePorts.length === 0 && (
              <p className="text-xs text-warn">{t("robotConfig.port.none")}</p>
            )}

            {/* Instructions DURING a detect stay on screen. Both of the user's
                hands are on the arm, so a tooltip is unreachable exactly when
                it is needed. */}
            {detecting && (
              <p className="text-xs text-ok">
                {detectionMethod === "swing"
                  ? t("robotConfig.port.detectLive")
                  : t("robotConfig.port.detectLiveProbe")}
              </p>
            )}
          </section>

          {/* 02 · Calibration */}
          {robot && (
            <section className="space-y-3 py-5">
              <div className="flex items-center gap-2">
                <PanelHeader step="02" title={t("robotConfig.files.step")} />
                <div className="ml-auto flex items-center gap-1.5">
                  {(autoCalibration || stepCalibration) && (
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-6 gap-1.5 px-2 text-xs"
                      onClick={handleCalibrateAll}
                      disabled={
                        !robotName ||
                        !anyArmAvailable ||
                        armActionsBlocked ||
                        manualCalibLive ||
                        zeroCalQueue.length > 0 ||
                        panelCalibration ||
                        calibrationStatus.calibration_active ||
                        batchAutoCal.active
                      }
                      title={
                        anyArmAvailable
                          ? t(
                              stepCalibration
                                ? "robotConfig.files.calibrateAllZeroTitle"
                                : "robotConfig.files.calibrateAllTitle",
                            )
                          : t("robotConfig.files.calibrateAllDisabledTitle")
                      }
                    >
                      <Wand2 className="h-4 w-4" />
                      {t("robotConfig.files.calibrateAll")}
                    </Button>
                  )}
                  {/* One folder per device type — both same-side slots share a
                    single directory (so_leader / so_follower for an SO-101;
                    maker_follower or metal_follower for the CAN followers,
                    with rebot_102_leader SHARED by both CAN leaders), so a
                    single leader + follower pair covers single AND bimanual
                    modes (no per-slot duplication). */}
                  {showLeader && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 gap-1.5 px-2 text-xs text-muted-foreground hover:text-foreground"
                      onClick={() => openCalibrationFolder("teleop")}
                      aria-label={t("robotConfig.files.openLeaderFolder")}
                      title={t("robotConfig.files.openLeaderFolder")}
                    >
                      <FolderOpen className="h-4 w-4" />
                      {t("robotConfig.files.leader")}
                    </Button>
                  )}
                  {showFollower && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 gap-1.5 px-2 text-xs text-muted-foreground hover:text-foreground"
                      onClick={() => openCalibrationFolder("robot")}
                      aria-label={t("robotConfig.files.openFollowerFolder")}
                      title={t("robotConfig.files.openFollowerFolder")}
                    >
                      <FolderOpen className="h-4 w-4" />
                      {t("robotConfig.files.follower")}
                    </Button>
                  )}
                </div>
              </div>
              {/* An extension's own calibration page (manifest
                  calibration.kind "panel"): nothing in this window can start
                  it yet, so the rows' calibrate buttons stay disabled and
                  this says why. TB6b mounts the panel here. */}
              {panelCalibration && (
                <Alert className="border-info/40 bg-info/10 text-info">
                  <Activity className="h-4 w-4" />
                  <AlertDescription>
                    {t("robotConfig.calib.panel.notice")}
                  </AlertDescription>
                </Alert>
              )}
              {(isBimanual
                ? // Bimanual: each of the four slots gets the same free-naming
                  // picker as single mode — names are arbitrary now, and the
                  // SLOT (not the name) decides which arm the file drives.
                  ([
                    {
                      labelKey: "robotConfig.files.row.leftLeader",
                      device: "teleop",
                      cfgField: "leader_config",
                    },
                    {
                      labelKey: "robotConfig.files.row.leftFollower",
                      device: "robot",
                      cfgField: "follower_config",
                    },
                    {
                      labelKey: "robotConfig.files.row.rightLeader",
                      device: "teleop",
                      cfgField: "right_leader_config",
                    },
                    {
                      labelKey: "robotConfig.files.row.rightFollower",
                      device: "robot",
                      cfgField: "right_follower_config",
                    },
                  ] as const)
                : ([
                    {
                      labelKey: "robotConfig.files.row.leader",
                      device: "teleop",
                      cfgField: "leader_config",
                    },
                    {
                      labelKey: "robotConfig.files.row.follower",
                      device: "robot",
                      cfgField: "follower_config",
                    },
                  ] as const)
              )
                // A row for an arm the layout hides is dropped, like its slot.
                .filter((row) =>
                  row.device === "teleop" ? showLeader : showFollower,
                )
                .map((row) => {
                  const cfg = (robot[row.cfgField] as string) || "";
                  // The same config may drive both same-side slots only by
                  // mistake (one physical arm on two arms), so exclude the
                  // counterpart slot's config from this picker in bimanual mode.
                  const counterpartField =
                    row.cfgField === "leader_config"
                      ? "right_leader_config"
                      : row.cfgField === "right_leader_config"
                        ? "leader_config"
                        : row.cfgField === "follower_config"
                          ? "right_follower_config"
                          : "follower_config";
                  const excludeConfig = isBimanual
                    ? (robot[counterpartField] as string) || undefined
                    : undefined;
                  // The counterpart slot's config field, so the library can
                  // SWAP assignments when the user picks its in-use config
                  // (this slot takes it; the counterpart takes this slot's).
                  const excludeConfigField = isBimanual
                    ? counterpartField
                    : undefined;
                  // Which physical arm this row's slot drives, for retargeting
                  // the calibration flow when its + button is clicked.
                  const rowArm: "left" | "right" = row.cfgField.startsWith(
                    "right_",
                  )
                    ? "right"
                    : "left";
                  // The arm slot this row stands for — rows and slots are 1:1 on
                  // cfgField in both modes, so the panel's "Auto-calibrate" can
                  // target this row's arm and nothing else.
                  const rowSlot = armSlots.find(
                    (s) => s.cfgField === row.cfgField,
                  );
                  const isNewCalibOpen = newCalibFor === row.cfgField;
                  const rowLabel = t(row.labelKey);
                  return (
                    // Keyed on the config field, not the label — the label is
                    // localized and would remount the row on a language switch.
                    <div key={row.cfgField}>
                      <div className="flex items-center gap-2 text-sm">
                        {cfg ? (
                          <CheckCircle className="h-4 w-4 text-ok" />
                        ) : (
                          <Circle className="h-4 w-4 text-muted-foreground" />
                        )}
                        <span
                          className={
                            cfg ? "text-foreground" : "text-muted-foreground"
                          }
                        >
                          {rowLabel}
                        </span>
                      </div>
                    {/* Picker, Calibrate and the overflow menu are one welded
                        control group rendered by CalibrationLibrary — the
                        Calibrate segment used to be a separate outlined "+"
                        button sitting outside its border. Pressing it expands
                        the calibration flow (auto/manual, demo, advanced
                        torque) right below this row, for this arm slot. */}
                    <CalibrationLibrary
                      armType={armType}
                      leaderKind={
                        row.device === "teleop" ? leaderKindParam : undefined
                      }
                      device={row.device}
                      assignedConfig={cfg}
                      configField={row.cfgField}
                      excludeConfig={excludeConfig}
                      excludeConfigField={excludeConfigField}
                      robotName={robotName}
                      onAssigned={fetchRobot}
                      onLibraryChanged={() => setCalibReloadToken((t) => t + 1)}
                      reloadToken={calibReloadToken}
                      onCalibrate={() =>
                        toggleNewCalibration(row.cfgField, row.device, rowArm)
                      }
                      calibrateDisabled={
                        armActionsBlocked ||
                        panelCalibration ||
                        hardwareBusy ||
                        manualCalibLive
                      }
                      calibrateOpen={isNewCalibOpen}
                    />
                    {/* Slides open in place, like the studio's entry forms. */}
                    <Collapsible open={isNewCalibOpen}>
                      <CollapsibleContent className={SLIDE}>
                        {newCalibrationPanel(rowLabel, rowSlot)}
                      </CollapsibleContent>
                    </Collapsible>
                  </div>
                );
              })}
            </section>
          )}

          {/* 03 · Cameras — follower-side, so a leader-only controller has
              none to configure. */}
          {showFollower && (
            <section className="space-y-4 py-5">
              <div className="flex items-center gap-2">
                <PanelHeader step="03" title={t("robotConfig.cameras.step")} />
                <div className="ml-auto flex items-center gap-2">
                  <Label
                    htmlFor="cameras-toggle"
                    className="cursor-pointer text-sm text-muted-foreground"
                  >
                    {camerasActive
                      ? t("robotConfig.cameras.on")
                      : t("robotConfig.cameras.off")}
                  </Label>
                  <Switch
                    id="cameras-toggle"
                    checked={camerasActive}
                    onCheckedChange={handleCamerasActiveChange}
                    aria-label={t("robotConfig.cameras.toggleLabel")}
                  />
                </div>
              </div>
              {/* Mounted whether the switch is on or off: it renders nothing
                  while off, so the camera picked before switching off is still
                  picked, and still previewing, when it comes back on. */}
              <CameraConfiguration
                active={camerasActive}
                cameras={cameras}
                onCamerasChange={handleCamerasChange}
                releaseStreamsRef={releaseStreamsRef}
              />
              {!camerasActive && (
                <div className="space-y-3 rounded-md border border-border bg-muted/30 p-6 text-center">
                  <Camera className="mx-auto h-10 w-10 text-muted-foreground" />
                  <div className="space-y-1">
                    <p className="font-medium text-foreground">
                      {t("robotConfig.cameras.offTitle")}
                    </p>
                    <p className="mx-auto max-w-md text-sm text-muted-foreground">
                      {t("robotConfig.cameras.offDescription")}
                    </p>
                    {cameras.length > 0 && (
                      <p className="pt-1 text-xs text-muted-foreground">
                        {t("robotConfig.cameras.saved", {
                          count: cameras.length,
                        })}
                      </p>
                    )}
                  </div>
                  <p className="flex items-center justify-center gap-1.5 text-xs text-muted-foreground">
                    <ShieldQuestion className="h-3.5 w-3.5" />
                    {t("robotConfig.cameras.permissionHint")}
                  </p>
                </div>
              )}
            </section>
          )}
          {robot && armType === "metal" && showFollower && (
            <Collapsible key={robotName} className="group py-5">
              <CollapsibleTrigger className="flex w-full items-center justify-between text-sm font-semibold text-foreground">
                {t("robotConfig.advanced.title")}
                <ChevronDown className="h-4 w-4 transition-transform group-data-[state=open]:rotate-180" />
              </CollapsibleTrigger>
              <CollapsibleContent className={SLIDE}>
                <div className="space-y-3 pt-4">
                  <div className="flex items-center justify-between gap-3">
                    <Label htmlFor="gripperHoldingTorque">{t("robotConfig.advanced.holdingLabel")}</Label>
                    <output htmlFor="gripperHoldingTorque" className="font-mono text-sm tabular-nums">
                      {(holdingDraft ?? 0.5).toFixed(1)} N·m
                    </output>
                  </div>
                  <input id="gripperHoldingTorque" type="range" min={0.1} max={2} step={0.1}
                    value={holdingDraft ?? 0.5} disabled={saving}
                    onChange={(event) => setHoldingDraft(Number(event.target.value))}
                    aria-valuetext={`${(holdingDraft ?? 0.5).toFixed(1)} N·m`}
                    className="h-1.5 w-full cursor-pointer accent-primary disabled:cursor-not-allowed"
                    list="gripperHoldingTorqueTicks" />
                  <datalist id="gripperHoldingTorqueTicks"><option value={0.5} /></datalist>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>0.1 N·m</span>
                    <Button type="button" variant="ghost" size="sm" className="h-6 px-2 text-xs"
                      disabled={saving || holdingDraft === 0.5} onClick={() => setHoldingDraft(0.5)}>
                      {t("robotConfig.advanced.holdingDefault")}
                    </Button>
                    <span>2.0 N·m</span>
                  </div>
                  {holdingDraft === null && <p className="text-xs text-muted-foreground">
                    {t("robotConfig.advanced.holdingDisabled")}
                  </p>}
                  <p className="text-xs text-muted-foreground">{t("robotConfig.advanced.holdingHint")}</p>
                </div>
              </CollapsibleContent>
            </Collapsible>
          )}
        </div>

        {/* Window footer — Save is the ONLY path that writes the robot record;
            every port, camera, and motor-power edit stays a local draft until
            pressed. Quit closes the window, confirming first if there are
            unsaved drafts (or a live manual calibration to abort). */}
        <div className="flex shrink-0 items-center justify-between gap-3 border-t border-border bg-background px-6 py-3">
          {/* Keep the footer brief; the tooltip explains any missing setup. */}
          <span
            className={`text-sm ${
              isDirty ? "text-warn" : "text-muted-foreground"
            }`}
            title={
              !isDirty && robot && !robotLayoutReady(robot)
                ? formatRobotSetupGap(t, robot, setupScopeForArms(robot.arms))
                : undefined
            }
          >
            {isDirty
              ? t("robotConfig.window.unsaved")
              : robot && !robotLayoutReady(robot)
                ? t("robotConfig.window.savedWithGap", {
                    // The saved layout's own scope: a station reads as ready
                    // once its follower is, a controller once its leader is.
                    gap: formatRobotSetupGap(
                      t,
                      robot,
                      setupScopeForArms(robot.arms),
                    ),
                  })
                : t("robotConfig.window.allSaved")}
          </span>
          <div className="flex gap-2">
            <Button variant="outline" onClick={requestClose}>
              {t("robotConfig.window.quit")}
            </Button>
            <Button onClick={handleSave} disabled={!isDirty || saving || !holdingValid}>
              {saving ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <CheckCircle className="mr-2 h-4 w-4" />
              )}
              {saving
                ? t("robotConfig.window.saving")
                : justSaved && !isDirty
                  ? t("robotConfig.window.justSaved")
                  : t("robotConfig.window.save")}
            </Button>
          </div>
        </div>

        <Dialog
          open={batchAutoCalPromptOpen}
          onOpenChange={setBatchAutoCalPromptOpen}
        >
          <DialogContent>
            <DialogHeader>
              {/* Same gate for both entry points; a row's "Auto-calibrate"
                  makes the one-arm wording the common case. */}
              <DialogTitle>
                {selectedBatchSlots.length === 1
                  ? t("robotConfig.batch.prompt.titleSingle", {
                      arm:
                        selectedBatchSlots[0]?.label ??
                        t("robotConfig.batch.prompt.titleFallbackArm"),
                    })
                  : t("robotConfig.batch.prompt.titleMulti", {
                      count: selectedBatchSlots.length,
                    })}
              </DialogTitle>
              <DialogDescription>
                {selectedBatchSlots.length === 1 ? (
                  <Trans
                    i18nKey="robotConfig.batch.prompt.bodySingle"
                    components={[<strong key="0" />]}
                  />
                ) : (
                  <Trans
                    i18nKey="robotConfig.batch.prompt.bodyMulti"
                    values={{ count: selectedBatchSlots.length }}
                    components={[<strong key="0" />]}
                  />
                )}
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => setBatchAutoCalPromptOpen(false)}
              >
                {t("common.cancel")}
              </Button>
              <Button onClick={() => startBatchAutoCalibration()}>
                {t("robotConfig.batch.prompt.confirm")}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <AlertDialog
          open={portAssignPrompt !== null}
          onOpenChange={(next) => {
            if (!next) setPortAssignPrompt(null);
          }}
        >
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>
                {portAssignPrompt?.swapPort
                  ? t("robotConfig.portAssign.swapTitle")
                  : portAssignPrompt?.source === "detect"
                    ? t("robotConfig.portAssign.detectTitle")
                    : t("robotConfig.portAssign.assignTitle")}
              </AlertDialogTitle>
              <AlertDialogDescription>
                <Trans
                  i18nKey={
                    portAssignPrompt?.source === "detect"
                      ? "robotConfig.portAssign.leadDetect"
                      : "robotConfig.portAssign.leadAssign"
                  }
                  values={{
                    port: portAssignPrompt?.port,
                    target: portAssignPrompt?.targetLabel,
                  }}
                  components={[
                    <span key="0" className="font-mono text-foreground" />,
                    <strong key="1" />,
                  ]}
                />
                {portAssignPrompt?.releasedLabel &&
                  (portAssignPrompt.swapPort ? (
                    <>
                      {" "}
                      <Trans
                        i18nKey="robotConfig.portAssign.swapClause"
                        values={{
                          released: portAssignPrompt.releasedLabel,
                          swapPort: portAssignPrompt.swapPort,
                        }}
                        components={[
                          <strong key="0" />,
                          <strong key="1" />,
                          <span
                            key="2"
                            className="font-mono text-foreground"
                          />,
                        ]}
                      />
                    </>
                  ) : (
                    <>
                      {" "}
                      <Trans
                        i18nKey="robotConfig.portAssign.takeClause"
                        values={{ released: portAssignPrompt.releasedLabel }}
                        components={[<strong key="0" />, <strong key="1" />]}
                      />
                    </>
                  ))}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
              <AlertDialogAction onClick={handleConfirmPortAssign}>
                {portAssignPrompt?.swapPort
                  ? t("robotConfig.portAssign.confirmSwap")
                  : portAssignPrompt?.releasedLabel
                    ? t("robotConfig.portAssign.confirmMove")
                    : t("robotConfig.portAssign.confirmAssign")}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        <AlertDialog open={quitPromptOpen} onOpenChange={setQuitPromptOpen}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>
                {t("robotConfig.window.discard.title")}
              </AlertDialogTitle>
              <AlertDialogDescription>
                {t("robotConfig.window.discard.description")}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>
                {t("robotConfig.window.discard.cancel")}
              </AlertDialogCancel>
              <AlertDialogAction
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                onClick={confirmQuit}
              >
                {t("robotConfig.window.discard.confirm")}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        <AlertDialog open={abortPromptOpen} onOpenChange={setAbortPromptOpen}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>
                {t("robotConfig.window.abort.title")}
              </AlertDialogTitle>
              <AlertDialogDescription>
                {t("robotConfig.window.abort.description")}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>
                {t("robotConfig.window.abort.cancel")}
              </AlertDialogCancel>
              <AlertDialogAction
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                onClick={confirmAbortAndClose}
              >
                {t("robotConfig.window.abort.confirm")}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </DialogContent>
    </Dialog>
  );
};

export default RobotConfigDialog;
