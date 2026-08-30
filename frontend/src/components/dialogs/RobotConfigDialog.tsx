import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
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
  XCircle,
  AlertCircle,
  AlertTriangle,
  ChevronDown,
  Loader2,
  Play,
  Plus,
  Square,
  Circle,
  Camera,
  ShieldQuestion,
  Hand,
  RefreshCw,
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
import CameraConfiguration, {
  CameraConfig,
} from "@/components/recording/CameraConfiguration";
import CalibrationLibrary from "@/components/calibration/CalibrationLibrary";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { PanelHeader, SLIDE } from "@/components/studio/panel/primitives";
import {
  RobotRecord,
  formatRobotSetupGap,
  isCanArmType,
} from "@/hooks/useRobots";
import { useLanguage } from "@/contexts/LanguageContext";
import { isCaselessScript } from "@/i18n/config";
import { cn } from "@/lib/utils";

// Wire text: matched with startsWith() against the backend's own error string,
// so this must stay byte-identical to what the server sends. Never translated;
// the heading rendered for it is `robotConfig.calib.discontinuityTitle`.
const DISCONTINUITY_ERROR_PREFIX = "Motor discontinuity detected";

interface CalibrationStatus {
  calibration_active: boolean;
  /**
   * SO-101 range sweep: "idle" | "connecting" | "recording" | "completed" |
   * "error" | "stopping".
   *
   * CAN zero-pose flow: "idle" | "connecting" | "awaiting_zero" | "saving" |
   * "completed" | "error" | "stopping". `/calibration-status` serves whichever
   * flow is live from one endpoint; the two payloads are field-compatible
   * where they overlap.
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
   * Zero-pose flow only: the arm is connected with torque OFF and we are
   * waiting for the user to pose it by hand. Always false on the SO-101 sweep
   * (the backend defaults it), which is what lets the panel switch on it.
   */
  awaiting_pose?: boolean;
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
const ArmSlotCard = ({
  slot,
  selected,
  port,
  portDetected,
  configured,
  onSelect,
}: {
  slot: ArmSlot;
  selected: boolean;
  port: string;
  portDetected: boolean;
  configured: boolean;
  onSelect: () => void;
}) => {
  const { t } = useTranslation();
  // A saved port that isn't currently detected outranks "ready": the arm may
  // be unplugged (or moved to another port, or renamed by the OS), and a green
  // check there reads as "connected, all good" when nothing is on that bus.
  // Same rule the Port dropdown and the batch already apply — no detected
  // port, no port.
  const undetected = !!port && !portDetected;
  const ready = !!port && portDetected && configured;
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      onClick={onSelect}
      className={cn(
        "w-full rounded-md border px-3 py-2 text-left transition-colors",
        selected
          ? "border-primary bg-accent"
          : "border-border bg-card hover:bg-accent",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium text-foreground">
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
          <CheckCircle className="h-4 w-4 shrink-0 text-ok" />
        ) : null}
      </div>
      {/* The path stays visible (it says WHICH port went missing); the warn
          colour is what marks it as absent, matching the "no port assigned"
          styling below. */}
      <p
        className={cn(
          "mt-0.5 truncate font-mono text-xs",
          port && !undetected ? "text-muted-foreground" : "text-warn/80",
        )}
      >
        {port || t("robotConfig.slotCard.noPort")}
      </p>
    </button>
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
 * surface changed: a shadcn Dialog styled with the skill studio's vocabulary
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

  const demoVideoRef = useRef<HTMLDivElement>(null);

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
  // The hardware family this robot is. Gates three things in this window:
  // which calibration flow the Calibrate step runs (a CAN arm — Maker or
  // Metal — has no range sweep and no automatic calibration, only a zero
  // pose), which port detection endpoint runs, and which calibration library
  // the config lists and file actions address. Records written before the
  // Maker arm existed read back as "so101", so the fallback here is only for
  // the pre-fetch render where `robot` is still null.
  const armType = robot?.arm_type ?? "so101";
  const isCanArm = isCanArmType(armType);
  const isMetalArm = armType === "metal";
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
  // The CAN families mint the arm type into the default ("<name>_maker" /
  // "<name>_metal") because their Star-leader calibrations share ONE library
  // directory while the presets' zero poses differ — an unsuffixed default
  // would let a Maker robot and a Metal robot silently share a zero that is
  // wrong for one of them. Mirrors the server's default_slot_config_name()
  // in makermodslab/utils/config.py — change both together. (The explicit
  // config_file this window sends at calibration start WINS over the server's
  // own default, so the two must agree.)
  const defaultBaseName = robotName
    ? isCanArm
      ? `${robotName}_${armType}`
      : robotName
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
    if (newCalibFor === field) {
      setNewCalibFor(null);
      return;
    }
    setDeviceType(device);
    setArm(whichArm);
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
  const [wiggling, setWiggling] = useState(false);
  // Touch-to-identify: watching every port for a hand-moved shoulder-pan swing.
  const [detecting, setDetecting] = useState(false);
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
  const [cameras, setCameras] = useState<CameraConfig[]>([]);
  const releaseStreamsRef = useRef<(() => void) | null>(null);
  // Off by default so merely opening the settings window never grabs a camera.
  // The user explicitly starts a scan, which is when cameras are turned on,
  // enumerated, and the browser permission prompt is requested.
  const [camerasActive, setCamerasActive] = useState(false);

  const handleCamerasActiveChange = (active: boolean) => {
    if (!active) {
      releaseStreamsRef.current?.();
    }
    setCamerasActive(active);
  };

  useEffect(() => {
    return () => {
      releaseStreamsRef.current?.();
    };
  }, []);

  // Arm slots the multi-arm auto-cal picker can offer. Bimanual exposes all
  // four (left/right × leader/follower); single-arm exposes the leader +
  // follower pair. Each maps to the record's config/port fields for prefill.
  const armSlots: ArmSlot[] = useMemo(
    () =>
      isBimanual
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
          ],
    [isBimanual, t],
  );

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
            body: JSON.stringify({ device_type: device, arm_type: armType }),
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
    [baseUrl, fetchWithHeaders, toast, t, armType],
  );

  // List the USB-serial ports for the dropdown (filtered to arm-like devices by
  // the backend). Refreshable so plugging in an arm and rescanning works.
  const fetchPorts = useCallback(async () => {
    setPortsLoading(true);
    try {
      const res = await fetchWithHeaders(`${baseUrl}/api/v1/available-ports`);
      const data = await res.json();
      setAvailablePorts(Array.isArray(data.ports) ? data.ports : []);
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
    },
  );
  const [isPolling, setIsPolling] = useState(false);

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

  const pollStatus = async () => {
    try {
      const response = await fetchWithHeaders(
        `${baseUrl}/api/v1/calibration-status`,
      );
      if (response.ok) {
        const status = await response.json();
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

  const handleWiggle = async () => {
    if (!port) {
      toast({
        title: t("robotConfig.port.toast.missingPortTitle"),
        description: t("robotConfig.port.toast.missingPortWiggle"),
        variant: "destructive",
      });
      return;
    }
    setWiggling(true);
    try {
      const res = await fetchWithHeaders(`${baseUrl}/api/v1/wiggle`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ port }),
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
  /**
   * Find the port for the currently selected CAN arm slot (Maker or Metal).
   *
   * Two strategies, cheapest first:
   *
   * 1. **Probe.** A CAN rig's follower and leader speak different protocols
   *    on different adapters (RobStride or Damiao over CAN vs FashionStar
   *    over UART), so simply asking each port which one answers identifies
   *    them with no gesture at all. Used whenever the probe finds exactly ONE
   *    port for this side — which is the whole single-arm case.
   * 2. **Gesture.** A bimanual rig has two identical arms per side, so the
   *    probe finds two ports and cannot say which is left and which is right.
   *    Only the user knows, so fall back to watching for a hand swing, exactly
   *    as the SO-101 flow does. (The server refuses this for a Metal FOLLOWER
   *    — the Damiao handshake energizes the motors — and its refusal message
   *    is surfaced by handleDetect's failure toast like any other.)
   *
   * Both requests carry the robot's arm_type: the endpoints default to
   * "maker", and a Metal port answers a Metal probe, not a Maker one.
   *
   * Returns the same `{success, port, message}` shape as /identify-arm so the
   * caller's assignment/confirmation path is untouched.
   */
  const detectCanArmPort = async () => {
    const probeRes = await fetchWithHeaders(
      `${baseUrl}/api/v1/maker/probe-ports`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // No ports listed = probe every detected port.
        body: JSON.stringify({ arm_type: armType }),
      },
    );
    const probe = await probeRes.json().catch(() => ({}));
    const candidates: string[] =
      (deviceType === "teleop" ? probe?.leader_ports : probe?.follower_ports) ??
      [];

    if (candidates.length === 1) {
      return {
        success: true,
        port: candidates[0],
        message: probe.message,
      };
    }

    // Zero candidates (nothing answered) or several (a bimanual rig): the
    // gesture is the only thing that can resolve it. Its own message covers
    // the nothing-found case too, so the probe's is not surfaced here.
    const res = await fetchWithHeaders(`${baseUrl}/api/v1/maker/identify-arm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device_type: deviceType, arm_type: armType }),
    });
    return await res.json();
  };

  const handleDetect = async () => {
    setDetecting(true);
    try {
      const data = isCanArm
        ? await detectCanArmPort()
        : await (
            await fetchWithHeaders(`${baseUrl}/api/v1/identify-arm`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({}), // empty = watch all detected ports
            })
          ).json();
      if (data.success && data.port) {
        // Which OTHER slot (if any) currently holds the detected port? Reuses
        // the same portFields set the dropdown uses (right_* only in bimanual),
        // so a single-arm robot's stale right_* ports don't trigger a release.
        const conflictingField = robot
          ? portFields.find(
              (f) => f !== portField && draftPort(f) === data.port,
            )
          : undefined;
        // The port THIS slot currently holds — handed to the other slot on a
        // swap. Null/empty means the swap degenerates to a take-with-warning.
        const currentPort = draftPort(portField);

        // Stage the result and open the confirmation dialog. No assignment or
        // persist happens here — that's deferred to handleConfirmPortAssign.
        setPortAssignPrompt({
          source: "detect",
          port: data.port,
          message: data.message,
          targetLabel: portFieldLabel(portField),
          releasedField: conflictingField ?? null,
          releasedLabel: conflictingField
            ? portFieldLabel(conflictingField)
            : null,
          swapPort: conflictingField && currentPort ? currentPort : null,
        });
      } else {
        toast({
          title: t("robotConfig.port.toast.noArmTitle"),
          description: data.message,
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
      setDetecting(false);
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

    setPort(prompt.port);
    const detected = prompt.source === "detect";

    if (prompt.releasedField) {
      const nextRobot = await persistPorts({
        [prompt.releasedField]: prompt.swapPort ?? "",
        [portField]: prompt.port,
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
      persistPort(prompt.port);
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
  const handleSelectPort = (nextPort: string) => {
    const conflictingField = robot
      ? portFields.find((f) => f !== portField && draftPort(f) === nextPort)
      : undefined;
    if (conflictingField) {
      const currentPort = draftPort(portField);
      setPortAssignPrompt({
        source: "manual",
        port: nextPort,
        message: "",
        targetLabel: portFieldLabel(portField),
        releasedField: conflictingField,
        releasedLabel: portFieldLabel(conflictingField),
        swapPort: currentPort || null,
      });
      return;
    }
    setPort(nextPort);
    persistPort(nextPort);
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
          } else {
            toast({
              title: t("robotConfig.batch.toast.issuesTitle"),
              description: t("robotConfig.batch.summary", {
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
          (e instanceof ApiError ? (e.detail ?? e.message) : String(e)),
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
      if (error instanceof ApiError) {
        // 409 session.held renders as the shared localized "robot is busy"
        // line; every other coded refusal shows the server's own prose.
        toast({
          title: t("robotConfig.calib.toast.startFailedTitle"),
          description:
            formatSessionHeld(t, error) ??
            error.detail ??
            t("robotConfig.calib.toast.startFailedFallback"),
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

  const handleCompleteStep = async () => {
    if (!calibrationStatus.calibration_active) return;

    try {
      const response = await fetchWithHeaders(
        `${baseUrl}/api/v1/complete-calibration-step`,
        { method: "POST" },
      );

      const data = await response.json();

      if (data.success) {
        toast({
          title: t("robotConfig.calib.toast.stepCompletedTitle"),
          description: data.message,
        });
      } else {
        toast({
          title: t("robotConfig.calib.toast.stepFailedTitle"),
          description:
            data.message || t("robotConfig.calib.toast.stepFailedFallback"),
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
    }
  };

  useEffect(() => {
    if (
      calibrationStatus.status === "error" &&
      calibrationStatus.error?.startsWith(DISCONTINUITY_ERROR_PREFIX)
    ) {
      demoVideoRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }, [calibrationStatus.status, calibrationStatus.error]);

  useEffect(() => {
    if (!isPolling) return;
    // Single stable interval. Reads calibration_active from the ref each tick so
    // the interval doesn't tear down/recreate on every status change.
    pollStatus();
    const interval = setInterval(() => {
      pollStatus();
    }, 200);
    return () => clearInterval(interval);
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

  // Refresh the robot record when a calibration completes so the checklist
  // flips to ✓ for the side that was just saved. (No auto-advance of Device
  // Type anymore — the calibration flow is anchored to the calibration-file
  // row whose "New calibration" panel is open, and switching device here
  // would drag that panel to another row mid-look.)
  useEffect(() => {
    if (calibrationStatus.status !== "completed") return;
    // A completed calibration may have written a new named file — nudge the
    // per-side libraries to re-fetch their config lists so it shows up.
    setCalibReloadToken((t) => t + 1);
    fetchRobot();
  }, [calibrationStatus.status, fetchRobot]);

  // Stage the current side's port into the local draft (no network write). A
  // re-detected USB port (which shuffles on reboot/reconnect) is recorded here
  // and only committed on Save. An empty string is a valid value: it CLEARS the
  // assignment (arm disconnected). The batched Save sends every dirty port slot
  // together so the backend's duplicate-port guard sees the merged record.
  const persistPort = useCallback(
    (nextPort: string) => {
      if (!robotName) return;
      setPortDraft((prev) => ({ ...prev, [portField]: nextPort }));
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
  // sessions (teleop/record/skill runs) run at stock LeRobot torque and
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
  const isDirty = camerasDirty || portsDirty || motorDirty;

  const handleSave = useCallback(async () => {
    if (!robotName || !robot) return;
    const patch: Record<string, unknown> = {};
    if (camerasDirty) patch.cameras = cameras;
    if (motorDirty) patch.motor_power = motorPercent;
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
        setCameras((data.robot as RobotRecord).cameras ?? []);
        setJustSaved(true);
        toast({ title: t("robotConfig.window.toast.saved") });
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
    portsDirty,
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
      // Zero-pose flow (CAN arms) — see CalibrationStatus.status.
      case "awaiting_zero":
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
  const newCalibrationPanel = (rowLabel: string, rowSlot?: ArmSlot) => (
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

      <div className="grid grid-cols-[minmax(0,1fr)_320px] items-start gap-3">
        {/* Main vertical: actions, batch picker, status, live data. */}
        <div className="flex min-w-0 flex-col gap-3">
          {calibrationStatus.calibration_active ? (
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
          ) : (
            // Auto-calibrate is the default calibration mode: it's the
            // primary action and calibrates THIS row's arm only, straight
            // through the batch's pre-start confirmation (the multi-arm
            // picker is the header's "Calibrate all"). Manual step-by-step
            // calibration stays fully available as the secondary button.
            //
            // A CAN arm (Maker, Metal) has NEITHER of those. Auto-calibration drives the
            // arm under torque against its stops and writes Feetech EEPROM —
            // there is no CAN equivalent — and it needs no range sweep at all,
            // because its joint limits are fixed constants. Its one flow is
            // the zero pose, so it gets a single primary button.
            <>
              {isCanArm ? (
                <Button
                  onClick={() => handleStartCalibration()}
                  disabled={!robotName || !deviceType || !portDetected}
                  className="w-full"
                >
                  <Play className="mr-2 h-4 w-4" />
                  {t("robotConfig.calib.zeroPose.start")}
                </Button>
              ) : (
                <>
                  <Button
                    onClick={() => rowSlot && handleAutoCalibrateSlot(rowSlot)}
                    className="w-full"
                    disabled={!robotName || !rowSlot || !slotPort(rowSlot)}
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
                    {t("robotConfig.calib.auto")}
                  </Button>
                  <Button
                    onClick={() => handleStartCalibration()}
                    variant="outline"
                    disabled={!robotName || !deviceType || !portDetected}
                    className="w-full"
                  >
                    <Play className="mr-2 h-4 w-4" />
                    {t("robotConfig.calib.manual")}
                  </Button>
                </>
              )}
            </>
          )}

          {/* Picker + live status. The checkbox list is the multi-arm path
              only (batchAutoCalOpen); a batch started from a row's own
              "Auto-calibrate" leaves it closed and this box shows nothing but
              progress, per-arm rows, and logs — the stop button sits above.
              The third term keeps a FINISHED run's results up after `active`
              goes false: on the row path the first two terms are both false by
              then, which used to unmount the results (and the error tooltips)
              the instant they became worth reading. Dismiss clears it. */}
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
                  <p className="text-xs text-muted-foreground">
                    <Trans
                      i18nKey="robotConfig.batch.pickerIntro"
                      components={[<strong key="0" />]}
                    />
                  </p>
                  <div className="space-y-2">
                    {armSlots.map((slot) => {
                      const selected = !!batchSelected[slot.key];
                      const assignedPort = slotPort(slot);
                      const hasPort = !!assignedPort;
                      // Distinguish "never assigned" from "assigned but
                      // not currently detected" so the hint is actionable.
                      const savedButUndetected =
                        !hasPort && !!slotSavedPort(slot);
                      return (
                        <label
                          key={slot.key}
                          className={`flex items-center gap-2 rounded-md border p-2 ${
                            selected
                              ? "border-ring bg-accent"
                              : "border-border bg-background"
                          } ${
                            hasPort
                              ? "cursor-pointer"
                              : "cursor-not-allowed opacity-60"
                          }`}
                        >
                          <Checkbox
                            checked={selected}
                            disabled={!hasPort}
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
                        </label>
                      );
                    })}
                  </div>
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
              ) : batchAutoCal.active ? (
                <p className="text-xs text-muted-foreground">
                  {t("robotConfig.batch.progress", {
                    count: batchAutoCal.total,
                    done: batchAutoCal.completed + batchAutoCal.failed,
                    total: batchAutoCal.total,
                  })}
                </p>
              ) : null}

              {/* Per-arm status rows (running + terminal), shown live. */}
              {batchAutoCal.arms.length > 0 && (
                <div className="space-y-1">
                  {batchAutoCal.arms.map((a) => (
                    <div
                      key={`${a.device_type}:${a.port}`}
                      className="flex items-center justify-between gap-2 rounded bg-muted px-2 py-1 text-xs"
                    >
                      <span className="truncate font-mono text-foreground">
                        {a.name || a.port}
                      </span>
                      <span
                        className={
                          a.status === "completed"
                            ? "text-ok"
                            : a.status === "failed"
                              ? "text-destructive"
                              : a.status === "stopped"
                                ? "text-warn"
                                : "text-info"
                        }
                        title={a.error ?? undefined}
                      >
                        {a.status === "completed"
                          ? t("robotConfig.batch.armStatus.completed")
                          : a.status === "failed"
                            ? t("robotConfig.batch.armStatus.failed")
                            : a.status === "stopped"
                              ? t("robotConfig.batch.armStatus.stopped")
                              : t("robotConfig.batch.armStatus.running")}
                      </span>
                    </div>
                  ))}
                  {!batchAutoCal.active && batchAutoCal.total > 0 && (
                    <p className="pt-1 text-xs text-muted-foreground">
                      {t("robotConfig.batch.summary", {
                        completed: batchAutoCal.completed,
                        failed: batchAutoCal.failed,
                      })}
                    </p>
                  )}
                </div>
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

          {/* Manual calibration only: torque is off the whole session,
              which surprises novices (the arm is deliberately floppy).
              Auto-cal needs no standing warning — it ends gracefully
              (fold on completion, freeze + return-to-start on Stop) and
              the multi-arm pre-start confirmation dialog carries the
              safety guidance. */}
          {calibrationStatus.calibration_active && (
            <Alert className="border-warn/40 bg-warn/10 text-warn">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                {t("robotConfig.calib.torqueOffWarning")}
              </AlertDescription>
            </Alert>
          )}

          {calibrationStatus.status === "connecting" && (
            <Alert className="border-warn/40 bg-warn/10 text-warn">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                {t("robotConfig.calib.connecting")}
              </AlertDescription>
            </Alert>
          )}

          {/* Zero-pose calibration (CAN arms). One step, no range sweep: the
              arm's joint limits are fixed constants, so all this establishes
              is where zero is. Torque is off for the whole wait — the user is
              physically moving the arm — and the live readout below is a pure
              read of where each joint currently sits. The pose text is per
              FAMILY — the two zero poses are opposites on the gripper (Maker:
              fully open; Metal: closed), so each renders its own key (the
              localized twin of the server's status message). */}
          {calibrationStatus.status === "awaiting_zero" && (
            <div className="space-y-3">
              <Alert className="border-info/40 bg-info/10 text-info">
                <Activity className="h-4 w-4" />
                <AlertDescription>
                  {isMetalArm
                    ? t("robotConfig.calib.zeroPose.instructionsMetal")
                    : t("robotConfig.calib.zeroPose.instructions")}
                </AlertDescription>
              </Alert>

              {calibrationStatus.current_positions &&
                Object.keys(calibrationStatus.current_positions).length > 0 && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Activity className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm font-medium text-foreground">
                        {t("robotConfig.calib.zeroPose.liveAngles")}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                      {Object.entries(calibrationStatus.current_positions).map(
                        ([motor, angle]) => (
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
                        ),
                      )}
                    </div>
                  </div>
                )}

              <Button
                onClick={handleCompleteStep}
                disabled={!calibrationStatus.calibration_active}
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

          {calibrationStatus.status === "recording" &&
            calibrationStatus.recorded_ranges && (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium text-foreground">
                    {t("robotConfig.calib.liveData")}
                  </span>
                </div>
                <div className="rounded-md border border-border bg-muted/30 p-4">
                  <div className="space-y-3">
                    {Object.entries(calibrationStatus.recorded_ranges).map(
                      ([motor, range]) => {
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
                          <div key={motor} className="space-y-2">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <span className="text-sm font-medium text-foreground">
                                  {motor}
                                </span>
                                {rangeComplete && (
                                  <CheckCircle
                                    className="h-4 w-4 text-ok"
                                    aria-label={t(
                                      "robotConfig.calib.rangeComplete",
                                    )}
                                  />
                                )}
                              </div>
                              <span className="font-mono text-xs text-foreground">
                                {range.current}
                              </span>
                            </div>
                            <div className="relative">
                              <div className="h-3 w-full rounded-full bg-secondary">
                                <div
                                  className="relative h-3 rounded-full bg-muted-foreground/20"
                                  style={{ width: "100%" }}
                                >
                                  <div
                                    className={`absolute top-0 h-3 w-1 rounded-full transition-all duration-100 ${
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
                              <div className="mt-1 flex justify-between text-xs text-muted-foreground">
                                <span>{range.min}</span>
                                <span>{range.max}</span>
                              </div>
                            </div>
                          </div>
                        );
                      },
                    )}
                  </div>
                </div>
              </div>
            )}

          {calibrationStatus.status === "recording" &&
            (() => {
              const ranges = calibrationStatus.recorded_ranges ?? {};
              const motors = Object.entries(ranges);
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
                  <Button
                    onClick={handleCompleteStep}
                    disabled={!calibrationStatus.calibration_active}
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
                  <Alert className="border-info/40 bg-info/10 text-info">
                    <Activity className="h-4 w-4" />
                    <AlertDescription>
                      <Trans
                        i18nKey="robotConfig.calib.rangeHint"
                        components={[<strong key="0" />, <strong key="1" />]}
                      />
                    </AlertDescription>
                  </Alert>
                </div>
              );
            })()}

          {calibrationStatus.status === "completed" && (
            <Alert className="border-ok/40 bg-ok/10 text-ok">
              <CheckCircle className="h-4 w-4" />
              <AlertDescription>
                {t("robotConfig.calib.completed")}
              </AlertDescription>
            </Alert>
          )}

          {calibrationStatus.status === "error" &&
            calibrationStatus.error &&
            (calibrationStatus.error.startsWith(DISCONTINUITY_ERROR_PREFIX) ? (
              <Alert className="border-destructive/40 bg-destructive/10 text-destructive">
                <XCircle className="h-4 w-4" />
                <AlertDescription>
                  <div className="mb-1 text-base font-semibold">
                    {t("robotConfig.calib.discontinuityTitle")}
                  </div>
                  <div>{t("robotConfig.calib.discontinuityBody")}</div>
                </AlertDescription>
              </Alert>
            ) : (
              <Alert className="border-destructive/40 bg-destructive/10 text-destructive">
                <XCircle className="h-4 w-4" />
                <AlertDescription>
                  <strong>{t("robotConfig.calib.errorLabel")}</strong>{" "}
                  {calibrationStatus.error}
                </AlertDescription>
              </Alert>
            ))}
        </div>

        {/* The demo is big, so it sits beside the main vertical instead of
            pushing the controls down. Hidden on a CAN arm: the clip is
            lerobot's SO-101 range sweep, which is both the wrong arm and the
            wrong procedure for a zero pose — showing it would actively
            mis-instruct. */}
        {!isCanArm && (
          <div
            ref={demoVideoRef}
            className="space-y-2 self-start rounded-md border border-border bg-muted/30 p-3"
          >
            <h3 className="eyebrow">{t("robotConfig.calib.demoTitle")}</h3>
            <div className="overflow-hidden rounded-md bg-muted">
              <video className="h-auto w-full" controls preload="auto" muted>
                <source
                  src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/calibrate_so101_2.mp4"
                  type="video/mp4"
                />
                <p className="py-4 text-center text-sm text-muted-foreground">
                  {t("robotConfig.calib.videoUnsupported")}
                  <br />
                  <a
                    href="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/calibrate_so101_2.mp4"
                    className="underline"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {t("robotConfig.calib.videoLink")}
                  </a>
                </p>
              </video>
            </div>
          </div>
        )}
      </div>

      {/* Auto-calibration drive torque lives under Advanced parameters
          (the studio's collapsible pattern, as in RecordingForm): sent
          with the auto-calibrate start (current slider position),
          persisted on Save. Manual calibration and regular sessions
          don't use it. Full panel width, below the controls/demo grid,
          so expanding it grows the panel evenly instead of stretching
          only the left column. Hidden on a CAN arm — the slider's only
          consumer is the auto-calibration subprocess, which that arm has no
          equivalent of, and its drive effort comes from the MIT follow gains
          set at connect() instead. */}
      {robot && !isCanArm && (
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
    </div>
  );

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
          <DialogTitle className="pt-1 text-base font-semibold">
            {t("robotConfig.window.title", { name: robotName })}
          </DialogTitle>
          <DialogDescription className="sr-only">
            {t("robotConfig.window.srDescription", { name: robotName })}
          </DialogDescription>
        </DialogHeader>

        {/* Scrollable window body */}
        <div className="flex-1 divide-y divide-border overflow-y-auto px-6">
          {/* 01 · Device */}
          <section className="space-y-4 py-5">
            <PanelHeader step="01" title={t("robotConfig.device.step")} />
            <div className="space-y-2">
              <Label>{t("robotConfig.device.label")}</Label>
              {isBimanual ? (
                <div
                  role="radiogroup"
                  aria-label={t("robotConfig.device.groupBimanual")}
                  className="grid grid-cols-2 gap-3"
                >
                  <div className="space-y-2">
                    {armSlots
                      .filter((slot) => slot.arm === "left")
                      .map((slot) => (
                        <ArmSlotCard
                          key={slot.key}
                          slot={slot}
                          selected={
                            deviceType === slot.device && arm === slot.arm
                          }
                          port={draftPort(slot.portField)}
                          portDetected={slotPortDetected(slot)}
                          configured={!!(robot?.[slot.cfgField] as string)}
                          onSelect={() => {
                            setDeviceType(slot.device);
                            setArm(slot.arm);
                          }}
                        />
                      ))}
                  </div>
                  <div className="space-y-2">
                    {armSlots
                      .filter((slot) => slot.arm === "right")
                      .map((slot) => (
                        <ArmSlotCard
                          key={slot.key}
                          slot={slot}
                          selected={
                            deviceType === slot.device && arm === slot.arm
                          }
                          port={draftPort(slot.portField)}
                          portDetected={slotPortDetected(slot)}
                          configured={!!(robot?.[slot.cfgField] as string)}
                          onSelect={() => {
                            setDeviceType(slot.device);
                            setArm(slot.arm);
                          }}
                        />
                      ))}
                  </div>
                </div>
              ) : (
                <div
                  role="radiogroup"
                  aria-label={t("robotConfig.device.groupSingle")}
                  className="grid grid-cols-2 gap-3"
                >
                  {armSlots.map((slot) => (
                    <ArmSlotCard
                      key={slot.key}
                      slot={slot}
                      selected={deviceType === slot.device}
                      port={draftPort(slot.portField)}
                      portDetected={slotPortDetected(slot)}
                      configured={!!(robot?.[slot.cfgField] as string)}
                      onSelect={() => {
                        setDeviceType(slot.device);
                        setArm(slot.arm);
                      }}
                    />
                  ))}
                </div>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="port">{t("robotConfig.port.label")}</Label>
              <div className="flex flex-wrap gap-2">
                <Select value={port} onValueChange={handleSelectPort}>
                  <SelectTrigger id="port" className="min-w-[200px] flex-1">
                    <SelectValue
                      placeholder={
                        availablePorts.length
                          ? t("robotConfig.port.select")
                          : t("robotConfig.port.none")
                      }
                    />
                  </SelectTrigger>
                  <SelectContent>
                    {availablePorts.map((p) => {
                      // In-use ports stay selectable: picking one prompts a
                      // swap (this slot's current port goes to the other arm)
                      // or, if this slot is empty, a take-with-warning.
                      const usedByOtherArm = otherArmPorts.includes(p);
                      return (
                        <SelectItem key={p} value={p}>
                          <span className="flex items-center gap-2 font-mono text-xs">
                            {p}
                            {usedByOtherArm && (
                              <span
                                className={cn(
                                  "rounded border border-warn/40 px-1 font-body text-[10px] text-warn",
                                  isCaselessScript(language)
                                    ? ""
                                    : "uppercase tracking-wide",
                                )}
                              >
                                {t("robotConfig.port.otherArm")}
                              </span>
                            )}
                          </span>
                        </SelectItem>
                      );
                    })}
                    {/* A saved-but-undetected port is intentionally NOT offered
                        here: an unplugged bus can't be calibrated against, so
                        it's treated as no port. The trigger falls back to the
                        placeholder, and the port re-selects on its own once the
                        arm is plugged back in and ports are rescanned. */}
                  </SelectContent>
                </Select>
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  onClick={() => {
                    setPort("");
                    persistPort("");
                  }}
                  // Also gated during calibration: clearing wouldn't stop the
                  // running session (the subprocess holds the serial port),
                  // it would just desync the UI from the arm being measured.
                  disabled={
                    !port ||
                    calibrationStatus.calibration_active ||
                    batchAutoCal.active
                  }
                  title={t("robotConfig.port.clearTitle")}
                  aria-label={t("robotConfig.port.clear")}
                  className="shrink-0 text-muted-foreground hover:text-destructive"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  onClick={fetchPorts}
                  disabled={portsLoading}
                  title={t("robotConfig.port.rescan")}
                  aria-label={t("robotConfig.port.rescan")}
                  className="shrink-0 text-muted-foreground hover:text-foreground"
                >
                  <RefreshCw
                    className={`h-4 w-4 ${portsLoading ? "animate-spin" : ""}`}
                  />
                </Button>
              </div>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleDetect}
                  disabled={
                    detecting ||
                    wiggling ||
                    calibrationStatus.calibration_active ||
                    batchAutoCal.active
                  }
                  title={t("robotConfig.port.detectTitle")}
                  className="w-28 shrink-0"
                >
                  {detecting ? (
                    <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                  ) : (
                    <Hand className="mr-1 h-4 w-4" />
                  )}
                  {detecting
                    ? t("robotConfig.port.detecting")
                    : t("robotConfig.port.detect")}
                </Button>
                <p className="min-w-[200px] flex-1 text-xs text-muted-foreground">
                  {isMetalArm
                    ? t("robotConfig.port.detectHelpMetal")
                    : isCanArm
                      ? t("robotConfig.port.detectHelpMaker")
                      : t("robotConfig.port.detectHelp")}
                </p>
              </div>
              {/* Wiggle drives the gripper through Feetech registers to show
                  which arm is on a port. A CAN rig needs no such
                  confirmation — its follower and leader answer different
                  protocols, so Detect already identifies each unambiguously —
                  and the CAN/UART buses have no equivalent write anyway. */}
              {!isCanArm && (
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handleWiggle}
                    disabled={
                      !port ||
                      wiggling ||
                      detecting ||
                      calibrationStatus.calibration_active ||
                      batchAutoCal.active
                    }
                    title={t("robotConfig.port.wiggleTitle")}
                    className="w-28 shrink-0"
                  >
                    <Hand className="mr-1 h-4 w-4" />
                    {wiggling
                      ? t("robotConfig.port.wiggling")
                      : t("robotConfig.port.wiggle")}
                  </Button>
                  <p className="min-w-[200px] flex-1 text-xs text-muted-foreground">
                    {t("robotConfig.port.wiggleHelp")}
                  </p>
                </div>
              )}
              {detecting && (
                <p className="text-xs text-ok">
                  {isMetalArm
                    ? t("robotConfig.port.detectLiveMetal")
                    : isCanArm
                      ? t("robotConfig.port.detectLiveMaker")
                      : t("robotConfig.port.detectLive")}
                </p>
              )}
            </div>
          </section>

          {/* 02 · Calibration files */}
          {robot && (
            <section className="space-y-3 py-5">
              <div className="flex items-center gap-2">
                <PanelHeader step="02" title={t("robotConfig.files.step")} />
                {/* The multi-arm entry point: same batch flow as a row's own
                    "Auto-calibrate" (which does its arm alone), but
                    pre-selecting every detected arm and opening the picker so
                    the selection can be reviewed before confirming.
                    Hidden entirely on a CAN arm, which has no automatic
                    calibration to batch — each arm's zero pose has to be set
                    by hand anyway, so there is nothing to run concurrently. */}
                {!isCanArm && (
                  <Button
                    size="sm"
                    variant="outline"
                    className="ml-auto h-6 gap-1.5 px-2 text-xs"
                    onClick={handleCalibrateAll}
                    disabled={
                      !robotName ||
                      !anyArmAvailable ||
                      calibrationStatus.calibration_active ||
                      batchAutoCal.active
                    }
                    title={
                      anyArmAvailable
                        ? t("robotConfig.files.calibrateAllTitle")
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
              </div>
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
              ).map((row) => {
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
                    <div className="flex items-start gap-2">
                      <div className="min-w-0 flex-1">
                        <CalibrationLibrary
                          armType={armType}
                          device={row.device}
                          assignedConfig={cfg}
                          configField={row.cfgField}
                          excludeConfig={excludeConfig}
                          excludeConfigField={excludeConfigField}
                          robotName={robotName}
                          onAssigned={fetchRobot}
                          onLibraryChanged={() =>
                            setCalibReloadToken((t) => t + 1)
                          }
                          reloadToken={calibReloadToken}
                        />
                      </div>
                      {/* Expands the calibration flow (auto/manual, demo,
                          advanced torque) right below this row, targeted at
                          this arm slot. */}
                      <Button
                        type="button"
                        variant={isNewCalibOpen ? "secondary" : "outline"}
                        className="mt-1 shrink-0"
                        onClick={() =>
                          toggleNewCalibration(row.cfgField, row.device, rowArm)
                        }
                        aria-expanded={isNewCalibOpen}
                        title={t("robotConfig.files.newCalibrationTitle")}
                      >
                        <Plus className="mr-1 h-4 w-4" />
                        {t("robotConfig.files.newCalibration")}
                      </Button>
                    </div>
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

          {/* 03 · Cameras */}
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
            {camerasActive ? (
              <CameraConfiguration
                cameras={cameras}
                onCamerasChange={handleCamerasChange}
                releaseStreamsRef={releaseStreamsRef}
              />
            ) : (
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
        </div>

        {/* Window footer — Save is the ONLY path that writes the robot record;
            every port, camera, and motor-power edit stays a local draft until
            pressed. Quit closes the window, confirming first if there are
            unsaved drafts (or a live manual calibration to abort). */}
        <div className="flex shrink-0 items-center justify-between gap-3 border-t border-border bg-background px-6 py-3">
          {/* Left label: draft state first; when everything is saved but the
              robot still isn't ready (a silently-disabled Save explains
              nothing), name the concrete setup gap instead of a bare
              "All changes saved". */}
          <span
            className={`text-sm ${
              isDirty ? "text-warn" : "text-muted-foreground"
            }`}
          >
            {isDirty
              ? t("robotConfig.window.unsaved")
              : robot && !robot.is_clean
                ? t("robotConfig.window.savedWithGap", {
                    gap: formatRobotSetupGap(t, robot),
                  })
                : t("robotConfig.window.allSaved")}
          </span>
          <div className="flex gap-2">
            <Button variant="outline" onClick={requestClose}>
              {t("robotConfig.window.quit")}
            </Button>
            <Button onClick={handleSave} disabled={!isDirty || saving}>
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
                  : t("robotConfig.batch.prompt.titleMulti")}
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
