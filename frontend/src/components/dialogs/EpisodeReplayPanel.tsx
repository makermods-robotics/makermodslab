import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Loader2, Square, TriangleAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useRobots } from "@/hooks/useRobots";
import { formatRobotSetupGap } from "@/lib/robotSetupGap";
import { useSessionHeartbeat } from "@/hooks/useSessionHeartbeat";
import { useUnloadWarning } from "@/hooks/useUnloadWarning";
import { useLiveJointReadout } from "@/hooks/useLiveJointReadout";
import { ApiError } from "@/lib/apiClient";
import { startSession, stopSession, formatSessionHeld } from "@/lib/sessionApi";
import { tabOwnerId } from "@/lib/sessionOwner";
import {
  ReplayStatus,
  ReplayPhase,
  getReplayStatus,
  stopReplay,
} from "@/lib/replayHardwareApi";

import { ThermalReplayStatus } from "./ThermalReplayStatus";

const POLL_MS = 1000;

// Catalog KEYS, not resolved copy: this map is built once at import time, so
// storing translated strings here would freeze whichever language loaded first.
// The ReplayPhase values are backend data; only the labels are display.
const PHASE_LABEL_KEY: Record<ReplayPhase, string> = {
  idle: "dialogs.replay.phase.idle",
  easing_in: "dialogs.replay.phase.easingIn",
  playing: "dialogs.replay.phase.playing",
  stopping: "dialogs.replay.phase.stopping",
  done: "dialogs.replay.phase.done",
  error: "dialogs.replay.phase.error",
};

export interface EpisodeReplayPanelProps {
  repoId: string;
  episodeIndex: number;
  onElapsedChange?: (elapsedS: number, phase: ReplayPhase) => void;
}

const EpisodeReplayPanel: React.FC<EpisodeReplayPanelProps> = ({
  repoId,
  episodeIndex,
  onElapsedChange,
}) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { selectedRecord } = useRobots();
  const [status, setStatus] = useState<ReplayStatus | null>(null);
  const [thermalEnabled, setThermalEnabled] = useState(false);
  const [restConfirmed, setRestConfirmed] = useState(false);
  const [experiment, setExperiment] = useState<"baseline" | "shoulder_kp_85">("baseline");
  const thermalAlertRef = useRef("");
  const thermalAvailable = selectedRecord?.arm_type === "maker" && selectedRecord.mode === "single";
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  // Identity of the session THIS panel started (POST /api/v1/sessions).
  const [sessionId, setSessionId] = useState<string | null>(null);
  const doneRef = useRef(false);
  const startPendingRef = useRef(false);
  const generationRef = useRef(0);
  const localT0Ref = useRef<number | null>(null);

  const { joints: liveJoints } = useLiveJointReadout(status?.replay_active === true);

  // While the replay is live, renew its lease; an abandoned page makes the
  // SERVER stop the arm via the missed heartbeats — the replacement for the
  // retired exit guard's stop beacon. The courtesy beforeunload only keeps an
  // accidental tab-close from being silent.
  const replayLive = status?.replay_active === true;
  useSessionHeartbeat(sessionId, tabOwnerId(), replayLive);
  useUnloadWarning(replayLive);

  useEffect(() => {
    doneRef.current = false;
    localT0Ref.current = null;
    setRestConfirmed(false);
  }, [repoId, episodeIndex, selectedRecord?.name]);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      if (doneRef.current || startPendingRef.current) return;
      const generation = generationRef.current;
      try {
        const next = await getReplayStatus(baseUrl, fetchWithHeaders);
        if (cancelled || startPendingRef.current || generation !== generationRef.current) return;
        setStatus(next);
        const trial = next.thermal_test;
        if (trial && trial.result !== "running" && (trial.result !== "completed" || trial.rest_reached) && trial.message !== thermalAlertRef.current) {
          thermalAlertRef.current = trial.message;
          toast({
            title: trial.result === "completed" ? "Thermal test completed" : "Thermal test stopped",
            description: trial.message,
            variant: ["completed", "stopped"].includes(trial.result) ? "default" : "destructive",
            duration: 15000,
          });
        }
        if (next.phase === "playing" && localT0Ref.current === null) {
          localT0Ref.current = performance.now() / 1000;
        }
        if (next.phase !== "playing") {
          localT0Ref.current = null;
        }
        const elapsed =
          next.thermal_test && next.phase === "playing" && next.duration_s
            ? next.thermal_test.elapsed_s % next.duration_s
            : next.phase === "playing" && localT0Ref.current !== null
            ? performance.now() / 1000 - localT0Ref.current
            : next.elapsed_s;
        onElapsedChange?.(elapsed, next.phase);
        if (!next.replay_active && (next.phase === "done" || next.phase === "error")) {
          if (next.phase === "error" && !next.thermal_test) {
            toast({
              title: t("dialogs.replay.toast.failedTitle"),
              // The backend's hint/error is prose we don't translate; only the
              // client-side fallback beside it is a catalog string.
              description:
                next.hint ?? next.error ?? t("dialogs.replay.toast.seeLog"),
              variant: "destructive",
              duration: 10000,
            });
          }
          doneRef.current = true;
        }
      } catch (e) {
        if (!cancelled) {
          toast({
            title: t("dialogs.replay.toast.lostConnectionTitle"),
            description: e instanceof Error ? e.message : String(e),
            variant: "destructive",
          });
        }
      }
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [baseUrl, fetchWithHeaders, toast, onElapsedChange, t]);

  const handleStart = async () => {
    if (!selectedRecord) return;
    setStarting(true);
    startPendingRef.current = true;
    generationRef.current += 1;
    doneRef.current = false;
    thermalAlertRef.current = "";
    localT0Ref.current = null;
    try {
      // Robot NAME + episode selection only — the follower port/config
      // resolve server-side from the saved record. The owner attaches the
      // lease the heartbeat above renews while the replay plays.
      const { session, warnings } = await startSession(baseUrl, fetchWithHeaders, {
        kind: "replay",
        robot: selectedRecord.name,
        owner: tabOwnerId(),
        options: {
          repo_id: repoId,
          episode_index: episodeIndex,
          ...(thermalEnabled && thermalAvailable ? {
            thermal_test: { duration_s: 1800, experiment, rest_pose_confirmed: restConfirmed },
          } : {}),
        },
      });
      setSessionId(session.id);
      setStatus({ replay_active: true, phase: "easing_in", episode_index: episodeIndex,
        elapsed_s: 0, duration_s: null });
      doneRef.current = false;
      if (warnings?.length) {
        // Warn-but-allow arm-identity finding: the replay RUNS, but the user
        // should see it. Backend prose, rendered verbatim.
        toast({
          title: t("dialogs.replay.toast.startedWarningTitle"),
          description: warnings.join(" "),
          duration: 8000,
        });
      }
    } catch (e) {
      toast({
        title: t("dialogs.replay.toast.startFailedTitle"),
        // 409 session.held renders as the shared localized "robot is busy"
        // line; everything else is the server's raw error text.
        description:
          formatSessionHeld(t, e) ??
          (e instanceof Error ? e.message : String(e)),
        variant: "destructive",
      });
    } finally {
      startPendingRef.current = false;
      setStarting(false);
    }
  };

  const handleStop = async () => {
    setStopping(true);
    try {
      // Stop by session id (a 404 means the replay already ended — fine);
      // fall back to the kind-level stop when this panel never started one.
      if (status?.phase === "stopping") {
        await stopReplay(baseUrl, fetchWithHeaders, true);
      } else if (sessionId) {
        try {
          await stopSession(baseUrl, fetchWithHeaders, sessionId);
        } catch (e) {
          if (!(e instanceof ApiError && e.status === 404)) throw e;
        }
      } else {
        await stopReplay(baseUrl, fetchWithHeaders);
      }
    } catch (e) {
      toast({
        title: t("dialogs.replay.toast.stopFailedTitle"),
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setStopping(false);
    }
  };

  const active = status?.replay_active === true;
  const jointEntries = Object.entries(liveJoints);
  const hasSides = jointEntries.some(([name]) => /^(left|right)_/.test(name));
  const jointGroups = hasSides
    ? (["left", "right"] as const).map((side) => ({
        label: t(side === "left" ? "shared.visualizer.leftArm" : "shared.visualizer.rightArm"),
        entries: jointEntries
          .filter(([name]) => name.startsWith(`${side}_`))
          .map(([name, value]) => [name.slice(side.length + 1), value] as const),
      }))
    : [{ label: "", entries: jointEntries }];

  if (!selectedRecord || !selectedRecord.follower_ready) {
    return (
      <div className="rounded-md border border-dashed border-border bg-muted/30 p-3 text-xs text-muted-foreground">
        {selectedRecord
          ? t("dialogs.replay.robotNotReady", {
              // Localized diagnosis of WHY the record is unclean — the shared
              // renderer, not the English `robotSetupGap`.
              gap: formatRobotSetupGap(t, selectedRecord, "follower"),
            })
          : t("dialogs.replay.noRobot")}
      </div>
    );
  }

  if (!active) {
    return (
      <div className="space-y-3 rounded-md border border-border bg-muted/40 p-3">
        {thermalAvailable && <div className="space-y-2 text-xs">
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={thermalEnabled} onChange={(event) => { setThermalEnabled(event.target.checked); setRestConfirmed(false); }} />
            Repeat as a 30-minute thermal test (experimental)
          </label>
          {thermalEnabled && <>
            <label className="flex items-center gap-2">Experiment
              <select aria-label="Thermal experiment" value={experiment} onChange={(event) => setExperiment(event.target.value as typeof experiment)} className="rounded border bg-background p-1">
                <option value="baseline">Baseline — unchanged gains</option>
                <option value="shoulder_kp_85">Shoulder stiffness −15%</option>
              </select>
            </label>
            <p className="text-muted-foreground">Record a motion that starts and ends at the same supported rest pose. The test returns there at 110°C; 135°C is a manufacturer-reported critical reference. Feedback or tracking faults also stop the test.</p>
            {experiment === "shoulder_kp_85" && <p className="text-amber-700">Lower stiffness can increase position error. This is not a torque cap; recorded timing stays unchanged.</p>}
            <label className="flex items-start gap-2">
              <input type="checkbox" checked={restConfirmed} onChange={(event) => setRestConfirmed(event.target.checked)} />
              The arm is at its supported rest pose, the gripper is empty, and the motion and direct return path are clear. I will supervise the test.
            </label>
          </>}
        </div>}
        <ThermalReplayStatus status={status?.thermal_test} />
        <Button onClick={handleStart} disabled={starting || (thermalEnabled && thermalAvailable && !restConfirmed)} size="sm" className="gap-2">
          {starting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
          {t("dialogs.replay.start")}
        </Button>
        <p className="flex items-center gap-1 text-[11px] text-muted-foreground">
          <TriangleAlert className="h-3 w-3 shrink-0" />
          {t(selectedRecord.mode === "bimanual" ? "dialogs.replay.movesArmsWarning" : "dialogs.replay.movesArmWarning", { robot: selectedRecord.name })}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2 rounded-md border border-border bg-muted/40 p-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium">
          {t(PHASE_LABEL_KEY[status?.phase ?? "idle"] as never)}
        </span>
        {/* During the stopping phase the arm is returning to its start pose;
            a second press asks the server to release it now (the same
            two-press contract as teleoperation). */}
        <Button onClick={handleStop} disabled={stopping} size="sm" variant="destructive" className="gap-2">
          <Square className="h-3 w-3" />
          {status?.phase === "stopping" ? t("dialogs.replay.releaseNow") : t("dialogs.replay.stop")}
        </Button>
      </div>
      <ThermalReplayStatus status={status?.thermal_test} />
      {jointGroups.filter(({ entries }) => entries.length > 0).map(({ label, entries }) => (
        <section key={label} aria-label={label || undefined}>
          {label ? <h4 className="mb-1 text-xs font-medium">{label}</h4> : null}
          <div className="grid grid-cols-3 gap-x-3 gap-y-1 font-mono text-[10.5px] text-muted-foreground">
            {entries.map(([name, value]) => (
              <div key={name} className="flex justify-between">
                <span>{name.replace(/\.pos$/, "")}</span>
                <span className="tabular-nums text-foreground">{value.toFixed(1)}</span>
              </div>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
};

export default EpisodeReplayPanel;
