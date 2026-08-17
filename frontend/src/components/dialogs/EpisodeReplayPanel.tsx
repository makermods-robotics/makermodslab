import React, { useEffect, useRef, useState } from "react";
import { Loader2, Square, TriangleAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useRobots, robotSetupGap } from "@/hooks/useRobots";
import { useSessionExitGuard } from "@/hooks/useSessionExitGuard";
import { useLiveJointReadout } from "@/hooks/useLiveJointReadout";
import {
  ReplayStatus,
  ReplayPhase,
  startReplay,
  getReplayStatus,
  stopReplay,
} from "@/lib/replayHardwareApi";

const POLL_MS = 1000;

const PHASE_LABEL: Record<ReplayPhase, string> = {
  idle: "Idle",
  easing_in: "Easing to start position…",
  playing: "Replaying",
  stopping: "Stopping…",
  done: "Done",
  error: "Error",
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
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { selectedRecord } = useRobots();
  const [status, setStatus] = useState<ReplayStatus | null>(null);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const doneRef = useRef(false);
  const localT0Ref = useRef<number | null>(null);

  const { joints: liveJoints } = useLiveJointReadout(status?.replay_active === true);

  const { markHandled } = useSessionExitGuard({
    active: status?.replay_active === true,
    confirmMessage: "Leaving stops the running replay. Continue?",
    beaconUrl: `${baseUrl}/stop-replay`,
    onLeave: () => {
      stopReplay(baseUrl, fetchWithHeaders).catch(() => {});
    },
    beaconFlagKey: "makermodslab:replay-stopped",
  });

  useEffect(() => {
    doneRef.current = false;
    localT0Ref.current = null;
  }, [repoId, episodeIndex]);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      if (doneRef.current) return;
      try {
        const next = await getReplayStatus(baseUrl, fetchWithHeaders);
        if (cancelled) return;
        setStatus(next);
        if (next.phase === "playing" && localT0Ref.current === null) {
          localT0Ref.current = performance.now() / 1000;
        }
        if (next.phase !== "playing") {
          localT0Ref.current = null;
        }
        const elapsed =
          next.phase === "playing" && localT0Ref.current !== null
            ? performance.now() / 1000 - localT0Ref.current
            : next.elapsed_s;
        onElapsedChange?.(elapsed, next.phase);
        if (!next.replay_active && (next.phase === "done" || next.phase === "error")) {
          if (next.phase === "error") {
            toast({
              title: "Replay failed",
              description: next.hint ?? next.error ?? "See the server log for details.",
              variant: "destructive",
              duration: 10000,
            });
          }
          markHandled();
          doneRef.current = true;
        }
      } catch (e) {
        if (!cancelled) {
          toast({
            title: "Lost connection to backend",
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
  }, [baseUrl, fetchWithHeaders, toast, markHandled, onElapsedChange]);

  const handleStart = async () => {
    if (!selectedRecord) return;
    setStarting(true);
    doneRef.current = false;
    try {
      const result = await startReplay(baseUrl, fetchWithHeaders, {
        repo_id: repoId,
        episode_index: episodeIndex,
        follower_port: selectedRecord.follower_port,
        follower_config: selectedRecord.follower_config,
        robot_name: selectedRecord.name,
      });
      if (result.warning) {
        toast({ title: "Started with a warning", description: result.warning, duration: 8000 });
      }
    } catch (e) {
      toast({
        title: "Could not start replay",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setStarting(false);
    }
  };

  const handleStop = async () => {
    setStopping(true);
    try {
      await stopReplay(baseUrl, fetchWithHeaders);
    } catch (e) {
      toast({
        title: "Could not stop replay",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setStopping(false);
    }
  };

  const active = status?.replay_active === true;

  if (!selectedRecord || !selectedRecord.follower_ready) {
    return (
      <div className="rounded-md border border-dashed border-border bg-muted/30 p-3 text-xs text-muted-foreground">
        {selectedRecord
          ? `Select a robot ready to replay: this robot ${robotSetupGap(selectedRecord, "follower")}.`
          : "Select a robot with a connected follower arm to replay this episode on hardware."}
      </div>
    );
  }

  if (!active) {
    return (
      <div className="flex items-center gap-3 rounded-md border border-border bg-muted/40 p-3">
        <Button onClick={handleStart} disabled={starting} size="sm" className="gap-2">
          {starting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
          Replay on hardware
        </Button>
        <p className="flex items-center gap-1 text-[11px] text-muted-foreground">
          <TriangleAlert className="h-3 w-3 shrink-0" />
          Moves {selectedRecord.name}&apos;s arm — make sure the area is clear.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2 rounded-md border border-border bg-muted/40 p-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium">{PHASE_LABEL[status?.phase ?? "idle"]}</span>
        <Button onClick={handleStop} disabled={stopping} size="sm" variant="destructive" className="gap-2">
          <Square className="h-3 w-3" />
          Stop
        </Button>
      </div>
      {Object.keys(liveJoints).length > 0 ? (
        <div className="grid grid-cols-3 gap-x-3 gap-y-1 font-mono text-[10.5px] text-muted-foreground">
          {Object.entries(liveJoints).map(([name, value]) => (
            <div key={name} className="flex justify-between">
              <span>{name.replace(/\.pos$/, "")}</span>
              <span className="tabular-nums text-foreground">{value.toFixed(1)}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
};

export default EpisodeReplayPanel;
