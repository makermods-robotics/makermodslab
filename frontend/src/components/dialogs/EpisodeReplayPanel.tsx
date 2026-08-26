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
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  // Identity of the session THIS panel started (POST /api/v1/sessions).
  const [sessionId, setSessionId] = useState<string | null>(null);
  const doneRef = useRef(false);
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
    doneRef.current = false;
    try {
      // Robot NAME + episode selection only — the follower port/config
      // resolve server-side from the saved record. The owner attaches the
      // lease the heartbeat above renews while the replay plays.
      const session = await startSession(baseUrl, fetchWithHeaders, {
        kind: "replay",
        robot: selectedRecord.name,
        owner: tabOwnerId(),
        options: {
          repo_id: repoId,
          episode_index: episodeIndex,
        },
      });
      setSessionId(session.id);
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
      setStarting(false);
    }
  };

  const handleStop = async () => {
    setStopping(true);
    try {
      // Stop by session id (a 404 means the replay already ended — fine);
      // fall back to the kind-level stop when this panel never started one.
      if (sessionId) {
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
      <div className="flex items-center gap-3 rounded-md border border-border bg-muted/40 p-3">
        <Button onClick={handleStart} disabled={starting} size="sm" className="gap-2">
          {starting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
          {t("dialogs.replay.start")}
        </Button>
        <p className="flex items-center gap-1 text-[11px] text-muted-foreground">
          <TriangleAlert className="h-3 w-3 shrink-0" />
          {t("dialogs.replay.movesArmWarning", { robot: selectedRecord.name })}
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
        <Button onClick={handleStop} disabled={stopping} size="sm" variant="destructive" className="gap-2">
          <Square className="h-3 w-3" />
          {t("dialogs.replay.stop")}
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
