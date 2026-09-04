import React, { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import UrdfViewer from "@/components/UrdfViewer";
import JointAngleReadout from "@/components/control/JointAngleReadout";
import RobotLayoutChip from "@/components/launchpad/RobotLayoutChip";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useHostingStatus } from "@/hooks/useHostingStatus";
import { useStationStatus } from "@/hooks/useStationStatus";
import { useRobots } from "@/hooks/useRobots";
import { isCanArmType, type ArmType } from "@/lib/armTypes";
import { getHostingStatus, type HostingPhase } from "@/lib/remoteApi";
import { getCurrentSession, stopSession } from "@/lib/sessionApi";
import { cn } from "@/lib/utils";

export interface HostingDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Station mode only: open the hosted-robot picker (StationRobotDialog).
   * The owner closes this view first — the picker is a separate dialog, not
   * nested under this one's ESC handler. */
  onChangeRobot?: () => void;
}

// Static catalog keys per phase (never a runtime-built template) so
// keyUsage.test.ts can verify each one; the phase VALUE is data.
const PHASE_KEYS: Record<HostingPhase, string> = {
  parked: "dialogs.hosting.phase.parked",
  engaging: "dialogs.hosting.phase.engaging",
  engaged: "dialogs.hosting.phase.engaged",
  parking: "dialogs.hosting.phase.parking",
};

const PHASE_STYLES: Record<HostingPhase, string> = {
  parked: "border-border bg-muted text-muted-foreground",
  engaging: "border-warn/40 bg-warn/10 text-warn",
  engaged: "border-ok/40 bg-ok/10 text-ok",
  parking: "border-warn/40 bg-warn/10 text-warn",
};

/**
 * The station side of remote teleoperation as a STATUS VIEW. Hosting is not
 * started here — a station is launched with `makermodslab --sfu --host
 * <robot>` and hosts from startup, re-arming after any local session — so
 * this dialog only shows the live session (phase, seat holder, room, the
 * follower on the same viewer local teleop uses) and offers the one action
 * a person at the station needs: releasing the arm for local use.
 *
 * That release is a stop of the hosting session by id (resolved from
 * /api/v1/sessions/current — station mode starts the session owner-less, so
 * there is no lease to heartbeat). Engaged, it follows teleoperation's
 * two-press contract (return to rest, then release; a second press releases
 * now); parked, the arm is already at rest with torque off and the stop is
 * immediate. Closing the dialog (ESC, the Close button, unmount) never stops
 * anything: the station keeps hosting whether or not anyone is watching.
 */
const HostingDialog: React.FC<HostingDialogProps> = ({
  open,
  onOpenChange,
  onChangeRobot,
}) => {
  const { t } = useTranslation();
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { records } = useRobots();
  const { status } = useHostingStatus({ enabled: open, intervalMs: 2000 });
  // The station's remembered choice — names the robot while hosting is down
  // (the descriptor only rides on a live session) and gates the picker.
  const { status: station } = useStationStatus({ enabled: open, intervalMs: 3000 });

  // How many release presses have been sent (0, 1, 2). Two is terminal.
  const pressesRef = useRef(0);
  // The session our first press stopped — a later tick that finds a
  // different (re-armed) session or none knows the release landed.
  const stoppedIdRef = useRef<string | null>(null);
  const [releasing, setReleasing] = useState(false);
  const [stopping, setStopping] = useState(false);

  // Fresh per open.
  useEffect(() => {
    if (open) {
      pressesRef.current = 0;
      stoppedIdRef.current = null;
      setReleasing(false);
      setStopping(false);
    }
  }, [open]);

  const warnIfCleanupFailed = useCallback(
    (lastCleanupError: string | null, hint: string | null) => {
      if (!lastCleanupError) return;
      toast({
        title: t("dialogs.hosting.toast.checkArm"),
        description: hint ? `${hint} (${lastCleanupError})` : lastCleanupError,
        variant: "destructive",
      });
    },
    [toast, t],
  );

  // After the first (engaged) press: notice the release landing so the
  // dialog can close on its own. Station mode re-arms hosting seconds after
  // a release, so "hosting inactive" alone could be missed between two
  // polls — the session id is the reliable tell.
  useEffect(() => {
    if (!open || !releasing) return;
    let cancelled = false;
    const tick = async () => {
      if (cancelled || pressesRef.current >= 2) return;
      try {
        const [{ session }, hosting] = await Promise.all([
          getCurrentSession(baseUrl, fetchWithHeaders),
          getHostingStatus(baseUrl, fetchWithHeaders),
        ]);
        if (cancelled || pressesRef.current >= 2) return;
        const gone =
          session === null ||
          session.kind !== "hosting" ||
          session.id !== stoppedIdRef.current;
        if (!gone && hosting.releasing) return;
        pressesRef.current = 2;
        warnIfCleanupFailed(hosting.last_cleanup_error, hosting.hint);
        onOpenChange(false);
      } catch {
        /* best-effort; the next tick retries */
      }
    };
    const id = setInterval(tick, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [open, releasing, baseUrl, fetchWithHeaders, onOpenChange, warnIfCleanupFailed]);

  /** One release press against the live hosting session. */
  const pressRelease = useCallback(async () => {
    if (pressesRef.current >= 2 || stopping) return;
    setStopping(true);
    try {
      let sessionId = stoppedIdRef.current;
      if (sessionId === null) {
        const { session } = await getCurrentSession(baseUrl, fetchWithHeaders);
        if (!session || session.kind !== "hosting") {
          // Nothing live to release (a local session already has the arm).
          pressesRef.current = 2;
          onOpenChange(false);
          return;
        }
        sessionId = session.id;
        stoppedIdRef.current = sessionId;
      }
      pressesRef.current += 1;
      const { result } = await stopSession(baseUrl, fetchWithHeaders, sessionId);
      const data = result as {
        success?: boolean;
        releasing?: boolean;
        message?: string;
        warning?: string;
      } | null;
      if (data?.warning) {
        pressesRef.current = 2;
        toast({
          title: t("dialogs.hosting.toast.stoppedCheckArm"),
          description: data.warning,
          variant: "destructive",
        });
        onOpenChange(false);
        return;
      }
      if (data?.releasing) {
        // Engaged: the arm returns to rest first; the dialog stays up until
        // the release lands (or a second press releases now).
        setReleasing(true);
        toast({
          title: t("dialogs.hosting.toast.stopped"),
          description: data.message ?? t("dialogs.hosting.toast.releasing"),
        });
        return;
      }
      // Parked (immediate), or the second press: done.
      pressesRef.current = 2;
      toast({
        title: t("dialogs.hosting.toast.stopped"),
        description: t("dialogs.hosting.toast.disconnected"),
      });
      onOpenChange(false);
    } catch {
      // 404: the session is already gone (stopped elsewhere, preempted).
      pressesRef.current = 2;
      onOpenChange(false);
    } finally {
      setStopping(false);
    }
  }, [stopping, baseUrl, fetchWithHeaders, onOpenChange, toast, t]);

  // ESC closes the status view — it never stops hosting.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      onOpenChange(false);
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open, onOpenChange]);

  const descriptor = status?.hosting ?? null;
  const active = status?.hosting_active === true;
  const phase = descriptor?.phase ?? null;
  // The hosted robot is the descriptor's while a session is live, else the
  // station's remembered choice — never the corner's selected robot, which
  // drives LOCAL flows and may legitimately differ.
  const robotName = descriptor?.robot ?? station?.robot ?? null;
  const hostedRecord = robotName !== null ? records[robotName] : undefined;
  const armType = (descriptor?.arm_type ?? hostedRecord?.arm_type) as
    | ArmType
    | undefined;
  const readoutOnly = isCanArmType(armType);
  const bimanual = (descriptor?.mode ?? hostedRecord?.mode) === "bimanual";
  // The layout chip beside the name — from the hosted robot's local record.
  const layoutArms = hostedRecord?.arms;
  const title = robotName
    ? t("dialogs.hosting.titleWithRobot", { robot: robotName })
    : t("dialogs.hosting.title");
  const died =
    status?.outcome === "failed" || status?.outcome === "ran_with_warning";
  const diedWarn = status?.outcome === "ran_with_warning";

  if (!open) return null;

  const viewer = (jointsKey: "joints" | "joints_right") =>
    readoutOnly ? (
      <JointAngleReadout
        jointsKey={jointsKey === "joints" ? "joints_deg" : "joints_deg_right"}
      />
    ) : (
      <UrdfViewer jointsKey={jointsKey} variant="light" compact />
    );

  return (
    <div
      role="dialog"
      aria-label={title}
      className={`fixed left-1/2 top-1/2 z-50 flex -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-lg border border-border bg-background shadow-2xl ${
        bimanual ? "w-[min(94vw,1000px)]" : "w-[min(92vw,640px)]"
      }`}
    >
      <div className="flex items-center gap-2 border-b border-border px-4 py-2">
        <span
          className={cn(
            "h-2 w-2 rounded-full",
            phase === "engaged"
              ? "animate-pulse bg-destructive"
              : active
                ? "bg-ok"
                : "bg-muted-foreground/60",
          )}
        />
        <span className="text-sm font-semibold text-foreground">{title}</span>
        <RobotLayoutChip arms={layoutArms} />
        {phase && (
          <span
            className={cn(
              "rounded-full border px-2 py-px text-[11px] font-semibold",
              PHASE_STYLES[phase],
            )}
          >
            {t(PHASE_KEYS[phase] as never, { defaultValue: phase })}
          </span>
        )}
        <div className="ml-auto flex items-center gap-1.5">
          {station?.station_mode && onChangeRobot && (
            <Button
              size="sm"
              variant="ghost"
              disabled={stopping || releasing}
              onClick={() => {
                onOpenChange(false);
                onChangeRobot();
              }}
            >
              {t("dialogs.hosting.changeRobot")}
            </Button>
          )}
          <Button
            size="sm"
            variant="ghost"
            onClick={() => onOpenChange(false)}
          >
            {t("common.close")}
          </Button>
          <Button
            size="sm"
            onClick={pressRelease}
            disabled={!active || stopping || pressesRef.current >= 2}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
          >
            {releasing
              ? t("dialogs.hosting.releaseNow")
              : t("dialogs.hosting.release")}
          </Button>
        </div>
      </div>

      <div className="flex flex-col gap-3 p-3">
        {releasing && (
          <div className="shrink-0 rounded-lg border border-warn/40 bg-warn/10 p-3 text-sm text-warn">
            {t("dialogs.hosting.releasingBanner")}
          </div>
        )}

        {status && !active && !releasing && (
          died ? (
            // The last session ended badly — say why (station mode retries
            // on its own; the person at the station still wants the reason).
            <div
              className={`shrink-0 rounded-lg border p-4 ${
                diedWarn
                  ? "border-warn/40 bg-warn/10"
                  : "border-destructive/40 bg-destructive/10"
              }`}
            >
              <div
                className={`flex items-center gap-2 text-sm font-semibold ${
                  diedWarn ? "text-warn" : "text-destructive"
                }`}
              >
                <span
                  className={`h-2 w-2 rounded-full ${
                    diedWarn ? "bg-warn" : "bg-destructive"
                  }`}
                />
                {diedWarn
                  ? t("dialogs.hosting.endedWithWarning")
                  : t("dialogs.hosting.failed")}
              </div>
              {status.hint && (
                <p
                  className={`mt-2 text-sm leading-relaxed ${
                    diedWarn ? "text-warn/90" : "text-destructive/90"
                  }`}
                >
                  {status.hint}
                </p>
              )}
              {status.error && (
                <pre className="mt-3 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-muted p-2 text-xs text-muted-foreground">
                  {status.error}
                </pre>
              )}
            </div>
          ) : (
            <div className="shrink-0 rounded-lg border border-border bg-muted/40 p-3 text-sm text-muted-foreground">
              {t("dialogs.hosting.inactive")}
            </div>
          )
        )}

        {/* Phase, operator identity and room name are data — verbatim. */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
          {phase && (
            <span className="flex items-center gap-1.5">
              <span>{t("dialogs.hosting.phaseLabel")}:</span>
              <span className="font-medium text-foreground">
                {t(PHASE_KEYS[phase] as never, { defaultValue: phase })}
              </span>
            </span>
          )}
          <span className="flex items-center gap-1.5">
            <span
              aria-hidden
              className={`h-2 w-2 rounded-full ${
                descriptor?.active_operator
                  ? "bg-ok"
                  : "animate-pulse bg-muted-foreground/60"
              }`}
            />
            <span>{t("dialogs.hosting.operatorLabel")}:</span>
            <span className="font-mono text-foreground">
              {descriptor?.active_operator ??
                t("dialogs.hosting.waitingOperator")}
            </span>
          </span>
          {descriptor?.room && (
            <span className="flex items-center gap-1.5">
              <span>{t("dialogs.hosting.roomLabel")}:</span>
              <span className="font-mono text-foreground">{descriptor.room}</span>
            </span>
          )}
        </div>

        {descriptor?.station_mode && (
          <p className="text-xs text-muted-foreground">
            {t("dialogs.hosting.stationModeNote")}
          </p>
        )}

        {bimanual ? (
          <div className="flex gap-3">
            <div className="flex-1">
              <span className="mb-1 block text-xs text-muted-foreground">
                {t("dialogs.hosting.leftArm")}
              </span>
              <div className="h-[400px] overflow-hidden rounded-md border border-border">
                {viewer("joints")}
              </div>
            </div>
            <div className="flex-1">
              <span className="mb-1 block text-xs text-muted-foreground">
                {t("dialogs.hosting.rightArm")}
              </span>
              <div className="h-[400px] overflow-hidden rounded-md border border-border">
                {viewer("joints_right")}
              </div>
            </div>
          </div>
        ) : (
          <div className="h-[440px] overflow-hidden rounded-md border border-border">
            {viewer("joints")}
          </div>
        )}
      </div>
    </div>
  );
};

export default HostingDialog;
