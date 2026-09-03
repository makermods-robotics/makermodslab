import React, { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import UrdfViewer from "@/components/UrdfViewer";
import JointAngleReadout from "@/components/control/JointAngleReadout";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useRobots } from "@/hooks/useRobots";
import { useSessionHeartbeat } from "@/hooks/useSessionHeartbeat";
import { useUnloadWarning } from "@/hooks/useUnloadWarning";
import { isCanArmType, type ArmType } from "@/lib/armTypes";
import { getHostingStatus, type HostingDescriptor } from "@/lib/remoteApi";
import { stopSession } from "@/lib/sessionApi";
import { tabOwnerId } from "@/lib/sessionOwner";

export interface HostingDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Identity of the `hosting` session the launcher started via
   * POST /api/v1/sessions — this dialog heartbeats its lease and stops it
   * by id. */
  sessionId: string | null;
}

/**
 * The station side of remote teleoperation as a centered floating viewer —
 * TeleopDialog's twin for a `hosting` session. The follower is held exactly
 * as local teleop holds it (the WS broadcast carries its joints, so the same
 * viewer works unchanged), and the stop is teleoperation's two-press
 * contract: the first Stop returns the arm to rest before torque is released
 * (the dialog stays up, polling, until the release lands), a second press
 * releases now. ESC and unmount count as the first press and close at once —
 * the return finishes in the background, like TeleopDialog's Done. A mid-loop
 * death surfaces as an inline banner; a browser-level leave is the lease's
 * job (missed heartbeats → safety stop).
 */
const HostingDialog: React.FC<HostingDialogProps> = ({
  open,
  onOpenChange,
  sessionId,
}) => {
  const { t } = useTranslation();
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { selectedRecord } = useRobots();

  // How many stop presses have been sent for this session (0, 1, 2). Two is
  // terminal — nothing more to stop or announce, however the user leaves.
  const pressesRef = useRef(0);
  // First press answered `releasing`: the arm is on its way to rest.
  const [releasing, setReleasing] = useState(false);
  // The live descriptor (operator, room) from the status poll.
  const [descriptor, setDescriptor] = useState<HostingDescriptor | null>(null);
  // Terminal outcome of a session that ended UNDER us — inline banner.
  const [finished, setFinished] = useState<{
    outcome: "ran_with_warning" | "failed";
    error: string | null;
    hint: string | null;
  } | null>(null);

  // Fresh session per open.
  useEffect(() => {
    if (open) {
      pressesRef.current = 0;
      setReleasing(false);
      setDescriptor(null);
      setFinished(null);
    }
  }, [open]);

  useSessionHeartbeat(sessionId, tabOwnerId(), open && finished === null);
  useUnloadWarning(open && finished === null);

  // The one-shot post-release check TeleopDialog does: the release happens
  // after the stop response returns, so look once, later, whether cleanup
  // left an error (the toast store is global, so this outlives the dialog).
  const scheduleCleanupCheck = useCallback(() => {
    setTimeout(async () => {
      try {
        const status = await getHostingStatus(baseUrl, fetchWithHeaders);
        if (status.last_cleanup_error) {
          toast({
            title: t("dialogs.hosting.toast.checkArm"),
            description: status.hint
              ? `${status.hint} (${status.last_cleanup_error})`
              : status.last_cleanup_error,
            variant: "destructive",
          });
        }
      } catch {
        /* best-effort */
      }
    }, 13000);
  }, [baseUrl, fetchWithHeaders, toast, t]);

  // Poll the hosting status: keeps the operator/room line fresh, catches a
  // mid-loop death, and — after the first stop press — notices the release
  // landing so the dialog can close on its own.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const tick = async () => {
      if (cancelled || pressesRef.current >= 2) return;
      try {
        const status = await getHostingStatus(baseUrl, fetchWithHeaders);
        if (cancelled || pressesRef.current >= 2) return;
        if (status.hosting) setDescriptor(status.hosting);
        if (status.hosting_active || status.releasing) return;
        if (pressesRef.current === 1) {
          // Our stop's return-to-rest finished: the session is gone.
          pressesRef.current = 2;
          if (status.last_cleanup_error) {
            toast({
              title: t("dialogs.hosting.toast.checkArm"),
              description: status.hint
                ? `${status.hint} (${status.last_cleanup_error})`
                : status.last_cleanup_error,
              variant: "destructive",
            });
          }
          onOpenChange(false);
          return;
        }
        if (
          status.outcome === "failed" ||
          status.outcome === "ran_with_warning"
        ) {
          // Died under us — nothing left to stop; show why.
          pressesRef.current = 2;
          setFinished({
            outcome: status.outcome,
            error: status.error ?? null,
            hint: status.hint ?? null,
          });
        }
      } catch {
        /* best-effort; the next tick retries */
      }
    };
    const id = setInterval(tick, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [open, baseUrl, fetchWithHeaders, onOpenChange, toast, t]);

  /** One stop press. Returns true when the session is still releasing (the
   * caller decides whether to wait for it or leave). */
  const pressStop = useCallback(async (): Promise<boolean> => {
    if (pressesRef.current >= 2 || !sessionId) return false;
    pressesRef.current += 1;
    try {
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
        return false;
      }
      if (data?.releasing) {
        setReleasing(true);
        toast({
          title: t("dialogs.hosting.toast.stopped"),
          description: data.message ?? t("dialogs.hosting.toast.releasing"),
        });
        return true;
      }
      pressesRef.current = 2;
      if (data?.success) {
        toast({
          title: t("dialogs.hosting.toast.stopped"),
          description: t("dialogs.hosting.toast.disconnected"),
        });
      }
      return false;
    } catch {
      // 404: the session is already gone (safety-stopped, stopped elsewhere).
      pressesRef.current = 2;
      return false;
    }
  }, [sessionId, baseUrl, fetchWithHeaders, toast, t]);

  // Leaving (ESC, unmount) counts as the first press and closes at once —
  // the return-to-rest finishes in the background, checked once later.
  const leave = useCallback(async () => {
    if (pressesRef.current === 0) {
      const stillReleasing = await pressStop();
      if (stillReleasing) scheduleCleanupCheck();
    }
    // A second press is never sent on leave: the arm keeps returning to rest.
    pressesRef.current = 2;
  }, [pressStop, scheduleCleanupCheck]);

  useEffect(() => {
    if (!open) return;
    return () => {
      leave();
    };
  }, [open, leave]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      leave();
      onOpenChange(false);
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open, leave, onOpenChange]);

  // The Stop button: first press waits for the release (dialog stays up);
  // second press releases now and closes.
  const handleStop = async () => {
    if (pressesRef.current === 0) {
      const stillReleasing = await pressStop();
      if (!stillReleasing) onOpenChange(false);
      return;
    }
    if (pressesRef.current === 1) {
      await pressStop();
      pressesRef.current = 2;
      scheduleCleanupCheck();
      onOpenChange(false);
    }
  };

  const finishedWarn = finished?.outcome === "ran_with_warning";
  const robotName = descriptor?.robot ?? selectedRecord?.name ?? null;
  const armType = (descriptor?.arm_type ?? selectedRecord?.arm_type) as
    | ArmType
    | undefined;
  const readoutOnly = isCanArmType(armType);
  const bimanual = (descriptor?.mode ?? selectedRecord?.mode) === "bimanual";
  const title = robotName
    ? t("dialogs.hosting.titleWithRobot", { robot: robotName })
    : t("dialogs.hosting.title");

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
        <span className="h-2 w-2 animate-pulse rounded-full bg-destructive" />
        <span className="text-sm font-semibold text-foreground">{title}</span>
        <Button
          size="sm"
          onClick={handleStop}
          disabled={finished !== null}
          className="ml-auto bg-destructive text-destructive-foreground hover:bg-destructive/90"
        >
          {releasing
            ? t("dialogs.hosting.releaseNow")
            : t("dialogs.hosting.stop")}
        </Button>
      </div>

      <div className="flex flex-col gap-3 p-3">
        {finished && (
          <div
            className={`shrink-0 rounded-lg border p-4 ${
              finishedWarn
                ? "border-warn/40 bg-warn/10"
                : "border-destructive/40 bg-destructive/10"
            }`}
          >
            <div
              className={`flex items-center gap-2 text-sm font-semibold ${
                finishedWarn ? "text-warn" : "text-destructive"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  finishedWarn ? "bg-warn" : "bg-destructive"
                }`}
              />
              {finishedWarn
                ? t("dialogs.hosting.endedWithWarning")
                : t("dialogs.hosting.failed")}
            </div>
            {finished.hint && (
              <p
                className={`mt-2 text-sm leading-relaxed ${
                  finishedWarn ? "text-warn/90" : "text-destructive/90"
                }`}
              >
                {finished.hint}
              </p>
            )}
            {finished.error && (
              <pre className="mt-3 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-muted p-2 text-xs text-muted-foreground">
                {finished.error}
              </pre>
            )}
          </div>
        )}

        {releasing && !finished && (
          <div className="shrink-0 rounded-lg border border-warn/40 bg-warn/10 p-3 text-sm text-warn">
            {t("dialogs.hosting.releasingBanner")}
          </div>
        )}

        {/* Operator identity and room name are data — rendered verbatim. */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
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
