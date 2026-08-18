import React, { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import UrdfViewer from "@/components/UrdfViewer";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useRobots } from "@/hooks/useRobots";

export interface TeleopDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/**
 * Live teleoperation as a centered floating viewer — not a window dialog: a
 * big URDF visualizer card in the middle of the screen. The session state
 * machine is ported verbatim from pages/Teleoperation.tsx — every exit path
 * (Done, ESC, unmount, browser-level leave) stops the session exactly once,
 * and a mid-loop death surfaces as an inline banner.
 */
const TeleopDialog: React.FC<TeleopDialogProps> = ({ open, onOpenChange }) => {
  const { t } = useTranslation();
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();
  // The teleop session is for the currently-selected robot; show two arms when
  // it's bimanual.
  const { selectedRecord } = useRobots();
  const bimanual = selectedRecord?.mode === "bimanual";

  // Stop teleoperation exactly once, however the user leaves, so Done, the
  // dialog close, and the unmount safety net can't double-stop or
  // double-toast. Reset when a new session opens the dialog.
  const stoppedRef = useRef(false);

  // Terminal outcome of a session that ended UNDER us (the status poll below
  // caught the worker dying mid-loop) — rendered as an inline banner: failed
  // red, ran_with_warning amber. Null while the session is live.
  const [finished, setFinished] = useState<{
    outcome: "ran_with_warning" | "failed";
    error: string | null;
    hint: string | null;
  } | null>(null);

  // Fresh session per open.
  useEffect(() => {
    if (open) {
      stoppedRef.current = false;
      setFinished(null);
    }
  }, [open]);

  // Poll the session status so a mid-loop death (unplugged bus, camera crash)
  // surfaces here instead of failing silently — the backend clears the
  // outcome fields on start, so a previous session's result can't trigger
  // this on mount. Stops polling once we've caught an outcome or the user
  // initiated a stop (the stop flow owns its own toasts).
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const tick = async () => {
      if (cancelled || stoppedRef.current) return;
      try {
        const res = await fetchWithHeaders(`${baseUrl}/teleoperation-status`);
        if (!res.ok) return;
        const status = await res.json();
        if (cancelled || stoppedRef.current) return;
        if (
          !status.teleoperation_active &&
          !status.releasing &&
          (status.outcome === "failed" || status.outcome === "ran_with_warning")
        ) {
          // The session is already gone — mark the leave safety net handled so
          // close doesn't POST a spurious stop against a dead session.
          stoppedRef.current = true;
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
  }, [open, baseUrl, fetchWithHeaders]);

  const stopTeleoperation = useCallback(async () => {
    if (stoppedRef.current) return;
    stoppedRef.current = true;
    try {
      const res = await fetchWithHeaders(`${baseUrl}/stop-teleoperation`, {
        method: "POST",
      });
      const data = await res.json();
      if (data?.warning) {
        // Cleanup could not release an arm — torque may still be enabled and
        // the arm can stay rigid. Make this loud instead of claiming success.
        toast({
          title: t("dialogs.teleop.toast.stoppedCheckArm"),
          // `data.warning` is backend prose — rendered verbatim.
          description: data.warning,
          variant: "destructive",
        });
      } else if (data?.releasing) {
        // The backend drives the follower straight back to its session-start
        // pose (no timed hold), then releases torque.
        toast({
          title: t("dialogs.teleop.toast.stopped"),
          // The backend's own message wins; only the fallback is a catalog
          // string.
          description: data.message ?? t("dialogs.teleop.toast.releasing"),
        });
        // The release happens after this response returns, so check once
        // after the return (progress-based, 10 s ceiling) whether it actually
        // succeeded (the toast store is global, so this fires even after the
        // dialog closes).
        setTimeout(async () => {
          try {
            const status = await fetchWithHeaders(
              `${baseUrl}/teleoperation-status`,
            ).then((r) => r.json());
            if (status?.last_cleanup_error) {
              toast({
                title: t("dialogs.teleop.toast.checkArm"),
                // Lead with the plain-language hint when the backend mapped
                // one (e.g. gripper overload) — the raw text follows.
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
      } else if (data?.success) {
        toast({
          title: t("dialogs.teleop.toast.stopped"),
          description: t("dialogs.teleop.toast.disconnected"),
        });
      }
    } catch {
      /* best-effort */
    }
  }, [baseUrl, fetchWithHeaders, toast, t]);

  // Cover every exit path while a session is live so it can't keep running and
  // block the next start with "already active":
  //   - Done and dialog-close await/fire stopTeleoperation;
  //   - unmount while open stops via cleanup;
  //   - a browser-level leave (URL change, reload, tab close) never runs React
  //     cleanup, so `pagehide` fires a keepalive stop that survives the unload
  //     and stashes a flag the next page reads to confirm the clean disconnect.
  //     It uses a bare fetch (no JSON Content-Type) so the request stays a CORS
  //     "simple request" and isn't dropped to a preflight mid-unload.
  useEffect(() => {
    if (!open) return;
    const handlePageHide = () => {
      try {
        sessionStorage.setItem("makermodslab:teleop-stopped", "1");
      } catch {
        /* sessionStorage may be unavailable; the stop below still runs */
      }
      fetch(`${baseUrl}/stop-teleoperation`, {
        method: "POST",
        keepalive: true,
      }).catch(() => {});
    };
    window.addEventListener("pagehide", handlePageHide);

    return () => {
      window.removeEventListener("pagehide", handlePageHide);
      stopTeleoperation();
    };
  }, [open, baseUrl, stopTeleoperation]);

  // ESC ends the session (the Radix dialog used to own this). Capture phase +
  // preventDefault so StudioOverlay's own ESC handler (bubble, gated on
  // defaultPrevented) doesn't also close the studio underneath.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      stopTeleoperation();
      onOpenChange(false);
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open, stopTeleoperation, onOpenChange]);

  const handleDone = async () => {
    await stopTeleoperation();
    onOpenChange(false);
  };

  const finishedWarn = finished?.outcome === "ran_with_warning";

  // One string for the window's aria-label and its visible heading — the
  // robot's own name is data, interpolated verbatim.
  const title = selectedRecord
    ? t("dialogs.teleop.titleWithRobot", { robot: selectedRecord.name })
    : t("dialogs.teleop.title");

  if (!open) return null;

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
          onClick={handleDone}
          className="ml-auto bg-destructive text-destructive-foreground hover:bg-destructive/90"
        >
          {t("dialogs.teleop.done")}
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
                ? t("dialogs.teleop.endedWithWarning")
                : t("dialogs.teleop.failed")}
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

        {bimanual ? (
          <div className="flex gap-3">
            <div className="flex-1">
              <span className="mb-1 block text-xs text-muted-foreground">
                {t("dialogs.teleop.leftArm")}
              </span>
              <div className="h-[400px] overflow-hidden rounded-md border border-border">
                <UrdfViewer jointsKey="joints" variant="light" compact />
              </div>
            </div>
            <div className="flex-1">
              <span className="mb-1 block text-xs text-muted-foreground">
                {t("dialogs.teleop.rightArm")}
              </span>
              <div className="h-[400px] overflow-hidden rounded-md border border-border">
                <UrdfViewer jointsKey="joints_right" variant="light" compact />
              </div>
            </div>
          </div>
        ) : (
          <div className="h-[440px] overflow-hidden rounded-md border border-border">
            <UrdfViewer variant="light" compact />
          </div>
        )}
      </div>
    </div>
  );
};

export default TeleopDialog;
