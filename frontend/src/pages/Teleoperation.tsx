import React, { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import VisualizerPanel from "@/components/control/VisualizerPanel";
import TeleopCameraPanel from "@/components/control/TeleopCameraPanel";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useRobots } from "@/hooks/useRobots";

const TeleoperationPage = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  // The teleop session is for the currently-selected robot; show two arms when
  // it's bimanual.
  const { selectedRecord } = useRobots();
  const bimanual = selectedRecord?.mode === "bimanual";
  // No Maker URDF ships yet, so a Maker session shows the live numeric joint
  // readout in the viewer's place rather than animating the SO-101 model with
  // a different arm's angles. See JointAngleReadout.
  const readoutOnly = selectedRecord?.arm_type === "maker";

  // Stop teleoperation exactly once, however the user leaves, so the back
  // button, an in-app link, and the unmount safety net can't double-stop or
  // double-toast.
  const stoppedRef = useRef(false);

  // Terminal outcome of a session that ended UNDER us (the status poll below
  // caught the worker dying mid-loop, or a stop from elsewhere whose cleanup
  // tripped) — rendered as an inline banner: failed red, ran_with_warning
  // amber (mirrors Inference.tsx). Null while the session is live.
  const [finished, setFinished] = useState<{
    outcome: "ran_with_warning" | "failed";
    error: string | null;
    hint: string | null;
  } | null>(null);

  // Poll the session status so a mid-loop death (unplugged bus, camera crash)
  // surfaces here instead of failing silently — the backend clears the
  // outcome fields on start, so a previous session's result can't trigger
  // this on mount. Stops polling once we've caught an outcome or the user
  // initiated a stop (the stop flow owns its own toasts).
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      if (cancelled || stoppedRef.current) return;
      try {
        const res = await fetchWithHeaders(
          `${baseUrl}/api/v1/teleoperation-status`,
        );
        if (!res.ok) return;
        const status = await res.json();
        if (cancelled || stoppedRef.current) return;
        if (
          !status.teleoperation_active &&
          !status.releasing &&
          (status.outcome === "failed" || status.outcome === "ran_with_warning")
        ) {
          // The session is already gone — mark the leave safety net handled so
          // unmount doesn't POST a spurious stop against a dead session.
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
  }, [baseUrl, fetchWithHeaders]);
  const stopTeleoperation = useCallback(async () => {
    if (stoppedRef.current) return;
    stoppedRef.current = true;
    try {
      const res = await fetchWithHeaders(`${baseUrl}/api/v1/stop-teleoperation`, {
        method: "POST",
      });
      const data = await res.json();
      if (data?.warning) {
        // Cleanup could not release an arm — torque may still be enabled and
        // the arm can stay rigid. Make this loud instead of claiming success.
        toast({
          title: t("pages.teleop.stoppedWarnTitle"),
          description: data.warning,
          variant: "destructive",
        });
      } else if (data?.releasing) {
        // The backend drives the follower straight back to its session-start
        // pose (no timed hold), then releases torque.
        toast({
          title: t("pages.teleop.stoppedTitle"),
          description:
            data.message ?? t("pages.teleop.releasingFallback"),
        });
        // The release happens after this response returns, so check once
        // after the return (progress-based, 10 s ceiling) whether it actually
        // succeeded (the toast store is global, so this fires even after
        // navigating away).
        setTimeout(async () => {
          try {
            const status = await fetchWithHeaders(
              `${baseUrl}/api/v1/teleoperation-status`
            ).then((r) => r.json());
            if (status?.last_cleanup_error) {
              toast({
                title: t("pages.teleop.checkArmTitle"),
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
          title: t("pages.teleop.stoppedTitle"),
          description: t("pages.teleop.disconnectedCleanly"),
        });
      }
    } catch {
      /* best-effort */
    }
  }, [baseUrl, fetchWithHeaders, toast]);

  // Deliberate in-app exits stop the session: the back button awaits
  // stopTeleoperation() then navigates (below), and any other in-app
  // navigation stops via this cleanup — on this legacy page, leaving the page
  // IS ending the session. There is no browser-unload stop beacon any more:
  // a session started through /api/v1/sessions carries a lease the server
  // safety-stops when its owner's heartbeats cease, and a beacon here could
  // kill a session some OTHER tab is legitimately running (the crossfire the
  // retired SingleTabGuard existed to prevent).
  useEffect(() => {
    return () => {
      stopTeleoperation();
    };
  }, [stopTeleoperation]);

  const handleGoBack = async () => {
    await stopTeleoperation();
    navigate("/");
  };

  const finishedWarn = finished?.outcome === "ran_with_warning";

  return (
    <div className="relative min-h-screen bg-background flex items-center justify-center p-2 sm:p-4">
      {finished && (
        <div
          className={`absolute top-4 left-1/2 -translate-x-1/2 z-50 w-[calc(100%-2rem)] max-w-xl rounded-lg border p-4 ${
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
              className={`w-2 h-2 rounded-full ${
                finishedWarn ? "bg-warn" : "bg-destructive"
              }`}
            />
            {finishedWarn
              ? t("pages.teleop.endedWithWarning")
              : t("pages.teleop.failed")}
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
            <pre className="mt-3 max-h-40 overflow-auto rounded bg-muted p-2 text-xs text-muted-foreground whitespace-pre-wrap break-words">
              {finished.error}
            </pre>
          )}
        </div>
      )}
      <div className="w-full h-[95vh] flex">
        <VisualizerPanel
          onGoBack={handleGoBack}
          className="lg:w-full"
          bimanual={bimanual}
          readoutOnly={readoutOnly}
          rightSlot={<TeleopCameraPanel />}
        />
      </div>
    </div>
  );
};

export default TeleoperationPage;
