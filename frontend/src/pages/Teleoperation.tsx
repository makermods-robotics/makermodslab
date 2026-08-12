import React, { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import VisualizerPanel from "@/components/control/VisualizerPanel";
import TeleopCameraPanel from "@/components/control/TeleopCameraPanel";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useRobots } from "@/hooks/useRobots";
import { useOnboarding } from "@/contexts/OnboardingContext";
import { useOnceFlag } from "@/lib/onboarding/storage";
import { teleopTour } from "@/lib/onboarding/tours";

const TeleoperationPage = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();
  // The teleop session is for the currently-selected robot; show two arms when
  // it's bimanual.
  const { selectedRecord } = useRobots();
  const bimanual = selectedRecord?.mode === "bimanual";

  // Stop teleoperation exactly once, however the user leaves, so the back
  // button, an in-app link, and the unmount safety net can't double-stop or
  // double-toast.
  const stoppedRef = useRef(false);

  const { seen: hasSeenTeleopTour, markSeen: markTeleopTourSeen } =
    useOnceFlag("makerlab:seen-teleop-tour");
  const { start: startTour } = useOnboarding();

  // First visit to this page, walk through the single-step reveal.
  useEffect(() => {
    if (!hasSeenTeleopTour) startTour(teleopTour, markTeleopTourSeen);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
      const res = await fetchWithHeaders(`${baseUrl}/stop-teleoperation`, {
        method: "POST",
      });
      const data = await res.json();
      if (data?.warning) {
        // Cleanup could not release an arm — torque may still be enabled and
        // the arm can stay rigid. Make this loud instead of claiming success.
        toast({
          title: "Teleoperation stopped — check the arm",
          description: data.warning,
          variant: "destructive",
        });
      } else if (data?.releasing) {
        // The backend drives the follower straight back to its session-start
        // pose (no timed hold), then releases torque.
        toast({
          title: "Teleoperation stopped",
          description:
            data.message ?? "The arm returns to its starting position, then goes limp.",
        });
        // The release happens after this response returns, so check once
        // after the return (progress-based, 10 s ceiling) whether it actually
        // succeeded (the toast store is global, so this fires even after
        // navigating away).
        setTimeout(async () => {
          try {
            const status = await fetchWithHeaders(`${baseUrl}/teleoperation-status`).then((r) =>
              r.json()
            );
            if (status?.last_cleanup_error) {
              toast({
                title: "Check the arm",
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
          title: "Teleoperation stopped",
          description: "The arm was disconnected cleanly.",
        });
      }
    } catch {
      /* best-effort */
    }
  }, [baseUrl, fetchWithHeaders, toast]);

  // Cover every exit path so a session can't keep running and block the next
  // start with "already active":
  //   - the back button awaits stopTeleoperation() then navigates (below);
  //   - any other in-app navigation unmounts this component → stop via cleanup;
  //   - a browser-level leave (URL change, reload, tab close) never runs React
  //     cleanup, so `pagehide` fires a keepalive stop that survives the unload
  //     and stashes a flag the next page reads to confirm the clean disconnect.
  //     It uses a bare fetch (no JSON Content-Type) so the request stays a CORS
  //     "simple request" and isn't dropped to a preflight mid-unload.
  useEffect(() => {
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
  }, [baseUrl, stopTeleoperation]);

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
              ? "Teleoperation ended with a cleanup warning"
              : "Teleoperation failed"}
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
      <div className="w-full h-[95vh] flex" data-tour="teleop-page">
        <VisualizerPanel
          onGoBack={handleGoBack}
          className="lg:w-full"
          bimanual={bimanual}
          rightSlot={<TeleopCameraPanel />}
        />
      </div>
    </div>
  );
};

export default TeleoperationPage;
