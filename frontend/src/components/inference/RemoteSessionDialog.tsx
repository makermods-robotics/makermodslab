import React, { useCallback, useState } from "react";
import { useTranslation } from "react-i18next";
import { Loader2 } from "lucide-react";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { useApi } from "@/contexts/ApiContext";
import { useStudio } from "@/contexts/StudioContext";
import { useToast } from "@/hooks/use-toast";
import { useGpuLauncher } from "@/hooks/useGpuLauncher";
import { useRemoteInferenceStatus } from "@/hooks/useRemoteInferenceStatus";
import { useSessionHeartbeat } from "@/hooks/useSessionHeartbeat";
import { useUnloadWarning } from "@/hooks/useUnloadWarning";
import { getCurrentSession, stopSession } from "@/lib/sessionApi";
import { tabOwnerId } from "@/lib/sessionOwner";
import RemoteSessionBody from "@/components/remote-inference/RemoteSessionBody";

/**
 * The session dialog for a REMOTE-inference run.
 *
 * A sibling of `InferenceSessionDialog` rather than a branch inside it, and
 * that is a deliberate structural choice: the local dialog's ~40 hooks are all
 * about a local rollout (its 1 Hz `/inference-status` poll, its log fetch, the
 * coaching key handler, a lease keyed on `inference_active`), and a remote run
 * satisfies none of them — a heartbeat gated on the LOCAL active flag would go
 * quiet under a live remote run and let the expiry watchdog safety-stop it 60 s
 * in. So the two share what an operator actually sees — the Dialog shell and
 * `sessionFrame`'s pill / timer / phase line — and nothing else.
 *
 * Everything below the frame is `RemoteSessionBody`. The GPU is not controlled
 * from here: Start GPU / Stop GPU and the idle countdown stay on the Deploy
 * panel, because the GPU outlives the run.
 */
const RemoteSessionDialog: React.FC<{
  /** Identity from POST /api/v1/sessions. Null after a reload or for a run
   * this tab did not start — the stop below then resolves it from the
   * server, the same fallback the retired inline panel used. */
  sessionId: string | null;
  onExit: () => void;
}> = ({ sessionId, onExit }) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { openStudio, deployPrefill } = useStudio();
  const { toast } = useToast();
  const [stopping, setStopping] = useState(false);

  // 1 Hz while the run is live; the `session_changed` hint refetches eagerly.
  const { status } = useRemoteInferenceStatus(true);
  // Read-only, and only for one line: WHO IS BEING BILLED. The launcher is the
  // only thing that knows the Modal profile, and there are no GPU controls
  // here — the lifecycle stays on the panel.
  const gpu = useGpuLauncher(true);

  // Treat the pre-first-status window as live: the session was just claimed,
  // and a dismissible dialog there would leave an energized arm behind.
  const live = status == null || status.remote_inference_active === true;
  useSessionHeartbeat(sessionId, tabOwnerId(), live);
  useUnloadWarning(live);

  const handleStop = useCallback(async () => {
    setStopping(true);
    try {
      let id = sessionId;
      if (!id) {
        // A reload, or a run started from another tab / the SDK. Stopping is
        // never owner-gated server-side — safety outranks ownership — so
        // whoever can reach the API can stop the arm.
        const current = await getCurrentSession(baseUrl, fetchWithHeaders);
        if (current.session?.kind === "remote_inference") id = current.session.id;
      }
      if (!id) {
        toast({
          title: t("remoteInference.toast.stopFailed"),
          description: t("remoteInference.toast.noSession"),
          variant: "destructive",
        });
        return;
      }
      await stopSession(baseUrl, fetchWithHeaders, id);
    } catch (e) {
      toast({
        title: t("remoteInference.toast.stopFailed"),
        // The server's own error text.
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setStopping(false);
    }
  }, [sessionId, baseUrl, fetchWithHeaders, toast, t]);

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !live) onExit();
      }}
    >
      <DialogContent
        hideClose
        onOpenAutoFocus={(e) => {
          e.preventDefault();
          (e.currentTarget as HTMLElement | null)?.focus();
        }}
        onEscapeKeyDown={(e) => {
          if (live) e.preventDefault();
        }}
        onPointerDownOutside={(e) => {
          if (live) e.preventDefault();
        }}
        onInteractOutside={(e) => {
          if (live) e.preventDefault();
        }}
        className="max-h-[92vh] w-max max-w-[95vw] min-w-[min(36rem,95vw)] gap-0 overflow-y-auto p-6"
        aria-describedby={undefined}
      >
        <DialogTitle className="sr-only">
          {t("inference.dialogTitle")}
        </DialogTitle>

        {!status ? (
          <div className="flex items-center justify-center py-20 text-muted-foreground">
            <Loader2 className="mr-3 h-6 w-6 animate-spin" />
            {t("remoteInference.status.connectingSubtitle")}
          </div>
        ) : (
          <RemoteSessionBody
            status={status}
            gpuProfile={gpu.status?.profile ?? null}
            gpuDeviceName={gpu.status?.device_name ?? null}
            gpuType={gpu.status?.gpu ?? null}
            onStop={() => void handleStop()}
            stopping={stopping}
            onClose={onExit}
            // The same offer the local dialog makes when a run ends badly, and
            // for the same reason: the moment the operator KNOWS the policy is
            // imperfect is the moment they watch it finish. It needs a policy
            // to hand on, which only a prefill carries.
            onCoach={
              deployPrefill
                ? () => {
                    onExit();
                    openStudio("deploy", {
                      deploy: { ...deployPrefill, mode: "coach" },
                    });
                  }
                : null
            }
          />
        )}
      </DialogContent>
    </Dialog>
  );
};

export default RemoteSessionDialog;
