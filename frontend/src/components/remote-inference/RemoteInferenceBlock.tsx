import React, { useCallback, useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useGpuLauncher, useGpuTargets } from "@/hooks/useGpuLauncher";
import { getCurrentSession, stopSession } from "@/lib/sessionApi";
import type { RemoteInferenceStatus } from "@/hooks/useRemoteInferenceStatus";
import type { UseRemoteInferenceTransport } from "@/hooks/useRemoteInferenceTransport";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import CameraRoleBindings, {
  type CameraRoleOption,
  type CameraRoleSlot,
} from "./CameraRoleBindings";
import GpuLaunchSection from "./GpuLaunchSection";
import ModalRunLine from "./ModalRunLine";
import RemoteInferenceStatusPanel from "./RemoteInferenceStatusPanel";
import RemoteRunFields from "./RemoteRunFields";
import TransportSection from "./TransportSection";
import type { RemoteRunConfig } from "./remoteRunConfig";

/**
 * The whole remote-inference surface, as ONE mount point in the Deploy panel.
 *
 * Deliberately self-contained. The studio panels are being reworked on another
 * branch, so this keeps its footprint in the shared files down to a run-mode
 * entry, this element, and two guard flags — everything else can be rebased as
 * a unit.
 *
 * Order is the order of the work: what to configure, what to launch on the GPU,
 * what the transport looks like, and then what the run is doing. S3.8 filled in
 * the second of those: the Lab launches `modal run` itself (GpuLaunchSection),
 * and the hand-typed command below it collapses under "Run it yourself
 * instead" — kept, because it is the only route when `modal` is missing or
 * unauthenticated, the only route to a hand-tuned flag, and the ground truth an
 * operator compares against when the fingerprint watchdog fires.
 */
const RemoteInferenceBlock: React.FC<{
  /** The remote verb is the armed one — show the form. A live run keeps the
   * status panel up regardless. */
  armed: boolean;
  config: RemoteRunConfig;
  onConfigChange: (next: RemoteRunConfig) => void;
  hubIdDefault: string;
  /** Whether the selected checkpoint's policy family can be in-painted, i.e.
   * whether the `rtc` engine is meaningful for it. */
  rtcSupported: boolean;
  /** The checkpoint's `n_action_steps` — the ceiling on the horizon. */
  checkpointHorizon: number | null;
  /** Checkpoint camera roles with NO name match on this robot, in the order
   * the checkpoint declares them. Empty (the usual case) renders no section. */
  cameraRoleSlots: CameraRoleSlot[];
  /** The robot's cameras, as the options for those roles. */
  cameraRoleOptions: CameraRoleOption[];
  /** How many roles bound themselves by name and therefore need no control. */
  cameraRoleNameMatched: number;
  onCameraRoleChange: (requestKey: string, cameraName: string | null) => void;
  /** The effective task (typed, else the checkpoint's inherited default) —
   * the same string the start request carries. */
  task: string;
  transportState: UseRemoteInferenceTransport;
  status: RemoteInferenceStatus | null;
  /** Known when THIS tab started the run. Null after a reload or for a run
   * started elsewhere — the Stop below then resolves it from the server. */
  sessionId: string | null;
  onStopped: () => void;
}> = ({
  armed,
  config,
  onConfigChange,
  hubIdDefault,
  rtcSupported,
  checkpointHorizon,
  cameraRoleSlots,
  cameraRoleOptions,
  cameraRoleNameMatched,
  onCameraRoleChange,
  task,
  transportState,
  status,
  sessionId,
  onStopped,
}) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [stopping, setStopping] = useState(false);
  const [showManual, setShowManual] = useState(false);
  // The GPU is a LAB-LEVEL resource, not part of this session: it holds no
  // hardware and stopping it is not a safety action. It deliberately does NOT
  // gate the remote verb — that stays the transport probe's `operator_present`,
  // which observes the room rather than a log line.
  const gpu = useGpuLauncher(armed);
  // WHICH WORKSPACE PAYS. Its own hook beside the launcher rather than inside
  // it: the listing is a read of this MACHINE (two `modal … list --json`
  // calls), not of the launch, and it must keep answering — and keep being
  // pickable — while a GPU is up.
  const gpuTargets = useGpuTargets(armed);

  const active = status?.remote_inference_active === true;
  const showStatus = active || status?.exited === true;

  const stop = useCallback(async () => {
    setStopping(true);
    try {
      let id = sessionId;
      if (!id) {
        // A reload, or a run started from another tab / the SDK. Stopping is
        // never owner-gated server-side — safety outranks ownership — so
        // whoever can reach the API can stop the arm.
        const current = await getCurrentSession(baseUrl, fetchWithHeaders);
        if (current.session?.kind === "remote_inference") {
          id = current.session.id;
        }
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
      onStopped();
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
  }, [sessionId, baseUrl, fetchWithHeaders, toast, t, onStopped]);

  return (
    <div className="space-y-4">
      {armed ? (
        <>
          <p className="text-sm leading-relaxed text-muted-foreground">
            {t("remoteInference.intro")}
          </p>
          <RemoteRunFields
            config={config}
            onChange={onConfigChange}
            hubIdDefault={hubIdDefault}
            rtcSupported={rtcSupported}
            checkpointHorizon={checkpointHorizon}
            disabled={active}
          />
          {/* Between the run's own fields and the GPU launch, because that is
              where it sits in the work: the roles are part of what this run
              sends, and the command below is generated from a configured run.
              Renders nothing at all when every checkpoint camera matched a
              robot camera by name, which is the ordinary case. */}
          <CameraRoleBindings
            slots={cameraRoleSlots}
            cameras={cameraRoleOptions}
            nameMatchedCount={cameraRoleNameMatched}
            onChange={onCameraRoleChange}
            disabled={active}
          />
          <GpuLaunchSection
            launcher={gpu}
            targets={gpuTargets}
            config={config}
            hubIdDefault={hubIdDefault}
            task={task}
          />
          <Collapsible open={showManual} onOpenChange={setShowManual}>
            <CollapsibleTrigger className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
              <ChevronDown
                className={`h-3 w-3 transition-transform ${showManual ? "" : "-rotate-90"}`}
              />
              {t("remoteInference.modalRun.manualToggle")}
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-2">
              <ModalRunLine
                config={config}
                transport={transportState.transport}
                hubIdDefault={hubIdDefault}
                task={task}
                // So the pasted line bills the same workspace Start GPU would.
                profile={gpuTargets.profile}
                environment={gpuTargets.environment}
              />
            </CollapsibleContent>
          </Collapsible>
          <TransportSection transportState={transportState} />
        </>
      ) : null}

      {showStatus && status ? (
        <RemoteInferenceStatusPanel
          status={status}
          onStop={() => void stop()}
          stopping={stopping}
        />
      ) : null}
    </div>
  );
};

export default RemoteInferenceBlock;
