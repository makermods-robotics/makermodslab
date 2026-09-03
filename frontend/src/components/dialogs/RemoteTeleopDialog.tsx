import React, { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Loader2, RefreshCw, VideoOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ToastAction } from "@/components/ui/toast";
import UrdfViewer from "@/components/UrdfViewer";
import JointAngleReadout from "@/components/control/JointAngleReadout";
import RobotLayoutChip from "@/components/launchpad/RobotLayoutChip";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useEyebrowClass } from "@/hooks/useEyebrowClass";
import { useNodes } from "@/hooks/useNodes";
import type { RobotRecord } from "@/hooks/useRobots";
import { useSessionHeartbeat } from "@/hooks/useSessionHeartbeat";
import { useUnloadWarning } from "@/hooks/useUnloadWarning";
import { isCanArmType, type ArmType } from "@/lib/armTypes";
import { NodeEntry, hostingNodes, nodeDisplayName } from "@/lib/nodesApi";
import {
  formatRemoteRefusal,
  getRemoteTeleoperationStatus,
  remoteCameraUrl,
  type RemoteTeleoperationStatus,
} from "@/lib/remoteApi";
import { startSession, stopSession } from "@/lib/sessionApi";
import { tabOwnerId } from "@/lib/sessionOwner";
import { cn } from "@/lib/utils";

export interface RemoteTeleopDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The LOCAL robot record whose leader arm drives the remote follower. */
  robot: RobotRecord | null;
  /** The start was refused with 409 system.extra_missing — the owner opens
   * the remote-extra install flow. */
  onInstallRequested: () => void;
}

/**
 * One remote camera tile: the station's frames re-streamed as MJPEG by this
 * node. A dropped stream shows a retry tile (click to retry now) — the
 * lightweight sibling of BackendCameraStream, without its why-probe. Clearing
 * `src` on detach is what makes the browser drop the HTTP connection.
 */
const RemoteCameraTile: React.FC<{ name: string }> = ({ name }) => {
  const { t } = useTranslation();
  const { baseUrl } = useApi();
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [down, setDown] = useState(false);

  const attachImg = useCallback((node: HTMLImageElement | null) => {
    if (node === null && imgRef.current) imgRef.current.src = "";
    imgRef.current = node;
  }, []);

  const retry = () => {
    setAttempt((a) => a + 1);
    setDown(false);
  };

  return (
    <div className="flex flex-col gap-1">
      {/* Camera names are data (the station's own) — rendered verbatim. */}
      <span className="truncate font-mono text-[11px] text-muted-foreground">
        {name}
      </span>
      {down ? (
        <button
          type="button"
          onClick={retry}
          className="flex aspect-video w-full flex-col items-center justify-center gap-1 rounded-md border border-border bg-muted text-muted-foreground"
        >
          <VideoOff className="h-5 w-5" />
          <span className="text-[10px]">
            {t("dialogs.remoteTeleop.cameraFailed")}
          </span>
        </button>
      ) : (
        <img
          key={attempt}
          ref={attachImg}
          src={`${remoteCameraUrl(baseUrl, name)}?r=${attempt}`}
          onError={() => setDown(true)}
          alt={t("dialogs.remoteTeleop.cameraAlt", { name })}
          className="aspect-video w-full rounded-md border border-border bg-black object-contain"
        />
      )}
    </div>
  );
};

/**
 * Whether a station's hosted arm belongs to a different family than the
 * local record's. The server refuses that pairing (robot.schema_mismatch),
 * so the picker greys the row out instead of letting Start find out. Both
 * ids are data, compared verbatim.
 */
const armFamilyDiffers = (node: NodeEntry, localArmType: ArmType | undefined) => {
  const hosted = node.capabilities?.hosting;
  return !!hosted && !!localArmType && hosted.arm_type !== localArmType;
};

/** The picker's station row — ComputeSelector's radio-row look. */
const StationRow: React.FC<{
  node: NodeEntry;
  checked: boolean;
  /** The local record's arm family; a station hosting another family is
   * greyed out with the reason. */
  localArmType: ArmType | undefined;
  onPick: () => void;
}> = ({ node, checked, localArmType, onPick }) => {
  const { t } = useTranslation();
  const hosted = node.capabilities?.hosting;
  const mismatch = armFamilyDiffers(node, localArmType);
  return (
    <button
      type="button"
      role="radio"
      aria-checked={checked}
      aria-disabled={mismatch || undefined}
      disabled={mismatch}
      onClick={onPick}
      className={cn(
        "flex w-full items-center gap-2.5 bg-background px-3 py-2 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        checked ? "bg-accent/60" : "text-muted-foreground hover:text-foreground",
      )}
    >
      <span
        aria-hidden
        className={cn(
          "relative h-3.5 w-3.5 shrink-0 rounded-full border",
          checked ? "border-primary" : "border-muted-foreground/60",
        )}
      >
        {checked ? (
          <span className="absolute inset-[2.5px] rounded-full bg-primary" />
        ) : null}
      </span>
      <span aria-hidden className="h-[7px] w-[7px] shrink-0 rounded-full bg-ok" />
      <span className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span
          className={cn(
            "truncate text-sm font-medium",
            checked && "text-foreground",
          )}
        >
          {nodeDisplayName(node)}
        </span>
        {hosted ? (
          <span className="truncate text-xs text-muted-foreground">
            {t("dialogs.remoteTeleop.hostingRobot", { robot: hosted.robot })}
            {mismatch ? (
              <>
                {" · "}
                <span className="text-warn">
                  {t("dialogs.remoteTeleop.armMismatch")}
                </span>
              </>
            ) : null}
          </span>
        ) : null}
      </span>
      {hosted ? (
        // The arm family tag: the value is data, only the label localizes
        // (the same fallback rule as the robot picker).
        <span className="whitespace-nowrap rounded border border-border px-1.5 py-px text-[11px] font-medium text-muted-foreground">
          {t(`robot.corner.armType.${hosted.arm_type}` as never, {
            defaultValue: hosted.arm_type,
          })}
        </span>
      ) : null}
    </button>
  );
};

/**
 * The operator side of remote teleoperation: a station picker (the node
 * registry's peers that are hosting right now), then — once the session
 * starts — TeleopDialog's floating viewer fed by the WS broadcast (which
 * carries the REMOTE follower's joints in the local shape), the station's
 * cameras re-streamed as MJPEG, and the transport round-trip readout. Stop
 * is single-press (there is no follower here to return to rest). Every
 * in-app exit stops the session once by id; a browser-level leave is the
 * lease's job.
 *
 * Mounted only while open (the shell below) so the registry poll runs only
 * while the picker is on screen.
 */
const RemoteTeleopDialogBody: React.FC<Omit<RemoteTeleopDialogProps, "open">> = ({
  onOpenChange,
  robot,
  onInstallRequested,
}) => {
  const { t } = useTranslation();
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();
  const eyebrow = useEyebrowClass();
  const { nodes, loading: nodesLoading, forceRefresh } = useNodes();

  const stations = hostingNodes(nodes);
  const [stationId, setStationId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState<RemoteTeleoperationStatus | null>(null);
  const [finished, setFinished] = useState<{
    outcome: "ran_with_warning" | "failed";
    error: string | null;
    hint: string | null;
  } | null>(null);
  const stoppedRef = useRef(false);
  const live = sessionId !== null;

  // The picked station, while it is still listed; its hosted arm type picks
  // the viewer (the remote follower is what the broadcast shows). A station
  // whose family stopped matching (it re-hosted another robot) can't be
  // started against — the row is greyed out, and Start follows it.
  const station = stations.find((n) => n.instance_id === stationId) ?? null;
  const stationMismatch =
    !!station && armFamilyDiffers(station, robot?.arm_type);
  const [hostedArmType, setHostedArmType] = useState<ArmType | undefined>();

  useSessionHeartbeat(sessionId, tabOwnerId(), live && finished === null);
  useUnloadWarning(live && finished === null);

  const handleStart = async () => {
    if (!robot || !stationId || !station || stationMismatch) return;
    setStarting(true);
    try {
      const { session, warnings } = await startSession(baseUrl, fetchWithHeaders, {
        kind: "remote_teleoperation",
        robot: robot.name,
        owner: tabOwnerId(),
        // `station` is the peer's instance id — data.
        options: { station: stationId },
      });
      setHostedArmType(station.capabilities?.hosting?.arm_type as ArmType | undefined);
      stoppedRef.current = false;
      setSessionId(session.id);
      if (warnings?.length) {
        toast({
          title: t("dialogs.remoteTeleop.toast.startedWarningTitle"),
          description: warnings.join(" "),
          duration: 10000,
        });
      } else {
        toast({
          title: t("dialogs.remoteTeleop.toast.startedTitle"),
          description: t("dialogs.remoteTeleop.toast.startedFallback", {
            station: nodeDisplayName(station),
          }),
        });
      }
    } catch (e) {
      const refusal = formatRemoteRefusal(
        t,
        e,
        t("robot.remote.failedFallback"),
      );
      if (refusal) {
        toast({
          title: t("robot.remote.failedTitle"),
          description: refusal.message,
          variant: "destructive",
          action: refusal.needsInstall ? (
            <ToastAction
              altText={t("robot.remote.installAction")}
              onClick={onInstallRequested}
            >
              {t("robot.remote.installAction")}
            </ToastAction>
          ) : undefined,
        });
      } else {
        toast({
          title: t("common.connectionError.title"),
          description: t("common.connectionError.description"),
          variant: "destructive",
        });
      }
    } finally {
      setStarting(false);
    }
  };

  // Poll the session status while live: cameras, metrics, and a mid-loop
  // death (the station dropped, the leader unplugged) as an inline banner.
  useEffect(() => {
    if (!live) return;
    let cancelled = false;
    const tick = async () => {
      if (cancelled || stoppedRef.current) return;
      try {
        const next = await getRemoteTeleoperationStatus(baseUrl, fetchWithHeaders);
        if (cancelled || stoppedRef.current) return;
        setStatus(next);
        if (
          !next.remote_teleoperation_active &&
          (next.outcome === "failed" || next.outcome === "ran_with_warning")
        ) {
          stoppedRef.current = true;
          setFinished({
            outcome: next.outcome,
            error: next.error ?? null,
            hint: next.hint ?? null,
          });
        }
      } catch {
        /* best-effort; the next tick retries */
      }
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [live, baseUrl, fetchWithHeaders]);

  const stopRemote = useCallback(async () => {
    if (stoppedRef.current || !sessionId) return;
    stoppedRef.current = true;
    try {
      const { result } = await stopSession(baseUrl, fetchWithHeaders, sessionId);
      const data = result as {
        success?: boolean;
        message?: string;
        warning?: string;
      } | null;
      if (data?.warning) {
        toast({
          title: t("dialogs.remoteTeleop.toast.stoppedCheckArm"),
          description: data.warning,
          variant: "destructive",
        });
      } else {
        toast({
          title: t("dialogs.remoteTeleop.toast.stopped"),
          description:
            data?.message ?? t("dialogs.remoteTeleop.toast.disconnected"),
        });
      }
    } catch {
      /* 404: already gone — nothing left to stop or announce */
    }
  }, [sessionId, baseUrl, fetchWithHeaders, toast, t]);

  // Unmounting while live stops the session — closing the viewer IS ending
  // the session, exactly as TeleopDialog does it.
  useEffect(() => {
    if (!live) return;
    return () => {
      stopRemote();
    };
  }, [live, stopRemote]);

  // ESC: in the picker just closes; live, it ends the session.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      if (live) stopRemote();
      onOpenChange(false);
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [live, stopRemote, onOpenChange]);

  const handleStop = async () => {
    await stopRemote();
    onOpenChange(false);
  };

  const finishedWarn = finished?.outcome === "ran_with_warning";
  const readoutOnly = isCanArmType(hostedArmType ?? robot?.arm_type);
  const bimanual = robot?.mode === "bimanual";
  const title = robot
    ? t("dialogs.remoteTeleop.titleWithRobot", { robot: robot.name })
    : t("dialogs.remoteTeleop.title");
  const metrics = status?.metrics ?? null;
  const ms = (v: number | null | undefined) =>
    v == null ? "—" : `${Math.round(v)} ms`;

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
        live ? "w-[min(94vw,1000px)]" : "w-[min(92vw,560px)]"
      }`}
    >
      <div className="flex items-center gap-2 border-b border-border px-4 py-2">
        {live ? (
          <span className="h-2 w-2 animate-pulse rounded-full bg-destructive" />
        ) : null}
        <span className="text-sm font-semibold text-foreground">{title}</span>
        <RobotLayoutChip arms={robot?.arms} />
        {live ? (
          <Button
            size="sm"
            onClick={handleStop}
            className="ml-auto bg-destructive text-destructive-foreground hover:bg-destructive/90"
          >
            {t("dialogs.remoteTeleop.stop")}
          </Button>
        ) : (
          <Button
            size="sm"
            variant="ghost"
            className="ml-auto"
            onClick={() => onOpenChange(false)}
          >
            {t("common.cancel")}
          </Button>
        )}
      </div>

      {!live ? (
        <div className="flex flex-col gap-3 p-3">
          <div className="flex items-center justify-between gap-2 px-1">
            <span className={eyebrow}>
              {t("dialogs.remoteTeleop.stationsHeading")}
            </span>
            <button
              type="button"
              onClick={forceRefresh}
              title={t("dialogs.remoteTeleop.refreshStations")}
              aria-label={t("dialogs.remoteTeleop.refreshStations")}
              className="flex items-center rounded p-0.5 text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground"
            >
              <RefreshCw className="h-3 w-3" />
            </button>
          </div>
          <div
            role="radiogroup"
            aria-label={t("dialogs.remoteTeleop.stationsHeading")}
            className="divide-y divide-border overflow-hidden rounded-md border border-border"
          >
            {nodesLoading && stations.length === 0 ? (
              <div className="flex items-center gap-2 bg-background px-3 py-2.5 text-xs text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" />
                {t("dialogs.remoteTeleop.stationsLoading")}
              </div>
            ) : stations.length === 0 ? (
              <div className="bg-background px-3 py-2.5 text-xs text-muted-foreground">
                {t("dialogs.remoteTeleop.stationsEmpty")}
              </div>
            ) : (
              stations.map((node) => (
                <StationRow
                  key={node.instance_id ?? node.url ?? nodeDisplayName(node)}
                  node={node}
                  checked={node.instance_id === stationId}
                  localArmType={robot?.arm_type}
                  onPick={() => setStationId(node.instance_id)}
                />
              ))
            )}
          </div>
          <div className="flex justify-end">
            <Button
              size="sm"
              disabled={!robot || !station || stationMismatch || starting}
              onClick={handleStart}
            >
              {starting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  {t("dialogs.remoteTeleop.starting")}
                </>
              ) : (
                t("dialogs.remoteTeleop.start")
              )}
            </Button>
          </div>
        </div>
      ) : (
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
                  ? t("dialogs.remoteTeleop.endedWithWarning")
                  : t("dialogs.remoteTeleop.failed")}
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

          {/* Station name, room and the metrics are data — verbatim. */}
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span>{t("dialogs.remoteTeleop.stationLabel")}:</span>
              <span className="font-mono text-foreground">
                {status?.station
                  ? (status.station.name ?? status.station.url)
                  : station
                    ? nodeDisplayName(station)
                    : "—"}
              </span>
            </span>
            {status?.room && (
              <span className="flex items-center gap-1.5">
                <span>{t("dialogs.remoteTeleop.roomLabel")}:</span>
                <span className="font-mono text-foreground">{status.room}</span>
              </span>
            )}
            <span className="flex items-center gap-1.5">
              <span>{t("dialogs.remoteTeleop.latency")}:</span>
              {metrics && metrics.rtt_ms_last != null ? (
                <span className="font-mono tabular-nums text-foreground">
                  {t("dialogs.remoteTeleop.latencyLast")} {ms(metrics.rtt_ms_last)}
                  {" · "}
                  {t("dialogs.remoteTeleop.latencyMean")} {ms(metrics.rtt_ms_mean)}
                  {" · "}
                  {t("dialogs.remoteTeleop.latencyP95")} {ms(metrics.rtt_ms_p95)}
                  {" · "}
                  {t("dialogs.remoteTeleop.observations")} {metrics.observations}
                  {" · "}
                  {t("dialogs.remoteTeleop.dropped")} {metrics.states_dropped}
                </span>
              ) : (
                <span>{t("dialogs.remoteTeleop.latencyWaiting")}</span>
              )}
            </span>
          </div>

          <div className="flex gap-3">
            <div className={cn("flex gap-3", bimanual ? "flex-[2]" : "flex-1")}>
              {bimanual ? (
                <>
                  <div className="flex-1">
                    <span className="mb-1 block text-xs text-muted-foreground">
                      {t("dialogs.remoteTeleop.leftArm")}
                    </span>
                    <div className="h-[400px] overflow-hidden rounded-md border border-border">
                      {viewer("joints")}
                    </div>
                  </div>
                  <div className="flex-1">
                    <span className="mb-1 block text-xs text-muted-foreground">
                      {t("dialogs.remoteTeleop.rightArm")}
                    </span>
                    <div className="h-[400px] overflow-hidden rounded-md border border-border">
                      {viewer("joints_right")}
                    </div>
                  </div>
                </>
              ) : (
                <div className="h-[440px] flex-1 overflow-hidden rounded-md border border-border">
                  {viewer("joints")}
                </div>
              )}
            </div>
            <div className="flex w-[min(40%,320px)] shrink-0 flex-col gap-2">
              <span className="text-xs text-muted-foreground">
                {t("dialogs.remoteTeleop.cameras")}
              </span>
              {status && status.cameras.length === 0 ? (
                <span className="text-xs text-muted-foreground">
                  {t("dialogs.remoteTeleop.noCameras")}
                </span>
              ) : (
                <div className="flex max-h-[440px] flex-col gap-2 overflow-auto">
                  {(status?.cameras ?? []).map((name) => (
                    <RemoteCameraTile key={name} name={name} />
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const RemoteTeleopDialog: React.FC<RemoteTeleopDialogProps> = ({
  open,
  ...rest
}) => (open ? <RemoteTeleopDialogBody {...rest} /> : null);

export default RemoteTeleopDialog;
