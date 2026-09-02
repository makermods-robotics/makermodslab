import React, { useEffect, useState } from "react";
import { Activity, Radio, ShieldCheck, TriangleAlert } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/badge";
import { useApi } from "@/contexts/ApiContext";
import {
  countServoFaults,
  getRemoteTeleoperationStatus,
  getServoHealthStatus,
  type RemoteTeleoperationStatus,
  type ServoHealthStatus,
} from "@/lib/armStatusApi";

const value = (number: number | null, suffix: string) =>
  number == null ? "—" : `${number}${suffix}`;

const RemoteServoStatusPanel: React.FC = () => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const [remote, setRemote] = useState<RemoteTeleoperationStatus | null>(null);
  const [health, setHealth] = useState<ServoHealthStatus | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let disposed = false;
    const refresh = async () => {
      const [remoteResult, healthResult] = await Promise.allSettled([
        getRemoteTeleoperationStatus(
          baseUrl,
          fetchWithHeaders,
          controller.signal
        ),
        getServoHealthStatus(baseUrl, fetchWithHeaders, controller.signal),
      ]);
      if (disposed) return;
      if (remoteResult.status === "fulfilled") setRemote(remoteResult.value);
      if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    };
    refresh();
    const interval = window.setInterval(refresh, 1500);
    return () => {
      disposed = true;
      controller.abort();
      window.clearInterval(interval);
    };
  }, [baseUrl, fetchWithHeaders]);

  const faults = countServoFaults(health);
  const motors = health?.arms.flatMap((arm) => arm.motors) ?? [];

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-muted-foreground" />
          <h2 className="text-sm font-semibold text-foreground">
            {t("pages.teleop.armLink.heading")}
          </h2>
        </div>
        <Badge variant="outline" className="font-mono text-[10px]">
          {remote?.state ?? t("pages.teleop.armLink.loading")}
        </Badge>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
        <div className="rounded border border-border bg-background p-2">
          <div className="flex items-center gap-1 text-muted-foreground">
            <Radio className="h-3 w-3" />
            {t("pages.teleop.armLink.remote")}
          </div>
          <p className="mt-1 font-medium text-foreground">
            {remote?.live_hardware_enabled
              ? t("pages.teleop.armLink.live")
              : t("pages.teleop.armLink.simulationOnly")}
          </p>
        </div>
        <div className="rounded border border-border bg-background p-2">
          <div className="flex items-center gap-1 text-muted-foreground">
            <ShieldCheck className="h-3 w-3" />
            {t("pages.teleop.armLink.maintenance")}
          </div>
          <p className="mt-1 font-medium text-foreground">
            {health?.maintenance.state === "disabled"
              ? t("pages.teleop.armLink.disabled")
              : health?.maintenance.state ?? "—"}
          </p>
        </div>
      </div>

      <div className="mt-3 flex items-center justify-between text-xs">
        <span className="text-muted-foreground">
          {t("pages.teleop.armLink.servoHealth")}
        </span>
        <span
          className={
            faults > 0 ? "font-medium text-destructive" : "text-foreground"
          }
        >
          {health?.available
            ? faults > 0
              ? t("pages.teleop.armLink.faults", { count: faults })
              : t("pages.teleop.armLink.healthy", { count: motors.length })
            : t("pages.teleop.armLink.noOwner")}
        </span>
      </div>

      {health?.last_error && (
        <div className="mt-2 flex gap-2 rounded border border-destructive/30 bg-destructive/5 p-2 text-xs text-destructive">
          <TriangleAlert className="mt-0.5 h-3 w-3 flex-none" />
          <span>{health.last_error}</span>
        </div>
      )}

      {motors.length > 0 && (
        <div className="mt-3 max-h-36 space-y-1 overflow-y-auto font-mono text-[10px]">
          {motors.map((motor) => (
            <div
              key={`${motor.joint}:${motor.id}`}
              className="grid grid-cols-[1fr_auto_auto] gap-2 rounded bg-background px-2 py-1.5"
            >
              <span className="truncate text-foreground">
                {motor.joint} · {motor.model} #{motor.id}
              </span>
              <span className="text-muted-foreground">
                {value(motor.temperature_c, "°C")}
              </span>
              <span
                className={
                  motor.faults?.length
                    ? "text-destructive"
                    : "text-muted-foreground"
                }
              >
                {value(motor.current_a, "A")}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default RemoteServoStatusPanel;
