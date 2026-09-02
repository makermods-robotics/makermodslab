import { useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  remoteRuntimeView,
  type RemoteTeleoperationStatus,
} from "@/lib/remoteTeleoperationApi";

const displayNumber = (value: number | null, suffix = "") =>
  value == null ? "—" : `${Math.max(0, Math.round(value))}${suffix}`;

const ReceiptValue = ({ value }: { value: boolean | null | undefined }) => {
  const { t } = useTranslation();
  if (value === true) {
    return <span className="font-medium text-ok">{t("remoteTeleop.receipt.yes")}</span>;
  }
  if (value === false) {
    return (
      <span className="font-medium text-destructive">
        {t("remoteTeleop.receipt.no")}
      </span>
    );
  }
  return (
    <span className="font-medium text-warn">
      {t("remoteTeleop.receipt.unknown")}
    </span>
  );
};

const Metric = ({
  label,
  value,
  danger = false,
}: {
  label: string;
  value: React.ReactNode;
  danger?: boolean;
}) => (
  <div className="min-w-0 rounded-lg border border-border bg-background/60 p-3">
    <dt className="text-xs text-muted-foreground">{label}</dt>
    <dd
      className={cn(
        "mt-1 truncate text-sm font-medium",
        danger && "text-destructive",
      )}
    >
      {value}
    </dd>
  </div>
);

export interface RemoteRuntimeStatusPanelProps {
  status: RemoteTeleoperationStatus | null;
  loading?: boolean;
}

/** Compact, operational status. Unknown receipt fields remain visibly unknown. */
export default function RemoteRuntimeStatusPanel({
  status,
  loading = false,
}: RemoteRuntimeStatusPanelProps) {
  const { t } = useTranslation();
  const view = remoteRuntimeView(status);
  const rejectionCount = Object.values(view.rejections).reduce(
    (total, count) => total + (Number.isFinite(count) ? count : 0),
    0,
  );
  const receipt = view.stopReceipt;
  const stateLabel = loading
    ? t("remoteTeleop.status.loading")
    : view.state || t("remoteTeleop.status.unknown");

  return (
    <section className="rounded-xl border border-border bg-card p-4 shadow-1">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-base font-semibold">
            {t("remoteTeleop.status.heading")}
          </h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            {t("remoteTeleop.status.liveDescription")}
          </p>
        </div>
        <Badge
          variant={view.faultLockout ? "destructive" : "outline"}
          className={cn(
            !view.faultLockout && view.enabled && "border-ok/40 text-ok",
          )}
        >
          {view.faultLockout
            ? t("remoteTeleop.status.locked")
            : view.enabled
              ? t("remoteTeleop.status.enabled")
              : t("remoteTeleop.status.disabled")}
        </Badge>
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3">
        <Metric label={t("remoteTeleop.status.state")} value={stateLabel} />
        <Metric
          label={t("remoteTeleop.status.owner")}
          value={view.owner ?? "—"}
        />
        <Metric
          label={t("remoteTeleop.status.credential")}
          value={view.credentialId ?? "—"}
        />
        <Metric
          label={t("remoteTeleop.status.actionWatchdog")}
          value={displayNumber(view.watchdog.action_remaining_ms ?? null, " ms")}
        />
        <Metric
          label={t("remoteTeleop.status.controlWatchdog")}
          value={displayNumber(view.watchdog.control_remaining_ms ?? null, " ms")}
        />
        <Metric
          label={t("remoteTeleop.status.browserWatchdog")}
          value={displayNumber(view.watchdog.browser_remaining_ms ?? null, " ms")}
        />
        <Metric
          label={t("remoteTeleop.status.lastSequence")}
          value={displayNumber(view.lastSequence)}
        />
        <Metric
          label={t("remoteTeleop.status.rejections")}
          value={rejectionCount}
          danger={rejectionCount > 0}
        />
        <Metric
          label={t("remoteTeleop.status.latency")}
          value={displayNumber(view.latencyMs, " ms")}
        />
        <Metric
          label={t("remoteTeleop.status.clockUncertainty")}
          value={displayNumber(view.clockUncertaintyMs, " ms")}
        />
        <Metric
          label={t("remoteTeleop.status.torque")}
          value={
            view.torqueOffConfirmed === true ? (
              <span className="text-ok">{t("remoteTeleop.status.torqueOff")}</span>
            ) : view.torqueOffConfirmed === false ? (
              <span className="text-destructive">
                {t("remoteTeleop.status.torqueEnabled")}
              </span>
            ) : (
              <span className="text-warn">
                {t("remoteTeleop.status.torqueUnknown")}
              </span>
            )
          }
        />
        <Metric
          label={t("remoteTeleop.status.faultLockout")}
          value={
            view.faultLockout
              ? t("remoteTeleop.status.active")
              : t("remoteTeleop.status.inactive")
          }
          danger={view.faultLockout}
        />
      </dl>

      <div className="mt-3 rounded-lg border border-border bg-background/60 p-3">
        <p className="text-xs font-medium text-muted-foreground">
          {t("remoteTeleop.status.stopReceipt")}
        </p>
        <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-2 text-xs sm:grid-cols-3">
          <div>
            <dt className="text-muted-foreground">
              {t("remoteTeleop.receipt.accepted")}
            </dt>
            <dd><ReceiptValue value={receipt?.accepted} /></dd>
          </div>
          <div>
            <dt className="text-muted-foreground">
              {t("remoteTeleop.receipt.halted")}
            </dt>
            <dd><ReceiptValue value={receipt?.advancement_halted} /></dd>
          </div>
          <div>
            <dt className="text-muted-foreground">
              {t("remoteTeleop.receipt.disableRequested")}
            </dt>
            <dd><ReceiptValue value={receipt?.torque_disable_requested} /></dd>
          </div>
          <div>
            <dt className="text-muted-foreground">
              {t("remoteTeleop.receipt.hardwareStopped")}
            </dt>
            <dd><ReceiptValue value={receipt?.hardware_stop_completed} /></dd>
          </div>
          <div>
            <dt className="text-muted-foreground">
              {t("remoteTeleop.receipt.torqueOff")}
            </dt>
            <dd><ReceiptValue value={receipt?.torque_off_confirmed} /></dd>
          </div>
          <div>
            <dt className="text-muted-foreground">
              {t("remoteTeleop.receipt.closed")}
            </dt>
            <dd><ReceiptValue value={receipt?.close_completed} /></dd>
          </div>
        </dl>
        {(receipt?.verification || receipt?.fault) && (
          <p className="mt-2 break-words font-mono text-[11px] text-muted-foreground">
            {[receipt.verification, receipt.fault].filter(Boolean).join(" · ")}
          </p>
        )}
      </div>

      <div
        className={cn(
          "mt-3 rounded-lg border px-3 py-2 text-xs",
          view.faults.length > 0
            ? "border-destructive/40 bg-destructive/10 text-destructive"
            : "border-border text-muted-foreground",
        )}
      >
        {view.faults.length > 0
          ? view.faults.join(" · ")
          : status
            ? t("remoteTeleop.status.noReportedFaults")
            : t("remoteTeleop.status.notAvailable")}
      </div>
    </section>
  );
}
