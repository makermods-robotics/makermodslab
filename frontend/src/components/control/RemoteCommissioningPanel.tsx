import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Loader2, ShieldCheck, TriangleAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import {
  confirmedRemotePhysicalSafeguards,
  remoteCommissioningAction,
  type ConfirmedRemotePhysicalSafeguards,
  type RemotePhysicalSafeguardChecks,
  type RemoteTeleoperationStatus,
} from "@/lib/remoteTeleoperationApi";

const blankChecks = (): RemotePhysicalSafeguardChecks => ({
  arm_secured: false,
  workspace_clear: false,
  physical_power_cutoff_reachable: false,
  acknowledge_live_torque_enable_risk: false,
});

const CHECKS = [
  "arm_secured",
  "workspace_clear",
  "physical_power_cutoff_reachable",
  "acknowledge_live_torque_enable_risk",
] as const;

interface RemoteCommissioningPanelProps {
  status: RemoteTeleoperationStatus | null;
  busyAction: "commission" | "recover" | null;
  onCommission: (
    safeguards: ConfirmedRemotePhysicalSafeguards,
  ) => Promise<void>;
  onRecover: (
    safeguards: ConfirmedRemotePhysicalSafeguards,
  ) => Promise<void>;
}

export default function RemoteCommissioningPanel({
  status,
  busyAction,
  onCommission,
  onRecover,
}: RemoteCommissioningPanelProps) {
  const { t } = useTranslation();
  const [checks, setChecks] = useState(blankChecks);
  const safeguards = confirmedRemotePhysicalSafeguards(checks);
  const action = remoteCommissioningAction(status);
  const commissioning = status?.commissioning;
  const durableFault = status?.durable_fault;
  const registry = status?.hardware_registry;
  const profileDigest =
    commissioning?.record?.profile_digest ??
    durableFault?.record?.profile_digest ??
    null;

  const run = async () => {
    if (!safeguards || !action || busyAction) return;
    try {
      if (action === "recover") await onRecover(safeguards);
      else await onCommission(safeguards);
    } finally {
      // Physical conditions must be re-attested for every probe, including a
      // retry after a failed or interrupted commissioning attempt.
      setChecks(blankChecks());
    }
  };

  return (
    <section className="rounded-xl border border-border bg-card p-4 shadow-1">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-base font-semibold">
            <ShieldCheck className="h-4 w-4" />
            {t("remoteTeleop.commissioning.heading")}
          </h2>
          <p className="mt-1 text-xs text-muted-foreground">
            {t("remoteTeleop.commissioning.description")}
          </p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <Badge variant={commissioning?.commissioned ? "secondary" : "outline"}>
            {commissioning?.commissioned
              ? t("remoteTeleop.commissioning.commissioned")
              : t("remoteTeleop.commissioning.notCommissioned")}
          </Badge>
          {durableFault?.fault_lockout && (
            <Badge variant="destructive">
              {t("remoteTeleop.commissioning.faultLocked")}
            </Badge>
          )}
        </div>
      </div>

      <dl className="mt-3 grid gap-2 text-xs sm:grid-cols-2">
        <div className="rounded-lg border border-border bg-background/60 p-2.5">
          <dt className="text-muted-foreground">
            {t("remoteTeleop.commissioning.profileDigest")}
          </dt>
          <dd className="mt-1 break-all font-mono">
            {profileDigest ?? "—"}
          </dd>
        </div>
        <div className="rounded-lg border border-border bg-background/60 p-2.5">
          <dt className="text-muted-foreground">
            {t("remoteTeleop.commissioning.hardwareRegistry")}
          </dt>
          <dd className="mt-1 font-medium">
            {registry
              ? `${registry.state}${registry.owner ? ` · ${registry.owner}` : ""}`
              : "—"}
          </dd>
          {registry?.pending_unresolved && (
            <dd className="mt-1 break-words text-destructive">
              {t("remoteTeleop.commissioning.pendingLatch")}
              {registry.pending_kind ? ` · ${registry.pending_kind}` : ""}
              {registry.pending_owner ? ` · ${registry.pending_owner}` : ""}
            </dd>
          )}
        </div>
      </dl>

      {(commissioning?.error || durableFault?.error) && (
        <div className="mt-3 flex gap-2 rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
          <TriangleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          {commissioning?.error ?? durableFault?.error}
        </div>
      )}

      {durableFault?.fault_lockout && durableFault.record && (
        <div className="mt-3 rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-xs text-destructive">
          <p className="font-medium">
            {durableFault.record.reason_code ??
              t("remoteTeleop.commissioning.faultLocked")}
          </p>
          {durableFault.record.fault_codes?.length ? (
            <p className="mt-1 break-words font-mono text-[11px]">
              {durableFault.record.fault_codes.join(" · ")}
            </p>
          ) : null}
        </div>
      )}

      {action ? (
        <div className="mt-4 rounded-lg border border-warn/40 bg-warn/5 p-3">
          <p className="text-xs font-medium">
            {t("remoteTeleop.commissioning.confirmEveryCheck")}
          </p>
          <div className="mt-3 space-y-3">
            {CHECKS.map((check) => {
              const id = `remote-${check.replace(/_/g, "-")}`;
              return (
                <div key={check} className="flex items-start gap-2.5">
                  <Checkbox
                    id={id}
                    checked={checks[check]}
                    disabled={busyAction !== null}
                    onCheckedChange={(checked) =>
                      setChecks((current) => ({
                        ...current,
                        [check]: checked === true,
                      }))
                    }
                  />
                  <Label htmlFor={id} className="text-xs leading-relaxed">
                    {t(`remoteTeleop.commissioning.checks.${check}`)}
                  </Label>
                </div>
              );
            })}
          </div>
          <Button
            type="button"
            variant={action === "recover" ? "destructive" : "default"}
            className="mt-4 w-full sm:w-auto"
            disabled={!safeguards || busyAction !== null}
            onClick={() => void run()}
          >
            {busyAction ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {t("remoteTeleop.commissioning.running")}
              </>
            ) : action === "recover" ? (
              t("remoteTeleop.commissioning.recover")
            ) : commissioning?.commissioned ? (
              t("remoteTeleop.commissioning.recommission")
            ) : (
              t("remoteTeleop.commissioning.commission")
            )}
          </Button>
        </div>
      ) : status?.runtime_enabled ? (
        <p className="mt-3 text-xs text-muted-foreground">
          {t("remoteTeleop.commissioning.disableFirst")}
        </p>
      ) : registry?.held || registry?.pending_unresolved ? (
        <p className="mt-3 text-xs text-destructive">
          {t("remoteTeleop.commissioning.registryHeld")}
        </p>
      ) : (
        <p className="mt-3 text-xs text-muted-foreground">
          {t("remoteTeleop.commissioning.configureFirst")}
        </p>
      )}
    </section>
  );
}
