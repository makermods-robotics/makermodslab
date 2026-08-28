import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Monitor, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { forgetDevice, updatePresenceSettings } from "@/lib/jobsApi";
import RemoteRunCard from "./RemoteRunCard";
import { useJobsData } from "./JobsDataContext";

/**
 * Local runs happening on the user's OTHER devices, plus this device's own
 * sharing switch.
 *
 * Why this is a separate section rather than rows in the run dropdown: every
 * entry in that dropdown is actionable — stop it, resume it, open it — and
 * these are not. Mixing them in would put un-actionable rows in a list whose
 * whole shape promises actions. A distinct group with its own heading says
 * "these are elsewhere" before the user reaches for a button that isn't there.
 *
 * `Forget this device` lives on the GROUP header, not on a run card. It acts on
 * the presence board, which is something this machine genuinely can do —
 * unlike anything on the run itself.
 */
const RemoteDevicesSection: React.FC = () => {
  const { t } = useTranslation();
  const { presence, jobs, refresh } = useJobsData();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [busy, setBusy] = useState(false);

  const onToggle = useCallback(
    async (enabled: boolean) => {
      setBusy(true);
      try {
        await updatePresenceSettings(baseUrl, fetchWithHeaders, { enabled });
        await refresh();
      } catch (e) {
        toast({
          title: e instanceof Error ? e.message : String(e),
          variant: "destructive",
        });
      } finally {
        setBusy(false);
      }
    },
    [baseUrl, fetchWithHeaders, refresh, toast],
  );

  const onForget = useCallback(
    async (deviceId: string) => {
      try {
        await forgetDevice(baseUrl, fetchWithHeaders, deviceId);
        toast({ title: t("jobs.remote.forgotten") });
        await refresh();
      } catch (e) {
        toast({
          title: e instanceof Error ? e.message : String(e),
          variant: "destructive",
        });
      }
    },
    [baseUrl, fetchWithHeaders, refresh, toast, t],
  );

  // R11: default-on publishing must never be silent. The moment this device
  // has actually written to the board, say so once — naming the repo that now
  // exists on the user's account — then record that it was said. The backend
  // deliberately does not self-mark this: doing so marked the notice delivered
  // when nothing had been shown, so it could never fire.
  const announcedRef = useRef(false);
  useEffect(() => {
    if (!presence.published || presence.announced || announcedRef.current) return;
    if (!presence.repo_id) return;
    announcedRef.current = true;
    toast({
      // The repo id is DATA, interpolated verbatim.
      title: t("jobs.remote.announced", { repo: presence.repo_id ?? "" }),
      description: t("jobs.remote.sharingHint"),
    });
    updatePresenceSettings(baseUrl, fetchWithHeaders, { announced: true }).catch(
      () => {
        // Failing to record the ack only risks showing the notice again next
        // session; never worth surfacing an error of its own.
      },
    );
  }, [
    presence.published,
    presence.announced,
    presence.repo_id,
    baseUrl,
    fetchWithHeaders,
    toast,
    t,
  ]);

  // A remote run whose id matches one of THIS machine's records is the same run
  // seen twice — possible when `outputs/` is on shared storage, or after a
  // device-id reset left an orphaned file on the board. Drop it: two rows for
  // one run, one of them actionless, is worse than no remote row at all.
  const localIds = useMemo(() => new Set(jobs.map((j) => j.id)), [jobs]);
  const devices = useMemo(
    () =>
      presence.devices
        .filter((d) => d.device_id !== presence.device_id)
        .map((d) => ({
          ...d,
          runs: d.runs.filter((r) => !localIds.has(r.job_id)),
        })),
    [presence.devices, presence.device_id, localIds],
  );

  const disabledNote = presence.disabled_reason
    ? t(`jobs.remote.disabled.${presence.disabled_reason}`, { defaultValue: "" })
    : "";

  // The sharing switch is ALWAYS rendered once the backend has answered, even
  // with no other device on the board. Publishing is on by default, so a user
  // with a single machine must still be able to see that it is happening and
  // turn it off — hiding the control until a second device appears made
  // default-on sharing undiscoverable in the commonest case.
  if (!presence.device_id) return null;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
          <Monitor className="w-3.5 h-3.5" />
          {t("jobs.remote.groupLabel")}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-muted-foreground">
            {t("jobs.remote.sharing")}
          </span>
          <Switch
            checked={presence.enabled && !presence.disabled_reason}
            disabled={busy || !!presence.disabled_reason}
            onCheckedChange={onToggle}
            aria-label={t("jobs.remote.sharing")}
          />
        </div>
      </div>

      {disabledNote ? (
        <p className="text-[11px] text-warn">{disabledNote}</p>
      ) : null}

      {devices.length === 0 ? (
        <p className="text-[11px] text-muted-foreground">
          {t("jobs.remote.noneYet")}
        </p>
      ) : null}

      {devices.map((device) => (
        <div key={device.device_id} className="space-y-1.5">
          <div className="flex items-center justify-between gap-2">
            <span className="truncate text-[11px] font-medium text-muted-foreground">
              {device.device_label}
            </span>
            {/* Only offered once a device has gone quiet for good — forgetting
                a machine that is actively reporting would just be undone by its
                next publish. */}
            {device.liveness === "presumed_stopped" ? (
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 text-muted-foreground hover:text-destructive"
                title={t("jobs.remote.forgetTitle")}
                aria-label={t("jobs.remote.forget")}
                onClick={() => onForget(device.device_id)}
              >
                <Trash2 className="w-3 h-3" />
              </Button>
            ) : null}
          </div>
          {device.runs.map((run) => (
            <RemoteRunCard key={run.job_id} device={device} run={run} />
          ))}
        </div>
      ))}
    </div>
  );
};

export default RemoteDevicesSection;
