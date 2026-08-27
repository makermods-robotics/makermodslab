import React from "react";
import { useTranslation } from "react-i18next";
import { Card, CardContent } from "@/components/ui/card";
import MetaRows from "@/components/library/MetaRows";
import { RemoteDevice, RemoteRun } from "@/lib/jobsApi";
import {
  CheckCircle2,
  HelpCircle,
  Loader2,
  Monitor,
  XCircle,
} from "lucide-react";

interface Props {
  device: RemoteDevice;
  run: RemoteRun;
}

/**
 * One local training run on ANOTHER of the user's devices.
 *
 * Deliberately thinner than JobCard, and deliberately **actionless**. There is
 * no control channel to the other machine and its checkpoints are on its own
 * disk, so stop / resume / download / run cannot work from here. The card
 * therefore carries no buttons at all rather than disabled ones: a greyed-out
 * Stop implies stopping is a thing that could happen here, and it is not. The
 * honesty is in the absence, with one line of prose to say where to go instead.
 *
 * The other rule this card exists to keep: a device we have stopped believing
 * NEVER renders as running. A machine unplugged mid-run never wrote a goodbye,
 * so its last published payload says "running" forever — the backend rewrites
 * those to "unknown" (see project_device), and the presentation below has no
 * spinner and no progress bar for that state.
 */

/** Run state -> presentation. `labelKey` is a translation KEY, not a word:
 * this map is built at import time, so a resolved label here would freeze
 * whichever language loaded first instead of following a language switch. The
 * keys are spelled out literally because the i18n catalog is strictly typed and
 * a template-literal key would not resolve. */
const statePresentation = {
  running: {
    labelKey: "jobs.remote.state.running",
    color: "text-ok",
    Icon: Loader2,
    spin: true,
  },
  done: {
    labelKey: "jobs.remote.state.done",
    color: "text-muted-foreground",
    Icon: CheckCircle2,
    spin: false,
  },
  failed: {
    labelKey: "jobs.remote.state.failed",
    color: "text-destructive",
    Icon: XCircle,
    spin: false,
  },
  interrupted: {
    labelKey: "jobs.remote.state.interrupted",
    color: "text-warn",
    Icon: XCircle,
    spin: false,
  },
  unknown: {
    labelKey: "jobs.remote.state.unknown",
    color: "text-muted-foreground",
    Icon: HelpCircle,
    spin: false,
  },
} as const;

function relativeTime(epochSec: number | null): string {
  if (epochSec == null) return "—";
  const diff = Math.max(0, Date.now() / 1000 - epochSec);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

const RemoteRunCard: React.FC<Props> = ({ device, run }) => {
  const { t } = useTranslation();
  const present = statePresentation[run.state] ?? statePresentation.unknown;
  const Icon = present.Icon;
  // Only a run we still believe is running gets a live treatment.
  const isLive = run.state === "running" && device.liveness === "live";
  const pct =
    run.total_steps > 0
      ? Math.min(100, (run.current_step / run.total_steps) * 100)
      : 0;

  // Values are data — run names, dataset ids, policy types, step counts — and
  // are never translated. Only the labels beside them are.
  const metaRows: Array<[string, string]> = [];
  if (run.dataset_repo_id)
    metaRows.push([t("jobs.meta.dataset"), run.dataset_repo_id]);
  if (run.policy_type) metaRows.push([t("jobs.meta.policy"), run.policy_type]);
  if (run.total_steps > 0)
    metaRows.push([
      t("jobs.meta.steps"),
      `${run.current_step.toLocaleString()} / ${run.total_steps.toLocaleString()}`,
    ]);

  // The device's freshness line. `live` says nothing — freshness is only worth
  // words once it is in doubt.
  const note =
    device.liveness === "live"
      ? null
      : device.liveness === "unknown"
        ? t("jobs.remote.lastSeen", { when: relativeTime(device.last_seen) })
        : t("jobs.remote.presumedStopped");

  return (
    <Card className="bg-card border-border rounded-md h-full">
      <CardContent className="flex h-full flex-col gap-2.5 p-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2">
            <div
              className={`flex items-center gap-1.5 text-xs font-semibold ${present.color}`}
            >
              <Icon
                className={`w-3.5 h-3.5 ${isLive && present.spin ? "animate-spin" : ""}`}
              />
              {t(present.labelKey)}
            </div>
            <div
              className="flex min-w-0 items-center gap-1 text-[11px] font-medium text-muted-foreground"
              title={device.device_label}
            >
              <Monitor className="w-3 h-3 shrink-0" />
              <span className="truncate">{device.device_label}</span>
            </div>
          </div>
        </div>

        <div className="text-foreground font-semibold truncate">
          {run.display_name || run.name || run.job_id}
        </div>

        <MetaRows rows={metaRows} />

        {/* A progress bar is a claim that something is moving. Only a run we
            still believe is running gets one. */}
        {isLive && run.total_steps > 0 ? (
          <div className="relative h-5 w-full overflow-hidden rounded-md bg-muted border border-border">
            <div
              className="h-full bg-info transition-[width] duration-500"
              style={{ width: `${pct}%` }}
            />
            <div className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-white tabular-nums drop-shadow">
              {pct.toFixed(1)}%
            </div>
          </div>
        ) : null}

        {note ? (
          <div className="text-[11px] text-warn">{note}</div>
        ) : null}

        {/* The whole affordance story, in one line. There are no buttons on
            this card and this says why, so their absence reads as deliberate
            rather than as something still loading. */}
        <div className="mt-auto pt-1 text-[11px] text-muted-foreground">
          {t("jobs.remote.manageThere", { device: device.device_label })}
        </div>
      </CardContent>
    </Card>
  );
};

export default RemoteRunCard;
