import React from "react";
import { useTranslation } from "react-i18next";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import MetaRows from "@/components/library/MetaRows";
import { HubJob, isHubJobActive } from "@/lib/jobsApi";
import {
  ExternalLink,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Trash2,
  XCircle,
  Clock,
  HelpCircle,
} from "lucide-react";

interface Props {
  job: HubJob;
  // Hide this job from the list (persisted backend-side; the Hub record is
  // untouched). The trash button is only offered on terminal stages — an
  // active run can't be dismissed out of sight.
  onDismiss?: (id: string) => void;
}

function relativeTime(iso: string | null): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "—";
  const diff = Math.max(0, (Date.now() - t) / 1000);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

/**
 * Hub stage → presentation, keyed by the Hub's own stage string (data, never
 * translated). `labelKey` is a translation KEY, not a word: this map is built
 * at import time, so a resolved label here would freeze the first language
 * loaded instead of following a language switch.
 */
const stagePresentation = {
  RUNNING: {
    labelKey: "jobs.stage.running",
    color: "text-ok",
    Icon: Loader2,
    spin: true,
  },
  QUEUED: { labelKey: "jobs.stage.queued", color: "text-warn", Icon: Clock },
  SCHEDULING: {
    labelKey: "jobs.stage.scheduling",
    color: "text-warn",
    Icon: Clock,
  },
  COMPLETED: {
    labelKey: "jobs.stage.completed",
    color: "text-muted-foreground",
    Icon: CheckCircle2,
  },
  FAILED: {
    labelKey: "jobs.stage.failed",
    color: "text-destructive",
    Icon: XCircle,
  },
  // HF API uses "CANCELED" (single L); accept both spellings.
  CANCELED: {
    labelKey: "jobs.stage.cancelled",
    color: "text-warn",
    Icon: AlertTriangle,
  },
  CANCELLED: {
    labelKey: "jobs.stage.cancelled",
    color: "text-warn",
    Icon: AlertTriangle,
  },
} as const;

const HubJobCard: React.FC<Props> = ({ job, onDismiss }) => {
  const { t } = useTranslation();
  const stage = job.status?.stage?.toUpperCase() ?? "";
  const present = stagePresentation[stage] ?? {
    labelKey: null,
    color: "text-muted-foreground",
    Icon: HelpCircle,
    spin: false,
  };
  const Icon = present.Icon;
  // A stage this bundle has no word for keeps the RAW Hub stage string — it is
  // data, and showing it beats showing nothing; only the last-resort word is
  // translated.
  const stageLabel = present.labelKey
    ? t(present.labelKey)
    : stage || t("jobs.stage.unknown");
  // The run name when the Hub could give us one. The image name is the last
  // resort: every cloud run uses the same image, so titling by it makes every
  // untracked job on the account read as "huggingface/lerobot-gpu:latest".
  // All three are data; only the final fallback is a translated sentence.
  const title =
    job.name ??
    job.docker_image ??
    job.space_id ??
    t("jobs.hubJob.fallbackTitle", { id: job.id.slice(0, 12) });

  // Unified metadata rows (same format as the dataset/job/model cards). The
  // run-identity rows lead, in JobCard's order, so a foreign run's card reads
  // like a tracked one; each is omitted when the job's argv didn't answer it
  // (a resumed run names a config_path, not a policy or dataset). Only the
  // LABELS are translated — the values are data (policy type, repo id) or a
  // pre-formatted number.
  const metaRows: Array<[string, string]> = [];
  if (job.policy_type) metaRows.push([t("jobs.meta.policy"), job.policy_type]);
  if (job.dataset) metaRows.push([t("jobs.meta.dataset"), job.dataset]);
  if (job.total_steps)
    metaRows.push([t("jobs.meta.steps"), job.total_steps.toLocaleString()]);
  metaRows.push([t("jobs.meta.flavor"), job.flavor ?? "—"]);
  metaRows.push([t("jobs.meta.created"), relativeTime(job.created_at)]);
  if (job.owner) metaRows.push([t("jobs.meta.owner"), job.owner]);
  // Only worth a row once it isn't the title; keeps the image visible for the
  // "which image did this run on" question without spending a row twice.
  if (job.name && job.docker_image)
    metaRows.push([t("jobs.meta.image"), job.docker_image]);

  return (
    <Card
      onClick={() => window.open(job.url, "_blank", "noopener,noreferrer")}
      className="bg-card border-border rounded-md cursor-pointer hover:border-ring/50 hover:bg-muted/40 transition-colors h-full"
    >
      <CardContent className="flex h-full flex-col gap-2.5 p-3">
        <div className="flex items-start justify-between gap-2">
          <div
            className={`flex items-center gap-1.5 text-xs font-semibold ${present.color}`}
          >
            <Icon
              className={`w-3.5 h-3.5 ${present.spin ? "animate-spin" : ""}`}
            />
            {stageLabel}
          </div>
          <div className="flex items-center gap-0.5">
            <Button
              variant="ghost"
              size="icon"
              asChild
              className="h-7 w-7 text-muted-foreground hover:text-foreground"
              aria-label={t("jobs.actions.viewOnHub")}
            >
              <a
                href={job.url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
              >
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </Button>
            {onDismiss && !isHubJobActive(job) ? (
              <Button
                variant="ghost"
                size="icon"
                onClick={(e) => {
                  e.stopPropagation();
                  // Left in English on purpose: a native confirm's OK/Cancel
                  // come from the BROWSER's locale, so a translated question
                  // above English buttons reads worse than an English one.
                  // Converting it to an AlertDialog is a separate UX change.
                  if (
                    window.confirm(
                      "Remove this job from the list? The job record on Hugging Face is unaffected.",
                    )
                  )
                    onDismiss(job.id);
                }}
                className="h-7 w-7 text-muted-foreground hover:text-destructive"
                aria-label={t("jobs.hubJob.removeAria")}
                title={t("jobs.hubJob.removeTitle")}
              >
                <Trash2 className="w-3.5 h-3.5" />
              </Button>
            ) : null}
          </div>
        </div>
        <div>
          <div className="text-foreground font-semibold truncate" title={title}>
            {title}
          </div>
        </div>
        <MetaRows rows={metaRows} />
        {job.status?.message ? (
          <div
            className="text-xs text-muted-foreground truncate"
            title={job.status.message}
          >
            {job.status.message}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
};

export default HubJobCard;
