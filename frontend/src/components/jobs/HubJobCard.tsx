import React from "react";
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

interface StagePresentation {
  label: string;
  color: string;
  Icon: React.ComponentType<{ className?: string }>;
  spin?: boolean;
}

const stagePresentation: Record<string, StagePresentation> = {
  RUNNING: { label: "Running", color: "text-ok", Icon: Loader2, spin: true },
  QUEUED: { label: "Queued", color: "text-warn", Icon: Clock },
  SCHEDULING: { label: "Scheduling", color: "text-warn", Icon: Clock },
  COMPLETED: { label: "Done", color: "text-muted-foreground", Icon: CheckCircle2 },
  FAILED: { label: "Failed", color: "text-destructive", Icon: XCircle },
  // HF API uses "CANCELED" (single L); accept both spellings.
  CANCELED: { label: "Cancelled", color: "text-warn", Icon: AlertTriangle },
  CANCELLED: { label: "Cancelled", color: "text-warn", Icon: AlertTriangle },
};

const HubJobCard: React.FC<Props> = ({ job, onDismiss }) => {
  const stage = job.status?.stage?.toUpperCase() ?? "";
  const present: StagePresentation = stagePresentation[stage] ?? {
    label: stage || "Unknown",
    color: "text-muted-foreground",
    Icon: HelpCircle,
  };
  const Icon = present.Icon;
  // The run name when the Hub could give us one. The image name is the last
  // resort: every cloud run uses the same image, so titling by it makes every
  // untracked job on the account read as "huggingface/lerobot-gpu:latest".
  const title =
    job.name ?? job.docker_image ?? job.space_id ?? `Job ${job.id.slice(0, 12)}…`;

  // Unified metadata rows (same format as the dataset/job/model cards).
  const metaRows: Array<[string, string]> = [
    ["Flavor", job.flavor ?? "—"],
    ["Created", relativeTime(job.created_at)],
  ];
  if (job.owner) metaRows.push(["Owner", job.owner]);
  // Only worth a row once it isn't the title; keeps the image visible for the
  // "which image did this run on" question without spending a row twice.
  if (job.name && job.docker_image) metaRows.push(["Image", job.docker_image]);

  return (
    <Card
      onClick={() => window.open(job.url, "_blank", "noopener,noreferrer")}
      className="bg-card border-border rounded-md cursor-pointer hover:border-ring/50 hover:bg-muted/40 transition-colors h-full"
    >
      <CardContent className="flex h-full flex-col gap-2.5 p-3">
        <div className="flex items-start justify-between gap-2">
          <div className={`flex items-center gap-1.5 text-xs font-semibold ${present.color}`}>
            <Icon className={`w-3.5 h-3.5 ${present.spin ? "animate-spin" : ""}`} />
            {present.label}
          </div>
          <div className="flex items-center gap-0.5">
            <Button
              variant="ghost"
              size="icon"
              asChild
              className="h-7 w-7 text-muted-foreground hover:text-foreground"
              aria-label="View on Hub"
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
                  if (
                    window.confirm(
                      "Remove this job from the list? The job record on Hugging Face is unaffected.",
                    )
                  )
                    onDismiss(job.id);
                }}
                className="h-7 w-7 text-muted-foreground hover:text-destructive"
                aria-label="Remove job from list"
                title="Remove from list"
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
          <div className="text-xs text-muted-foreground truncate" title={job.status.message}>
            {job.status.message}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
};

export default HubJobCard;
