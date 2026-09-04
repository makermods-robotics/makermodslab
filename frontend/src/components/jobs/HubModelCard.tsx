import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import MetaRows from "@/components/library/MetaRows";
import { middleEllipsis } from "@/lib/modelNames";
import { HubModel } from "@/lib/jobsApi";
import { useTruncationTitle } from "@/hooks/useTruncationTitle";
import { ExternalLink, Lock, Play, Sparkles, Upload } from "lucide-react";

interface Props {
  model: HubModel;
  /**
   * Run inference / Fine-tune on this untracked Hub repo. The parent lazily
   * auto-imports the repo (registering it as a tracked imported model), then
   * proceeds exactly as it would for an imported-model card — so this card's
   * primary actions match a regular model card without duplicating any flow.
   */
  onAction?: (
    repoId: string,
    action: "inference" | "finetune",
  ) => void | Promise<void>;
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

const HubModelCard: React.FC<Props> = ({ model, onAction }) => {
  const { t } = useTranslation();
  const [acting, setActing] = useState<"inference" | "finetune" | null>(null);
  const url = `https://huggingface.co/${model.repo_id}`;
  // Same title rule as the imported card next to it in the grid: namespace off
  // (the subtitle below repeats the full repo id), and shortened from the
  // middle rather than the end, since an uploaded repo's tail is its timestamp
  // — the only thing separating two uploads of the same task.
  const shortName = middleEllipsis(
    model.repo_id.includes("/")
      ? model.repo_id.split("/").slice(1).join("/")
      : model.repo_id,
  );
  // Both of this title's shortenings are the caller's own — the namespace peel
  // and middleEllipsis — so the flag is just "is what we render the whole repo
  // id?"; the div's `truncate` on top of that is measured on hover. An
  // unnamespaced repo whose name fits is therefore the one case with no title,
  // and correctly so: the text on screen already IS the repo id.
  const nameHover = useTruncationTitle(
    model.repo_id,
    shortName !== model.repo_id,
  );

  const runAction = async (
    e: React.MouseEvent,
    action: "inference" | "finetune",
  ) => {
    e.stopPropagation();
    if (!onAction || acting) return;
    setActing(action);
    try {
      await onAction(model.repo_id, action);
    } finally {
      setActing(null);
    }
  };

  return (
    <Card
      onClick={() => window.open(url, "_blank", "noopener,noreferrer")}
      className="@container bg-card border-border rounded-md cursor-pointer hover:border-ring/50 hover:bg-muted/40 transition-colors h-full"
    >
      <CardContent className="flex h-full flex-col gap-2.5 p-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-info">
            <Upload className="w-3.5 h-3.5" />
            {t("jobs.hubModelCard.uploaded")}
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
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
              >
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </Button>
            {/* Hub-repo deletion was removed from the UI — the route remains
                for a future management surface; for now repos are deleted on
                huggingface.co itself. */}
          </div>
        </div>
        <div>
          <div
            className="text-foreground font-semibold truncate flex items-center gap-1.5"
            {...nameHover}
          >
            {model.private ? (
              <Lock className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
            ) : null}
            <span className="truncate">{shortName}</span>
          </div>
          <div className="text-xs text-muted-foreground truncate" title={model.repo_id}>
            {model.repo_id}
          </div>
        </div>
        <MetaRows
          rows={[[t("jobs.meta.updated"), relativeTime(model.last_modified)]]}
        />
        {/* Same primary actions as an imported model card. Clicking either
            lazily auto-imports the repo (in the parent) and then runs the
            action — so a model trained on another machine is a first-class
            citizen here. */}
        {onAction ? (
          <div className="mt-auto flex items-center gap-1.5 pt-1">
            <Button
              size="sm"
              onClick={(e) => runAction(e, "inference")}
              disabled={acting !== null}
              className="h-8 gap-1 bg-primary hover:bg-primary/90 text-primary-foreground"
              aria-label={t("jobs.actions.runInferenceModel")}
              title={t("jobs.actions.runInference")}
            >
              <Play className="w-3.5 h-3.5" /> {t("jobs.actions.run")}
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={(e) => runAction(e, "finetune")}
              disabled={acting !== null}
              className="h-8 shrink-0 gap-1 border-primary/40 text-primary hover:bg-primary/10"
              // Same words on both, so one key rather than two that could
              // drift apart.
              aria-label={t("jobs.actions.fineTuneHint")}
              title={t("jobs.actions.fineTuneHint")}
            >
              <Sparkles className="w-3.5 h-3.5" />
              {/* Label only when the card fits the whole row on one line;
                  the tooltip covers the narrow case. */}
              <span className="hidden @[13rem]:inline">
                {t("jobs.actions.fineTune")}
              </span>
            </Button>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
};

export default HubModelCard;
