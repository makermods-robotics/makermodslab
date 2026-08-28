import React, { useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import {
  ExternalLink,
  GraduationCap,
  Heart,
  Play,
  Sparkles,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useStudio } from "@/contexts/StudioContext";
import { useApi } from "@/contexts/ApiContext";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useLanguage } from "@/contexts/LanguageContext";
import { isCaselessScript } from "@/i18n/config";
import { cn } from "@/lib/utils";
import { useRobots } from "@/hooks/useRobots";
import { policyTypeDisplayName } from "@/components/training/types";
import { ModelInfo, ModelItem, getModelInfo } from "@/lib/modelsApi";
import {
  SkillBadgePill,
  classifySkill,
  formatCount,
  isWipSkillId,
  skillDisplayAuthorLabel,
  skillDisplayTitle,
  skillNamespace,
  skillThumbnail,
} from "@/components/launchpad/SkillCard";

const formatBytes = (bytes: number | null | undefined): string => {
  if (bytes == null) return "";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
};

export interface SkillDetailDialogProps {
  model: ModelItem | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/**
 * Skill detail — the marketplace's connective tissue. Shows badges, author, the
 * real stats the payload carries (policy type, steps, base dataset lineage,
 * size), and the three market actions: Run on the corner robot (→ Deploy panel,
 * prefilled), Fine-tune (→ Train panel, base prefilled), and a Hub link. Likes
 * are display-only text — the API supports neither a count nor a like action, so
 * nothing is fabricated and there is no Like button.
 */
const SkillDetailDialog: React.FC<SkillDetailDialogProps> = ({
  model,
  open,
  onOpenChange,
}) => {
  const { t } = useTranslation();
  const { language } = useLanguage();
  const { openStudio } = useStudio();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { auth } = useHfAuth();
  const { selectedRecord } = useRobots();
  const [info, setInfo] = useState<ModelInfo | null>(null);

  const username = auth.status === "authenticated" ? auth.username : null;
  // The robot's own name is data; only the no-robot-selected fallback is copy.
  const robotName = selectedRecord?.name ?? t("dialogs.skillDetail.robotFallback");

  // Lazily enrich with /models/info while the dialog is open — a hub-only row's
  // list entry has null dataset/steps that model_info can recover (base dataset
  // lineage, size). Best-effort: a failure leaves the list-derived fields.
  // A WIP preview has no repo behind it, so there is nothing to enrich and the
  // lookup would only 404 into the silent catch below.
  useEffect(() => {
    if (!open || !model || isWipSkillId(model.id)) {
      setInfo(null);
      return;
    }
    const controller = new AbortController();
    getModelInfo(baseUrl, fetchWithHeaders, model.id, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setInfo(data);
      })
      .catch(() => {
        // Degrade silently to the list-derived fields.
      });
    return () => controller.abort();
  }, [open, model, baseUrl, fetchWithHeaders]);

  if (!model) return null;

  const isWip = isWipSkillId(model.id);
  const badge = isWip ? "wip" : classifySkill(model, username);
  const title = skillDisplayTitle(t, model);
  const author = skillDisplayAuthorLabel(t, model);
  // "by <org>" reads right only for a real namespace — a bare local run and a
  // WIP preview both carry a standalone label instead.
  const hasAuthor = !isWip && skillNamespace(model) !== null;
  const policyType = info?.policy_type ?? model.policy_type;
  const policy = policyType ? policyTypeDisplayName(policyType) : null;
  const steps = info?.steps ?? model.steps;
  const dataset = info?.dataset ?? model.dataset;
  const sizeBytes = info?.size_bytes ?? null;
  const hubRepoId = model.hf_repo_id ?? info?.hf_repo_id ?? null;

  const stats: string[] = [];
  if (policy) stats.push(policy);
  // `steps` and the byte size arrive pre-formatted — the catalog only supplies
  // the words around them.
  if (steps != null)
    stats.push(t("dialogs.skillDetail.steps", { steps: formatCount(steps) }));
  if (sizeBytes != null) stats.push(formatBytes(sizeBytes));
  if (model.private) stats.push(t("dialogs.skillDetail.private"));

  // Only a Hub-ONLY model needs the repo-id (lazy-import) path. A model with a
  // local copy (`local` or `both`) already has a job registry entry — its run
  // id is the job id — and must deploy/fine-tune through it: importing a
  // second Hub pseudo-job would duplicate the record and break offline runs.
  const hubOnly = model.source === "hub";

  const openDeploy = (mode?: "coach") => {
    const base =
      hubOnly && hubRepoId
        ? ({ source: "hub", id: hubRepoId } as const)
        : ({ source: "job", id: model.id } as const);
    openStudio("deploy", { deploy: { ...base, ...(mode ? { mode } : {}) } });
    onOpenChange(false);
  };

  const handleRun = () => openDeploy();

  // The third place an operator knows a skill is imperfect — the other two are
  // a finished run and an evaluation summary. All three used to offer no route
  // to improving it except "fine-tune", which needs data they do not have yet.
  // Coaching is how they GET that data.
  const handleCoach = () => openDeploy("coach");

  const handleFineTune = () => {
    openStudio("train", {
      train:
        hubOnly && hubRepoId
          ? { baseModelRepoId: hubRepoId }
          : { baseJobId: model.id },
    });
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="font-display tracking-tight">
            {title}
          </DialogTitle>
        </DialogHeader>

        <div className="grid gap-5 sm:grid-cols-[1.2fr_1fr]">
          {skillThumbnail(model) ? (
            <img
              src={skillThumbnail(model)}
              alt={t("dialogs.skillDetail.previewAlt", { title })}
              className="aspect-[4/3] w-full rounded-md object-cover"
            />
          ) : (
            <div
              className="media-slot aspect-[4/3] w-full"
              data-label={t("dialogs.skillDetail.previewPlaceholder")}
            />
          )}
          <div className="flex flex-col">
            <div className="mb-2 flex flex-wrap items-center gap-1.5">
              <SkillBadgePill badge={badge} />
              {model.source === "both" && (
                <span
                  className={cn(
                    "rounded-full border border-border px-2 py-0.5 text-[10px] font-semibold text-muted-foreground",
                    // `uppercase` does nothing to Chinese, but the
                    // letter-spacing that pairs with it does — both come off
                    // together (mirrors SkillBadgePill).
                    isCaselessScript(language) ? "" : "uppercase tracking-[0.06em]",
                  )}
                >
                  {t("dialogs.skillDetail.localAndHub")}
                </span>
              )}
            </div>

            <p className="mb-1 font-mono text-xs text-muted-foreground">
              {hasAuthor ? t("dialogs.skillDetail.byAuthor", { author }) : author}
            </p>
            {stats.length > 0 && (
              <p className="mb-2 text-sm text-muted-foreground">
                {stats.join(" · ")}
              </p>
            )}
            {dataset && (
              <p className="mb-3 text-sm text-muted-foreground">
                <Trans
                  i18nKey="dialogs.skillDetail.trainedOn"
                  values={{ dataset }}
                  components={[
                    <span key="0" className="text-foreground" />,
                    <span key="1" className="font-mono" />,
                  ]}
                />
              </p>
            )}

            <div className="mt-auto flex flex-col gap-2 pt-2">
              {isWip ? (
                <p className="rounded-md border border-warn/40 px-3 py-2 text-sm text-warn">
                  {t("dialogs.skillDetail.notTrained")}
                </p>
              ) : (
                <>
                  <Button onClick={handleRun} className="w-full gap-2">
                    <Play className="h-4 w-4" />
                    {t("dialogs.skillDetail.run", { robot: robotName })}
                  </Button>
                  {/* Between running and fine-tuning, because that is the real
                      order: fine-tuning needs data the operator does not have
                      yet, and coaching is how they get it. */}
                  <Button
                    variant="outline"
                    onClick={handleCoach}
                    className="w-full gap-2"
                  >
                    <GraduationCap className="h-4 w-4" />
                    {t("dialogs.skillDetail.coach")}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={handleFineTune}
                    className="w-full gap-2"
                  >
                    <Sparkles className="h-4 w-4" />
                    {t("dialogs.skillDetail.fineTune")}
                  </Button>
                </>
              )}
              <div className="flex gap-2">
                <span className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-md border border-border px-3 py-2 text-sm text-muted-foreground">
                  <Heart className="h-4 w-4" />
                  {t("dialogs.skillDetail.likesUnavailable")}
                </span>
                {hubRepoId && (
                  <a
                    href={`https://huggingface.co/${hubRepoId}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-md border border-border px-3 py-2 text-sm text-foreground transition-colors hover:bg-accent"
                  >
                    <ExternalLink className="h-4 w-4" />
                    {t("dialogs.skillDetail.viewOnHub")}
                  </a>
                )}
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default SkillDetailDialog;
