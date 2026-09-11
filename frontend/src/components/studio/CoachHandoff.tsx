import React, { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Trans, useTranslation } from "react-i18next";
import { GitMerge, GraduationCap, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useStudio } from "@/contexts/StudioContext";

/** Left in router state by the inference session dialog when a coaching
 * session ends with corrections on disk. Mirrors `RecordedInfo`. */
export interface CoachedInfo {
  /** The `rollout_…` dataset the session actually created. */
  repo_id: string;
  corrections_saved: number;
  /** Job registry id of the skill that was coached — the fine-tune base. */
  base_job_id?: string;
  base_name?: string;
  /** What that skill was last trained on: the other half of the merge. */
  training_repo_id?: string;
}

/**
 * Post-coaching handoff banner on the Launchpad.
 *
 * Sibling of `CollectHandoff`, and deliberately the same shape: a session that
 * produced data ends by putting the next step where the operator LANDS, not
 * inside the modal they are about to close.
 *
 * That distinction is the whole reason this exists. The merge + fine-tune
 * offer previously lived in the session dialog's own summary, which meant it
 * died with the dialog — closed it, or reloaded the page, and the only route
 * onward was to reconstruct both dataset names by hand. Corrections are worth
 * nothing until they are merged with what the checkpoint was last trained on
 * and the checkpoint fine-tuned on the result (TrainPanel takes exactly one
 * dataset, so merging is mandatory, not an optimisation) — so the offer has to
 * outlive the session that produced it.
 *
 * Router state rather than a context field, for the same reason CollectHandoff
 * uses it: it survives the dialog unmounting, and `navigate(".", {state: null})`
 * on dismiss means a reload does not resurrect a handoff already dealt with.
 */
const CoachHandoff: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { openStudio } = useStudio();
  const { t } = useTranslation();
  const [dismissed, setDismissed] = useState(false);

  const coached = location.state?.coached as CoachedInfo | undefined;
  if (!coached || dismissed) return null;

  const dismiss = () => {
    setDismissed(true);
    navigate(".", { replace: true, state: null });
  };

  // Both halves have to be known for the one-click path to mean anything. A
  // session launched outside the studio has no lineage, so it falls back to
  // naming the dataset and pointing at the library rather than offering a
  // button that would have to guess which dataset to merge against.
  const canMerge = Boolean(coached.training_repo_id && coached.base_job_id);

  const mergeAndFinetune = () => {
    if (!coached.training_repo_id) return;
    // Named after the TRAINING dataset: the merged result is the next training
    // set, and inheriting the corrections' `rollout_` prefix would claim it
    // came straight off a deployment.
    const base = coached.training_repo_id.split("/").pop() ?? "training";
    setDismissed(true);
    navigate(".", { replace: true, state: null });
    openStudio("collect", {
      merge: {
        sources: [coached.training_repo_id, coached.repo_id],
        suggestedOutput: `${base}_coached`,
        finetuneBaseJobId: coached.base_job_id,
        finetuneBaseName: coached.base_name,
      },
    });
  };

  return (
    <div className="w-full space-y-3">
      <div className="w-full rounded-lg border border-border bg-card p-4 shadow-1">
        <div className="flex items-start gap-3">
          <GraduationCap className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600 dark:text-emerald-400" />
          <div className="min-w-0 flex-1">
            <p className="font-medium text-foreground">
              {/* <0> is the dataset name — data, in the Latin script. */}
              <Trans
                i18nKey="studio.coachHandoff.saved"
                count={coached.corrections_saved}
                values={{ dataset: coached.repo_id }}
                components={[
                  <span key="0" className="break-all font-mono text-foreground" />,
                ]}
              />
            </p>
            <p className="mt-0.5 text-sm text-muted-foreground">
              {canMerge ? (
                <Trans
                  i18nKey="studio.coachHandoff.next"
                  values={{ dataset: coached.training_repo_id ?? "" }}
                  components={[<span key="0" className="break-all font-mono" />]}
                />
              ) : (
                /* <0> emphasises "last" — the mistake this sentence exists to
                   prevent. */
                <Trans
                  i18nKey="studio.coachHandoff.manual"
                  components={[<em key="0" />]}
                />
              )}
            </p>
            {canMerge && (
              <Button onClick={mergeAndFinetune} className="mt-3 gap-2">
                <GitMerge className="h-4 w-4" />
                {t("studio.coachHandoff.action")}
              </Button>
            )}
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={dismiss}
            aria-label={t("studio.common.dismiss")}
            className="h-7 w-7 shrink-0"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
};

export default CoachHandoff;
