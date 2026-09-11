import React from "react";
import { useTranslation } from "react-i18next";
import { GitBranch, Sparkles } from "lucide-react";
import { RunKind } from "@/lib/jobsApi";

/**
 * The "what is this run" chip, shared by the local job card and the Hub job
 * card so the two can't drift apart.
 *
 * Rendered in the card's HEADER row, beside the state badge — the same slot
 * that already carries JobCard's Local/Cloud chip, which is the established
 * place a card says what kind of thing it is.
 *
 * Two of the four kinds render NOTHING, deliberately:
 *   * `scratch`    — random weights is the unremarkable case.
 *   * `foundation` — starting from lerobot/smolvla_base (or the pi0 family's
 *     equivalents) is the DEFAULT for those policies, not a choice: jobs.py
 *     pins it whenever a VLA run names no starting point. A chip that appears
 *     on every VLA card would carry no information and would read as a claim
 *     the user fine-tuned something when they did not. The Base row still names
 *     the checkpoint, which is the part that IS informative.
 */
const RunKindChip: React.FC<{ kind?: RunKind | null }> = ({ kind }) => {
  const { t } = useTranslation();
  if (kind !== "finetune" && kind !== "resume") return null;
  const Icon = kind === "finetune" ? Sparkles : GitBranch;
  return (
    <div className="flex items-center gap-1 text-[11px] font-medium text-muted-foreground">
      <Icon className="w-3 h-3" />
      {t(`jobs.kind.${kind}`)}
    </div>
  );
};

export default RunKindChip;
