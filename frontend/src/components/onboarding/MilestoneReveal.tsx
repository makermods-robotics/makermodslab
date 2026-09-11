import React from "react";
import { useTranslation } from "react-i18next";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";

export interface MilestoneRevealProps {
  /** Already-resolved copy: every caller (TrainPanel, DeployPanel,
   * CollectHandoff) owns the milestone's wording and passes `t(...)` in.
   * Keeping plain strings rather than keys means this component never has to
   * know which namespace a milestone lives in. */
  title: string;
  description: string;
  onDismiss: () => void;
}

/**
 * Inline "here's what you can do next" card shown once after a first-time
 * milestone (first recording, first Hub upload, first training job, first
 * deploy). Not spotlight-anchored — some of these fire when the next step's
 * UI isn't even open yet (e.g. pointing at Train from the recording banner).
 */
const MilestoneReveal: React.FC<MilestoneRevealProps> = ({
  title,
  description,
  onDismiss,
}) => {
  const { t } = useTranslation();
  return (
    <div
      role="status"
      aria-live="polite"
      className="w-full rounded-lg border border-ring/40 bg-ring/5 p-4"
    >
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <p className="font-medium text-foreground">{title}</p>
          <p className="mt-0.5 text-sm text-muted-foreground">{description}</p>
        </div>
        <Button
          variant="ghost"
          size="icon"
          aria-label={t("studio.common.dismiss")}
          onClick={onDismiss}
          className="h-7 w-7 shrink-0 text-muted-foreground"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};

export default MilestoneReveal;
