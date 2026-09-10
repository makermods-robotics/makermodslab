import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface EpisodeTaskPromptProps {
  /** 1-based index of the episode just recorded. */
  episode: number;
  /** What the box starts filled with — the previous episode's task, or the
   * dataset-level task before the first one is named. */
  defaultTask: string;
  /** A submission is in flight; the button is locked until it resolves. */
  submitting: boolean;
  onSubmit: (task: string) => void;
}

/**
 * The blocking card shown between an episode's recording phase and the reset
 * gap when the session records a per-episode task (record.py's "naming"
 * phase). There is no bypass — the session only advances once a non-empty task
 * is submitted — so this renders in place of the whole live HUD, not beside
 * it. Mount it keyed by episode so each episode starts from its own prefill.
 */
const EpisodeTaskPrompt: React.FC<EpisodeTaskPromptProps> = ({
  episode,
  defaultTask,
  submitting,
  onSubmit,
}) => {
  const { t } = useTranslation();
  const [value, setValue] = useState(defaultTask);
  const canSubmit = value.trim().length > 0 && !submitting;

  const submit = () => {
    if (!canSubmit) return;
    onSubmit(value.trim());
  };

  return (
    <div className="flex flex-col items-center gap-4 py-6 text-center">
      <div>
        <h3 className="text-lg font-semibold">
          {t("recording.session.naming.title", { index: episode })}
        </h3>
        <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">
          {t("recording.session.naming.description", { index: episode })}
        </p>
      </div>
      <div className="w-full max-w-md space-y-2 text-left">
        <Label htmlFor="episodeTask">
          {t("recording.session.naming.inputLabel", { index: episode })}
        </Label>
        <Input
          id="episodeTask"
          autoFocus
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              submit();
            }
          }}
          placeholder={t("recording.session.naming.placeholder")}
        />
      </div>
      <Button
        onClick={submit}
        disabled={!canSubmit}
        className="w-full max-w-md font-semibold"
      >
        {t("recording.session.button.saveEpisodeTask")}
      </Button>
    </div>
  );
};

export default EpisodeTaskPrompt;
