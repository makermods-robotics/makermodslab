import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface EpisodeTaskPromptProps {
  /** 1-based index of the episode about to record. */
  episode: number;
  defaultTask: string;
  submitting: boolean;
  onSubmit: (task: string) => void;
  onRerecord?: () => void;
}

/** No countdown: finish editing with Enter, then press Space when ready.
 * Spaces inside the description remain ordinary text input. */
const EpisodeTaskPrompt: React.FC<EpisodeTaskPromptProps> = ({
  episode,
  defaultTask,
  submitting,
  onSubmit,
  onRerecord,
}) => {
  const { t } = useTranslation();
  const [value, setValue] = useState(defaultTask);
  const startButton = useRef<HTMLButtonElement>(null);
  const canSubmit = value.trim().length > 0 && !submitting;

  const submit = () => {
    if (canSubmit) onSubmit(value.trim());
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target instanceof HTMLElement ? event.target : null;
      if (event.repeat || event.isComposing || !canSubmit) return;
      if (target?.matches("input, textarea") || target?.isContentEditable) return;
      // Other buttons retain their own keyboard action, including session exits.
      if (target?.closest("button") && target.closest("button") !== startButton.current) return;
      if (event.key === " ") {
        event.preventDefault();
        event.stopImmediatePropagation();
        onSubmit(value.trim());
      }
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [canSubmit, onSubmit, value]);

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
            if (e.key === "Enter" && !e.nativeEvent.isComposing) {
              e.preventDefault();
              if (canSubmit) startButton.current?.focus();
            }
          }}
          placeholder={t("recording.session.naming.placeholder")}
          aria-describedby="episodeTaskHint"
        />
        <p id="episodeTaskHint" className="text-xs text-muted-foreground">
          {t("recording.session.naming.keyboardHint")}
        </p>
      </div>
      <Button
        ref={startButton}
        onClick={submit}
        disabled={!canSubmit}
        className="w-full max-w-md font-semibold"
      >
        {t("recording.session.button.saveEpisodeTask")}
        <span className="ml-3 text-xs font-mono">SPACE</span>
      </Button>
      {onRerecord && (
        <Button variant="outline" disabled={submitting} onClick={onRerecord}>
          {t("recording.session.naming.rerecordPrevious")}
        </Button>
      )}
    </div>
  );
};

export default EpisodeTaskPrompt;
