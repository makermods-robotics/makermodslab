import React from "react";
import { Trans, useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import {
  AlertTriangle,
  CheckCircle2,
  Copy,
  Loader2,
  XCircle,
} from "lucide-react";
import type { InstallState, LogEntry } from "@/hooks/useInstallExtra";

interface InstallProgressProps {
  state: InstallState;
  error: string | null;
  logs: LogEntry[];
  logBoxRef: React.RefObject<HTMLDivElement>;
  onInstall: () => void;
  onRetry: () => void;

  installHint: string;
  packageName: string;
  idleTitle: string;
  idleDescription: React.ReactNode;
  doneDescription: React.ReactNode;
}

/** The dialog/card title for an install flow. Takes `t` rather than resolving
 * copy at module scope: every caller is already inside a component. `idleTitle`
 * arrives already translated from the caller, which owns the "what is missing"
 * wording. */
export function installTitle(
  t: TFunction,
  state: InstallState,
  idleTitle: string,
): string {
  switch (state) {
    case "done":
      return t("training.install.titleDone");
    case "error":
      return t("training.install.titleError");
    case "installing":
      return t("training.install.titleInstalling");
    default:
      return idleTitle;
  }
}

export function InstallTitleIcon({ state }: { state: InstallState }) {
  if (state === "done") return <CheckCircle2 className="w-6 h-6 text-ok" />;
  if (state === "error") return <XCircle className="w-6 h-6 text-destructive" />;
  if (state === "installing")
    return <Loader2 className="w-6 h-6 text-info animate-spin" />;
  return <AlertTriangle className="w-6 h-6 text-warn" />;
}

export const InstallProgress: React.FC<InstallProgressProps> = ({
  state,
  error,
  logs,
  logBoxRef,
  onInstall,
  onRetry,
  installHint,
  packageName,
  idleDescription,
  doneDescription,
}) => {
  const { toast } = useToast();
  const { t } = useTranslation();

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(installHint);
      // `installHint` is the backend's own command string — shown verbatim.
      toast({ title: t("training.install.copiedTitle"), description: installHint });
    } catch {
      toast({
        title: t("training.install.copyFailedTitle"),
        description: t("training.install.copyFailedDescription"),
        variant: "destructive",
      });
    }
  };

  return (
    <>
      {state === "idle" && (
        <>
          <p className="text-muted-foreground">{idleDescription}</p>
          <div className="flex items-center gap-2">
            <code className="flex-1 bg-muted border border-border rounded-lg px-3 py-2 text-sm text-foreground font-mono">
              {installHint}
            </code>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleCopy}
              className="text-muted-foreground hover:text-foreground"
              aria-label={t("training.install.copyAria")}
            >
              <Copy className="w-4 h-4" />
            </Button>
          </div>
          <Button
            onClick={onInstall}
            className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold"
          >
            {t("training.install.installNow")}
          </Button>
        </>
      )}

      {state === "installing" && (
        <p className="text-muted-foreground">
          <Trans
            i18nKey="training.install.installing"
            values={{ packageName }}
            components={[
              <code key="0" className="px-1 py-0.5 rounded bg-muted text-info" />,
            ]}
          />
        </p>
      )}

      {state === "done" && (
        <div className="space-y-3 text-muted-foreground">{doneDescription}</div>
      )}

      {state === "error" && (
        <>
          {/* The backend's own error text when there is one; the local
              fallback is the only part that is ours to translate. */}
          <p className="text-destructive">
            {error || t("training.install.failedFallback")}
          </p>
          <Button
            onClick={onRetry}
            className="bg-secondary hover:bg-secondary/80 text-secondary-foreground"
          >
            {t("training.install.tryAgain")}
          </Button>
        </>
      )}

      {state === "error" && logs.length > 0 && (
        <div
          ref={logBoxRef}
          className="bg-muted rounded-lg p-3 h-48 overflow-y-auto font-mono text-xs border border-border text-muted-foreground whitespace-pre-wrap break-words"
        >
          {logs.map((log, idx) => (
            <div key={idx}>{log.message}</div>
          ))}
        </div>
      )}
    </>
  );
};

// The installed package is only consumed by training subprocesses (fresh
// processes that see the new install immediately), and the backend probes
// availability live per request — so no server restart is needed. Reload the
// page to pick it up.
//
// Takes the finished sentence rather than a noun to slot into one: English
// built "Install complete — {purpose} is available…" from a bare noun phrase
// ("training", "W&B logging", "ACT inference"), which no translation survives.
// Each caller passes its own complete `training.install.ready*` string.
export const ReadyInstructions: React.FC<{ text: string }> = ({ text }) => (
  <p>{text}</p>
);
