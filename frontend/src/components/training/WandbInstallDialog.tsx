import React from "react";
import { Trans, useTranslation } from "react-i18next";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useInstallExtra } from "@/hooks/useInstallExtra";
import {
  InstallProgress,
  InstallTitleIcon,
  ReadyInstructions,
  installTitle,
} from "./InstallProgress";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The backend's own pip command — data, shown verbatim. */
  installHint: string;
}

const WandbInstallDialog: React.FC<Props> = ({ open, onOpenChange, installHint }) => {
  const install = useInstallExtra("system/wandb-extra", open);
  const { t } = useTranslation();
  const idleTitle = t("training.wandbDialog.title");

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-background border-border max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-foreground">
            <InstallTitleIcon state={install.state} />
            {installTitle(t, install.state, idleTitle)}
          </DialogTitle>
          <DialogDescription className="sr-only">
            {t("training.wandbDialog.srDescription")}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <InstallProgress
            state={install.state}
            error={install.error}
            logs={install.logs}
            logBoxRef={install.logBoxRef}
            onInstall={install.handleInstall}
            onRetry={install.handleRetry}
            installHint={installHint}
            packageName="wandb"
            idleTitle={idleTitle}
            idleDescription={
              <Trans
                i18nKey="training.wandbDialog.description"
                components={[
                  <code
                    key="0"
                    className="px-1 py-0.5 rounded bg-muted text-info"
                  />,
                ]}
              />
            }
            doneDescription={
              <ReadyInstructions text={t("training.install.readyWandb")} />
            }
          />
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default WandbInstallDialog;
