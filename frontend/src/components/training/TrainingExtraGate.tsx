import React from "react";
import { Trans, useTranslation } from "react-i18next";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useInstallExtra } from "@/hooks/useInstallExtra";
import {
  InstallProgress,
  InstallTitleIcon,
  ReadyInstructions,
  installTitle,
} from "./InstallProgress";

interface Props {
  /** The backend's own pip command — data, shown verbatim. */
  installHint: string;
}

const TrainingExtraGate: React.FC<Props> = ({ installHint }) => {
  const install = useInstallExtra("system/training-extra");
  const { t } = useTranslation();
  const idleTitle = t("training.extraGate.title");

  return (
    <div className="max-w-3xl mx-auto">
      <Card className="bg-card border-border rounded-xl">
        <CardHeader>
          <CardTitle className="flex items-center gap-3 text-foreground">
            <InstallTitleIcon state={install.state} />
            {installTitle(t, install.state, idleTitle)}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <InstallProgress
            state={install.state}
            error={install.error}
            logs={install.logs}
            logBoxRef={install.logBoxRef}
            onInstall={install.handleInstall}
            onRetry={install.handleRetry}
            installHint={installHint}
            packageName="accelerate"
            idleTitle={idleTitle}
            idleDescription={
              <Trans
                i18nKey="training.extraGate.description"
                components={[
                  <code
                    key="0"
                    className="px-1 py-0.5 rounded bg-muted text-info"
                  />,
                ]}
              />
            }
            doneDescription={
              <ReadyInstructions text={t("training.install.readyTraining")} />
            }
          />
        </CardContent>
      </Card>
    </div>
  );
};

export default TrainingExtraGate;
