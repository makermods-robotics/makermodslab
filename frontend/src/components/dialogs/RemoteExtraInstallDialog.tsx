import React, { useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useApi } from "@/contexts/ApiContext";
import { useInstallExtra } from "@/hooks/useInstallExtra";
import { getRemoteExtra } from "@/lib/remoteApi";
import {
  InstallProgress,
  InstallTitleIcon,
  ReadyInstructions,
  installTitle,
} from "@/components/training/InstallProgress";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/**
 * Install flow for the `remote` optional extra — what a hosting or
 * remote-teleoperation start refuses without (409 system.extra_missing).
 * The twin of WandbInstallDialog over the system/remote-extra trio; the
 * backend's own pip command is fetched on open and shown verbatim.
 */
const RemoteExtraInstallDialog: React.FC<Props> = ({ open, onOpenChange }) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const install = useInstallExtra("system/remote-extra", open);
  const [installHint, setInstallHint] = useState("");
  const idleTitle = t("dialogs.remoteExtra.title");

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    getRemoteExtra(baseUrl, fetchWithHeaders)
      .then((status) => {
        if (!cancelled) setInstallHint(status.install_hint);
      })
      .catch(() => {
        /* the hint is a convenience; the Install button still works */
      });
    return () => {
      cancelled = true;
    };
  }, [open, baseUrl, fetchWithHeaders]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-background border-border max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-foreground">
            <InstallTitleIcon state={install.state} />
            {installTitle(t, install.state, idleTitle)}
          </DialogTitle>
          <DialogDescription className="sr-only">
            {t("dialogs.remoteExtra.srDescription")}
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
            packageName="remote"
            idleTitle={idleTitle}
            idleDescription={
              <Trans
                i18nKey="dialogs.remoteExtra.description"
                components={[
                  <code
                    key="0"
                    className="px-1 py-0.5 rounded bg-muted text-info"
                  />,
                ]}
              />
            }
            doneDescription={
              <ReadyInstructions text={t("dialogs.remoteExtra.ready")} />
            }
          />
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default RemoteExtraInstallDialog;
