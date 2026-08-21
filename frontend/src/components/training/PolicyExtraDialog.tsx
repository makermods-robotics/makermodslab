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
import { policyTypeShortLabel } from "./types";
import {
  InstallProgress,
  InstallTitleIcon,
  ReadyInstructions,
  installTitle,
} from "./InstallProgress";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  policyType: string;
  packageName: string; // the probed module, e.g. "transformers"
  installTarget: string; // e.g. "lerobot[smolvla]"
  installHint: string; // e.g. "pip install 'lerobot[smolvla]'"
  purpose: "training" | "inference"; // what the caller was about to do
}

/**
 * Per-purpose catalog KEYS, not copy.
 *
 * This used to be `{ verb, noun }` fragments ("Training"/"training",
 * "Running"/"inference") slotted into shared English templates — grammar as
 * data, which does not survive translation: the verb form, the word order and
 * the noun's classifier all differ per language. Each purpose now owns
 * complete sentences instead. Keys (never resolved strings) because a
 * module-level constant is evaluated at import time.
 */
const PURPOSE_KEYS = {
  training: {
    srDescription: "training.policyExtra.srDescriptionTraining",
    description: "training.policyExtra.descriptionTraining",
    ready: "training.install.readyPolicyTraining",
  },
  inference: {
    srDescription: "training.policyExtra.srDescriptionInference",
    description: "training.policyExtra.descriptionInference",
    ready: "training.install.readyPolicyInference",
  },
} as const satisfies Record<Props["purpose"], Record<string, string>>;

// Some policies (smolvla, pi0, pi0_fast, pi05, diffusion) need an optional
// LeRobot extra. This catches the missing package before training/inference
// starts and offers a one-click install, instead of the run dying with a
// buried ImportError.
const PolicyExtraDialog: React.FC<Props> = ({
  open,
  onOpenChange,
  policyType,
  packageName,
  installTarget,
  installHint,
  purpose,
}) => {
  const install = useInstallExtra(`system/policy-extra/${policyType}`, open);
  const { t } = useTranslation();
  // A product name (ACT, SmolVLA…) — never translated.
  const shortLabel = policyTypeShortLabel(policyType);
  const title = t("training.policyExtra.title", { policy: shortLabel });
  const keys = PURPOSE_KEYS[purpose];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-background border-border max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-foreground">
            <InstallTitleIcon state={install.state} />
            {installTitle(t, install.state, title)}
          </DialogTitle>
          <DialogDescription className="sr-only">
            {t(keys.srDescription, {
              target: installTarget,
              policy: shortLabel,
            })}
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
            packageName={installTarget}
            idleTitle={title}
            idleDescription={
              <Trans
                i18nKey={keys.description}
                values={{
                  policy: shortLabel,
                  packageName,
                  target: installTarget,
                }}
                components={[
                  <span key="0" className="font-semibold" />,
                  <code
                    key="1"
                    className="px-1 py-0.5 rounded bg-muted text-info"
                  />,
                  <code
                    key="2"
                    className="px-1 py-0.5 rounded bg-muted text-info"
                  />,
                ]}
              />
            }
            doneDescription={
              <ReadyInstructions text={t(keys.ready, { policy: shortLabel })} />
            }
          />
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default PolicyExtraDialog;
