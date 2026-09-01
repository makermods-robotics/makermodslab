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
  /** When set, the install runs on THIS LAN node (through the server-to-server
   * proxy) instead of the local environment — the offloaded run imports from
   * the peer's site-packages, so that is where the extra must land. `name` is
   * the node's display name (data, rendered verbatim). */
  node?: { instanceId: string; name: string };
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

// The node variant owns complete sentences too — the extra lands on the PEER,
// and every line must say so or the user "fixes" the wrong machine.
const NODE_KEYS = {
  srDescription: "training.policyExtra.srDescriptionTrainingNode",
  description: "training.policyExtra.descriptionTrainingNode",
  ready: "training.install.readyPolicyTrainingNode",
} as const;

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
  node,
}) => {
  // On a node target the whole flow — status seed, install POST, progress
  // poll — runs through the server-to-server proxy; the pip subprocess runs
  // on the peer, in the environment its training subprocesses import from.
  const install = useInstallExtra(
    node
      ? `nodes/${node.instanceId}/policy-extra/${policyType}`
      : `system/policy-extra/${policyType}`,
    open,
  );
  const { t } = useTranslation();
  // A product name (ACT, SmolVLA…) — never translated.
  const shortLabel = policyTypeShortLabel(policyType);
  const title = node
    ? t("training.policyExtra.titleNode", { policy: shortLabel, node: node.name })
    : t("training.policyExtra.title", { policy: shortLabel });
  const keys = node ? NODE_KEYS : PURPOSE_KEYS[purpose];

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
              node: node?.name ?? "",
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
                  node: node?.name ?? "",
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
                  <span key="3" className="font-semibold" />,
                ]}
              />
            }
            doneDescription={
              <ReadyInstructions
                text={t(keys.ready, {
                  policy: shortLabel,
                  node: node?.name ?? "",
                })}
              />
            }
          />
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default PolicyExtraDialog;
