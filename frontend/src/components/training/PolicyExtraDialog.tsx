import React from "react";
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

const PURPOSE_COPY: Record<Props["purpose"], { verb: string; noun: string }> = {
  training: { verb: "Training", noun: "training" },
  inference: { verb: "Running", noun: "inference" },
};

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
  const shortLabel = policyTypeShortLabel(policyType);
  const title = `${shortLabel} needs an extra package`;
  const { verb, noun } = PURPOSE_COPY[purpose];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-background border-border max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-foreground">
            <InstallTitleIcon state={install.state} />
            {installTitle(install.state, title)}
          </DialogTitle>
          <DialogDescription className="sr-only">
            Install {installTarget} for {noun} with {shortLabel}.
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
              <>
                {verb} a <span className="font-semibold">{shortLabel}</span> policy needs the{" "}
                <code className="px-1 py-0.5 rounded bg-muted text-info">{packageName}</code>{" "}
                package (installed via{" "}
                <code className="px-1 py-0.5 rounded bg-muted text-info">{installTarget}</code>),
                which isn't in this environment yet. Install it to {purpose === "training" ? "train" : "run"} this policy.
              </>
            }
            doneDescription={<ReadyInstructions purpose={`${shortLabel} ${noun}`} />}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default PolicyExtraDialog;
