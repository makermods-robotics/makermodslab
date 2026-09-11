import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Terminal, ExternalLink, Copy, Check } from "lucide-react";

const ONE_LINER =
  "uv tool install git+https://github.com/makermods-robotics/makermodslab.git && makermodslab";
const LOCAL_URL = "http://localhost:8000/";

interface UsageInstructionsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  dismissible?: boolean;
}

const UsageInstructionsModal: React.FC<UsageInstructionsModalProps> = ({
  open,
  onOpenChange,
  dismissible = true,
}) => {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);

  const blockClose = (e: Event) => {
    if (!dismissible) e.preventDefault();
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(ONE_LINER);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (err) {
      console.warn("Clipboard write failed:", err);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={dismissible ? onOpenChange : () => undefined}
    >
      <DialogContent
        className="text-muted-foreground sm:max-w-xl"
        hideClose={!dismissible}
        onEscapeKeyDown={blockClose}
        onPointerDownOutside={blockClose}
        onInteractOutside={blockClose}
      >
        <DialogHeader className="text-center sm:text-center min-w-0">
          <DialogTitle className="text-foreground flex items-center justify-center gap-2 text-xl">
            <Terminal className="w-6 h-6" />
            {t("landing.usageInstructions.title")}
          </DialogTitle>
          <DialogDescription>
            {t("landing.usageInstructions.description")}
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-2 min-w-0">
          <button
            type="button"
            onClick={handleCopy}
            aria-label={t("landing.usageInstructions.copyAria")}
            className="group relative w-full bg-muted hover:bg-muted/80 rounded-lg border border-border hover:border-foreground/20 text-left transition-colors cursor-pointer"
          >
            <pre className="p-4 pr-12 text-xs sm:text-sm overflow-x-auto whitespace-pre-wrap break-all">
              <code className="text-ok">{ONE_LINER}</code>
            </pre>
            <span className="absolute right-2 top-2 flex items-center gap-1 px-2 py-1 rounded text-xs text-muted-foreground group-hover:text-foreground bg-background/80">
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-ok" />
                  {t("landing.usageInstructions.copied")}
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  {t("landing.usageInstructions.copy")}
                </>
              )}
            </span>
          </button>
          <p className="text-muted-foreground text-sm text-center">
            {t("landing.usageInstructions.afterRunning")}
          </p>
          <Button
            asChild
            className="w-full"
          >
            <a href={LOCAL_URL} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="w-4 h-4 mr-2" />
              {t("landing.usageInstructions.open")}
            </a>
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default UsageInstructionsModal;
