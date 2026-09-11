import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Check, Copy, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useHfAuth } from "@/contexts/HfAuthContext";

interface HfAuthDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const HfAuthDialog: React.FC<HfAuthDialogProps> = ({ open, onOpenChange }) => {
  const { t } = useTranslation();
  const { auth, refetch } = useHfAuth();
  const [copied, setCopied] = useState(false);
  const [refetching, setRefetching] = useState(false);

  if (auth.status !== "unauthenticated") {
    return null;
  }

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(auth.loginCommand);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (err) {
      console.warn("Clipboard write failed:", err);
    }
  };

  const handleRefetch = async () => {
    setRefetching(true);
    try {
      await refetch();
    } finally {
      setRefetching(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="text-amber-700 dark:text-amber-200">
            {t("landing.hfAuthDialog.title")}
          </DialogTitle>
          <DialogDescription className="text-muted-foreground">
            {t("landing.hfAuthDialog.description")}
          </DialogDescription>
        </DialogHeader>
        <pre className="bg-muted p-3 rounded border border-border text-xs sm:text-sm overflow-x-auto flex items-center justify-between gap-2">
          <code className="text-ok">{auth.loginCommand}</code>
          <button
            type="button"
            onClick={handleCopy}
            className="flex-shrink-0 text-muted-foreground hover:text-foreground transition-colors"
            aria-label={t("landing.hfAuthDialog.copyAria")}
          >
            {copied ? (
              <Check className="w-4 h-4 text-ok" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </button>
        </pre>
        <Button
          variant="outline"
          size="sm"
          onClick={handleRefetch}
          disabled={refetching}
          className="border-amber-500/50 bg-transparent text-amber-700 dark:text-amber-200 hover:bg-amber-500/10"
        >
          <RefreshCw
            className={`w-4 h-4 mr-2 ${refetching ? "animate-spin" : ""}`}
          />
          {t("landing.hfAuthDialog.recheck")}
        </Button>
      </DialogContent>
    </Dialog>
  );
};

export default HfAuthDialog;
