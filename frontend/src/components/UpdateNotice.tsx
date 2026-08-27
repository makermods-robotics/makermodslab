import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Loader2, Copy, Sparkles, ChevronRight } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useUpdateCheck } from "@/hooks/useUpdateCheck";

/**
 * App-level popup that notifies the user when a newer MakerMods Lab is available on
 * GitHub. Offers a copy-able upgrade command, a best-effort "Update now" button
 * (runs the pip upgrade on the backend), and a "don't ask again" opt-out.
 */
const UpdateNotice = () => {
  const { status, open, dismiss } = useUpdateCheck();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { t } = useTranslation();
  const [dontAsk, setDontAsk] = useState(false);
  const [updating, setUpdating] = useState(false);
  const [output, setOutput] = useState<string | null>(null);

  if (!status) return null;

  const behind =
    typeof status.commits_behind === "number" && status.commits_behind > 0
      ? t("shared.update.behind", { count: status.commits_behind })
      : t("shared.update.available");

  const copyCommand = async () => {
    if (!status.update_command) return;
    try {
      await navigator.clipboard.writeText(status.update_command);
      toast({
        title: t("shared.update.copiedTitle"),
        description: t("shared.update.copiedDescription"),
      });
    } catch {
      toast({
        title: t("shared.update.copyFailedTitle"),
        description: t("shared.update.copyFailedDescription"),
        variant: "destructive",
      });
    }
  };

  const runUpdate = async () => {
    setUpdating(true);
    setOutput(null);
    try {
      const r = await fetchWithHeaders(`${baseUrl}/system/update`, {
        method: "POST",
      });
      const body: { success: boolean; message: string; output: string } =
        await r.json();
      if (body.success) {
        toast({
          // body.message is backend prose — shown as-is.
          title: t("shared.update.updatedTitle"),
          description: body.message,
        });
        dismiss(false);
      } else {
        setOutput(body.output || body.message);
        toast({
          title: t("shared.update.failedTitle"),
          description: body.message,
          variant: "destructive",
        });
      }
    } catch (e) {
      toast({
        title: t("shared.update.failedTitle"),
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setUpdating(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o && !updating) dismiss(dontAsk);
      }}
    >
      <DialogContent
        className="bg-background border-border text-foreground max-w-lg"
        onOpenAutoFocus={(e) => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-foreground">
            <Sparkles className="w-5 h-5 text-warn" />
            {t("shared.update.title")}
          </DialogTitle>
          <DialogDescription className="text-muted-foreground">
            {t("shared.update.body", { behind })}
            <br />
            {t("shared.update.bodyLine2")}
            {status.compare_url && (
              <>
                {" "}
                <a
                  href={status.compare_url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-info underline hover:text-info/80"
                >
                  {t("shared.update.seeChanges")}
                </a>
                .
              </>
            )}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <Collapsible>
            <CollapsibleTrigger className="group flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors">
              <ChevronRight className="w-3.5 h-3.5 transition-transform group-data-[state=open]:rotate-90" />
              {t("shared.update.manual")}
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-2">
              <div className="flex items-start gap-2">
                <code className="min-w-0 flex-1 px-2 py-1.5 rounded bg-muted text-info text-xs break-all whitespace-pre-wrap">
                  {status.update_command}
                </code>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={copyCommand}
                  title={t("shared.update.copyCommand")}
                  className="shrink-0 bg-background border-border text-foreground hover:bg-accent"
                >
                  <Copy className="w-4 h-4" />
                </Button>
              </div>
            </CollapsibleContent>
          </Collapsible>

          {output && (
            <pre className="max-h-40 overflow-auto rounded bg-muted p-2 text-xs text-muted-foreground whitespace-pre-wrap">
              {output}
            </pre>
          )}

          <div className="flex items-center justify-between gap-3 pt-1">
            <label className="flex items-center gap-2 text-sm text-muted-foreground">
              <Checkbox
                checked={dontAsk}
                onCheckedChange={(v) => setDontAsk(v === true)}
                className="border-border data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground"
              />
              {t("shared.update.dontAsk")}
            </label>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                onClick={() => dismiss(dontAsk)}
                disabled={updating}
                className="text-muted-foreground hover:bg-accent hover:text-foreground"
              >
                {t("shared.update.later")}
              </Button>
              {status.can_auto_update && (
                <Button onClick={runUpdate} disabled={updating}>
                  {updating ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      {t("shared.update.updating")}
                    </>
                  ) : (
                    t("shared.update.now")
                  )}
                </Button>
              )}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default UpdateNotice;
