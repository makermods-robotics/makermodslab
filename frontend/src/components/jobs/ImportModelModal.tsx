import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertTriangle, Download, Loader2 } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { importModel, jobDisplayName } from "@/lib/jobsApi";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImported: () => void;
}

const ImportModelModal: React.FC<Props> = ({ open, onOpenChange, onImported }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { t } = useTranslation();
  const [source, setSource] = useState("");
  const [name, setName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    const src = source.trim();
    if (!src) return;
    setSubmitting(true);
    setError(null);
    try {
      const record = await importModel(
        baseUrl,
        fetchWithHeaders,
        src,
        name.trim() || undefined,
      );
      if (record.already_imported) {
        // Duplicate source: the backend returned the existing entry (id and
        // display alias preserved) instead of registering a second one.
        toast({
          title: t("jobs.importModal.alreadyImportedTitle"),
          // The name is the user's own — interpolated, never translated.
          description: t("jobs.importModal.alreadyImportedDescription", {
            name: jobDisplayName(record),
          }),
        });
      }
      setSource("");
      setName("");
      onOpenChange(false);
      onImported();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-background border-border sm:max-w-[520px] p-8">
        <DialogHeader>
          <DialogTitle className="text-foreground text-center text-2xl font-bold">
            {t("jobs.importModal.title")}
          </DialogTitle>
          <DialogDescription className="text-muted-foreground text-center">
            {t("jobs.importModal.description")}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="source" className="text-sm font-medium text-muted-foreground">
              {t("jobs.importModal.sourceLabel")}
            </Label>
            <Input
              id="source"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              placeholder={t("jobs.importModal.sourcePlaceholder")}
              className="bg-background border-input"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="name" className="text-sm font-medium text-muted-foreground">
              {t("jobs.importModal.nameLabel")}
            </Label>
            <Input
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={t("jobs.importModal.namePlaceholder")}
              className="bg-background border-input"
            />
          </div>

          {error ? (
            <Alert className="bg-destructive/10 border-destructive/40 text-destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}

          <div className="flex gap-3 justify-center pt-2">
            <Button
              onClick={handleSubmit}
              disabled={!source.trim() || submitting}
              className="bg-primary hover:bg-primary/90 text-primary-foreground px-8 disabled:opacity-40"
            >
              {submitting ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Download className="w-4 h-4 mr-2" />
              )}
              {submitting
                ? t("jobs.importModal.submitting")
                : t("jobs.importModal.submit")}
            </Button>
            <Button
              onClick={() => onOpenChange(false)}
              variant="outline"
              className="border-border px-8 text-muted-foreground bg-background hover:bg-muted"
            >
              {t("common.cancel")}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ImportModelModal;
