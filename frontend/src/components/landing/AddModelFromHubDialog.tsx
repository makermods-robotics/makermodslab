import React, { useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Download, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { HUB_REPO_ID_RE } from "@/lib/repoId";

interface AddModelFromHubDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Add the typed Hub id to the picker (pin + select). When `download` is true
   * the parent also kicks off a background download into the local models dir. */
  onAdd: (repoId: string, download: boolean) => void;
}

/**
 * "Add a model from Hugging Face" form — the models twin of
 * AddDatasetFromHubDialog. Takes a `namespace/name` Hub repo id, validated with
 * the same rule as the backend, and adds it to the picker's Hugging Face list
 * (pin + select via the parent). An optional "Download to this machine now"
 * toggle additionally starts a background download of the checkpoint so
 * inference can run on it offline (source flips to "both"); left off, the model
 * is listed and inference fetches it from the Hub on demand.
 */
const AddModelFromHubDialog: React.FC<AddModelFromHubDialogProps> = ({
  open,
  onOpenChange,
  onAdd,
}) => {
  const { t } = useTranslation();
  const [repoId, setRepoId] = useState("");
  const [download, setDownload] = useState(false);

  useEffect(() => {
    if (open) {
      setRepoId("");
      setDownload(false);
    }
  }, [open]);

  const trimmed = repoId.trim();
  const isValid = HUB_REPO_ID_RE.test(trimmed);
  const showError = trimmed.length > 0 && !isValid;

  const handleConfirm = () => {
    if (!isValid) return;
    onAdd(trimmed, download);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{t("landing.addModelFromHub.title")}</DialogTitle>
          <DialogDescription className="text-muted-foreground">
            {t("landing.addModelFromHub.description")}
          </DialogDescription>
        </DialogHeader>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleConfirm();
          }}
          className="space-y-4"
        >
          <div>
            <Label htmlFor="hub-model-id" className="text-muted-foreground">
              {t("landing.addModelFromHub.idLabel")}
            </Label>
            <Input
              id="hub-model-id"
              autoFocus
              value={repoId}
              onChange={(e) =>
                setRepoId(e.target.value.replace(/[^A-Za-z0-9._\-/]/g, ""))
              }
              placeholder="org/name"
              aria-invalid={showError}
              className="mt-1 aria-[invalid=true]:border-destructive/70"
            />
            {showError && (
              <p className="mt-1 text-xs text-destructive">
                {/* "org/name" is a format literal, so it rides inside the
                    sentence as <Trans> markup rather than a concatenation. */}
                <Trans
                  i18nKey="landing.addModelFromHub.idError"
                  components={[<span key="0" className="font-mono" />]}
                />
              </p>
            )}
          </div>
          <label className="flex items-start gap-2 text-sm text-muted-foreground cursor-pointer">
            <input
              type="checkbox"
              checked={download}
              onChange={(e) => setDownload(e.target.checked)}
              className="mt-0.5 h-4 w-4 accent-blue-500"
            />
            <span>
              {t("landing.addModelFromHub.downloadNow")}
              <span className="block text-xs text-muted-foreground">
                {t("landing.addModelFromHub.downloadNowHint")}
              </span>
            </span>
          </label>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              className=""
            >
              {t("common.cancel")}
            </Button>
            <Button
              type="submit"
              disabled={!isValid}
              className=""
            >
              {download ? (
                <Download className="w-4 h-4 mr-2" />
              ) : (
                <Plus className="w-4 h-4 mr-2" />
              )}
              {download
                ? t("landing.addModelFromHub.submitWithDownload")
                : t("landing.addModelFromHub.submit")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default AddModelFromHubDialog;
