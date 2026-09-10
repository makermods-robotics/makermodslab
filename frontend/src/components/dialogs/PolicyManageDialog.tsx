import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Play } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { ModelItem, hideModel, removeCustomModel } from "@/lib/modelsApi";
import { resolveDeleteAction } from "@/lib/deleteSemantics";
import ModelInfoCard from "@/components/landing/ModelInfoCard";

export interface PolicyManageDialogProps {
  model: ModelItem | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Refresh the models listing after upload / download / delete. */
  onChanged: () => void;
  /** Run this policy on the corner robot (→ Deploy panel, prefilled). */
  onRun: (model: ModelItem) => void;
}

/**
 * Manage one of MY policies — wraps the existing ModelInfoCard (unmodified) in a
 * dialog so the model-library management surface survives the Layout D
 * redesign: Hub upload, checkpoint download, rename-adjacent metadata, and the
 * unified delete pipeline (local delete / local-copy removal / unpin / hide via
 * resolveDeleteAction — the Hub repo itself is never touched). Ported from the
 * old ModelsPanel's confirm pipeline.
 */
const PolicyManageDialog: React.FC<PolicyManageDialogProps> = ({
  model,
  open,
  onOpenChange,
  onChanged,
  onRun,
}) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [pendingDelete, setPendingDelete] = useState<ModelItem | null>(null);

  if (!model) return null;

  // Destructive deletes (a local run's files, a "both" row's local copy) are
  // deliberately no longer offered here — that surface moved out of the
  // library UI (the backend routes remain for a future management menu). Only
  // the non-destructive listing management survives: unpinning a pinned custom
  // hub row and hiding an own-namespace hub row.
  const modelAction = resolveDeleteAction("model", model).action;
  const canRemoveFromList = modelAction === "unpin" || modelAction === "hide";

  const res = pendingDelete ? resolveDeleteAction("model", pendingDelete) : null;

  // Ported from ModelsPanel.confirmDeleteModel, minus the destructive arms:
  // resolveDeleteAction still decides the semantics, but only rows resolving
  // to unpin/hide ever reach here (canRemoveFromList gates the button).
  const confirmDelete = async () => {
    const item = pendingDelete;
    if (!item) return;
    setPendingDelete(null);
    const resolution = resolveDeleteAction("model", item);
    try {
      if (resolution.action === "unpin") {
        await removeCustomModel(baseUrl, fetchWithHeaders, item.id);
        // The model name is data — it stays the toast's verbatim description.
        toast({
          title: t("dialogs.policyManage.toast.removedFromList"),
          description: item.name,
        });
      } else if (resolution.action === "hide") {
        await hideModel(baseUrl, fetchWithHeaders, item.hf_repo_id ?? item.id);
        toast({
          title: t("dialogs.policyManage.toast.removedFromList"),
          description: item.name,
        });
      }
      onChanged();
      onOpenChange(false);
    } catch (e) {
      toast({
        title: t("dialogs.policyManage.toast.removeFailed"),
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle className="break-all font-mono text-base">
              {model.name}
            </DialogTitle>
          </DialogHeader>

          <ModelInfoCard
            id={model.id}
            // Upload + local-detail affordances are local-only, mirroring the
            // old ModelsPanel gating.
            isLocal={model.source === "local"}
            // Only list-management rows (unpin / hide) keep the affordance;
            // destructive deletes are gone from this surface.
            canDelete={canRemoveFromList}
            onDelete={() => setPendingDelete(model)}
            onUploaded={onChanged}
            onDownloaded={onChanged}
          />

          <Button onClick={() => onRun(model)} className="w-full gap-2">
            <Play className="h-4 w-4" />
            {t("dialogs.policyManage.runOnRobot")}
          </Button>
        </DialogContent>
      </Dialog>

      <AlertDialog
        open={pendingDelete !== null}
        onOpenChange={(o) => !o && setPendingDelete(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            {/* `resolveDeleteAction` names WHICH sentence to render; the
                title is a whole question with the model name — data — dropped
                in as `{{label}}` wherever the language wants it. */}
            <AlertDialogTitle className="break-words">
              {res ? t(res.titleKey as never, { label: pendingDelete?.name }) : null}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {res ? t(res.descriptionKey as never) : null}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmDelete}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {res ? t(res.confirmKey as never) : null}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
};

export default PolicyManageDialog;
