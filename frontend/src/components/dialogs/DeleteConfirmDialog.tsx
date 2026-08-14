import React from "react";
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
import {
  DeletableItem,
  DeleteResolution,
  resolveDeleteAction,
} from "@/lib/deleteSemantics";

export interface DeleteConfirmDialogProps {
  kind: "dataset" | "model";
  /** null closes the dialog. */
  item: DeletableItem | null;
  /** Interpolated into the confirm title, e.g. the repo id or model name. */
  label: string | undefined;
  onOpenChange: (open: boolean) => void;
  /** The caller performs the actual API call and is responsible for
   * clearing `item` (via onOpenChange or its own state) once done. */
  onConfirm: (resolution: DeleteResolution) => void;
}

/**
 * Shared confirm popup for the dataset/model delete pipeline. Wraps
 * resolveDeleteAction's local-delete / local-copy-removal / unpin / hide
 * semantics in one AlertDialog so the two dataset entry points
 * (DatasetDetailDialog, LibrarySheet's "My datasets" row) don't each
 * hand-roll their own copy of this markup. SkillManageDialog.tsx keeps its
 * own inline version for models — not migrated here, out of scope.
 */
const DeleteConfirmDialog: React.FC<DeleteConfirmDialogProps> = ({
  kind,
  item,
  label,
  onOpenChange,
  onConfirm,
}) => {
  const res = item ? resolveDeleteAction(kind, item) : null;

  return (
    <AlertDialog open={item !== null} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="break-words">
            {res?.titlePrefix} "<span className="break-all">{label}</span>"?
          </AlertDialogTitle>
          <AlertDialogDescription>{res?.description}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => res && onConfirm(res)}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
          >
            {res?.confirmLabel}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
};

export default DeleteConfirmDialog;
