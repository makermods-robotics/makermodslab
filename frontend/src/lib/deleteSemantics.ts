/**
 * Shared delete semantics for the dataset and model pickers/info cards.
 *
 * One resolver maps a listing row to what its trash button actually does, so
 * every entry point (picker row, info card) opens the same confirm dialog with
 * the same action. The rules (user-decided):
 *
 *  - Delete NEVER deletes or mutates a Hub repo.
 *  - "both" (local + hub): TWO-PRESS. First press removes only the local copy —
 *    the row flips to a plain hub row and stays listed, and the selection is
 *    kept (the dataset/model still exists). A second press (now hub-only)
 *    hides it.
 *  - Pinned custom hub rows are unpinned (existing behavior).
 *  - Own-namespace hub-only rows are hidden via the persistent hidden-list
 *    (they'd otherwise resurface on every Hub listing).
 *  - Local-only rows are destructive local file deletes (existing routes).
 */

export type DeleteAction = "delete-local" | "delete-local-copy" | "unpin" | "hide";

export interface DeletableItem {
  source: "local" | "hub" | "both";
  saved_custom?: boolean;
}

export interface DeleteResolution {
  action: DeleteAction;
  /** Dialog title verb; the caller interpolates the item label after it. */
  titlePrefix: string;
  description: string;
  confirmLabel: string;
  /** True when a confirmed action removes the row from the listing entirely
   * (hide / unpin / local-only delete) — the persisted selection should then
   * be cleared. False for the both→hub flip, where the row stays listed and
   * the selection is kept. */
  clearsSelection: boolean;
}

const LOCAL_DELETE_DESCRIPTION: Record<"dataset" | "model", string> = {
  dataset:
    "This moves the dataset to trash. You can undo it from the Recently " +
    "deleted list for 24 hours, but the disk space isn't freed until then.",
  model:
    "This permanently removes the model's local files from disk — including " +
    "its checkpoints. You can't undo this. A Hub copy, if any, is not affected.",
};

export function resolveDeleteAction(
  kind: "dataset" | "model",
  item: DeletableItem,
): DeleteResolution {
  // "both" outranks saved_custom: the first press always removes the local
  // copy; the (possibly pinned) hub row survives for a second press.
  if (item.source === "both") {
    return {
      action: "delete-local-copy",
      titlePrefix: "Remove local copy of",
      description:
        "This removes the local copy from disk — the Hub copy stays, and it " +
        `remains listed as a Hub ${kind}.`,
      confirmLabel: "Remove local copy",
      clearsSelection: false,
    };
  }
  if (item.source === "hub" && item.saved_custom) {
    return {
      action: "unpin",
      titlePrefix: "Remove",
      description:
        `This just removes the ${kind} from your list. The Hub repo and any ` +
        "local copy are untouched — you can re-add it any time from the " +
        `Add ${kind} menu.`,
      confirmLabel: "Remove",
      clearsSelection: true,
    };
  }
  if (item.source === "hub") {
    return {
      action: "hide",
      titlePrefix: "Remove",
      description:
        `This hides the ${kind} from your list. The Hub repo is not deleted — ` +
        `you can re-add it any time from the Add ${kind} menu.`,
      confirmLabel: "Remove",
      clearsSelection: true,
    };
  }
  return {
    action: "delete-local",
    titlePrefix: "Delete",
    description: LOCAL_DELETE_DESCRIPTION[kind],
    confirmLabel: "Delete",
    clearsSelection: true,
  };
}
