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
 *  - EXCEPT a "both" row whose local side is a training RUN (`local_kind`).
 *    The two-press rule rests on the local copy being replaceable from the Hub,
 *    which holds for a downloaded model but not for a run: publishing uploads
 *    the checkpoints the user chose, so a run's unpublished checkpoints exist
 *    nowhere else and "the Hub copy stays" would understate the loss. Such a
 *    row gets the destructive local delete instead.
 *  - Pinned custom hub rows are unpinned (existing behavior).
 *  - Own-namespace hub-only rows are hidden via the persistent hidden-list
 *    (they'd otherwise resurface on every Hub listing).
 *  - Local-only rows are destructive local file deletes (existing routes).
 *
 * The dialog COPY is not built here any more. It used to be assembled by
 * splicing the noun "dataset"/"model" into a shared template, and the title was
 * handed over as a prefix for the caller to concatenate a name after — both
 * untranslatable, since another language puts that noun in a different clause
 * and never builds a sentence out of two independently-chosen halves. Each
 * (action × kind) combination now names one complete sentence in the `library`
 * catalog, and the resolver only says WHICH one: see `titleKey` below.
 */

// Type-only: the catalog shape constrains COPY_GROUP so a renamed group is a
// compile error rather than a key that silently resolves to nothing.
import type enLibrary from "@/i18n/locales/en/library";

export type DeleteAction = "delete-local" | "delete-local-copy" | "unpin" | "hide";

/** What is being deleted. A data value — it selects copy and API routes, and is
 * never shown to the user. */
export type DeletableKind = "dataset" | "model";

export interface DeletableItem {
  source: "local" | "hub" | "both";
  saved_custom?: boolean;
  /** What the LOCAL side of this row is: a training run (holds every checkpoint
   * it saved, only some of which may be published) or a copy pulled from the
   * Hub / imported from disk. Absent for datasets and for hub-only rows. */
  local_kind?: "run" | "downloaded";
}

export interface DeleteResolution {
  action: DeleteAction;
  /**
   * i18next key for the confirm dialog's title — a WHOLE question, with the
   * item's display name interpolated as `{{label}}`:
   * `t(res.titleKey, { label: item.name })`. The name is data and is rendered
   * verbatim wherever the language wants it.
   */
  titleKey: string;
  /** i18next key for the dialog body. Takes no interpolation. */
  descriptionKey: string;
  /** i18next key for the destructive confirm button's label. */
  confirmKey: string;
  /** True when a confirmed action removes the row from the listing entirely
   * (hide / unpin / local-only delete) — the persisted selection should then
   * be cleared. False for the both→hub flip, where the row stays listed and
   * the selection is kept. */
  clearsSelection: boolean;
}

/** `DeleteAction` → the catalog group holding its copy. The action strings are
 * data (they pick the API route); these are only their names in `library`. */
const COPY_GROUP = {
  "delete-local-copy": "localCopy",
  unpin: "unpin",
  hide: "hide",
  "delete-local": "local",
} as const satisfies Record<DeleteAction, keyof typeof enLibrary.delete>;

/** The catalog keys for one action × kind. */
function copyFor(action: DeleteAction, kind: DeletableKind) {
  const group = COPY_GROUP[action];
  return {
    titleKey: `library.delete.${group}.${kind}.title`,
    descriptionKey: `library.delete.${group}.${kind}.description`,
    confirmKey: `library.delete.${group}.${kind}.confirm`,
  };
}

export function resolveDeleteAction(
  kind: DeletableKind,
  item: DeletableItem,
): DeleteResolution {
  // A published training run is "both", but its local side is not a replaceable
  // copy — only the checkpoints the user chose to publish are on the Hub, so
  // this is a destructive delete (the `local` copy: "including its checkpoints
  // … a Hub copy is not affected") rather than the two-press "remove local
  // copy" flow, whose "the Hub copy stays" wording would understate the loss.
  if (item.source === "both" && item.local_kind === "run") {
    return {
      action: "delete-local",
      ...copyFor("delete-local", kind),
      clearsSelection: true,
    };
  }
  // "both" outranks saved_custom: the first press always removes the local
  // copy; the (possibly pinned) hub row survives for a second press.
  if (item.source === "both") {
    return {
      action: "delete-local-copy",
      ...copyFor("delete-local-copy", kind),
      clearsSelection: false,
    };
  }
  if (item.source === "hub" && item.saved_custom) {
    return {
      action: "unpin",
      ...copyFor("unpin", kind),
      clearsSelection: true,
    };
  }
  if (item.source === "hub") {
    return {
      action: "hide",
      ...copyFor("hide", kind),
      clearsSelection: true,
    };
  }
  return {
    action: "delete-local",
    ...copyFor("delete-local", kind),
    clearsSelection: true,
  };
}
