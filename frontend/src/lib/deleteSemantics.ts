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
 *
 * The dialog COPY is not built here any more. It used to be assembled by
 * splicing the noun "dataset"/"model" into a shared template, and the title was
 * handed over as a prefix for the caller to concatenate a name after — both
 * untranslatable, since another language puts that noun in a different clause
 * and never builds a sentence out of two independently-chosen halves. Each
 * (action × kind) combination now names one complete sentence in the `library`
 * catalog, and the resolver only says WHICH one: see `titleKey` below.
 */

import enLibrary from "@/i18n/locales/en/library";

export type DeleteAction = "delete-local" | "delete-local-copy" | "unpin" | "hide";

/** What is being deleted. A data value — it selects copy and API routes, and is
 * never shown to the user. */
export type DeletableKind = "dataset" | "model";

export interface DeletableItem {
  source: "local" | "hub" | "both";
  saved_custom?: boolean;
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
  /**
   * @deprecated English fallbacks for call sites that have not moved to the
   * keys above. Read straight out of the English catalog so the two can't
   * drift; delete these three fields once every caller uses `t()`.
   */
  titlePrefix: string;
  /** @deprecated Use `descriptionKey` with `t()`. */
  description: string;
  /** @deprecated Use `confirmKey` with `t()`. */
  confirmLabel: string;
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

/** Legacy English title prefixes, kept only for the `titlePrefix` fallback —
 * the translated titles are whole sentences and have no prefix form. */
const LEGACY_TITLE_PREFIX: Record<DeleteAction, string> = {
  "delete-local-copy": "Remove local copy of",
  unpin: "Remove",
  hide: "Remove",
  "delete-local": "Delete",
};

/** The catalog keys (and their English fallbacks) for one action × kind. */
function copyFor(action: DeleteAction, kind: DeletableKind) {
  const group = COPY_GROUP[action];
  const en = enLibrary.delete[group][kind];
  return {
    titleKey: `library.delete.${group}.${kind}.title`,
    descriptionKey: `library.delete.${group}.${kind}.description`,
    confirmKey: `library.delete.${group}.${kind}.confirm`,
    titlePrefix: LEGACY_TITLE_PREFIX[action],
    description: en.description as string,
    confirmLabel: en.confirm as string,
  };
}

export function resolveDeleteAction(
  kind: DeletableKind,
  item: DeletableItem,
): DeleteResolution {
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
