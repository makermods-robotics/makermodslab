/**
 * Presentation helpers for model/skill card titles.
 *
 * The derivation itself is the backend's — `derive_imported_title` in
 * makermodslab/utils/naming.py peels an imported model's repo id down to its
 * task segment, and `dedupe_display_names` keeps two of them apart — so a card
 * only has to render the name it is given. What CSS can't do is what lives
 * here: `truncate` always eats the tail, which is the wrong half for a name
 * whose distinguishing text sits at the end.
 */

import { validateModelName } from "@/lib/datasetName";

/**
 * Shorten `text` from the MIDDLE, keeping both ends readable.
 *
 * End-truncation is the wrong default for repo-derived names: their head is
 * boilerplate the author repeats across every model and their tail is the part
 * that identifies this one, so cutting the tail hides exactly what the user is
 * scanning for. Returns `text` untouched when it already fits, which is the
 * normal case for a derived title — this only bites on the fallback path, a
 * community repo whose name we can't parse and therefore can't shorten safely.
 */
export function middleEllipsis(text: string, max = 32): string {
  if (max < 5 || text.length <= max) return text;
  // One char of the budget goes to the ellipsis; the head keeps the odd char,
  // since a name's first word is usually the more recognizable of the two.
  const head = Math.ceil((max - 1) / 2);
  const tail = max - 1 - head;
  return `${text.slice(0, head)}…${text.slice(text.length - tail)}`;
}

/**
 * A trailing " (…)" the backend appended to break a name collision.
 *
 * `dedupe_display_names` (makermodslab/utils/naming.py) resolves two rows that
 * derived the same title by appending the one fact that separates them — the
 * run's date, its date and time, or an ordinal. That suffix is therefore the
 * MOST load-bearing text in the whole label and the LEAST survivable: it sits
 * at the very end, where `truncate` starts eating. "eraser_place (2026-08-…"
 * and "eraser_place (2026-08-…" are two rows the suffix was added to tell
 * apart, rendered identically.
 *
 * Splitting lets the caller render the base and the suffix as separate spans,
 * so only the base shrinks (see DisplayName). A name the user typed that
 * happens to end in parentheses splits too — harmless, since both halves are
 * rendered adjacent and read exactly as before.
 */
export function splitDedupeSuffix(name: string): {
  base: string;
  suffix: string | null;
} {
  const match = /^(.*\S)\s\(([^()]+)\)$/.exec(name);
  if (!match) return { base: name, suffix: null };
  return { base: match[1], suffix: match[2] };
}

/**
 * A dedupe suffix's shape when it is a timestamp: `YYYY-MM-DD` or
 * `YYYY-MM-DD HH:MM` — the ladder `utils/naming.imported_name_suffixes` /
 * `iso_time_suffixes` walk, as `splitDedupeSuffix` hands it back (parentheses
 * already stripped). Anchored and fully specified so nothing else can match:
 * the same ladder ends in a namespace (`lerobot`) or a bare ordinal (`2`) when
 * no timestamp separates the collision, and neither has a year to drop.
 */
const DEDUPE_DATE_SUFFIX_RE = /^\d{4}-(\d{2}-\d{2}(?: \d{2}:\d{2})?)$/;

/**
 * The DISPLAY form of a dedupe suffix: the same suffix with its year dropped.
 *
 * The suffix is the only thing telling two same-titled cards apart, so it is
 * the half that must render whole (DisplayName makes it `shrink-0` for exactly
 * that reason) — but at a narrow card width the full `2026-07-31 17:35` leaves
 * the base name almost nothing, and something has to give. The year is what
 * gives: it is the least distinguishing field in it (a collision is between two
 * imports of the same task, which nearly always happened in the same year, and
 * the hh:mm is what actually separates two runs on one day). Dropping it buys
 * ~5 characters for the base. The year stays reachable: dropping it counts as
 * shortening, so the hover title carries the untouched name.
 *
 * The residual risk this trades for that room: two runs exactly a year apart
 * now read identically until hovered. Only the timestamp shapes are touched —
 * an ordinal or namespace suffix has no year to drop and passes through
 * verbatim — and the backend's stored suffix is untouched either way.
 */
export function displayDedupeSuffix(suffix: string): string {
  const match = DEDUPE_DATE_SUFFIX_RE.exec(suffix);
  return match ? match[1] : suffix;
}

// The session stamp a recording appends to the dataset name the user typed
// ("eraser_stack" -> "eraser_stack_20260718_175437"). Peeling it recovers the
// TASK the dataset captured, which is the name the model trained on it wants.
const DATASET_SESSION_STAMP_RE = /_\d{8}_\d{6}$/;

// The run stamp a training run appends (jobs._named_job_id /
// _generate_job_id — mirrors utils/naming.py's RUN_REPO_TIMESTAMP_RE). Peeled
// for the same reason when the proposal comes from an existing model.
const RUN_TIMESTAMP_RE = /_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$/;

/**
 * Propose a model name from what the run is ABOUT — its dataset's repo id, or
 * the model it fine-tunes from — so accepting the default costs no keystrokes.
 *
 * Peels the namespace and whichever machine stamp the source carries, leaving
 * the task stem: `makermods/eraser_stack_20260718_175437` -> `eraser_stack`,
 * `makermods/eraser_stack_2026-08-04_11-02-19` -> `eraser_stack`.
 *
 * Returns "" when the result wouldn't be a legal model name — a display title
 * like "SMOLVLA · user/ds" or a hand-typed "smolvla 5k" is a label, not an
 * identifier. A prefill that the form would immediately mark invalid is worse
 * than an empty field, so the caller falls back (or leaves it to the user).
 */
export function proposeModelName(source: string): string {
  const path = source.trim().replace(/\\/g, "/").replace(/\/+$/, "");
  const basename = path.split("/").pop() ?? "";
  const stem = basename
    .replace(DATASET_SESSION_STAMP_RE, "")
    .replace(RUN_TIMESTAMP_RE, "");
  return validateModelName(stem) === null ? stem : "";
}
