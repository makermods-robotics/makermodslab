/**
 * Presentation helpers for model/policy card titles.
 *
 * The derivation itself is the backend's — `derive_imported_title` in
 * makermodslab/utils/naming.py peels an imported model's repo id down to its
 * task segment, and `dedupe_display_names` keeps two of them apart — so a card
 * only has to render the name it is given. What CSS can't do is what lives
 * here: `truncate` always eats the tail, which is the wrong half for a name
 * whose distinguishing text sits at the end.
 */

import { POLICY_TYPE_OPTIONS } from "@/components/training/types";

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
 * The part of a job id that tells two runs apart: its run timestamp.
 *
 * Needed because a display NAME does not tell two runs apart within a resume
 * chain, and cannot: a continuation continues the same model, so every run on
 * the chain carries the same name by design. Anywhere the UI attributes
 * something to its owning run inside a lineage — the checkpoint dropdown, the
 * card hints — the name alone renders identical text for different runs.
 *
 * The id is where the difference lives. `_named_job_id` / `_generate_job_id`
 * (makermodslab/jobs.py) both end in `_%Y-%m-%d_%H-%M-%S`, optionally followed
 * by the `-2`, `-3`, … guard against two runs created in the same second. That
 * tail is returned VERBATIM rather than reformatted into something prettier:
 * it has to stay a literal substring of the id, so what the user reads here
 * matches what the backend's refusals print (`'alias' (id)`) and what they can
 * search the list for. Prettifying it would break that correspondence for a
 * few saved characters.
 *
 * Returns null when the id has no such tail — an imported record, or an id
 * some other tool wrote — so callers can fall back rather than render a
 * confident-looking lie.
 */
const JOB_ID_STAMP_RE = /_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:-\d+)?)$/;

export function jobIdStamp(jobId: string): string | null {
  return JOB_ID_STAMP_RE.exec(jobId)?.[1] ?? null;
}

/**
 * How a run is named when it has to be told apart from its own chain: the
 * display name plus the id's distinguishing tail.
 *
 * Falls back to a middle-ellipsized id when there is no timestamp tail to
 * quote — middle, not end, for the usual reason (see `middleEllipsis`): an
 * id's tail is the half that identifies it.
 */
export function jobRunStamp(jobId: string): string {
  return jobIdStamp(jobId) ?? middleEllipsis(jobId, 24);
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

// The separator jobs.start puts between the two halves of a generated run name
// (`f"{policy_type.upper()} · {dataset_repo_id}"`, jobs.py). A space-padded
// middle dot, not a hyphen — nothing a dataset or policy name contains.
const RUN_NAME_SEPARATOR = " · ";

// The policy vocabulary the peel is gated on. Reuses the picker's own option
// list rather than restating it: every generated name this frontend can meet
// was produced from a `policy_type` the picker offered, so the two lists cannot
// drift apart in the direction that matters. The backend's KNOWN_POLICY_TYPES
// (utils/naming.py) is the wider set — it also knows types lerobot can load but
// the trainer doesn't offer (pi05, wall_x, multi_task_dit) — so a name built
// from one of those falls through unpeeled here. That is the safe direction:
// the name renders exactly as it does today, nothing is mangled.
const KNOWN_POLICY_TYPES = new Set(POLICY_TYPE_OPTIONS.map((o) => o.value));

/**
 * A run's name peeled down to the TASK it learned — the frontend mirror of
 * `_run_identity_name` (makermodslab/models.py), which does this for the model
 * library's cards.
 *
 *   "SMOLVLA · makermods/eraser_place_unblurry_real"  ->  "eraser_place_unblurry_real"
 *
 * Both peeled halves are stated elsewhere on every surface that renders this:
 * the policy by its own chip / meta row, the dataset (namespace included) by
 * its own field. The title line is the widest and the first to truncate, so
 * spending it on facts printed twice is what makes every row of one
 * policy+namespace collapse to the same "SMOLVLA · makermods/era…".
 *
 * Only the GENERATED shape is peeled, and only when the head is a policy type
 * we recognize — a human-typed name that happens to contain " · " keeps every
 * word, and so does a name whose head isn't a policy (the retired
 * "Imported · …" prefix among them). The namespace peel sits behind that same
 * gate, so it can never eat a slash out of a name a person wrote. Callers pass
 * `jobDisplayName(job)`, which prefers a rename outright — an alias reaches
 * here only if the user typed this exact shape themselves, and the gate then
 * hands it back whole.
 *
 * Pure and total: anything unrecognized returns byte-identical, which is what
 * keeps imported records and `{name}_{timestamp}` runs rendering as they do.
 */
export function runTaskTitle(name: string): string {
  const at = name.indexOf(RUN_NAME_SEPARATOR);
  if (at < 0) return name;
  const head = name.slice(0, at);
  const tail = name.slice(at + RUN_NAME_SEPARATOR.length);
  if (!tail || !KNOWN_POLICY_TYPES.has(head.toLowerCase())) return name;
  // The tail is a dataset repo id: keep everything after the namespace. A bare
  // name with no slash is already the task.
  const slash = tail.indexOf("/");
  if (slash < 0) return tail;
  return tail.slice(slash + 1) || tail;
}
