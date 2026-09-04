/**
 * Deciding what the Deploy panel may say about a checkpoint's training task.
 *
 * Extracted from DeployPanel for the same reason as deployGuards: the rules
 * here decide what sentence is sent to a physical robot — and, on a coaching
 * run, what is written into every frame of the recorded dataset — and none of
 * it was reachable by a test while it lived inside an effect.
 *
 * The governing rule is that the panel must never assert something it does not
 * know. Three situations used to collapse into one message reading "No task
 * found on the training dataset": a dataset that genuinely has no task, a Hub
 * summary whose tasks were never fetched, and a lookup that failed outright.
 * Only the first of those is what that sentence claims.
 */

import { ApiError } from "@/lib/apiClient";
import type { DatasetInfo, DatasetTask } from "@/lib/replayApi";

/** What the panel knows about the training dataset's tasks.
 *
 * `unknown` is the state that did not exist before: the lookup did not produce
 * an answer, so the panel must ask rather than claim. Its `reason` is carried
 * so the copy can say WHICH way it failed — "not on this machine" and "couldn't
 * reach it" send an operator to different places. */
export type TaskPrefillState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "loaded"; tasks: string[] }
  | { kind: "unknown"; reason: "not_found" | "unreachable" };

/** How long one dot stays on screen while the lookup is in flight. */
export const TASK_LOADING_DOT_MS = 400;

/** How long the animation runs before the field stops saying "loading" and
 * invites the operator to type instead.
 *
 * A lookup that has not answered in this long is either a slow Hub round-trip
 * or never coming back, and either way the operator should not be left watching
 * dots — they can type the sentence faster than the wait. The request is NOT
 * abandoned: if it lands afterwards its answer replaces whatever is on screen,
 * because a real task is always better than a prompt to invent one. */
export const TASK_LOADING_MAX_MS = 8000;

/** The trailing dots of the loading placeholder: one, two, three, then back to
 * one. A count rather than a string so the caller owns the word being animated
 * and this stays a pure function of the tick. */
export function loadingDots(tick: number): string {
  return ".".repeat((tick % 3) + 1);
}

/** The words the placeholder cycles through while it waits.
 *
 * "Loading" comes first and is on screen for the whole first dot cycle, so the
 * field states what it is doing before it starts having fun. The rest are all
 * rummaging-for-something verbs, which is literally the job: going through a
 * dataset's metadata looking for the sentence it was recorded under.
 *
 * Spelled as FULL literal key paths rather than assembled from a suffix,
 * deliberately: i18n/keyUsage.test.ts scans the source for key-shaped literals
 * and asserts each resolves in every language, so writing them out means a
 * missing translation is a test failure instead of a raw key path rendered at
 * an operator. */
export const TASK_LOADING_WORD_KEYS = [
  "studio.deploy.task.loading.loading",
  "studio.deploy.task.loading.rummaging",
  "studio.deploy.task.loading.digging",
  "studio.deploy.task.loading.foraging",
  "studio.deploy.task.loading.excavating",
  "studio.deploy.task.loading.spelunking",
  "studio.deploy.task.loading.ferreting",
  "studio.deploy.task.loading.scrounging",
] as const;

/** Which word is on screen at `tick`.
 *
 * One word per full dot cycle (three ticks), so the dots animate visibly
 * underneath a word that changes about once a second — fast enough to read as
 * alive, slow enough to actually read. Wraps, so a wait longer than the list
 * loops rather than running out. */
export function loadingWordKey(tick: number): (typeof TASK_LOADING_WORD_KEYS)[number] {
  const index = Math.floor(tick / 3) % TASK_LOADING_WORD_KEYS.length;
  return TASK_LOADING_WORD_KEYS[index];
}

/** Order tasks most-represented first, but ONLY when every count is known.
 *
 * `num_episodes` is null for "unknown" (unreadable episode metadata, or a Hub
 * summary that never fetched it — see makermodslab/datasets.py). Coercing that
 * to 0 would let an unreadable file decide the ranking while every number still
 * looked plausible, and the real margins are thin: the merged datasets this was
 * measured against separate two near-identical task strings by 99 vs 100
 * episodes. When any count is unknown the server's own order (task_index) is
 * kept — not a ranking, but an order the data actually has.
 *
 * Does not mutate its input. */
export function rankDatasetTasks(tasks: DatasetTask[]): string[] {
  const known = tasks.every((t) => t.num_episodes !== null);
  const ordered = known
    ? [...tasks].sort((a, b) => (b.num_episodes ?? 0) - (a.num_episodes ?? 0))
    : tasks;
  return ordered.map((t) => t.task).filter(Boolean);
}

/** Classify what came back from `getDatasetInfo`.
 *
 * A 404 is a dataset that is not on this machine and not resolvable on the Hub
 * — deleted, renamed, or never downloaded. Anything else thrown is a transport
 * or server failure. Neither is "this dataset has no task", which is what the
 * old bare catch reported. */
export function classifyTaskLookup(info: DatasetInfo): TaskPrefillState;
export function classifyTaskLookup(error: unknown, failed: true): TaskPrefillState;
export function classifyTaskLookup(
  value: DatasetInfo | unknown,
  failed?: true,
): TaskPrefillState {
  if (failed) {
    const notFound = value instanceof ApiError && value.status === 404;
    return { kind: "unknown", reason: notFound ? "not_found" : "unreachable" };
  }
  const info = value as DatasetInfo;
  const tasks = rankDatasetTasks(info.tasks ?? []);
  // A Hub summary carries its task STRINGS but never its counts, so an empty
  // list there is a real "this dataset lists no task", same as local — the
  // server already tried. Nothing to special-case: `loaded` with no tasks is
  // the honest answer for both sources.
  return { kind: "loaded", tasks };
}

/** The tasks the panel may offer. Empty for every non-`loaded` state, so
 * neither a failed lookup nor one still in flight can contribute a default. */
export function tasksFrom(state: TaskPrefillState): string[] {
  return state.kind === "loaded" ? state.tasks : [];
}

/** The task suggested when the operator leaves the box empty.
 *
 * Empty when the dataset offers SEVERAL, which is the point: with more than one
 * candidate there is no defensible guess, and sending one silently is how a
 * coaching dataset ends up labelled with a sentence nobody chose. The operator
 * picks from the chips instead (see deployGuards' taskAmbiguous). */
export function defaultTaskFrom(state: TaskPrefillState): string {
  const tasks = tasksFrom(state);
  return tasks.length === 1 ? tasks[0] : "";
}

/** Whether the operator is being asked to choose between several tasks and has
 * not yet done so. Drives both the guard and the hint copy. */
export function taskIsAmbiguous(state: TaskPrefillState, typed: string): boolean {
  return typed.trim() === "" && tasksFrom(state).length > 1;
}

/** Whether the task input is on screen.
 *
 * The field renders for a language-conditioned policy (the string steers it) or
 * a coaching run (the string is saved with every correction). Anywhere else the
 * operator never sees it — which is why the launch payload must not carry a
 * value from here either. */
export function taskFieldVisible(
  requiresTask: boolean,
  runMode: "single" | "eval" | "coach",
): boolean {
  return requiresTask || runMode === "coach";
}

/** The task actually sent with a launch.
 *
 * Empty whenever the field was not on screen: a value the operator could not
 * see, could not confirm and could not correct has no business reaching
 * lerobot's `--task=`. For the modes that DO show it, a typed sentence wins and
 * an empty box falls back to the single unambiguous suggestion. */
export function effectiveTaskFor(
  typed: string,
  state: TaskPrefillState,
  requiresTask: boolean,
  runMode: "single" | "eval" | "coach",
): string {
  if (!taskFieldVisible(requiresTask, runMode)) return "";
  return typed.trim() || defaultTaskFrom(state);
}
