/**
 * The session dialog's FRAME palette — the bits every live run wears, whichever
 * machine the policy is running on.
 *
 * A local rollout and a remote (DRTC) run are the same session to the operator:
 * a status pill, a big timer, the policy line, one full-width Stop, and a phase
 * line under it. They differ only in what fills the middle — a coaching tally
 * for one, the GPU's telemetry for the other — and in which status endpoint
 * they read.
 *
 * Extracted here so the two bodies cannot drift apart cosmetically. Nothing in
 * this file knows about either status shape: the caller resolves a tone and a
 * label and hands them over. (Values only, no components — a .ts on purpose, so
 * importing a colour map does not cost the importer its fast refresh.)
 */

/** Tone → dot colour. The dot is the only always-visible carrier of state on a
 * phase line, so it never appears without the word beside it. */
export const PHASE_DOT: Record<"amber" | "green" | "red", string> = {
  amber: "bg-warn",
  green: "bg-ok",
  red: "bg-destructive",
};

export const PHASE_TEXT: Record<"amber" | "green" | "red", string> = {
  amber: "text-warn",
  green: "text-ok",
  red: "text-destructive",
};

/** Pill (status chip) background + text per tone. Mirrors the dot/text maps so
 * the finished-failed/warning states reuse the same palette as the phases. */
export const PILL_BG: Record<"amber" | "green" | "red", string> = {
  amber: "bg-warn/15 text-warn",
  green: "bg-ok/15 text-ok",
  red: "bg-destructive/15 text-destructive",
};

/** mm:ss. Never localized: it is a clock reading, and the digits and the colon
 * are the same in every language this app ships. */
export function formatTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}
