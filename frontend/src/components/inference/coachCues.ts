/**
 * Which audio cue a coaching phase change earns, as a pure function.
 *
 * Audio is the only channel that reaches an operator whose eyes are on the arm,
 * so a cue that says the wrong thing is worse than no cue. Extracted from
 * InferenceSessionDialog so the one case that matters most — a discard — can be
 * tested.
 */

export type PhaseCue = "granted" | "handback" | null;

export interface PhaseCueInput {
  /** The phase last cued, or null if none has been seen yet. */
  previous: string | null;
  /** The phase now. */
  next: string | null;
  /**
   * A discard landed on this same update. The runner reports a discard as
   * `correcting -> paused` plus a new `discard_notice`, which is the same phase
   * transition as an ordinary hand-back.
   */
  discardPending: boolean;
}

export function phaseCue({ previous, next, discardPending }: PhaseCueInput): PhaseCue {
  if (previous === next) return null;
  if (next === "correcting") return "granted";
  if (previous === "correcting" && next != null) {
    // A discard is NOT a hand-back.
    //
    // Both leave `correcting` for `paused`, so the handback cue fired first and
    // the discard thud followed. The handback tone falls 660->440 — it reads as
    // "the policy has it back, all is well" — so the destructive event's audio
    // signature OPENED with the reassuring one. In a workshop the operator
    // hears the first 200ms and stops listening, and the correction they just
    // threw away is unrecoverable.
    if (discardPending) return null;
    return "handback";
  }
  return null;
}
