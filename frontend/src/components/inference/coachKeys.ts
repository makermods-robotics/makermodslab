/**
 * The coaching dialog's keyboard decision, as a pure function.
 *
 * Extracted from InferenceSessionDialog's keydown effect so it can be tested.
 * The logic guards a physical robot arm — "does a held key repeat a command",
 * "is this keystroke the operator's or the browser's" — and none of that was
 * reachable by a test while it lived inside a closure over component state.
 *
 * The effect keeps ownership of the SIDE EFFECTS (sending commands, the
 * one-shot explanatory toasts). This decides only what should happen.
 */

import type { CoachingPhase } from "@/lib/inferenceApi";

/**
 * May control change hands RIGHT NOW?
 *
 * Lives here rather than in the dialog because three callers share it — the
 * key, the button's disabled state, and the toggle handler — and because the
 * cost of getting it wrong is a demonstration silently filed as a correction.
 * See `CoachKeyState.controlToggleAllowed` for why the excluded phases are
 * excluded; this is the phase table that produces that flag.
 *
 *   correcting  the operator is driving; the toggle hands back.
 *   poised      the middle of a takeover — the leader is glided onto the
 *               follower and held under torque, and this press is the one that
 *               releases it and starts recording. Allowed unconditionally: the
 *               only way in is a first press this same gate already allowed,
 *               from a phase where the policy was driving the attempt, so the
 *               frames are a correction to that attempt and not a fresh
 *               demonstration. Refusing here would strand the operator holding
 *               a rigid arm with the only key that finishes the takeover dead.
 *   autonomous/paused mid-attempt — the policy is driving, or was frozen while
 *               driving. Parked (`awaitingAttempt`) is NOT this: there the next
 *               move is the next attempt, and a takeover records a plain
 *               teleoperated episode into a corrections dataset.
 *
 * Everything else — `handing_over`, `saving`, `resetting`, and no phase at all
 * — is refused.
 */
export function controlToggleAllowedFor(
  phase: CoachingPhase | null,
  awaitingAttempt: boolean,
): boolean {
  if (phase === "correcting" || phase === "poised") return true;
  return (phase === "autonomous" || phase === "paused") && !awaitingAttempt;
}

export type CoachKeyAction =
  | "takeover-toggle"
  | "hold"
  | "recovered"
  | "advance"
  | "discard"
  | "drop-last"
  | "nothing-to-drop-hint"
  | "escape-hint"
  | null;

/** The subset of KeyboardEvent this decision reads. */
export interface CoachKeyEvent {
  code: string;
  key: string;
  repeat: boolean;
  shiftKey: boolean;
  metaKey: boolean;
  ctrlKey: boolean;
  altKey: boolean;
}

export interface CoachKeyState {
  /** The runner is live and can accept commands. */
  coachLive: boolean;
  /** A command is in flight. */
  coachBusy: boolean;
  /**
   * The keystroke landed on something that handles keys itself — a text field,
   * or a control the browser will activate on Space/Enter. See `decideCoachKey`.
   */
  targetHandlesKey: boolean;
  /**
   * Whether control may change hands RIGHT NOW.
   *
   * False during a reset, a handover, a save, and while parked waiting for the
   * next attempt. Taking over in those phases does not produce a correction:
   * the policy is not driving, so there is no failure to correct and nothing to
   * compare against. What it produces is a plain teleoperated demonstration
   * written into a corrections dataset — the operator recording a fresh episode
   * while believing they are coaching. That is worse than a no-op, because the
   * resulting frames are indistinguishable from real corrections at training
   * time.
   *
   * TRUE while `poised` — the middle of a takeover, where the leader is glided
   * onto the follower and held under torque and the operator's second press is
   * what releases it and starts recording. That is not an exception to the rule
   * above: reaching `poised` at all required a first press this same gate
   * allowed, from a phase where the policy was driving the attempt. Blocking
   * the key there would leave a takeover half-finished with the operator
   * holding a rigid arm.
   */
  controlToggleAllowed: boolean;
  /**
   * The policy is ACTUALLY driving the arm right now (`autonomous`).
   *
   * The only phase in which a freeze means anything: from `paused` the arm is
   * already frozen, and from `correcting` the operator is holding it
   * themselves. Gating on this is what stops Shift+Space being a second name
   * for hand-back and for start-next-attempt.
   */
  policyIsDriving: boolean;
  /**
   * The operator is holding the leader and a correction is being recorded RIGHT
   * NOW (`correcting`).
   *
   * This is the only phase in which Backspace can mean "throw away what is
   * being recorded", because it is the only phase in which something is being
   * recorded. Outside it the same key means "throw away the previous
   * correction" — see the Backspace branch in `decideCoachKey`.
   *
   * `poised` is deliberately NOT this: the takeover has begun but no frame has
   * been kept yet, so there is nothing in flight to discard and the key keeps
   * its delete-the-previous-one meaning there.
   */
  correcting: boolean;
  /**
   * The runner is still holding the previous correction in memory and would
   * honour `drop_last` (`droppable_correction != null` on the status payload).
   *
   * Read straight from the backend, never inferred from the phase: the window
   * opens at the hand-back and closes when the runner commits the correction at
   * the next takeover, and both dagger_protocol.py and inferenceApi.ts say in
   * so many words that the browser must not guess at those edges. Guessing
   * would either hide a delete that is still available or offer one that is
   * already too late.
   */
  hasDroppableCorrection: boolean;
}

export interface CoachKeyDecision {
  preventDefault: boolean;
  stopPropagation: boolean;
  action: CoachKeyAction;
}

const IGNORE: CoachKeyDecision = {
  preventDefault: false,
  stopPropagation: false,
  action: null,
};

/** Swallow the key but do nothing with it. */
const SWALLOW: CoachKeyDecision = {
  preventDefault: true,
  stopPropagation: false,
  action: null,
};

const bare = (e: CoachKeyEvent) => !e.metaKey && !e.ctrlKey && !e.altKey;

export function decideCoachKey(e: CoachKeyEvent, state: CoachKeyState): CoachKeyDecision {
  // Never hijack a keystroke aimed at something that handles it itself: a text
  // field, or any focusable control the browser activates on Space/Enter.
  if (state.targetHandlesKey) return IGNORE;

  // `e.repeat` FIRST, before any command branch.
  //
  // This guard used to sit after the Space branch, so every command key was
  // protected EXCEPT the primary one. `coachBusy` is not a barrier — it clears
  // after each ~50ms round trip — so holding space issued a continuous stream
  // of alternating takeover/handback commands at the OS key-repeat rate, each
  // one a physical handover on a moving arm.
  //
  // Repeats are still SWALLOWED rather than ignored: a held space that fell
  // through to the browser would scroll the dialog.
  if (e.repeat) return SWALLOW;

  if (e.code === "Space") {
    // Always swallowed, even before the runner is live. During the 10-30s
    // startup an operator taught that "space is the whole interaction" will
    // press it at the arm; letting it through would activate whatever the
    // dialog has focused.
    if (!state.coachLive) return SWALLOW;
    if (state.coachBusy) return SWALLOW;

    // Shift+Space is the FREEZE, and it now means one thing only.
    //
    // It used to route through a hold/resume/handback toggle, which made it a
    // second key for something another key already did in two of the three
    // phases: from `correcting` it handed back, exactly like bare Space; from
    // `paused` it resumed the next attempt, exactly like Enter. An operator
    // with their hands on the leader cannot be asked to hold "which of my two
    // space chords applies right now" in their head, and the phases where it
    // duplicated something were the phases they were most likely to be in.
    //
    // What survives is the half nothing else can do: stop the policy while it
    // is driving, without taking over and without opening a correction. That
    // is the one gesture with no other route, so it stays — but it is now
    // inert everywhere else rather than quietly meaning a third thing.
    if (e.shiftKey) {
      return {
        preventDefault: true,
        stopPropagation: false,
        action: state.policyIsDriving ? "hold" : null,
      };
    }

    // Swallowed, not ignored: a stray space during a reset must not fall
    // through to the browser either, and the operator's hands are on the arm.
    if (!state.controlToggleAllowed) return SWALLOW;
    return {
      preventDefault: true,
      stopPropagation: false,
      action: "takeover-toggle",
    };
  }

  if (!state.coachLive) return IGNORE;

  if (e.key.toLowerCase() === "g" && bare(e)) {
    return {
      preventDefault: true,
      stopPropagation: false,
      action: state.coachBusy ? null : "recovered",
    };
  }

  if (e.key === "Enter" && bare(e)) {
    return {
      preventDefault: true,
      stopPropagation: false,
      action: state.coachBusy ? null : "advance",
    };
  }

  if (e.key === "Backspace" || e.key === "Delete") {
    // ONE key, one meaning — "throw away the correction" — but WHICH correction
    // depends on whether one is being recorded right now.
    //
    // The key stays live in every phase, and that part is deliberate: gating it
    // on `correcting` once left discard dead for the 1-3s after a takeover,
    // precisely the window in which a takeover goes wrong. What changed is what
    // it does outside `correcting`. It used to send CMD_CANCEL from every
    // phase, on the belief that the runner no-ops a cancel that does not apply.
    // It does not: the runner accepts CMD_CANCEL from EVERY phase and always
    // ends the attempt, easing the follower across the workspace and releasing
    // it limp. A stray Backspace while the policy was driving therefore moved
    // the arm, with no confirmation and no modifier.
    //
    // So outside `correcting` the key un-records the PREVIOUS correction
    // instead: same intent, and it touches the dataset rather than the arm.
    if (state.coachBusy) return { preventDefault: true, stopPropagation: true, action: null };
    if (state.correcting) {
      return { preventDefault: true, stopPropagation: true, action: "discard" };
    }
    // The runner holds exactly ONE correction, so the second press in a row has
    // nothing to take. That must not be silence: the operator pressed a key
    // meaning "undo that", and an undo that quietly does nothing reads as a
    // dropped keystroke and gets pressed again, harder. Say instead that the
    // earlier corrections are already written and cannot be taken back.
    return {
      preventDefault: true,
      stopPropagation: true,
      action: state.hasDroppableCorrection ? "drop-last" : "nothing-to-drop-hint",
    };
  }

  if (e.key === "Escape") {
    // Swallowed, and it reaches the runner not at all: an Escape arriving at
    // StudioOverlay would close the studio out from under a running policy.
    // The action it returns is a one-shot toast, not a robot command.
    return { preventDefault: true, stopPropagation: true, action: "escape-hint" };
  }

  return IGNORE;
}

/**
 * Does this element handle Space/Enter itself?
 *
 * Text entry, plus anything the browser will activate as a control. The old
 * check listed only INPUT/TEXTAREA/SELECT — a DOM this dialog does not have
 * (its controls are `<button>`, and Radix selects render a
 * `<button role="combobox">`), so every button in the dialog had its Space and
 * Enter swallowed and could not be operated by keyboard at all.
 */
export function targetHandlesKey(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || typeof el.closest !== "function") return false;
  if (el.isContentEditable) return true;
  if (["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName)) return true;
  return (
    el.closest(
      'button,select,textarea,input,a[href],[role="button"],[role="combobox"],' +
        '[role="switch"],[role="menuitem"],[contenteditable="true"],' +
        '[tabindex]:not([tabindex="-1"])',
    ) !== null
  );
}
