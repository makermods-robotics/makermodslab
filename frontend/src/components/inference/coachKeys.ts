/**
 * The coaching dialog's keyboard decision, as a pure function.
 *
 * Extracted from InferenceSessionDialog's keydown effect so it can be tested.
 * The logic guards a physical robot arm — "does a held key repeat a command",
 * "is this keystroke the operator's or the browser's" — and none of that was
 * reachable by a test while it lived inside a closure over component state.
 *
 * The effect keeps ownership of the SIDE EFFECTS (sending commands, the
 * one-shot Escape toast). This decides only what should happen.
 */

export type CoachKeyAction =
  | "takeover-toggle"
  | "hold-toggle"
  | "recovered"
  | "advance"
  | "discard"
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
   */
  controlToggleAllowed: boolean;
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
    // Swallowed, not ignored: a stray space during a reset must not fall
    // through to the browser either, and the operator's hands are on the arm.
    if (!state.controlToggleAllowed) return SWALLOW;
    return {
      preventDefault: true,
      stopPropagation: false,
      action: e.shiftKey ? "hold-toggle" : "takeover-toggle",
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
    // Always sent, never gated on phase: the runner no-ops a CANCEL that does
    // not apply, and gating left discard dead for 1-3s after a takeover —
    // precisely the window in which a takeover goes wrong.
    return {
      preventDefault: true,
      stopPropagation: true,
      action: state.coachBusy ? null : "discard",
    };
  }

  if (e.key === "Escape") {
    // Swallowed and inert: an Escape reaching StudioOverlay would close the
    // studio out from under a running policy.
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
