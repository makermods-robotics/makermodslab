import { describe, expect, it } from "vitest";

import {
  controlToggleAllowedFor,
  decideCoachKey,
  targetHandlesKey,
} from "./coachKeys";
import type { CoachKeyEvent, CoachKeyState } from "./coachKeys";

const key = (over: Partial<CoachKeyEvent> = {}): CoachKeyEvent => ({
  code: "",
  key: "",
  repeat: false,
  shiftKey: false,
  metaKey: false,
  ctrlKey: false,
  altKey: false,
  ...over,
});

const live: CoachKeyState = {
  coachLive: true,
  coachBusy: false,
  targetHandlesKey: false,
  controlToggleAllowed: true,
  policyIsDriving: true,
  correcting: false,
  hasDroppableCorrection: true,
};

const SPACE = key({ code: "Space", key: " " });

describe("held keys never repeat a command", () => {
  // THE regression. The `e.repeat` guard used to sit after the Space branch,
  // so the primary control was the one key it did not protect. Holding space
  // issued alternating takeover/handback at the OS repeat rate, each one a
  // physical handover on a moving arm.
  it("does not act on a repeated space", () => {
    const decision = decideCoachKey({ ...SPACE, repeat: true }, live);
    expect(decision.action).toBeNull();
  });

  it("still swallows a repeated space so it cannot scroll the dialog", () => {
    expect(
      decideCoachKey({ ...SPACE, repeat: true }, live).preventDefault,
    ).toBe(true);
  });

  it.each([
    ["shift+space", { ...SPACE, shiftKey: true }],
    ["enter", key({ key: "Enter" })],
    ["g", key({ key: "g" })],
    ["backspace", key({ key: "Backspace" })],
  ])("does not act on a repeated %s", (_label, ev) => {
    expect(decideCoachKey({ ...ev, repeat: true }, live).action).toBeNull();
  });

  it("acts on the FIRST press of each command key", () => {
    expect(decideCoachKey(SPACE, live).action).toBe("takeover-toggle");
    expect(decideCoachKey({ ...SPACE, shiftKey: true }, live).action).toBe(
      "hold",
    );
    expect(decideCoachKey(key({ key: "Enter" }), live).action).toBe("advance");
    expect(decideCoachKey(key({ key: "g" }), live).action).toBe("recovered");
    expect(
      decideCoachKey(key({ key: "Backspace" }), {
        ...live,
        correcting: true,
      }).action,
    ).toBe("discard");
  });
});

describe("keystrokes aimed at a control are left to the browser", () => {
  // Every control in this dialog is a <button>; the old guard listed only
  // INPUT/TEXTAREA/SELECT, so Space and Enter were swallowed for all of them
  // and no button was operable by keyboard while a session was live.
  const onControl: CoachKeyState = { ...live, targetHandlesKey: true };

  it.each([
    ["space", SPACE],
    ["enter", key({ key: "Enter" })],
    ["backspace", key({ key: "Backspace" })],
    ["escape", key({ key: "Escape" })],
  ])("never preventDefaults %s when focus is on a control", (_label, ev) => {
    const decision = decideCoachKey(ev, onControl);
    expect(decision.preventDefault).toBe(false);
    expect(decision.action).toBeNull();
  });
});

describe("startup: keys are swallowed, not acted on", () => {
  const starting: CoachKeyState = {
    coachLive: false,
    coachBusy: false,
    targetHandlesKey: false,
    controlToggleAllowed: false,
    policyIsDriving: false,
    correcting: false,
    hasDroppableCorrection: false,
  };

  it("swallows space before the runner is live so it cannot activate Stop", () => {
    const decision = decideCoachKey(SPACE, starting);
    expect(decision.preventDefault).toBe(true);
    expect(decision.action).toBeNull();
  });

  it("ignores the other command keys entirely before the runner is live", () => {
    expect(decideCoachKey(key({ key: "Enter" }), starting).action).toBeNull();
    expect(decideCoachKey(key({ key: "g" }), starting).action).toBeNull();
  });
});

describe("a command in flight blocks a second one", () => {
  const busy: CoachKeyState = { ...live, coachBusy: true };

  it.each([
    ["space", SPACE],
    ["enter", key({ key: "Enter" })],
    ["g", key({ key: "g" })],
    ["backspace", key({ key: "Backspace" })],
  ])("swallows %s but issues nothing while busy", (_label, ev) => {
    const decision = decideCoachKey(ev, busy);
    expect(decision.preventDefault).toBe(true);
    expect(decision.action).toBeNull();
  });
});

describe("modifiers and unrelated keys", () => {
  it.each([
    ["meta", { metaKey: true }],
    ["ctrl", { ctrlKey: true }],
    ["alt", { altKey: true }],
  ])("leaves %s+enter alone so browser shortcuts survive", (_label, mod) => {
    expect(
      decideCoachKey(key({ key: "Enter", ...mod }), live).action,
    ).toBeNull();
    expect(
      decideCoachKey(key({ key: "Enter", ...mod }), live).preventDefault,
    ).toBe(false);
  });

  it("ignores keys it does not own", () => {
    expect(decideCoachKey(key({ key: "k" }), live)).toEqual({
      preventDefault: false,
      stopPropagation: false,
      action: null,
    });
  });

  it("swallows Escape and reports it, so the studio cannot close mid-session", () => {
    const decision = decideCoachKey(key({ key: "Escape" }), live);
    expect(decision).toEqual({
      preventDefault: true,
      stopPropagation: true,
      action: "escape-hint",
    });
  });

  it("stops propagation on Backspace so it cannot also navigate back", () => {
    // In every phase, and whether or not it resolves to a command: a Backspace
    // that reached the browser would be a history-back out of a live session.
    for (const correcting of [true, false]) {
      for (const hasDroppableCorrection of [true, false]) {
        expect(
          decideCoachKey(key({ key: "Backspace" }), {
            ...live,
            correcting,
            hasDroppableCorrection,
          }).stopPropagation,
        ).toBe(true);
      }
    }
  });
});

describe("targetHandlesKey", () => {
  it("detects a button, including a click on a child of one", () => {
    const button = document.createElement("button");
    const span = document.createElement("span");
    button.appendChild(span);
    document.body.appendChild(button);
    expect(targetHandlesKey(button)).toBe(true);
    expect(targetHandlesKey(span)).toBe(true);
    button.remove();
  });

  it("detects a Radix select trigger, which is a button with role=combobox", () => {
    const el = document.createElement("button");
    el.setAttribute("role", "combobox");
    document.body.appendChild(el);
    expect(targetHandlesKey(el)).toBe(true);
    el.remove();
  });

  it("detects text entry", () => {
    const input = document.createElement("input");
    document.body.appendChild(input);
    expect(targetHandlesKey(input)).toBe(true);
    input.remove();
  });

  it("is false for the dialog body, where the hands-on operator's keys land", () => {
    const div = document.createElement("div");
    document.body.appendChild(div);
    expect(targetHandlesKey(div)).toBe(false);
    div.remove();
  });

  it("is false for a programmatically-focused container (tabindex -1)", () => {
    const div = document.createElement("div");
    div.setAttribute("tabindex", "-1");
    document.body.appendChild(div);
    expect(targetHandlesKey(div)).toBe(false);
    div.remove();
  });

  it("is false for null", () => {
    expect(targetHandlesKey(null)).toBe(false);
  });
});

describe("control cannot change hands outside the phases that mean it", () => {
  // Taking over during a reset, a handover, a save, or while parked is not a
  // correction: the policy is not driving, so there is nothing being corrected.
  // It records a plain teleoperated demonstration into a corrections dataset,
  // which is indistinguishable from a real correction at training time.
  // Parked means the policy is not driving either — the two travel together,
  // and a state with `policyIsDriving` still true does not exist in the session.
  const parked: CoachKeyState = {
    ...live,
    controlToggleAllowed: false,
    policyIsDriving: false,
  };

  it("swallows space rather than taking over", () => {
    const decision = decideCoachKey(SPACE, parked);
    expect(decision.action).toBeNull();
    expect(decision.preventDefault).toBe(true);
  });

  it("swallows shift+space too — freezing a parked arm means nothing", () => {
    expect(
      decideCoachKey({ ...SPACE, shiftKey: true }, parked).action,
    ).toBeNull();
  });

  it("still lets Enter through, which is the only move that IS valid there", () => {
    expect(decideCoachKey(key({ key: "Enter" }), parked).action).toBe(
      "advance",
    );
  });

  it("still lets the operator delete the last correction and mark recovery", () => {
    // Backspace outside `correcting` unwinds the PREVIOUS correction rather
    // than cancelling the attempt: there is nothing in flight to cancel here,
    // and CMD_CANCEL from this phase would move the arm.
    expect(decideCoachKey(key({ key: "Backspace" }), parked).action).toBe(
      "drop-last",
    );
    expect(decideCoachKey(key({ key: "g" }), parked).action).toBe("recovered");
  });
});


describe("shift+space is the freeze, and ONLY the freeze", () => {
  // The duplicate-key complaint, pinned. Shift+Space used to be a toggle that
  // resolved to hand-back from `correcting` (bare Space already did that) and
  // to resume from `paused` (Enter already did that), so the same chord meant
  // three different things depending on a phase the operator — hands on the
  // leader, eyes on the arm — could not see.

  it("freezes while the policy is driving", () => {
    expect(decideCoachKey({ ...SPACE, shiftKey: true }, live).action).toBe(
      "hold",
    );
  });

  it("does NOTHING once the policy is not driving", () => {
    // `paused` and `correcting` both land here. This is the assertion that
    // fails the moment anyone reintroduces the toggle.
    const notDriving: CoachKeyState = { ...live, policyIsDriving: false };
    expect(decideCoachKey({ ...SPACE, shiftKey: true }, notDriving).action).toBeNull();
  });

  it("still swallows the keystroke when it does nothing", () => {
    // Inert is not the same as ignored: a space that fell through to the
    // browser would scroll the dialog or press whatever it has focused, and
    // the operator's hands are on the arm rather than the keyboard.
    const notDriving: CoachKeyState = { ...live, policyIsDriving: false };
    expect(
      decideCoachKey({ ...SPACE, shiftKey: true }, notDriving).preventDefault,
    ).toBe(true);
  });

  it("never resolves to takeover-toggle, whatever the phase", () => {
    for (const policyIsDriving of [true, false]) {
      for (const controlToggleAllowed of [true, false]) {
        const state: CoachKeyState = {
          ...live,
          policyIsDriving,
          controlToggleAllowed,
        };
        expect(
          decideCoachKey({ ...SPACE, shiftKey: true }, state).action,
        ).not.toBe("takeover-toggle");
      }
    }
  });
});

describe("Backspace throws away a correction — which one depends on the phase", () => {
  // Backspace used to send CMD_CANCEL from every phase, on a comment claiming
  // the runner no-ops a cancel that does not apply. It does not: the runner
  // accepts CMD_CANCEL everywhere and always ends the attempt, driving the
  // follower home and dropping it limp. So a stray press while the policy was
  // driving MOVED THE ARM, with no confirmation and no modifier.
  const correcting: CoachKeyState = { ...live, correcting: true };

  it.each([
    ["backspace", key({ key: "Backspace" })],
    ["delete", key({ key: "Delete" })],
  ])("discards the in-flight correction on %s while correcting", (_l, ev) => {
    expect(decideCoachKey(ev, correcting).action).toBe("discard");
  });

  it.each([
    ["backspace", key({ key: "Backspace" })],
    ["delete", key({ key: "Delete" })],
  ])("deletes the PREVIOUS correction on %s outside correcting", (_l, ev) => {
    expect(decideCoachKey(ev, live).action).toBe("drop-last");
  });

  it("never cancels the attempt from a phase with nothing in flight", () => {
    // The assertion that fails the moment anyone restores the old behaviour.
    // `discard` outside `correcting` degrades to a plain reset in the runner,
    // which is a moving arm rather than a no-op.
    for (const hasDroppableCorrection of [true, false]) {
      for (const policyIsDriving of [true, false]) {
        const state: CoachKeyState = {
          ...live,
          correcting: false,
          hasDroppableCorrection,
          policyIsDriving,
        };
        expect(decideCoachKey(key({ key: "Backspace" }), state).action).not.toBe(
          "discard",
        );
      }
    }
  });

  it("explains itself rather than no-opping when nothing is droppable", () => {
    // Only ONE correction is held, so a second press in a row has nothing to
    // take. Silence there reads as a dropped keystroke and gets pressed again;
    // the operator has to be told the earlier ones are already written.
    const nothingHeld: CoachKeyState = {
      ...live,
      hasDroppableCorrection: false,
    };
    expect(decideCoachKey(key({ key: "Backspace" }), nothingHeld).action).toBe(
      "nothing-to-drop-hint",
    );
  });

  it("swallows the key in every phase, droppable or not", () => {
    for (const correctingNow of [true, false]) {
      for (const hasDroppableCorrection of [true, false]) {
        const decision = decideCoachKey(key({ key: "Backspace" }), {
          ...live,
          correcting: correctingNow,
          hasDroppableCorrection,
        });
        expect(decision.preventDefault).toBe(true);
        expect(decision.stopPropagation).toBe(true);
      }
    }
  });

  it("still cannot be fired by a held key, in either meaning", () => {
    // The `e.repeat` guard sits ahead of every command branch and stays there:
    // its absence once let a held key issue a command stream at the OS repeat
    // rate. A held Backspace must not delete a run of corrections either.
    for (const correctingNow of [true, false]) {
      const decision = decideCoachKey(
        { ...key({ key: "Backspace" }), repeat: true },
        { ...live, correcting: correctingNow },
      );
      expect(decision.action).toBeNull();
      expect(decision.preventDefault).toBe(true);
    }
  });

  it("issues nothing at all while a command is in flight", () => {
    for (const correctingNow of [true, false]) {
      for (const hasDroppableCorrection of [true, false]) {
        expect(
          decideCoachKey(key({ key: "Backspace" }), {
            ...live,
            coachBusy: true,
            correcting: correctingNow,
            hasDroppableCorrection,
          }).action,
        ).toBeNull();
      }
    }
  });
});

describe("poised: the second press of a two-press takeover", () => {
  // Takeover used to be one press: the policy stopped, the leader was glided
  // onto the follower, and control passed in the same instant. When the glide
  // raised — which it did on the first takeover of a real session — the offset
  // meant to absorb the gap walked the FOLLOWER 114 degrees across the
  // workspace to meet a leader that had never moved. `poised` is the stop
  // between the two halves: both arms still, leader held under torque, nothing
  // recorded, waiting for the operator to take hold and press again.
  const poised: CoachKeyState = {
    ...live,
    // What the dialog computes for this phase, via controlToggleAllowedFor.
    controlToggleAllowed: controlToggleAllowedFor("poised", false),
    // The policy is stopped: this is mid-takeover, not mid-attempt.
    policyIsDriving: false,
    // Nothing is being recorded yet — that starts on the press below.
    correcting: false,
  };

  it("lets space through — it is the press that finishes the takeover", () => {
    expect(decideCoachKey(SPACE, poised).action).toBe("takeover-toggle");
  });

  it("does not repeat the press when the key is held", () => {
    const decision = decideCoachKey({ ...SPACE, repeat: true }, poised);
    expect(decision.action).toBeNull();
    expect(decision.preventDefault).toBe(true);
  });

  it("keeps shift+space inert — the policy is already stopped", () => {
    expect(
      decideCoachKey({ ...SPACE, shiftKey: true }, poised).action,
    ).toBeNull();
  });

  it("never discards on Backspace: there is nothing in flight to discard", () => {
    // The takeover has begun but no frame has been kept, so the key keeps its
    // outside-`correcting` meaning — un-record the PREVIOUS correction, which
    // touches the dataset rather than the arm.
    for (const ev of [key({ key: "Backspace" }), key({ key: "Delete" })]) {
      expect(decideCoachKey(ev, poised).action).not.toBe("discard");
    }
    expect(decideCoachKey(key({ key: "Backspace" }), poised).action).toBe(
      "drop-last",
    );
    expect(
      decideCoachKey(key({ key: "Backspace" }), {
        ...poised,
        hasDroppableCorrection: false,
      }).action,
    ).toBe("nothing-to-drop-hint");
  });
});

describe("controlToggleAllowedFor", () => {
  // The gate that decides whether a keystroke can hand control over. Its whole
  // reason for existing: taking over in a phase where the policy is not driving
  // records a plain teleoperated demonstration into a corrections dataset, and
  // at training time those frames cannot be told apart from real corrections.
  it("allows the two-press takeover: the first press and the second", () => {
    expect(controlToggleAllowedFor("autonomous", false)).toBe(true);
    expect(controlToggleAllowedFor("poised", false)).toBe(true);
  });

  it("allows a hand-back mid-correction, and a takeover from a mid-attempt hold", () => {
    expect(controlToggleAllowedFor("correcting", false)).toBe(true);
    expect(controlToggleAllowedFor("paused", false)).toBe(true);
  });

  it.each(["resetting", "handing_over", "saving"] as const)(
    "refuses %s, where a takeover would record a demonstration, not a correction",
    (phase) => {
      expect(controlToggleAllowedFor(phase, false)).toBe(false);
      // And the key is swallowed rather than left to the browser: the
      // operator's hands are on the arm, not the keyboard.
      const decision = decideCoachKey(SPACE, {
        ...live,
        controlToggleAllowed: controlToggleAllowedFor(phase, false),
        policyIsDriving: false,
      });
      expect(decision.action).toBeNull();
      expect(decision.preventDefault).toBe(true);
    },
  );

  it("refuses while parked, where the next move is the next attempt", () => {
    expect(controlToggleAllowedFor("paused", true)).toBe(false);
    expect(controlToggleAllowedFor("autonomous", true)).toBe(false);
  });

  it("refuses before the runner has reported any phase at all", () => {
    expect(controlToggleAllowedFor(null, false)).toBe(false);
  });
});
