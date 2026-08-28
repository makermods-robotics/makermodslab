import { describe, expect, it } from "vitest";

import { decideCoachKey, targetHandlesKey } from "./coachKeys";
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
      "hold-toggle",
    );
    expect(decideCoachKey(key({ key: "Enter" }), live).action).toBe("advance");
    expect(decideCoachKey(key({ key: "g" }), live).action).toBe("recovered");
    expect(decideCoachKey(key({ key: "Backspace" }), live).action).toBe(
      "discard",
    );
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

  it("stops propagation on discard so Backspace cannot also navigate back", () => {
    expect(
      decideCoachKey(key({ key: "Backspace" }), live).stopPropagation,
    ).toBe(true);
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
  const parked: CoachKeyState = { ...live, controlToggleAllowed: false };

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

  it("still lets the operator discard and mark recovery", () => {
    expect(decideCoachKey(key({ key: "Backspace" }), parked).action).toBe(
      "discard",
    );
    expect(decideCoachKey(key({ key: "g" }), parked).action).toBe("recovered");
  });
});
