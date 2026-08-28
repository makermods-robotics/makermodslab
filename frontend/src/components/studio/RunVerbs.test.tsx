import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import { RunVerbs } from "./DeployPanel";

type Props = React.ComponentProps<typeof RunVerbs>;

const setup = (over: Partial<Props> = {}) => {
  const onArm = vi.fn();
  const onLaunch = vi.fn();
  render(
    <RunVerbs
      active="single"
      onArm={onArm}
      onLaunch={onLaunch}
      blockedReason={() => null}
      ready
      busy={false}
      counts={{ eval: 10, coach: 10 }}
      {...over}
    />,
  );
  return { onArm, onLaunch };
};

const coachButton = () => screen.getByRole("button", { name: /Coach it/i });

describe("a blocked run verb can still be armed", () => {
  // THE deadlock. Coaching is blocked while the task is empty, but the task
  // field only renders once coach is ARMED. While the button was `disabled`,
  // `disabled:pointer-events-none` swallowed hover and focus, so it could not
  // be armed — and coaching became unreachable with no route out.
  const blockedCoach = (mode: string) =>
    mode === "coach" ? "Describe the task first — it's saved with every correction." : null;

  it("arms on hover even while blocked, so its fields can be reached", () => {
    const { onArm } = setup({ blockedReason: blockedCoach });
    fireEvent.mouseEnter(coachButton());
    expect(onArm).toHaveBeenCalledWith("coach");
  });

  it("arms on focus, so a keyboard operator is not shut out either", () => {
    const { onArm } = setup({ blockedReason: blockedCoach });
    fireEvent.focus(coachButton());
    expect(onArm).toHaveBeenCalledWith("coach");
  });

  it("is still reachable by keyboard (not removed from the tab order)", () => {
    setup({ blockedReason: blockedCoach });
    expect(coachButton()).not.toHaveAttribute("disabled");
    expect(coachButton()).toHaveAttribute("aria-disabled", "true");
  });

  it("carries its reason where it can actually be read", () => {
    setup({ blockedReason: blockedCoach });
    expect(coachButton()).toHaveAttribute("title", expect.stringContaining("Describe the task"));
  });

  it("refuses to LAUNCH while blocked — clicking only arms", () => {
    const { onArm, onLaunch } = setup({ blockedReason: blockedCoach });
    fireEvent.click(coachButton());
    expect(onLaunch).not.toHaveBeenCalled();
    expect(onArm).toHaveBeenCalledWith("coach");
  });

  it("leaves unblocked verbs launchable", () => {
    const { onLaunch } = setup({ blockedReason: blockedCoach });
    fireEvent.click(screen.getByRole("button", { name: /Just run it/i }));
    expect(onLaunch).toHaveBeenCalledWith("single");
  });
});

describe("not-ready and busy states are equally non-inert", () => {
  it("still arms while the panel is not ready", () => {
    const { onArm, onLaunch } = setup({ ready: false });
    fireEvent.click(coachButton());
    expect(onLaunch).not.toHaveBeenCalled();
    expect(onArm).toHaveBeenCalledWith("coach");
  });

  it("does not launch a second run while one is starting", () => {
    const { onLaunch } = setup({ busy: true });
    fireEvent.click(coachButton());
    expect(onLaunch).not.toHaveBeenCalled();
  });
});

describe("armed state is exposed to assistive tech", () => {
  it("marks the active verb pressed and the others not", () => {
    setup({ active: "coach" });
    expect(coachButton()).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /Just run it/i })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
  });
});

describe("Score it is a normal verb", () => {
  // The scored-evaluation engine it launches shipped in #63; the verb is just
  // the launch control for it. It carries no WIP badge, no shield, and no
  // tab-out — same as the other two.
  const evalButton = () => screen.getByRole("button", { name: /Score it/i });

  it("wears no WIP badge", () => {
    setup();
    expect(evalButton()).not.toHaveTextContent("WIP");
  });

  it("has no shield over it", () => {
    setup();
    expect(screen.queryByTestId("wip-shield")).toBeNull();
  });

  it("stays in the tab order", () => {
    setup();
    expect(evalButton()).not.toHaveAttribute("tabindex");
  });

  it("launches on click when nothing blocks it", () => {
    const { onLaunch } = setup();
    fireEvent.click(evalButton());
    expect(onLaunch).toHaveBeenCalledWith("eval");
  });

  it("arms rather than launches while blocked", () => {
    const { onArm, onLaunch } = setup({
      blockedReason: (m) =>
        m === "eval" ? "Bind every camera the checkpoint expects." : null,
    });
    fireEvent.click(evalButton());
    expect(onLaunch).not.toHaveBeenCalled();
    expect(onArm).toHaveBeenCalledWith("eval");
  });
});
