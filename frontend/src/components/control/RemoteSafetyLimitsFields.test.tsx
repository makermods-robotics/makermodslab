import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import RemoteSafetyLimitsFields, {
  type RemoteSafetyLimits,
} from "./RemoteSafetyLimitsFields";

const initial: RemoteSafetyLimits = {
  action_watchdog_ms: 200,
  first_action_deadline_ms: 1000,
  control_deadline_ms: 1000,
  browser_deadline_ms: 2000,
  max_velocity_per_s: 60,
  max_acceleration_per_s2: 300,
};

function Harness() {
  const [value, setValue] = useState(initial);
  return <RemoteSafetyLimitsFields value={value} onChange={setValue} />;
}

describe("RemoteSafetyLimitsFields", () => {
  it("shows bounded inputs and keeps dependent deadlines ordered", () => {
    render(<Harness />);

    const action = screen.getByLabelText("Action watchdog (ms)");
    const first = screen.getByLabelText("First action deadline (ms)");
    const control = screen.getByLabelText("Control deadline (ms)");
    const browser = screen.getByLabelText("Browser deadline (ms)");
    const velocity = screen.getByLabelText(
      "Max velocity (position units/s)",
    );
    const acceleration = screen.getByLabelText(
      "Max acceleration (position units/s²)",
    );

    expect(action).toHaveAttribute("min", "20");
    expect(action).toHaveAttribute("max", "2000");
    expect(first).toHaveAttribute("min", "200");
    expect(first).toHaveAttribute("max", "5000");
    expect(control).toHaveAttribute("min", "100");
    expect(control).toHaveAttribute("max", "5000");
    expect(browser).toHaveAttribute("min", "1000");
    expect(browser).toHaveAttribute("max", "10000");
    expect(velocity).toHaveAttribute("min", "0.01");
    expect(velocity).toHaveAttribute("max", "10000");
    expect(acceleration).toHaveAttribute("min", "0.01");
    expect(acceleration).toHaveAttribute("max", "100000");

    fireEvent.change(action, { target: { value: "240" } });
    fireEvent.change(first, { target: { value: "850" } });
    fireEvent.change(control, { target: { value: "900" } });
    fireEvent.change(browser, { target: { value: "1800" } });
    fireEvent.change(velocity, { target: { value: "35.5" } });
    fireEvent.change(acceleration, { target: { value: "120.25" } });

    expect(action).toHaveValue(240);
    expect(first).toHaveValue(850);
    expect(control).toHaveValue(900);
    expect(browser).toHaveValue(1800);
    expect(velocity).toHaveValue(35.5);
    expect(acceleration).toHaveValue(120.25);
    expect(first).toHaveAttribute("min", "240");
    expect(browser).toHaveAttribute("min", "900");
  });
});
