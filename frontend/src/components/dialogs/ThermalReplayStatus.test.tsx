import { afterEach, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { ThermalReplayStatus } from "./ThermalReplayStatus";
import type { ThermalReplayStatus as Status } from "@/lib/replayHardwareApi";

afterEach(cleanup);
const status: Status = {
  result: "overheated", message: "shoulder_lift reached 110°C", experiment: "baseline",
  elapsed_s: 10, duration_s: 1800, cycles: 1, critical: false, rest_reached: true, log_dir: "/run",
  actuators: [{ actuator: "follower.shoulder_lift", temperature_c: 110, torque_nm: 6.2, status: "OVERHEATING" }],
};
it("shows the thermal alert and temperature and torque together", () => {
  render(<ThermalReplayStatus status={status} />);
  expect(screen.getByRole("alert")).toHaveTextContent("110°C");
  expect(screen.getByText(/6.20 Nm/)).toHaveClass("text-red-800");
});
it("does not show PASS until rest arrival is confirmed", () => {
  const view = render(<ThermalReplayStatus status={{ ...status, result: "completed", rest_reached: false }} />);
  expect(screen.queryByText(/PASS/)).not.toBeInTheDocument();
  view.rerender(<ThermalReplayStatus status={{ ...status, result: "completed" }} />);
  expect(screen.getByText(/PASS/)).toBeInTheDocument();
});
