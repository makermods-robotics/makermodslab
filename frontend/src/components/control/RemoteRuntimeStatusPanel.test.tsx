import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import RemoteRuntimeStatusPanel from "./RemoteRuntimeStatusPanel";

describe("RemoteRuntimeStatusPanel", () => {
  it("shows confirmed off while an enabled robot waits for its first action", () => {
    render(
      <RemoteRuntimeStatusPanel
        status={{
          runtime_enabled: true,
          state: "waiting_for_first_action",
          runtime: {
            active: {
              executor: {
                safety: { torque_off_confirmed: true },
              },
            },
          },
        }}
      />,
    );

    expect(screen.getByText("Confirmed off")).toBeInTheDocument();
    expect(screen.queryByText("Unknown — treat as energized")).toBeNull();
  });

  it("reports an unknown torque receipt as unknown, never as off", () => {
    render(
      <RemoteRuntimeStatusPanel
        status={{
          state: "stopped",
          runtime_enabled: false,
          stop_receipt: {
            accepted: true,
            advancement_halted: true,
            torque_disable_requested: true,
            torque_off_confirmed: null,
            close_completed: null,
          },
        }}
      />,
    );

    expect(screen.getByText("Unknown — treat as energized")).toBeInTheDocument();
    expect(screen.queryByText("Confirmed off")).toBeNull();
    expect(screen.getAllByText("Unknown").length).toBeGreaterThan(0);
  });

  it("renders active unknown torque instead of a stale confirmed-off receipt", () => {
    render(
      <RemoteRuntimeStatusPanel
        status={{
          runtime_enabled: true,
          state: "active",
          stop_receipt: { torque_off_confirmed: true },
          runtime: {
            state: "active",
            active: {
              executor: {
                safety: { torque_off_confirmed: null },
              },
            },
          },
        }}
      />,
    );

    expect(screen.getByText("Unknown — treat as energized")).toBeInTheDocument();
    expect(screen.queryByText("Confirmed off")).toBeNull();
  });

  it("makes fault lockout and a failed torque confirmation visible", () => {
    render(
      <RemoteRuntimeStatusPanel
        status={{
          state: "faulted",
          runtime_enabled: false,
          fault_lockout: true,
          faults: ["torque_state_unknown"],
          stop_receipt: {
            torque_disable_requested: true,
            torque_off_confirmed: false,
            fault: "torque_still_enabled",
          },
        }}
      />,
    );

    expect(screen.getAllByText("Fault lockout").length).toBeGreaterThan(0);
    expect(screen.getByText("Enabled")).toBeInTheDocument();
    expect(screen.getByText("torque_state_unknown")).toBeInTheDocument();
    expect(screen.getByText(/torque_still_enabled/)).toBeInTheDocument();
  });
});
