import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import RemoteCommissioningPanel from "./RemoteCommissioningPanel";
import type { RemoteTeleoperationStatus } from "@/lib/remoteTeleoperationApi";

const registry = {
  held: false,
  state: "free",
  kind: null,
  owner: null,
  generation: 4,
};

const commissionable: RemoteTeleoperationStatus = {
  configured: true,
  role: "robot",
  runtime_enabled: false,
  state: "disabled",
  commissioning: { commissioned: false, record: null },
  durable_fault: { fault_lockout: false, record: null },
  hardware_registry: registry,
};

const confirmSafeguards = () => {
  fireEvent.click(
    screen.getByLabelText(
      "The follower is physically secured against unexpected motion or a drop.",
    ),
  );
  fireEvent.click(
    screen.getByLabelText(
      "The workspace is clear and everyone is outside the arm's reach.",
    ),
  );
  fireEvent.click(
    screen.getByLabelText("A physical power cutoff is immediately reachable."),
  );
  fireEvent.click(
    screen.getByLabelText(
      "I understand that live execution enables Feetech torque and requires an immediately reachable physical cutoff.",
    ),
  );
};

describe("RemoteCommissioningPanel", () => {
  it("requires all four fresh safeguards and resets them after commissioning", async () => {
    const onCommission = vi.fn(async () => {});
    render(
      <RemoteCommissioningPanel
        status={commissionable}
        busyAction={null}
        onCommission={onCommission}
        onRecover={vi.fn(async () => {})}
      />,
    );

    const button = screen.getByRole("button", {
      name: "Commission this profile",
    });
    expect(button).toBeDisabled();
    confirmSafeguards();
    expect(button).toBeEnabled();
    fireEvent.click(button);

    await waitFor(() =>
      expect(onCommission).toHaveBeenCalledWith({
        arm_secured: true,
        workspace_clear: true,
        physical_power_cutoff_reachable: true,
        acknowledge_live_torque_enable_risk: true,
      }),
    );
    await waitFor(() => expect(button).toBeDisabled());
  });

  it("offers recovery instead of commissioning only for a durable fault", () => {
    render(
      <RemoteCommissioningPanel
        status={{
          ...commissionable,
          durable_fault: {
            fault_lockout: true,
            record: {
              profile_digest: "a".repeat(64),
              reason_code: "shutdown_unconfirmed",
              fault_codes: ["torque_state_unknown"],
            },
          },
          hardware_registry: {
            ...registry,
            held: true,
            state: "unresolved",
            owner: "local:secured-arm",
          },
        }}
        busyAction={null}
        onCommission={vi.fn(async () => {})}
        onRecover={vi.fn(async () => {})}
      />,
    );

    expect(
      screen.getByRole("button", { name: "Run secured recovery" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Commission this profile" }),
    ).toBeNull();
    expect(screen.getByText("torque_state_unknown")).toBeInTheDocument();
  });

  it("offers no hardware probe while the remote role is enabled", () => {
    render(
      <RemoteCommissioningPanel
        status={{ ...commissionable, runtime_enabled: true }}
        busyAction={null}
        onCommission={vi.fn(async () => {})}
        onRecover={vi.fn(async () => {})}
      />,
    );

    expect(
      screen.getByText(
        "Disable the remote role before commissioning or recovery.",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByRole("checkbox")).toBeNull();
  });

  it("surfaces and blocks on a pending unresolved registry latch", () => {
    render(
      <RemoteCommissioningPanel
        status={{
          ...commissionable,
          hardware_registry: {
            ...registry,
            pending_unresolved: true,
            pending_kind: "remote_recovery",
            pending_owner: "durable:robot-fault",
          },
        }}
        busyAction={null}
        onCommission={vi.fn(async () => {})}
        onRecover={vi.fn(async () => {})}
      />,
    );

    expect(
      screen.getByText(/Pending unresolved hardware latch/),
    ).toHaveTextContent(
      "Pending unresolved hardware latch · remote_recovery · durable:robot-fault",
    );
    expect(screen.queryByRole("checkbox")).toBeNull();
    expect(
      screen.getByText(
        "Hardware is retained by another or unresolved lease. Recovery is required if a durable fault is present.",
      ),
    ).toBeInTheDocument();
  });
});
