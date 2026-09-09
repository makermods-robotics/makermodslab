import { MAKER, METAL, SO101 } from "./robotConfigFixtures";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  cleanup,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import RobotConfigDialog from "./RobotConfigDialog";

const mocks = vi.hoisted(() => ({
  toast: vi.fn(),
  fetch: vi.fn(),
  start: vi.fn(),
}));
vi.mock("@/hooks/useArms", () => ({
  useArms: () => ({
    byId: (id: string) => [MAKER, METAL, SO101].find((arm) => arm.id === id),
    arms: [MAKER, METAL, SO101],
    loading: false,
  }),
}));
vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({ toast: mocks.toast }),
}));
vi.mock("@/contexts/ApiContext", () => ({
  useApi: () => ({ baseUrl: "http://test", fetchWithHeaders: mocks.fetch }),
}));
vi.mock("@/hooks/useSessionHeartbeat", () => ({
  useSessionHeartbeat: () => {},
}));
vi.mock("@/hooks/useUnloadWarning", () => ({ useUnloadWarning: () => {} }));
vi.mock("@/lib/sessionApi", () => ({
  startSession: mocks.start,
  stopSession: vi.fn(),
  getCurrentSession: vi.fn(async () => null),
  formatSessionHeld: () => null,
}));
vi.mock("@/components/recording/CameraConfiguration", () => ({
  default: () => null,
}));
vi.mock("@/components/calibration/CalibrationLibrary", () => ({
  default: ({
    onCalibrate,
    configField,
    calibrateDisabled,
  }: {
    onCalibrate: () => void;
    configField: string;
    calibrateDisabled: boolean;
  }) => (
    <button disabled={calibrateDisabled} onClick={onCalibrate}>
      Calibrate {configField}
    </button>
  ),
}));

let status: Record<string, unknown>;
let terminal: "completed" | "error";
let mode: "single" | "bimanual";

beforeEach(() => {
  vi.clearAllMocks();
  mode = "single";
  terminal = "completed";
  status = { status: "idle", calibration_active: false };
  mocks.fetch.mockImplementation(async (url: string) => {
    let data: unknown = {};
    if (url.includes("/robots/")) {
      data = {
        robot: {
          name: "test",
          mode,
          arm_type: "maker",
          cameras: [],
          leader_port: "leader",
          follower_port: "follower",
          right_leader_port: "right-leader",
          right_follower_port: "right-follower",
        },
      };
    } else if (url.includes("/available-ports")) {
      data = {
        ports: ["leader", "follower", "right-leader", "right-follower"],
      };
    } else if (url.includes("/auto-calibration-batch-status")) {
      data = {
        active: false,
        arms: [],
        total: 0,
        completed: 0,
        failed: 0,
        logs: [],
      };
    } else if (url.includes("/calibration-status")) {
      data = { ...status };
    } else if (url.includes("/complete-calibration-step")) {
      status = {
        status: terminal,
        calibration_active: false,
        error:
          terminal === "error" ? "No response from motor 'shoulder'" : null,
      };
      data = { success: true, message: "Zero pose confirmed" };
    }
    return { ok: true, json: async () => data };
  });
  mocks.start.mockImplementation(async () => {
    status = {
      status: "awaiting_step",
      step: 1,
      live_positions: true,
      calibration_active: true,
    };
    return { session: { id: "session" } };
  });
});
afterEach(cleanup);

async function openAll() {
  render(<RobotConfigDialog open robotName="test" onOpenChange={() => {}} />);
  const button = await screen.findByRole("button", { name: "Calibrate all" });
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
}

async function startAndSave() {
  const start = await screen.findByRole("button", { name: "Set zero pose" });
  await waitFor(() => expect(start).toBeEnabled());
  fireEvent.click(start);
  fireEvent.click(
    await screen.findByRole("button", { name: "Set zero and save" }),
  );
}

describe("zero-pose calibration", () => {
  it("keeps the live joint grid in place when subsequent samples arrive in a different order", async () => {
    const positions = {
      wrist_roll: 5.7,
      wrist_yaw: -1.1,
      gripper: -7.2,
      wrist_flex: -0.2,
      elbow_flex: 0,
      shoulder_lift: 0.1,
      shoulder_pan: -0.3,
    };
    mocks.start.mockImplementation(async () => {
      status = {
        status: "awaiting_step",
        step: 1,
        live_positions: true,
        calibration_active: true,
        current_positions: positions,
      };
      return { session: { id: "session" } };
    });
    await openAll();
    fireEvent.click(
      await screen.findByRole("button", { name: "Set zero pose" }),
    );
    await screen.findByText("wrist_roll");
    const grid =
      screen.getByText("Live joint angles").parentElement!.nextElementSibling!;
    const rows = Array.from(grid.children);
    expect(rows.map((row) => row.firstElementChild!.textContent)).toEqual([
      "shoulder_pan",
      "shoulder_lift",
      "elbow_flex",
      "wrist_flex",
      "wrist_yaw",
      "wrist_roll",
      "gripper",
    ]);
    status = {
      ...status,
      current_positions: {
        ...Object.fromEntries(Object.entries(positions).reverse()),
        gripper: -6.2,
      },
    };
    await screen.findByText("-6.2°");
    expect(Array.from(grid.children)).toEqual(rows);
    expect(rows.at(-1)!.textContent).toBe("gripper-6.2°");
  });

  it("toasts a connection failure immediately after Set zero pose", async () => {
    mocks.start.mockImplementation(async () => {
      status = {
        status: "error",
        calibration_active: false,
        error: "Handshake failed. No response",
      };
      return { session: { id: "session" } };
    });
    await openAll();
    fireEvent.click(
      await screen.findByRole("button", { name: "Set zero pose" }),
    );
    await waitFor(() =>
      expect(mocks.toast).toHaveBeenCalledWith(
        expect.objectContaining({
          description: "Arm not responding. Check power and cable.",
          variant: "destructive",
        }),
      ),
    );
    expect(
      screen.queryByRole("button", { name: "Set zero and save" }),
    ).not.toBeInTheDocument();
  });

  it("calibrates all four bimanual slots in order, requiring each pose", async () => {
    mode = "bimanual";
    await openAll();
    const slots = [
      ["teleop", "left", "leader"],
      ["robot", "left", "follower"],
      ["teleop", "right", "right-leader"],
      ["robot", "right", "right-follower"],
    ];
    for (const [index, [device_type, arm, port]] of slots.entries()) {
      await startAndSave();
      expect(mocks.start).toHaveBeenCalledTimes(index + 1);
      expect(mocks.start.mock.calls[index][2].options).toMatchObject({
        device_type,
        arm,
        port,
      });
      if (index < 3)
        await screen.findByRole("button", { name: "Set zero pose" });
    }
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Set zero and save" }),
      ).not.toBeInTheDocument(),
    );
    expect(
      mocks.fetch.mock.calls.some(([url]) =>
        url.endsWith("calibration-status?arm_type=maker"),
      ),
    ).toBe(true);
  });

  it("shows one short error toast, stops the queue, and allows retry", async () => {
    terminal = "error";
    await openAll();
    await startAndSave();
    const failure = {
      title: "Calibration Failed",
      description: "No response from shoulder. Check power and wiring.",
      variant: "destructive",
    };
    await waitFor(() => expect(mocks.toast).toHaveBeenCalledWith(failure));
    expect(
      mocks.toast.mock.calls.filter(
        ([toast]) => toast.variant === "destructive",
      ),
    ).toHaveLength(1);
    expect(
      mocks.toast.mock.calls.some(
        ([toast]) => toast.title === "Step Completed",
      ),
    ).toBe(false);
    expect(mocks.start).toHaveBeenCalledTimes(1);
    expect(
      screen.queryByText(/arm remaining|arms remaining/),
    ).not.toBeInTheDocument();
    await startAndSave();
    await waitFor(() =>
      expect(
        mocks.toast.mock.calls.filter(
          ([toast]) => toast.variant === "destructive",
        ),
      ).toHaveLength(2),
    );
  });

  it("cancels the remaining sequence without starting hardware", async () => {
    await openAll();
    fireEvent.click(await screen.findByRole("button", { name: "Cancel all" }));
    expect(mocks.start).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Calibrate all" })).toBeEnabled();
  });
});
