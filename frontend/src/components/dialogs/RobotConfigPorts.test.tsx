import { MAKER, METAL, SO101 } from "./robotConfigFixtures";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";

const { request } = vi.hoisted(() => ({ request: vi.fn() }));
vi.mock("@/hooks/useArms", () => ({
  useArms: () => ({
    byId: (id: string) => [MAKER, METAL, SO101].find((arm) => arm.id === id),
    arms: [MAKER, METAL, SO101],
    loading: false,
  }),
}));
vi.mock("@/contexts/ApiContext", () => ({
  useApi: () => ({ baseUrl: "", fetchWithHeaders: request }),
}));
vi.mock("@/components/recording/CameraConfiguration", () => ({
  default: () => null,
}));
vi.mock("@/components/calibration/CalibrationLibrary", () => ({
  default: () => null,
}));
import RobotConfigDialog from "./RobotConfigDialog";

const setup = async (
  mode = "single",
  multiple = false,
  leader_kind = "star",
) => {
  request.mockImplementation(async (url: string) => {
    let data: object = {
      active: false,
      arms: [],
      logs: [],
      calibration_active: false,
      status: "idle",
    };
    if (url === "/api/v1/robots/bench") {
      data = {
        robot: {
          name: "bench",
          arm_type: "metal",
          mode,
          leader_kind,
          leader_port: "l1",
          follower_port: "f1",
          cameras: [],
        },
      };
    } else if (url.endsWith("available-ports")) {
      data = { ports: ["l1", "l2", "f1", "f2"] };
    } else if (url.endsWith("maker/probe-ports")) {
      data = {
        success: true,
        leader_ports: ["l1"],
        follower_ports: multiple ? ["f1", "f2"] : ["f1"],
      };
    }
    return new Response(JSON.stringify(data));
  });
  render(<RobotConfigDialog open robotName="bench" onOpenChange={vi.fn()} />);
  await waitFor(() =>
    expect(
      screen.getByRole("combobox", {
        name:
          mode === "bimanual" ? "Port for Left Follower" : "Port for Follower",
      }),
    ).toHaveTextContent("f1"),
  );
};

beforeEach(() => {
  request.mockReset();
});
afterEach(cleanup);

describe("Star to Metal port controls", () => {
  it("single mode offers auto detection in addition to swing and wiggle", async () => {
    await setup();
    expect(screen.getAllByRole("button", { name: "Auto detect" })).toHaveLength(
      2,
    );
    expect(
      screen.getAllByRole("button", { name: "Detect by swing" }),
    ).toHaveLength(1);
    expect(screen.getAllByRole("button", { name: "Wiggle" })).toHaveLength(1);
    fireEvent.click(screen.getAllByRole("button", { name: "Auto detect" })[0]);
    await waitFor(() =>
      expect(request).toHaveBeenCalledWith(
        "/api/v1/maker/probe-ports",
        expect.anything(),
      ),
    );
    expect(
      request.mock.calls.some(([url]) => url.endsWith("identify-arm")),
    ).toBe(false);
  });

  it("bimanual mode offers only swing leaders and wiggle followers", async () => {
    await setup("bimanual");
    expect(
      screen.queryByRole("button", { name: "Auto detect" }),
    ).not.toBeInTheDocument();
    expect(
      screen.getAllByRole("button", { name: "Detect by swing" }),
    ).toHaveLength(2);
    expect(screen.getAllByRole("button", { name: "Wiggle" })).toHaveLength(2);
  });

  it.each(["single", "bimanual"])(
    "Metal-to-Metal %s uses wiggle on both sides",
    async (mode) => {
      await setup(mode, false, "metal");
      expect(
        screen.queryByRole("button", { name: "Detect by swing" }),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: "Auto detect" }),
      ).not.toBeInTheDocument();
      const buttons = screen.getAllByRole("button", { name: "Wiggle" });
      expect(buttons).toHaveLength(mode === "single" ? 2 : 4);
      fireEvent.click(buttons[0]);
      await waitFor(() =>
        expect(request).toHaveBeenCalledWith(
          "/api/v1/maker/wiggle-gripper",
          expect.objectContaining({
            body: JSON.stringify({
              arm_type: "metal",
              device_type: "teleop",
              port: "l1",
              leader_kind: "metal",
            }),
          }),
        ),
      );
    },
  );

  it("lets a second arm be set to no port without a switch prompt", async () => {
    request.mockImplementation(async (url: string) => {
      let data: object = {
        active: false,
        arms: [],
        logs: [],
        calibration_active: false,
        status: "idle",
      };
      if (url === "/api/v1/robots/bench") {
        data = {
          robot: {
            name: "bench",
            arm_type: "metal",
            mode: "single",
            leader_kind: "star",
            leader_port: "",
            follower_port: "f1",
            cameras: [],
          },
        };
      } else if (url.endsWith("available-ports")) {
        data = { ports: ["f1", "f2"] };
      }
      return new Response(JSON.stringify(data));
    });
    render(<RobotConfigDialog open robotName="bench" onOpenChange={vi.fn()} />);
    const follower = await screen.findByRole("combobox", {
      name: "Port for Follower",
    });
    await waitFor(() => expect(follower).toHaveTextContent("f1"));

    fireEvent.click(follower);
    fireEvent.click(await screen.findByRole("option", { name: "No port" }));

    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
    await waitFor(() => expect(follower).toHaveTextContent("No port"));
  });

  it("removes both auto controls if multiple arms answer", async () => {
    await setup("single", true);
    fireEvent.click(screen.getAllByRole("button", { name: "Auto detect" })[0]);
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Auto detect" }),
      ).not.toBeInTheDocument(),
    );
    expect(
      screen.getByRole("button", { name: "Detect by swing" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Wiggle" })).toBeInTheDocument();
  });
});


it("shows the saved Star vertical grip choice and its closed calibration hint", async () => {
  await setup("single", false, "star_vertical");
  expect(screen.getByRole("combobox", { name: "Leader arm" })).toHaveTextContent("Star arm vertical grip");
  expect(screen.getByText(/41.2° of grip travel/)).toBeInTheDocument();
  expect(screen.queryByText(/This leader supports its own weight/)).not.toBeInTheDocument();
});
