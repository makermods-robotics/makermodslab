import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { RobotRecord } from "@/hooks/useRobots";
import SessionCameraGrid from "./SessionCameraGrid";

vi.mock("./CameraFeed", () => ({ default: ({ label, cameraIndex }: { label: string; cameraIndex: number }) => <div data-testid="device">{label}:{cameraIndex}</div> }));
vi.mock("./RecordingCameraStream", () => ({ default: ({ cameraName }: { cameraName: string }) => <div data-testid="snapshot">{cameraName}</div> }));
const robot = {
  mode: "single",
  cameras: [0, 1, 2, 3].map(i => ({ id: `c${i}`, name: `view_${i}`, camera_index: i + 2 })),
} as RobotRecord;

describe("session camera sources", () => {
  it("shows every configured device immediately and releases feeds when switched off", () => {
    render(<SessionCameraGrid robot={robot} />);
    expect(screen.getAllByTestId("device")).toHaveLength(4);
    expect(screen.getByText("view_0:2")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("switch"));
    expect(screen.queryAllByTestId("device")).toHaveLength(0);
  });
  it("uses recorder snapshots instead of opening preview devices", () => {
    render(<SessionCameraGrid robot={robot} recording />);
    expect(screen.queryAllByTestId("device")).toHaveLength(0);
    expect(screen.getAllByTestId("snapshot").map(el => el.textContent)).toEqual(["view_0", "view_1", "view_2", "view_3"]);
  });
  it("uses the recorder's left-arm camera prefix for bimanual robots", () => {
    render(<SessionCameraGrid robot={{ ...robot, mode: "bimanual" }} recording paused />);
    expect(screen.getAllByTestId("snapshot").map(el => el.textContent)).toEqual(["left_view_0", "left_view_1", "left_view_2", "left_view_3"]);
    expect(screen.getByText(/last captured frames/)).toBeInTheDocument();
  });
});
