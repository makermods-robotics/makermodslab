import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

const mocks = vi.hoisted(() => ({
  record: {
    name: "pair", mode: "bimanual", arm_type: "maker", arm_available: true,
    arms: "followers", follower_ready: true,
    follower_port: "left-port", follower_config: "left-calibration",
    right_follower_port: "right-port", right_follower_config: "right-calibration",
  },
  status: { replay_active: false, phase: "idle", elapsed_s: 0, episode_index: 2, duration_s: 60 },
  joints: {} as Record<string, number>,
  start: vi.fn(async (..._args: unknown[]) => ({ session: { id: "replay-1" } })),
  stop: vi.fn(async () => ({})),
  toast: vi.fn(),
  fetch: vi.fn(),
}));

vi.mock("@/hooks/useRobots", () => ({ useRobots: () => ({ selectedRecord: mocks.record }) }));
vi.mock("@/contexts/ApiContext", () => ({
  useApi: () => ({ baseUrl: "http://test", fetchWithHeaders: mocks.fetch }),
}));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ toast: mocks.toast }) }));
vi.mock("@/hooks/useSessionHeartbeat", () => ({ useSessionHeartbeat: () => {} }));
vi.mock("@/hooks/useUnloadWarning", () => ({ useUnloadWarning: () => {} }));
vi.mock("@/hooks/useLiveJointReadout", () => ({ useLiveJointReadout: () => ({ joints: mocks.joints }) }));
vi.mock("@/lib/sessionApi", () => ({
  startSession: mocks.start, stopSession: mocks.stop, formatSessionHeld: () => null,
}));
vi.mock("@/lib/replayHardwareApi", () => ({
  getReplayStatus: async () => mocks.status, stopReplay: mocks.stop,
}));

import EpisodeReplayPanel from "./EpisodeReplayPanel";

beforeEach(() => {
  vi.clearAllMocks();
  mocks.record.mode = "bimanual";
  mocks.record.follower_ready = true;
  mocks.status.replay_active = false;
  mocks.status.phase = "idle";
  mocks.joints = {};
});
afterEach(cleanup);

describe("hardware replay", () => {
  it("starts a bimanual replay through its saved robot record", async () => {
    render(<EpisodeReplayPanel repoId="owner/dataset" episodeIndex={2} />);
    expect(screen.getByText(/Moves both arms of pair/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Replay on hardware" }));
    await waitFor(() => expect(mocks.start).toHaveBeenCalledOnce());
    expect(mocks.start.mock.calls[0]).toEqual([
      "http://test", mocks.fetch,
      expect.objectContaining({ kind: "replay", robot: "pair", options: { repo_id: "owner/dataset", episode_index: 2 } }),
    ]);
  });

  it("keeps left and right readings distinct and exposes stop", async () => {
    mocks.status.replay_active = true;
    mocks.status.phase = "playing";
    mocks.joints = { "left_elbow_flex.pos": 12, "right_elbow_flex.pos": -8 };
    render(<EpisodeReplayPanel repoId="owner/dataset" episodeIndex={2} />);
    const left = await screen.findByRole("region", { name: "Left arm" });
    const right = screen.getByRole("region", { name: "Right arm" });
    expect(within(left).getByText("12.0")).toBeInTheDocument();
    expect(within(right).getByText("-8.0")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Stop" }));
    await waitFor(() => expect(mocks.stop).toHaveBeenCalledOnce());
  });

  it("retains the single-arm warning", () => {
    mocks.record.mode = "single";
    render(<EpisodeReplayPanel repoId="owner/dataset" episodeIndex={2} />);
    expect(screen.getByText(/Moves pair's arm/)).toBeInTheDocument();
    expect(screen.queryByText(/Moves both arms/)).not.toBeInTheDocument();
  });
});
