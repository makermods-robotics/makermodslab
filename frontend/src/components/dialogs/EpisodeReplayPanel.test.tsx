import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TooltipProvider } from "@/components/ui/tooltip";
import EpisodeReplayPanel from "./EpisodeReplayPanel";

const mocks = vi.hoisted(() => ({ status: vi.fn(), stop: vi.fn() }));
vi.mock("@/contexts/ApiContext", () => ({ useApi: () => ({ baseUrl: "http://test", fetchWithHeaders: vi.fn() }) }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ toast: vi.fn() }) }));
vi.mock("@/hooks/useRobots", () => ({ useRobots: () => ({ selectedRecord: { name: "test", follower_ready: true } }) }));
vi.mock("@/hooks/useSessionHeartbeat", () => ({ useSessionHeartbeat: () => {} }));
vi.mock("@/hooks/useUnloadWarning", () => ({ useUnloadWarning: () => {} }));
vi.mock("@/hooks/useLiveJointReadout", () => ({ useLiveJointReadout: () => ({ joints: {} }) }));
vi.mock("@/lib/replayHardwareApi", () => ({ getReplayStatus: mocks.status, stopReplay: mocks.stop }));

afterEach(() => { cleanup(); vi.clearAllMocks(); });

describe("replay stop affordance", () => {
  it.each([
    ["playing", "Stop", "Returns the arm to its start pose, then releases torque."],
    ["stopping", "Release now", "Skips the return and releases torque immediately."],
  ])("describes the %s phase stop accurately", async (phase, label, tooltip) => {
    mocks.status.mockResolvedValue({ replay_active: true, phase, elapsed_s: 0 });
    mocks.stop.mockResolvedValue({});
    render(<TooltipProvider><EpisodeReplayPanel repoId="test/data" episodeIndex={0} /></TooltipProvider>);
    const button = await screen.findByRole("button", { name: label });
    fireEvent.focus(button);
    expect(await screen.findByRole("tooltip")).toHaveTextContent(tooltip);
    fireEvent.click(button);
    expect(mocks.stop).toHaveBeenCalledOnce();
  });
});
