import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, waitFor, cleanup } from "@testing-library/react";

const mocks = vi.hoisted(() => ({
  fetch: vi.fn(),
  startSession: vi.fn(async () => ({ session: { id: "sess-1" } })),
  toast: vi.fn(),
}));

vi.mock("@/hooks/useRobots", () => ({
  useRobots: () => ({ records: { bench: { name: "bench", cameras: [] } } }),
}));
vi.mock("@/contexts/ApiContext", () => ({
  useApi: () => ({ baseUrl: "http://test", fetchWithHeaders: mocks.fetch }),
}));
vi.mock("@/contexts/LanguageContext", () => ({
  useLanguage: () => ({ language: "en" }),
}));
vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({ toast: mocks.toast }),
}));
vi.mock("@/hooks/useSessionHeartbeat", () => ({ useSessionHeartbeat: () => {} }));
vi.mock("@/hooks/useUnloadWarning", () => ({ useUnloadWarning: () => {} }));
vi.mock("@/lib/sessionApi", () => ({
  startSession: mocks.startSession,
  stopSession: vi.fn(),
  formatSessionHeld: () => null,
}));
vi.mock("@/lib/recordingAudio", () => ({
  getMuted: () => true,
  setMuted: vi.fn(),
  playRecordingStartCue: vi.fn(),
  playResetStartCue: vi.fn(),
  playAutoAdvanceWarning: vi.fn(),
}));
vi.mock("@/components/control/SessionLiveView", () => ({ default: () => null }));
vi.mock("@/components/LogPanel", () => ({ default: () => null }));

import RecordingSessionDialog from "./RecordingSessionDialog";

const CONFIG = {
  robot: "bench",
  dataset_repo_id: "alice/varied",
  single_task: "pick the cube",
  per_episode_task: false,
  num_episodes: 3,
  episode_time_s: 60,
  reset_time_s: 0,
  fps: 30,
  video: true,
  push_to_hub: false,
  resume: false,
  streaming_encoding: true,
};

const jsonResponse = (body: unknown) => ({ ok: true, json: async () => body });

const endedStatus = (overrides: Record<string, unknown> = {}) => ({
  recording_active: false,
  current_phase: "completed",
  current_episode: 2,
  total_episodes: 3,
  saved_episodes: 2,
  session_ended: true,
  outcome: "failed",
  error: "RuntimeError: camera disconnected",
  hint: "A camera disconnected mid-session.",
  discarded_empty: false,
  dataset_repo_id: "alice/varied",
  available_controls: {
    stop_recording: false,
    exit_early: false,
    rerecord_episode: false,
    pause_recording: false,
    resume_recording: false,
    submit_episode_task: false,
  },
  ...overrides,
});

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(cleanup);

const mockStatus = (status: ReturnType<typeof endedStatus>) => {
  mocks.fetch.mockImplementation(async (url: string) => {
    if (url.endsWith("/recording-status")) return jsonResponse(status);
    if (url.endsWith("/recording-log")) return jsonResponse({ logs: "" });
    return jsonResponse({});
  });
};

describe("a session that ended with an issue but saved episodes", () => {
  it("offers exactly one button — no immediate-upload or hard-delete choice", async () => {
    mockStatus(endedStatus());
    render(<RecordingSessionDialog config={CONFIG} onExit={vi.fn()} />);

    await screen.findByRole("button", { name: /review & finalize/i });
    expect(screen.queryByRole("button", { name: /keep episodes/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /discard/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /back home/i })).not.toBeInTheDocument();
  });

  it("never calls delete-dataset — Finalize's own checkboxes own deletion now", async () => {
    mockStatus(endedStatus());
    render(<RecordingSessionDialog config={CONFIG} onExit={vi.fn()} />);

    await screen.findByRole("button", { name: /review & finalize/i });
    const deleteCalls = mocks.fetch.mock.calls.filter(([url]) =>
      String(url).includes("/delete-dataset"),
    );
    expect(deleteCalls).toHaveLength(0);
  });

  it("clicking it exits with the recorded payload, unchanged", async () => {
    mockStatus(endedStatus());
    const onExit = vi.fn();
    render(<RecordingSessionDialog config={CONFIG} onExit={onExit} />);

    fireEvent.click(await screen.findByRole("button", { name: /review & finalize/i }));
    expect(onExit).toHaveBeenCalledWith({
      repo_id: "alice/varied",
      saved_episodes: 2,
      discarded_empty: false,
    });
  });
});

describe("a session that ended with an issue and saved nothing", () => {
  it("offers a Continue button instead, which exits with discarded_empty", async () => {
    mockStatus(endedStatus({ saved_episodes: 0, discarded_empty: true }));
    const onExit = vi.fn();
    render(<RecordingSessionDialog config={CONFIG} onExit={onExit} />);

    const button = await screen.findByRole("button", { name: /^continue$/i });
    expect(
      screen.queryByRole("button", { name: /review & finalize/i }),
    ).not.toBeInTheDocument();
    fireEvent.click(button);
    await waitFor(() =>
      expect(onExit).toHaveBeenCalledWith({
        repo_id: "alice/varied",
        saved_episodes: 0,
        discarded_empty: true,
      }),
    );
  });
});
