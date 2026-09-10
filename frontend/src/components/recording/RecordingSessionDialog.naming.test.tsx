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
  // Per-episode-task sessions carry no dataset-level task.
  single_task: "",
  per_episode_task: true,
  num_episodes: 3,
  episode_time_s: 60,
  reset_time_s: 15,
  fps: 30,
  video: true,
  push_to_hub: false,
  resume: false,
  streaming_encoding: true,
};

const namingStatus = {
  recording_active: true,
  current_phase: "naming",
  current_episode: 2,
  total_episodes: 3,
  saved_episodes: 1,
  session_ended: false,
  current_episode_task_default: "pick the cube",
  available_controls: {
    stop_recording: true,
    exit_early: false,
    rerecord_episode: false,
    pause_recording: false,
    resume_recording: false,
    submit_episode_task: true,
  },
};

const jsonResponse = (body: unknown) => ({
  ok: true,
  json: async () => body,
});

beforeEach(() => {
  vi.clearAllMocks();
  mocks.fetch.mockImplementation(async (url: string) => {
    if (url.endsWith("/recording-status")) return jsonResponse(namingStatus);
    if (url.endsWith("/recording-log")) return jsonResponse({ logs: "" });
    if (url.endsWith("/recording-episode-task"))
      return jsonResponse({ success: true, message: "Episode task saved" });
    return jsonResponse({});
  });
});

afterEach(cleanup);

describe("the recording dialog during the naming phase", () => {
  it("prompts for the episode's task and posts it to the naming endpoint", async () => {
    render(<RecordingSessionDialog config={CONFIG} onExit={vi.fn()} />);

    const box = await screen.findByRole("textbox");
    expect(box).toHaveValue("pick the cube");

    fireEvent.change(box, { target: { value: "fold the napkin" } });
    fireEvent.click(screen.getByRole("button", { name: /start recording/i }));

    await waitFor(() => {
      const call = mocks.fetch.mock.calls.find(([u]) =>
        String(u).endsWith("/api/v1/recording-episode-task"),
      );
      expect(call).toBeTruthy();
      expect(JSON.parse(call![1].body)).toEqual({ task: "fold the napkin" });
    });
  });

  it("offers Done/Quit while waiting for the next task", async () => {
    render(<RecordingSessionDialog config={CONFIG} onExit={vi.fn()} />);
    await screen.findByRole("textbox");
    expect(
      screen.queryByRole("button", { name: /^finish session$/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /^quit$/i }),
    ).toBeInTheDocument();
  });
});
