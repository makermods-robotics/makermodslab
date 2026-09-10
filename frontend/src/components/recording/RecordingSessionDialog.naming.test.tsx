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
  reset_time_s: 0,
  fps: 30,
  video: true,
  push_to_hub: false,
  resume: false,
  streaming_encoding: true,
};

const namingStatus = {
  recording_active: true,
  current_phase: "naming",
  current_episode: 1,
  total_episodes: 3,
  saved_episodes: 0,
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

let status = structuredClone(namingStatus);

beforeEach(() => {
  status = structuredClone(namingStatus);
  vi.clearAllMocks();
  mocks.fetch.mockImplementation(async (url: string) => {
    if (url.endsWith("/recording-status")) return jsonResponse(status);
    if (url.endsWith("/recording-log")) return jsonResponse({ logs: "" });
    if (url.endsWith("/recording-episode-task")) {
      status = { ...status, current_phase: "recording", available_controls: {
        ...status.available_controls, exit_early: true, submit_episode_task: false,
      } };
      return jsonResponse({ success: true });
    }
    if (url.endsWith("/recording-exit-early")) {
      status = { ...namingStatus, current_episode: 2 };
      return jsonResponse({ success: true });
    }
    return jsonResponse({});
  });
});

afterEach(cleanup);

const submitFirstTask = () => {
  const box = screen.getByRole("textbox");
  fireEvent.change(box, { target: { value: "pick the cube" } });
  fireEvent.keyDown(box, { key: "Enter" });
  fireEvent.keyDown(screen.getByRole("button", { name: /start recording/i }), { key: " " });
};

const taskCalls = () => mocks.fetch.mock.calls.filter(([url]) =>
  String(url).endsWith("/api/v1/recording-episode-task"),
);

describe("task before recording", () => {
  it("shows the first task immediately and allows cancel without starting hardware", () => {
    const onExit = vi.fn();
    render(<RecordingSessionDialog config={CONFIG} onExit={onExit} />);
    expect(screen.getByRole("textbox")).toHaveValue("");
    expect(mocks.startSession).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onExit).toHaveBeenCalledOnce();
    expect(mocks.startSession).not.toHaveBeenCalled();
  });

  it("starts the session and submits the first task with one Space", async () => {
    render(<RecordingSessionDialog config={CONFIG} onExit={vi.fn()} />);
    submitFirstTask();
    await waitFor(() => expect(taskCalls()).toHaveLength(1));
    expect(mocks.startSession).toHaveBeenCalledOnce();
    expect(mocks.startSession).toHaveBeenCalledWith("http://test", mocks.fetch,
      expect.objectContaining({ options: expect.objectContaining({ single_task: "pick the cube" }) }),
    );
    expect(JSON.parse(taskCalls()[0][1].body)).toEqual({ task: "pick the cube" });
    await screen.findByRole("button", { name: /^done/i }, { timeout: 2500 });
    expect(taskCalls()).toHaveLength(1);
  });

  it("Space ends capture, then asks for the next task until Space starts it", async () => {
    render(<RecordingSessionDialog config={CONFIG} onExit={vi.fn()} />);
    submitFirstTask();
    await screen.findByRole("button", { name: /^done/i }, { timeout: 2500 });
    fireEvent.keyDown(window, { key: " " });
    const box = await screen.findByRole("textbox", {}, { timeout: 2500 });
    expect(screen.getByRole("button", { name: /^finish session$/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^quit$/i })).toBeInTheDocument();
    expect(taskCalls()).toHaveLength(1);
    fireEvent.change(box, { target: { value: "fold the napkin" } });
    fireEvent.keyDown(box, { key: "Enter" });
    fireEvent.keyDown(screen.getByRole("button", { name: /start recording/i }), { key: " " });
    await waitFor(() => expect(taskCalls()).toHaveLength(2));
    expect(JSON.parse(taskCalls()[1][1].body)).toEqual({ task: "fold the napkin" });
  });
});
