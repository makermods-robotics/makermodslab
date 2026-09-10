import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

vi.mock("@/contexts/HfAuthContext", () => ({
  useHfAuth: () => ({
    auth: { status: "unauthenticated", loginCommand: "hf auth login" },
    refetch: vi.fn(),
  }),
}));
// The read-only camera list pulls in useApi/useAvailableCameras — not part of
// this form's contract here.
vi.mock("@/components/recording/CameraConfiguration", () => ({
  SessionCameraList: () => null,
}));

import RecordingForm from "./RecordingForm";

const baseProps = {
  robot: null,
  datasetName: "sock_sort",
  setDatasetName: vi.fn(),
  singleTask: "sort the socks",
  setSingleTask: vi.fn(),
  numEpisodes: 5,
  setNumEpisodes: vi.fn(),
  episodeTimeS: 60,
  setEpisodeTimeS: vi.fn(),
  resetTimeS: 15,
  setResetTimeS: vi.fn(),
  streamingEncoding: true,
  setStreamingEncoding: vi.fn(),
  pushToHub: true,
  setPushToHub: vi.fn(),
  perEpisodeTask: false,
  setPerEpisodeTask: vi.fn(),
};

describe("the recording form's per-episode-task checkbox", () => {
  it("is offered unchecked next to the task description", () => {
    render(<RecordingForm {...baseProps} />);
    const box = screen.getByRole("checkbox", {
      name: /name each episode's task after recording it/i,
    });
    expect(box).toHaveAttribute("data-state", "unchecked");
  });

  it("turns the mode on when the operator ticks it", () => {
    const setPerEpisodeTask = vi.fn();
    render(
      <RecordingForm {...baseProps} setPerEpisodeTask={setPerEpisodeTask} />,
    );
    fireEvent.click(
      screen.getByRole("checkbox", {
        name: /name each episode's task after recording it/i,
      }),
    );
    expect(setPerEpisodeTask).toHaveBeenCalledWith(true);
  });

  it("shows the dataset-level task field only while the mode is off", () => {
    const { rerender } = render(<RecordingForm {...baseProps} />);
    expect(screen.getByLabelText(/task description/i)).toBeInTheDocument();

    rerender(<RecordingForm {...baseProps} perEpisodeTask={true} />);
    expect(screen.queryByLabelText(/task description/i)).not.toBeInTheDocument();
    // The checkbox itself stays put.
    expect(
      screen.getByRole("checkbox", {
        name: /name each episode's task after recording it/i,
      }),
    ).toBeInTheDocument();
  });
});
