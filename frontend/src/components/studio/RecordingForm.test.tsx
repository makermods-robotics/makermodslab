import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

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
  numEpisodes: 5,
  setNumEpisodes: vi.fn(),
  episodeTimeS: 60,
  setEpisodeTimeS: vi.fn(),
  streamingEncoding: true,
  setStreamingEncoding: vi.fn(),
  pushToHub: true,
  setPushToHub: vi.fn(),
};

describe("recording setup", () => {
  it("asks only for dataset settings before opening the task prompt", () => {
    render(<RecordingForm {...baseProps} />);
    expect(screen.getByLabelText(/dataset name/i)).toBeInTheDocument();
    expect(screen.queryByLabelText(/task description/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/reset duration/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: /episode.*task/i })).not.toBeInTheDocument();
  });
});
