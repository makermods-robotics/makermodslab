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
import { StudioProvider, useStudio } from "@/contexts/StudioContext";

const baseProps = {
  robot: null,
  datasetName: "sock_sort",
  setDatasetName: vi.fn(),
  singleTask: "sort socks",
  setSingleTask: vi.fn(),
  perEpisodeTask: false,
  setPerEpisodeTask: vi.fn(),
  resetTimeS: 15,
  setResetTimeS: vi.fn(),
  numEpisodes: 5,
  setNumEpisodes: vi.fn(),
  episodeTimeS: 60,
  setEpisodeTimeS: vi.fn(),
  streamingEncoding: true,
  setStreamingEncoding: vi.fn(),
  pushToHub: true,
  setPushToHub: vi.fn(),
};

function FormWithDraft() {
  const { collectForm, updateCollectForm } = useStudio();
  return <RecordingForm {...baseProps} {...collectForm}
    setPerEpisodeTask={(perEpisodeTask) => updateCollectForm({ perEpisodeTask })}
    setSingleTask={(singleTask) => updateCollectForm({ singleTask })}
  />;
}

describe("recording setup", () => {
  it("defaults to one task and enables the per-episode flow only when toggled", () => {
    render(<StudioProvider><FormWithDraft /></StudioProvider>);
    const toggle = screen.getByRole("switch", { name: "Describe each episode" });
    expect(toggle).not.toBeChecked();
    const task = screen.getByLabelText(/task description/i);
    fireEvent.change(task, { target: { value: "sort socks" } });
    expect(screen.getByLabelText(/reset duration/i)).toBeInTheDocument();

    fireEvent.click(toggle);
    expect(toggle).toBeChecked();
    expect(screen.queryByLabelText(/task description/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/reset duration/i)).not.toBeInTheDocument();

    fireEvent.click(toggle);
    expect(toggle).not.toBeChecked();
    expect(screen.getByLabelText(/task description/i)).toHaveValue("sort socks");
    expect(screen.getByLabelText(/reset duration/i)).toBeInTheDocument();
  });
});
