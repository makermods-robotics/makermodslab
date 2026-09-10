import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import RemoteNetworkSection from "./RemoteNetworkSection";
import { DEFAULT_REMOTE_RUN_CONFIG, type RemoteRunConfig } from "./remoteRunConfig";

function Harness({ codec = "MJPEG", engine = "rtc", disabled = false }: { codec?: RemoteRunConfig["videoCodec"]; engine?: RemoteRunConfig["engine"]; disabled?: boolean }) {
  const [config, setConfig] = useState({ ...DEFAULT_REMOTE_RUN_CONFIG, engine, videoCodec: codec, fps: 20 });
  return <><RemoteNetworkSection config={config} onChange={setConfig} disabled={disabled} /><output>{JSON.stringify(config)}</output></>;
}

describe("network controls", () => {
  it("keeps control FPS unchanged when limiting camera sends", () => {
    render(<Harness />);
    fireEvent.click(screen.getByText("Network"));
    expect(screen.getByLabelText("MJPEG quality")).toHaveValue(90);
    expect(screen.queryByLabelText("H264 bitrate (kbps)")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Camera send rate (Hz)"), { target: { value: "5" } });
    expect(screen.getByRole("status")).toHaveTextContent('"cameraSendHz":5');
    expect(screen.getByRole("status")).toHaveTextContent('"fps":20');
    fireEvent.click(screen.getByText("Advanced"));
    expect(screen.getByLabelText("Jitter margin")).toHaveValue(1.5);
  });
  it("shows bitrate for H264 and hides RTC-only controls for sync", () => {
    render(<Harness codec="H264" engine="sync" />);
    fireEvent.click(screen.getByText("Network"));
    expect(screen.getByLabelText("H264 bitrate (kbps)")).toHaveValue(4000);
    expect(screen.queryByLabelText("Camera send rate (Hz)")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Jitter margin")).not.toBeInTheDocument();
  });
  it("locks settings during a robot run", () => {
    render(<Harness disabled />);
    fireEvent.click(screen.getByText("Network"));
    expect(screen.getByLabelText("GPU region")).toBeDisabled();
    expect(screen.getByLabelText("Camera send rate (Hz)")).toBeDisabled();
    expect(screen.getByLabelText("Frame matching tolerance")).toBeDisabled();
  });
});


it("offers automatic region assignment", () => {
  render(<Harness />);
  fireEvent.click(screen.getByText("Network"));
  fireEvent.keyDown(screen.getByLabelText("GPU region"), { key: "Enter" });
  fireEvent.keyDown(screen.getByRole("option", { name: "Auto (less wait time)" }), { key: "Enter" });
  expect(screen.getByRole("status")).toHaveTextContent('"region":"auto"');
  expect(screen.getByText("Modal chooses the region. Network delay may vary.")).toBeInTheDocument();
});
