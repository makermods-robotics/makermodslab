import React from "react";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, renderHook, screen } from "@testing-library/react";
import { GPU_ACTIVE_POLL_MS, GPU_RESTART_STOP_TIMEOUT_MS, gpuKnobSupport, useGpuKnobs, useGpuLauncher, type GpuStartRequest, type GpuStatus, type GpuState } from "@/hooks/useGpuLauncher";
import { DEFAULT_REMOTE_RUN_CONFIG } from "./remoteRunConfig";
import GpuLaunchSection from "./GpuLaunchSection";
import ModalRunLine from "./ModalRunLine";

const { fetcher } = vi.hoisted(() => ({ fetcher: vi.fn() }));
vi.mock("@/contexts/ApiContext", () => ({ useApi: () => ({ baseUrl: "http://lab.test", fetchWithHeaders: fetcher }) }));

const restart = vi.fn(async () => {});
const start = vi.fn(async () => {});
const stop = vi.fn(async () => {});
const targets = { targets: null, profile: "", environment: "", setProfile: vi.fn(), setEnvironment: vi.fn(), refresh: vi.fn() };
const config = { ...DEFAULT_REMOTE_RUN_CONFIG, policyHubId: "someone/p" };

function Harness({ state = "idle", pending = false, restarting = false, task = "pick up the block", targetError, network }: { state?: GpuState; pending?: boolean; restarting?: boolean; task?: string; targetError?: { code: string; message: string }; network?: { region: string; tolerance: number } }) {
  const knobs = useGpuKnobs();
  const support = gpuKnobSupport(null);
  const status = state === "idle" ? null : {
    state, phase: "connected", engine: config.engine, policy_hub_id: "someone/p", task: "pick up the block",
    horizon: config.horizon, fps: config.fps, video_codec: config.videoCodec, s_min: config.sMin,
    slack: 5, elapsed_s: 0, idle_stop_in_s: null, ...network,
  } as GpuStatus;
  return <>
    <GpuLaunchSection
      launcher={{ status, pending, restarting, error: null, start, stop, restart, refresh: vi.fn(), launched: null }}
      targets={{ ...targets, targets: targetError ? { profiles: [], environments: [], profile: null, error: targetError } : null }} knobs={knobs} knobSupport={support} config={config}
      hubIdDefault="someone/p" task={task} taskRequired={true} extraImageRoles={[]}
    />
    <ModalRunLine
      config={config} transport={null} hubIdDefault="someone/p" task="pick up the block"
      profile="" environment="" knobs={knobs} knobSupport={support} extraImageRoles={[]}
    />
  </>;
}

beforeAll(() => {
  Element.prototype.hasPointerCapture = vi.fn(() => false);
  Element.prototype.setPointerCapture = vi.fn();
  Element.prototype.releasePointerCapture = vi.fn();
  Element.prototype.scrollIntoView = vi.fn();
});

beforeEach(() => {
  localStorage.clear();
  start.mockClear();
  stop.mockClear();
  restart.mockClear();
  fetcher.mockReset();
});

describe("sync slack", () => {
  it("updates launch and copied command together and remembers the selection", () => {
    const view = render(<Harness />);
    fireEvent.click(screen.getByText("GPU tuning"));
    const select = screen.getByLabelText("Buffer (steps)");
    expect(select).toHaveTextContent("5");
    fireEvent.keyDown(select, { key: "Enter" });
    fireEvent.keyDown(screen.getByRole("option", { name: "2" }), { key: "Enter" });
    expect(select).toHaveTextContent("2");
    expect(screen.getByText(/modal run makermodslab/)).toHaveTextContent("--slack 2");
    fireEvent.click(screen.getByRole("button", { name: "Start GPU" }));
    expect(start).toHaveBeenCalledWith(expect.objectContaining({ slack: 2 }));
    view.unmount();
    render(<Harness />);
    expect(screen.getByLabelText("Buffer (steps)")).toHaveTextContent("2");
  });

  it.each(["starting", "ready", "stopping"] as const)("is locked while the GPU is %s", (state) => {
    render(<Harness state={state} />);
    expect(screen.getByLabelText("Buffer (steps)")).toBeDisabled();
  });

  it("shows the running slack when a remembered selection differs after reload", () => {
    localStorage.setItem("makermodslab.gpuSyncSlack", "2");
    render(<Harness state="ready" />);
    expect(screen.getByLabelText("Buffer (steps)")).toHaveTextContent("2");
    expect(screen.getByText(/ · slack 5/)).toBeInTheDocument();
  });

  it("is locked during a launch request and rejects an invalid saved choice", () => {
    localStorage.setItem("makermodslab.gpuSyncSlack", "1");
    render(<Harness pending />);
    fireEvent.click(screen.getByText("GPU tuning"));
    const select = screen.getByLabelText("Buffer (steps)");
    expect(select).toHaveTextContent("5");
    expect(select).toBeDisabled();
  });
});


function response(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

const replacement: GpuStartRequest = {
  engine: "rtc", policy_hub_id: "someone/new", task: "lift the block", horizon: 30,
  fps: 30, video_codec: "MJPEG", s_min: 4, slack: 2, profile: "lab", environment: "main",
  model_dtype: "bfloat16", gpu: "H100", flow_steps: 6, extra_image_roles: ["cam2"],
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => { resolve = r; });
  return { promise, resolve };
}

afterEach(() => vi.useRealTimers());

describe("GPU restart", () => {
  it("wires the restart button to the selected settings and keeps Stop separate", () => {
    localStorage.setItem("makermodslab.gpuSyncSlack", "2");
    render(<Harness state="ready" />);
    fireEvent.click(screen.getByRole("button", { name: "Apply and restart GPU" }));
    expect(restart).toHaveBeenCalledWith(expect.objectContaining({ slack: 2, task: "pick up the block" }));
    expect(start).not.toHaveBeenCalled();
    expect(stop).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Stop GPU" }));
    expect(stop).toHaveBeenCalledOnce();
    expect(restart).toHaveBeenCalledOnce();
  });

  it("shows pending restart even after the old GPU reaches idle", () => {
    render(<Harness state="idle" pending restarting />);
    expect(screen.getByRole("button", { name: "Restarting GPU…" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Start GPU" })).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Starting a GPU with the new settings");
    expect(screen.getByLabelText("Buffer (steps)")).toBeDisabled();
  });

  it("does not stop a GPU when the replacement is missing a required task", () => {
    render(<Harness state="ready" task="" />);
    expect(screen.getByRole("button", { name: "Apply and restart GPU" })).toBeDisabled();
  });

  it("waits for shutdown, snapshots every setting, and prevents overlapping actions", async () => {
    vi.useFakeTimers();
    const stopping = deferred<Response>();
    const starting = deferred<Response>();
    fetcher.mockReturnValueOnce(stopping.promise)
      .mockResolvedValueOnce(response({ state: "idle" }))
      .mockReturnValueOnce(starting.promise);
    const { result } = renderHook(() => useGpuLauncher(false));
    const selected = { ...replacement, extra_image_roles: [...replacement.extra_image_roles] };
    let done!: Promise<void>;
    act(() => {
      done = result.current.restart(selected);
      void result.current.restart(selected);
      void result.current.start(selected);
      void result.current.stop();
    });
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(result.current.pending).toBe(true);
    selected.slack = 5;
    selected.extra_image_roles.push("cam3");
    await act(async () => stopping.resolve(response({ state: "stopping" })));
    expect(fetcher).toHaveBeenCalledTimes(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(GPU_ACTIVE_POLL_MS); });
    expect(fetcher.mock.calls.map(([url]) => url.split("/").pop())).toEqual(["stop", "gpu", "start"]);
    expect(JSON.parse(fetcher.mock.calls[2][1].body)).toEqual(replacement);
    expect(result.current.pending).toBe(true);
    expect(result.current.restarting).toBe(true);
    await act(async () => {
      starting.resolve(response({ started: true, gpu: { state: "starting" } }));
      await done;
    });
    expect(result.current.pending).toBe(false);
    expect(result.current.restarting).toBe(false);
    expect(result.current.launched).toEqual(replacement);
    expect(result.current.error).toBeNull();
  });

  it.each(["request", "cleanup", "timeout"])("never launches a replacement after a %s failure", async (kind) => {
    vi.useFakeTimers();
    if (kind === "request") fetcher.mockResolvedValue(response({ detail: "Stop refused" }, 409));
    else if (kind === "cleanup") fetcher.mockResolvedValue(response({ state: "failed", message: "Modal cleanup failed", hint: "Check ap-test." }));
    else fetcher.mockResolvedValue(response({ state: "stopping" }));
    const { result } = renderHook(() => useGpuLauncher(false));
    let done!: Promise<void>;
    await act(async () => { done = result.current.restart(replacement); });
    if (kind === "timeout") await act(async () => { await vi.advanceTimersByTimeAsync(GPU_RESTART_STOP_TIMEOUT_MS + GPU_ACTIVE_POLL_MS); });
    await act(async () => { await done; });
    expect(fetcher.mock.calls.some(([url]) => url.endsWith("/start"))).toBe(false);
    expect(result.current.error).toContain("No replacement was requested");
    if (kind === "cleanup") expect(result.current.error).toContain("Check ap-test");
    expect(result.current.pending).toBe(false);
  });

  it("reports replacement failure after successful shutdown and releases the action gate", async () => {
    fetcher.mockResolvedValueOnce(response({ state: "idle" }))
      .mockResolvedValueOnce(response({ detail: "Select a valid Modal profile" }, 400))
      .mockResolvedValueOnce(response({ started: true, gpu: { state: "starting" } }));
    const { result } = renderHook(() => useGpuLauncher(false));
    await act(async () => { await result.current.restart(replacement); });
    expect(result.current.error).toContain("old GPU stopped");
    expect(result.current.error).toContain("Select a valid Modal profile");
    expect(result.current.launched).toBeNull();
    await act(async () => { await result.current.start(replacement); });
    expect(result.current.error).toBeNull();
    expect(result.current.launched).toEqual(replacement);
  });

  it("ordinary stop never submits a start request", async () => {
    fetcher.mockResolvedValue(response({ state: "stopping" }));
    const { result } = renderHook(() => useGpuLauncher(false));
    await act(async () => { await result.current.stop(); });
    expect(fetcher).toHaveBeenCalledOnce();
    expect(fetcher.mock.calls[0][0]).toMatch(/\/stop$/);
  });
});


describe("Modal setup guidance", () => {
  it.each([
    ["gpu.cli_missing", "Install Modal to start a GPU."],
    ["gpu.unauthenticated", "Sign in to Modal to start a GPU."],
  ])("explains %s and rechecks after setup", (code, title) => {
    targets.refresh.mockClear();
    const view = render(<Harness targetError={{ code, message: "Long technical diagnostic" }} />);
    expect(screen.getByText(title)).toBeVisible();
    expect(screen.getByText("Run these commands on the computer hosting MakerMods Lab.")).toBeVisible();
    expect(screen.getByText(/modal setup/)).toBeVisible();
    if (code === "gpu.cli_missing") expect(screen.getByText(/uv tool install modal/)).toBeVisible();
    expect(screen.getByText("Long technical diagnostic")).not.toBeVisible();
    expect(screen.getByRole("button", { name: "Start GPU" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Check again" }));
    expect(targets.refresh).toHaveBeenCalledOnce();
    view.rerender(<Harness />);
    expect(screen.queryByText(title)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start GPU" })).toBeEnabled();
  });
});


it("restarts with the form region and tolerance when the running GPU differs", () => {
  render(<Harness state="ready" network={{ region: "eu", tolerance: 2.5 }} />);
  fireEvent.click(screen.getByRole("button", { name: /Restart GPU/i }));
  expect(restart).toHaveBeenCalledWith(expect.objectContaining({ region: "us-west", tolerance: 1.5 }));
});


it("sends and remembers Auto GPU without changing the A10G default", () => {
  const view = render(<Harness />);
  expect(screen.getByLabelText("GPU")).toHaveTextContent("A10G");
  fireEvent.keyDown(screen.getByLabelText("GPU"), { key: "Enter" });
  fireEvent.keyDown(screen.getByRole("option", { name: "Auto (less wait time)" }), { key: "Enter" });
  expect(screen.getByText("A10G, L4 or larger. Price varies.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Start GPU" }));
  expect(start).toHaveBeenCalledWith(expect.objectContaining({ gpu: "auto" }));
  expect(screen.getByText(/modal run makermodslab/)).toHaveTextContent("DRTC_GPU=auto");
  view.unmount();
  render(<Harness />);
  expect(screen.getByLabelText("GPU")).toHaveTextContent("Auto (less wait time)");
});
