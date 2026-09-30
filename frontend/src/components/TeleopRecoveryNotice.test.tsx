import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import TeleopRecoveryNotice from "./TeleopRecoveryNotice";
import { Toaster } from "@/components/ui/toaster";
import { toast } from "@/hooks/use-toast";

const mocks = vi.hoisted(() => ({ fetch: vi.fn(), current: vi.fn(), stop: vi.fn() }));
vi.mock("@/contexts/ApiContext", () => ({
  useApi: () => ({ baseUrl: "http://station", fetchWithHeaders: mocks.fetch }),
}));
vi.mock("react-i18next", () => {
  const t = (key: string) => key;
  return { useTranslation: () => ({ t }) };
});
vi.mock("@/lib/sessionApi", () => ({ getCurrentSession: mocks.current, stopSession: mocks.stop }));

beforeEach(() => {
  vi.clearAllMocks();
  mocks.fetch.mockResolvedValue({ ok: true, json: async () => ({ rest_failed: true, message: "10.9 deg away" }) });
  mocks.current.mockResolvedValue({ session: { id: "failed-session", kind: "teleoperation", phase: "rest_failed" } });
  mocks.stop.mockResolvedValue({ result: { success: true } });
});

afterEach(cleanup);

const renderNotice = () => render(<><TeleopRecoveryNotice /><Toaster /></>);

describe("TeleopRecoveryNotice", () => {
  it("requires physical support acknowledgement and targets the failed session explicitly", async () => {
    renderNotice();
    const button = await screen.findByRole("button", { name: "dialogs.teleop.releaseTorque" });
    expect(button).toBeDisabled();
    expect(screen.getByText("dialogs.teleop.restFailed").closest("li")).toHaveClass("bg-destructive");
    fireEvent.click(button);
    expect(mocks.stop).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(button);
    await waitFor(() => expect(mocks.stop).toHaveBeenCalledWith("http://station", mocks.fetch, "failed-session", true));
  });

  it("does not offer release for a replacement session", async () => {
    mocks.current.mockResolvedValue({ session: { id: "new-session", kind: "recording", phase: "recording" } });
    renderNotice();
    await waitFor(() => expect(mocks.current).toHaveBeenCalled());
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("retains the support acknowledgement and shows a release error in the toast", async () => {
    mocks.stop.mockRejectedValue(new Error("CAN release did not confirm"));
    renderNotice();
    const button = await screen.findByRole("button", { name: "dialogs.teleop.releaseTorque" });
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(button);
    expect(await screen.findByRole("alert")).toHaveTextContent("CAN release did not confirm");
    expect(button).toBeEnabled();
  });

  it("reoffers recovery after another toast replaces it without releasing torque", async () => {
    renderNotice();
    await screen.findByRole("checkbox");
    act(() => { toast({ title: "Another message" }); });
    expect(screen.queryByRole("checkbox")).toBeNull();
    await waitFor(() => expect(screen.getByRole("checkbox")).toBeInTheDocument(), { timeout: 3000 });
    expect(mocks.stop).not.toHaveBeenCalled();
  });
});
