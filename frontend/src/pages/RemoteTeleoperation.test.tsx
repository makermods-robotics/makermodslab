import { MemoryRouter } from "react-router-dom";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { TooltipProvider } from "@/components/ui/tooltip";
import type { RobotRecord } from "@/hooks/useRobots";
import type {
  RemoteConfigurationRequest,
  RemoteTeleoperationStatus,
} from "@/lib/remoteTeleoperationApi";
import {
  commissionRemoteFollower,
  disableRemoteTeleoperation,
  enableRemoteTeleoperation,
  getRemoteTeleoperationStatus,
  pairRemoteOperator,
  removeRemoteConfiguration,
  saveRemoteConfiguration,
  sendRemoteBrowserHeartbeat,
  stopRemoteTeleoperation,
} from "@/lib/remoteTeleoperationApi";
import RemoteTeleoperationPage from "./RemoteTeleoperation";

const apiMocks = vi.hoisted(() => ({
  commission: vi.fn(),
  disable: vi.fn(),
  enable: vi.fn(),
  getStatus: vi.fn(),
  heartbeat: vi.fn(),
  openPairingWindow: vi.fn(),
  pair: vi.fn(),
  recover: vi.fn(),
  remove: vi.fn(),
  revoke: vi.fn(),
  save: vi.fn(),
  stop: vi.fn(),
}));

const fetchWithHeaders = vi.hoisted(() => vi.fn());

vi.mock("@/contexts/ApiContext", () => ({
  useApi: () => ({
    baseUrl: "http://127.0.0.1:8000",
    wsBaseUrl: "ws://127.0.0.1:8000",
    fetchWithHeaders,
  }),
}));

const so101Record: RobotRecord = {
  name: "so101-bench",
  mode: "single",
  arm_type: "so101",
  leader_port: "/dev/leader",
  follower_port: "/dev/follower",
  leader_config: "leader-calibration-v1",
  follower_config: "follower-calibration-v1",
  right_leader_port: "",
  right_follower_port: "",
  right_leader_config: "",
  right_follower_config: "",
  cameras: [],
  motor_power: 38,
  is_clean: true,
  follower_ready: true,
};

vi.mock("@/hooks/useRobots", () => ({
  useRobots: () => ({ records: { [so101Record.name]: so101Record } }),
}));

vi.mock("@/lib/remoteTeleoperationApi", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/remoteTeleoperationApi")>()),
  commissionRemoteFollower: apiMocks.commission,
  disableRemoteTeleoperation: apiMocks.disable,
  enableRemoteTeleoperation: apiMocks.enable,
  getRemoteTeleoperationStatus: apiMocks.getStatus,
  openRemotePairingWindow: apiMocks.openPairingWindow,
  pairRemoteOperator: apiMocks.pair,
  recoverRemoteHardware: apiMocks.recover,
  removeRemoteConfiguration: apiMocks.remove,
  revokeRemoteCredential: apiMocks.revoke,
  saveRemoteConfiguration: apiMocks.save,
  sendRemoteBrowserHeartbeat: apiMocks.heartbeat,
  stopRemoteTeleoperation: apiMocks.stop,
}));

const registryFree = {
  held: false,
  state: "free",
  kind: null,
  owner: null,
  generation: 1,
};

const dormantStatus = (): RemoteTeleoperationStatus => ({
  configured: false,
  role: null,
  configuration: null,
  runtime_enabled: false,
  state: "disabled",
  commissioning: { commissioned: false, record: null },
  durable_fault: { fault_lockout: false, record: null },
  hardware_registry: registryFree,
});

let backendStatus: RemoteTeleoperationStatus;

const cloneStatus = (): RemoteTeleoperationStatus =>
  JSON.parse(JSON.stringify(backendStatus)) as RemoteTeleoperationStatus;

const renderPage = () =>
  render(
    <MemoryRouter>
      <TooltipProvider>
        <RemoteTeleoperationPage />
      </TooltipProvider>
    </MemoryRouter>,
  );

const change = (label: string | RegExp, value: string) => {
  fireEvent.change(screen.getByLabelText(label), { target: { value } });
};

const confirmPhysicalSafeguards = () => {
  fireEvent.click(
    screen.getByLabelText(
      "The follower is physically secured against unexpected motion or a drop.",
    ),
  );
  fireEvent.click(
    screen.getByLabelText(
      "The workspace is clear and everyone is outside the arm's reach.",
    ),
  );
  fireEvent.click(
    screen.getByLabelText("A physical power cutoff is immediately reachable."),
  );
  fireEvent.click(
    screen.getByLabelText(
      "I understand that live execution enables Feetech torque and requires an immediately reachable physical cutoff.",
    ),
  );
};

beforeEach(() => {
  backendStatus = dormantStatus();
  localStorage.clear();
  vi.clearAllMocks();
  vi.mocked(getRemoteTeleoperationStatus).mockImplementation(async () =>
    cloneStatus(),
  );
  vi.mocked(sendRemoteBrowserHeartbeat).mockResolvedValue({ success: true });
  apiMocks.openPairingWindow.mockResolvedValue({});
  apiMocks.recover.mockResolvedValue({ success: true });
  apiMocks.revoke.mockResolvedValue({ success: true });
});

describe("RemoteTeleoperationPage full lifecycle", () => {
  it("uses the published trial ports by default", async () => {
    renderPage();
    await waitFor(() =>
      expect(getRemoteTeleoperationStatus).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /Remote robot/ }));
    expect(screen.getByLabelText("TLS control port")).toHaveValue(7443);
    expect(screen.getByLabelText("UDP action port")).toHaveValue(7444);
  });

  it("configures, commissions, starts, stops, disables, and removes the robot role", async () => {
    const certificatePath = "/opt/makermods/tls/robot-cert.pem";
    const privateKeyPath = "/opt/makermods/tls/robot-key.pem";
    const calibrationDigest = "ab".repeat(32);
    let submittedConfiguration: RemoteConfigurationRequest | null = null;

    vi.mocked(saveRemoteConfiguration).mockImplementation(
      async (_baseUrl, _fetcher, configuration) => {
        submittedConfiguration = configuration;
        backendStatus = {
          configured: true,
          role: "robot",
          configuration: {
            node_id: "robot-laptop",
            robot_name: "so101-bench",
            bind_address: "100.64.0.2",
            control_port: 9443,
            udp_port: 9411,
            leader_calibration_id: "leader-calibration-v1",
            leader_calibration_digest: calibrationDigest,
            action_rate_hz: 60,
            action_watchdog_ms: 240,
            first_action_deadline_ms: 850,
            control_deadline_ms: 900,
            browser_deadline_ms: 1800,
            max_velocity_per_s: 35.5,
            max_acceleration_per_s2: 120.25,
            tls_certificate_configured: true,
            tls_private_key_configured: true,
          },
          runtime_enabled: false,
          state: "configured",
          commissioning: { commissioned: false, record: null },
          durable_fault: { fault_lockout: false, record: null },
          hardware_registry: registryFree,
        };
        return cloneStatus();
      },
    );
    vi.mocked(commissionRemoteFollower).mockImplementation(
      async (_baseUrl, _fetcher, safeguards) => {
        backendStatus = {
          ...backendStatus,
          commissioning: {
            commissioned: true,
            record: { profile_digest: "cd".repeat(32) },
          },
        };
        return { success: true, status: cloneStatus(), safeguards };
      },
    );
    vi.mocked(enableRemoteTeleoperation).mockImplementation(async () => {
      backendStatus = {
        ...backendStatus,
        runtime_enabled: true,
        state: "listening",
        runtime: {
          state: "listening",
          runtime_enabled: true,
          owner: "remote_robot:robot-laptop",
          torque_off_confirmed: true,
        },
      };
      return { success: true, status: cloneStatus() };
    });
    vi.mocked(stopRemoteTeleoperation).mockImplementation(async () => {
      backendStatus = {
        ...backendStatus,
        state: "stopped",
        stop_receipt: {
          accepted: true,
          advancement_halted: true,
          hardware_stop_completed: true,
          torque_disable_requested: true,
          torque_off_confirmed: true,
          close_completed: true,
        },
      };
      return { success: true, status: cloneStatus() };
    });
    vi.mocked(disableRemoteTeleoperation).mockImplementation(async () => {
      backendStatus = {
        ...backendStatus,
        runtime_enabled: false,
        state: "disabled",
        runtime: null,
      };
      return { success: true, status: cloneStatus() };
    });
    vi.mocked(removeRemoteConfiguration).mockImplementation(async () => {
      backendStatus = dormantStatus();
      return cloneStatus();
    });

    renderPage();
    await waitFor(() =>
      expect(getRemoteTeleoperationStatus).toHaveBeenCalled(),
    );

    fireEvent.click(screen.getByRole("button", { name: /Remote robot/ }));
    change("Laptop node ID", "robot-laptop");
    change("SO-101 follower record", "so101-bench");
    change("Exact private bind IP", "100.64.0.2");
    change("TLS control port", "9443");
    change("UDP action port", "9411");
    change("TLS certificate path", certificatePath);
    change("TLS private key path", privateKeyPath);
    change("Allowed leader calibration ID", "leader-calibration-v1");
    change("Allowed leader calibration SHA-256", calibrationDigest);
    change("Action watchdog (ms)", "240");
    change("First action deadline (ms)", "850");
    change("Control deadline (ms)", "900");
    change("Browser deadline (ms)", "1800");
    change("Max velocity (position units/s)", "35.5");
    change("Max acceleration (position units/s²)", "120.25");
    change("Action rate (Hz)", "60");
    fireEvent.click(screen.getByRole("button", { name: "Save configuration" }));

    await waitFor(() => expect(saveRemoteConfiguration).toHaveBeenCalledOnce());
    expect(submittedConfiguration).toEqual({
      role: "robot",
      robot: {
        node_id: "robot-laptop",
        robot_name: "so101-bench",
        bind_address: "100.64.0.2",
        control_port: 9443,
        udp_port: 9411,
        tls_certificate_path: certificatePath,
        tls_private_key_path: privateKeyPath,
        leader_calibration_id: "leader-calibration-v1",
        leader_calibration_digest: calibrationDigest,
        action_rate_hz: 60,
        action_watchdog_ms: 240,
        first_action_deadline_ms: 850,
        control_deadline_ms: 900,
        browser_deadline_ms: 1800,
        max_velocity_per_s: 35.5,
        max_acceleration_per_s2: 120.25,
      },
      operator: null,
    });
    expect(screen.getByLabelText("TLS certificate path")).toHaveValue("");
    expect(screen.getByLabelText("TLS private key path")).toHaveValue("");
    expect(JSON.stringify(backendStatus)).not.toContain(certificatePath);
    expect(JSON.stringify(backendStatus)).not.toContain(privateKeyPath);
    expect(JSON.stringify({ ...localStorage })).not.toContain(privateKeyPath);

    const enable = screen.getByRole("button", {
      name: "Enable robot listener",
    });
    expect(enable).toBeDisabled();
    confirmPhysicalSafeguards();
    fireEvent.click(
      screen.getByRole("button", { name: "Commission this profile" }),
    );
    await waitFor(() =>
      expect(commissionRemoteFollower).toHaveBeenCalledOnce(),
    );
    expect(commissionRemoteFollower).toHaveBeenCalledWith(
      "http://127.0.0.1:8000",
      fetchWithHeaders,
      {
        arm_secured: true,
        workspace_clear: true,
        physical_power_cutoff_reachable: true,
        acknowledge_live_torque_enable_risk: true,
      },
    );
    await waitFor(() => expect(screen.getByText("Commissioned")).toBeVisible());
    expect(enable).toBeEnabled();

    fireEvent.click(enable);
    await waitFor(() =>
      expect(enableRemoteTeleoperation).toHaveBeenCalledOnce(),
    );
    await waitFor(() => expect(screen.getByText("Enabled")).toBeVisible());

    fireEvent.click(screen.getByRole("button", { name: /^STOPEsc$/ }));
    await waitFor(() => expect(stopRemoteTeleoperation).toHaveBeenCalledOnce());
    await waitFor(() => expect(screen.getAllByText("Yes")).toHaveLength(6));
    expect(screen.getByText("Confirmed off")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Disable role" }));
    await waitFor(() =>
      expect(disableRemoteTeleoperation).toHaveBeenCalledOnce(),
    );
    await waitFor(() => expect(screen.getByText("Disabled")).toBeVisible());

    vi.spyOn(window, "confirm").mockReturnValueOnce(true);
    fireEvent.click(
      screen.getByRole("button", { name: "Remove saved remote config" }),
    );
    await waitFor(() =>
      expect(removeRemoteConfiguration).toHaveBeenCalledOnce(),
    );
    expect(screen.getByText("Not saved")).toBeVisible();
  }, 15_000);

  it("configures and pairs the operator, heartbeats only while enabled, then stops and removes it", async () => {
    const pairingToken = "single-use-pairing-secret";
    const fingerprint = "ef".repeat(32);
    let submittedConfiguration: RemoteConfigurationRequest | null = null;

    vi.mocked(saveRemoteConfiguration).mockImplementation(
      async (_baseUrl, _fetcher, configuration) => {
        submittedConfiguration = configuration;
        backendStatus = {
          configured: true,
          role: "operator",
          configuration: {
            node_id: "operator-laptop",
            robot_id: "robot-laptop",
            leader_robot_name: "so101-bench",
            control_uri: "wss://100.64.0.2:9443",
            certificate_fingerprint: fingerprint,
            action_rate_hz: 60,
          },
          runtime_enabled: false,
          state: "configured",
          commissioning: { commissioned: false, record: null },
          durable_fault: { fault_lockout: false, record: null },
          hardware_registry: registryFree,
        };
        return cloneStatus();
      },
    );
    vi.mocked(pairRemoteOperator).mockImplementation(async () => ({
      credential_id: "credential-operator-1",
      robot_id: "robot-laptop",
      certificate_fingerprint: fingerprint,
    }));
    vi.mocked(enableRemoteTeleoperation).mockImplementation(async () => {
      backendStatus = {
        ...backendStatus,
        runtime_enabled: true,
        state: "streaming",
        runtime: {
          state: "streaming",
          runtime_enabled: true,
          credential_id: "credential-operator-1",
        },
      };
      return { success: true, status: cloneStatus() };
    });
    vi.mocked(stopRemoteTeleoperation).mockImplementation(async () => {
      backendStatus = {
        ...backendStatus,
        state: "stopped",
        runtime: {
          state: "stopped",
          runtime_enabled: true,
          credential_id: "credential-operator-1",
          last_stop: {
            robot_receipt: {
              safety: {
                stop_accepted: true,
                software_dispatch_halted: true,
                disable_requested: true,
                hardware_stop_completed: true,
                hardware_close_completed: true,
                torque_off_confirmed: true,
              },
            },
          },
        },
      };
      return { success: true, status: cloneStatus() };
    });
    vi.mocked(disableRemoteTeleoperation).mockImplementation(async () => {
      backendStatus = {
        ...backendStatus,
        runtime_enabled: false,
        state: "disabled",
        runtime: null,
      };
      return { success: true, status: cloneStatus() };
    });
    vi.mocked(removeRemoteConfiguration).mockImplementation(async () => {
      backendStatus = dormantStatus();
      return cloneStatus();
    });

    renderPage();
    await waitFor(() =>
      expect(getRemoteTeleoperationStatus).toHaveBeenCalled(),
    );
    expect(sendRemoteBrowserHeartbeat).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /Remote operator/ }));
    change("Laptop node ID", "operator-laptop");
    change("SO-101 leader record", "so101-bench");
    change("Robot ID", "robot-laptop");
    change("Robot TLS WebSocket address", "wss://100.64.0.2:9443");
    change("Pinned certificate SHA-256", fingerprint);
    change("Action rate (Hz)", "60");
    fireEvent.click(screen.getByRole("button", { name: "Save configuration" }));

    await waitFor(() => expect(saveRemoteConfiguration).toHaveBeenCalledOnce());
    expect(submittedConfiguration).toEqual({
      role: "operator",
      robot: null,
      operator: {
        node_id: "operator-laptop",
        robot_id: "robot-laptop",
        leader_robot_name: "so101-bench",
        control_uri: "wss://100.64.0.2:9443",
        certificate_fingerprint: fingerprint,
        action_rate_hz: 60,
      },
    });

    change("One-time pairing token", pairingToken);
    change("Operator label", "MakerMods test laptop");
    fireEvent.click(screen.getByRole("button", { name: "Pair operator" }));
    await waitFor(() => expect(pairRemoteOperator).toHaveBeenCalledOnce());
    expect(pairRemoteOperator).toHaveBeenCalledWith(
      "http://127.0.0.1:8000",
      fetchWithHeaders,
      {
        pairing_token: pairingToken,
        operator_label: "MakerMods test laptop",
      },
    );
    expect(screen.getByLabelText("One-time pairing token")).toHaveValue("");
    expect(JSON.stringify(backendStatus)).not.toContain(pairingToken);
    expect(JSON.stringify({ ...localStorage })).not.toContain(pairingToken);

    fireEvent.click(screen.getByRole("button", { name: "Connect operator" }));
    await waitFor(() =>
      expect(enableRemoteTeleoperation).toHaveBeenCalledOnce(),
    );
    await waitFor(() => expect(sendRemoteBrowserHeartbeat).toHaveBeenCalled());
    expect(sendRemoteBrowserHeartbeat).toHaveBeenCalledWith(
      "http://127.0.0.1:8000",
      fetchWithHeaders,
    );

    fireEvent.click(screen.getByRole("button", { name: /^STOPEsc$/ }));
    await waitFor(() => expect(stopRemoteTeleoperation).toHaveBeenCalledOnce());
    await waitFor(() =>
      expect(screen.getByText("Confirmed off")).toBeVisible(),
    );

    fireEvent.click(screen.getByRole("button", { name: "Disable role" }));
    await waitFor(() =>
      expect(disableRemoteTeleoperation).toHaveBeenCalledOnce(),
    );
    const heartbeatCountAfterDisable = vi.mocked(sendRemoteBrowserHeartbeat)
      .mock.calls.length;
    await new Promise((resolve) => window.setTimeout(resolve, 800));
    expect(sendRemoteBrowserHeartbeat).toHaveBeenCalledTimes(
      heartbeatCountAfterDisable,
    );

    vi.spyOn(window, "confirm").mockReturnValueOnce(true);
    fireEvent.click(
      screen.getByRole("button", { name: "Remove saved remote config" }),
    );
    await waitFor(() =>
      expect(removeRemoteConfiguration).toHaveBeenCalledOnce(),
    );
    expect(screen.getByText("Not saved")).toBeVisible();
  }, 15_000);
});
