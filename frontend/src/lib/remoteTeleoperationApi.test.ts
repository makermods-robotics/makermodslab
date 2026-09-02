import { describe, expect, it } from "vitest";
import {
  commissionRemoteFollower,
  confirmedRemotePhysicalSafeguards,
  disableRemoteTeleoperation,
  enableRemoteTeleoperation,
  getRemoteTeleoperationStatus,
  openRemotePairingWindow,
  pairRemoteOperator,
  recoverRemoteHardware,
  removeRemoteConfiguration,
  remoteCommissioningAction,
  remoteRuntimeView,
  revokeRemoteCredential,
  saveRemoteConfiguration,
  sendRemoteBrowserHeartbeat,
  stopRemoteTeleoperation,
} from "./remoteTeleoperationApi";

const ok = (body: unknown = {}) =>
  Promise.resolve(
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    }),
  );

describe("remote teleoperation API", () => {
  it("uses only the dedicated versioned local routes", async () => {
    const requests: Array<{ url: string; options?: RequestInit }> = [];
    const fetcher = (url: string, options?: RequestInit) => {
      requests.push({ url, options });
      return ok({ credential_id: "operator-1" });
    };
    const baseUrl = "http://127.0.0.1:8000";

    await getRemoteTeleoperationStatus(baseUrl, fetcher);
    await enableRemoteTeleoperation(baseUrl, fetcher);
    await disableRemoteTeleoperation(baseUrl, fetcher);
    await removeRemoteConfiguration(baseUrl, fetcher);
    await openRemotePairingWindow(baseUrl, fetcher);
    const safeguards = {
      arm_secured: true,
      workspace_clear: true,
      physical_power_cutoff_reachable: true,
      acknowledge_live_torque_enable_risk: true,
    } as const;
    await commissionRemoteFollower(baseUrl, fetcher, safeguards);
    await recoverRemoteHardware(baseUrl, fetcher, safeguards);
    await pairRemoteOperator(baseUrl, fetcher, {
      pairing_token: "one-time-secret",
      operator_label: "bench laptop",
    });
    await sendRemoteBrowserHeartbeat(baseUrl, fetcher);
    await stopRemoteTeleoperation(baseUrl, fetcher);
    await revokeRemoteCredential(baseUrl, fetcher, "operator/one");

    expect(requests.map(({ url }) => url)).toEqual([
      `${baseUrl}/api/v1/arms/remote-teleoperation`,
      `${baseUrl}/api/v1/arms/remote-teleoperation/enable`,
      `${baseUrl}/api/v1/arms/remote-teleoperation/disable`,
      `${baseUrl}/api/v1/arms/remote-teleoperation/configuration`,
      `${baseUrl}/api/v1/arms/remote-teleoperation/pairing-window`,
      `${baseUrl}/api/v1/arms/remote-teleoperation/commission`,
      `${baseUrl}/api/v1/arms/remote-teleoperation/recover-hardware`,
      `${baseUrl}/api/v1/arms/remote-teleoperation/pair`,
      `${baseUrl}/api/v1/arms/remote-teleoperation/browser-heartbeat`,
      `${baseUrl}/api/v1/arms/remote-teleoperation/stop`,
      `${baseUrl}/api/v1/arms/remote-teleoperation/credentials/operator%2Fone/revoke`,
    ]);
    expect(requests.map(({ options }) => options?.method ?? "GET")).toEqual([
      "GET",
      "POST",
      "POST",
      "DELETE",
      "POST",
      "POST",
      "POST",
      "POST",
      "POST",
      "POST",
      "POST",
    ]);
    expect(requests[5].options?.body).toBe(JSON.stringify(safeguards));
    expect(requests[6].options?.body).toBe(JSON.stringify(safeguards));
    expect(requests[7].options?.body).toBe(
      JSON.stringify({
        pairing_token: "one-time-secret",
        operator_label: "bench laptop",
      }),
    );
    expect(requests[7].url).not.toContain("one-time-secret");
    expect(requests[9].options?.body).toBe(
      JSON.stringify({ reason: "local_ui_stop" }),
    );
  });

  it("sends role configuration as one discriminated PUT body", async () => {
    let captured: { url: string; options?: RequestInit } | null = null;
    const fetcher = (url: string, options?: RequestInit) => {
      captured = { url, options };
      return ok({ configured: true, role: "operator" });
    };
    await saveRemoteConfiguration("http://local", fetcher, {
      role: "operator",
      robot: null,
      operator: {
        node_id: "operator-one",
        robot_id: "robot-one",
        leader_robot_name: "so101-pair",
        control_uri: "wss://100.64.0.2:8443",
        certificate_fingerprint: "ab".repeat(32),
        action_rate_hz: 50,
      },
    });

    expect(captured).toMatchObject({
      url: "http://local/api/v1/arms/remote-teleoperation/configuration",
      options: { method: "PUT" },
    });
    expect(JSON.parse(String(captured!.options?.body))).toMatchObject({
      role: "operator",
      robot: null,
      operator: { leader_robot_name: "so101-pair" },
    });
  });

  it("sends every robot timing and motion limit in the exact PUT body", async () => {
    let body: unknown;
    const fetcher = (_url: string, options?: RequestInit) => {
      body = JSON.parse(String(options?.body));
      return ok({ configured: true, role: "robot" });
    };
    const robot = {
      node_id: "robot-one",
      robot_name: "so101-pair",
      bind_address: "100.64.0.2",
      control_port: 8443,
      udp_port: 8411,
      tls_certificate_path: "/private/cert.pem",
      tls_private_key_path: "/private/key.pem",
      leader_calibration_id: "leader-calibration",
      leader_calibration_digest: "ab".repeat(32),
      action_rate_hz: 50,
      action_watchdog_ms: 240,
      first_action_deadline_ms: 850,
      control_deadline_ms: 900,
      browser_deadline_ms: 1800,
      max_velocity_per_s: 35.5,
      max_acceleration_per_s2: 120.25,
    };

    await saveRemoteConfiguration("http://local", fetcher, {
      role: "robot",
      robot,
      operator: null,
    });

    expect(body).toEqual({ role: "robot", robot, operator: null });
  });

  it("normalizes foundation and live status without claiming unknown torque is off", () => {
    const view = remoteRuntimeView({
      live_hardware_enabled: true,
      state: "streaming",
      status: {
        authority: {
          highest_sequence: 18,
          counters: { duplicate: 2 },
        },
        watchdog_ms: 73,
        stop_receipt: { torque_disable_requested: true },
      },
    });
    expect(view.enabled).toBe(true);
    expect(view.lastSequence).toBe(18);
    expect(view.rejections).toEqual({ duplicate: 2 });
    expect(view.watchdog.action_remaining_ms).toBe(73);
    expect(view.torqueOffConfirmed).toBeNull();
  });

  it("flattens the robot service's watchdog, executor, UDP, and safety status", () => {
    const view = remoteRuntimeView({
      configured: true,
      role: "robot",
      runtime_enabled: true,
      state: "active",
      runtime: {
        state: "active",
        runtime_enabled: true,
        active: {
          owner_credential_id: "operator-7",
          watchdog: {
            action_remaining_ms: 88,
            control_remaining_ms: 470,
            browser_remaining_ms: 900,
          },
          udp: { counters: { datagram_rejected: 3, datagram_dispatched: 20 } },
          executor: {
            authority: { highest_sequence: 21, counters: { accepted: 20 } },
            safety: {
              stop_accepted: true,
              software_dispatch_halted: true,
              disable_requested: true,
              hardware_stop_completed: true,
              hardware_close_completed: false,
              torque_off_confirmed: null,
              fault_lockout: true,
              faults: ["close:TimeoutError"],
            },
          },
        },
      },
    });

    expect(view.credentialId).toBe("operator-7");
    expect(view.lastSequence).toBe(21);
    expect(view.watchdog.action_remaining_ms).toBe(88);
    expect(view.rejections).toEqual({ datagram_rejected: 3 });
    expect(view.faultLockout).toBe(true);
    expect(view.faults).toEqual(["close:TimeoutError"]);
    expect(view.stopReceipt).toMatchObject({
      accepted: true,
      advancement_halted: true,
      torque_disable_requested: true,
      torque_off_confirmed: null,
      close_completed: false,
    });
  });

  it("prefers an active unknown torque state over a stale confirmed-off receipt", () => {
    const view = remoteRuntimeView({
      runtime_enabled: true,
      state: "active",
      stop_receipt: { torque_off_confirmed: true },
      runtime: {
        state: "active",
        active: {
          executor: {
            safety: { torque_off_confirmed: null },
          },
        },
      },
    });

    expect(view.stopReceipt?.torque_off_confirmed).toBe(true);
    expect(view.torqueOffConfirmed).toBeNull();
  });

  it("keeps the robot's acknowledged STOP evidence visible on the operator host", () => {
    const view = remoteRuntimeView({
      role: "operator",
      state: "idle",
      runtime: {
        state: "idle",
        credential_id: "operator-9",
        last_stop: {
          robot_confirmation_available: true,
          leader_closed: true,
          robot_receipt: {
            reason: "local_ui_stop",
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
    });

    expect(view.credentialId).toBe("operator-9");
    expect(view.stopReceipt).toMatchObject({
      accepted: true,
      advancement_halted: true,
      torque_disable_requested: true,
      hardware_stop_completed: true,
      close_completed: true,
      torque_off_confirmed: true,
    });
    expect(view.torqueOffConfirmed).toBe(true);
  });

  it("requires every physical safeguard and selects only an eligible robot action", () => {
    const incomplete = confirmedRemotePhysicalSafeguards({
      arm_secured: true,
      workspace_clear: true,
      physical_power_cutoff_reachable: false,
      acknowledge_live_torque_enable_risk: true,
    });
    expect(incomplete).toBeNull();

    const complete = confirmedRemotePhysicalSafeguards({
      arm_secured: true,
      workspace_clear: true,
      physical_power_cutoff_reachable: true,
      acknowledge_live_torque_enable_risk: true,
    });
    expect(complete).toEqual({
      arm_secured: true,
      workspace_clear: true,
      physical_power_cutoff_reachable: true,
      acknowledge_live_torque_enable_risk: true,
    });

    expect(
      remoteCommissioningAction({
        configured: true,
        role: "robot",
        runtime_enabled: false,
        durable_fault: { fault_lockout: false, record: null },
        hardware_registry: {
          held: false,
          state: "free",
          kind: null,
          owner: null,
          generation: 1,
        },
      }),
    ).toBe("commission");
    expect(
      remoteCommissioningAction({
        configured: true,
        role: "robot",
        runtime_enabled: false,
        durable_fault: { fault_lockout: true, record: null },
        hardware_registry: {
          held: true,
          state: "unresolved",
          kind: null,
          owner: null,
          generation: 2,
        },
      }),
    ).toBe("recover");
    expect(
      remoteCommissioningAction({
        configured: true,
        role: "robot",
        runtime_enabled: false,
        durable_fault: { fault_lockout: false, record: null },
        hardware_registry: {
          held: false,
          state: "free",
          kind: null,
          owner: null,
          generation: 3,
          pending_unresolved: true,
          pending_kind: "remote_recovery",
          pending_owner: "durable:robot-fault",
        },
      }),
    ).toBeNull();
    expect(
      remoteCommissioningAction({
        configured: true,
        role: "robot",
        runtime_enabled: true,
      }),
    ).toBeNull();
  });
});
