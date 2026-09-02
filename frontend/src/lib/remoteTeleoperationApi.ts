import { apiRequest, type Fetcher } from "./apiClient";

const REMOTE_PATH = "/api/v1/arms/remote-teleoperation";

export type RemoteRole = "robot" | "operator";

export interface RemoteRobotConfiguration {
  node_id: string;
  robot_name: string;
  bind_address: string;
  control_port: number;
  udp_port: number;
  tls_certificate_path: string;
  tls_private_key_path: string;
  leader_calibration_id: string;
  leader_calibration_digest: string;
  action_rate_hz: number;
  action_watchdog_ms: number;
  first_action_deadline_ms: number;
  control_deadline_ms: number;
  browser_deadline_ms: number;
  max_velocity_per_s: number;
  max_acceleration_per_s2: number;
}

export interface RemoteOperatorConfiguration {
  node_id: string;
  robot_id: string;
  leader_robot_name: string;
  control_uri: string;
  certificate_fingerprint: string;
  action_rate_hz: number;
}

export type RemoteConfigurationRequest =
  | {
      role: "robot";
      robot: RemoteRobotConfiguration;
      operator: null;
    }
  | {
      role: "operator";
      robot: null;
      operator: RemoteOperatorConfiguration;
    };

export interface PublicRemoteConfiguration {
  node_id?: string;
  robot_name?: string;
  leader_robot_name?: string;
  robot_id?: string;
  bind_address?: string;
  control_port?: number;
  udp_port?: number;
  control_uri?: string;
  certificate_fingerprint?: string;
  leader_calibration_id?: string;
  leader_calibration_digest?: string;
  action_rate_hz?: number;
  action_watchdog_ms?: number;
  first_action_deadline_ms?: number;
  control_deadline_ms?: number;
  browser_deadline_ms?: number;
  max_velocity_per_s?: number;
  max_acceleration_per_s2?: number;
  tls_certificate_configured?: boolean;
  tls_private_key_configured?: boolean;
}

export interface RemoteStopReceipt {
  accepted?: boolean | null;
  advancement_halted?: boolean | null;
  hardware_stop_completed?: boolean | null;
  torque_disable_requested?: boolean | null;
  torque_off_confirmed?: boolean | null;
  close_completed?: boolean | null;
  verification?: string | null;
  fault?: string | null;
  reason?: string | null;
}

export interface RemoteWatchdogStatus {
  action_remaining_ms?: number | null;
  control_remaining_ms?: number | null;
  browser_remaining_ms?: number | null;
  first_action_remaining_ms?: number | null;
}

export interface RemoteRuntimeDetails {
  mode?: string;
  state?: string;
  owner?: string | null;
  credential_id?: string | null;
  runtime_enabled?: boolean;
  watchdog?: RemoteWatchdogStatus;
  watchdog_ms?: number;
  last_sequence?: number | null;
  highest_sequence?: number | null;
  rejections?: Record<string, number>;
  counters?: Record<string, number>;
  latency_ms?: number | null;
  clock_uncertainty_ms?: number | null;
  fault_lockout?: boolean;
  faults?: string[];
  stop_receipt?: RemoteStopReceipt | null;
  close_receipt?: RemoteStopReceipt | null;
  torque_off_confirmed?: boolean | null;
  authority?: {
    highest_sequence?: number | null;
    counters?: Record<string, number>;
  };
  owner_credential_id?: string | null;
  clock_uncertainty_ns?: number | null;
  fault?: string | null;
  last_stop?: Record<string, unknown> | null;
  active?: {
    owner_credential_id?: string | null;
    watchdog?: RemoteWatchdogStatus;
    executor?: RemoteRuntimeDetails & {
      safety?: Record<string, unknown>;
      latest_latency_ns?: number | null;
    };
    udp?: { counters?: Record<string, number> } | null;
  } | null;
  credentials?: Array<{
    credential_id: string;
    operator_label: string;
    revoked: boolean;
  }>;
}

export interface RemoteCommissioningStatus {
  commissioned: boolean;
  record: {
    profile_digest?: string;
    rig_id?: string;
    follower_calibration_id?: string;
    leader_calibration_id?: string;
    commissioned_at_utc?: string;
  } | null;
  error?: string;
}

export interface RemoteDurableFaultStatus {
  fault_lockout: boolean;
  record: {
    profile_digest?: string;
    reason_code?: string;
    fault_codes?: string[];
    hardware_stop_completed?: boolean;
    device_closed?: boolean;
    torque_off_confirmed?: boolean;
    occurred_at_utc?: string;
  } | null;
  error?: string;
}

export interface RemoteHardwareRegistryStatus {
  held: boolean;
  state: string;
  kind: string | null;
  owner: string | null;
  generation: number;
  pending_unresolved?: boolean;
  pending_kind?: string | null;
  pending_owner?: string | null;
}

export interface RemotePhysicalSafeguardChecks {
  arm_secured: boolean;
  workspace_clear: boolean;
  physical_power_cutoff_reachable: boolean;
  acknowledge_live_torque_enable_risk: boolean;
}

export interface ConfirmedRemotePhysicalSafeguards {
  arm_secured: true;
  workspace_clear: true;
  physical_power_cutoff_reachable: true;
  acknowledge_live_torque_enable_risk: true;
}

/**
 * The status model is intentionally additive. During the contribution's PR
 * sequence the hardware-free foundation reports its runtime under `status`,
 * while the live service may promote the same fields to the top level. The UI
 * reads both without inventing a safer state when a field is absent.
 */
export interface RemoteTeleoperationStatus {
  configured?: boolean;
  role?: RemoteRole | null;
  config?: PublicRemoteConfiguration | null;
  configuration?: PublicRemoteConfiguration | null;
  runtime_enabled?: boolean;
  live_hardware_enabled?: boolean;
  simulation_only?: boolean;
  state?: string;
  owner?: string | null;
  credential_id?: string | null;
  watchdog?: RemoteWatchdogStatus;
  last_sequence?: number | null;
  rejections?: Record<string, number>;
  latency_ms?: number | null;
  clock_uncertainty_ms?: number | null;
  fault_lockout?: boolean;
  faults?: string[];
  stop_receipt?: RemoteStopReceipt | null;
  close_receipt?: RemoteStopReceipt | null;
  torque_off_confirmed?: boolean | null;
  status?: RemoteRuntimeDetails | null;
  runtime?: RemoteRuntimeDetails | null;
  simulation?: unknown;
  commissioning?: RemoteCommissioningStatus;
  durable_fault?: RemoteDurableFaultStatus;
  hardware_registry?: RemoteHardwareRegistryStatus;
}

export type RemoteCommissioningAction = "commission" | "recover" | null;

export function remoteCommissioningAction(
  status: RemoteTeleoperationStatus | null,
): RemoteCommissioningAction {
  if (
    !status?.configured ||
    status.role !== "robot" ||
    status.runtime_enabled
  ) {
    return null;
  }
  if (status.durable_fault?.fault_lockout) return "recover";
  return status.hardware_registry?.held ||
    status.hardware_registry?.pending_unresolved
    ? null
    : "commission";
}

export function confirmedRemotePhysicalSafeguards(
  checks: RemotePhysicalSafeguardChecks,
): ConfirmedRemotePhysicalSafeguards | null {
  if (!Object.values(checks).every((value) => value === true)) return null;
  return {
    arm_secured: true,
    workspace_clear: true,
    physical_power_cutoff_reachable: true,
    acknowledge_live_torque_enable_risk: true,
  };
}

export interface RemoteRuntimeView {
  state: string;
  owner: string | null;
  credentialId: string | null;
  enabled: boolean;
  watchdog: RemoteWatchdogStatus;
  lastSequence: number | null;
  rejections: Record<string, number>;
  latencyMs: number | null;
  clockUncertaintyMs: number | null;
  faultLockout: boolean;
  faults: string[];
  stopReceipt: RemoteStopReceipt | null;
  torqueOffConfirmed: boolean | null;
}

const finiteOrNull = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

const recordOrNull = (value: unknown): Record<string, unknown> | null =>
  value != null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : null;

const torqueStateFrom = (
  source: unknown,
): boolean | null | undefined => {
  const record = recordOrNull(source);
  if (
    !record ||
    !Object.prototype.hasOwnProperty.call(record, "torque_off_confirmed")
  ) {
    return undefined;
  }
  const value = record.torque_off_confirmed;
  return typeof value === "boolean" ? value : null;
};

const receiptFrom = (value: unknown): RemoteStopReceipt | null => {
  const receipt = recordOrNull(value);
  if (!receipt) return null;
  const robotReceipt = recordOrNull(receipt.robot_receipt);
  const safety =
    recordOrNull(receipt.safety) ??
    recordOrNull(robotReceipt?.safety) ??
    robotReceipt ??
    receipt;
  const bool = (primary: string, fallback?: string) => {
    const value = safety[primary] ?? (fallback ? safety[fallback] : undefined);
    return typeof value === "boolean" ? value : null;
  };
  return {
    accepted: bool("accepted", "stop_accepted"),
    advancement_halted: bool("advancement_halted", "software_dispatch_halted"),
    hardware_stop_completed: bool("hardware_stop_completed"),
    torque_disable_requested: bool(
      "torque_disable_requested",
      "disable_requested",
    ),
    torque_off_confirmed: bool("torque_off_confirmed"),
    close_completed: bool("close_completed", "hardware_close_completed"),
    verification:
      typeof safety.verification === "string" ? safety.verification : null,
    fault: typeof safety.fault === "string" ? safety.fault : null,
    reason: typeof receipt.reason === "string" ? receipt.reason : null,
  };
};

const rejectionCounters = (
  ...sources: Array<Record<string, number> | undefined>
): Record<string, number> => {
  const output: Record<string, number> = {};
  for (const source of sources) {
    for (const [key, value] of Object.entries(source ?? {})) {
      if (
        Number.isFinite(value) &&
        /(reject|invalid|stale|duplicate|spoof|oversize|rate|future|expired|auth)/i.test(
          key,
        )
      ) {
        output[key] = value;
      }
    }
  }
  return output;
};

export function remoteRuntimeView(
  source: RemoteTeleoperationStatus | null,
): RemoteRuntimeView {
  const nested = source?.runtime ?? source?.status ?? {};
  const active = nested.active ?? null;
  const executor = active?.executor ?? null;
  const executorSafety = recordOrNull(executor?.safety);
  const directTorqueState = torqueStateFrom(source);
  const nestedTorqueState = torqueStateFrom(nested);
  const activeExecutorTorqueState = active
    ? torqueStateFrom(executorSafety)
    : undefined;
  const currentTorqueState =
    directTorqueState !== undefined
      ? directTorqueState
      : nestedTorqueState !== undefined
        ? nestedTorqueState
        : activeExecutorTorqueState;
  const receipt =
    source?.stop_receipt ??
    nested.stop_receipt ??
    receiptFrom(nested.last_stop) ??
    receiptFrom(executorSafety);
  const counters = rejectionCounters(
    source?.rejections,
    nested.rejections,
    nested.authority?.counters,
    nested.counters,
    executor?.authority?.counters,
    executor?.counters,
    active?.udp?.counters,
  );
  const fault = nested.fault;
  const executorFaults = Array.isArray(executorSafety?.faults)
    ? executorSafety.faults.filter((value): value is string => typeof value === "string")
    : [];
  return {
    state: source?.state ?? nested.state ?? "disabled",
    owner:
      source?.owner ??
      nested.owner ??
      source?.hardware_registry?.owner ??
      null,
    credentialId:
      source?.credential_id ??
      nested.credential_id ??
      nested.owner_credential_id ??
      active?.owner_credential_id ??
      null,
    enabled: Boolean(
      source?.runtime_enabled ??
        source?.live_hardware_enabled ??
        nested.runtime_enabled ??
        false,
    ),
    watchdog:
      source?.watchdog ??
      nested.watchdog ??
      active?.watchdog ??
      (nested.watchdog_ms != null
        ? { action_remaining_ms: nested.watchdog_ms }
        : {}),
    lastSequence:
      finiteOrNull(source?.last_sequence) ??
      finiteOrNull(nested.last_sequence) ??
      finiteOrNull(executor?.authority?.highest_sequence) ??
      finiteOrNull(nested.authority?.highest_sequence) ??
      finiteOrNull(nested.highest_sequence),
    rejections: counters,
    latencyMs:
      finiteOrNull(source?.latency_ms) ??
      finiteOrNull(nested.latency_ms) ??
      (finiteOrNull(executor?.latest_latency_ns) == null
        ? null
        : Number(executor?.latest_latency_ns) / 1_000_000),
    clockUncertaintyMs:
      finiteOrNull(source?.clock_uncertainty_ms) ??
      finiteOrNull(nested.clock_uncertainty_ms) ??
      (finiteOrNull(nested.clock_uncertainty_ns) == null
        ? null
        : Number(nested.clock_uncertainty_ns) / 1_000_000),
    faultLockout: Boolean(
      (source?.fault_lockout ??
        nested.fault_lockout ??
        executorSafety?.fault_lockout ??
        source?.durable_fault?.fault_lockout) ??
        (source?.state === "fault_lockout" ||
          source?.state === "fault" ||
          Boolean(fault)),
    ),
    faults:
      source?.faults ??
      nested.faults ??
      executorFaults.concat(
        typeof fault === "string" && fault ? [fault] : [],
        source?.durable_fault?.record?.fault_codes ?? [],
      ),
    stopReceipt: receipt,
    torqueOffConfirmed:
      currentTorqueState !== undefined
        ? currentTorqueState
        : (receipt?.torque_off_confirmed ?? null),
  };
}

export interface PairingWindowResponse {
  open?: boolean;
  robot_address?: string;
  control_port?: number;
  certificate_fingerprint?: string;
  pairing_token?: string;
  expires_at?: string;
  expires_in_ms?: number;
  expires_monotonic_ns?: number;
  payload?: {
    robot_address?: string;
    control_port?: number;
    certificate_fingerprint?: string;
    pairing_token?: string;
    expires_at?: string;
    expires_in_ms?: number;
  };
}

export interface PairRemoteOperatorRequest {
  pairing_token: string;
  operator_label: string;
}

export interface PairRemoteOperatorResponse {
  credential_id: string;
  robot_id?: string;
  certificate_fingerprint?: string;
}

export interface RemoteActionResponse {
  success?: boolean;
  state?: string;
  status?: RemoteTeleoperationStatus;
  stop_receipt?: RemoteStopReceipt | null;
}

export function getRemoteTeleoperationStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<RemoteTeleoperationStatus> {
  return apiRequest(baseUrl, fetcher, REMOTE_PATH, {
    signal,
    action: "Get remote teleoperation status",
  });
}

export function saveRemoteConfiguration(
  baseUrl: string,
  fetcher: Fetcher,
  configuration: RemoteConfigurationRequest,
): Promise<RemoteTeleoperationStatus> {
  return apiRequest(baseUrl, fetcher, `${REMOTE_PATH}/configuration`, {
    method: "PUT",
    body: configuration,
    action: "Save remote teleoperation configuration",
  });
}

export function removeRemoteConfiguration(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<RemoteTeleoperationStatus> {
  return apiRequest(baseUrl, fetcher, `${REMOTE_PATH}/configuration`, {
    method: "DELETE",
    action: "Remove remote teleoperation configuration",
  });
}

const post = <T = RemoteActionResponse>(
  baseUrl: string,
  fetcher: Fetcher,
  suffix: string,
  body?: unknown,
): Promise<T> =>
  apiRequest(baseUrl, fetcher, `${REMOTE_PATH}${suffix}`, {
    method: "POST",
    body,
    action: `Remote teleoperation ${suffix.slice(1)}`,
  });

export const enableRemoteTeleoperation = (baseUrl: string, fetcher: Fetcher) =>
  post(baseUrl, fetcher, "/enable");

export const disableRemoteTeleoperation = (
  baseUrl: string,
  fetcher: Fetcher,
) => post(baseUrl, fetcher, "/disable");

export const stopRemoteTeleoperation = (baseUrl: string, fetcher: Fetcher) =>
  post(baseUrl, fetcher, "/stop", { reason: "local_ui_stop" });

export const sendRemoteBrowserHeartbeat = (
  baseUrl: string,
  fetcher: Fetcher,
) => post(baseUrl, fetcher, "/browser-heartbeat");

export const openRemotePairingWindow = (
  baseUrl: string,
  fetcher: Fetcher,
) => post<PairingWindowResponse>(baseUrl, fetcher, "/pairing-window");

export const pairRemoteOperator = (
  baseUrl: string,
  fetcher: Fetcher,
  request: PairRemoteOperatorRequest,
) =>
  post<PairRemoteOperatorResponse>(baseUrl, fetcher, "/pair", request);

export const commissionRemoteFollower = (
  baseUrl: string,
  fetcher: Fetcher,
  safeguards: ConfirmedRemotePhysicalSafeguards,
) => post<RemoteTeleoperationStatus>(baseUrl, fetcher, "/commission", safeguards);

export const recoverRemoteHardware = (
  baseUrl: string,
  fetcher: Fetcher,
  safeguards: ConfirmedRemotePhysicalSafeguards,
) =>
  post<RemoteTeleoperationStatus>(
    baseUrl,
    fetcher,
    "/recover-hardware",
    safeguards,
  );

export const revokeRemoteCredential = (
  baseUrl: string,
  fetcher: Fetcher,
  credentialId: string,
) =>
  post(
    baseUrl,
    fetcher,
    `/credentials/${encodeURIComponent(credentialId)}/revoke`,
  );
