import { apiRequest, type Fetcher } from "./apiClient";

export interface RemoteTeleoperationStatus {
  simulation_only: boolean;
  live_hardware_enabled: boolean;
  state: string;
  status: {
    authority?: {
      highest_sequence?: number | null;
      counters?: Record<string, number>;
    };
    watchdog_ms?: number;
  } | null;
  recorded_events: number;
}

export interface ServoMotorHealth {
  joint: string;
  id: number;
  model: string;
  position_degrees: number | null;
  velocity_rpm_estimate: number | null;
  load_percent: number | null;
  current_a: number | null;
  voltage_v: number | null;
  temperature_c: number | null;
  moving: boolean | null;
  torque_enabled: boolean | null;
  faults: string[] | null;
  complete: boolean;
}

export interface ServoArmHealth {
  arm: string;
  available: boolean;
  complete: boolean;
  communication_errors: number;
  last_error: string | null;
  motors: ServoMotorHealth[];
}

export interface ServoHealthStatus {
  read_only: boolean;
  available: boolean;
  complete: boolean;
  owner: string | null;
  arms: ServoArmHealth[];
  last_error: string | null;
  maintenance: { state: string; write_operations?: string[] };
}

export function getRemoteTeleoperationStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal
): Promise<RemoteTeleoperationStatus> {
  return apiRequest(baseUrl, fetcher, "/api/v1/arms/remote-teleoperation", {
    signal,
    action: "Get remote teleoperation status",
  });
}

export function getServoHealthStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal
): Promise<ServoHealthStatus> {
  return apiRequest(baseUrl, fetcher, "/api/v1/arms/servo-health", {
    signal,
    action: "Get servo health",
  });
}

export function countServoFaults(status: ServoHealthStatus | null): number {
  if (!status) return 0;
  return status.arms.reduce(
    (total, arm) =>
      total +
      arm.motors.reduce(
        (armTotal, motor) => armTotal + (motor.faults?.length ?? 0),
        0
      ),
    0
  );
}
