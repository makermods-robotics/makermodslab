import { describe, expect, it } from "vitest";
import {
  countServoFaults,
  getRemoteTeleoperationStatus,
  getServoHealthStatus,
  type ServoHealthStatus,
} from "./armStatusApi";

const response = (body: unknown) =>
  Promise.resolve(
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    })
  );

describe("arm status API", () => {
  it("uses only versioned read-only routes", async () => {
    const paths: string[] = [];
    const fetcher = (url: string) => {
      paths.push(url);
      return response({
        simulation_only: true,
        live_hardware_enabled: false,
        state: "idle",
        status: null,
        recorded_events: 0,
      });
    };
    await getRemoteTeleoperationStatus("http://robot", fetcher);
    await getServoHealthStatus("http://robot", fetcher);
    expect(paths).toEqual([
      "http://robot/api/v1/arms/remote-teleoperation",
      "http://robot/api/v1/arms/servo-health",
    ]);
  });

  it("counts only explicit decoded faults", () => {
    const status = {
      read_only: true,
      available: true,
      complete: true,
      owner: "teleoperation",
      last_error: null,
      maintenance: { state: "disabled" },
      arms: [
        {
          arm: "left",
          available: true,
          complete: true,
          communication_errors: 0,
          last_error: null,
          motors: [
            { faults: ["over_temperature", "over_current"] },
            { faults: null },
            { faults: [] },
          ],
        },
      ],
    } as unknown as ServoHealthStatus;
    expect(countServoFaults(status)).toBe(2);
    expect(countServoFaults(null)).toBe(0);
  });
});
