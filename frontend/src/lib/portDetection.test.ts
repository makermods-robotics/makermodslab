import { describe, expect, it, vi } from "vitest";
import { detectArmPort } from "./portDetection";

const response = (data: object) => new Response(JSON.stringify(data));

describe("port identification", () => {
  it.each(["teleop", "robot"] as const)(
    "explicit auto detection identifies the %s by protocol without a swing",
    async (device) => {
      const request = vi.fn().mockResolvedValue(
        response({
          success: true,
          follower_ports: ["usbmodem"],
          leader_ports: ["usbserial"],
        }),
      );
      const result = await detectArmPort(
        request,
        "",
        "metal",
        device,
        "Select and wiggle",
        "auto",
      );
      expect(result).toMatchObject({
        success: true,
        port: device === "teleop" ? "usbserial" : "usbmodem",
      });
      expect(request).toHaveBeenCalledTimes(1);
      expect(request.mock.calls[0][0]).toBe("/api/v1/maker/probe-ports");
    },
  );

  it("auto detecting a Star leader refuses an ambiguous follower setup", async () => {
    const request = vi.fn().mockResolvedValue(
      response({
        success: true,
        follower_ports: ["f1", "f2"],
        leader_ports: ["l1"],
      }),
    );
    const result = await detectArmPort(
      request,
      "",
      "metal",
      "teleop",
      "Select and wiggle",
      "auto",
    );
    expect(result).toEqual({
      success: false,
      multiple: true,
      message: "Select and wiggle",
    });
    expect(request).toHaveBeenCalledTimes(1);
  });

  it.each(["so101", "maker", "metal"])(
    "%s leaders always wait for a swing",
    async (family) => {
      const request = vi
        .fn()
        .mockResolvedValue(response({ success: true, port: "swung" }));
      const result = await detectArmPort(
        request,
        "",
        family,
        "teleop",
        "Select and wiggle",
        "swing",
        { portProbe: family !== "so101" },
      );
      expect(result.port).toBe("swung");
      expect(request).toHaveBeenCalledTimes(1);
      expect(request.mock.calls[0][0]).toBe(
        `/api/v1/${family === "so101" ? "" : "maker/"}identify-arm`,
      );
      if (family !== "so101") {
        expect(JSON.parse(request.mock.calls[0][1].body)).toEqual({
          arm_type: family,
          device_type: "teleop",
        });
      }
    },
  );

  it.each(["maker", "metal"])(
    "%s only assigns a single unambiguous follower",
    async (family) => {
      const request = vi.fn().mockResolvedValue(
        response({
          success: true,
          follower_ports: ["f1"],
          leader_ports: ["l1"],
        }),
      );
      const result = await detectArmPort(
        request,
        "",
        family,
        "robot",
        "Select and wiggle",
      );
      expect(result).toMatchObject({ success: true, port: "f1" });
      expect(request.mock.calls[0][0]).toBe("/api/v1/maker/probe-ports");
    },
  );

  it.each([
    { follower_ports: ["f1", "f2"], leader_ports: ["l1"] },
    { follower_ports: ["f1"], leader_ports: ["l1", "l2"] },
    { follower_ports: ["f1", "f2"], leader_ports: ["l1", "l2"] },
  ])(
    "never assigns or asks to swing a follower when either role is ambiguous: %j",
    async (ports) => {
      const request = vi
        .fn()
        .mockResolvedValue(response({ success: true, ...ports }));
      const result = await detectArmPort(
        request,
        "",
        "metal",
        "robot",
        "Select and wiggle",
      );
      expect(result).toEqual({
        success: false,
        multiple: true,
        message: "Select and wiggle",
      });
      expect(request).toHaveBeenCalledTimes(1);
    },
  );

  it("does not fall back to follower motion when nothing answers", async () => {
    const request = vi
      .fn()
      .mockResolvedValue(response({ success: false, message: "No hardware" }));
    expect(
      await detectArmPort(request, "", "metal", "robot", "Select and wiggle"),
    ).toEqual({ success: false, message: "No hardware" });
    expect(request).toHaveBeenCalledTimes(1);
  });
});
