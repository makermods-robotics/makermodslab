import { describe, expect, it } from "vitest";
import i18n from "@/i18n";
import type { TFunction } from "i18next";
import { ApiError } from "@/lib/apiClient";
import {
  formatRemoteRefusal,
  formatStationRefusal,
  remoteCameraUrl,
} from "@/lib/remoteApi";

/** A coded refusal the way apiRequest surfaces it. */
function refusal(code: string | null, detail = "server prose", status = 409) {
  return new ApiError("Start session failed", status, detail, code);
}

describe("formatRemoteRefusal", () => {
  const t = i18n.getFixedT("en") as unknown as TFunction;

  it("maps each remote-teleop refusal code to its catalog line", () => {
    expect(formatRemoteRefusal(t, refusal("node.not_hosting"), "fb")).toEqual({
      message:
        "That station isn't hosting. Start it there with makermodslab --sfu --host <robot> first.",
      needsInstall: false,
    });
    expect(
      formatRemoteRefusal(t, refusal("robot.schema_mismatch"), "fb")?.message,
    ).toMatch(/doesn't match the hosted one/);
    expect(
      formatRemoteRefusal(t, refusal("sfu.disabled"), "fb")?.message,
    ).toMatch(/makermodslab --sfu/);
  });

  it("renders the taken seat as a friendly refusal, never the holder id", () => {
    const taken = new ApiError("seat", 409, "seat held by op-1", "sfu.seat_taken", {
      holder: "op-1",
    });
    expect(formatRemoteRefusal(t, taken, "fb")).toEqual({
      message:
        "Someone else is driving this robot. Wait for them to stop, then try again.",
      needsInstall: false,
    });
  });

  it("flags system.extra_missing so the caller offers the install flow", () => {
    expect(formatRemoteRefusal(t, refusal("system.extra_missing"), "fb")).toEqual({
      message: "The remote-teleoperation extra isn't installed on this node.",
      needsInstall: true,
    });
  });

  it("routes session.held through the shared busy line", () => {
    const held = new ApiError("busy", 409, "held", "session.held", {
      holder: { kind: "hosting", session_id: "s1" },
    });
    expect(formatRemoteRefusal(t, held, "fb")?.message).toBe(
      "The robot is busy — remote-teleop hosting is running. Stop it first.",
    );
  });

  it("shows the server's prose for any other coded or uncoded refusal", () => {
    expect(formatRemoteRefusal(t, refusal("robot.not_ready"), "fb")).toEqual({
      message: "server prose",
      needsInstall: false,
    });
    expect(
      formatRemoteRefusal(t, new ApiError("x", 500, null, null), "fb"),
    ).toEqual({ message: "fb", needsInstall: false });
  });

  it("returns null for a non-API failure", () => {
    expect(formatRemoteRefusal(t, new TypeError("fetch failed"), "fb")).toBeNull();
  });
});

describe("remoteCameraUrl", () => {
  it("builds the v1 re-stream URL with the camera name escaped", () => {
    expect(remoteCameraUrl("http://x", "wrist cam")).toBe(
      "http://x/api/v1/remote-teleoperation/camera/wrist%20cam",
    );
  });
});

describe("formatStationRefusal", () => {
  const t = i18n.getFixedT("en") as unknown as TFunction;

  it("renders session.held as the station's own line, never the holder", () => {
    const held = new ApiError("busy", 409, "An operator ('op-1') is driving", "session.held", {
      holder: { kind: "hosting", session_id: null },
    });
    expect(formatStationRefusal(t, held, "fb")).toBe(
      "An operator is driving right now — change the hosted robot once they leave.",
    );
  });

  it("shows the server's prose for robot.not_ready / robot.not_found", () => {
    expect(formatStationRefusal(t, refusal(`robot.not_ready`, "needs a port", 400), "fb")).toBe(
      "needs a port",
    );
    expect(formatStationRefusal(t, new ApiError("x", 404, null, `robot.not_found`), "fb")).toBe(
      "fb",
    );
  });

  it("returns null for a non-API failure", () => {
    expect(formatStationRefusal(t, new TypeError("fetch failed"), "fb")).toBeNull();
  });
});
