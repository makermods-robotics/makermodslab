import { describe, expect, it } from "vitest";
import i18n from "@/i18n";
import { calibrationErrorMessage } from "./calibrationError";

describe("calibration error toasts", () => {
  it.each([
    "Handshake failed. Missing motors: shoulder, gripper",
    "Read timeout",
  ])("explains an unresponsive arm briefly: %s", (error) => {
    expect(calibrationErrorMessage(error, i18n.t)).toBe(
      "Arm not responding. Check power and cable.",
    );
  });

  it.each([
    ["No response from motor 'shoulder_pan' (send ID: 0x01).", "shoulder pan"],
    ["Servo shoulder (id=1) has never responded.", "shoulder"],
    [
      "Failed to connect to CAN bus: Handshake failed. The following motors did not respond: ['gripper']. Check power (24V) and CAN wiring.",
      "gripper",
    ],
  ])("names the joint that failed: %s", (error, joints) => {
    expect(calibrationErrorMessage(error, i18n.t)).toBe(
      `No response from ${joints}. Check power and wiring.`,
    );
  });

  it("preserves a different cause instead of blaming power", () => {
    expect(calibrationErrorMessage("Disk full", i18n.t)).toBe("Disk full");
    expect(
      calibrationErrorMessage("Fault reported by motor 'shoulder'", i18n.t),
    ).toBe("Motor reported a fault. Check the arm before retrying.");
    expect(
      calibrationErrorMessage(
        "Handshake failed. reported fault: ['shoulder']",
        i18n.t,
      ),
    ).toBe("Motor reported a fault. Check the arm before retrying.");
    expect(
      calibrationErrorMessage(
        "Timed out waiting for the zero pose to be confirmed.",
        i18n.t,
      ),
    ).toBe("Timed out waiting for the zero pose to be confirmed.");
  });

  it("keeps unexpected errors to one short line", () => {
    expect(calibrationErrorMessage("Disk full\nTraceback: ...", i18n.t)).toBe(
      "Disk full",
    );
    expect(
      calibrationErrorMessage("x".repeat(200), i18n.t).length,
    ).toBeLessThanOrEqual(160);
  });
});
