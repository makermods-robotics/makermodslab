import { describe, expect, it } from "vitest";
import i18n from "@/i18n";
import type { TFunction } from "i18next";
import { ApiError, type Fetcher } from "@/lib/apiClient";
import {
  formatSessionHeld,
  sessionHeldHolder,
  startSession,
  type SessionInfo,
} from "@/lib/sessionApi";

/** A 409 session.held the way apiRequest surfaces it. */
function heldError(kind: string | null, sessionId: string | null = "s1") {
  return new ApiError(
    "Start session failed: busy",
    409,
    "The robot hardware is held by an active session. Stop it first.",
    "session.held",
    { holder: { kind, session_id: sessionId } }
  );
}

const SESSION: SessionInfo = {
  id: "abc123",
  kind: "teleoperation",
  robot: "bench",
  owner: "ui:1",
  started_at: 1,
  revision: 1,
  phase: null,
  lease: null,
};

/** A Fetcher that returns the given 201 body for POST /api/v1/sessions. */
function fetcherReturning(body: unknown): Fetcher {
  return async () =>
    new Response(JSON.stringify(body), {
      status: 201,
      headers: { "Content-Type": "application/json" },
    });
}

describe("startSession warnings relay", () => {
  const args = {
    kind: "teleoperation" as const,
    robot: "bench",
    owner: "ui:1",
    options: {},
  };

  it("passes the 201's warn-but-allow findings through", async () => {
    const { session, warnings } = await startSession(
      "http://x",
      fetcherReturning({ session: SESSION, warnings: ["EEPROM drift"] }),
      args
    );
    expect(session.id).toBe("abc123");
    expect(warnings).toEqual(["EEPROM drift"]);
  });

  it("normalizes an absent or null warnings field to null", async () => {
    for (const body of [
      { session: SESSION },
      { session: SESSION, warnings: null },
    ]) {
      const { warnings } = await startSession(
        "http://x",
        fetcherReturning(body),
        args
      );
      expect(warnings).toBeNull();
    }
  });
});

describe("sessionHeldHolder", () => {
  it("extracts the holder kind from a coded 409", () => {
    expect(sessionHeldHolder(heldError("recording"))).toEqual({
      kind: "recording",
    });
  });

  it("keeps a null holder kind (server couldn't name it)", () => {
    expect(sessionHeldHolder(heldError(null))).toEqual({ kind: null });
  });

  it("returns null for any other error", () => {
    expect(sessionHeldHolder(new Error("boom"))).toBeNull();
    expect(
      sessionHeldHolder(new ApiError("x failed", 400, "bad", "robot.not_ready"))
    ).toBeNull();
    expect(sessionHeldHolder(undefined)).toBeNull();
  });

  it("tolerates a held error whose details are missing", () => {
    const bare = new ApiError("x failed", 409, "busy", "session.held", null);
    expect(sessionHeldHolder(bare)).toEqual({ kind: null });
  });
});

describe("formatSessionHeld", () => {
  const t = i18n.getFixedT("en") as unknown as TFunction;

  it("names the holding activity for every session kind", () => {
    expect(formatSessionHeld(t, heldError("teleoperation"))).toBe(
      "The robot is busy — teleoperation is running. Stop it first."
    );
    expect(formatSessionHeld(t, heldError("recording"))).toBe(
      "The robot is busy — a recording session is running. Stop it first."
    );
    expect(formatSessionHeld(t, heldError("auto_calibration"))).toBe(
      "The robot is busy — an auto-calibration is running. Stop it first."
    );
    expect(formatSessionHeld(t, heldError("hosting"))).toBe(
      "The robot is busy — remote-teleop hosting is running. Stop it first."
    );
    expect(formatSessionHeld(t, heldError("remote_teleoperation"))).toBe(
      "The robot is busy — remote teleoperation is running. Stop it first."
    );
  });

  it("falls back to the generic line for an unnamed or unknown holder", () => {
    expect(formatSessionHeld(t, heldError(null))).toBe(
      "The robot is busy with another session. Stop it first."
    );
    expect(formatSessionHeld(t, heldError("future_kind"))).toBe(
      "The robot is busy with another session. Stop it first."
    );
  });

  it("returns null for anything that isn't a session.held error", () => {
    expect(formatSessionHeld(t, new Error("network down"))).toBeNull();
    expect(
      formatSessionHeld(t, new ApiError("x failed", 409, "busy", null))
    ).toBeNull();
  });
});
