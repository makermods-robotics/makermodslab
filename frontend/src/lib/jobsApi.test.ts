import { describe, expect, it } from "vitest";
import { JobState, isTerminalJobState, updatePresenceSettings } from "./jobsApi";

describe("isTerminalJobState", () => {
  it("marks the three states a run can never leave", () => {
    for (const state of ["done", "failed", "interrupted"] as JobState[]) {
      expect(isTerminalJobState(state)).toBe(true);
    }
  });

  it("keeps queued and running non-terminal — they take Stop/Cancel, not Delete", () => {
    for (const state of ["queued", "running"] as JobState[]) {
      expect(isTerminalJobState(state)).toBe(false);
    }
  });
});

describe("updatePresenceSettings", () => {
  it("sends the changes as an object, not a JSON string", async () => {
    // apiRequest serializes the body itself. Pre-stringifying here put a JSON
    // *string* on the wire where POST /api/v1/jobs/devices/settings wants an
    // object, and the endpoint answered 422 — so the sharing toggle and the
    // device rename both failed for every user, silently, from the UI's side.
    let sent: string | undefined;
    const fetcher = async (_url: string, init?: RequestInit) => {
      sent = init?.body as string;
      return new Response(JSON.stringify({ enabled: false, label: "desktop" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    };

    await updatePresenceSettings("http://x", fetcher, { enabled: false });

    expect(JSON.parse(sent!)).toEqual({ enabled: false });
  });
});
