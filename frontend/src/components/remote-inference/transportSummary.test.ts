import { describe, expect, it } from "vitest";
import type { RemoteInferenceTransportStatus } from "@/hooks/useRemoteInferenceTransport";
import { SFU_OFF_SUMMARY_KEY, summarizeTransport } from "./transportSummary";

const base: RemoteInferenceTransportStatus = {
  extra_installed: true,
  configured: true,
  url: "ws://100.64.0.1:7880",
  room: "drtc-bench",
  source: "sfu",
  sfu_enabled: true,
  sfu_url: "ws://127.0.0.1:7880",
  sfu_modal_url: "ws://100.64.0.1:7880",
  sfu_external_ip: true,
  policy_token: "jwt.policy.abc",
  sfu_install_hint: null,
  endpoint_reachable: true,
  operator_present: true,
  error_code: null,
  message: null,
};

describe("summarizeTransport picks the first thing to fix", () => {
  it("reports a failed read before anything else", () => {
    const s = summarizeTransport(base, false, "boom");
    expect(s.key).toBe("remoteInference.transport.summary.fetchFailed");
    expect(s.values).toEqual({ error: "boom" });
    expect(s.tone).toBe("error");
  });

  it("distinguishes 'checking' from 'never checked'", () => {
    expect(summarizeTransport(null, true, null).key).toBe(
      "remoteInference.transport.summary.checking",
    );
    expect(summarizeTransport(null, false, null).key).toBe(
      "remoteInference.transport.summary.notChecked",
    );
  });

  it("reports a stopped SFU", () => {
    const s = summarizeTransport(
      { ...base, sfu_enabled: false, configured: false, source: "none" },
      false, null,
    );
    expect(s.key).toBe(SFU_OFF_SUMMARY_KEY);
    expect(s.tone).toBe("error");
  });

  it("reports unavailable configuration when the SFU is enabled", () => {
    const s = summarizeTransport({ ...base, configured: false }, false, null);
    expect(s.key).toBe("remoteInference.transport.summary.notConfigured");
    expect(s.tone).toBe("error");
  });

  // null is "the probe did not run" — a third state, never a failure.
  it("keeps an un-run probe out of the failure tones", () => {
    const s = summarizeTransport(
      { ...base, endpoint_reachable: null, operator_present: null },
      false,
      null,
    );
    expect(s.key).toBe("remoteInference.transport.summary.notProbed");
    expect(s.tone).toBe("muted");
  });

  it("is ready only with an operator in the room", () => {
    expect(summarizeTransport(base, false, null)).toEqual({
      key: "remoteInference.transport.summary.ready",
      values: { room: "drtc-bench" },
      tone: "ok",
    });
    const absent = summarizeTransport(
      { ...base, operator_present: false },
      false,
      null,
    );
    expect(absent.key).toBe("remoteInference.transport.summary.operatorAbsent");
    expect(absent.tone).toBe("muted");
  });

  it.each([
    ["starting", "gpuStarting"],
    ["ready", "gpuWaiting"],
    ["stopping", "gpuWaiting"],
    ["idle", "operatorAbsent"],
  ])("keeps the waiting message consistent with a %s GPU", (state, key) => {
    expect(summarizeTransport({ ...base, operator_present: false }, false, null, state).key)
      .toBe(`remoteInference.transport.summary.${key}`);
  });
});
