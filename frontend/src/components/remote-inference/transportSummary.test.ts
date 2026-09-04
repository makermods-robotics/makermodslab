import { describe, expect, it } from "vitest";
import type { RemoteInferenceTransportStatus } from "@/hooks/useRemoteInferenceTransport";
import { SFU_OFF_SUMMARY_KEY, summarizeTransport } from "./transportSummary";

const base: RemoteInferenceTransportStatus = {
  extra_installed: true,
  configured: true,
  missing_vars: [],
  url: "ws://100.64.0.1:7880",
  room: "drtc-bench",
  source: "sfu",
  sfu_enabled: true,
  sfu_url: "ws://127.0.0.1:7880",
  sfu_modal_url: "ws://100.64.0.1:7880",
  sfu_external_ip: true,
  sfu_key_id: "local-abc",
  sfu_key_file: "/keys.yaml",
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

  // The SFU being off is the REASON credentials are missing when the Lab's own
  // server is the intended source, so it is named ahead of them.
  it("blames the stopped SFU before the missing variables", () => {
    const s = summarizeTransport(
      {
        ...base,
        sfu_enabled: false,
        configured: false,
        missing_vars: ["LIVEKIT_URL", "LIVEKIT_API_KEY"],
        source: "none",
      },
      false,
      null,
    );
    expect(s.key).toBe(SFU_OFF_SUMMARY_KEY);
    expect(s.tone).toBe("error");
  });

  // ...but a Cloud operator's Lab never hosts one, and that is not a fault.
  it("says nothing about the SFU when livekit.env already configured it", () => {
    const s = summarizeTransport(
      { ...base, sfu_enabled: false, source: "cloud" },
      false,
      null,
    );
    expect(s.key).toBe("remoteInference.transport.summary.ready");
  });

  it("still names the missing variables when the SFU is running", () => {
    const s = summarizeTransport(
      {
        ...base,
        sfu_enabled: true,
        configured: false,
        missing_vars: ["LIVEKIT_API_KEY"],
      },
      false,
      null,
    );
    expect(s.key).toBe("remoteInference.transport.summary.missingVars");
  });

  it("names the missing variables verbatim", () => {
    const s = summarizeTransport(
      {
        ...base,
        configured: false,
        missing_vars: ["LIVEKIT_URL", "LIVEKIT_API_KEY"],
      },
      false,
      null,
    );
    expect(s.key).toBe("remoteInference.transport.summary.missingVars");
    expect(s.values).toEqual({ vars: "LIVEKIT_URL, LIVEKIT_API_KEY" });
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
    expect(absent.tone).toBe("warn");
  });
});
