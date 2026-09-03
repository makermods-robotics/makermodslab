import { describe, expect, it } from "vitest";

import {
  DEFAULT_HORIZON,
  DEFAULT_REMOTE_RUN_CONFIG,
  DEFAULT_S_MIN,
  defaultEngineForPolicyType,
  policySupportsRtc,
} from "./remoteRunConfig";

/**
 * Which engine a checkpoint gets by default.
 *
 * This is the one piece of the remote panel that makes a hardware decision on
 * the operator's behalf: it picks which chunk player drives the arm, and — via
 * the generated command — which GPU server they are told to start. Both wrong
 * answers are quiet ones. Defaulting a flow policy to `sync` leaves the ~1 Hz
 * seam twitch in place with a perfectly healthy transport; defaulting anything
 * else to `rtc` pairs the arm with a server the operator never launched.
 */

describe("only flow policies can be in-painted", () => {
  it.each(["smolvla", "pi0", "pi05", "diffusion"])(
    "recognises %s as a flow family",
    (policyType) => {
      expect(policySupportsRtc(policyType)).toBe(true);
      expect(defaultEngineForPolicyType(policyType)).toBe("rtc");
    },
  );

  it("keeps ACT on the sync engine", () => {
    // ACT serves a plain chunk and ignores the RTC state fields entirely, so
    // `rtc` buys it nothing and costs it the play-to-completion contract it
    // was evaluated in.
    expect(policySupportsRtc("act")).toBe(false);
    expect(defaultEngineForPolicyType("act")).toBe("sync");
  });

  it("treats an unknown or absent policy type as sync", () => {
    // "Everything else" on purpose. Guessing rtc for a policy family nobody
    // has classified would pair the arm with a GPU server the operator was
    // never told to start — a session that connects, looks healthy, and
    // receives nothing.
    for (const value of [null, undefined, "", "some-new-policy"]) {
      expect(defaultEngineForPolicyType(value)).toBe("sync");
    }
  });

  it("matches the policy_type case-insensitively", () => {
    // The value comes from the checkpoint's own config; nothing guarantees its
    // casing, and a miss here reads as "this policy can't be in-painted".
    expect(policySupportsRtc("SmolVLA")).toBe(true);
  });
});

describe("the horizon default follows the engine", () => {
  it("is one ACT block for sync and the full flow chunk_size for rtc", () => {
    // 50 is what smolvla / pi0 / pi05 always denoise — a smaller horizon only
    // truncates — and it is what robot_rtc and modal_policy_rtc.py both
    // default to, so a run started with neither side touched still agrees.
    expect(DEFAULT_HORIZON.sync).toBe(16);
    expect(DEFAULT_HORIZON.rtc).toBe(50);
  });

  it("starts the panel on the engine that suits any policy", () => {
    expect(DEFAULT_REMOTE_RUN_CONFIG.engine).toBe("sync");
    expect(DEFAULT_REMOTE_RUN_CONFIG.horizon).toBe(DEFAULT_HORIZON.sync);
  });

  it("starts s_min at the value BOTH sides default to", () => {
    // The robot computes `overlap_end = H - max(s_min, d)` and the GPU server
    // trusts that field; the two defaults matching is what keeps an untouched
    // run correct.
    expect(DEFAULT_S_MIN).toBe(4);
    expect(DEFAULT_REMOTE_RUN_CONFIG.sMin).toBe(DEFAULT_S_MIN);
  });
});
