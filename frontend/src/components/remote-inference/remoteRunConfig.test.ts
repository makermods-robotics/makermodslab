import { describe, expect, it } from "vitest";

import {
  DEFAULT_HORIZON,
  DEFAULT_REMOTE_RUN_CONFIG,
  DEFAULT_S_MIN,
  defaultEngineForPolicy,
  horizonForEngine,
  policySupportsRtc,
} from "./remoteRunConfig";

/**
 * Which engine a checkpoint gets by default, and which horizon.
 *
 * This is the one piece of the remote panel that makes a hardware decision on
 * the operator's behalf: it picks which chunk player drives the arm, and — via
 * the generated command — which GPU server they are told to start. Both wrong
 * answers are quiet ones. Defaulting a flow policy to `sync` leaves the ~1 Hz
 * seam twitch in place with a perfectly healthy transport; defaulting anything
 * else to `rtc` pairs the arm with a server the operator never launched. A
 * horizon above the checkpoint's own chunk width is quieter still: the Portal
 * peers disagree about the action-chunk shape, the wire-schema fingerprint
 * stops matching, and every packet is dropped with nothing said anywhere.
 */

describe("the server's supports_rtc is the answer when it has one", () => {
  it("takes a true straight from the server, whatever the type is called", () => {
    // molmoact2 is not in the frontend's own family list and never will be
    // reliably — the fork gains policy types faster than this file does.
    expect(
      policySupportsRtc({ policy_type: "molmoact2", supports_rtc: true }),
    ).toBe(true);
    expect(
      defaultEngineForPolicy({ policy_type: "molmoact2", supports_rtc: true }),
    ).toBe("rtc");
  });

  it("takes a false straight from the server, over the frontend's list", () => {
    // The two lists genuinely disagree: `diffusion` is in FLOW_POLICY_TYPES,
    // and the fork's classes say it cannot be in-painted. The server read the
    // classes; this file guessed.
    expect(
      policySupportsRtc({ policy_type: "diffusion", supports_rtc: false }),
    ).toBe(false);
    expect(
      defaultEngineForPolicy({ policy_type: "diffusion", supports_rtc: false }),
    ).toBe("sync");
  });
});

describe("the family list is the fallback, and only that", () => {
  it.each(["smolvla", "pi0", "pi05", "diffusion"])(
    "recognises %s as a flow family when the server didn't say",
    (policy_type) => {
      // supports_rtc null = a server too old to answer, or a policy type newer
      // than its table. Never read as "no".
      expect(policySupportsRtc({ policy_type, supports_rtc: null })).toBe(true);
      expect(defaultEngineForPolicy({ policy_type, supports_rtc: null })).toBe(
        "rtc",
      );
    },
  );

  it("keeps ACT on the sync engine", () => {
    // ACT serves a plain chunk and ignores the RTC state fields entirely, so
    // `rtc` buys it nothing and costs it the play-to-completion contract it
    // was evaluated in.
    expect(policySupportsRtc({ policy_type: "act", supports_rtc: null })).toBe(
      false,
    );
    expect(
      defaultEngineForPolicy({ policy_type: "act", supports_rtc: null }),
    ).toBe("sync");
  });

  it("treats an unknown or absent policy type as sync", () => {
    // "Everything else" on purpose. Guessing rtc for a policy family nobody
    // has classified would pair the arm with a GPU server the operator was
    // never told to start — a session that connects, looks healthy, and
    // receives nothing.
    for (const policy_type of [null, undefined, "", "some-new-policy"]) {
      expect(defaultEngineForPolicy({ policy_type, supports_rtc: null })).toBe(
        "sync",
      );
    }
    expect(defaultEngineForPolicy(null)).toBe("sync");
    expect(defaultEngineForPolicy(undefined)).toBe("sync");
  });

  it("matches the policy_type case-insensitively", () => {
    // The value comes from the checkpoint's own config; nothing guarantees its
    // casing, and a miss here reads as "this policy can't be in-painted".
    expect(
      policySupportsRtc({ policy_type: "SmolVLA", supports_rtc: null }),
    ).toBe(true);
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

describe("n_action_steps is a ceiling on the horizon, not a target", () => {
  it("falls back to the engine default when the checkpoint doesn't say", () => {
    for (const absent of [null, undefined, 0, -1]) {
      expect(horizonForEngine("sync", absent)).toBe(DEFAULT_HORIZON.sync);
      expect(horizonForEngine("rtc", absent)).toBe(DEFAULT_HORIZON.rtc);
    }
  });

  it("comes down to a checkpoint that returns fewer steps", () => {
    // The motivating case: lerobot/MolmoAct2-SO100_101-LeRobot returns 30, and
    // the rtc default of 50 makes the fingerprint mismatch that drops every
    // packet in silence.
    expect(horizonForEngine("rtc", 30)).toBe(30);
    expect(horizonForEngine("sync", 8)).toBe(8);
  });

  it("does NOT rise to a checkpoint that returns more", () => {
    // An ACT config declaring 100 still starts at one open-loop block, and a
    // flow checkpoint declaring 100 still starts at the chunk width both
    // servers default to. The ceiling only ever lowers the seed.
    expect(horizonForEngine("sync", 100)).toBe(DEFAULT_HORIZON.sync);
    expect(horizonForEngine("rtc", 100)).toBe(DEFAULT_HORIZON.rtc);
  });

  it("is idempotent at the ceiling itself", () => {
    // The engine-switch re-seed compares against this value on BOTH sides, so
    // a horizon sitting exactly on the ceiling must not read as hand-typed.
    expect(horizonForEngine("rtc", 50)).toBe(50);
  });
});
