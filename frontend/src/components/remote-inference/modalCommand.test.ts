import { describe, expect, it } from "vitest";

import {
  buildModalRunLine,
  LOCAL_SECRET_PLACEHOLDER,
  POLICY_PATH_PLACEHOLDER,
} from "./modalCommand";
import type { ModalRunLineInput } from "./modalCommand";

/**
 * The generated line is a shell command an operator pastes into another
 * terminal, so it is asserted VERBATIM. A "cosmetic" change to any character
 * here is a change to a command that launches a GPU container, and a horizon
 * or codec that drifts from the robot side's does not fail loudly — Portal
 * silently drops every packet whose schema fingerprint differs.
 */
const cloud: ModalRunLineInput = {
  policyHubId: "makermods/pick-place",
  task: "",
  horizon: 16,
  fps: 30,
  videoCodec: "H264",
  room: "portal-lerobot-inference",
  url: "wss://example.livekit.cloud",
  source: "cloud",
};

describe("the generated modal run line", () => {
  it("carries the transport triple and the room for a Cloud transport", () => {
    expect(buildModalRunLine(cloud)).toBe(
      "modal run makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        "--horizon 16 --fps 30 --video-codec H264 " +
        "--livekit-room portal-lerobot-inference",
    );
  });

  it("does NOT add the tailnet flags for a Cloud transport", () => {
    // A Cloud URL is reachable from a Modal container as-is; --tailscale there
    // would demand a TS_AUTHKEY the run does not need and fail at startup.
    expect(buildModalRunLine(cloud)).not.toContain("--tailscale");
    expect(buildModalRunLine(cloud)).not.toContain("--livekit-url");
  });

  it("adds --tailscale, the url and both secret placeholders for a local SFU", () => {
    const line = buildModalRunLine({
      ...cloud,
      source: "local_override",
      url: "ws://100.64.0.1:7880",
    });
    expect(line).toBe(
      "modal run makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        "--horizon 16 --fps 30 --video-codec H264 " +
        "--livekit-room portal-lerobot-inference " +
        "--tailscale --livekit-url ws://100.64.0.1:7880 " +
        `--livekit-api-key ${LOCAL_SECRET_PLACEHOLDER} ` +
        `--livekit-api-secret ${LOCAL_SECRET_PLACEHOLDER}`,
    );
  });

  it("never emits a real key or secret — the API does not expose them", () => {
    const line = buildModalRunLine({ ...cloud, source: "local_override" });
    expect(line).toContain(`--livekit-api-key ${LOCAL_SECRET_PLACEHOLDER}`);
    expect(line).toContain(`--livekit-api-secret ${LOCAL_SECRET_PLACEHOLDER}`);
  });

  it("forwards the codec identifier verbatim", () => {
    expect(buildModalRunLine({ ...cloud, videoCodec: "MJPEG" })).toContain(
      "--video-codec MJPEG",
    );
  });

  it("stays copy-able before a Hub id is typed", () => {
    expect(buildModalRunLine({ ...cloud, policyHubId: "   " })).toContain(
      `--policy-path ${POLICY_PATH_PLACEHOLDER}`,
    );
  });

  it("omits the room flag when the transport reports none", () => {
    // Rather than emitting `--livekit-room ` and having Modal parse the next
    // flag as its value.
    expect(buildModalRunLine({ ...cloud, room: "" })).not.toContain(
      "--livekit-room",
    );
  });
});

describe("the task travels to the GPU side too", () => {
  // A language-conditioned policy is STEERED by this string. Sending it to the
  // robot side and not to the container does not fail loudly — it just makes
  // the policy worse in ways that read as the policy being bad.
  it("quotes the task and places it where the entrypoint declares it", () => {
    expect(
      buildModalRunLine({ ...cloud, task: "Put the lego brick in the box" }),
    ).toBe(
      "modal run makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        '--task "Put the lego brick in the box" ' +
        "--horizon 16 --fps 30 --video-codec H264 " +
        "--livekit-room portal-lerobot-inference",
    );
  });

  it("escapes what a double-quoted shell word still expands", () => {
    // All four of \ " $ ` survive inside double quotes, so all four are
    // escaped. A task is arbitrary user text that reaches a shell.
    expect(
      buildModalRunLine({
        ...cloud,
        task: 'Put the "red" block in $HOME\\bin `now`',
      }),
    ).toBe(
      "modal run makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        '--task "Put the \\"red\\" block in \\$HOME\\\\bin \\`now\\`" ' +
        "--horizon 16 --fps 30 --video-codec H264 " +
        "--livekit-room portal-lerobot-inference",
    );
  });

  it("omits the flag entirely for an empty or whitespace task", () => {
    expect(buildModalRunLine(cloud)).not.toContain("--task");
    expect(buildModalRunLine({ ...cloud, task: "   " })).not.toContain("--task");
  });
});
