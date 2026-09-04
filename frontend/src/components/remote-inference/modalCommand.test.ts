import { describe, expect, it } from "vitest";

import {
  buildModalRunLine,
  LOCAL_KEY_ID_PLACEHOLDER,
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
  engine: "sync",
  task: "",
  horizon: 16,
  fps: 30,
  videoCodec: "H264",
  room: "portal-lerobot-inference",
  url: "wss://example.livekit.cloud",
  source: "cloud",
  sMin: 4,
  // The key NAME the transport endpoint reports. Real, and unused on the Cloud
  // path — only the SFU line carries it.
  sfuKeyId: "APIkey123",
  // Nothing chosen: the CLI resolves the profile and environment itself, which
  // is what every assertion below the target block assumes.
  profile: "",
  environment: "",
  // Neither GPU-side knob chosen either: no --model-dtype flag (the dtype the
  // checkpoint was saved with) and no DRTC_GPU assignment (the wrapper's own
  // pin). That is what keeps every verbatim line below byte-for-byte pre-S3.8e.
  modelDtype: "",
  gpu: "",
};

/** The rtc pairing: the other wrapper, the flow families' full chunk_size, and
 * the one flag whose value the server trusts from the robot. */
const rtcCloud: ModalRunLineInput = {
  ...cloud,
  engine: "rtc",
  horizon: 50,
  sMin: 4,
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

  it("adds --tailscale, the url, the real key id and a secret placeholder for the Lab SFU", () => {
    const line = buildModalRunLine({
      ...cloud,
      source: "sfu",
      url: "ws://100.64.0.1:7880",
    });
    expect(line).toBe(
      "modal run makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        "--horizon 16 --fps 30 --video-codec H264 " +
        "--livekit-room portal-lerobot-inference " +
        "--tailscale --livekit-url ws://100.64.0.1:7880 " +
        "--livekit-api-key APIkey123 " +
        `--livekit-api-secret ${LOCAL_SECRET_PLACEHOLDER}`,
    );
  });

  it("never emits the API secret — the endpoint does not expose it", () => {
    const line = buildModalRunLine({ ...cloud, source: "sfu" });
    // The key ID is real: it names the pair, it does not authorize with it.
    expect(line).toContain("--livekit-api-key APIkey123");
    expect(line).toContain(`--livekit-api-secret ${LOCAL_SECRET_PLACEHOLDER}`);
    expect(line).not.toContain("s3cret");
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

  it("falls back to a key-id placeholder before the SFU has reported one", () => {
    // Still copy-able, and obviously incomplete — the same contract the
    // policy-path placeholder has.
    const line = buildModalRunLine({ ...cloud, source: "sfu", sfuKeyId: "  " });
    expect(line).toContain(`--livekit-api-key ${LOCAL_KEY_ID_PLACEHOLDER}`);
  });

  it("omits the room flag when the transport reports none", () => {
    // Rather than emitting `--livekit-room ` and having Modal parse the next
    // flag as its value.
    expect(buildModalRunLine({ ...cloud, room: "" })).not.toContain(
      "--livekit-room",
    );
  });
});

describe("which workspace the pasted line bills", () => {
  // The line has to match what the Lab's own launcher runs, flag for flag and
  // variable for variable — it is the manual fallback AND the ground truth an
  // operator compares against. `modal_launcher.build_argv` / `child_env` are
  // the other half of this pair.
  it("prefixes the profile as an env-var assignment, never a flag", () => {
    // `modal run` has no --profile, and `modal profile activate` would rewrite
    // the ~/.modal.toml every other terminal on this machine shares.
    const line = buildModalRunLine({ ...cloud, profile: "work-account" });
    expect(line).toBe(
      "MODAL_PROFILE=work-account modal run makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        "--horizon 16 --fps 30 --video-codec H264 " +
        "--livekit-room portal-lerobot-inference",
    );
    expect(line).not.toContain("--profile");
    expect(line).not.toContain("profile activate");
  });

  it("puts --env before the wrapper path, where `modal run` reads it", () => {
    // `modal run [OPTIONS] FUNC_REF`. After the path, Click hands --env to the
    // wrapper's own local_entrypoint, which has no such parameter.
    const line = buildModalRunLine({ ...cloud, environment: "staging" });
    expect(line).toBe(
      "modal run --env staging makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        "--horizon 16 --fps 30 --video-codec H264 " +
        "--livekit-room portal-lerobot-inference",
    );
    expect(line.indexOf("--env")).toBeLessThan(line.indexOf("modal_policy.py"));
  });

  it("carries both together on the rtc wrapper", () => {
    expect(
      buildModalRunLine({
        ...rtcCloud,
        profile: "work-account",
        environment: "staging",
      }),
    ).toBe(
      "MODAL_PROFILE=work-account modal run --env staging " +
        "makermodslab/drtc/modal_policy_rtc.py " +
        "--policy-path makermods/pick-place " +
        "--horizon 50 --fps 30 --s-min 4 --video-codec H264 " +
        "--livekit-room portal-lerobot-inference",
    );
  });

  it("emits neither when nothing is chosen, so the CLI resolves them", () => {
    // Byte for byte the pre-S3.8b line. `--env ""` would name an environment
    // called "", and `MODAL_PROFILE= ` a profile called "".
    const line = buildModalRunLine({
      ...cloud,
      profile: "  ",
      environment: " ",
    });
    expect(line.startsWith("modal run makermodslab/")).toBe(true);
    expect(line).not.toContain("--env");
    expect(line).not.toContain("MODAL_PROFILE");
  });
});

describe("what the pasted line runs as, and what it runs on", () => {
  // The two S3.8e knobs, and the reason they travel differently: one is a flag
  // the wrapper's local_entrypoint declares, the other is a value the wrapper
  // reads at IMPORT, before Click has parsed anything. `modal_launcher`'s
  // build_argv / child_env are the other half of this pair.
  it("passes the precision as a flag, between --task and --horizon", () => {
    const line = buildModalRunLine({
      ...cloud,
      task: "Put the eraser on the mat",
      modelDtype: "bfloat16",
    });
    expect(line).toBe(
      "modal run makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        '--task "Put the eraser on the mat" ' +
        "--model-dtype bfloat16 " +
        "--horizon 16 --fps 30 --video-codec H264 " +
        "--livekit-room portal-lerobot-inference",
    );
  });

  it("omits the precision flag when nothing is chosen", () => {
    // Unset is not a default this line picks — it is the dtype the checkpoint
    // was saved with, and `--model-dtype ""` would be a dtype named "".
    expect(buildModalRunLine({ ...cloud, modelDtype: "  " })).not.toContain(
      "--model-dtype",
    );
  });

  it("puts the GPU type in front as an env-var assignment, never as a flag", () => {
    // `_FN_KWARGS["gpu"]` is evaluated when `modal run` IMPORTS the wrapper on
    // the operator's own machine, so by the time a flag could be parsed the
    // Modal function is already declared. DRTC_GPU is the only channel.
    const line = buildModalRunLine({ ...cloud, gpu: "H100" });
    expect(line).toBe(
      "DRTC_GPU=H100 modal run makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        "--horizon 16 --fps 30 --video-codec H264 " +
        "--livekit-room portal-lerobot-inference",
    );
    expect(line).not.toContain("--gpu");
  });

  it("carries both assignments in the order the launcher exports them", () => {
    const line = buildModalRunLine({
      ...rtcCloud,
      profile: "work-account",
      gpu: "A100-80GB",
      modelDtype: "float16",
    });
    expect(line).toBe(
      "MODAL_PROFILE=work-account DRTC_GPU=A100-80GB modal run " +
        "makermodslab/drtc/modal_policy_rtc.py " +
        "--policy-path makermods/pick-place " +
        "--model-dtype float16 " +
        "--horizon 50 --fps 30 --s-min 4 --video-codec H264 " +
        "--livekit-room portal-lerobot-inference",
    );
  });

  it("emits neither when nothing is chosen", () => {
    const line = buildModalRunLine({ ...cloud, gpu: " ", modelDtype: " " });
    expect(line.startsWith("modal run makermodslab/")).toBe(true);
    expect(line).not.toContain("DRTC_GPU");
    expect(line).not.toContain("--model-dtype");
  });
});

describe("the engine picks the wrapper the GPU side must run", () => {
  // The two servers publish DIFFERENT state schemas — the rtc one carries five
  // extra in-painting fields — and Portal fingerprints the schema. Running the
  // sync wrapper against an rtc robot is therefore a session that connects,
  // reports a healthy transport, and never receives a single chunk.
  it("builds the rtc line: the rtc wrapper, --s-min, and horizon 50", () => {
    expect(buildModalRunLine(rtcCloud)).toBe(
      "modal run makermodslab/drtc/modal_policy_rtc.py " +
        "--policy-path makermods/pick-place " +
        "--horizon 50 --fps 30 --s-min 4 --video-codec H264 " +
        "--livekit-room portal-lerobot-inference",
    );
  });

  it("keeps the sync line on the sync wrapper, with no --s-min at all", () => {
    // modal_policy.py's local_entrypoint has no s_min parameter, so the flag
    // would make the line fail to parse rather than fall back to a default.
    const line = buildModalRunLine(cloud);
    expect(line).toContain("modal run makermodslab/drtc/modal_policy.py");
    expect(line).not.toContain("--s-min");
  });

  it("forwards a non-default s-min verbatim on the rtc engine", () => {
    // The robot computes `overlap_end = H - max(s_min, d)` and the server
    // TRUSTS that field; the two values existing to be equal is the whole
    // reason s_min is on the session at all.
    expect(buildModalRunLine({ ...rtcCloud, sMin: 2 })).toContain("--s-min 2");
  });

  it("carries the task and the tailnet block on the rtc wrapper too", () => {
    expect(
      buildModalRunLine({
        ...rtcCloud,
        task: "Put the eraser on the mat",
        source: "sfu",
        url: "ws://100.64.0.1:7880",
      }),
    ).toBe(
      "modal run makermodslab/drtc/modal_policy_rtc.py " +
        "--policy-path makermods/pick-place " +
        '--task "Put the eraser on the mat" ' +
        "--horizon 50 --fps 30 --s-min 4 --video-codec H264 " +
        "--livekit-room portal-lerobot-inference " +
        "--tailscale --livekit-url ws://100.64.0.1:7880 " +
        "--livekit-api-key APIkey123 " +
        `--livekit-api-secret ${LOCAL_SECRET_PLACEHOLDER}`,
    );
  });

  it("leaves slack / tolerance / rtc-schedule at the wrapper's defaults", () => {
    // Deliberately not exposed. They are knobs whose wrong values present as
    // "the arm is sluggish" rather than as an error, and the wrapper's own
    // defaults are the ones the August bench runs validated.
    const line = buildModalRunLine(rtcCloud);
    for (const flag of [
      "--slack",
      "--tolerance",
      "--max-guidance-weight",
      "--rtc-schedule",
    ]) {
      expect(line).not.toContain(flag);
    }
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
    expect(buildModalRunLine({ ...cloud, task: "   " })).not.toContain(
      "--task",
    );
  });
});
