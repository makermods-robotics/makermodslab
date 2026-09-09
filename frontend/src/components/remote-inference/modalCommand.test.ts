import { describe, expect, it } from "vitest";

import {
  buildModalRunLine,
  POLICY_PATH_PLACEHOLDER,
  TOKEN_PLACEHOLDER,
} from "./modalCommand";
import type { ModalRunLineInput } from "./modalCommand";

/**
 * The generated line is a shell command an operator pastes into another
 * terminal, so it is asserted VERBATIM. A "cosmetic" change to any character
 * here is a change to a command that launches a GPU container, and a horizon
 * or codec that drifts from the robot side's does not fail loudly — Portal
 * silently drops every packet whose schema fingerprint differs.
 */
const sfu: ModalRunLineInput = {
  policyHubId: "makermods/pick-place",
  engine: "sync",
  task: "",
  horizon: 16,
  fps: 30,
  videoCodec: "H264",
  room: "mml-abcdef012345",
  // The TAILNET url a container dials — never the loopback one a local child
  // uses.
  url: "ws://100.64.0.1:7880",
  source: "sfu",
  sMin: 4,
  // The operator-role JWT the transport endpoint reports. Short-lived, one
  // room, one identity — which is why it may sit on a command line.
  policyToken: "jwt.policy.abc",
  // Nothing chosen: the CLI resolves the profile and environment itself, which
  // is what every assertion below the target block assumes.
  profile: "",
  environment: "",
  // No GPU-side knob chosen either: no --model-dtype flag (the dtype the
  // checkpoint was saved with), no --flow-steps (its own sampler count) and no
  // DRTC_GPU assignment (the wrapper's own pin). That is what keeps every
  // verbatim line below byte-for-byte pre-S3.8e.
  modelDtype: "",
  flowSteps: null,
  extraImageRoles: [],
  gpu: "",
};

const TAIL =
  "--livekit-room mml-abcdef012345 " +
  "--tailscale --livekit-url ws://100.64.0.1:7880 " +
  "--livekit-token jwt.policy.abc";

/** The rtc pairing: the other wrapper, the flow families' full chunk_size, and
 * the one flag whose value the server trusts from the robot. */
const rtcSfu: ModalRunLineInput = {
  ...sfu,
  engine: "rtc",
  horizon: 50,
  sMin: 4,
};

describe("the generated modal run line", () => {
  it.each([sfu, rtcSfu])("passes a selected slack to either wrapper", (input) => {
    expect(buildModalRunLine({ ...input, slack: 2 })).toContain("--slack 2");
    expect(buildModalRunLine({ ...input, slack: 5 })).not.toContain("--slack");
  });
  it("carries the transport triple, the room, the tailnet url and the token", () => {
    expect(buildModalRunLine(sfu)).toBe(
      "modal run makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        "--horizon 16 --fps 30 --video-codec H264 " +
        TAIL,
    );
  });

  it("never emits an API key or secret — the container holds a token only", () => {
    // The station's signing secret never leaves the station; the GPU side
    // joins with an operator-role JWT the station signed for it.
    const line = buildModalRunLine(sfu);
    expect(line).not.toContain("--livekit-api-key");
    expect(line).not.toContain("--livekit-api-secret");
    expect(line).toContain("--livekit-token jwt.policy.abc");
  });

  it("emits no tailnet block and no token when the Lab runs no SFU", () => {
    // With `none` there is no room to join; the line stays copy-able so the
    // operator can read what WOULD run, but it carries no transport.
    const line = buildModalRunLine({ ...sfu, source: "none", url: "", room: "" });
    expect(line).toBe(
      "modal run makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        "--horizon 16 --fps 30 --video-codec H264",
    );
  });

  it("forwards the codec identifier verbatim", () => {
    expect(buildModalRunLine({ ...sfu, videoCodec: "MJPEG" })).toContain(
      "--video-codec MJPEG",
    );
  });

  it("stays copy-able before a Hub id is typed", () => {
    expect(buildModalRunLine({ ...sfu, policyHubId: "   " })).toContain(
      `--policy-path ${POLICY_PATH_PLACEHOLDER}`,
    );
  });

  it("falls back to a token placeholder before the SFU has reported one", () => {
    // Still copy-able, and obviously incomplete — the same contract the
    // policy-path placeholder has.
    const line = buildModalRunLine({ ...sfu, policyToken: "  " });
    expect(line).toContain(`--livekit-token ${TOKEN_PLACEHOLDER}`);
  });

  it("omits the url flag when tailscale reported no address", () => {
    // Rather than offering a loopback url a container cannot reach.
    const line = buildModalRunLine({ ...sfu, url: "" });
    expect(line).toContain("--tailscale");
    expect(line).not.toContain("--livekit-url");
  });

  it("omits the room flag when the transport reports none", () => {
    // Rather than emitting `--livekit-room ` and having Modal parse the next
    // flag as its value.
    expect(buildModalRunLine({ ...sfu, room: "" })).not.toContain(
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
    const line = buildModalRunLine({ ...sfu, profile: "work-account" });
    expect(line).toBe(
      "MODAL_PROFILE=work-account modal run makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        "--horizon 16 --fps 30 --video-codec H264 " +
        TAIL,
    );
    expect(line).not.toContain("--profile");
    expect(line).not.toContain("profile activate");
  });

  it("puts --env before the wrapper path, where `modal run` reads it", () => {
    // `modal run [OPTIONS] FUNC_REF`. After the path, Click hands --env to the
    // wrapper's own local_entrypoint, which has no such parameter.
    const line = buildModalRunLine({ ...sfu, environment: "staging" });
    expect(line).toBe(
      "modal run --env staging makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        "--horizon 16 --fps 30 --video-codec H264 " +
        TAIL,
    );
    expect(line.indexOf("--env")).toBeLessThan(line.indexOf("modal_policy.py"));
  });

  it("carries both together on the rtc wrapper", () => {
    expect(
      buildModalRunLine({
        ...rtcSfu,
        profile: "work-account",
        environment: "staging",
      }),
    ).toBe(
      "MODAL_PROFILE=work-account modal run --env staging " +
        "makermodslab/drtc/modal_policy_rtc.py " +
        "--policy-path makermods/pick-place " +
        "--horizon 50 --fps 30 --s-min 4 --video-codec H264 " +
        TAIL,
    );
  });

  it("emits neither when nothing is chosen, so the CLI resolves them", () => {
    // `--env ""` would name an environment called "", and `MODAL_PROFILE= ` a
    // profile called "".
    const line = buildModalRunLine({
      ...sfu,
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
      ...sfu,
      task: "Put the eraser on the mat",
      modelDtype: "bfloat16",
    });
    expect(line).toBe(
      "modal run makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        '--task "Put the eraser on the mat" ' +
        "--model-dtype bfloat16 " +
        "--horizon 16 --fps 30 --video-codec H264 " +
        TAIL,
    );
  });

  it("omits the precision flag when nothing is chosen", () => {
    // Unset is not a default this line picks — it is the dtype the checkpoint
    // was saved with, and `--model-dtype ""` would be a dtype named "".
    expect(buildModalRunLine({ ...sfu, modelDtype: "  " })).not.toContain(
      "--model-dtype",
    );
  });

  it("puts the GPU type in front as an env-var assignment, never as a flag", () => {
    // `_FN_KWARGS["gpu"]` is evaluated when `modal run` IMPORTS the wrapper on
    // the operator's own machine, so by the time a flag could be parsed the
    // Modal function is already declared. DRTC_GPU is the only channel.
    const line = buildModalRunLine({ ...sfu, gpu: "H100" });
    expect(line).toBe(
      "DRTC_GPU=H100 modal run makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        "--horizon 16 --fps 30 --video-codec H264 " +
        TAIL,
    );
    expect(line).not.toContain("--gpu");
  });

  it("carries both assignments in the order the launcher exports them", () => {
    const line = buildModalRunLine({
      ...rtcSfu,
      profile: "work-account",
      gpu: "A100-80GB",
      modelDtype: "float32",
    });
    expect(line).toBe(
      "MODAL_PROFILE=work-account DRTC_GPU=A100-80GB modal run " +
        "makermodslab/drtc/modal_policy_rtc.py " +
        "--policy-path makermods/pick-place " +
        "--model-dtype float32 " +
        "--horizon 50 --fps 30 --s-min 4 --video-codec H264 " +
        TAIL,
    );
  });

  it("emits neither when nothing is chosen", () => {
    const line = buildModalRunLine({ ...sfu, gpu: " ", modelDtype: " " });
    expect(line.startsWith("modal run makermodslab/")).toBe(true);
    expect(line).not.toContain("DRTC_GPU");
    expect(line).not.toContain("--model-dtype");
  });

  // The third knob (S3.8f). A flag like the precision, right after it.
  it("passes the flow steps as a flag, right after the precision", () => {
    const line = buildModalRunLine({
      ...rtcSfu,
      modelDtype: "bfloat16",
      flowSteps: 4,
    });
    expect(line).toBe(
      "modal run makermodslab/drtc/modal_policy_rtc.py " +
        "--policy-path makermods/pick-place " +
        "--model-dtype bfloat16 --flow-steps 4 " +
        "--horizon 50 --fps 30 --s-min 4 --video-codec H264 " +
        TAIL,
    );
  });

  it("omits the flow-steps flag when nothing is chosen", () => {
    // 0 is not a step count any sampler would take — the pin raises on it — so
    // the flag's ABSENCE is the only way to say "the checkpoint's own".
    expect(buildModalRunLine({ ...sfu, flowSteps: null })).not.toContain(
      "--flow-steps",
    );
  });

  // The fourth knob (S3.8g), and the only one that is half the WIRE: the robot
  // side publishes a video track per bound camera role, so a hand-run command
  // that omits this flag declares fewer views than the arm is sending — and
  // Portal answers a fingerprint mismatch by dropping every packet in silence.
  it("passes the extra camera roles as ONE comma-joined flag, after the steps", () => {
    const line = buildModalRunLine({
      ...rtcSfu,
      modelDtype: "bfloat16",
      flowSteps: 4,
      extraImageRoles: ["cam2", "cam3"],
    });
    expect(line).toBe(
      "modal run makermodslab/drtc/modal_policy_rtc.py " +
        "--policy-path makermods/pick-place " +
        "--model-dtype bfloat16 --flow-steps 4 " +
        "--extra-image-roles cam2,cam3 " +
        "--horizon 50 --fps 30 --s-min 4 --video-codec H264 " +
        TAIL,
    );
  });

  it("omits the extra-image-roles flag when no view was added", () => {
    // Comma-joined into one flag because both wrappers take a Click str, not a
    // repeatable option — and absent is the only way to say "the views the
    // checkpoint was published with".
    expect(
      buildModalRunLine({ ...sfu, extraImageRoles: [] }),
    ).not.toContain("--extra-image-roles");
  });
});

describe("the engine picks the wrapper the GPU side must run", () => {
  // The two servers publish DIFFERENT state schemas — the rtc one carries five
  // extra in-painting fields — and Portal fingerprints the schema. Running the
  // sync wrapper against an rtc robot is therefore a session that connects,
  // reports a healthy transport, and never receives a single chunk.
  it("builds the rtc line: the rtc wrapper, --s-min, and horizon 50", () => {
    expect(buildModalRunLine(rtcSfu)).toBe(
      "modal run makermodslab/drtc/modal_policy_rtc.py " +
        "--policy-path makermods/pick-place " +
        "--horizon 50 --fps 30 --s-min 4 --video-codec H264 " +
        TAIL,
    );
  });

  it("keeps the sync line on the sync wrapper, with no --s-min at all", () => {
    // modal_policy.py's local_entrypoint has no s_min parameter, so the flag
    // would make the line fail to parse rather than fall back to a default.
    const line = buildModalRunLine(sfu);
    expect(line).toContain("modal run makermodslab/drtc/modal_policy.py");
    expect(line).not.toContain("--s-min");
  });

  it("forwards a non-default s-min verbatim on the rtc engine", () => {
    // The robot computes `overlap_end = H - max(s_min, d)` and the server
    // TRUSTS that field; the two values existing to be equal is the whole
    // reason s_min is on the session at all.
    expect(buildModalRunLine({ ...rtcSfu, sMin: 2 })).toContain("--s-min 2");
  });

  it("carries the task and the tailnet block on the rtc wrapper too", () => {
    expect(
      buildModalRunLine({
        ...rtcSfu,
        task: "Put the eraser on the mat",
      }),
    ).toBe(
      "modal run makermodslab/drtc/modal_policy_rtc.py " +
        "--policy-path makermods/pick-place " +
        '--task "Put the eraser on the mat" ' +
        "--horizon 50 --fps 30 --s-min 4 --video-codec H264 " +
        TAIL,
    );
  });

  it("leaves slack / tolerance / rtc-schedule at the wrapper's defaults", () => {
    // Deliberately not exposed. They are knobs whose wrong values present as
    // "the arm is sluggish" rather than as an error, and the wrapper's own
    // defaults are the ones the August bench runs validated.
    const line = buildModalRunLine(rtcSfu);
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
      buildModalRunLine({ ...sfu, task: "Put the lego brick in the box" }),
    ).toBe(
      "modal run makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        '--task "Put the lego brick in the box" ' +
        "--horizon 16 --fps 30 --video-codec H264 " +
        TAIL,
    );
  });

  it("escapes what a double-quoted shell word still expands", () => {
    // All four of \ " $ ` survive inside double quotes, so all four are
    // escaped. A task is arbitrary user text that reaches a shell.
    expect(
      buildModalRunLine({
        ...sfu,
        task: 'Put the "red" block in $HOME\\bin `now`',
      }),
    ).toBe(
      "modal run makermodslab/drtc/modal_policy.py " +
        "--policy-path makermods/pick-place " +
        '--task "Put the \\"red\\" block in \\$HOME\\\\bin \\`now\\`" ' +
        "--horizon 16 --fps 30 --video-codec H264 " +
        TAIL,
    );
  });

  it("omits the flag entirely for an empty or whitespace task", () => {
    expect(buildModalRunLine(sfu)).not.toContain("--task");
    expect(buildModalRunLine({ ...sfu, task: "   " })).not.toContain(
      "--task",
    );
  });
});


it("pins the selected region and matching tolerance in the manual command", () => {
  const line = buildModalRunLine({ ...sfu, region: "eu", tolerance: 2.5 });
  expect(line).toContain('DRTC_REGION="eu" modal run');
  expect(line).toContain("--tolerance 2.5");
});


it("passes Auto GPU and region as explicit environment settings", () => {
  const line = buildModalRunLine({ ...sfu, gpu: "auto", region: "auto" });
  expect(line).toContain('DRTC_GPU=auto DRTC_REGION="auto" modal run');
});
