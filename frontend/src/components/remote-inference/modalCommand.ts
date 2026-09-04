import type { TransportSource } from "@/hooks/useRemoteInferenceTransport";
import type { RemoteEngine } from "./remoteRunConfig";

/**
 * The `modal run` line the operator must paste into the OTHER terminal.
 *
 * This is the highest-value element of the remote-inference UI, and the reason
 * `policy_hub_id` / `horizon` / `fps` / `video_codec` are options on the
 * session rather than constants: the panel generates the GPU side's command
 * from the SAME object the robot side is started with, so the two cannot
 * disagree. They must not disagree — Portal fingerprints the wire schema and
 * SILENTLY DROPS packets whose fingerprint differs, so a mismatched horizon or
 * codec presents as a healthy session with zero chunks, never as an error.
 *
 * EVERY CHARACTER OF THE OUTPUT IS DATA. It is a shell command; it is never
 * translated, never localized, never case-folded. Only the prose AROUND it in
 * the panel goes through i18n.
 *
 * The Lab does not launch Modal (lifecycle A — see docs/drtc/SLICE3.md §2), so
 * this line is how the human closes the loop. It becomes dead weight the day
 * the Lab launches the GPU side itself, and that is fine.
 */

export interface ModalRunLineInput {
  /** "<owner>/<repo>" the container resolves with `from_pretrained`. */
  policyHubId: string;
  /** Which robot-side chunk player the session will spawn — and therefore
   * which SCRIPT this line has to run. The two servers publish different state
   * schemas (the rtc one carries five extra in-painting fields), and Portal
   * fingerprints the schema, so pairing the wrong two is a healthy-looking
   * session that receives nothing. */
  engine: RemoteEngine;
  /** RTC only. Must equal the robot's `--s_min`: the robot computes
   * `overlap_end = H - max(s_min, d)` per request and the server TRUSTS that
   * field, falling back to its own `H - s_min` when it is absent. */
  sMin: number;
  /** The task string the run is started with. The GPU side takes it as
   * `--task`, and a language-conditioned policy is STEERED by it — omitting it
   * there while the robot side has it does not fail, it just makes the policy
   * worse in ways that look like the policy being bad. Empty ⇒ no flag. */
  task: string;
  /** The precision the checkpoint is loaded at. Empty ⇒ NO flag, which is the
   * dtype the checkpoint was saved with — not a default this line picks. */
  modelDtype: string;
  /** How many flow-matching / denoising steps the sampler takes per chunk.
   * Null ⇒ NO flag, which is the checkpoint's own count — null is to this
   * field what `""` is to `modelDtype`. It is also null whenever the selected
   * checkpoint's config has no such field, so a remembered pick for another
   * policy never appears in a line the operator is about to paste. */
  flowSteps: number | null;
  /** The Modal GPU to run on. It cannot be a flag: `_FN_KWARGS["gpu"]` is
   * evaluated when `modal run` imports the wrapper, before Click parses
   * anything — so it is emitted as an env-var ASSIGNMENT prefixing the
   * command, exactly as the Lab's own launcher passes it. Empty ⇒ omitted,
   * which leaves the wrapper's own pin. */
  gpu: string;
  horizon: number;
  fps: number;
  videoCodec: "H264" | "MJPEG";
  /** The room the transport endpoint reports, verbatim. */
  room: string;
  /** The URL the GPU side must dial, verbatim. Under the Lab's own SFU that is
   * `sfu_modal_url` — the TAILNET address — never the loopback url a local
   * child uses; on LiveKit Cloud it is unused (the container's secret carries
   * its own). */
  url: string;
  /** Which layer supplied the transport. Only `sfu` needs the tailnet flags:
   * a Cloud URL is reachable from a Modal container as-is, and the container's
   * own `LiveKit-cloud` secret already holds the credentials for it. */
  source: TransportSource;
  /** The `--livekit-api-key` value: the key's NAME, which the API does expose
   * because it identifies rather than authorizes. Empty ⇒ a placeholder is
   * emitted instead. */
  sfuKeyId: string;
  /** WHICH WORKSPACE PAYS, mirroring what the Lab's own launch does with the
   * same two values: the profile as a `MODAL_PROFILE=` assignment PREFIXING
   * the command (the CLI reads it per process, and `modal profile activate`
   * would rewrite the ~/.modal.toml every other terminal on this machine
   * shares), the environment as `modal run --env`. Empty ⇒ omitted, which is
   * the CLI's own resolution. */
  profile: string;
  environment: string;
}

/** Literal stand-in for the API SECRET. The API deliberately never exposes it
 * (a status endpoint that hands out a signing key is a credential leak wearing
 * a diagnostic hat), so the line ships a placeholder and the panel names the
 * file to read it from. The key ID beside it is real — it identifies rather
 * than authorizes. */
export const LOCAL_SECRET_PLACEHOLDER = "<from livekit_keys.yaml>";

/** Stand-in for a key ID the transport endpoint has not reported yet, so the
 * line stays copy-able (and obviously incomplete) before the SFU is up. */
export const LOCAL_KEY_ID_PLACEHOLDER = "<key id>";

/** Where the secret actually lives. Mirrors `utils/config.LIVEKIT_KEY_FILE` —
 * a real path, so it is data. The transport endpoint reports the effective one
 * in `sfu_key_file`; this is the fallback for rendering before it has. */
export const LOCAL_SFU_KEY_FILE =
  "~/.cache/huggingface/lerobot/livekit_keys.yaml";

/** Stand-in for an unknown Hub id, so the line is still copy-able (and
 * obviously incomplete) before the operator fills the field in. */
export const POLICY_PATH_PLACEHOLDER = "<owner>/<repo>";

/** The wrapper each engine pairs with. Paths, run from the repo root, exactly
 * as docs/drtc/README.md states them — `modal run` takes the wrapper in FILE
 * form, never as `python -m`. */
export const MODAL_WRAPPERS: Record<RemoteEngine, string> = {
  sync: "makermodslab/drtc/modal_policy.py",
  rtc: "makermodslab/drtc/modal_policy_rtc.py",
};

/**
 * The task as a double-quoted shell word.
 *
 * Double quotes, not single, because the operator will read this line and a
 * `'`-quoted English sentence breaks on the first apostrophe ("don't drop the
 * block"). Inside double quotes a POSIX shell still expands four characters,
 * so all four are backslash-escaped: `\` first (it is the escape itself),
 * then `"`, `$` and the backtick. A task is arbitrary user text and it reaches
 * a shell — treating it as a plain string is how "$(rm …)" becomes a command.
 */
export function shellQuote(value: string): string {
  return `"${value.replace(/[\\"$`]/g, (c) => `\\${c}`)}"`;
}

export function buildModalRunLine(input: ModalRunLineInput): string {
  const task = input.task.trim();
  const profile = input.profile.trim();
  const environment = input.environment.trim();
  const modelDtype = input.modelDtype.trim();
  const gpu = input.gpu.trim();
  const parts = [
    // Two env-var ASSIGNMENTS in front of the command, not flags — and for two
    // different reasons. `modal run` has no --profile, and the CLI reads
    // MODAL_PROFILE per process, which is what makes this line safe to paste
    // without re-pointing every other terminal on the machine. DRTC_GPU has no
    // flag it COULD be: both wrappers build `@app.function(gpu=…)` at import,
    // before Click parses anything, so the environment is the only channel.
    `${profile ? `MODAL_PROFILE=${profile} ` : ""}${gpu ? `DRTC_GPU=${gpu} ` : ""}modal run` +
      // `--env` is a `modal run` OPTION, so it goes BEFORE the wrapper path.
      // After it, Click hands it to the wrapper's own local_entrypoint — which
      // has no such parameter — and the command dies on an unknown flag.
      `${environment ? ` --env ${environment}` : ""} ` +
      MODAL_WRAPPERS[input.engine],
    `--policy-path ${input.policyHubId.trim() || POLICY_PATH_PLACEHOLDER}`,
    // Flag order follows each wrapper's own local_entrypoint signature, so the
    // line reads the same way the function does. `--s-min` sits between --fps
    // and --video-codec there, which is where it goes here.
    ...(task ? [`--task ${shellQuote(task)}`] : []),
    // Between --task and --horizon, where both wrappers' local_entrypoint
    // signatures put it. Omitted when unset, like --task: unset is not a
    // default, it is the dtype the checkpoint was saved with.
    ...(modelDtype ? [`--model-dtype ${modelDtype}`] : []),
    // Right after it, the order the two knobs were added in — and omitted the
    // same way, for a stronger reason: 0 is not a step count any sampler would
    // take (the pin raises on it), so the flag's absence is the only way to
    // say "leave the checkpoint's own".
    ...(input.flowSteps ? [`--flow-steps ${input.flowSteps}`] : []),
    `--horizon ${input.horizon}`,
    `--fps ${input.fps}`,
    // RTC only. The sync wrapper has no --s-min flag at all, so emitting it
    // there would make the line fail to parse rather than run with a default.
    ...(input.engine === "rtc" ? [`--s-min ${input.sMin}`] : []),
    `--video-codec ${input.videoCodec}`,
  ];
  // The room is what makes the two sides meet. The GPU side otherwise takes it
  // from the LiveKit-cloud secret's own LIVEKIT_ROOM — which the Lab cannot
  // read and therefore cannot check — so pinning it here is what removes the
  // one mismatch that is invisible by construction.
  if (input.room) parts.push(`--livekit-room ${input.room}`);
  // The Lab's own SFU is only reachable from a Modal container over the
  // tailnet, and its key pair is not the Cloud one the container's secret
  // carries — so the whole transport has to travel on the command line. The
  // key ID is real; only the secret is a placeholder.
  if (input.source === "sfu") {
    parts.push("--tailscale");
    if (input.url) parts.push(`--livekit-url ${input.url}`);
    parts.push(
      `--livekit-api-key ${input.sfuKeyId.trim() || LOCAL_KEY_ID_PLACEHOLDER}`,
    );
    parts.push(`--livekit-api-secret ${LOCAL_SECRET_PLACEHOLDER}`);
  }
  return parts.join(" ");
}
