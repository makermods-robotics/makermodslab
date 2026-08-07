import { Fetcher, apiRequest } from "./apiClient";

export interface StartInferenceRequest {
  follower_port: string;
  follower_config: string;
  policy_ref: string;
  task: string;
  // Which of the ROBOT RECORD's cameras plays each camera role the checkpoint
  // was trained with: { policy-expected camera name: robot-record camera name }.
  // Only the name pairing is sent — which device, and how it's opened (index,
  // unique_id, fps, fourcc, backend), is read out of the record named by
  // `robot_name`, so a run can never open a camera set the saved robot doesn't
  // have. (Capture resolution is the exception — see `camera_dims`.) A binding
  // naming a camera the record lacks is a 400.
  camera_bindings: Record<string, string>;
  // Capture resolution per policy-expected camera name, read off the
  // checkpoint's `image_features` (same forward-the-checkpoint's-own-metadata
  // pattern as `checkpoint_state_dim`). The ONE camera setting not taken from
  // the robot record: lerobot's rollout does not resize frames to the policy's
  // input shape, so a camera must capture at the resolution the policy was
  // trained on. Omitted entries fall back to the record's own width/height.
  camera_dims?: Record<string, { width: number; height: number }>;
  duration_s: number;
  // Bimanual: "single" (default) drives one follower; "bimanual" drives two.
  // In bimanual mode follower_port/follower_config above is the LEFT arm and
  // the right_* fields carry the RIGHT arm. Inference has no leader arms.
  mode?: "single" | "bimanual";
  right_follower_port?: string;
  right_follower_config?: string;
  // Robot record name. Names the record `camera_bindings` resolve against
  // (required whenever any binding is set), and — bimanual only — the BiSO
  // calibration-staging base id.
  robot_name?: string;
  // Checkpoint's flat state width (6 = single arm, 12 = bimanual). Lets the
  // server reject an arm-count mismatch before spawning the rollout subprocess.
  checkpoint_state_dim?: number;
  // Multi-episode EVALUATION mode. Omitted/1 = the historical single rollout.
  // >1 runs N sequential episodes in one session (one model download, one arm
  // preflight, one camera handover), each scored success/failure, ending in an
  // accuracy. Clamped server-side to [1, 200].
  eval_episodes?: number;
  // Which lerobot inference engine runs the rollout. "sync" (the server-side
  // default) does one policy forward per control tick, so the arm pauses
  // between action chunks; "rtc" (Real-Time Chunking) overlaps that forward
  // with action playback on a background thread. Experimental: RTC changes the
  // policy's action-generation path, not just its scheduling.
  inference_engine?: "sync" | "rtc";
}

// Structured startup sub-phase, mirrored from rollout.py's phase constants.
// Names which substep a slow startup is in so the UI can say "Downloading
// model…" / "Connecting to arm…" instead of one opaque spinner. Absent/null
// when no session has seeded a phase yet.
// The startup phases repeat PER EPISODE in evaluation mode (each episode is its
// own rollout subprocess); `resetting`/`finished`/`aborted` are eval-only.
export type InferencePhase =
  | "downloading_model"
  | "starting"
  | "loading_policy"
  | "connecting"
  | "running"
  | "stopping"
  | "stopped"
  | "error"
  // Eval only: an episode ended and was scored; the session is parked waiting
  // for the user to rearrange the scene and start the next one. Also where a
  // CRASHED episode parks, with `error`/`hint` populated.
  | "resetting"
  // Eval only, terminal: every episode ran — `accuracy` is populated.
  | "finished"
  // Eval only, terminal: the user aborted. Partial tally, NO accuracy claimed.
  | "aborted";

// One episode's verdict. `error` (a crash: serial glitch, camera drop, policy
// blow-up) is deliberately NEITHER success nor failure — it's excluded from the
// accuracy denominator so one hardware hiccup can't poison a 20-episode number.
export type EpisodeResult = "success" | "failure" | "error";

// How a finished run turned out (present only on the exited status payload):
//   ok               — clean exit.
//   ran_with_warning — the rollout ran but a noisy shutdown/cleanup tripped
//                      (e.g. torque-disable on a gripper still holding an
//                      object). NOT a real failure — render amber, not red.
//   failed           — a real failure (never got going, or crashed mid-run).
export type InferenceOutcome = "ok" | "ran_with_warning" | "failed";

export interface InferenceStatus {
  inference_active: boolean;
  started_at: number | null;
  rollout_started_at: number | null;
  elapsed_s: number;
  rollout_elapsed_s: number;
  duration_s: number | null;
  policy_ref: string | null;
  log_path: string | null;
  phase?: InferencePhase | null;
  exited?: boolean;
  exit_code?: number | null;
  // Present only on the exited payload. `outcome` classifies the run;
  // `error` is a short snippet mined from the log tail; `hint` is a
  // plain-language, actionable diagnosis. All null when not applicable.
  outcome?: InferenceOutcome | null;
  error?: string | null;
  hint?: string | null;
  // Byte progress of the Hub model download, populated only during the
  // `downloading_model` phase (all null outside it / for a local checkpoint).
  // `download_percent` is null while the total is still unknown → the UI shows
  // an indeterminate bar. The total can grow as file sizes are discovered, so
  // the bar may legitimately step backwards — that's honest, not a glitch.
  download_bytes_done?: number | null;
  download_bytes_total?: number | null;
  download_percent?: number | null;
  // Warn-but-allow arm-identity finding, surfaced once the run is up (the
  // preflight now runs server-side in the background, after the POST returned).
  warning?: string | null;
  // --- Multi-episode evaluation -------------------------------------------
  // False (with null companions) for a plain single rollout — the shape is
  // stable, so `eval_mode` is the only flag worth branching on.
  eval_mode?: boolean;
  // 1-based index of the episode running / just scored, clamped to the total.
  episode_index?: number | null;
  episodes_total?: number | null;
  // Verdicts in episode order; its length is how many episodes have finished.
  episode_results?: EpisodeResult[] | null;
  // successes / (successes + failures). Claimed ONLY on the `finished` payload:
  // an aborted session reports its partial tally with accuracy null, and so
  // does a session where every episode crashed.
  accuracy?: number | null;
}

// The POST now returns immediately: it only validates the request cheaply, then
// hands the model download + arm preflight + subprocess spawn to a background
// worker. So there's no log_path or warning here yet — the inference page polls
// /inference-status for the download progress, the warn-but-allow finding, and
// any failure. A 4xx still surfaces here (mutex/arm-count/policy-ref shape).
export async function startInference(
  baseUrl: string,
  fetcher: Fetcher,
  request: StartInferenceRequest,
): Promise<{ message: string }> {
  return apiRequest<{ message: string }>(
    baseUrl,
    fetcher,
    "/start-inference",
    { method: "POST", body: request, action: "Start inference" },
  );
}

export async function stopInference(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ message: string }> {
  return apiRequest<{ message: string }>(baseUrl, fetcher, "/stop-inference", {
    method: "POST",
    action: "Stop inference",
  });
}

// Evaluation mode only: end the CURRENT episode early and score it a SUCCESS
// ("the robot did the task"). The session stays up and moves into its reset
// phase — this is NOT stopInference, which aborts the whole run.
export async function stopInferenceEpisode(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ message: string }> {
  return apiRequest<{ message: string }>(
    baseUrl,
    fetcher,
    "/inference-episode-stop",
    { method: "POST", action: "Stop inference episode" },
  );
}

// Evaluation mode only: leave the reset phase and start the next episode. The
// reset is user-ended (no auto-timer) — rearranging a bench scene shouldn't be
// on a clock.
export async function startNextInferenceEpisode(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ message: string }> {
  return apiRequest<{ message: string }>(
    baseUrl,
    fetcher,
    "/inference-next-episode",
    { method: "POST", action: "Start next episode" },
  );
}

export async function getInferenceStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<InferenceStatus> {
  return apiRequest<InferenceStatus>(baseUrl, fetcher, "/inference-status", {
    signal,
    action: "Get inference status",
  });
}

// Which run the returned log belongs to. The backend only ever serves a log it
// opened itself, so this is never a guess:
//   "active"   — the currently running session's own log
//   "last_run" — the most recent FINISHED run of this server process
//   null       — there is no log to show
// A live session reporting anything other than "active" simply has not produced
// output yet (it is still downloading/preflighting, or it failed before the
// rollout process started). Rendering `logs` in that case is how a previous
// run's output gets presented as the current run's — the defect this field
// exists to prevent.
export type InferenceLogOwner = "active" | "last_run" | null;

export interface InferenceLog {
  logs: string;
  log_path: string | null;
  belongs_to: InferenceLogOwner;
}

// Tail of the active/most-recent rollout's log file. Read-only + bounded on the
// server (last ~500 lines); empty `logs` (not an error) before output exists.
export async function getInferenceLog(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<InferenceLog> {
  return apiRequest<InferenceLog>(baseUrl, fetcher, "/inference-log", {
    signal,
    action: "Get inference log",
  });
}
