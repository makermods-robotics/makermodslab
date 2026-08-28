import { Fetcher, apiRequest } from "./apiClient";

// Starting a run no longer lives here: launch goes through POST
// /api/v1/sessions (lib/sessionApi.ts startSession, kind "inference") — the
// request carries the robot NAME plus InferenceSessionOptions, and the server
// resolves ports/configs/cameras from the saved record. This module keeps the
// run's rich status/log polling and the kind-specific controls.

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
  | "aborted"
  // Coaching only. These REPLACE `running` for the life of a coaching session,
  // because "running" doesn't answer the one question an operator standing at
  // the arm needs answered: is the robot driving, or am I?
  //   watching   — the policy drives; watch for the failure
  //   holding    — frozen; the arm holds its pose, nothing is recorded
  //   correcting — you are driving through the leader; frames ARE recorded
  | "watching"
  | "holding"
  | "correcting"
  // Coaching only: the arm is physically travelling into position for a
  // handover. Neither steady state is true for that ~2s window.
  | "handing_over"
  // Coaching only: the correction is being written to disk. Synchronous on the
  // runner's control loop, so the arm is frozen and the policy hasn't resumed.
  | "saving";

// lerobot's own DAggerPhase, passed through unrenamed so a status payload and a
// lerobot log line agree. The operator-facing wording is a rendering concern —
// see PHASE_META in InferenceSessionDialog.
export type CoachingPhase =
  | "autonomous"
  | "paused"
  | "correcting"
  // NOT one of lerobot's — ours. Covers the window where `_apply_transition`
  // is driving an arm and no lerobot phase describes what is happening:
  // single-arm the leader is being driven to the follower's pose, bimanual
  // BOTH followers are sliding to meet the leaders. Reporting either steady
  // state there tells the operator the arm is still when it is moving.
  | "handing_over"
  // Also ours: `save_episode()` is writing the parquet and encoding video, on
  // the control loop, with the arm held. Without this the banner inherited
  // "correcting" and kept claiming the operator was driving after they let go.
  | "saving";

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
  // --- Coaching (DAgger) ---------------------------------------------------
  // False (with null companions) for every non-coaching run — the shape is
  // stable, so `coaching` is the only flag worth branching on. Unlike the eval
  // block, this one SURVIVES onto the terminal payload: the dataset name and
  // tally are what the follow-up card needs to offer "merge" and "fine-tune",
  // and they have to outlive the session that produced them.
  coaching?: boolean;
  // Null until the runner reports one — there is a window after the session
  // goes live where neither "the policy is driving" nor "the arm is frozen" is
  // known to be true, and the UI must say "Starting…" rather than guess.
  coaching_phase?: CoachingPhase | null;
  corrections_saved?: number | null;
  corrections_target?: number | null;
  // Recorded correction time across the session, in seconds. Reported next to
  // the count because ten one-second twitches and ten ten-second recoveries are
  // very different datasets.
  correction_seconds?: number | null;
  // The dataset lerobot ACTUALLY created, timestamp and all. Null until the
  // session reports it (shortly after the policy loads) — it cannot be derived
  // from `coaching_dataset_name`, which is only what was asked for.
  coaching_dataset?: string | null;
  // Set when a takeover was refused because the leader sits too far from the
  // follower, with the offending joints named. Cleared on the next successful
  // phase change, so it always describes the LAST attempt, not a history.
  align_error?: string | null;
}

// Kind-agnostic fallback stop. The session dialog stops by session id
// (lib/sessionApi.ts stopSession); this endpoint remains for surfaces that
// know only "an inference run is active" (DeployPanel's Stop button) — stops
// are never owner-gated, so it may target a run another tab started.
export async function stopInference(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ message: string }> {
  return apiRequest<{ message: string }>(
    baseUrl,
    fetcher,
    "/api/v1/stop-inference",
    {
      method: "POST",
      action: "Stop inference",
    },
  );
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
    "/api/v1/inference-episode-stop",
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
    "/api/v1/inference-next-episode",
    { method: "POST", action: "Start next episode" },
  );
}

// --- Coaching controls ------------------------------------------------------
// All five are 409 when no coaching session is live or it is still starting up.
// None of them pre-checks the phase client-side: the runner's phase is
// authoritative and the browser only ever holds a copy that is one poll stale,
// so a command that arrives at the wrong moment is a harmless no-op the runner
// logs — far better than a button that refuses something the arm was ready for.

// Take control from the policy and start recording. ONE press covers the whole
// handover: the policy pauses, the leader arm is driven to the follower's pose
// (so you pick up an arm already where the robot is), and only then does the
// correction begin. Upstream lerobot needs two keys and four presses per cycle;
// the composition lives in the runner so the operator's job is one button.
export async function coachingTakeover(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ message: string }> {
  return apiRequest<{ message: string }>(baseUrl, fetcher, "/coaching-takeover", {
    method: "POST",
    action: "Take over",
  });
}

// End the correction, SAVE it, and hand control back to the policy.
export async function coachingHandback(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ message: string }> {
  return apiRequest<{ message: string }>(baseUrl, fetcher, "/coaching-handback", {
    method: "POST",
    action: "Hand back",
  });
}

// End the correction and DISCARD it. The fumbled-takeover escape: a botched
// correction is poison training data, and upstream lerobot has no way to reject
// one once it starts.
export async function coachingCancel(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ message: string }> {
  return apiRequest<{ message: string }>(baseUrl, fetcher, "/coaching-cancel", {
    method: "POST",
    action: "Discard correction",
  });
}

// Freeze the policy without taking over — the arm holds its pose and nothing is
// recorded. For deciding, or repositioning the scene, without committing.
export async function coachingHold(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ message: string }> {
  return apiRequest<{ message: string }>(baseUrl, fetcher, "/coaching-hold", {
    method: "POST",
    action: "Hold",
  });
}

// Hand control back to the policy from a hold.
export async function coachingResume(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ message: string }> {
  return apiRequest<{ message: string }>(baseUrl, fetcher, "/coaching-resume", {
    method: "POST",
    action: "Resume policy",
  });
}

export async function getInferenceStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<InferenceStatus> {
  return apiRequest<InferenceStatus>(baseUrl, fetcher, "/api/v1/inference-status", {
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
  return apiRequest<InferenceLog>(baseUrl, fetcher, "/api/v1/inference-log", {
    signal,
    action: "Get inference log",
  });
}
