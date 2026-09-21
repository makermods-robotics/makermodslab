import React, { useCallback, useEffect, useRef, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import {
  CheckCircle2,
  Hand,
  Loader2,
  Pause,
  Play,
  GitMerge,
  GraduationCap,
  Square,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useApi } from "@/contexts/ApiContext";
import { useStudio } from "@/contexts/StudioContext";
import { useToast } from "@/hooks/use-toast";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import {
  controlToggleAllowedFor,
  decideCoachKey,
  targetHandlesKey,
} from "./coachKeys";
import { phaseCue } from "./coachCues";
import { PHASE_DOT, PHASE_TEXT, PILL_BG, formatTime } from "./sessionFrame";
import {
  CoachingPhase,
  CoachingState,
  coachStateIsNewer,
  pickCoachingState,
  EpisodeResult,
  InferenceStatus,
  InferenceLogOwner,
  InferencePhase,
  getInferenceStatus,
  getInferenceLog,
  startNextInferenceEpisode,
  stopInference,
  stopInferenceEpisode,
} from "@/lib/inferenceApi";
import LogPanel from "@/components/LogPanel";
import { deleteDataset } from "@/lib/replayApi";
import { formatBytes } from "@/lib/formatBytes";
import { useSessionHeartbeat } from "@/hooks/useSessionHeartbeat";
import { useUnloadWarning } from "@/hooks/useUnloadWarning";
import { ApiError } from "@/lib/apiClient";
import { stopSession, sendCoachingCommand } from "@/lib/sessionApi";
import type { CoachingCommand } from "@/lib/sessionApi";
import { tabOwnerId } from "@/lib/sessionOwner";
import type { CoachingLineage } from "@/contexts/InferenceSessionContext";
import { useCoachingStateSignal } from "@/hooks/useCoachingStateSignal";
import { useCoachingCues } from "@/hooks/useCoachingCues";

const POLL_MS = 1000;

// Human-readable label + tone for each startup sub-phase. Drives the status
// line above the log panel so a slow startup names its substep ("Downloading
// model…", "Connecting to arm…") instead of an opaque spinner. `pulse` marks
// the still-working phases; terminal phases render steady.
const PHASE_META: Record<
  InferencePhase,
  { labelKey: string; tone: "amber" | "green" | "red"; pulse: boolean }
> = {
  downloading_model: { labelKey: "inference.phase.downloadingModel", tone: "amber", pulse: true },
  starting: { labelKey: "inference.phase.starting", tone: "amber", pulse: true },
  loading_policy: { labelKey: "inference.phase.loadingPolicy", tone: "amber", pulse: true },
  connecting: { labelKey: "inference.phase.connecting", tone: "amber", pulse: true },
  running: { labelKey: "inference.phase.running", tone: "green", pulse: true },
  stopping: { labelKey: "inference.phase.stopping", tone: "amber", pulse: true },
  stopped: { labelKey: "inference.phase.stopped", tone: "green", pulse: false },
  error: { labelKey: "inference.phase.error", tone: "red", pulse: false },
  resetting: { labelKey: "inference.phase.resetting", tone: "amber", pulse: false },
  finished: { labelKey: "inference.phase.finished", tone: "green", pulse: false },
  aborted: { labelKey: "inference.phase.aborted", tone: "amber", pulse: false },
  // Coaching. These are the ONLY phases in this map that describe who is
  // holding the arm, so they are worded from the operator's point of view
  // rather than the system's — "the policy is running" is a fact about the
  // software; "watching" is an instruction to the person standing at the robot.
  watching: { labelKey: "inference.phase.watching", tone: "green", pulse: true },
  holding: { labelKey: "inference.phase.holding", tone: "amber", pulse: false },
  correcting: { labelKey: "inference.phase.correcting", tone: "amber", pulse: true },
  handing_over: { labelKey: "inference.phase.handingOver", tone: "amber", pulse: true },
  saving: { labelKey: "inference.phase.saving", tone: "amber", pulse: true },
  attempt_reset: { labelKey: "inference.phase.attemptReset", tone: "amber", pulse: false },
};

// The big banner shown during a coaching session, keyed by lerobot's own phase.
// Deliberately louder than everything else in this dialog: the operator is
// looking at the ARM, not the screen, and has to be able to tell at a glance —
// peripherally — whether the robot or they are in control. `hintKey` is the
// second line; `accent` drives border and text, never colour alone (the word
// carries the meaning on its own for anyone who can't distinguish the hues).
// The four beats a coaching session repeats, in order. The banner shows ALL
// FOUR, always, with the current one lit.
//
// A phase name says where you are; it does not say what comes next, and "what
// comes next" is the question an operator standing at a robot with both hands
// on a leader arm actually has. Reading a position in a fixed cycle answers
// both at once, and it answers the second one without reading at all — the lit
// stop moves left to right, so peripheral vision alone tells you the session
// advanced. Never translated: these are four fixed landmarks, and their job is
// to be the same four shapes in the same four places every time.
const COACH_STEPS = ["WATCH", "DRIVE", "SAVE", "RESET"] as const;

// Which stop each phase lights (1-4), and the qualifier that rides beside the
// title. `step: 0` means no stop is lit — true only before the session has
// reported a phase, when claiming a position in the loop would be a guess.
//
// `badgeKey` carries what the stop cannot: four different states light RESET,
// and the three parked ones differ ONLY in whether the arm is safe to grab —
// the one difference on this screen an operator can be hurt by getting wrong.
type CoachBanner = {
  titleKey: string;
  hintKey: string;
  accent: string;
  bg: string;
  step: 0 | 1 | 2 | 3 | 4;
  badgeKey?: string;
};

const COACH_BANNER: Record<CoachingPhase, CoachBanner> = {
  autonomous: {
    titleKey: "inference.coachBanner.watching.title",
    hintKey: "inference.coachBanner.watching.hint",
    accent: "text-ok border-ok/40",
    bg: "bg-ok/10",
    step: 1,
  },
  paused: {
    // Deliberately the QUIETEST of the four. Held is the safe state — nothing
    // is moving and nothing is being recorded — so it gets neutral/muted, not
    // the warning colour it used to share with "you're driving".
    titleKey: "inference.coachBanner.held.title",
    hintKey: "inference.coachBanner.held.hint",
    accent: "text-muted-foreground border-border",
    bg: "bg-muted/50",
    // Still on WATCH, not a stop of its own: nothing has advanced, the policy
    // has simply stopped. The badge is what says so.
    step: 1,
    badgeKey: "inference.coachBadge.paused",
  },
  handing_over: {
    titleKey: "inference.coachBanner.handingOver.title",
    hintKey: "inference.coachBanner.handingOver.hint",
    accent: "text-warn border-warn",
    bg: "bg-warn/20",
    step: 2,
    badgeKey: "inference.coachBadge.armMoving",
  },
  resetting: {
    titleKey: "inference.coachBanner.resetting.title",
    hintKey: "inference.coachBanner.resetting.hint",
    accent: "text-muted-foreground border-border",
    bg: "bg-muted/50",
    step: 4,
    badgeKey: "inference.coachBadge.armMoving",
  },
  saving: {
    titleKey: "inference.coachBanner.saving.title",
    hintKey: "inference.coachBanner.saving.hint",
    accent: "text-muted-foreground border-border",
    bg: "bg-muted/50",
    step: 3,
  },
  poised: {
    // The one banner that is an INSTRUCTION rather than a report, and it has to
    // read as one from across the room. `poised` and `paused` are both lerobot
    // `paused` — both arms still, nothing recording — but they ask for opposite
    // things: HELD says "nothing is happening", this says "your arm is ready,
    // take it". So it deliberately does NOT inherit HELD's muted treatment; it
    // gets a saturated fill like `correcting`, in a different hue so the two
    // loud states are never confused in peripheral vision (amber = act, red =
    // you are driving and every frame is being kept).
    titleKey: "inference.coachBanner.poised.title",
    hintKey: "inference.coachBanner.poised.hint",
    accent: "text-white border-warn",
    bg: "bg-warn",
    // DRIVE, not WATCH: the takeover has begun and the loop has advanced — the
    // second press finishes it rather than starting something new.
    step: 2,
    // Says the thing the operator is about to bet their hand on: the leader is
    // lined up with the follower and is holding still under torque.
    badgeKey: "inference.coachBadge.aligned",
  },
  correcting: {
    // The LOUDEST thing on the display. This is the only phase where a human is
    // driving a robot and every frame is being written to disk, and it has to
    // be distinguishable from HELD across a workshop, in peripheral vision.
    // Saturated fill and a live pulse do that; a 5% background alpha did not.
    titleKey: "inference.coachBanner.correcting.title",
    hintKey: "inference.coachBanner.correcting.hint",
    accent: "text-destructive-foreground border-destructive",
    bg: "bg-destructive",
    step: 2,
    badgeKey: "inference.coachBadge.recording",
  },
};

// Shown before the runner has reported any phase at all. Not a control-state
// claim, because at that point we do not have one: `coaching_phase` is null
// between the session going live and the control loop's first PHASE event.
// Parked straight after a reset: the arm is home and limp, the scene is being
// rearranged, and the next move is the NEXT ATTEMPT. Distinct from plain HELD
// because the instruction is different — telling the operator "space to take
// over" here is what produced the unwanted correction.
const COACH_BANNER_PARKED: CoachBanner = {
  titleKey: "inference.coachBanner.parked.title",
  hintKey: "inference.coachBanner.parked.hint",
  // Warn/amber, matching the eval reset screen below (border-warn/40 bg-warn/10)
  // and the recording dialog's reset phase. Green read as "all good, nothing to
  // do" — the opposite of the truth here, where the session is waiting on the
  // operator to physically rearrange the scene before anything else happens.
  accent: "text-warn border-warn/40",
  bg: "bg-warn/10",
  step: 4,
  // The one badge that is a safety claim rather than a label.
  badgeKey: "inference.coachBadge.limp",
};

// Same parked moment, but the reset did NOT finish cleanly: the follower never
// reached home. The scene is safe to rearrange, the arm is not where the next
// attempt expects it, and saying "RESET" there would be a lie the operator
// only discovers when the next attempt starts from the wrong pose.
const COACH_BANNER_PARKED_STUCK: CoachBanner = {
  titleKey: "inference.coachBanner.parkedStuck.title",
  hintKey: "inference.coachBanner.parkedStuck.hint",
  accent: "text-warn border-warn",
  bg: "bg-warn/20",
  step: 4,
  badgeKey: "inference.coachBadge.notHome",
};

// Parked and homed, but we cannot confirm the follower went limp — either the
// runner said so outright, or it's an older runner that reports nothing (null).
// Both take this branch: never promise a limp arm we cannot confirm, because
// the operator acts on that promise by grabbing it.
const COACH_BANNER_PARKED_RIGID: CoachBanner = {
  titleKey: "inference.coachBanner.parkedRigid.title",
  hintKey: "inference.coachBanner.parkedRigid.hint",
  accent: "text-warn border-warn",
  bg: "bg-warn/20",
  step: 4,
  badgeKey: "inference.coachBadge.mayBeStiff",
};

// Shown ONLY once the operator has marked the recovery boundary, i.e. only when
// they have told us this takeover had a rescue in it. It used to be the other
// way round — RECOVERING was the default and this was reachable only after
// pressing G — which meant every operator who had not adopted the gesture was
// told, on the largest element on screen, to rewind an arm that needed no
// rewinding. RaC's decomposition is worth surfacing, but it is a claim the
// OPERATOR makes; the UI must not make it on their behalf.
const COACH_BANNER_CORRECTING: CoachBanner = {
  titleKey: "inference.coachBanner.correcting2.title",
  hintKey: "inference.coachBanner.correcting2.hint",
  accent: "text-destructive-foreground border-destructive",
  bg: "bg-destructive",
  step: 2,
  badgeKey: "inference.coachBadge.recording",
};

const COACH_BANNER_STARTING: CoachBanner = {
  titleKey: "inference.coachBanner.starting.title",
  hintKey: "inference.coachBanner.starting.hint",
  accent: "text-muted-foreground border-border",
  bg: "bg-muted/50",
  // No stop lit. The session has not reported a phase, so it has no position
  // in the loop yet, and lighting WATCH to avoid an empty rail would claim the
  // policy is driving an arm that is not even connected.
  step: 0,
};

// Per-episode verdict styling for the tally + the final per-episode list.
const RESULT_META: Record<
  EpisodeResult,
  { labelKey: string; dot: string; text: string }
> = {
  success: { labelKey: "inference.result.success", dot: "bg-ok", text: "text-ok" },
  failure: { labelKey: "inference.result.failure", dot: "bg-muted-foreground", text: "text-muted-foreground" },
  error: { labelKey: "inference.result.error", dot: "bg-destructive", text: "text-destructive" },
};

function tally(results: EpisodeResult[]): Record<EpisodeResult, number> {
  return {
    success: results.filter((r) => r === "success").length,
    failure: results.filter((r) => r === "failure").length,
    error: results.filter((r) => r === "error").length,
  };
}

// The pill / dot / text palettes and the mm:ss clock now live in
// `sessionFrame.tsx`, shared with the remote (DRTC) session body so the two
// dialogs cannot drift apart cosmetically. Nothing about their behaviour
// changed in the move.

/**
 * The live inference run as a modal dialog over whatever launched it —
 * replaces the old /inference page (the polling/safety logic is ported
 * verbatim; every navigate-home became `onExit`). While the run is live
 * (including the connecting/setup window before the first status lands) the
 * dialog can't be dismissed by ESC / outside click / X: leaving stops the
 * arm, so the only in-app exits are the explicit Stop flow and the
 * clean-finish auto-close. An abandoned page (tab close, crash) is covered by
 * the session's server-side lease: the missed heartbeats make the server stop
 * the run. Once the run has ended, dismissal is free.
 */
const InferenceSessionDialog: React.FC<{
  /** Identity from POST /api/v1/sessions — this dialog heartbeats its lease
   * and stops it by id. Null only for defensive robustness (e.g. a stale
   * launcher); the stop then falls back to the kind-level endpoint. */
  sessionId: string | null;
  /** Called for every exit — closes the dialog, landing back where the run
   * was launched from. */
  onExit: () => void;
  /** Coaching only: which skill is being coached and what it was trained on,
   * so the end-of-session summary can offer the merge + fine-tune rather than
   * describe it. Null for a plain run or an eval, and null for a coaching
   * session whose page was reloaded mid-run — see CoachingLineage. */
  coachingLineage?: CoachingLineage | null;
}> = ({ sessionId, onExit, coachingLineage }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { openStudio, deployPrefill } = useStudio();
  const navigate = useNavigate();
  const { toast } = useToast();
  const { t } = useTranslation();
  const [status, setStatus] = useState<InferenceStatus | null>(null);
  const [logs, setLogs] = useState("");
  // Which run the fetched log belongs to. Never inferred from the text: the
  // backend says so explicitly, because a log file on disk carries no evidence
  // of which run wrote it.
  const [logOwner, setLogOwner] = useState<InferenceLogOwner>(null);
  const [stopping, setStopping] = useState(false);
  // Eval mode only: in-flight guards for the two per-episode controls, so a
  // double-click can't fire "succeeded" or "next episode" twice.
  const [endingEpisode, setEndingEpisode] = useState(false);
  const [startingNext, setStartingNext] = useState(false);
  // Coaching only: one in-flight guard for every coaching control. Most are
  // mutually exclusive transitions of a single state machine, so a second
  // command sent while the first is still crossing the wire can only ever be
  // one the runner will reject — better to hold the buttons for the ~50ms round
  // trip. (RECOVERED is an annotation rather than a transition, but shares the
  // guard: double-marking a boundary is exactly as unwanted.)
  const [coachBusy, setCoachBusy] = useState(false);
  // What the operator just asked for, shown the instant they ask. The banner
  // TITLE stays server-truth — never claim who holds the arm before the runner
  // says so — but a press that produces no visible change for a whole poll
  // interval reads as a dead control, and the trained response to a dead
  // control is to press it again. This is the acknowledgement, not a
  // prediction. Cleared when the next status lands.
  const [pendingAction, setPendingAction] = useState<string | null>(null);
  // Deleting the corrections dataset from the end-of-session summary.
  // `confirmDelete` makes the button a two-press control rather than opening a
  // dialog on top of a dialog — the action is cheap to offer and irreversible
  // to take, so it needs a beat, not a modal.
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deletingDataset, setDeletingDataset] = useState(false);
  const [datasetDeleted, setDatasetDeleted] = useState(false);
  const exitedRef = useRef(false);
  // Independent flag: we may request a stop (safety net) before the run
  // is actually inactive. We must not flip exitedRef yet — that
  // would block the natural completion path on the next tick.
  const stopRequestedRef = useRef(false);
  // Set once we've captured a finished (exited) payload we want to stay on —
  // a failure/warning we're surfacing inline. Freezes further polling so the
  // next idle status (which lacks outcome/error/hint, since the subprocess is
  // already reaped) can't clobber the error display.
  const doneRef = useRef(false);
  // The warn-but-allow arm-identity finding now arrives on the status payload
  // (the preflight runs server-side in the background), not the start response.
  // Toast it once when first seen so it isn't repeated on every poll.
  const warnedRef = useRef(false);
  // The Escape-moved notice fires once per mount, not once per press: an
  // operator jabbing Escape because nothing happened must not be answered with
  // a stack of identical toasts.
  const escapeHintShownRef = useRef(false);

  // Same one-per-mount discipline for "there is nothing left to take back".
  // The runner holds exactly one correction, so a second Backspace in a row
  // lands here; the operator needs the rule explained, not repeated at them
  // every press while they are looking at the arm.
  const nothingToDropHintShownRef = useRef(false);

  // Safety net: a policy must never keep driving the arm with nobody
  // watching. That guarantee is server-side now — while the run is live
  // (treating the pre-first-status window as live, since the launcher just
  // started it) this tab renews the session's lease, and if the page goes
  // away the missed heartbeats make the SERVER stop the run. The courtesy
  // beforeunload only keeps an accidental tab-close from being silent; the
  // old exit guard's beacon/back-confirm/unmount-stop machinery is retired.
  const sessionLive = status == null || status.inference_active === true;
  useSessionHeartbeat(sessionId, tabOwnerId(), sessionLive);
  useUnloadWarning(sessionLive);

  // Stop this run: by session id when we have one (a 404 means the session is
  // already gone — fine, the poll shows the ending), else the kind-level stop.
  const stopThisRun = useCallback(async () => {
    if (sessionId) {
      try {
        await stopSession(baseUrl, fetchWithHeaders, sessionId);
        return;
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return;
        throw e;
      }
    }
    await stopInference(baseUrl, fetchWithHeaders);
  }, [sessionId, baseUrl, fetchWithHeaders]);

  useEffect(() => {
    let cancelled = false;
    const stopIfHung = async () => {
      try {
        await stopThisRun();
      } catch {
        // The next status poll will surface the failure if it persists.
      }
    };
    const tick = async () => {
      // Once we've frozen on a finished-with-error payload, stop polling: a
      // later idle status would drop the outcome/error/hint we're showing.
      if (doneRef.current) return;
      try {
        const next = await getInferenceStatus(baseUrl, fetchWithHeaders);
        if (cancelled) return;
        // The pending acknowledgement stands until the runner's phase actually
        // moves — so "Taking over…" stays up for the whole handover rather than
        // blinking out on the first poll while the arm is still travelling.
        setStatus((prev) => {
          if (prev?.coaching_phase !== next.coaching_phase)
            setPendingAction(null);
          // The poll REPLACES, and it carries a snapshot taken when the request
          // was issued. A push that landed while it was in flight would be
          // overwritten with older state — reverting the banner to "the policy
          // is driving" while the leader is already gliding under torque, and
          // firing the handback cue in the middle of a correction. Coaching
          // fields therefore keep whichever version is newer, and the poll
          // stays authoritative for everything else.
          if (prev && coachStateIsNewer(prev, next)) {
            return { ...next, ...pickCoachingState(prev) };
          }
          return next;
        });
        // Surface the server's warn-but-allow arm-identity finding once.
        if (next.warning && !warnedRef.current) {
          warnedRef.current = true;
          toast({
            title: t("inference.toast.startedWarningTitle"),
            description: next.warning,
            duration: 10000,
          });
        }
        // Pull the rollout log tail on the same tick so the panel stays live.
        // Best-effort: a log fetch failure must not disturb status handling.
        try {
          const log = await getInferenceLog(baseUrl, fetchWithHeaders);
          if (!cancelled) {
            setLogs(log.logs);
            setLogOwner(log.belongs_to);
          }
        } catch {
          // Ignore; the next tick retries.
        }
        // Handle a finished run.
        if (!next.inference_active && !exitedRef.current) {
          // A real failure or a cleanup-warning: keep the user here so the
          // hint + error snippet (rendered near the log panel) are readable
          // instead of flashing a toast and bouncing away. Freeze polling on
          // this payload.
          if (next.exited && next.outcome && next.outcome !== "ok") {
            doneRef.current = true;
            // Also surface a simple bottom-right toast (min_stable behavior):
            // the full hint + error snippet stay readable in the dialog, the
            // toast is the at-a-glance "it broke" signal.
            const failed = next.outcome === "failed";
            toast({
              title: failed
                ? t("inference.toast.failedTitle")
                : t("inference.toast.ranWithWarningTitle"),
              // next.hint / next.error are backend prose — shown as-is.
              description:
                next.hint ??
                next.error?.split("\n").at(-1) ??
                t("inference.toast.seeLog"),
              variant: failed ? "destructive" : undefined,
              duration: 10000,
            });
            return;
          }
          // An evaluation that ran its course (or was aborted) ends on its
          // SUMMARY, not by bouncing away: the accuracy and the per-episode
          // list are the entire point of the run. Freeze here and let the user
          // close. Checked after the failure branch above so a session-level
          // startup failure still renders as an error, not a summary.
          // A coaching session likewise ends on its SUMMARY, not by bouncing
          // away — the dataset it produced and the two follow-up actions
          // (merge, fine-tune) are the whole reason the operator stood there
          // taking over. Checked before the eval branch only for symmetry; the
          // two modes are mutually exclusive server-side.
          // No markHandled() here: the exit guard this used to notify was
          // retired with the move to a server-side lease (see useUnloadWarning).
          if (next.coaching && next.exited) {
            exitedRef.current = true;
            doneRef.current = true;
            const saved = next.corrections_saved ?? 0;
            toast({
              title:
                next.phase === "aborted"
                  ? t("inference.toast.coachStoppedTitle")
                  : t("inference.toast.coachCompleteTitle"),
              description: t("inference.toast.coachSaved", { count: saved }),
              duration: 10000,
            });
            return;
          }
          if (next.eval_mode && next.exited) {
            exitedRef.current = true;
            doneRef.current = true;
            toast({
              title:
                next.phase === "aborted"
                  ? t("inference.toast.evalAbortedTitle")
                  : t("inference.toast.evalCompleteTitle"),
              description:
                next.phase === "aborted"
                  ? t("inference.toast.evalAbortedDescription")
                  : next.accuracy != null
                    ? t("inference.toast.evalAccuracy", {
                        percent: Math.round(next.accuracy * 100),
                      })
                    : t("inference.toast.evalNoScoreable"),
              duration: 10000,
            });
            return;
          }
          // A clean finish (completed / user stop): toast + auto-close.
          exitedRef.current = true;
          doneRef.current = true;
          if (next.exited) {
            toast({
              title: t("inference.toast.finishedTitle"),
              description: t("inference.toast.finishedDescription"),
            });
          }
          onExit();
          return;
        }
        // Safety net: only fire after the rollout *main loop* has actually
        // started (lerobot honours --duration there). Setup time — policy
        // load, snapshot_download, bus connect, camera connect — can take
        // 10–30s and must NOT count against the user's configured duration.
        if (
          next.inference_active &&
          next.rollout_started_at != null &&
          next.duration_s != null &&
          next.duration_s > 0 &&
          next.rollout_elapsed_s > next.duration_s + 10 &&
          !stopRequestedRef.current
        ) {
          stopRequestedRef.current = true;
          toast({
            title: t("inference.toast.hungTitle"),
            description: `Rollout past duration by ${Math.round(
              next.rollout_elapsed_s - next.duration_s,
            )}s. Stopping.`,
            variant: "destructive",
          });
          stopIfHung();
        }
      } catch (e) {
        if (!cancelled) {
          toast({
            title: t("inference.toast.lostConnectionTitle"),
            description: e instanceof Error ? e.message : String(e),
            variant: "destructive",
          });
        }
      }
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [baseUrl, fetchWithHeaders, onExit, toast, stopThisRun]);

  // Stops immediately — no confirmation dialog. The follower eases back to its
  // start pose and releases torque; `stopping` guards against double-fires
  // while the request is in flight.
  const handleStop = async () => {
    setStopping(true);
    try {
      await stopThisRun();
      // Status poll will catch the inactive state and close the dialog.
    } catch (e) {
      setStopping(false);
      toast({
        title: t("inference.toast.stopFailedTitle"),
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  // Eval mode: "the robot did the task". Ends THIS episode and scores it a
  // success; the session stays up and moves into its reset phase. Deliberately
  // not `handleStop` — that aborts the whole evaluation.
  const handleEpisodeSuccess = async () => {
    setEndingEpisode(true);
    try {
      await stopInferenceEpisode(baseUrl, fetchWithHeaders);
      // The status poll picks up the reset phase and the updated tally.
    } catch (e) {
      toast({
        title: t("inference.toast.endEpisodeFailedTitle"),
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setEndingEpisode(false);
    }
  };

  const handleNextEpisode = async () => {
    setStartingNext(true);
    try {
      await startNextInferenceEpisode(baseUrl, fetchWithHeaders);
    } catch (e) {
      toast({
        title: t("inference.toast.nextEpisodeFailedTitle"),
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setStartingNext(false);
    }
  };

  // --- Coaching (DAgger) ---------------------------------------------------
  const coachMode = status?.coaching === true;
  const coachPhase = (status?.coaching_phase ?? null) as CoachingPhase | null;
  const correctionsSaved = status?.corrections_saved ?? 0;
  const correctionsTarget = status?.corrections_target ?? 0;
  const correctionSeconds = status?.correction_seconds ?? 0;
  const coachDataset = status?.coaching_dataset ?? null;
  const alignError = status?.align_error ?? null;
  const discardNotice = status?.discard_notice ?? null;
  const awaitingAttempt = status?.awaiting_attempt === true;
  // Outcome of the last reset, used ONLY to pick the parked banner. Both stay
  // null on an older runner, which the banner treats as "cannot promise" —
  // see COACH_BANNER_PARKED_RIGID.
  const resetHomed = status?.reset_homed ?? null;
  const resetLimp = status?.reset_limp ?? null;
  // Null until the operator marks the recovery/correction boundary for the
  // takeover in progress. See handleCoachRecovered.
  const recoveryMarkedAt = status?.recovery_marked_at ?? null;
  const correctionsLabelled = status?.corrections_labelled ?? 0;
  // Only meaningful once the session is actually driving. `coachPhase` is null
  // until the runner reports one, which is what the "Starting…" banner covers.
  const coachLive =
    coachMode &&
    status?.inference_active === true &&
    status?.rollout_started_at != null;

  // Coaching state arrives by PUSH as well as by poll. The poll below stays
  // exactly as it was and remains the reconciler — this only removes latency
  // from the one signal where latency is a safety property. Merging rather
  // than replacing keeps every non-coaching field (elapsed, log offsets, the
  // hung-run watchdog's inputs) owned solely by the poll.
  const applyCoachingState = useCallback((coaching: CoachingState) => {
    setStatus((prev) => {
      if (prev == null) return prev;
      // Clearing the acknowledgement HERE as well as in the poll. It used to
      // clear only when the poll was first to see a phase change — and once
      // pushes started arriving first, the poll always found the phases equal
      // and never cleared it. "Taking over…" then replaced the banner's
      // instruction for the rest of the session, and the better the push
      // worked the more reliably it stuck.
      if (prev.coaching_phase !== coaching.coaching_phase)
        setPendingAction(null);
      return { ...prev, ...coaching };
    });
  }, []);
  useCoachingStateSignal(applyCoachingState, coachLive);

  // --- Audio cues ----------------------------------------------------------
  // The operator is watching the arm, not the screen. See useCoachingCues.
  const playCue = useCoachingCues(coachLive);
  const lastCuedPhaseRef = useRef<CoachingPhase | null>(null);
  const lastCuedSavedRef = useRef<number | null>(null);
  const lastCuedAlignRef = useRef<string | null | undefined>(undefined);
  const lastCuedDiscardRef = useRef<string | null | undefined>(undefined);

  useEffect(() => {
    if (!coachLive) {
      // Reset on session end so the next session's first phase cues rather
      // than being swallowed as "no change".
      lastCuedPhaseRef.current = null;
      lastCuedSavedRef.current = null;
      lastCuedAlignRef.current = undefined;
      lastCuedDiscardRef.current = undefined;
      return;
    }
    // Control changed hands. These two are the cues that matter most: they say
    // who is holding the arm, which is the thing the operator cannot safely be
    // wrong about and the thing the screen is worst at telling them.
    if (coachPhase !== lastCuedPhaseRef.current) {
      const previous = lastCuedPhaseRef.current;
      lastCuedPhaseRef.current = coachPhase;
      // This effect runs BEFORE the discard effect below, so the ref still
      // holds the previously-cued notice: a difference here means a discard
      // landed on this same update and this transition is not a hand-back.
      const cue = phaseCue({
        previous,
        next: coachPhase,
        discardPending:
          lastCuedDiscardRef.current !== undefined &&
          !!discardNotice &&
          discardNotice !== lastCuedDiscardRef.current,
      });
      if (cue) playCue(cue);
    }
  }, [coachLive, coachPhase, discardNotice, playCue]);

  useEffect(() => {
    if (!coachLive) return;
    const saved = status?.corrections_saved ?? 0;
    // First observation of a session establishes the baseline rather than
    // firing — a dialog reopened mid-session must not replay the tally.
    if (lastCuedSavedRef.current == null) {
      lastCuedSavedRef.current = saved;
      return;
    }
    if (saved > lastCuedSavedRef.current) playCue("saved");
    lastCuedSavedRef.current = saved;
  }, [coachLive, status?.corrections_saved, playCue]);

  useEffect(() => {
    if (!coachLive) return;
    // Keyed on the FIELD, not on the message text. Choosing the cue by
    // sniffing for the word "discarded" in backend prose meant a copy edit
    // would silently downgrade the discard thud to the refusal beeps.
    // First observation establishes a baseline rather than firing, matching the
    // corrections tally above: a dialog opened onto a session that already has
    // a standing notice must not replay an event the operator never saw.
    if (lastCuedDiscardRef.current === undefined) {
      lastCuedDiscardRef.current = discardNotice;
      return;
    }
    if (discardNotice && discardNotice !== lastCuedDiscardRef.current) {
      playCue("discarded");
    }
    lastCuedDiscardRef.current = discardNotice;
  }, [coachLive, discardNotice, playCue]);

  useEffect(() => {
    if (!coachLive) return;
    // Baseline on first observation, as with the discard notice and the tally:
    // a refusal already standing on the payload when the dialog mounts happened
    // before the operator was looking, and replaying it sends them to re-align
    // a leader arm that is already fine.
    if (lastCuedAlignRef.current === undefined) {
      lastCuedAlignRef.current = alignError;
      return;
    }
    if (alignError && alignError !== lastCuedAlignRef.current)
      playCue("refused");
    lastCuedAlignRef.current = alignError;
  }, [coachLive, alignError, playCue]);

  // One helper for every coaching control: send, and let the runner's own push
  // render the result — the backend emits the new coaching state the instant it
  // changes (see useCoachingStateSignal), with the 1 Hz poll behind it as the
  // reconciler. Nothing is optimistically applied to `coachPhase`: the runner
  // owns the phase, and a browser that painted "you're driving" a beat before
  // the arm actually handed over would be lying at the one moment it matters.
  // That is why the push mattered — it makes honesty cheap instead of slow.
  const sendCoachCommand = useCallback(
    async (command: CoachingCommand, label: string, pending: string) => {
      // No id, no command. Every coaching verb is addressed to a session, so a
      // dialog that somehow has no id cannot safely send one — refusing loudly
      // beats posting to a session-agnostic endpoint and hoping the running
      // session is the one on screen.
      if (!sessionId) {
        toast({
          title: t("inference.coach.cmd.failed", { action: label }),
          description:
            "This coaching session has no id — reload the page and start it again.",
          variant: "destructive",
        });
        return;
      }
      setCoachBusy(true);
      setPendingAction(pending);
      try {
        // Addressed to THIS session's id, never to "whatever is running".
        // A dialog left open across a session change then gets a 404 it can
        // report, rather than silently taking over an arm in a session the
        // operator is no longer looking at.
        await sendCoachingCommand(baseUrl, fetchWithHeaders, sessionId, command);
      } catch (e) {
        setPendingAction(null);
        // A session that has already ended is not a failure the operator did.
        //
        // `coachLive` is derived from the STATUS, which lags the runner by up
        // to a poll, so there is an unavoidable window where the keys are still
        // armed and the session behind them is gone — most often right after a
        // reset, because that is when the last correction lands and the runner
        // finalizes. Reporting that as a red "Reset failed" made a normal end
        // of session look like a crash.
        //
        // Kept narrow on purpose: only the two 409s that mean "there is nothing
        // to command any more". Every other failure still shouts.
        // Kept narrow: the two 409s that mean "there is nothing to command any
        // more", plus the 404 that means this dialog's session is already gone
        // — which only became reachable again when the command went back to
        // being session-scoped. Every other failure still shouts.
        const benign =
          e instanceof ApiError &&
          ((e.status === 409 &&
            /no coaching session is active|shutting down/i.test(e.detail ?? "")) ||
            e.status === 404);
        if (benign) {
          toast({
            title: t("inference.coach.sessionEnded.title"),
            description: t("inference.coach.sessionEnded.body"),
          });
        } else {
          toast({
            // `label` arrives already translated from the call site.
            title: t("inference.coach.cmd.failed", { action: label }),
            description: e instanceof Error ? e.message : String(e),
            variant: "destructive",
          });
        }
      } finally {
        setCoachBusy(false);
      }
    },
    [baseUrl, fetchWithHeaders, sessionId, toast, t],
  );

  // Wrap every coaching button's click so the button does not KEEP focus.
  //
  // This is the other half of the duplicate-key complaint, and the harder half
  // to see. `targetHandlesKey` deliberately lets a keystroke through to any
  // focused control — without it, Space and Enter were swallowed for every
  // button in the dialog and none could be operated by keyboard at all. But a
  // MOUSE click also leaves the button focused, and from then on Space and
  // Enter activate whatever was last clicked instead of reaching the coaching
  // handler. Click "Take over" with the mouse, then press Enter expecting
  // "task done", and you hand the arm back instead: one key, two meanings,
  // decided by something invisible.
  //
  // Blurring on click puts focus back on the dialog container, so the keys mean
  // what their labels say. Tab-focusing a button and pressing Enter still
  // activates that button, which is what a keyboard user asks for.
  const pressAndBlur = useCallback(
    (fn: () => void) => (e: React.MouseEvent<HTMLButtonElement>) => {
      e.currentTarget.blur();
      fn();
    },
    [],
  );

  // Space is the whole interaction. Upstream lerobot spends two keys and four
  // presses per correction cycle (pause, correct, end, resume) and asks the
  // operator to track a three-state machine while an arm is about to knock
  // something over. Here one key toggles control, and the state machine stays
  // the runner's problem.
  // Control may only change hands when the policy is ACTUALLY DRIVING, or held
  // mid-attempt. Not during a reset, a handover, a save, or while parked
  // waiting for the next attempt.
  //
  // Taking over in those phases produces no correction — the policy is not
  // driving, so there is no failure being corrected. It records a plain
  // teleoperated demonstration into the corrections dataset, and at training
  // time those frames are indistinguishable from real corrections. A stray
  // space during a reset was enough to do it.
  //
  // The phase table itself lives in coachKeys.ts beside the key that spends it,
  // so the rule is stated once and can be tested — `poised` (the second press
  // of a takeover) is allowed there, and every phase where the policy is not
  // driving the attempt still is not.
  const controlToggleAllowed = controlToggleAllowedFor(
    coachPhase,
    awaitingAttempt,
  );

  const handleCoachToggle = useCallback(() => {
    if (coachPhase === "correcting") {
      sendCoachCommand(
        "handback",
        t("inference.coach.cmd.handBack"),
        t("inference.coach.cmd.handingBack"),
      );
      return;
    }
    // Guarded here as well as on the key and the button: three callers, one
    // rule, and the cost of getting it wrong is a demonstration silently filed
    // as a correction.
    if (!controlToggleAllowed) return;
    sendCoachCommand(
      "takeover",
      t("inference.coach.cmd.takeOver"),
      t("inference.coach.cmd.takingOver"),
    );
  }, [coachPhase, controlToggleAllowed, sendCoachCommand, t]);

  // Shift+Space: stop the policy without taking over — "wait, let me think".
  // Maps to lerobot's bare pause, and that is now ALL it maps to.
  //
  // It used to be a three-way toggle: resume from `paused`, hand back from
  // `correcting`, hold otherwise. Two of those three were a second key for
  // something the operator already had one for — bare Space hands back, Enter
  // starts the next attempt — so the same chord meant a different thing in
  // every phase, and the phase is precisely what an operator with their hands
  // on the leader cannot see. What is left is the one gesture nothing else
  // offers: freezing a policy that is currently driving.
  const handleCoachHold = useCallback(() => {
    if (coachPhase !== "autonomous") return;
    sendCoachCommand(
      "hold",
      t("inference.coach.cmd.hold"),
      t("inference.coach.cmd.holding"),
    );
  }, [coachPhase, sendCoachCommand, t]);

  // Discard the correction in progress — THE feature that makes coaching usable.
  // A fumbled takeover is poison training data, and upstream lerobot saves every
  // correction permanently with no way to reject one.
  //
  // It now also brings the arm home and parks for a scene reset. A discard means
  // the last few seconds were a mess, and the scene almost always needs setting
  // up again afterwards; leaving the follower holding the pose the fumble ended
  // in made the operator work out what to press next. That pair — discard, then
  // reset — was previously a SECOND control ("Arm stuck? Recover") with its own
  // button, endpoint and keybinding, which is now gone: one decision, one
  // control. It is accepted from every phase, which was that control's reason to
  // exist and is what still gives a wedged correction a way out.
  const handleCoachDiscard = useCallback(() => {
    sendCoachCommand(
      "cancel",
      t("inference.coach.cmd.discard"),
      t("inference.coach.cmd.discarding"),
    );
  }, [sendCoachCommand, t]);

  // Both halves of the merge have to be known for the one-click path to mean
  // anything: the corrections we just recorded, and the dataset the coached
  // checkpoint was last trained on. Missing either — a run launched outside the
  // studio, or a page reloaded mid-session — falls back to the written
  // instructions rather than offering a button that would guess.
  const canHandOff = Boolean(
    coachDataset && coachingLineage?.trainingDatasetRepoId && !datasetDeleted,
  );

  // Close the session, open the merge with both datasets ticked, and remember
  // to open training on the result. The chaining lives in CollectPanel (which
  // owns the merge dialog); this only states the intent. See MergePrefill.
  //
  // No navigate("/") despite the studio overlay living on Launchpad: lineage is
  // only ever set by DeployPanel, which IS the studio, so a session that can
  // reach this button was launched from there and is already on that route.
  // Hand the next step to the surface the operator LANDS on, rather than
  // keeping it inside the modal they are closing. Same router-state contract
  // CollectHandoff uses after a recording session — see CoachHandoff.
  const exitWithHandoff = useCallback(() => {
    if (coachMode && correctionsSaved > 0 && coachDataset && !datasetDeleted) {
      navigate("/", {
        replace: true,
        state: {
          coached: {
            repo_id: coachDataset,
            corrections_saved: correctionsSaved,
            base_job_id: coachingLineage?.jobId,
            base_name: coachingLineage?.jobName,
            training_repo_id: coachingLineage?.trainingDatasetRepoId,
          },
        },
      });
    }
    onExit();
  }, [
    coachMode,
    correctionsSaved,
    coachDataset,
    datasetDeleted,
    coachingLineage,
    navigate,
    onExit,
  ]);

  const handleMergeAndFinetune = useCallback(() => {
    if (!canHandOff || !coachingLineage) return;
    // Named after the TRAINING dataset, not the corrections. The merged result
    // is the next training set — demos plus the rescues — so it belongs in that
    // lineage, and inheriting the corrections' `rollout_` prefix would claim it
    // came straight off a deployment when it did not. Still editable in the
    // dialog.
    const base =
      coachingLineage.trainingDatasetRepoId!.split("/").pop() ?? "training";
    onExit();
    openStudio("collect", {
      merge: {
        sources: [coachingLineage.trainingDatasetRepoId!, coachDataset!],
        suggestedOutput: `${base}_coached`,
        finetuneBaseJobId: coachingLineage.jobId,
        finetuneBaseName: coachingLineage.jobName,
      },
    });
  }, [canHandOff, coachingLineage, coachDataset, onExit, openStudio]);

  // RaC: "the arm is back somewhere sane — the correction starts here."
  //
  // An intervention is two things wearing one name: first the operator rewinds
  // the arm to a state the policy has actually seen, then they demonstrate what
  // should follow. lerobot's HIL guide names RaC (arXiv:2509.07953) as the
  // protocol its DAgger strategy follows, and RaC's whole data-efficiency claim
  // rests on that decomposition — but the strategy records both halves as one
  // undifferentiated intervention, and nobody can recover the boundary from a
  // finished episode afterwards. So it is marked live, or not at all.
  //
  // Optional. Never pressing it records the correction as unlabelled, which the
  // sidecar keeps distinct from "recovery took zero frames".
  const handleCoachRecovered = useCallback(() => {
    sendCoachCommand(
      "recovered",
      t("inference.coach.cmd.recovered"),
      t("inference.coach.cmd.marking"),
    );
  }, [sendCoachCommand, t]);

  const handleCoachReset = useCallback(() => {
    sendCoachCommand("reset", "Task done", "Returning home…");
  }, [sendCoachCommand]);

  // The correction that can still be un-recorded, or null when there is none.
  //
  // The runner holds the most recent correction in memory rather than writing
  // it the instant the operator hands back, and only commits it when the next
  // takeover begins (or the session ends). That deferral is what makes a real
  // delete possible at all: once `save_episode` has run, the frames are spread
  // across shared parquet chunks and a concatenated video file, and lerobot has
  // no supported way to take one back out of an open dataset. So this is not a
  // delete-after-the-fact — it is the session declining to write it.
  //
  // Consequently the window is narrow and the backend, not the browser, decides
  // when it is open. Null the moment the runner commits.
  const droppableCorrection =
    status?.droppable_correction ?? null;

  const handleCoachDropLast = useCallback(() => {
    sendCoachCommand(
      "drop_last",
      t("inference.coach.cmd.dropLast"),
      t("inference.coach.cmd.dropping"),
    );
  }, [sendCoachCommand, t]);

  // ENTER is the whole between-attempts loop, pressed twice per iteration:
  //
  //   Enter (task finished)  -> arm eases home and goes limp   [RESET]
  //   ...operator rearranges the scene by hand, no time limit...
  //   Enter (scene is set)   -> the next attempt starts        [WATCHING]
  //
  // and round again until `corrections_target` corrections are recorded, at
  // which point the runner ends the session on its own.
  //
  // Deliberately ONE key for both halves. The operator is looking at the robot
  // with both hands on it, not at the screen — asking them to remember that
  // finishing is one key and continuing is a different one is how you get the
  // wrong key pressed mid-scene. The phase decides which half Enter means, and
  // the banner always names the next press.
  const handleCoachAdvance = useCallback(() => {
    if (awaitingAttempt) {
      sendCoachCommand("resume", "Next attempt", "Starting…");
    } else {
      handleCoachReset();
    }
  }, [awaitingAttempt, sendCoachCommand, handleCoachReset]);

  // Delete the whole corrections dataset from the summary. Reuses the same
  // /delete-dataset the library uses, which refuses while anything is reading
  // or writing the directory — by this point the session has ended and the
  // runner has finalized, so that guard should pass.
  // Nothing was recorded, so there is nothing to throw away. Kept as its own
  // named flag because it gates BOTH the button and the handler below — the
  // disabled attribute stops the click, and the handler refuses anyway rather
  // than trusting the UI to be the only caller.
  const nothingToDelete = correctionsSaved === 0;

  const handleDeleteCoachDataset = useCallback(async () => {
    const repoId = status?.coaching_dataset;
    if (!repoId) return;
    // Belt and braces with the button's `disabled`. A session that saved
    // nothing still HAS a dataset directory, so `repoId` above is truthy and
    // would have carried a delete through to the server on an empty session.
    if (nothingToDelete) return;
    if (!confirmDelete) {
      setConfirmDelete(true);
      return;
    }
    setDeletingDataset(true);
    try {
      const res = await deleteDataset(baseUrl, fetchWithHeaders, repoId);
      if (res.success === false) {
        throw new Error(res.message ?? t("inference.coach.deleteRefused"));
      }
      setDatasetDeleted(true);
      toast({
        title: t("inference.coach.deletedToast.title"),
        description: t("inference.coach.deletedToast.body", { dataset: repoId }),
      });
      // Deleting the corrections is a WAY OUT of this dialog, not a state to
      // sit in. Everything the summary screen still offers past this point —
      // the merge-and-fine-tune handoff, the close button that carries the
      // handoff with it — refers to a dataset that no longer exists, so
      // leaving the operator parked on it means the next thing they click is
      // one of two buttons that cannot work.
      //
      // `onExit`, deliberately NOT `exitWithHandoff`. The handoff's whole
      // payload is the corrections dataset; carrying it to the Launchpad after
      // deleting it would offer a merge against a directory that is gone.
      // `exitWithHandoff` does guard on `datasetDeleted`, but that state was
      // set one line ago and this closure still sees the old value.
      onExit();
    } catch (e) {
      setConfirmDelete(false);
      toast({
        title: t("inference.coach.deleteFailed"),
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setDeletingDataset(false);
    }
  }, [
    status?.coaching_dataset,
    confirmDelete,
    nothingToDelete,
    baseUrl,
    fetchWithHeaders,
    toast,
    t,
    onExit,
  ]);

  useEffect(() => {
    // NOTE: deliberately NOT gated on `coachBusy`. Unmounting the listener
    // while a command was in flight meant a press in that window was not even
    // seen — no preventDefault, so a bare space scrolled the dialog instead —
    // and the operator got silence from both their press and their retry.
    // Mounted for the whole coaching session INCLUDING startup, not just once
    // the policy is driving. During the 10-30s load the only button on screen
    // is Stop, and Radix autofocuses it — so an operator who has been taught
    // that "space is the whole interaction" presses space at the arm and
    // aborts the run before it begins. The commands themselves stay gated on
    // `coachLive` below; what matters here is that the key is swallowed.
    if (!coachMode) return;
    const onKey = (e: KeyboardEvent) => {
      // The decision is a pure function (coachKeys.ts) so it can be tested:
      // this logic guards a physical arm, and none of it was reachable by a
      // test while it lived in a closure over component state.
      const { preventDefault, stopPropagation, action } = decideCoachKey(e, {
        coachLive,
        coachBusy,
        // Leave the browser alone when the keystroke is aimed at a control it
        // will activate itself. Without this, Space and Enter were swallowed
        // for every <button> in the dialog and NONE of them — including
        // "Discard this correction" and "End session" — could be operated by
        // keyboard. The dialog focuses its own container rather than a button
        // on open (see DialogContent's onOpenAutoFocus), so the hands-on
        // operator's keystrokes still land here, not on a control.
        targetHandlesKey: targetHandlesKey(e.target),
        controlToggleAllowed,
        // Shift+Space is inert unless the policy is actually driving. See
        // handleCoachHold: that is the only phase in which a freeze is a thing
        // no other key already does.
        policyIsDriving: coachPhase === "autonomous",
        // Which correction Backspace throws away. Only while `correcting` is
        // there one in flight to cancel; outside it a cancel would not no-op,
        // it would end the attempt and move the arm, so the key un-records the
        // previous correction instead. See the Backspace branch in coachKeys.
        correcting: coachPhase === "correcting",
        // Straight from the backend, never inferred from the phase — the drop
        // window runs from the hand-back until the next takeover and the
        // browser is explicitly told not to guess at those edges.
        hasDroppableCorrection: droppableCorrection != null,
      });
      if (preventDefault) e.preventDefault();
      if (stopPropagation) e.stopPropagation();
      switch (action) {
        case "takeover-toggle":
          handleCoachToggle();
          break;
        case "hold":
          handleCoachHold();
          break;
        case "recovered":
          handleCoachRecovered();
          break;
        case "advance":
          handleCoachAdvance();
          break;
        case "discard":
          handleCoachDiscard();
          break;
        case "drop-last":
          handleCoachDropLast();
          break;
        case "nothing-to-drop-hint":
          // The press was real and the intent was clear, so answer it. Only the
          // most recent correction is still in memory; everything before it is
          // already in the dataset's parquet chunks and video files, which
          // lerobot has no supported way to take an episode back out of. Saying
          // nothing here would read as a dropped keystroke and get the key
          // pressed again, harder.
          if (!nothingToDropHintShownRef.current) {
            nothingToDropHintShownRef.current = true;
            toast({
              title: t("inference.coach.nothingToDropHint.title"),
              description: t("inference.coach.nothingToDropHint.body"),
              duration: 8000,
            });
          }
          break;
        case "escape-hint":
          // Explains itself once per session, then stays quiet. Escape used to
          // discard, and an operator carrying that muscle memory has to be told
          // it moved rather than silently discovering it did nothing.
          if (!escapeHintShownRef.current) {
            escapeHintShownRef.current = true;
            toast({
              title: "Escape no longer discards",
              description:
                "Backspace or Delete discards the correction in progress, matching the recording dialog. Escape is held back so it can't close the studio mid-session.",
              duration: 8000,
            });
          }
          break;
        default:
          break;
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [
    coachMode,
    coachLive,
    coachBusy,
    coachPhase,
    controlToggleAllowed,
    handleCoachToggle,
    handleCoachHold,
    handleCoachDiscard,
    handleCoachDropLast,
    handleCoachAdvance,
    handleCoachRecovered,
    droppableCorrection,
    toast,
    t,
  ]);

  // Dismissal is blocked while the run is (or may still be) live — before the
  // first status lands we treat the session as live, since the launcher just
  // started it. A reset between episodes counts as live: the session still owns
  // the arm and the cameras.
  const live = status == null || status.inference_active === true;

  const setupElapsed = status?.elapsed_s ?? 0;
  const rolloutElapsed = status?.rollout_elapsed_s ?? 0;
  const duration = status?.duration_s ?? 0;
  // --- Multi-episode evaluation --------------------------------------------
  // `eval_mode` is the single flag to branch on: a plain run reports it false
  // with null companions, so everything below collapses to the old behaviour.
  const evalMode = status?.eval_mode === true;
  const episodesTotal = status?.episodes_total ?? null;
  const episodeIndex = status?.episode_index ?? null;
  const results = (status?.episode_results ?? []) as EpisodeResult[];
  const counts = tally(results);
  const accuracy = status?.accuracy ?? null;
  // Parked between episodes, waiting for the user to rearrange the scene.
  const isResetting = evalMode && status?.phase === "resetting";
  const evalFinished = evalMode && status?.phase === "finished";
  const evalAborted = evalMode && status?.phase === "aborted";
  const isEvalDone = evalFinished || evalAborted;
  // A crashed episode parks in the reset phase carrying its error — the reset
  // screen doubles as "this one broke, continue or abort?".
  const episodeCrashed = isResetting && !!status?.error;

  const isSettingUp =
    status != null &&
    status.inference_active &&
    !isResetting &&
    status.rollout_started_at == null;
  const isRunning =
    status != null &&
    status.inference_active &&
    status.rollout_started_at != null;

  // A finished run we're staying on to surface (see the tick): a real failure
  // (red) or a cleanup-only warning (amber). `ran_with_warning` must NOT read
  // as the red failed state — the run actually worked, only teardown was noisy.
  const isFinished = status?.exited === true && !status?.inference_active;
  const outcome = status?.outcome ?? null;
  const finishedWarn = isFinished && outcome === "ran_with_warning";
  const finishedFailed = isFinished && outcome === "failed";
  const showOutcome = finishedWarn || finishedFailed;
  // What to put in the log panel. `logs` is only THIS run's output when the
  // backend says the log belongs to the active session; a finished run's own log
  // is equally fine to show once the session has ended. Anything else means this
  // run has produced no output, and printing the text anyway is how a previous
  // run's log gets read as the current one — a live incident, where a failed run
  // showed a three-day-old run's output and the user concluded the wrong policy
  // had executed.
  const logIsThisRun =
    logOwner === "active" ||
    (logOwner === "last_run" && !status?.inference_active);
  const logPlaceholder = finishedFailed
    ? t("inference.log.failedPlaceholder")
    : t("inference.log.emptyPlaceholder");
  // The live timer/progress block is replaced by the reset screen between
  // episodes and by the summary once an evaluation ends. A coaching session
  // hides it too: it runs unbounded on purpose, so a countdown against a
  // duration it isn't measuring would be noise at best and a false deadline at
  // worst. The correction tally is the progress indicator that means something.
  const showTimer = !isFinished && !isResetting && !coachMode;
  const coachDone = coachMode && isFinished;

  // When setting up: progress is uncertain — show a soft pulsing bar.
  // When rolling out: progress is rolloutElapsed / duration.
  const pct =
    isRunning && duration > 0
      ? Math.min(100, (rolloutElapsed / duration) * 100)
      : 0;
  const pillTone: "amber" | "green" | "red" = finishedFailed
    ? "red"
    : finishedWarn
      ? "amber"
      : evalAborted
        ? "amber"
        : evalFinished
          ? "green"
          : isResetting
            ? "amber"
            : isSettingUp
              ? "amber"
              : "green";
  const pillLabel = finishedFailed
    ? t("inference.pill.failed")
    : finishedWarn
    ? t("inference.pill.ranWithWarning")
    : coachDone
    ? status?.phase === "aborted"
      ? t("inference.pill.coachingStopped")
      : t("inference.pill.coachingComplete")
    : coachLive
    ? t("inference.pill.coaching")
    : evalAborted
    ? t("inference.pill.aborted")
    : evalFinished
    ? t("inference.pill.evaluationComplete")
    : isResetting
    ? t("inference.pill.resetTheScene")
    : isSettingUp
    ? t("inference.pill.settingUp")
    : isRunning
    ? t("inference.pill.running")
    : t("inference.pill.finished");
  const timerSeconds = isRunning ? rolloutElapsed : setupElapsed;

  // Granular startup phase (from the same status poll). Suppressed once we're
  // showing the terminal outcome banner, which carries its own tone + label.
  // Null before any session has seeded a phase, or for an unrecognised value —
  // then we show nothing and let the timer/pill carry the state.
  const phaseMeta =
    !showOutcome && status?.phase ? (PHASE_META[status.phase] ?? null) : null;

  // Hub model download: show a real byte-progress bar during the
  // downloading_model phase. Indeterminate (pulsing) until the total is known —
  // the total can grow as file sizes are discovered, so the bar may legitimately
  // step backwards. Mirrors the sibling branch's DownloadProgressBar shape.
  const isDownloading = !showOutcome && status?.phase === "downloading_model";
  const dlDone = status?.download_bytes_done ?? null;
  const dlTotal = status?.download_bytes_total ?? null;
  const dlPercent = status?.download_percent ?? null;
  const dlDeterminate = dlPercent != null && dlTotal != null;

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        // Closing a finished coaching session leaves the merge + fine-tune
        // offer on the Launchpad rather than letting it die with the dialog.
        if (!open && !live) exitWithHandoff();
      }}
    >
      <DialogContent
        hideClose
        // Focus the dialog itself on open, not the first button in it.
        //
        // Radix's default lands focus on a control — during startup that is
        // Stop. Two things then went wrong at once: the coaching keydown
        // handler had to swallow Space unconditionally to stop an operator's
        // reflex press from aborting the run, and swallowing it that broadly
        // left every button in the dialog keyboard-inoperable. Focusing the
        // container decouples them — the hands-on operator's keys reach the
        // handler (the container is not a control), and a keyboard operator
        // who tabs to a button gets normal Space/Enter activation.
        onOpenAutoFocus={(e) => {
          e.preventDefault();
          (e.currentTarget as HTMLElement | null)?.focus();
        }}
        onEscapeKeyDown={(e) => {
          if (live) e.preventDefault();
        }}
        onPointerDownOutside={(e) => {
          if (live) e.preventDefault();
        }}
        onInteractOutside={(e) => {
          if (live) e.preventDefault();
        }}
        // w-max, not w-fit: with left-1/2 positioning, fit-content shrink-wraps
        // into the half-viewport left by the offset; max-content sizes to the
        // log's longest line and the 95vw clamp does the capping.
        className="max-h-[92vh] w-max min-w-[min(36rem,95vw)] max-w-[95vw] gap-0 overflow-y-auto p-6"
        aria-describedby={undefined}
      >
        <DialogTitle className="sr-only">{t("inference.dialogTitle")}</DialogTitle>

        {!status ? (
          <div className="flex items-center justify-center py-20 text-muted-foreground">
            <Loader2 className="w-6 h-6 animate-spin mr-3" /> Connecting to
            inference…
          </div>
        ) : (
          // min-w-0 keeps the grid item from inheriting the log's unwrapped
          // line width — overflow scrolls inside the log panel, not the dialog.
          <div className="min-w-0">
            <div className="text-center mb-6">
              <div
                className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold tracking-widest ${PILL_BG[pillTone]}`}
              >
                <span
                  className={`w-2 h-2 rounded-full ${PHASE_DOT[pillTone]} ${
                    isFinished ? "" : "animate-pulse"
                  }`}
                />
                {pillLabel}
              </div>
            </div>

            {/* Coaching banner — who is holding the arm, right now. The single
                most important thing on this screen, and sized accordingly: the
                operator reads it peripherally while looking at the robot. */}
            {coachLive &&
              (() => {
                // No phase yet ⇒ say so. Never fall back to a control-state
                // claim; "the arm is frozen" shown while the policy is starting
                // is the worst default this banner could have.
                // A takeover is two jobs, and the banner names whichever one
                // the operator is on. Before the boundary is marked the
                // instruction is "get the arm back somewhere the policy has
                // seen"; after it, "now show it the right thing". Same phase,
                // same recording, different task — and per RaC the operator
                // working in those terms IS the mechanism, which they will not
                // do while the screen calls the whole thing "you're driving".
                const banner = !coachPhase
                  ? COACH_BANNER_STARTING
                  : awaitingAttempt && coachPhase === "paused"
                    ? // An unknown outcome (older runner) takes the cautious
                      // branch: never promise a limp arm we cannot confirm.
                      resetHomed === false
                      ? COACH_BANNER_PARKED_STUCK
                      : resetLimp === false || resetLimp == null
                        ? COACH_BANNER_PARKED_RIGID
                        : COACH_BANNER_PARKED
                    : coachPhase === "correcting" && recoveryMarkedAt != null
                      ? COACH_BANNER_CORRECTING
                      : COACH_BANNER[coachPhase];
                // A coaching session is the same four beats over and over, so
                // the banner shows ALL FOUR with the current one lit rather than
                // naming the state and leaving the operator to remember what
                // follows it. A phase name says where you are; a position in a
                // fixed cycle says that AND what comes next, which is the
                // question someone with both hands on a leader arm actually has.
                //
                // `poised` gets a HALF-lit DRIVE rather than a fifth stop. It is
                // genuinely half of that step — the arms are lined up and held,
                // and the second press is what commits to it — so a partial fill
                // says "you are here but have not started" without costing a
                // fifth label's worth of small text on a screen read from across
                // a bench. Four stops stay scannable; five would not.
                const poised = coachPhase === "poised";
                return (
                  <div
                    className={`mb-4 rounded-lg border-2 p-5 ${banner.bg} ${banner.accent} ${
                      coachPhase === "correcting" ? "animate-pulse" : ""
                    }`}
                  >
                    <div className="grid grid-cols-4 gap-1.5">
                      {COACH_STEPS.map((label, i) => {
                        const step = i + 1;
                        const lit = banner.step === step;
                        return (
                          <div
                            key={label}
                            className={`flex flex-col gap-1.5 ${lit ? "" : "opacity-30"}`}
                          >
                            <div className="h-1.5 overflow-hidden rounded-full bg-current/25">
                              {/* Half-fill for the poised hold; a full bar
                                  otherwise. The lit stop is the only one that
                                  ever paints, so an unlit stop's inner bar is
                                  deliberately zero-width rather than hidden. */}
                              <div
                                className="h-full rounded-full bg-current transition-[width] duration-300"
                                style={{ width: lit ? (poised ? "50%" : "100%") : "0%" }}
                              />
                            </div>
                            <div className="font-mono text-[10px] tracking-[0.12em]">
                              {label}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    <div className="mt-4 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                      <span className="text-4xl font-bold leading-none tracking-tight">
                        {t(banner.titleKey as never)}
                      </span>
                      {banner.badgeKey && (
                        <span className="rounded-full border border-current px-2 py-0.5 font-mono text-[10px] tracking-[0.12em]">
                          {t(banner.badgeKey as never)}
                        </span>
                      )}
                    </div>
                    <p className="mt-2 text-sm leading-relaxed opacity-90">
                      {pendingAction ? (
                        <span className="inline-flex items-center gap-2 font-semibold">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          {pendingAction}
                        </span>
                      ) : (
                        t(banner.hintKey as never)
                      )}
                    </p>
                  </div>
                );
              })()}

            {/* A refused takeover. Not an error state — the session is fine and
                the operator just has to move the leaders nearer before trying
                again — so it sits inline rather than taking over the dialog. */}
            {/* A correction the operator did NOT ask to lose. Its own slot, and
                its own colour: a refused takeover is "try again", this is
                "your work is gone and here is how to avoid it next time". */}
            {coachLive && discardNotice && (
              <div className="mb-4 rounded-lg border border-warn/40 bg-warn/10 p-3">
                <p className="text-sm leading-relaxed text-warn">
                  {discardNotice}
                </p>
              </div>
            )}

            {coachLive && alignError && (
              <div className="mb-4 rounded-lg border border-warn/40 bg-warn/10 p-3">
                <p className="text-sm leading-relaxed text-warn">
                  {alignError}
                </p>
              </div>
            )}

            {/* Coaching tally. Rendered on the live screen AND the summary, so
                the count is never more than a glance away — same reasoning as
                the eval header below. */}
            {coachMode && (
              <div className="mb-6 rounded-lg border border-border bg-muted/30 p-4">
                <div className="flex items-baseline justify-between gap-4">
                  <span className="text-sm font-semibold">
                    {t("inference.coach.tally", {
                      saved: correctionsSaved,
                      target: correctionsTarget || "?",
                    })}
                  </span>
                  {/* Attempts deliberately NOT shown. It sat next to the
                      corrections count and read as a second progress number,
                      so "attempt 3" was taken to mean three corrections were
                      recorded when none had been. The only count that matters
                      here is the one the session ends on. */}
                  <span className="text-xs text-muted-foreground tabular-nums">
                    {t("inference.coach.recorded", {
                      duration: formatTime(correctionSeconds),
                    })}
                  </span>
                </div>
                {/* How many corrections carry a recovery boundary. Shown live
                    rather than only in the summary because the habit is worth
                    forming DURING the session — by the end it is too late to
                    mark the ones that went unmarked, and nobody can recover the
                    boundary from a finished episode. */}
                {correctionsSaved > 0 && (
                  <div className="mt-1 text-xs text-muted-foreground">
                    {correctionsLabelled} of {correctionsSaved} split into
                    recovery + correction
                    {correctionsLabelled < correctionsSaved && (
                      <span className="opacity-75">
                        {" "}
                        — press <span className="font-mono">g</span>{" "}
                        mid-takeover to mark where the rescue ends
                      </span>
                    )}
                  </div>
                )}
                <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full bg-ok transition-[width] duration-500"
                    style={{
                      width: `${
                        correctionsTarget > 0
                          ? Math.min(
                              100,
                              (correctionsSaved / correctionsTarget) * 100,
                            )
                          : 0
                      }%`,
                    }}
                  />
                </div>
                {coachDataset && (
                  <p className="mt-3 break-all text-xs text-muted-foreground">
                    {t("inference.coach.savingTo", { dataset: coachDataset })}
                  </p>
                )}
              </div>
            )}

            {/* Coaching summary — what to do with what you just collected. */}
            {coachDone && (
              <div
                className={`mb-6 rounded-lg border p-4 ${
                  correctionsSaved > 0
                    ? "border-ok/40 bg-ok/10"
                    : "border-warn/40 bg-warn/10"
                }`}
              >
                {correctionsSaved > 0 ? (
                  <>
                    <p className="text-sm leading-relaxed text-ok">
                      {/* Two whole sentences rather than a splice: the plural
                          is on the correction COUNT, and whether a dataset name
                          is known changes the wording, not just a fragment.
                          <0> emphasises the dataset name — data, not prose. */}
                      {coachDataset ? (
                        <Trans
                          i18nKey="inference.coach.summarySavedTo"
                          count={correctionsSaved}
                          values={{ dataset: coachDataset }}
                          components={[<strong key="0" className="break-all" />]}
                        />
                      ) : (
                        t("inference.coach.summarySaved", {
                          count: correctionsSaved,
                        })
                      )}
                    </p>
                    {/* The handoff. Corrections are worth nothing on their
                        own — they have to be merged with what the checkpoint
                        was last trained on and the checkpoint fine-tuned on the
                        result, and TrainPanel takes exactly one dataset, so the
                        merge is mandatory rather than an optimisation. This
                        used to be a paragraph of prose with no buttons, which
                        put the entire payoff of the feature behind a manual
                        chore the UI didn't help with. */}
                    {canHandOff ? (
                      <>
                        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                          {/* <0> is the dataset name — data, emphasised. */}
                          <Trans
                            i18nKey="inference.coach.handoffNext"
                            values={{
                              dataset:
                                coachingLineage?.trainingDatasetRepoId ?? "",
                            }}
                            components={[<strong key="0" className="break-all" />]}
                          />
                        </p>
                        <Button
                          onClick={handleMergeAndFinetune}
                          className="mt-3 w-full font-semibold"
                        >
                          <GitMerge className="mr-2 h-4 w-4" />
                          {t("inference.coach.handoffAction")}
                        </Button>
                        <p className="mt-2 text-xs text-muted-foreground">
                          {t("inference.coach.handoffHint")}
                        </p>
                      </>
                    ) : (
                      /* No lineage: either this wasn't launched from a skill in
                         this browser session, or the page was reloaded and the
                         handoff state went with it. The corrections are on disk
                         and mergeable by hand, so say how rather than offering
                         a button that can't know both halves. */
                      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                        {/* <0> emphasises "last" — the whole point of the
                            sentence, and the mistake it exists to prevent. */}
                        <Trans
                          i18nKey="inference.coach.summaryNextSteps"
                          components={[<em key="0" />]}
                        />
                      </p>
                    )}
                  </>
                ) : (
                  <p className="text-sm leading-relaxed text-warn">
                    {t("inference.coach.summaryNone")}
                  </p>
                )}
              </div>
            )}

            {/* Evaluation header — which episode we're on, and the tally so
                far. Rendered on every eval screen (running, reset, summary) so
                the score is never more than a glance away. */}
            {evalMode && (
              <div className="mb-6 rounded-lg border border-border bg-muted/30 p-4">
                <div className="flex items-baseline justify-between gap-4">
                  <span className="text-sm font-semibold">
                    {isEvalDone
                      ? t("inference.eval.episodesTotal", {
                          count: episodesTotal ?? results.length,
                        })
                      : t("inference.eval.episodeProgress", {
                          index: episodeIndex ?? 1,
                          total:
                            episodesTotal ?? t("inference.eval.unknownTotal"),
                        })}
                  </span>
                  <span className="text-xs text-muted-foreground tabular-nums">
                    {t("inference.eval.done", { count: results.length })}
                  </span>
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-sm tabular-nums">
                  {(["success", "failure", "error"] as EpisodeResult[]).map(
                    (key) => (
                      <span key={key} className="flex items-center gap-1.5">
                        <span
                          className={`h-2 w-2 rounded-full ${RESULT_META[key].dot}`}
                        />
                        <span className={RESULT_META[key].text}>
                          {t(RESULT_META[key].labelKey as never)}
                        </span>
                        <span className="font-semibold">{counts[key]}</span>
                      </span>
                    ),
                  )}
                </div>
                {counts.error > 0 && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    {t("inference.eval.errorsExcluded")}
                  </p>
                )}
              </div>
            )}

            {/* Evaluation summary — the point of the whole run. */}
            {isEvalDone && (
              <div
                className={`mb-6 rounded-lg border p-4 ${
                  evalAborted
                    ? "border-warn/40 bg-warn/10"
                    : "border-ok/40 bg-ok/10"
                }`}
              >
                {evalAborted ? (
                  <p className="text-sm leading-relaxed text-warn">
                    {t("inference.eval.abortedSummary", {
                      done: results.length,
                      total: episodesTotal ?? t("inference.eval.unknownTotal"),
                    })}
                  </p>
                ) : accuracy != null ? (
                  <div className="text-center">
                    <div className="text-5xl font-mono font-bold leading-none text-ok">
                      {Math.round(accuracy * 100)}%
                    </div>
                    <div className="mt-2 text-sm text-muted-foreground tabular-nums">
                      {t("inference.eval.succeeded", {
                        success: counts.success,
                        scored: counts.success + counts.failure,
                      })}
                      {counts.error > 0
                        ? t("inference.eval.excludedAsErrors", {
                            count: counts.error,
                          })
                        : ""}
                    </div>
                  </div>
                ) : (
                  <p className="text-sm leading-relaxed text-warn">
                    {t("inference.eval.noScoreable")}
                  </p>
                )}
                {results.length > 0 && (
                  <ol className="mt-4 space-y-1 text-xs tabular-nums">
                    {results.map((r, i) => (
                      <li
                        key={i}
                        className="flex items-center gap-2 text-muted-foreground"
                      >
                        <span className="w-10 shrink-0">#{i + 1}</span>
                        <span
                          className={`h-1.5 w-1.5 rounded-full ${RESULT_META[r].dot}`}
                        />
                        <span className={RESULT_META[r].text}>
                          {t(RESULT_META[r].labelKey as never)}
                        </span>
                      </li>
                    ))}
                  </ol>
                )}
              </div>
            )}

            {/* Reset between episodes — user-ended, no timer. */}
            {isResetting && (
              <div
                className={`mb-6 rounded-lg border p-4 ${
                  episodeCrashed
                    ? "border-destructive/40 bg-destructive/10"
                    : "border-warn/40 bg-warn/10"
                }`}
              >
                {episodeCrashed ? (
                  <>
                    <div className="flex items-center gap-2 text-sm font-semibold text-destructive">
                      <span className="h-2 w-2 rounded-full bg-destructive" />
                      {t("inference.eval.episodeCrashed", {
                        index: results.length,
                      })}
                    </div>
                    <p className="mt-2 text-sm leading-relaxed text-destructive/90">
                      {t("inference.eval.episodeCrashedBody")}
                    </p>
                    {status.hint && (
                      <p className="mt-2 text-sm leading-relaxed text-destructive/90">
                        {status.hint}
                      </p>
                    )}
                    {status.error && (
                      <pre className="mt-3 max-h-40 overflow-auto rounded bg-muted p-2 text-xs text-muted-foreground whitespace-pre-wrap break-words">
                        {status.error}
                      </pre>
                    )}
                  </>
                ) : (
                  <p className="text-sm leading-relaxed text-warn">
                    <Trans
                      i18nKey="inference.eval.episodeRecorded"
                      values={{
                        index: results.length,
                        result: t(
                          RESULT_META[results.at(-1) ?? "failure"]
                            .labelKey as never,
                        ),
                      }}
                      components={[<span key="0" />, <strong key="1" />]}
                    />
                  </p>
                )}
              </div>
            )}

            {showTimer && (
              <>
                <div className="text-center mb-4">
                  <div
                    className={`text-7xl font-mono font-bold leading-none ${
                      isSettingUp ? "text-warn" : "text-ok"
                    }`}
                  >
                    {formatTime(timerSeconds)}
                  </div>
                  <div className="text-sm text-muted-foreground mt-2">
                    {isSettingUp
                      ? t("inference.settingUp")
                      : `/ ${formatTime(duration)}`}
                  </div>
                </div>

                <div className="w-full bg-muted rounded-full h-1.5 mb-8">
                  <div
                    className={`h-1.5 rounded-full transition-all duration-500 ${
                      isSettingUp ? "bg-warn/40 animate-pulse w-full" : "bg-ok"
                    }`}
                    style={isSettingUp ? undefined : { width: `${pct}%` }}
                  />
                </div>
              </>
            )}

            <div className="text-xs text-muted-foreground break-all mb-6">
              {t("inference.policyRef", {
                ref: status.policy_ref ?? t("inference.unknownPolicy"),
              })}
            </div>

            {showOutcome && (
              <div
                className={`mb-6 rounded-lg border p-4 ${
                  finishedWarn
                    ? "border-warn/40 bg-warn/10"
                    : "border-destructive/40 bg-destructive/10"
                }`}
              >
                <div
                  className={`flex items-center gap-2 text-sm font-semibold ${
                    finishedWarn ? "text-warn" : "text-destructive"
                  }`}
                >
                  <span
                    className={`w-2 h-2 rounded-full ${
                      finishedWarn ? "bg-warn" : "bg-destructive"
                    }`}
                  />
                  {finishedWarn
                    ? t("inference.outcome.ranWithWarning")
                    : t("inference.outcome.runFailed")}
                </div>
                {status.hint && (
                  <p
                    className={`mt-2 text-sm leading-relaxed ${
                      finishedWarn ? "text-warn/90" : "text-destructive/90"
                    }`}
                  >
                    {status.hint}
                  </p>
                )}
                {status.error && (
                  <pre className="mt-3 max-h-40 overflow-auto rounded bg-muted p-2 text-xs text-muted-foreground whitespace-pre-wrap break-words">
                    {status.error}
                  </pre>
                )}
              </div>
            )}

            {isFinished ? (
              <div className="space-y-2">
                {/* The moment the operator KNOWS the policy is imperfect is the
                    moment they watch it finish — and until now every one of
                    those moments dead-ended in a Close button. Coaching stops
                    being a mode you have to already know about and becomes the
                    obvious next move after a bad result. Not offered on a
                    coaching run's own summary, which has its own follow-ups. */}
                {!coachMode && deployPrefill && (
                  <Button
                    onClick={() => {
                      // No markHandled(): the exit guard it notified was
                      // retired with the move to a server-side lease.
                      onExit();
                      openStudio("deploy", {
                        deploy: { ...deployPrefill, mode: "coach" },
                      });
                    }}
                    className="w-full font-semibold py-6 text-lg"
                  >
                    <GraduationCap className="w-5 h-5 mr-2" />
                    {evalMode && accuracy != null && accuracy < 1
                      ? t("inference.coach.offerWithGap", {
                          percent: Math.round((1 - accuracy) * 100),
                        })
                      : t("inference.coach.offer")}
                  </Button>
                )}
                {/* exitWithHandoff, NOT onExit. It is the only caller that
                    navigates with the `coached` payload, which is what carries
                    the merge + fine-tune offer to the Launchpad. Wired into
                    onOpenChange but not here, this button — the one the UI
                    invites you to press — silently dropped the handoff, while
                    Escape and outside-click kept it. It fired on the most
                    common path: the operator who declines the inline merge and
                    then closes. exitWithHandoff calls onExit unconditionally,
                    so nothing else changes. */}
                <Button
                  onClick={exitWithHandoff}
                  variant={
                    !coachMode && deployPrefill ? "outline" : "secondary"
                  }
                  className="w-full font-semibold py-6 text-lg"
                >
                  {t("inference.button.close")}
                </Button>
                {/* Throwing the corrections away is a first-class outcome, not
                    a failure path: most sessions while shaking the feature out
                    produce data nobody wants, and the alternative is hunting
                    for a timestamped name in the dataset library afterwards.
                    Only offered once a dataset actually exists. */}
                {coachDone && coachDataset && (
                  // Shown-but-dead when the session saved nothing. The dataset
                  // directory is created when the session STARTS, so it exists
                  // (and `coachDataset` is set) even after a session that
                  // recorded zero corrections — which is how a destructive
                  // button ended up live with nothing behind it to destroy.
                  // Disabled rather than hidden: the row vanishing between one
                  // session and the next reads as a bug, and the disabled label
                  // says plainly why there is nothing to do here.
                  <Button
                    onClick={handleDeleteCoachDataset}
                    disabled={
                      deletingDataset || datasetDeleted || nothingToDelete
                    }
                    variant="outline"
                    className="w-full font-semibold text-destructive hover:text-destructive disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    {datasetDeleted
                      ? t("inference.coach.deleted")
                      : deletingDataset
                        ? t("inference.coach.deleting")
                        : nothingToDelete
                          ? t("inference.coach.nothingToDelete")
                          : confirmDelete
                            ? t("inference.coach.deleteConfirm")
                            : t("inference.coach.delete")}
                  </Button>
                )}
              </div>
            ) : coachLive ? (
              // FOUR controls at most, and never six.
              //
              // The old screen stacked take-over, recovered, discard, advance,
              // recover and end-session as six equal-weight full-width buttons.
              // An operator who is looking at the arm and not the screen cannot
              // pick out of six, and the two that end or unwind the session sat
              // in the same visual register as the one they press every few
              // seconds. So: one primary, one that moves the session forward,
              // the correction-only pair that only exist while there IS a
              // correction, and the two "something is wrong" exits demoted to a
              // single quiet row at the bottom.
              //
              // Every button blurs itself on click — see `pressAndBlur`.
              <div className="space-y-2">
                <Button
                  onClick={pressAndBlur(handleCoachToggle)}
                  disabled={coachBusy || !controlToggleAllowed}
                  className="w-full font-semibold py-6 text-lg disabled:opacity-50"
                  variant={
                    coachPhase === "correcting" ? "secondary" : "default"
                  }
                >
                  <Hand className="w-5 h-5 mr-2" />
                  {/* Three labels, one button, because it is one key. While
                      `poised` the operator is NOT asking for control — they
                      already asked, the arms are lined up and holding, and this
                      press confirms their hand is on the leader and starts
                      recording. Labelling it "Take control" there would ask the
                      same question twice and hide the fact that the second
                      press is the one that keeps frames. */}
                  {coachPhase === "correcting"
                    ? t("inference.coach.handBack")
                    : coachPhase === "poised"
                      ? t("inference.coach.confirmHold")
                      : t("inference.coach.takeOver")}
                  {/* Key names are the physical keys — never translated. */}
                  <kbd className="ml-2 rounded bg-foreground/10 px-1.5 py-0.5 text-xs font-mono">
                    space
                  </kbd>
                </Button>
                {/* Shown mid-correction too, now that Enter SAVES the
                    correction in flight before resetting. Hiding it there
                    predated that, and left the operator who finished the task
                    while still driving with no visible way to say so. */}
                <Button
                  onClick={pressAndBlur(handleCoachAdvance)}
                  disabled={coachBusy}
                  variant={awaitingAttempt ? "default" : "outline"}
                  className="w-full font-semibold disabled:opacity-50"
                >
                  {awaitingAttempt ? (
                    <Play className="w-4 h-4 mr-2" />
                  ) : (
                    <CheckCircle2 className="w-4 h-4 mr-2" />
                  )}

                  {awaitingAttempt
                    ? "Scene is set — start the next one"
                    : coachPhase === "correcting"
                      ? "Task done — save this correction and reset"
                      : "Is the task done? Press Enter to say so"}
                  <kbd className="ml-2 rounded bg-foreground/10 px-1.5 py-0.5 text-xs font-mono">
                    enter
                  </kbd>
                </Button>
                {/* Bin the correction from the attempt that just ended.
                    Shown for exactly as long as the backend says it can be
                    honoured, and not one phase less. This was gated on
                    `awaitingAttempt` as well, which is precisely the inference
                    dagger_protocol.py and inferenceApi.ts both forbid: the drop
                    window runs from the hand-back until the next takeover, not
                    from a phase the browser can see. After a plain hand-back
                    the policy resumes, `awaiting_attempt` is false, and the
                    button vanished while the runner was still holding the
                    correction and still honouring the verb — confirmed on
                    hardware, pressed by keyboard in a window with no button.
                    See dagger_protocol's DROP_LAST. */}
                {droppableCorrection != null && (
                  <Button
                    onClick={pressAndBlur(handleCoachDropLast)}
                    disabled={coachBusy}
                    variant="outline"
                    className="w-full font-semibold text-destructive hover:text-destructive disabled:opacity-50"
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    {t("inference.coach.dropLast", {
                      seconds: formatTime(droppableCorrection.seconds),
                    })}
                    {/* The same key the discard button names, because it is the
                        same key: Backspace means "throw away the thing that just
                        happened" in both phases, and the operator has to be able
                        to see that it does. Withheld only while `correcting`,
                        where the key is claimed by the discard button below —
                        two visible ⌫ badges promising different deletions is
                        worse than one. */}
                    {coachPhase !== "correcting" && (
                      <kbd className="ml-2 rounded bg-foreground/10 px-1.5 py-0.5 text-xs font-mono">
                        ⌫
                      </kbd>
                    )}
                  </Button>
                )}
                {/* The correction-only pair. Both are meaningless outside a
                    takeover, so neither is on screen outside one — which is
                    most of what took the live screen from six buttons to
                    three. */}
                {coachPhase === "correcting" && recoveryMarkedAt == null && (
                  <Button
                    onClick={pressAndBlur(handleCoachRecovered)}
                    disabled={coachBusy}
                    variant="outline"
                    className="w-full font-semibold disabled:opacity-50"
                  >
                    <CheckCircle2 className="w-4 h-4 mr-2" />
                    {t("inference.coach.recovered")}
                    <kbd className="ml-2 rounded bg-foreground/10 px-1.5 py-0.5 text-xs font-mono">
                      g
                    </kbd>
                  </Button>
                )}
                {coachPhase === "correcting" && (
                  <Button
                    onClick={pressAndBlur(handleCoachDiscard)}
                    disabled={coachBusy}
                    variant="outline"
                    className="w-full font-semibold disabled:opacity-50"
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    {t("inference.coach.discardAndReset")}
                    <kbd className="ml-2 rounded bg-foreground/10 px-1.5 py-0.5 text-xs font-mono">
                      ⌫
                    </kbd>
                  </Button>
                )}
                {/* The two "something is wrong" exits, together and quiet.
                    They are the rarest presses on the screen and the most
                    expensive to hit by accident, and as full-width buttons they
                    read as ordinary steps in the loop. Recover comes first
                    because it is the one that does NOT cost the whole session. */}
                {/* The one "something is wrong" exit left, and the rarest and
                    most expensive press on the screen — so it is quiet.
                    "Arm stuck? Recover" used to sit beside it doing what
                    Discard now does on its own. */}
                <div className="flex justify-center pt-1">
                  <Button
                    onClick={pressAndBlur(handleStop)}
                    disabled={stopping}
                    variant="ghost"
                    className="text-xs text-muted-foreground disabled:opacity-50"
                  >
                    <Square className="w-3.5 h-3.5 mr-1.5" />
                    {stopping
                      ? t("inference.coach.ending")
                      : t("inference.coach.endSession")}
                  </Button>
                </div>
                {/* The freeze has no button on purpose — it is the one control
                    an operator reaches for without looking, and a seventh
                    button would undo the point of the cull. Named here so the
                    binding is not invisible. */}
                {coachPhase === "autonomous" && (
                  <p className="pt-1 text-center text-[11px] text-muted-foreground">
                    Need the policy to stop without taking over? Press{" "}
                    <span className="font-mono">shift+space</span>.
                  </p>
                )}
              </div>
            ) : isResetting ? (
              // Reset screen: continue is the primary action, abort stays
              // available alongside it.
              <div className="space-y-2">
                <Button
                  onClick={handleNextEpisode}
                  disabled={startingNext}
                  className="w-full font-semibold py-6 text-lg disabled:opacity-50"
                >
                  <Play className="w-5 h-5 mr-2" />
                  {startingNext
                    ? t("inference.button.starting")
                    : t("inference.button.startEpisode", {
                        index: Math.min(
                          results.length + 1,
                          episodesTotal ?? results.length + 1,
                        ),
                      })}
                </Button>
                <Button
                  onClick={handleStop}
                  disabled={stopping}
                  variant="outline"
                  className="w-full font-semibold disabled:opacity-50"
                >
                  <Square className="w-4 h-4 mr-2" />
                  {stopping
                    ? t("inference.button.aborting")
                    : t("inference.button.abortEvaluation")}
                </Button>
              </div>
            ) : evalMode ? (
              // Running an episode: calling it a success is the primary action,
              // and it is NOT the same button as aborting the whole run.
              <div className="space-y-2">
                <Button
                  onClick={handleEpisodeSuccess}
                  disabled={!isRunning || endingEpisode}
                  className="w-full font-semibold py-6 text-lg disabled:opacity-50"
                >
                  <CheckCircle2 className="w-5 h-5 mr-2" />
                  {endingEpisode
                    ? t("inference.button.endingEpisode")
                    : t("inference.button.taskSucceeded")}
                </Button>
                <Button
                  onClick={handleStop}
                  disabled={!status.inference_active || stopping}
                  variant="outline"
                  className="w-full font-semibold disabled:opacity-50"
                >
                  <Square className="w-4 h-4 mr-2" />
                  {stopping
                    ? t("inference.button.aborting")
                    : t("inference.button.abortEvaluation")}
                </Button>
              </div>
            ) : (
              <Button
                onClick={handleStop}
                disabled={!status.inference_active || stopping}
                variant="destructive"
                className="w-full font-semibold py-6 text-lg disabled:opacity-50"
              >
                <Square className="w-5 h-5 mr-2" />
                {stopping
                  ? t("inference.button.stopping")
                  : t("inference.button.stop")}
              </Button>
            )}

            {phaseMeta && (
              <div className="mt-6 flex items-center gap-2 text-sm">
                <span
                  className={`w-2 h-2 rounded-full ${PHASE_DOT[phaseMeta.tone]} ${
                    phaseMeta.pulse ? "animate-pulse" : ""
                  }`}
                />
                <span className={`font-medium ${PHASE_TEXT[phaseMeta.tone]}`}>
                  {t(phaseMeta.labelKey as never)}
                </span>
              </div>
            )}

            {isDownloading && (
              <div className="mt-3 space-y-1">
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                  {dlDeterminate ? (
                    <div
                      className="h-full rounded-full bg-warn transition-[width] duration-500"
                      style={{ width: `${dlPercent}%` }}
                    />
                  ) : (
                    <div className="h-full w-full animate-pulse rounded-full bg-warn/40" />
                  )}
                </div>
                <div className="text-[11px] tabular-nums text-muted-foreground">
                  {dlDeterminate
                    ? t("inference.download.progress", {
                        done: formatBytes(dlDone ?? 0),
                        total: formatBytes(dlTotal),
                      })
                    : dlDone != null
                      ? t("inference.download.soFar", {
                          done: formatBytes(dlDone),
                        })
                      : t("inference.download.starting")}
                </div>
              </div>
            )}

            <div className="mt-4">
              <LogPanel
                logs={logIsThisRun ? logs : logPlaceholder}
                title={t("inference.log.title")}
                defaultCollapsed
                wrap={false}
              />
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default InferenceSessionDialog;
