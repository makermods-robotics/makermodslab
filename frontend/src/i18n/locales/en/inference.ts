export default {
  dialogTitle: "Inference session",
  // Granular startup/run phases from the status poll. The PHASE enum values
  // are backend data; only these labels are display.
  phase: {
    downloadingModel: "Downloading model…",
    starting: "Starting up…",
    loadingPolicy: "Loading policy…",
    connecting: "Connecting to arm…",
    running: "Running",
    stopping: "Stopping…",
    stopped: "Stopped",
    error: "Error — see log",
    resetting: "Reset the scene",
    finished: "Evaluation complete",
    aborted: "Evaluation aborted",
  },
  result: {
    success: "Success",
    failure: "Failure",
    error: "Error",
  },
  // Status pill. Written lower-case here and uppercased by CSS in Latin
  // scripts; the uppercase/tracking classes come off for CJK.
  pill: {
    failed: "Failed",
    ranWithWarning: "Ran with warning",
    aborted: "Aborted",
    evaluationComplete: "Evaluation complete",
    resetTheScene: "Reset the scene",
    settingUp: "Setting up",
    running: "Running",
    finished: "Finished",
  },
  toast: {
    startedWarningTitle: "Started with a warning",
    failedTitle: "Inference failed",
    ranWithWarningTitle: "Ran with a cleanup warning",
    // Last-resort fallback: the backend hint and error text win when present.
    seeLog: "See the inference log for details.",
    evalAbortedTitle: "Evaluation aborted",
    evalCompleteTitle: "Evaluation complete",
    evalAbortedDescription: "Partial results — no accuracy recorded.",
    // {{percent}} arrives pre-rounded — formatting is done by the caller.
    evalAccuracy: "{{percent}}% success rate.",
    evalNoScoreable: "No scoreable episodes.",
    finishedTitle: "Inference finished",
    finishedDescription: "Run completed.",
    hungTitle: "Inference seems hung",
    // {{seconds}} arrives pre-rounded.
    hungDescription: "Rollout past duration by {{seconds}}s.",
    lostConnectionTitle: "Lost connection to backend",
    stopFailedTitle: "Stop failed",
    endEpisodeFailedTitle: "Couldn't end the episode",
    nextEpisodeFailedTitle: "Couldn't start the next episode",
  },
  eval: {
    // {{count}} episodes total, shown once the run is done.
    episodesTotal_one: "{{count}} episode",
    episodesTotal_other: "{{count}} episodes",
    episodeProgress: "Episode {{index}} of {{total}}",
    // Placeholder when the backend has not reported a total yet.
    unknownTotal: "?",
    done: "{{count}} done",
    errorsExcluded:
      "Errored episodes are excluded from the accuracy — a hardware hiccup isn't a policy failure.",
    abortedSummary:
      "Aborted after {{done}} of {{total}} episodes — no accuracy recorded for a partial run.",
    succeeded: "{{success}} / {{scored}} episodes succeeded",
    excludedAsErrors: " ({{count}} excluded as errors)",
    noScoreable:
      "No scoreable episodes — every episode errored, so there's no accuracy to report.",
    episodeCrashed: "Episode {{index}} crashed",
    episodeCrashedBody:
      "It counts as neither a success nor a failure. Continue to run the next episode, or abort the evaluation.",
    // <1> wraps the result label (Success / Failure / Error).
    episodeRecorded:
      "Episode {{index}} recorded as <1>{{result}}</1>. Rearrange the scene, then start the next episode — there's no timer, take as long as you need.",
  },
  settingUp: "Loading policy & connecting hardware…",
  // {{ref}} is a model/policy identifier — data, rendered verbatim.
  policyRef: "policy: {{ref}}",
  unknownPolicy: "(unknown)",
  outcome: {
    ranWithWarning: "Ran with a cleanup warning",
    runFailed: "Run failed",
  },
  button: {
    close: "Close",
    starting: "Starting…",
    startEpisode: "Start episode {{index}}",
    aborting: "Aborting…",
    abortEvaluation: "Abort evaluation",
    endingEpisode: "Ending episode…",
    taskSucceeded: "Task succeeded — end episode",
    stopping: "Stopping…",
    stop: "Stop",
  },
  download: {
    // Both sides arrive pre-formatted from formatBytes().
    progress: "{{done}} / {{total}}",
    soFar: "{{done}} so far",
    starting: "Starting download…",
  },
  log: {
    title: "Inference log",
    failedPlaceholder:
      "This run failed before the rollout process started, so it produced no log — see the error above.",
    emptyPlaceholder:
      "No log yet for this run — it hasn't started producing output.",
  },
} as const;
