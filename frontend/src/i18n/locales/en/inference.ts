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
    // Coaching phases. The only ones in this map worded from the OPERATOR's
    // point of view rather than the system's — they say who is holding the arm.
    watching: "Policy driving — watch for a failure",
    holding: "Held — the arm is holding its pose",
    correcting: "You're driving — recording",
    handingOver: "Handing over — the arm is moving",
    saving: "Saving the correction…",
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
    coaching: "Coaching",
    coachingComplete: "Coaching complete",
    coachingStopped: "Coaching stopped",
  },

  // The big in-session coaching banner. Titles are SHOUTED on purpose: the
  // operator is looking at the arm, not the screen, and reads this peripherally.
  coachBanner: {
    watching: {
      title: "WATCHING",
      hint: "The policy is driving. Press space to take over — the leader arm will move itself to the robot's pose first, so hold it loosely and let it.",
    },
    held: {
      title: "HELD",
      hint: "The arm is holding its pose. Nothing is being recorded.",
    },
    handingOver: {
      title: "HANDING OVER",
      hint: "The arm is moving into position — don't fight it. Wait for it to settle.",
    },
    saving: {
      title: "SAVING…",
      hint: "Writing the correction to disk. The arm is held; the policy resumes when this finishes.",
    },
    correcting: {
      title: "YOU'RE DRIVING",
      hint: "Recovering and correcting — every frame is being recorded.",
    },
    starting: {
      title: "STARTING…",
      hint: "Loading the policy and connecting the arms.",
    },
  },

  coach: {
    // In-session controls. Each button also renders its key, which is never
    // translated — "space" and "esc" are the physical keys.
    takeOver: "Take over",
    handBack: "Hand back to the policy",
    discard: "Discard this correction",
    hold: "Hold — freeze the arm",
    resume: "Let the policy continue",
    ending: "Ending…",
    endSession: "End session & keep corrections",
    // Live tally. {{saved}} and {{target}} are raw counts; {{target}} is "?"
    // until the runner reports one, so this is not a plural form.
    tally: "{{saved}} of {{target}} corrections",
    recorded: "{{duration}} recorded",
    savingTo: "saving to {{dataset}}",
    // Summary. Two forms because a known dataset name changes the sentence,
    // not just a fragment; <0> emphasises the name, which is data.
    summarySaved_one: "{{count}} correction saved.",
    summarySaved_other: "{{count}} corrections saved.",
    summarySavedTo_one: "{{count}} correction saved to <0>{{dataset}}</0>.",
    summarySavedTo_other: "{{count}} corrections saved to <0>{{dataset}}</0>.",
    summaryNextSteps:
      "To turn these into a better policy: merge this dataset with the one this checkpoint was <0>last</0> trained on — if you fine-tuned, that's the fine-tuning dataset, not the original demos — then fine-tune from this same checkpoint on the merged result. Training takes one dataset, so the merge isn't optional. Both steps are in the dataset library and the training panel.",
    summaryNone: "No corrections were saved — nothing to train on from this session.",
    // Discarding the whole dataset from the summary — a first-class outcome,
    // not a failure path.
    delete: "Delete these corrections",
    deleteConfirm: "Really delete? This can't be undone",
    deleting: "Deleting…",
    deleted: "Deleted",
    deleteRefused: "The server refused the delete.",
    deleteFailed: "Couldn't delete the dataset",
    deletedToast: {
      title: "Corrections deleted",
      body: "{{dataset}} was removed from disk.",
    },
    // Command labels. {{action}} in `failed` is one of the labels below, so
    // they read as sentence fragments rather than button text.
    cmd: {
      failed: "{{action}} failed",
      takeOver: "Take over",
      takingOver: "Taking over…",
      handBack: "Hand back",
      handingBack: "Handing back…",
      hold: "Hold",
      holding: "Holding…",
      resume: "Resume policy",
      resuming: "Resuming…",
      discard: "Discard correction",
      discarding: "Discarding…",
    },
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
    coachStoppedTitle: "Coaching session stopped",
    coachCompleteTitle: "Coaching complete",
    // count 0 is a real outcome here — a session can end having saved nothing.
    coachSaved_one: "{{count}} correction saved.",
    coachSaved_other: "{{count}} corrections saved.",
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
