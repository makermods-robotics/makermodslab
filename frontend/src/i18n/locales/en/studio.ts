/**
 * "studio" namespace — the policy studio overlay and its three panels
 * (1 · Collect, 2 · Train, 3 · Run/Deploy) plus the shared panel primitives.
 *
 * Key tree must match the other language exactly (see i18n/catalogs.test.ts).
 *
 * What is deliberately NOT here: dataset / model / job / robot / camera names,
 * repo ids, policy-type identifiers and checkpoint steps. Those are DATA and
 * are interpolated verbatim. Backend prose (`e.message`, `data.message`) is
 * likewise passed through untranslated — only the client-side wrapper around
 * it is a key.
 */
export default {
  // ── Studio shell (StudioOverlay) ──────────────────────────────────────────
  overlay: {
    // One key for the overlay's dialog aria-label AND the header eyebrow: the
    // same two words naming the same surface, so two keys could only drift.
    // The "by MakerMods" mark beside it is branding and stays English.
    title: "Policy studio",
    // aria-label and title on the same button — one key, two attributes.
    backToMenu: "Back to main menu",
    close: "Close studio",
    // aria-labels on the three panel <section>s. Deliberately more explicit
    // than the visible panel headings, which are read with their step digit.
    sections: {
      collect: "Collect dataset",
      train: "Train",
      deploy: "Deploy policy",
    },
  },

  // ── Shared panel chrome (panel/primitives.tsx) ────────────────────────────
  common: {
    // AdvancedSection's default heading. Resolved at render, never as a
    // module-level prop default — that would freeze the boot language.
    advancedParameters: "Advanced parameters",
    // aria-label on every dismiss "×": the handoff banner and MilestoneReveal.
    dismiss: "Dismiss",
  },

  // ── Panel 1 · Collect ─────────────────────────────────────────────────────
  collect: {
    title: "Collect",
    entry: "Record new dataset",
    start: "Start recording",
    library: {
      title: "Your datasets",
      // aria-label and title on the same button (both of these).
      clearSelected: "Clear selected dataset",
      refresh: "Refresh dataset list",
      merge: "Merge datasets",
    },
    form: {
      intro:
        "Name the dataset and set the capture parameters, then start recording on the selected robot.",
      noRobot:
        "Select or create a robot before recording — use the robot menu in the top-right corner of this window.",
      // <0> wraps the robot name; {{gap}} is the rendered setup-gap phrase
      // (robot.setupGap.* via formatRobotSetupGap).
      robotNotReady:
        "<0>{{name}}</0> {{gap}}. Open Robot settings before recording.",
      datasetName: "Dataset name *",
      // The <0>/<1>/<2> slots are <code> samples of the three allowed
      // punctuation characters — literal syntax, so they stay as they are.
      nameHint:
        "Letters, numbers, <0>.</0> <1>_</1> <2>-</2> only; start and end with a letter or number.",
      // <0> is the mono repo-id span; {{repoId}} is data, rendered verbatim.
      savedAs: "Will be saved as <0>{{repoId}}</0>",
      loginHint: "Log in to Hugging Face to set the repository owner.",
      task: "Task description *",
      taskPlaceholder:
        "e.g., pick up the red block and place it on the blue square",
      numEpisodes: "Number of episodes",
      episodeTime: "Episode duration (s)",
      resetTime: "Reset duration (s)",
      camerasEmptyRobot:
        "This robot has no cameras. Add them in Robot settings to record video.",
      camerasNoRobot: "Select a robot to see its cameras.",
      advancedSummary: "Streaming encoding, push to Hub",
      streamingLabel: "Streaming video encoding",
      streamingHint:
        "Encodes frames in real time during capture so each episode saves almost instantly. Uncheck to fall back to the slower PNG-then-encode flow.",
      pushToHubLabel: "Push to Hugging Face Hub",
      pushToHubHint:
        "Uploads the dataset to your Hugging Face account in the background once the session ends. Uncheck to keep it local — you can still upload it later from the dataset library.",
    },
    toast: {
      noRobotTitle: "No robot selected",
      noRobotBody:
        "Select or create a robot first — use the robot menu in the top-right corner.",
      missingDetailsTitle: "Missing dataset details",
      missingDetailsBody: "Please enter a dataset name and task description.",
      // The body is validateDatasetName's own message — client-side, but owned
      // by lib/datasetName.ts, so only this title is a key.
      invalidNameTitle: "Invalid dataset name",
      preparingCamerasTitle: "Preparing camera resources",
      releasingStreams_one:
        "Releasing {{count}} camera stream for recording...",
      releasingStreams_other:
        "Releasing {{count}} camera streams for recording...",
      camerasReadyTitle: "Camera resources ready",
      camerasReadyBody:
        "Camera streams released successfully. Starting recording...",
    },
  },

  // ── Post-recording handoff banner (CollectHandoff) ────────────────────────
  handoff: {
    emptyTitle: "No episodes were recorded",
    emptyBody:
      "Nothing was saved — the empty dataset was discarded so it doesn't take up disk space.",
    // <0> is the mono repo-id span; {{repoId}} is data.
    savedTitle: "Dataset <0>{{repoId}}</0> saved",
    episodes_one: "{{count}} episode",
    episodes_other: "{{count}} episodes",
    trainOnThis: "Train on this dataset",
    upload: {
      button: "Upload to Hub",
      uploading: "Uploading to Hub…",
      doneTitle: "Uploaded to Hub",
      // <0> wraps the "View dataset" link; {{repoId}} is data.
      doneBody: "{{repoId}} is now on the Hub. <0>View dataset</0>",
      failedTitle: "Upload failed",
      // The failure text itself comes from the backend and stays as sent; this
      // is only the label of the docs link rendered beside it.
      setupGuide: "Open setup guide",
      autoFailedTitle: "Automatic Hub upload not started",
    },
    milestone: {
      recording: {
        title: "Nice, your first episodes are recorded!",
        description:
          "Train a policy on this dataset from the Train panel, or upload it to the Hub to share it or train in the cloud.",
      },
      hubUpload: {
        title: "Uploaded to the Hub!",
        description:
          "Your dataset is public and shareable — reference its repo id anywhere in MakerLab, or fine-tune a policy on it from the Train panel.",
      },
    },
  },

  // ── Panel 2 · Train ───────────────────────────────────────────────────────
  train: {
    title: "Train",
    entry: "Start a new training",
    start: "Start training",
    intro: {
      fresh:
        "Choose what to train on, where the run executes, and how long it trains — then start.",
      resume:
        "Continuing an existing run — its dataset and weights are fixed. Set how much further it trains, then start.",
    },
    dataset: {
      label: "Dataset *",
      // The read-only variant shown while resuming a run.
      resumeLabel: "Dataset",
      resumeHint: "Inherited from the run being continued.",
      remove: "Remove {{repoId}}",
      searchPlaceholder: "Search datasets, or type a public org/name Hub id",
      // Note shown when the dataset arrived from the viewer curated down to
      // a subset of its episodes. The "of total" variant needs the dataset's
      // own episode count, which a Hub-only dataset may not resolve.
      episodeSubset:
        "Training on {{used}} episodes — adjust which ones from this dataset's viewer in My Library.",
      episodeSubsetOfTotal:
        "Training on {{used}} of {{total}} episodes — adjust which ones from this dataset's viewer in My Library.",
      // aria-label and title on the same button.
      choose: "Choose dataset",
      // Placeholder on the picker trigger before a dataset is chosen.
      pick: "Pick a dataset",
      // <0> is the mono repo-id span; {{repoId}} is the typed Hub id.
      useHub: "Use <0>{{repoId}}</0> from the Hub",
      useHubHint: "Public dataset — training fetches it on demand.",
      // <0> wraps the literal "org/name" id shape — syntax, not prose.
      noMatches:
        "No matching datasets. Type a full <0>org/name</0> id to use any public Hugging Face dataset.",
      hint: "Yours, or any public Hugging Face dataset.",
      // The per-row markers (episode count / weighted / Hub) live with the
      // picker that renders them now: `landing.datasetPicker.row.*`.
    },
    startingPoint: {
      label: "Starting point",
      // Both the <Select> placeholder and the "no base model" option's label.
      // The submitted option VALUE ("__none__") is untouched.
      scratch: "Train from scratch",
      // Shown instead of `scratch` for the foundation policies, which have no
      // real from-scratch: leaving Starting point unset fine-tunes their
      // public base checkpoint rather than training random weights.
      fromBase: "Train from base",
      loading: "Loading checkpoints…",
      finetuneHint: "Fine-tunes from this policy's latest checkpoint.",
      hint: "Fine-tune an existing policy, or start fresh.",
      // The `hint` counterpart for those same foundation policies.
      foundationHint:
        "Fine-tune an existing policy, or train from its public base.",
    },
    toast: {
      noCheckpointsTitle: "No checkpoints in this policy",
      noCheckpointsBody: "It has no saved checkpoint to fine-tune from.",
      // Body is the caught error's own message and stays as thrown.
      baseFailedTitle: "Couldn't load the starting point",
    },
    milestone: {
      title: "Training started!",
      description:
        "Watch progress from the jobs list above. Once it finishes, run it on your robot from the Deploy panel.",
    },
  },

  // ── Panel 3 · Run (Deploy) ────────────────────────────────────────────────
  // The post-coaching handoff, a sibling of CollectHandoff on the Launchpad:
  // a session that produced data puts the next step where the operator LANDS.
  coachHandoff: {
    saved_one: "{{count}} correction saved to <0>{{dataset}}</0>",
    saved_other: "{{count}} corrections saved to <0>{{dataset}}</0>",
    next: "Merge them with <0>{{dataset}}</0> — what this skill was last trained on — then fine-tune it on the result. Training takes one dataset, so the merge isn't optional.",
    manual:
      "To turn these into a better policy, merge them with the dataset this checkpoint was <0>last</0> trained on, then fine-tune from the same checkpoint on the merged result. Both steps are in the dataset library and the training panel.",
    action: "Merge & fine-tune",
  },

  deploy: {
    title: "Run",
    // The panel's entry control — the opener that slides the run form open,
    // matching collect.entry / train.entry.
    entry: "Run a policy",
    policy: {
      label: "Policy *",
    },
    picker: {
      placeholder: "Pick a policy",
      loading: "Loading policies…",
      empty: "No trained or imported policies yet",
      // Shown INSTEAD of `empty` when the listing could not be fetched —
      // an outage must not read as "you have no policies".
      error: "Couldn’t load policies. Check the server and try again.",
      // The "failed run" row badge lives with the picker that renders it now:
      // `landing.modelPicker.failedBadge`.
      hubDegraded:
        "Hub unreachable — showing your local policies and the last Hub listing.",
      // aria-label and title on the same button.
      import: "Import policy",
      hint: "Pick a trained checkpoint or an imported Hub policy to run on your robot.",
    },
    // Where a policy lives. Rendered as a small marker beside its name.
    source: {
      hub: "hub",
      local: "local",
      both: "local · hub",
    },
    // The run form's one-line brief, in the slot and voice Train uses.
    intro:
      "Pick a policy and its checkpoint, set how long it runs, and check the cameras — then start.",
    noRobot:
      "Select a robot to run on — use the robot menu in the top-right corner of this window.",
    // <0> wraps the robot name; {{gap}} is the rendered follower-scoped setup
    // gap. The plural is on the follower ARM count (one arm, or two when the
    // robot is bimanual) — the number itself is never printed.
    robotNotReady_one:
      "<0>{{name}}</0> {{gap}}. Open Robot settings before running inference. (Inference only uses the follower arm — leader setup isn't needed.)",
    robotNotReady_other:
      "<0>{{name}}</0> {{gap}}. Open Robot settings before running inference. (Inference only uses the follower arms — leader setup isn't needed.)",
    // Coaching's variant: it teleoperates through the leader, so {{gap}} here
    // is the ALL-arms gap and the parenthetical says why the leader matters.
    robotNotReadyCoach_one:
      "<0>{{name}}</0> {{gap}}. Open Robot settings before running inference. (Coaching also uses the leader arm — you teleoperate with it during takeovers, so it needs a port and a calibration.)",
    robotNotReadyCoach_other:
      "<0>{{name}}</0> {{gap}}. Open Robot settings before running inference. (Coaching also uses the leader arms — you teleoperate with them during takeovers, so they need a port and a calibration.)",
    // Which shape the run takes. Option VALUES ("single"/"eval"/"coach") are
    // identifiers the frontend switches on — only these labels are translated.
    runMode: {
      label: "What do you want to do with this skill?",
      // Each row states its COMMITMENT before it is chosen: these three are not
      // interchangeable menu items, and picking wrong is discovered at the arm.
      single: {
        title: "Just run it",
        what: "One attempt, then stop.",
        commitment: "hands off",
      },
      eval: {
        title: "Score it",
        what: "Repeat the task, and you judge every attempt into a success rate.",
        commitment:
          "hands on between episodes — you reset the scene and score each one",
      },
      coach: {
        title: "Coach it",
        what: "Take over when it's about to fail. Each rescue is saved as training data you can fine-tune on.",
        commitment: "hands on — you hold the leader arm the whole session",
      },
      remote: {
        title: "Run it remotely",
        what: "The arm runs here; the policy runs on a remote GPU over a LiveKit room.",
        commitment: "needs the GPU side running in another terminal",
      },
    },
    // Coaching-only parameters, shown when run mode is "coach".
    coaching: {
      correctionsLabel: "Corrections to collect",
      correctionsHint:
        "The session ends once you've saved this many. You can stop early at any point and keep everything recorded so far.",
      datasetLabel: "Corrections dataset",
      datasetPlaceholder: "e.g., fold_shirt_fixes",
      // Stand-in for the typed half of the name while the box is empty.
      datasetFallback: "correction",
      // <0> wraps {{prefix}}, the literal on-disk name — an identifier, so it
      // stays in the Latin script whatever the language.
      datasetHint:
        "Saved as <0>{{prefix}}</0> plus a timestamp. Leave it empty to use the greyed name, taken from the dataset this model was trained on; anything you type replaces it, and clearing the box brings it back.",
      leaderLabel: "Leader arm",
      leaderNoRobot: "Select a robot above.",
      leaderMissing:
        "This robot has no leader arm configured. Add its port and calibration in Robot settings — coaching can't run without one.",
      // {{configs}} is one or two calibration file names — data, never translated.
      leaderFrom:
        "Taken from {{name}}: {{configs}}. You'll teleoperate with it during takeovers.",
      bimanualWarning:
        "Bimanual: park the leader arms near the robot's pose before taking over. With two arms the robot moves to meet the leaders rather than the other way round, so a takeover from across the bench sweeps both arms through the scene. Takeovers that would travel too far are refused.",
    },
    checkpoint: {
      label: "Checkpoint",
      none: "No checkpoints available for this policy yet.",
      // Placeholder on the disabled dropdown shown before a policy is picked.
      pickPolicyFirst: "Pick a policy first",
    },
    // Checkpoint/robot arm-count mismatch. Each branch is one complete
    // sentence pair so word order is the translator's to choose. <0> is the
    // emphasised robot kind, <1> the robot's name; {{dim}} and {{arms}} are
    // raw numbers read off the checkpoint.
    armMismatch: {
      bimanualCheckpoint:
        "This checkpoint was trained on a <0>bimanual robot</0> ({{dim}}-dim state, {{arms}} arms), but <1>{{name}}</1> is a single-arm robot. Pick a single-arm checkpoint, or select a bimanual robot from the top-right robot menu.",
      singleCheckpoint:
        "This checkpoint was trained on a <0>single-arm robot</0> ({{dim}}-dim state), but <1>{{name}}</1> is a bimanual robot. Pick a bimanual checkpoint, or select a single-arm robot from the top-right robot menu.",
    },
    task: {
      label: "Task description",
      placeholder: "e.g., pick up the red block",
      // {{policyType}} is the policy identifier (act, smolvla, …) — data.
      // The field is always shown, so the helper answers "is this even read?"
      // in all three states: no policy picked yet, conditioned, not conditioned.
      hint: "This policy is language-conditioned ({{policyType}}).",
      hintUnknown:
        "Only language-conditioned policies use this — pick a policy to see whether yours does.",
      hintNotConditioned:
        "This policy ({{policyType}}) isn't language-conditioned — it ignores this.",
      // Appended to `hint` when the task was auto-filled from the checkpoint's
      // own training dataset. Leading space is added by the caller.
      prefilled: "Filled in from the dataset it was trained on.",
      // Placeholder when the lineage offered no task at all. Never an invented
      // example: a fake task greyed into the slot the REAL inherited one uses
      // is indistinguishable from one.
      placeholderNone: "No task found on the training dataset — type one",
      // Shown for a policy that does NOT read the task. Coaching still saves it.
      hintCoach:
        "Saved with every correction, so you can tell later what this session was teaching.",
      leaveEmpty:
        "Leave it empty to use the greyed task from the dataset it was trained on.",
      multiTaskHint_one:
        "Its training dataset has {{count}} task — pick the one you're running:",
      multiTaskHint_other:
        "Its training dataset has {{count}} tasks, most common first — pick the one you're running:",
    },
    duration: {
      label: "Max duration (s)",
      hint: "Per episode. An episode that runs this long without you calling it a success counts as a failure.",
      singleHint: "The run stops after this long.",
    },
    episodes: {
      label: "Episodes",
      // {{episodes}}, not {{count}}: the branch is chosen in code (> 1), so
      // i18next must not try to derive a plural of its own here.
      evalHint:
        "Evaluation run: {{episodes}} episodes with a reset between each, scored into an accuracy.",
      hint: "Leave at 1 for a single run. More than 1 starts a scored evaluation.",
      scoreHint: "How many episodes to score into the accuracy.",
    },
    engine: {
      label: "Inference engine",
      // Option labels only — the submitted values ("sync" / "rtc") are
      // identifiers the backend parses and are never translated.
      sync: "Sync (default)",
      rtc: "RTC — experimental, smoother control",
      syncHint:
        "One policy forward per control step. The arm pauses briefly between action chunks.",
      rtcHint:
        "Real-Time Chunking overlaps inference with motion, removing the pause between action chunks. It also changes how actions are generated — compare against Sync before trusting a result.",
      // Shown INSTEAD of the picker in coaching mode, which is pinned to sync.
      coachingNote:
        "Coaching always uses the Sync engine. Real-Time Chunking makes the arm jump back toward its pre-correction pose when the policy resumes, which isn't safe with a hand nearby.",
    },
    cameras: {
      title: "Cameras",
      loading: "Reading policy config…",
      // {{error}} is the backend's own message and is shown as sent.
      configError: "Couldn't load policy config: {{error}}",
      none: "This policy doesn't use cameras.",
      intro:
        "Bind one of this robot's cameras to each name the policy was trained with. Which camera and how it's opened come from the robot (edit in Robot settings); the capture resolution comes from the checkpoint.",
      captures: "Captures at {{width}}×{{height}} — the policy's resolution",
      // Shown when the bound camera's saved resolution differs from the one
      // the checkpoint asks for. {{name}} is the camera's own name — data.
      mismatch: "({{name}} is set to {{width}}×{{height}} in Robot settings)",
      disconnected: "Disconnected — reconnect it before starting",
      select: "Select a camera",
      robotHasNone: "This robot has no cameras — add them in Robot settings",
      // Empty state of the read-only camera list when no robot is selected.
      noRobot: "Select a robot to see its cameras.",
      // A camera the checkpoint names that the robot has nothing matching.
      // <0> emphasises the name; the name itself is DATA (the robot record's
      // own key), interpolated, never translated.
      unmatched:
        "The policy expects camera <0>{{name}}</0> but this robot has no camera named “{{name}}” — rename one in Robot settings.",
      // Matched by name, but the robot captures at a different size than the
      // checkpoint trained at. All four numbers are raw pixel dimensions.
      resolutionMismatch:
        "<0>{{name}}</0> is set to {{robotWidth}}×{{robotHeight}} in Robot settings, but the policy trained at {{policyWidth}}×{{policyHeight}} — the run captures at the policy's size.",
    },
    thumbnail: {
      // The preview tile's two placeholder states.
      released: "Released",
      noPreview: "No preview",
    },
    advanced: {
      summary: "Temporal ensembling for ACT",
      actionSelection: "Action selection",
      temporalEnsemble: "Temporal ensembling",
      temporalEnsembleHint:
        "Averages the overlapping action chunks the policy predicts at each step instead of executing one chunk open-loop — smoother motion, but the policy runs every control step, so it is slower.",
      coeffLabel: "Ensemble coefficient",
      // {{value}} is the pre-formatted default coefficient (0.01).
      coeffPlaceholder: "{{value}} (ACT paper default)",
      coeffInvalid: "Enter a number greater than 0.",
      coeffHint:
        "Weights are exp(-coeff × age): higher favours the newest prediction, lower averages more evenly. The ACT paper uses {{value}}.",
    },
    // The action row: each verb selects its mode and launches it in one press.
    runVerbs: {
      groupLabel: "Start a run",
      single: "Just run it",
      // {{count}} is the episode / correction target — a number, so no plural.
      eval: "Score it · {{count}}",
      coach: "Coach it · {{count}}",
      remote: "Run it remotely",
    },
    // Why a verb can't run, keyed so deployGuards.ts stays pure prose-free.
    blocked: {
      noRobot: "Select a robot above.",
      followerNotReady: "This robot's follower arm isn't ready.",
      noCheckpoint: "Pick a skill and a checkpoint.",
      armMismatch: "This checkpoint doesn't match the robot's arm count.",
      camerasUnbound: "Bind every camera the checkpoint expects.",
      temporalEnsemble: "Fix the temporal-ensemble setting.",
      runInProgress: "A run is already in progress.",
      taskRequired:
        "Describe the task first — this policy is language-conditioned.",
      leaderMissing:
        "Coaching needs a leader arm — add its port and calibration in Robot settings.",
      coachTaskRequired:
        "Describe the task first — it's saved with every correction.",
      transportNotReady:
        "The remote transport isn't ready — check it in the Remote run section below.",
      remoteArmUnsupported:
        "Remote runs need a single SO-101 arm. Bimanual rigs and the CAN arms aren't supported yet.",
    },
    actions: {
      start: "Start inference",
      // {{episodes}} rather than {{count}} — the branch is picked in code.
      startEval: "Start evaluation ({{episodes}})",
      // {{corrections}} rather than {{count}} — same reason as startEval.
      startCoach: "Start coaching ({{corrections}})",
      starting: "Starting…",
      checking: "Checking…",
      stop: "Stop inference",
      stopping: "Stopping…",
    },
    toast: {
      // Every *Failed title below is paired with the caught error's own
      // message, which stays exactly as thrown.
      loadPolicyFailed: "Couldn't load the policy",
      startFailed: "Couldn't start inference",
      stoppingTitle: "Stopping inference",
      stoppingBody: "The rollout is winding down.",
      stopFailed: "Stop failed",
    },
    milestone: {
      title: "First policy deployed!",
      description:
        "Your robot just ran a trained policy. Come back here anytime to redeploy it, swap checkpoints, or run a different policy.",
    },
  },
} as const;
