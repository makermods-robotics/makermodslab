/**
 * "training" namespace — the Train panel's configuration form, its install
 * gates, the cloud-upload notices, and the job monitor dialog.
 * Key tree must match the other language exactly (see i18n/catalogs.test.ts).
 *
 * NOT in here on purpose:
 *  - policy identifiers (`act`, `smolvla`) and their product names (ACT,
 *    SmolVLA) — POLICY_TYPE_OPTIONS in components/training/types.ts. The
 *    identifiers are wire values; the names are product names.
 *  - `formatDurationShort` output ("2h30m") in lib/jobTimeout.ts — it is the
 *    exact value typed back into the timeout input and re-parsed against the
 *    backend's regex, so its unit suffixes must never be localized.
 *  - hyperparameter keys (lr, batch_size, steps) and optimizer/W&B option
 *    values, which are sent to the backend verbatim.
 */
export default {
  header: {
    title: "Training",
  },

  // Copy shared by the two "this only exists on your machine" notices.
  cloudNotice: {
    // {{action}} is the Start button's own label, quoted so the notice and the
    // button can't drift apart.
    uploadHint: "Use “{{action}}” below to upload, then launch.",
  },

  // Read-only explanations for controls a resume rebuilds from the parent
  // run's checkpoint. Module-level KEYS live in components/training/types.ts.
  resumeInherited: {
    note: "Rebuilt from the parent run's checkpoint — a resume continues the same experiment, so changing these here has no effect. To train with different settings, fine-tune from this checkpoint instead.",
    short: "Rebuilt from the parent run's checkpoint.",
  },

  configurator: {
    checkingEnvironment: "Checking training environment…",
    // Blocks Start until the user raises the step target. {{step}} arrives
    // pre-formatted (toLocaleString) — formatting is the caller's.
    resumeStepError:
      "Total steps must be greater than the checkpoint's step ({{step}}).",
    resume: {
      // {{name}} is the parent run's name — data, rendered verbatim.
      titleFromStep: "Continuing “{{name}}” from step {{step}}",
      // Shown only when the total differs from the parent run's. lerobot
      // rebuilds the LR schedule from the new total, so the rate jumps at the
      // resume point instead of continuing to decay.
      lrSeam:
        "Total steps differ from the original run ({{from}} → {{to}}). LeRobot rebuilds the learning-rate schedule from the new total, so the rate can jump back up at the resume point instead of continuing to decay. Keep {{from}} for an unbroken schedule.",
      titleFromLatest: "Continuing “{{name}}” from its latest checkpoint",
      // <0> emphasises the "Steps" control by name. {{steps}} is the
      // pre-formatted prefill. One complete sentence per runner, because the
      // cloud one names an extra control mid-list.
      bodyLocal:
        "Settings are prefilled from that run and stay editable. The dataset, policy, batch size, and optimizer are rebuilt from the checkpoint itself, so changing them here won't affect the continuation — but <0>Steps</0> and the checkpoint cadence both apply. Set Steps above the resumed step to train further (prefilled to {{steps}}).",
      bodyCloud:
        "Settings are prefilled from that run and stay editable. The dataset, policy, batch size, and optimizer are rebuilt from the checkpoint itself, so changing them here won't affect the continuation — but <0>Steps</0>, the checkpoint cadence, and the job timeout all apply. Set Steps above the resumed step to train further (prefilled to {{steps}}).",
      // {{timeout}} is an HF-Jobs duration string — wire format, verbatim.
      jobTimeout:
        "Job timeout: <0>{{timeout}}</0> — a continuation needs at least as long as the tail it has left.",
      jobTimeoutDefault: "24h (default)",
    },
    finetune: {
      // Used when the checkpoint picker is shown: the picker names the step, so
      // repeating it here printed the same number twice.
      title: "Fine-tuning from “{{name}}”",
      titleWithStep: "Fine-tuning from “{{name}}” (step {{step}})",
      titleLatest: "Fine-tuning from “{{name}}” (latest checkpoint)",
      checkpointLabel: "Checkpoint",
      // <0> emphasises "Fresh run".
      body: "<0>Fresh run</0> from step 0 — new optimizer, with the policy weights loaded from this checkpoint.",
    },
    tooltip: {
      localBusy: "Another local training is already running",
      needAuth: "Log in to Hugging Face to use cloud compute",
      needFlavor: "Select a hardware flavor",
      offlineDataset:
        "Offline mode is on — the dataset can't be uploaded to the Hub",
      offlineCheckpoint:
        "Offline mode is on — the checkpoint can't be uploaded to the Hub",
    },
    button: {
      uploading: "Uploading…",
      starting: "Starting…",
      startTraining: "Start training",
      startFinetuning: "Start fine-tuning",
      continueTraining: "Continue training",
      uploadAndStart: "Upload & start training",
      uploadAndContinue: "Upload & continue training",
    },
    toast: {
      startedTitle: "Training Started",
      errorTitle: "Error",
      datasetRequired: "Dataset repository ID is required",
      uploadFailedTitle: "Upload failed",
    },
  },

  policyField: {
    label: "Policy",
    placeholder: "Select a policy type",
    hint: "The network architecture this run trains.",
    hintLocked:
      "Set by the starting point — the run trains the same architecture as its source checkpoint.",
  },

  target: {
    computeLabel: "Compute",
    runnerLocal: "Local — your machine",
    runnerCloud: "Hugging Face Cloud",
    resumeRunnerHint:
      "Defaults to the runner this run started on — switch it to continue somewhere else.",
    deviceLabel: "Device",
    deviceAuto: "Automatic (use GPU if available)",
    deviceCpu: "CPU",
    deviceHint: "lerobot auto-detects your GPU (CUDA/MPS); only CPU is forced.",
    hardwareLabel: "Hardware",
    hardwareLoading: "Loading…",
    hardwarePlaceholder: "Select hardware",
    loginToHf: "log in to HF",
    costHint:
      "Cost shown is per running hour. Final policy uploads to your HF account when training completes.",
  },

  essentials: {
    steps: "Training steps",
    // Resume only. The field is a TOTAL the run trains up to, not an increment
    // added to the steps already done — the label and hint both say so, and the
    // hint does the subtraction so the user never has to.
    stepsTotal: "Total training steps",
    stepsTotalHint:
      "Resuming from step {{from}}, training {{remaining}} more steps.",
    stepsTotalHintLatest: "Total step count, not additional steps.",
    stepsTotalTooLow:
      "Must be above {{from}} — the run has already trained that far, so this would train nothing.",
    batchSize: "Batch size",
    runName: "Run name",
    // Sits beside the Run name label on a continuation. {{step}} is
    // pre-formatted by the caller.
    resumedFromStep: "from step {{step}}",
    resumedFromLatest: "from latest checkpoint",
    // Stand-ins inside the run-name placeholder when nothing is chosen yet.
    // The policy half is upper-cased by the caller (a no-op on CJK).
    runNamePolicyFallback: "policy",
    runNameDatasetFallback: "dataset",
    runNameHint: "Optional — shown on the job card and searchable.",
    wandbEnable: "Log to Weights & Biases",
    wandbProject: "W&B project name",
    wandbEntity: "W&B entity (optional)",
    wandbNotes: "W&B notes (optional)",
    wandbNotesPlaceholder: "Training run notes...",
    wandbMode: "W&B mode",
    // Labels only — the <SelectItem> values stay "online"/"offline"/"disabled".
    wandbModeOnline: "Online",
    wandbModeOffline: "Offline",
    wandbModeDisabled: "Disabled",
    wandbDisableArtifact: "Disable artifacts",
  },

  advanced: {
    summary: "Optimizer, learning rate, log frequency, checkpoints, and more",
    sectionPolicyPreset: "Policy preset",
    useAmp: "Use automatic mixed precision",
    sectionTraining: "Training",
    randomSeed: "Random seed",
    sectionOptimizer: "Optimizer",
    optimizerLabel: "Optimizer",
    // Optimizer display names. Kept identical across languages — these are
    // algorithm names, not copy — but held as catalog entries so the label map
    // in AdvancedCard.tsx stores keys instead of freezing resolved strings at
    // import time. The identifiers (adam, adamw…) are backend values.
    optimizerName: {
      adam: "Adam",
      adamw: "AdamW",
      sgd: "SGD",
      multiAdam: "Multi Adam",
    },
    optimizerUnknown: "Set by the policy preset",
    // {{policy}} is a policy's short product name (ACT, SmolVLA…).
    optimizerFixedByPolicy:
      "Set by the {{policy}} policy preset — the optimizer class isn't adjustable.",
    optimizerFixedGeneric:
      "The policy preset picks the optimizer class; it isn't adjustable.",
    optimizerNoKnobs:
      "The {{policy}} preset builds its optimizer from per-parameter-group settings, so there are no learning-rate or weight-decay knobs to set here.",
    learningRate: "Learning rate",
    weightDecay: "Weight decay",
    gradientClipping: "Gradient clipping",
    noGradClip: "The {{policy}} policy exposes no gradient-clipping setting.",
    noGradClipOrWeightDecay:
      "The {{policy}} policy exposes no gradient-clipping setting or weight decay.",
    // Placeholders for the optimizer number inputs. {{value}} is a formatted
    // number from the backend's preset — data, rendered verbatim.
    policyDefaultValue: "{{value}} (policy default)",
    usePolicyDefault: "Use policy default",
    sectionDataLoading: "Data loading",
    numWorkers: "Number of workers",
    numWorkersHint: "DataLoader processes feeding the GPU.",
    sectionLogging: "Logging & checkpointing",
    logFreq: "Log frequency",
    logFreqExceeds:
      "⚠ Logging every {{logFreq}} steps exceeds the {{steps}}-step run — no metrics will be logged.",
    logFreqHint:
      "Steps between logged loss/lr points. Lower = higher-resolution charts (each point is a window average), but more log volume.",
    saveFreq: "Save frequency",
    saveFreqExceeds:
      "⚠ Saving every {{saveFreq}} steps exceeds the {{steps}}-step run — no checkpoint will be saved.",
    sectionCloud: "Cloud",
    jobTimeout: "Job timeout",
    // The placeholder shows the wire format the field accepts, so the duration
    // literals stay exactly as the backend parses them.
    jobTimeoutPlaceholder: "2h (default)",
    jobTimeoutInvalid:
      'Use a duration like "2h", "45m", or "3h30m" (units: s, m, h, d).',
    jobTimeoutHint:
      "HF Jobs kills the run after this long. Leave blank for the 2h default.",
  },

  datasetNotice: {
    title: "This dataset is only on this machine",
    // <0> is the HF_HUB_OFFLINE code sample, <1> the dataset repo id (data).
    offline:
      "Hugging Face Cloud trains from the Hub, but the server is in offline mode (<0>HF_HUB_OFFLINE</0>), so <1>{{repoId}}</1> can't be uploaded. Switch off offline mode, or run this training locally.",
    // Two complete sentences rather than an optional size fragment. <0> is the
    // repo id, <1> emphasises "private". {{size}} is pre-formatted.
    body: "Hugging Face Cloud trains from the Hub, so <0>{{repoId}}</0> will be uploaded as a <1>private</1> dataset before training starts.",
    bodyWithSize:
      "Hugging Face Cloud trains from the Hub, so <0>{{repoId}}</0> (~{{size}}) will be uploaded as a <1>private</1> dataset before training starts.",
    uploading:
      "Uploading to the Hub… this can take a few minutes for large datasets.",
  },

  checkpointNotice: {
    title: "This checkpoint is only on this machine",
    // Which checkpoint is moving, as a noun phrase slotted into the sentences
    // below. {{step}} is pre-formatted by the caller.
    stepLabel: "step {{step}}",
    latestLabel: "its latest checkpoint",
    // One complete sentence per mode — a resume moves the whole checkpoint, a
    // fine-tune only its weights, and each ends with its own advice.
    // <0> is the HF_HUB_OFFLINE code sample, <1> the run name (data).
    offlineResume:
      "Hugging Face Cloud continues from the Hub, but the server is in offline mode (<0>HF_HUB_OFFLINE</0>), so {{stepLabel}} of <1>{{runName}}</1> can't be uploaded. Switch off offline mode, or continue this run locally.",
    offlineFinetune:
      "Hugging Face Cloud loads base weights from the Hub, but the server is in offline mode (<0>HF_HUB_OFFLINE</0>), so {{stepLabel}} of <1>{{runName}}</1> can't be uploaded. Switch off offline mode, or run this fine-tune locally.",
    // <0> is the run name, <1> emphasises "private".
    bodyResume:
      "Hugging Face Cloud continues from the Hub, so {{stepLabel}} of <0>{{runName}}</0> — its weights and optimizer state — will be uploaded to a <1>private</1> repo in your account before the job starts. Continuing the same checkpoint again reuses that upload.",
    bodyFinetune:
      "Hugging Face Cloud loads base weights from the Hub, so {{stepLabel}} of <0>{{runName}}</0> — its weights, not its optimizer state — will be uploaded to a <1>private</1> repo in your account before the job starts. Fine-tuning the same checkpoint again reuses that upload.",
    privacy: "Private to your account — nothing is published.",
  },

  install: {
    titleDone: "Install Complete",
    titleError: "Install Failed",
    titleInstalling: "Installing…",
    copyAria: "Copy install command",
    copiedTitle: "Copied",
    copyFailedTitle: "Copy failed",
    copyFailedDescription: "Select the command and copy manually.",
    installNow: "Install Now",
    // <0> is the package name (data, rendered verbatim).
    installing:
      "Installing <0>{{packageName}}</0>. This usually takes about 10 seconds.",
    // Client-side fallback only — the backend's own error text wins when set.
    failedFallback: "Install failed.",
    tryAgain: "Try again",
    // Each caller gets its own complete "ready" sentence rather than slotting a
    // noun into a shared template.
    readyTraining:
      "Install complete — training is available immediately, no restart needed. Reload the page if it doesn't unlock on its own.",
    readyWandb:
      "Install complete — W&B logging is available immediately, no restart needed. Reload the page if it doesn't unlock on its own.",
    // {{policy}} is a policy's short product name (ACT, SmolVLA…).
    readyPolicyTraining:
      "Install complete — {{policy}} training is available immediately, no restart needed. Reload the page if it doesn't unlock on its own.",
    readyPolicyInference:
      "Install complete — {{policy}} inference is available immediately, no restart needed. Reload the page if it doesn't unlock on its own.",
  },

  extraGate: {
    title: "Training Extra Not Installed",
    // <0> is the `accelerate` package name — a literal, never translated.
    description:
      "Training requires the <0>accelerate</0> package, which isn't installed in this environment. Install it to enable the Training page.",
  },

  wandbDialog: {
    title: "Weights & Biases Not Installed",
    srDescription: "Install the wandb package to enable W&B logging.",
    // <0> is the `wandb` package name.
    description:
      "Enabling W&B logging requires the <0>wandb</0> package, which isn't installed in this environment. Install it to log this run to W&B.",
  },

  policyExtra: {
    // {{policy}} is a policy's short product name (ACT, SmolVLA…).
    title: "{{policy}} needs an extra package",
    // {{target}} is a pip install target ("lerobot[smolvla]") — data.
    srDescriptionTraining: "Install {{target}} for training with {{policy}}.",
    srDescriptionInference: "Install {{target}} for inference with {{policy}}.",
    // One complete sentence per purpose — English slotted a verb ("Training" /
    // "Running") and a noun into one template, which no translation survives.
    // <0> emphasises the policy name, <1> and <2> are the package name and the
    // install target (both data).
    descriptionTraining:
      "Training a <0>{{policy}}</0> policy needs the <1>{{packageName}}</1> package (installed via <2>{{target}}</2>), which isn't in this environment yet. Install it to train this policy.",
    descriptionInference:
      "Running a <0>{{policy}}</0> policy needs the <1>{{packageName}}</1> package (installed via <2>{{target}}</2>), which isn't in this environment yet. Install it to run this policy.",
  },

  monitoring: {
    progress: "Progress",
    startingUp: "Training starting…",
    eta: "ETA",
    warmingUp: "warming up…",
    loss: "Loss",
    learningRate: "Learning rate",
    waitingForMetrics: "Waiting for first metric tick…",
    logsTitle: "Training logs",
    logsEmpty: "No training logs yet. Start training to see output.",
  },

  jobDialog: {
    srTitle: "Training job status",
    back: "Skill studio",
    // {{jobId}} is data; {{errorText}} is the backend's own message, left as it
    // arrived.
    loadFailed: "Couldn't load job {{jobId}}: {{errorText}}",
    loading: "Loading job…",
    runnerLocal: "Local",
    // Stand-in when a cloud job has no flavor recorded.
    cloudFallback: "cloud",
    viewOnHub: "View on Hub ↗",
    viewOnWandb: "View on W&B ↗",
    stop: "Stop",
    delete: "Delete",
    runInference: "Run inference",
    noCheckpoints: "No checkpoints yet — wait for the first save.",
    runOnRobot: "Run on robot",
    toast: {
      stoppingTitle: "Stopping…",
      stopFailedTitle: "Stop failed",
      removedTitle: "Job removed",
      deleteFailedTitle: "Delete failed",
    },
  },
} as const;
