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
      titleWithStep: "Fine-tuning from “{{name}}” (step {{step}})",
      titleLatest: "Fine-tuning from “{{name}}” (latest checkpoint)",
      // <0> emphasises "fresh run", <1> the word "dataset".
      body: "This starts a <0>fresh run</0> (new optimizer, from step 0) with the policy weights initialized from that model. Pick a <1>dataset</1> to train on and set your training parameters as usual.",
    },
    tooltip: {
      // A busy local slot no longer blocks Start — the submission queues.
      willQueue:
        "A training is already running — this run will wait in the queue.",
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
      // Shown instead of "Start training" while the local slot is busy: the
      // click ENQUEUES the run rather than starting it.
      queueTraining: "Queue training",
      startFinetuning: "Start fine-tuning",
      continueTraining: "Continue training",
      uploadAndStart: "Upload & start training",
      uploadAndContinue: "Upload & continue training",
    },
    toast: {
      startedTitle: "Training Started",
      queuedTitle: "Training queued",
      // {{name}} is the run's name (data); {{position}} its 1-based place.
      queuedBody:
        "{{name}} — position #{{position}}. It starts when the current run finishes.",
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
    // The section header's live summary. <0> is the bold name — "This
    // machine" / the cloud label / a node's name (data).
    runOn: "Run on: <0>{{name}}</0>",
    runnerCloud: "Hugging Face Cloud",
    thisMachine: "This machine",
    // "MakerMods Lab" is the product name and stays untranslated.
    thisMachineSub: "Local — this MakerMods Lab server",
    cloudSub: "Rented GPU, billed per hour — pick hardware below",
    // The honest server-to-server sentence: choosing a node never moves the
    // browser off this server.
    selectorHint:
      "Where this run executes. The interface stays on this server — a node runs the job, driven server-to-server.",
    lanNodes: "LAN nodes",
    nodesLoading: "Looking for nodes…",
    // "Tailscale" is a product name. Shown only while a tailscale discovery
    // source is registered — otherwise the sentence would promise discovery
    // that will never happen (nodesEmptyNoDiscovery covers that case).
    nodesEmpty:
      "No nodes yet — other MakerMods Lab servers appear here when discovered via Tailscale, or add one by URL.",
    // <0> wraps the literal CLI flag --discover-tailscale: data, rendered
    // verbatim in a mono span, never translated.
    nodesEmptyNoDiscovery:
      "No nodes yet — add another MakerMods Lab server by URL. Starting the server with <0>--discover-tailscale</0> finds them over your tailnet automatically.",
    viaTailscale: "via tailscale",
    verifying: "Verifying…",
    verifyingTitle: "Discovered — waiting for the verify handshake to confirm it.",
    unreachable: "Unreachable",
    // {{when}} is a pre-formatted relative duration ("4m ago") — duration
    // formatting stays English everywhere (lib/relativeTime.ts).
    lastSeen: "last seen {{when}}",
    unreachableTitle:
      "Unreachable — it reappears as selectable when the next handshake succeeds.",
    unreachableTitleLastSeen:
      "Unreachable — last seen {{when}}. It reappears as selectable when the next handshake succeeds.",
    // "makermodslab" is the package name; {{version}} is data.
    nodeVersion: "makermodslab v{{version}}",
    nodeGone: "No longer registered",
    nodeGoneTitle:
      "This node has left the registry. Pick another target, or starting will be refused.",
    refreshNodes: "Refresh nodes",
    addNode: {
      button: "Add node…",
      title: "Register another MakerMods Lab server by URL",
      urlLabel: "Node URL",
      submit: "Add",
      adding: "Adding…",
      // Client-side sentences for the backend's CODED refusals (node.self /
      // node.duplicate / node.unreachable). Every uncoded refusal shows the
      // server's own detail verbatim instead.
      errorSelf: "That URL answers as this server itself.",
      errorDuplicate: "That node is already registered.",
      errorUnreachable: "No MakerMods Lab server answered at that URL.",
    },
    detail: {
      instance: "Instance",
      version: "Version",
      lastSeenLabel: "Last seen",
      workloadLoading: "Checking workload…",
      workloadUnreachable:
        "Unreachable — couldn't ask this node for its workload.",
      // {{name}} is the run's display name (data); {{pct}} is pre-formatted.
      workloadRunningPct: "Running: {{name}} · {{pct}}%",
      workloadRunning: "Running: {{name}}",
      workloadIdle: "Idle",
      // A plain figure, named `total` rather than `count` so i18next does not
      // treat it as a plural selector.
      workloadQueued: "+{{total}} queued",
      // <0>/<1> both wrap the node's name (data). "Hub" / "HF" are product
      // names.
      hubSyncHint:
        "The job runs on <0>{{name}}</0>; datasets sync via the Hub — this dataset uploads to your HF account first, and <1>{{name}}</1> pulls it from there.",
      goneBody:
        "This node is no longer in the registry. Pick another compute target — starting the run would be refused.",
      // The remote-restart button (two-step arm/confirm) and its status line.
      // {{name}} is the node's display name — data, rendered verbatim.
      restartAction: "Restart node",
      restartConfirm: "Confirm restart?",
      restartRequested:
        "Restart requested — {{name}} will drop off for a few seconds and come back.",
    },
    // The drill-in dialog for one run ON a peer node (NodeJobDialog). Run
    // names, numbers and log lines are data, rendered verbatim; the state
    // labels come from jobs.jobState, and the stop/delete toasts reuse
    // jobs.jobsData so a peer run reads exactly like a local one.
    nodeJob: {
      // One whole sentence; <0> wraps the node's display name (data).
      onNode: "Runs on <0>{{name}}</0> — driven server-to-server from here.",
      logsLabel: "Live log tail",
      logsEmpty: "No log lines since this dialog opened.",
      stop: "Stop",
      stopping: "Stopping…",
      delete: "Delete",
      deleting: "Deleting…",
      // Our sentence for the proxy's coded 502 (node.unreachable); every
      // uncoded refusal shows the server's own prose verbatim.
      unreachable: "Unreachable — couldn't reach this node just now.",
      // Hover text on the clickable running line / queued chips.
      openRunning: "View this run's details",
      openQueued: "View this queued run",
    },
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
    // {{node}} is a LAN node's display name — data, rendered verbatim.
    readyPolicyTrainingNode:
      "Install complete on {{node}} — {{policy}} training is available there immediately, no restart needed. Start the run again.",
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
    // The LAN-node variant: the extra lands on the PEER's environment, and
    // every sentence says so. {{node}} is the node's display name (data).
    titleNode: "{{policy}} needs an extra package on {{node}}",
    srDescriptionTrainingNode:
      "Install {{target}} on the node {{node}} for training with {{policy}}.",
    descriptionTrainingNode:
      "Training a <0>{{policy}}</0> policy on <3>{{node}}</3> needs the <1>{{packageName}}</1> package (installed via <2>{{target}}</2>), which isn't in that node's environment yet. Install it there — the pip install runs on the node.",
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
    runnerNode: "LAN node",
    // Stand-in when a cloud job has no flavor recorded.
    cloudFallback: "cloud",
    viewOnHub: "View on Hub ↗",
    viewOnWandb: "View on W&B ↗",
    stop: "Stop",
    cancelQueued: "Cancel",
    delete: "Delete",
    runInference: "Run inference",
    noCheckpoints: "No checkpoints yet — wait for the first save.",
    runOnRobot: "Run on robot",
    toast: {
      stoppingTitle: "Stopping…",
      stopFailedTitle: "Stop failed",
      cancelledTitle: "Removed from the queue",
      cancelFailedTitle: "Cancel failed",
      removedTitle: "Job removed",
      deleteFailedTitle: "Delete failed",
    },
  },

  publish: {
    title: "Publish to Hub",
    intro:
      "Share this run's checkpoints as a public model on the Hub — every step you pick lands in one repo, under one model card.",
    // Rendered after the pinned repo link, behind a literal "· " separator.
    hubUnknownShort: "couldn't check which checkpoints are published",
    publishedOf: "{{published}} of {{total}} checkpoints published",
    addCheckpoints: "Add checkpoints",
    uploadToHub: "Upload to Hub",
    // {{repo}} is the pinned repo id — data, rendered verbatim inside slot 0.
    addingTo:
      "Adding to <0>{{repo}}</0>. A run keeps one repo, so every checkpoint stays under the same model card.",
    repoNameLabel: "Repo name (optional)",
    // {{placeholder}} is the default repo id — data, verbatim inside slot 0.
    leaveBlank:
      "Leave blank to publish as <0>{{placeholder}}</0>. Later checkpoints go to this same repo.",
    repoInvalid: "That's not a valid repo name — use name or namespace/name.",
    // {{namespace}} is the namespace the user typed — data, verbatim.
    repoNotWritable: "Your token can't write to {{namespace}}.",
    checkpointsLabel: "Checkpoints",
    clearAll: "Clear all",
    // {{total}} is a plain display number, deliberately NOT `count` — no
    // plural selection wanted here.
    selectAllCount: "Select all ({{total}})",
    hubUnknownDetail:
      "Couldn't reach the Hub to check which checkpoints are already published — the badges below may be incomplete.",
    publishedBadge: "published",
    // Shown only for 2+ selections, so no _one form.
    multiNote_other:
      "{{count}} checkpoints upload one after another — each is a full copy of the policy weights.",
    overwriteNote: "Re-selecting a published checkpoint overwrites it in place.",
    selectPrompt: "Select a checkpoint",
    uploadCount_one: "Upload {{count}} checkpoint",
    uploadCount_other: "Upload {{count}} checkpoints",
    // {{current}} / {{total}} are display numbers (current is 1-based).
    uploadingOf: "Uploading {{current}} of {{total}}",
    uploading: "Uploading",
    publishingAria: "Publishing checkpoints",
    legacyRootNote:
      "This repo also holds a checkpoint at its root from an earlier upload. It stays readable, but the step-addressed copies above are what tools load.",
    toast: {
      publishedTitle_one: "Published {{count}} checkpoint",
      publishedTitle_other: "Published {{count}} checkpoints",
      // {{repoId}} is data; slot 0 is the View-model link.
      publishedBody: "{{repoId}} is on the Hub. <0>View model</0>",
      failedTitle: "Publish failed",
      // Appended (with its own parentheses) after the backend's verbatim
      // failure message when some checkpoints did land.
      failedLanded_one: "({{count}} checkpoint already published — retry the rest.)",
      failedLanded_other: "({{count}} checkpoints already published — retry the rest.)",
    },
  },
} as const;
