export default {
  network: {
    autoRegionHint: "Modal chooses the region. Network delay may vary.",
    "title": "Network",
    "region": "GPU region",
    "regionHint": "Choose a nearby region. Restart GPU to apply.",
    "advanced": "Advanced",
    "videoQuality": "MJPEG quality",
    "videoQualityHint": "Lower uses less bandwidth.",
    "videoBitrateKbps": "H264 bitrate (kbps)",
    "videoBitrateKbpsHint": "Per camera. Lower uses less bandwidth.",
    "cameraSendHz": "Camera send rate (Hz)",
    "cameraSendHzHint": "0 = automatic. Caps camera and state updates, not arm speed.",
    "latencyK": "Jitter margin",
    "latencyKHint": "Higher allows for more variation in delay.",
    "tolerance": "Frame matching tolerance",
    "toleranceHint": "In control ticks. Higher accepts less closely timed frames. Restart GPU to apply.",
    "budget": "RTC timing budget",
    "delay": "Estimated delay {{delay}} / {{budget}} ms",
    "within": "Within budget",
    "near": "Near limit",
    "over": "Over budget",
    "budgetHint": "Includes inference and jitter margin. Queue holds may still occur."
},
  form: {
    autoAssignment: "Auto (less wait time)",
    autoShort: "Auto",
    autoGpuHint: "A10G, L4 or larger. Price varies.",

    motionTuning: "Motion",
    gpuTuning: "GPU tuning",
    humanUnavailable: "Human in the loop is unavailable for remote runs.",

    hubIdLabel: "Hub policy id",
    // The engine's LABELS live under `studio.deploy.engine` — it is one field
    // for both places a run can happen. These hints are the pair that survived
    // the merge, because they describe what the two engines DO rather than
    // which of them is the default.
    //
    // There was a third: a remote-only warning that the selected checkpoint
    // couldn't be in-painted. It went with the fail-open half of the engine
    // rule — the rtc option is now disabled outright for such a checkpoint, on
    // both paths, so nobody can select it and be warned afterwards. The one
    // remaining sentence for that state is `studio.deploy.engine.rtcUnavailable`.
    engine: {
      syncHint:
        "Plays one chunk at a time.",
      rtcHint:
        "Plans the next chunk while the arm moves.",
    },
    // The Advanced trigger's summary line. Every value is live and is DATA —
    // the codec id is the wire value, the numbers are the ones that go out.
    // Two whole sentences rather than one plus a fragment: `s_min` only
    // reaches the wire for rtc, so it is only claimed there. The GPU's own
    // knobs used to be appended here; they live on the Modal card now, and the
    // line says "Transport" because that is all it still describes.
    advancedSummary: "{{fps}} Hz · {{horizon}} actions · {{codec}}",
    advancedSummaryRtc:
      "{{fps}} Hz · smoothing {{lpfHz}} Hz",
    horizonLabel: "Chunk size (actions)",
    fpsLabel: "Control rate (Hz)",
    codecLabel: "Video codec",
    // The checkpoint's own chunk width. {{steps}} is a number read off its
    // config, never a translated value.
    horizonFromCheckpoint:
      "Maximum chunk size: {{steps}} actions.",
    horizonOverCeiling:
      "Use {{steps}} actions or fewer to connect correctly.",
    filterLabel: "Smoothing (Hz)",
    filterHint: "Lower is smoother but slower to respond. 0 is off.",
    sMinLabel: "Replan interval (steps)",
    // "--s-min" is a flag name, kept in the Latin script like every other
    // identifier in this panel.
    sMinHint:
      "Minimum steps between plans. Start with 4.",
    // The GPU-side knobs, which live on the Modal card rather than under
    // Advanced: nothing here has to match the arm — they decide what the
    // container loads and what it loads onto. Labels only; the card is a strip
    // of selects beside Start GPU and carries no prose of its own.
    precisionLabel: "Precision",
    // The one option that is prose: it stands for passing no flag at all. The
    // others are torch dtype names — wire values, never translated.
    precisionCheckpoint: "Model default",
    gpuLabel: "GPU",
    gpuHint:
      "Choose a larger GPU if the model runs out of memory.",
    // Said where the disabled select is, because the reason belongs to THIS
    // checkpoint. No policy type is named: the fact the operator needs is that
    // this one has no such setting, not which family it belongs to.
    precisionUnavailable:
      "Uses the model’s saved precision.",
    // The flow-steps knob (S3.8f).
    flowStepsLabel: "Flow steps",
    slackLabel: "Buffer (steps)",
    slackHint:
      "Increase for uneven delivery. Restart GPU to apply.",
    // Prose, like the precision's first option: it stands for passing no flag.
    // The second form carries the number this checkpoint will actually run at,
    // which is data — the server works it out, never this file.
    flowStepsCheckpoint: "Model default",
    flowStepsCheckpointKnown: "Model default ({{steps}})",
    flowStepsUnavailable:
      "This model has no flow-step setting.",
  },
  // The per-role camera picker. It appears ONLY for checkpoint cameras that
  // matched nothing by name, so most runs never see it.
  cameraRoles: {
    title: "Camera roles",
    // Roles that bound themselves and need no control.
    nameMatched_one: "{{count}} other camera matched by name.",
    nameMatched_other: "{{count}} other cameras matched by name.",
    // Both numbers are raw pixel dimensions from the checkpoint's config.
    capturesAt: "The policy trained at {{width}}×{{height}}.",
    unbound: "Not chosen",
    noCameras: "This robot has no cameras. add one in Robot settings.",
    disconnected: "Not plugged in right now.",
    // S3.8g — a view the checkpoint does not declare at all. The role NAME
    // interpolated below is data (cam2), rendered verbatim in every language.
    addRole: "Add a camera role",
    addRoleHint:
      "This checkpoint was fine-tuned with the two views it declares, but the model underneath takes any number of them, so the GPU can be asked for one more. It costs latency (more image tokens per step) and the checkpoint's own authors never tested it. measure before trusting it. The extra camera needs to be chosen above, and the GPU must be started from this panel so both halves agree.",
    addRoleFull:
      "That is as many extra views as this launcher will add. Each one is more work per step, and past a point the chunk arrives after the arm needed it.",
    extraBadge: "Added for this run. not a view the checkpoint was trained with.",
    remove: "Remove",
    removeRole: "Remove the camera role {{role}}",
  },
  // Backend engine values. Matched on, never displayed raw — the raw value is
  // the fallback for an engine a newer server introduces.
  engine: {
    sync: "Adaptive sync",
    rtc: "Real-time chunking",
  },
  modalRun: {
    manualToggle: "Manual setup",
    title: "What the Lab will run",
    intro:
      "Copy this command to start the GPU from a terminal.",
    copy: "Copy",
    copiedTitle: "Command copied",
    copyFailedTitle: "Couldn't copy",
    copyFailedBody: "Select the command and copy it by hand.",
    noRoomYet:
      "Check the connection before copying the command.",
    tokenHint: "This room token expires in about an hour. Copy again to get a fresh token.",
    noTailnetUrl:
      "Sign in to Tailscale, then check the connection again.",
  },
  // The GPU half, which the Lab launches itself since S3.8. It does NOT gate
  // the remote verb — that stays the transport probe's operator check.
  gpu: {
    setup: {
      install: "Install Modal to start a GPU.",
      signIn: "Sign in to Modal to start a GPU.",
      where: "Run these commands on the computer hosting MakerMods Lab.",
      checkAgain: "Check again",
      failed: "GPU start failed. Details",
    },

    title: "Policy server on Modal",
    start: "Start GPU",
    retry: "Try again",
    stop: "Stop GPU",
    cancel: "Cancel",
    // {{wrapper}} is the wrapper's PATH and {{gpu}} the Modal GPU spec — both
    // data, shown verbatim. The GPU is interpolated rather than written into
    // the sentence because it is a choice now (S3.8e).
    idleHint:
      "{{gpu}} on Modal. First start may take 1–3 minutes.",
    // {{seconds}} is a plain integer, deliberately not i18next's magic `count`.
    elapsed: "{{seconds}}s",
    // Backend phase values. Matched on, never displayed raw — the raw value is
    // the fallback for a phase a newer server introduces.
    phase: {
      pending: "Waiting for GPU",
      tailscale_up: "Joining the tailnet",
      loading: "Loading the checkpoint",
      warmup: "Warming up the model",
      connecting: "Connecting to the room",
      connected: "In the room",
      claimed: "Driving",
    },
    // The two target pickers. Their OPTIONS are never translated: a profile
    // name, a workspace name and an environment name are identifiers the CLI
    // matches on, and the panel shows them exactly as `modal` reports them.
    profileLabel: "Modal profile",
    environmentLabel: "Environment",
    running: "GPU running. Billing active.",
    // {{profile}}, {{workspace}} and {{environment}} are all DATA — Modal's own
    // names, shown verbatim inside whatever sentence a translator writes.
    billingTo: "Billing to {{profile}}.",
    billingToWorkspace: "Billing to {{profile}} · {{workspace}}.",
    billingEnvironment: "Environment {{environment}}.",
    // {{minutes}} is a plain integer, deliberately not `count`.
    idleStopIn:
      "Auto-stop in {{minutes}} min if unused.",
    idleStopPaused: "Auto-stop pauses during a run.",
    // Drift between the form and the running server. {{fields}} is a list of
    // flag NAMES (engine, horizon, fps, codec, s_min, policy, task) — data.
    // The launched values follow the sentence, verbatim.
    // (model_dtype and gpu join that list since S3.8e: a running container
    // cannot change either — the precision is decided while the weights load,
    // the GPU when the container is created.)
    driftBody:
      "Changed: {{fields}}. Restart GPU to apply. Current settings:",
    restart: "Apply and restart GPU",
    restarting: "Restarting GPU…",
    restartingBody: "Starting a GPU with the new settings.",
    // Shown in the idle state while Start GPU is disabled for an empty task.
    taskRequired:
      "Enter a task before starting the GPU.",
    roomLabel: "Room",
    logLabel: "Log",
  },
  transport: {
    details: "Details",
    // What is left of the retired Transport section: the values a human has to
    // read with their eyes and retype somewhere else (the crib sheet beside the
    // hand-typed `modal run` line), the source label the session dialog's
    // policy line reads, and the one-sentence verdict under Start. Every value
    // behind these labels is data and appears verbatim.
    unresolved: "not set",
    source: {
      sfu: "the Lab's own SFU",
      none: "SFU is off",
    },
    roomLabel: "Room",
    extraMissing:
      "Install the remote extra in the main checkout.",
    sfuModalUrlLabel: "Address for the GPU",
    sfuNoTailnet: "no tailnet address",
    // A summary verdict like the ones below, kept out of `summary` because it
    // outlived the retired Transport section unchanged: it is the one case
    // whose remedy is a command, and the panel prints that command (and the
    // backend's install hint, when there is one) beneath this sentence.
    // "the flags below" is that `<pre>`. See transportSummary.ts.
    sfuNotRunning:
      "Start the Lab with --sfu.",
    // The transport as ONE sentence, chosen by the first thing that is wrong —
    // the order is the order an operator has to fix things in. It stands under
    // Start in place of the generic "not ready" line, so each of these has to
    // say what to DO. See transportSummary.ts.
    summary: {
      // {{error}} is the thrown error's own text — backend prose, verbatim.
      fetchFailed: "Connection check failed: {{error}}",
      checking: "Checking the room…",
      notChecked: "Connection not checked.",
      notConfigured:
        "Connection setup failed. Restart the Lab with --sfu.",
      // {{url}} is the address itself — data.
      unreachable:
        "Cannot reach {{url}}. Check the LiveKit server.",
      notProbed: "Cannot check this connection.",
      // {{room}} is the room NAME — data.
      ready: "GPU ready in {{room}}.",
      gpuStarting: "GPU is starting…",
      gpuWaiting: "Waiting for GPU…",
      operatorAbsent:
        "Start GPU to continue.",
    },
  },
  // Backend phase values. Matched on, never displayed raw — the raw value is
  // the fallback for a phase a newer server introduces.
  phase: {
    idle: "Not running",
    resolving: "Resolving the checkpoint",
    transport_check: "Checking the transport",
    preflight: "Preflight",
    starting: "Starting",
    connecting: "Connecting to the room",
    warming_up: "Waiting for the policy",
    easing: "Easing the arm into position",
    running: "Running",
    stopping: "Stopping",
    stopped: "Stopped",
    error: "Failed",
  },
  outcome: {
    ok: "Finished cleanly",
    failed: "The run failed",
    ran_with_warning: "Finished, with a cleanup warning",
  },
  status: {
    // The session dialog's own copy for a remote run. The pill, the button and
    // the log title are shared with a local run and live under `inference.*`.
    //
    // ONE fixed sentence for the whole setup: the live phase is named on the
    // phase line under the button, and a subtitle that changed with it made the
    // number under the clock the noisiest thing on the screen.
    connectingSubtitle: "Connecting to the GPU & the arm…",
    // A remote run with duration 0 runs until it is stopped. The "/" matches
    // the "/ 01:00" a bounded run shows in the same slot.
    unbounded: "/ ∞",
    unboundedDone: "/ ∞",
    // {{ref}} is the policy ref, {{room}} the room NAME and {{source}} the
    // resolved source label — all three data, shown verbatim.
    policyLine: "policy: {{ref}} · remote · {{room}} on {{source}}",
    policyLineNoRoom: "policy: {{ref}} · remote",
    gpuCardTitle: "Remote GPU",
    // {{profile}} is Modal's own profile name and {{gpu}} is Modal's GPU spec
    // string — both data. They and "billing" are facts about what this costs,
    // said where the operator is watching it run.
    gpuBilling: "Modal · {{profile}} · {{gpu}} · billing",
    // The child writes a log FILE and reports its path; nothing is streamed to
    // the browser, so the log slot holds the path for the operator to open.
    noLogYet: "No log yet.",
    returningToRest:
      "Returning to the starting position.",
    operator: "Operator",
    noOperatorYet: "waiting",
    chunks: "Chunks / requests",
    chunkAge: "Chunk age",
    e2e: "End-to-end p50 / p95",
    rtt: "Round trip",
    holdsRate: "Holds",
    // {{rate}} is a pre-formatted number, deliberately not `count`.
    holdsPerSecond: "{{rate}}/s",
    leadLabel: "Scheduler margin",
    // Both values are plain integers from the child's own sample.
    leadValue: "{{lead}} of {{margin}}",
    degradeHint: "quality is degrading",
    noSampleYet:
      "Waiting for the first sample.",
  },
  toast: {
    startFailed: "Couldn't start the remote run",
    stopFailed: "Couldn't stop the remote run",
    noSession: "No remote run is registered on this server.",
  },
} as const;
