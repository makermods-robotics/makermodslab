export default {
  form: {
    hubIdLabel: "Hub policy id",
    hubIdHint:
      "The repo the GPU container loads. It is not downloaded here — this machine only drives the arm.",
    hubIdInherited: "Left empty, this run's own output repo is used.",
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
        "Plays each action chunk to the end and asks for the next one just in time. Right for any policy, and the only choice for ACT.",
      rtcHint:
        "Sends the moves it has not made yet with every request, so the GPU shapes the next chunk to continue them. Removes the seam between chunks — only flow policies (SmolVLA, π0, π0.5, diffusion) can do this.",
    },
    // The Advanced trigger's summary line. Every value is live and is DATA —
    // the codec id is the wire value, the numbers are the ones that go out.
    // Two whole sentences rather than one plus a fragment: `s_min` only
    // reaches the wire for rtc, so it is only claimed there.
    // {{extra}} is the GPU side's own half of the line — " · A100 · bfloat16",
    // built from identifiers alone, so there is nothing inside it to translate
    // and it is appended rather than concatenated with words.
    advancedSummary:
      "Transport: horizon {{horizon}} · {{fps}} fps · {{codec}}{{extra}}",
    advancedSummaryRtc:
      "Transport: horizon {{horizon}} · {{fps}} fps · {{codec}} · minimum budget {{sMin}}{{extra}}",
    transportGroupHint:
      "These must match the GPU side exactly. A mismatch is not an error: the wire schema is fingerprinted, so mismatched packets are dropped silently and the run looks healthy while receiving nothing.",
    horizonLabel: "Horizon",
    fpsLabel: "Frames per second",
    codecLabel: "Video codec",
    // The checkpoint's own chunk width. {{steps}} is a number read off its
    // config, never a translated value.
    horizonFromCheckpoint:
      "This checkpoint returns {{steps}} steps per chunk, so the horizon starts from that and must not go above it.",
    horizonOverCeiling:
      "The horizon is above the {{steps}} steps this checkpoint returns. The two sides then disagree about the chunk shape and every packet is dropped in silence — the run will look connected and receive nothing.",
    sMinLabel: "Minimum budget",
    // "--s-min" is a flag name, kept in the Latin script like every other
    // identifier in this panel.
    sMinHint:
      "Steps of the plan the arm keeps in hand for the round trip. It must be the same number as --s-min in the command above: the arm works out which part of the next chunk is still fresh from it, and the GPU takes that answer on trust.",
    // The two GPU-side knobs (S3.8e). Nothing here has to match the arm — they
    // decide what the container loads and what it loads onto.
    gpuGroupHint:
      "These two are the GPU's alone: they decide how fast a chunk comes back, and what the hour costs. Changing either needs the GPU restarted.",
    precisionLabel: "Precision",
    // The one option that is prose: it stands for passing no flag at all. The
    // others are torch dtype names — wire values, never translated.
    precisionCheckpoint: "Checkpoint default",
    precisionHint:
      "float32 is what most checkpoints are saved as, and it runs without autocast — the slowest path, and the one that runs out of memory first. bfloat16 is the lever to pull when a chunk takes too long to come back or the container is out of VRAM. Left at the checkpoint default, nothing is overridden.",
    gpuLabel: "GPU",
    gpuHint:
      "The Modal GPU the policy server runs on. Bigger is faster and dearer per hour; it is the second lever after precision, and it is billed either way.",
    // Said where the disabled select is, because the reason belongs to THIS
    // checkpoint. No policy type is named: the fact the operator needs is that
    // this one has no such setting, not which family it belongs to.
    precisionUnavailable:
      "This checkpoint has no precision setting to override — it is loaded the way it was saved.",
    // The flow-steps knob (S3.8f).
    flowStepsLabel: "Flow steps",
    // Prose, like the precision's first option: it stands for passing no flag.
    // The second form carries the number this checkpoint will actually run at,
    // which is data — the server works it out, never this file.
    flowStepsCheckpoint: "Checkpoint default",
    flowStepsCheckpointKnown: "Checkpoint default ({{steps}})",
    flowStepsHint:
      "How many passes the model makes to shape one chunk of movement. Fewer is faster and coarser — the cheapest way to cut the wait, and the first one to cost quality. MolmoAct2 runs 10, and its work on the GPU currently takes about 880 ms against a 777 ms budget at horizon 30 and 30 fps.",
    flowStepsUnavailable:
      "This checkpoint does not build its actions in steps, so there is nothing to shorten.",
  },
  // The per-role camera picker. It appears ONLY for checkpoint cameras that
  // matched nothing by name, so most runs never see it.
  cameraRoles: {
    title: "Camera roles",
    hint: "This checkpoint names cameras that nothing on this robot is called. Choose which camera plays each role for this run.",
    // Roles that bound themselves and need no control.
    nameMatched_one: "{{count}} other camera matched by name.",
    nameMatched_other: "{{count}} other cameras matched by name.",
    // Both numbers are raw pixel dimensions from the checkpoint's config.
    capturesAt: "The policy trained at {{width}}×{{height}}.",
    unbound: "Not chosen",
    noCameras: "This robot has no cameras — add one in Robot settings.",
    disconnected: "Not plugged in right now.",
    identityNote:
      "The choice is remembered for this checkpoint and robot, and is sent with this run only. Nothing is renamed: the camera keeps the name it has in Robot settings, and the server still finds the device by it.",
  },
  // Backend engine values. Matched on, never displayed raw — the raw value is
  // the fallback for an engine a newer server introduces.
  engine: {
    sync: "Adaptive sync",
    rtc: "Real-time chunking",
  },
  modalRun: {
    manualToggle: "Run it yourself instead",
    title: "What the Lab will run",
    intro:
      "The same command, for launching by hand — the only route when the modal command is missing or not signed in, and the line to compare against when a run connects but receives nothing.",
    copy: "Copy",
    copiedTitle: "Command copied",
    copyFailedTitle: "Couldn't copy",
    copyFailedBody: "Select the command and copy it by hand.",
    noRoomYet:
      "No room resolved yet — re-check the transport below, then copy the command again.",
    // <0> is the literal placeholder text and <1> the literal key-file path.
    // Both are identifiers and stay in the Latin script.
    secretsHint:
      "Replace <0>{{placeholder}}</0> with the secret beside that key id in <1>{{path}}</1>. The key id in the line is real; the Lab never sends the secret over its own API.",
    noTailnetUrl:
      "No tailnet address, so the command has no URL for the GPU side to dial. Sign in to Tailscale on this machine and re-check the transport.",
  },
  // The GPU half, which the Lab launches itself since S3.8. It does NOT gate
  // the remote verb — that stays the transport probe's operator check.
  gpu: {
    title: "Policy server on Modal",
    start: "Start GPU",
    retry: "Try again",
    stop: "Stop GPU",
    cancel: "Cancel",
    // {{wrapper}} is the wrapper's PATH and {{gpu}} the Modal GPU spec — both
    // data, shown verbatim. The GPU is interpolated rather than written into
    // the sentence because it is a choice now (S3.8e).
    idleHint:
      "Runs {{wrapper}} on a Modal {{gpu}} from this machine. Cold start is usually 1-3 minutes; the room and the credentials are filled in for you.",
    // {{seconds}} is a plain integer, deliberately not i18next's magic `count`.
    elapsed: "{{seconds}}s",
    // Backend phase values. Matched on, never displayed raw — the raw value is
    // the fallback for a phase a newer server introduces.
    phase: {
      pending: "Starting the container",
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
    running: "GPU running — this is billing.",
    // {{profile}}, {{workspace}} and {{environment}} are all DATA — Modal's own
    // names, shown verbatim inside whatever sentence a translator writes.
    billingTo: "Billing to {{profile}}.",
    billingToWorkspace: "Billing to {{profile}} · {{workspace}}.",
    billingEnvironment: "Environment {{environment}}.",
    // {{minutes}} is a plain integer, deliberately not `count`.
    idleStopIn:
      "It stops itself in about {{minutes}} min if no remote run starts.",
    idleStopPaused: "A remote run is using it, so it won't stop itself.",
    // Drift between the form and the running server. {{fields}} is a list of
    // flag NAMES (engine, horizon, fps, codec, s_min, policy, task) — data.
    // The launched values follow the sentence, verbatim.
    // (model_dtype and gpu join that list since S3.8e: a running container
    // cannot change either — the precision is decided while the weights load,
    // the GPU when the container is created.)
    driftBody:
      "You changed {{fields}} since the GPU was started. A running server keeps the values it was started with, and a mismatch is a run that receives nothing — not an error. It is running:",
    restart: "Restart GPU with these settings",
    // Shown in the idle state while Start GPU is disabled for an empty task.
    taskRequired:
      "Describe the task first — this policy is language-conditioned, and the GPU's policy server refuses to start without one.",
    roomLabel: "Room",
    logLabel: "Log",
  },
  transport: {
    // What is left of the retired Transport section: the values a human has to
    // read with their eyes and retype somewhere else (the crib sheet beside the
    // hand-typed `modal run` line), the source label the session dialog's
    // policy line reads, and the one-sentence verdict under Start. Every value
    // behind these labels is data and appears verbatim.
    unresolved: "not set",
    source: {
      sfu: "the Lab's own SFU",
      cloud: "livekit.env (LiveKit Cloud)",
      process_env: "this process's environment",
      none: "nowhere — nothing is configured",
    },
    roomLabel: "Room",
    extraMissing:
      "The optional drtc extra isn't installed, so nothing could be checked. Install it from the primary checkout — an editable install run from a worktree re-points every other session.",
    sfuModalUrlLabel: "Address for the GPU",
    sfuNoTailnet: "no tailnet address",
    sfuKeyIdLabel: "Key id",
    sfuKeyFileLabel: "Secret is in",
    // A summary verdict like the ones below, kept out of `summary` because it
    // outlived the retired Transport section unchanged: it is the one case
    // whose remedy is a command, and the panel prints that command (and the
    // backend's install hint, when there is one) beneath this sentence.
    // "the flags below" is that `<pre>`. See transportSummary.ts.
    sfuNotRunning:
      "This Lab isn't running a LiveKit server. Start it with the flags below, or leave it off and use LiveKit Cloud credentials from livekit.env.",
    // The transport as ONE sentence, chosen by the first thing that is wrong —
    // the order is the order an operator has to fix things in. It stands under
    // Start in place of the generic "not ready" line, so each of these has to
    // say what to DO. See transportSummary.ts.
    summary: {
      // {{error}} is the thrown error's own text — backend prose, verbatim.
      fetchFailed: "Couldn't read the transport: {{error}}",
      checking: "Checking the room…",
      notChecked: "The room hasn't been checked yet.",
      // {{vars}} is a list of environment variable NAMES — data, verbatim.
      missingVars:
        "No LiveKit credentials: {{vars}} missing. Start the Lab with --sfu, or put Cloud credentials in livekit.env.",
      // {{url}} is the address itself — data.
      unreachable:
        "Nothing is answering at {{url}}. Check the LiveKit server is up and reachable from here.",
      notProbed: "The room could not be checked from here.",
      // {{room}} is the room NAME — data.
      ready: "A GPU is in {{room}}, ready to drive the arm.",
      operatorAbsent:
        "No GPU in {{room}} yet — start one above, or run the command yourself.",
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
    unbounded: "/ ∞ — stops when you do",
    unboundedDone: "/ ∞",
    // {{ref}} is the policy ref, {{room}} the room NAME and {{source}} the
    // resolved source label — all three data, shown verbatim.
    policyLine: "policy: {{ref}} · remote · {{room}} on {{source}}",
    policyLineNoRoom: "policy: {{ref}} · remote",
    gpuCardTitle: "Remote GPU",
    // {{profile}} is Modal's own profile name — data. A100 and "billing" are
    // facts about what this costs, said where the operator is watching it run.
    gpuBilling: "Modal · {{profile}} · A100 · billing",
    // The child writes a log FILE and reports its path; nothing is streamed to
    // the browser, so the log slot holds the path for the operator to open.
    noLogYet: "No log path yet — the run hasn't opened one.",
    returningToRest:
      "Easing the arm back to where it started before letting go.",
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
      "No sample yet — the first one lands a second after connecting.",
  },
  toast: {
    startFailed: "Couldn't start the remote run",
    stopFailed: "Couldn't stop the remote run",
    noSession: "No remote run is registered on this server.",
  },
} as const;
