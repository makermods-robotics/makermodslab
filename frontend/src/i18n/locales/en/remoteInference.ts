export default {
  intro:
    "The arm runs here; the policy runs on a remote GPU, and the two meet in a LiveKit room. This machine never loads the checkpoint — start the GPU side yourself with the command below.",
  form: {
    hubIdLabel: "Hub policy id",
    hubIdHint:
      "The repo the GPU container loads. It is not downloaded here — this machine only drives the arm.",
    hubIdInherited: "Left empty, this run's own output repo is used.",
    engineLabel: "Chunk engine",
    engine: {
      // Option VALUES ("sync" / "rtc") are backend identifiers — only these
      // labels are translated.
      sync: "Adaptive sync",
      rtc: "Real-time chunking",
      syncHint:
        "Plays each action chunk to the end and asks for the next one just in time. Right for any policy, and the only choice for ACT.",
      rtcHint:
        "Sends the moves it has not made yet with every request, so the GPU shapes the next chunk to continue them. Removes the seam between chunks — only flow policies (SmolVLA, π0, π0.5, diffusion) can do this.",
      rtcUnsupported:
        "This checkpoint isn't a flow policy, so it can't be guided this way — a real-time run would be no better than adaptive sync, and slower to start. Switch back to Adaptive sync to launch.",
    },
    transportGroup: "Transport",
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
    durationLabel: "Max duration (s)",
    durationHint:
      "The run stops itself after this long. Stop it early anytime.",
    durationUnbounded: "0 — the run continues until you stop it.",
    sMinLabel: "Minimum budget",
    // "--s-min" is a flag name, kept in the Latin script like every other
    // identifier in this panel.
    sMinHint:
      "Steps of the plan the arm keeps in hand for the round trip. It must be the same number as --s-min in the command above: the arm works out which part of the next chunk is still fresh from it, and the GPU takes that answer on trust.",
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
    // {{wrapper}} is the wrapper's PATH — data, shown verbatim.
    idleHint:
      "Runs {{wrapper}} on a Modal A100 from this machine. Cold start is usually 1-3 minutes; the room and the credentials are filled in for you.",
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
    title: "Transport",
    refresh: "Re-check",
    checking: "Checking…",
    notCheckedYet: "Not checked yet.",
    unresolved: "not set",
    sourceLabel: "Read from",
    source: {
      sfu: "the Lab's own SFU",
      cloud: "livekit.env (LiveKit Cloud)",
      process_env: "this process's environment",
      none: "nowhere — nothing is configured",
    },
    urlLabel: "URL",
    roomLabel: "Room",
    credentialsLabel: "Credentials",
    configured: "all present",
    // {{vars}} is a list of environment variable NAMES — data, shown verbatim.
    missingVars: "missing {{vars}}",
    reachableLabel: "Endpoint",
    reachable: "answering",
    unreachable: "not answering",
    notProbed: "not checked",
    operatorLabel: "GPU operator",
    operatorPresent: "in the room",
    operatorAbsent: "not in the room",
    extraMissing:
      "The optional drtc extra isn't installed, so nothing could be checked. Install it from the primary checkout — an editable install run from a worktree re-points every other session.",
    sfuRunningTitle: "This machine's LiveKit server",
    sfuModalUrlLabel: "Address for the GPU",
    sfuNoTailnet: "no tailnet address",
    sfuKeyIdLabel: "Key id",
    sfuKeyFileLabel: "Secret is in",
    sfuExternalIpLabel: "Public media address",
    sfuExternalIpOn: "advertised",
    sfuExternalIpOff: "not advertised",
    sfuExternalIpHint:
      "Without it a remote GPU can reach this server to say hello but has no way to send video or actions. Restart the Lab with the flag below to turn it on.",
    sfuNotRunning:
      "This Lab isn't running a LiveKit server. Start it with the flags below, or leave it off and use LiveKit Cloud credentials from livekit.env.",
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
    // {{elapsed}} and {{duration}} are pre-formatted; {{duration}} is "∞" for
    // an unbounded run, so neither may take i18next's magic `count`.
    elapsed: "{{elapsed}}s / {{duration}}",
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
    stop: "Stop the run",
    stopping: "Stopping…",
  },
  toast: {
    startFailed: "Couldn't start the remote run",
    stopFailed: "Couldn't stop the remote run",
    noSession: "No remote run is registered on this server.",
  },
} as const;
