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
    durationLabel: "Max duration (s)",
    durationHint: "The run stops itself after this long. Stop it early anytime.",
    durationUnbounded: "0 — the run continues until you stop it.",
    sMinLabel: "Minimum budget",
    // "--s-min" is a flag name, kept in the Latin script like every other
    // identifier in this panel.
    sMinHint:
      "Steps of the plan the arm keeps in hand for the round trip. It must be the same number as --s-min in the command above: the arm works out which part of the next chunk is still fresh from it, and the GPU takes that answer on trust.",
  },
  // Backend engine values. Matched on, never displayed raw — the raw value is
  // the fallback for an engine a newer server introduces.
  engine: {
    sync: "Adaptive sync",
    rtc: "Real-time chunking",
  },
  modalRun: {
    title: "Run this in the other terminal",
    intro:
      "The Lab drives the arm and verifies the room; it does not launch the GPU. Start this first and leave it running, then press the remote verb below.",
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
    returningToRest: "Easing the arm back to where it started before letting go.",
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
    noSampleYet: "No sample yet — the first one lands a second after connecting.",
    stop: "Stop the run",
    stopping: "Stopping…",
  },
  toast: {
    startFailed: "Couldn't start the remote run",
    stopFailed: "Couldn't stop the remote run",
    noSession: "No remote run is registered on this server.",
  },
} as const;
