export default {
  corner: {
    create: "Robot",
    createTooltip: "Create robot",
    settings: "Robot settings",
    settingsFor: "Robot settings for {{name}}",
    selectFirst: "Select a robot first",
    // Trailing space is intentional — it sits directly before the robot name.
    activeLabel: "Robot: ",
    selectRobot: "Select a robot",
    setUp: "Set up your robot",
    robots: "Robots",
    mode: {
      single: "single",
      bimanual: "bimanual",
    },
    // Short family tags for the picker rows; the long-form names live in
    // landing.createRobot.armTypes. The VALUES "so101"/"maker"/"metal" are
    // data (robot records on disk) — only these display labels localize.
    armType: {
      so101: "SO-101",
      maker: "Maker",
      metal: "Metal",
    },
    status: {
      ready: "ready",
      needsSetup: "needs setup",
    },
    empty:
      "No robots yet. Create one to get started — you'll set up ports, calibration, and cameras next.",
    createItem: "Create robot…",
    renameItem: "Rename robot…",
    deleteItem: "Delete robot…",
    teleop: "Teleop",
    // The remote-teleoperation menu beside Teleop: host this robot for an
    // operator elsewhere, or drive a station's robot with this leader.
    remote: "Remote",
    remoteTooltip: "Remote teleoperation over the network",
    hostItem: "Available for remote teleop",
    hostItemSub:
      "Publish this robot's follower and cameras for an operator on another node.",
    remoteItem: "Drive a remote robot…",
    remoteItemSub:
      "Use this robot's leader arm to teleoperate a station that is hosting.",
  },
  rename: {
    title: "Rename robot",
    description:
      "Calibration assignments, ports, and cameras move with the robot.",
    newName: "New name",
    submit: "Rename",
    submitting: "Renaming…",
  },
  delete: {
    // {{name}} falls back to `fallbackName` when nothing is selected.
    title: "Delete {{name}}?",
    fallbackName: "robot",
    description:
      "This removes the robot's saved configuration (ports, calibration assignments, cameras). Calibration files themselves stay in the library. This can't be undone.",
    confirm: "Delete robot",
  },
  teleop: {
    startedTitle: "Teleoperation started",
    startedFallback: "Started teleoperation for {{name}}.",
    startedWarningTitle: "Started with a warning",
    failedTitle: "Couldn't start teleoperation",
    failedFallback: "Failed to start.",
    // {{gap}} is the rendered setup-gap phrase below.
    disabledReason: "{{name}} {{gap}} — open Robot settings",
  },
  // Station side of remote teleoperation (session kind `hosting`).
  hosting: {
    startedTitle: "Hosting started",
    startedFallback: "{{name}} is available for remote teleoperation.",
    startedWarningTitle: "Hosting started with a warning",
    failedTitle: "Couldn't start hosting",
    failedFallback: "Failed to start.",
    // {{gap}} is the rendered follower-side setup-gap phrase.
    disabledReason: "{{name}} {{gap}} — open Robot settings",
  },
  // Operator side (session kind `remote_teleoperation`). The refusal lines
  // are display twins of the backend's error codes — matched on the code,
  // never on the prose. `makermodslab --sfu` is a CLI literal.
  remote: {
    failedTitle: "Couldn't start remote teleoperation",
    failedFallback: "Failed to start.",
    disabledReason:
      "{{name}} needs its leader arm set up (port and calibration) — open Robot settings",
    installAction: "Install",
    refusal: {
      notHosting:
        "That station isn't hosting. Ask for “Available for remote teleop” to be pressed there first.",
      nodeNotFound:
        "That station is no longer in the node registry. Refresh the list and pick it again.",
      nodeUnreachable:
        "That station didn't answer. Check that it is online and reachable from this machine.",
      schemaMismatch:
        "This robot doesn't match the hosted one — the arm family or motor set differs.",
      sfuDisabled:
        "This node isn't running the LiveKit SFU. Restart it with makermodslab --sfu.",
      extraMissing:
        "The remote-teleoperation extra isn't installed on this node.",
    },
  },
  // Setup-gap rendering. `robotSetupGaps()` (hooks/useRobots) returns structure;
  // these turn it into a sentence. English output must stay byte-identical to
  // the original hand-built string — see useRobots.setupGap.test.ts.
  setupGap: {
    missingCalibration: "is missing a calibration for the {{arms}}",
    noPort: "has no port assigned for the {{arms}}",
    stale:
      "references a calibration file that no longer exists — reassign or recalibrate",
    // Joins the two clauses above when a robot has both problems.
    clauseJoin: " and ",
    // Joins arm labels inside one clause.
    armJoin: " and ",
    armList_one: "{{arms}} arm",
    armList_other: "{{arms}} arms",
  },
  arm: {
    leader: "leader",
    follower: "follower",
    leftLeader: "left leader",
    leftFollower: "left follower",
    rightLeader: "right leader",
    rightFollower: "right follower",
  },
} as const;
