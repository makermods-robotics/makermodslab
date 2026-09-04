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
    // The primary button on a leader-only controller (Teleop's slot): drive
    // a station's robot with this leader.
    drive: "Drive remote",
    // The remote-teleoperation menu beside Teleop: drive a station's robot
    // with this leader. (Hosting is not started from the UI — a station is
    // launched with `makermodslab --sfu --host <robot>`.)
    remote: "Remote",
    remoteTooltip: "Remote teleoperation over the network",
    remoteItem: "Drive a remote robot…",
    remoteItemSub:
      "Use this robot's leader arm to teleoperate a station that is hosting.",
    // The status chip a station in station mode shows while it hosts: the
    // prefix, then the phase, joined by " · ". The phase VALUES are data
    // (remote_host.PHASES); {{operator}} is the seat holder's identity.
    hosting: {
      chip: "Hosting",
      tooltip:
        "This station is hosting {{robot}} for remote teleoperation — open the status view.",
      phase: {
        parked: "Parked",
        engaging: "Engaging…",
        engaged: "Engaged",
        parking: "Parking…",
      },
      engagedBy: "Engaged by {{operator}}",
    },
    // The station-mode chips shown while NOTHING is hosted: no robot chosen
    // yet (opens the hosted-robot picker), or a chosen robot whose hosting is
    // down right now (opens the status view). {{robot}} is data.
    station: {
      chooseChip: "Station · choose a robot to host",
      chooseTooltip:
        "This station has no robot to host yet. Pick a saved robot whose follower arm is set up.",
      idleChip: "Station · {{robot}}",
      idleTooltip:
        "This station hosts {{robot}} whenever nothing local holds the arm — open the status view.",
    },
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
  // Operator side (session kind `remote_teleoperation`). The refusal lines
  // are display twins of the backend's error codes — matched on the code,
  // never on the prose. `makermodslab --sfu` is a CLI literal.
  remote: {
    failedTitle: "Couldn't start remote teleoperation",
    failedFallback: "Failed to start.",
    // {{gap}} is the rendered leader-side setup-gap phrase.
    disabledReason: "{{name}} {{gap}} — open Robot settings",
    installAction: "Install",
    refusal: {
      notHosting:
        "That station isn't hosting. Start it there with makermodslab --sfu --host <robot> first.",
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
      // 409 sfu.seat_taken: the station's single operator seat is held.
      seatTaken:
        "Someone else is driving this robot. Wait for them to stop, then try again.",
    },
  },
  // Station side: changing the hosted robot (PUT /api/v1/station/robot).
  // 409 session.held here means an operator is driving the hosted robot;
  // robot.not_ready / robot.not_found show the server's prose.
  station: {
    failedTitle: "Couldn't change the hosted robot",
    failedFallback: "The station didn't accept the change.",
    refusal: {
      held: "An operator is driving right now — change the hosted robot once they leave.",
    },
  },
  // Layout chips beside a record's name wherever it is listed. The VALUES
  // ("both"/"follower"/"leader", the record's `arms` field) are data; a pair
  // gets no chip at all. A leader-only record is not really a robot — it is a
  // controller — so the chip says so.
  layout: {
    followerOnly: "Robot (follower only)",
    leaderOnly: "Controller (leader only)",
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
