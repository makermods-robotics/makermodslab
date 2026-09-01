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
