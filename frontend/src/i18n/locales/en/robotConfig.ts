/**
 * "robotConfig" namespace — the Robot settings window
 * (components/dialogs/RobotConfigDialog.tsx): ports, calibration files, the
 * manual + auto calibration flow, cameras, and the auto-calibration torque.
 *
 * Key tree must match the other language exactly (see i18n/catalogs.test.ts).
 *
 * NOT in here, deliberately — these are DATA, not copy, and are rendered
 * verbatim: robot names, serial port paths (`/dev/tty.*`), calibration file
 * names, camera names, the `device_type`/`arm` values sent to the backend, and
 * every `message`/`error` string the server sends back.
 */
export default {
  // ---- Window chrome + footer -------------------------------------------
  window: {
    // Rendered uppercase by the `.eyebrow` class (a no-op on Chinese).
    eyebrow: "ports · calibration · cameras · motor power",
    // {{name}} is the robot's own name — rendered verbatim, never translated.
    title: "Robot settings — {{name}}",
    srDescription:
      "Configure ports, calibration, cameras, and motor power for {{name}}.",
    unsaved: "Unsaved changes",
    // {{gap}} is the setup-gap predicate rendered from the `robot.setupGap.*`
    // keys, so this line reads as one sentence in both languages.
    savedWithGap: "Saved — but this robot {{gap}}",
    allSaved: "All changes saved",
    quit: "Quit",
    save: "Save",
    saving: "Saving…",
    // Transient post-save acknowledgment; the ✓ is decoration, keep it.
    justSaved: "Saved ✓",
    toast: {
      saved: "Changes saved",
      saveFailedTitle: "Couldn't save changes",
      // Client-side fallback only — the backend's own message wins when sent.
      saveFailedFallback: "Failed to save the configuration.",
    },
    discard: {
      title: "Discard unsaved changes?",
      description:
        "You have unsaved configuration changes (ports, cameras, or motor torque). Closing now discards them — nothing was written to the robot. Save first to keep them.",
      cancel: "Keep editing",
      confirm: "Discard & quit",
    },
    abort: {
      title: "Abort calibration?",
      description:
        "A manual calibration is running. Closing aborts it — nothing will be saved and the arm is released (it's limp; keep it supported).",
      cancel: "Keep calibrating",
      confirm: "Abort & close",
    },
    // Native window.confirm() text for the back-button / tab-close guard.
    leaveConfirm:
      "Leaving aborts this calibration — nothing will be saved and the arm is released. Continue?",
  },

  // ---- Arm slot labels ---------------------------------------------------
  // The four (device, arm) pairs, as shown on the Device cards, in toasts, and
  // in the port-assignment dialog. Display only: the wire payload always
  // carries `device_type` + `arm`, never these labels.
  arm: {
    leader: "Leader",
    follower: "Follower",
    leftLeader: "Left Leader",
    leftFollower: "Left Follower",
    rightLeader: "Right Leader",
    rightFollower: "Right Follower",
  },

  // The backend `device_type` enum rendered as a word inside a sentence. The
  // VALUE ("teleop"/"robot") is unchanged; only this label is localized, and
  // an unmapped value falls back to the raw string.
  deviceValue: {
    teleop: "teleop",
    robot: "robot",
  },

  // ---- 01 · Device -------------------------------------------------------
  device: {
    step: "Device",
    label: "Device",
    // aria-labels for the radiogroup; the bimanual grid also picks an arm.
    groupBimanual: "Device and arm",
    groupSingle: "Device",
  },

  slotCard: {
    // aria-label and title differ on purpose: the title adds the fix.
    undetectedLabel: "Saved port not currently detected",
    undetectedTitle:
      "Saved port not currently detected — plug in the arm and rescan",
    noPort: "no port assigned",
  },

  // ---- Port picker, Detect, Wiggle ---------------------------------------
  port: {
    label: "Port",
    select: "Select a port",
    none: "No arms detected — plug in & refresh",
    // Badge on a port another arm already holds. Uppercased in English only.
    otherArm: "other arm",
    // aria-label and title differ: the title explains what clearing does.
    clear: "Clear port",
    clearTitle: "Clear port — release it without assigning another",
    // aria-label and title are identical here, so they share one key.
    rescan: "Rescan ports",
    detect: "Detect",
    detecting: "Watching…",
    detectTitle:
      "Identify by hand: swing the arm's base wide, both left and right",
    detectHelp:
      "Identify by hand — swing the arm's base wide to the left AND the right (10–15° past where it started, each way); the port that moves is assigned. Small wiggles won't register.",
    detectLive:
      "Swing the base of the arm wide — clearly past its starting point both left and right. A small or one-sided wiggle is ignored (that's how bumps are filtered out). The port that sees the motion will be assigned to this arm.",
    wiggle: "Wiggle",
    wiggling: "Wiggling…",
    wiggleTitle: "Move the gripper on this port to see which arm it is",
    wiggleHelp:
      "Confirms an arm is on this port — briefly drives its gripper so you can see which arm responds.",
    toast: {
      missingPortTitle: "Missing port",
      missingPortWiggle:
        "Enter or detect the port first, then wiggle to confirm the arm.",
      wiggleStartedTitle: "Wiggling gripper",
      wiggleFailedTitle: "Wiggle failed",
      noArmTitle: "No arm detected",
      detectFailedTitle: "Detect failed",
      swappedDetectedTitle: "Arm identified — ports swapped",
      swappedTitle: "Ports swapped",
      // {{port}}/{{swapPort}} are device paths, {{released}} an arm label.
      // On the Detect path the backend's own message is prefixed ahead of
      // this sentence rather than folded into it.
      swappedDescription:
        "{{port}} is now this arm's; the {{released}} took {{swapPort}}.",
      movedDetectedTitle: "Arm identified — port moved",
      movedTitle: "Port moved",
      movedDescription:
        "{{port}} was assigned to the {{released}}; moved it here. The {{released}} now needs a port.",
      identifiedTitle: "Arm identified",
      assignedTitle: "Port assigned",
      // Follows the backend's detect message, which already names the port.
      identifiedDescription: "Port assigned to this arm.",
      assignedDescription: "{{port}} assigned to this arm.",
    },
  },

  // ---- Port assignment confirmation --------------------------------------
  portAssign: {
    swapTitle: "Swap ports?",
    detectTitle: "Assign detected port?",
    assignTitle: "Assign port?",
    // <0> is the monospaced port path, <1> the bold target arm label.
    leadDetect: "Detected <0>{{port}}</0> — assign it to the <1>{{target}}</1>?",
    leadAssign: "Assign <0>{{port}}</0> to the <1>{{target}}</1>?",
    // <0> and <1> both bold the same arm label; <2> is the mono port path.
    swapClause:
      "It's currently assigned to the <0>{{released}}</0>; confirming swaps them — the <1>{{released}}</1> takes this arm's current port <2>{{swapPort}}</2> in exchange, so neither arm is left without a port.",
    takeClause:
      "It's currently assigned to the <0>{{released}}</0>; this arm has no port to swap back, so confirming moves it here and leaves the <1>{{released}}</1> without a port.",
    confirmSwap: "Swap ports",
    confirmMove: "Move & assign",
    confirmAssign: "Assign port",
  },

  // ---- 02 · Calibration files -------------------------------------------
  files: {
    step: "Calibration files",
    calibrateAll: "Calibrate all",
    calibrateAllTitle: "Select every detected arm for auto-calibration",
    calibrateAllDisabledTitle: "No arms detected — plug in an arm and rescan",
    // aria-label and title are identical on both folder buttons.
    openLeaderFolder: "Open leader calibrations folder",
    openFollowerFolder: "Open follower calibrations folder",
    leader: "Leader",
    follower: "Follower",
    newCalibration: "New calibration",
    newCalibrationTitle: "Create a new calibration for this arm",
    // Row labels. The parenthetical names LeRobot's device class for the slot
    // (teleoperator = leader side, robot = follower side).
    row: {
      leader: "Leader (Teleoperator)",
      follower: "Follower (Robot)",
      leftLeader: "Left Leader (Teleoperator)",
      leftFollower: "Left Follower (Robot)",
      rightLeader: "Right Leader (Teleoperator)",
      rightFollower: "Right Follower (Robot)",
    },
    toast: {
      openFolderFailedTitle: "Couldn't open folder",
    },
  },

  // ---- The "New calibration" panel ---------------------------------------
  calib: {
    // {{row}} is the calibration-file row this panel is expanded under.
    panelTitle: "New calibration — {{row}}",
    // Backend `status` enum → label. The value itself is never translated;
    // an unmapped status falls back to `unknown`, as it did before i18n.
    status: {
      idle: "Idle",
      connecting: "Connecting",
      recording: "Recording ranges",
      completed: "Completed",
      error: "Error",
      stopping: "Stopping",
      unknown: "Unknown",
    },
    cancel: "Cancel calibration",
    auto: "Auto-calibrate",
    // {{arm}} is an arm label, {{port}} a device path.
    autoTitle:
      "Auto-calibrate {{arm}} on {{port}} — the arm will move on its own",
    autoDisabledTitle:
      "No detected port for this arm — assign or reconnect it above",
    manual: "Calibrate manually",
    torqueOffWarning:
      "Motor torque is off — the arm won't hold its pose during calibration, and stays limp after you cancel or finish. Keep it low and supported so it can't drop onto the table edge.",
    connecting: "Connecting to the device. Please ensure it's connected.",
    liveData: "Live position data",
    rangeComplete: "Range complete",
    save: "Save calibration",
    // <0> bolds "Important:", <1> bolds the wrist-roll exception.
    rangeHint:
      "<0>Important:</0> Move each joint through its full range — <1>except the wrist roll</1>: leave it near the middle. It rotates continuously and its range is set automatically. A check appears next to each joint once its range is wide enough.",
    completed: "Calibration completed successfully!",
    // Heading shown when the backend error starts with the discontinuity
    // prefix. The PREFIX the code matches on stays English — it is wire text.
    discontinuityTitle: "Motor discontinuity detected",
    discontinuityBody:
      "Make sure to start the calibration with the robot in a middle position — all joints in the middle of their ranges. See the calibration demo beside for the correct starting pose.",
    // Label in front of the backend's raw error text, which stays as sent.
    errorLabel: "Error:",
    demoTitle: "Calibration demo",
    videoUnsupported: "Your browser does not support the video tag.",
    videoLink: "Click here to view the calibration video",
    toast: {
      noRobotTitle: "No robot selected",
      // The ⚙ names the gear button in the robot corner.
      noRobotDescription:
        "Open Robot settings from the robot menu (⚙ Robot settings).",
      missingPortTitle: "Missing port",
      missingPortDescription: "Set the device's serial port before starting.",
      startedTitle: "Calibration Started",
      // {{device}} is the localized `deviceValue.*` label.
      startedDescription: "Calibration started for {{device}}",
      startFailedTitle: "Calibration Failed",
      startFailedFallback: "Failed to start calibration",
      errorTitle: "Error",
      startError: "Failed to start calibration",
      stoppedTitle: "Calibration Stopped",
      stoppedDescription: "Calibration has been stopped",
      stopFailedFallback: "Failed to stop calibration",
      stepCompletedTitle: "Step Completed",
      stepFailedTitle: "Step Failed",
      stepFailedFallback: "Could not complete step",
      stepError: "Could not complete calibration step",
    },
  },

  // ---- Concurrent multi-arm auto-calibration -----------------------------
  batch: {
    // Single vs multi are different sentences, not one plural form, so they
    // stay separate keys in every language.
    titleSingle: "Auto-calibration",
    titleMulti: "Multi-arm auto-calibration",
    stopSingle: "Stop auto-calibration",
    stopAll: "Stop all auto-calibration",
    // <0> bolds "at the same time" — the point of the batch.
    pickerIntro:
      "Pick the arms to calibrate. Each runs its own hands-off calibration <0>at the same time</0> on its assigned port — one arm failing doesn't stop the others. Ports come from each arm's assignment above; an arm with no port yet can't be picked. Each arm replaces its own existing calibration; rename any of them afterward from the calibration list above.",
    portUndetected: "port not detected",
    portMissing: "no port — assign above",
    start_one: "Auto-calibrate {{count}} arm",
    start_other: "Auto-calibrate {{count}} arms",
    // {{done}} of {{total}}; the plural is on how many arms are moving.
    progress_one:
      "{{done}} of {{total}} done — the arm is moving. Keep the workspace clear.",
    progress_other:
      "{{done}} of {{total}} done — the arms are moving. Keep the workspace clear.",
    // Per-arm terminal/running state from the backend. Glyphs are decoration.
    armStatus: {
      completed: "✓ done",
      failed: "✗ failed",
      stopped: "stopped",
      running: "running…",
    },
    // Shared by the results panel and the finished-with-issues toast.
    summary: "{{completed}} completed, {{failed}} failed/stopped.",
    dismiss: "Dismiss",
    prompt: {
      // Singular names the arm, plural counts them — different sentences.
      titleSingle: "Auto-calibrate {{arm}} — it will move",
      titleFallbackArm: "this arm",
      titleMulti: "Auto-calibrate multiple arms — they will move",
      // <0> bolds the "moves under power" safety warning.
      bodySingle:
        "This arm will <0>move on its own under power</0> to find each joint's range. Clear the workspace and keep hands away from it. It replaces its own existing calibration.",
      bodyMulti:
        "{{count}} arms will <0>move on their own under power</0> at the same time to find each joint's range. Clear the workspace and keep hands away from every arm. Each arm replaces its own existing calibration.",
      confirm: "Start auto-calibration",
    },
    toast: {
      noArmsTitle: "No arms selected",
      noArmsDescription: "Tick at least one arm to auto-calibrate.",
      noPortTitle: "Arm has no detected port",
      // {{arm}} is an arm label.
      noPortDescription:
        "{{arm}} has no port that's currently plugged in — assign/reconnect it above before starting.",
      duplicatePortTitle: "Duplicate port",
      duplicatePortDescription: "Each arm needs its own serial port.",
      startedTitle_one: "Auto-calibration started on {{count}} arm",
      startedTitle_other: "Auto-calibration started on {{count}} arms",
      startedDescription: "The arms are moving — keep the workspace clear.",
      startFailedTitle: "Couldn't start auto-calibration",
      finishedTitle_one: "Auto-calibrated {{count}} arm",
      finishedTitle_other: "Auto-calibrated {{count}} arms",
      issuesTitle: "Batch auto-calibration finished with issues",
    },
  },

  // ---- Advanced parameters (auto-calibration torque) ---------------------
  advanced: {
    title: "Advanced parameters",
    subtitle: "Auto-calibration torque",
    torqueLabel: "Auto-calibration torque",
    // "Torque_Limit" is the servo register name — never translated.
    torqueSliderLabel:
      "Auto-calibration torque (Torque_Limit register, 0-1000 scale)",
    // <0> wraps the register name in <code>. {{ref}}/{{min}} are raw units.
    torqueHint:
      "Raw servo <0>Torque_Limit</0> (tick = stock {{ref}}) — lower is gentler; below {{min}} the arm can't lift itself.",
  },

  // ---- 03 · Attached cameras ---------------------------------------------
  cameras: {
    step: "Attached cameras",
    on: "On",
    off: "Off",
    toggleLabel: "Turn cameras on or off",
    offTitle: "Cameras are off",
    offDescription:
      "Turn cameras on to scan for connected devices and preview them. The browser may briefly open a camera to read device labels, and configured cameras stay active while previews are visible; your browser will ask for camera permission. Nothing is recorded.",
    saved_one: "{{count}} camera saved to this robot.",
    saved_other: "{{count}} cameras saved to this robot.",
    permissionHint: "You'll be asked to grant camera access.",
  },
} as const;
