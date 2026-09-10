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
  gripperCurrentLimit: {
    title: "Experimental gripper current limit",
    enabled: "Enabled",
    disabled: "Disabled (stock behavior)",
    description: "Limits motor current during position control; startup behavior and grip force have not been validated on hardware. Can be combined with the soft squeeze limit.",
    percent: "Motor current cap (%)",
    velocity: "Maximum gripper speed (degrees/sec)",
    invalidCurrent: "Enter a finite value from 0.01 to 100%. No safe current cap is assumed.",
    currentHelp: "Percentage of the motor's current scale. This is not a jaw-force percentage.",
    invalidVelocity: "Enter a finite speed within the motor protocol's range (about 0.573–5729.578 degrees/sec). No safe speed is assumed.",
    velocityHelp: "Required with the current cap. Choose a speed for your gripper and object.",
    requirements: "Requires supported DM4310 firmware, checked at session start. Changes the motor control mode for the next local session; the prior mode is restored after the motor is disabled.",
    scope: "Applies from the next local teleoperation or recording session only.",
    scopeBimanual: "Applies to both grippers from the next local teleoperation or recording session only.",
  },

  gripperSoftLimit: {
    title: "Gripper soft squeeze limit",
    enabled: "Enabled",
    disabled: "Disabled (stock behavior)",
    description: "Caps how far the gripper target can push past its measured position while closing. Smaller values reduce sustained squeeze; opening remains responsive. This is not a force or torque guarantee.",
    degrees: "Maximum closing error (degrees)",
    invalid: "Enter a finite number greater than 0. No safe limit is assumed.",
    help: "Smaller values reduce sustained squeeze. Choose a value for your gripper and object.",
    scope: "Applies from the next local teleoperation or recording session only.",
    scopeBimanual: "Applies to both grippers from the next local teleoperation or recording session only.",
  },

  // ---- Window chrome + footer -------------------------------------------
  window: {
    // Rendered uppercase by the `.eyebrow` class (a no-op on Chinese).
    eyebrow: "ports · calibration · cameras · motor power",
    // {{name}} is the robot's own name — rendered verbatim, never translated.
    title: "Robot settings — {{name}}",
    srDescription:
      "Configure ports, calibration, cameras, and motor power for {{name}}.",
    // Shown once at the top when no installed arm family answers to the
    // record's arm_type. {{armType}} is the raw id — data, rendered verbatim.
    armUnavailable:
      'This robot\'s arm type "{{armType}}" is not installed. Install the extension that provides it, or delete this robot and create it again with an installed arm type — port detection and calibration are disabled until then.',
    // The manifest (GET /api/v1/arms) has not answered; ArmsProvider retries.
    armsNotLoaded:
      "Arm types have not loaded from the server yet — retrying. Port detection and calibration are held until they do.",
    unsaved: "Unsaved changes",
    // The footer tooltip explains which setup step is missing.
    savedWithGap: "Saved. Setup incomplete.",
    allSaved: "Saved",
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
        "You have unsaved configuration changes (ports, cameras, motor torque, or gripper settings). Closing now discards them — nothing was written to the robot. Save first to keep them.",
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

  // ---- Arm layout --------------------------------------------------------
  // "What's plugged into this machine?" — the record's `arms` field. The
  // VALUES ("both"/"follower"/"leader") are data; a leader-only record is a
  // controller, not a robot, and the option says so.
  layout: {
    question: "What's plugged into this machine?",
    both: "Leader and follower",
    followerOnly: "Follower only — a robot station",
    leaderOnly: "Leader only — a controller for a remote robot",
  },

  // ---- 01 · Device -------------------------------------------------------
  device: {
    step: "Device",
    label: "Device",
    // aria-labels for the radiogroup; the bimanual grid also picks an arm.
    groupBimanual: "Device and arm",
    groupSingle: "Device",
    // Row headings on a bimanual robot, one per side.
    left: "Left",
    right: "Right",
  },

  // ---- Leader kind ---------------------------------------------------------
  // Which of the family's leaders drives the robot; rendered only for a
  // family whose manifest entry lists more than one (the Metal arm: its Star
  // Arm 102, or a second gravity-compensated Metal arm). The option VALUES
  // are manifest ids sent as the record's `leader_kind` — data; only the
  // per-id labels under optionFor are localized (a kind without an entry
  // renders the manifest's own English label). The "unavailable" reason is
  // server prose and renders as-is.
  leaderKind: {
    label: "Leader arm",
    unavailable: "not installed",
    energizedHint:
      "This leader supports its own weight. Use Wiggle to find its port.",
    optionFor: {
      metal: {
        star: "Star Arm 102 leader",
        metal: "Metal arm leader (gravity-compensated)",
      },
    },
    toast: {
      savedTitle: "Leader arm saved",
      saveFailedTitle: "Could not change the leader arm",
    },
  },

  slotCard: {
    // aria-label and title differ on purpose: the title adds the fix.
    undetectedLabel: "Port not detected",
    undetectedTitle: "Saved port not detected. Plug in the arm and rescan.",
    noPort: "no port assigned",
    readyLabel: "Ready",
  },

  // ---- Port picker, Detect, Wiggle ---------------------------------------
  port: {
    label: "Port",
    select: "Select a port",
    none: "No arms detected. Plug in and rescan.",
    // Badge on a port another arm already holds. Uppercased in English only.
    otherArm: "other arm",
    // aria-label and title differ: the title explains what clearing does.
    clear: "Clear port",
    clearTitle: "Clear port — release it without assigning another",
    // aria-label and title are identical here, so they share one key.
    rescan: "Refresh",
    detect: "Detect by swing",
    detecting: "Watching…",
    detectTitle:
      "Identify by hand: swing the arm's base wide, both left and right",
    detectHelp:
      "Identify by hand — swing the arm's base wide to the left AND the right (10–15° past where it started, each way); the port that moves is assigned. Small wiggles won't register.",
    detectLive:
      "Swing the base wide, past where it started, both ways. Small or one-sided motion is ignored.",
    // The probe families' twin of detectLive: their follower and leader
    // answer different protocols (CAN vs UART), so detection asks each port
    // what it is instead of asking the user to move something. Only a
    // bimanual rig — two identical arms per side — still needs the gesture
    // to tell left from right.
    detectHelpMaker:
      "Detected automatically — the Maker arm's follower and leader answer different protocols, so no gesture is needed. On a bimanual rig, swing one arm's base left AND right to say which side it is.",
    detectHelpMetal:
      "Detected automatically — the Metal arm's follower and leader answer different protocols, so no gesture is needed. On a bimanual rig, swing one arm's base left AND right to say which side it is.",
    // Generic live text for ANY family whose follower answers a protocol
    // probe (supports_port_probe in the arms manifest); detectLive above is
    // the generic gesture text. The per-id entries in detectLiveFor override
    // either for a family that wants its own wording.
    detectLiveProbe:
      "Checking each port. If multiple arms answer, select a port and use Swing for an unpowered leader or Wiggle for a powered arm.",
    // Keyed by manifest id — the VALUES are data. An id without an entry
    // falls back to detectLiveProbe / detectLive by its detection mechanism.
    detectLiveFor: {
      maker:
        "Checking each port. If multiple arms answer, select a port and use Swing for an unpowered leader or Wiggle for a powered arm.",
      metal:
        "Checking each port. If multiple arms answer, select a port and use Swing for an unpowered leader or Wiggle for a powered arm.",
    },
    multipleHelp:
      "Identify leaders by swinging their base left and right. For followers, select a port and press Wiggle to see which gripper moves. The gripper returns to its starting position.",
    wiggle: "Wiggle",
    wiggling: "Wiggling…",
    wiggleTitle: "Move the gripper on this port to see which arm it is",
    wiggleHelp:
      "Confirms an arm is on this port — briefly drives its gripper so you can see which arm responds.",
    // Short forms for the icon tooltips that replaced the help paragraphs.
    // Kept to one or two clauses: a tooltip is read at a glance, and the long
    // detectHelp/wiggleHelp text above is still what the live panel shows.
    detectTip:
      "Swing the base wide, left and right. Small wiggles are ignored.",
    // The CAN arms answer a protocol probe instead, so there is no gesture to
    // describe — this only appears on a single-arm CAN robot.
    detectAuto: "Auto detect",
    detectTipAuto: "Probes each port. No gesture needed.",
    wiggleTip: "Moves the gripper, then returns it to its starting position.",
    // Appended to a failed Detect when the server names the gripper wiggle
    // as the identification of last resort (two Damiao arms on one rig:
    // the probe cannot tell them apart and the gesture would energize them).
    wiggleFallback:
      "Use Wiggle on a port instead: it moves only that arm's gripper, so you can see which arm it is and assign the port by hand.",
    // The blank first row of every port dropdown, and what an empty slot's
    // trigger shows. Selecting it clears the port.
    noneAssigned: "No port",
    forSlot: "Port for {{slot}}",
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
    leadDetect:
      "Detected <0>{{port}}</0> — assign it to the <1>{{target}}</1>?",
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
    step: "Calibration",
    calibrateAll: "Calibrate all",
    calibrateAllTitle: "Select every detected arm for auto-calibration",
    calibrateAllZeroTitle: "Set the zero pose for each detected arm",
    calibrateAllDisabledTitle: "No arms detected. Plug in an arm and rescan.",
    // aria-label and title are identical on both folder buttons.
    openLeaderFolder: "Open leader calibrations folder",
    openFollowerFolder: "Open follower calibrations folder",
    leader: "Leader",
    follower: "Follower",
    calibrate: "Calibrate",
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
      // Zero-pose flow (CAN arms) only.
      awaitingZero: "Waiting for zero pose",
      saving: "Saving calibration",
      completed: "Completed",
      error: "Error",
      stopping: "Stopping",
      unknown: "Unknown",
    },
    // The zero-pose calibration. It has no range sweep — the arm's joint
    // limits are fixed constants — so the whole flow is: torque off, pose the
    // arm by hand, confirm. Follower arm poses differ by family; both
    // grippers are fully closed. The shared Star Arm
    // 102 leader has one folded, closed-gripper pose on both rigs.
    zeroPose: {
      sequence_one: "Set each arm's zero pose. {{count}} arm remaining.",
      sequence_other: "Set each arm's zero pose. {{count}} arms remaining.",
      cancelAll: "Cancel all",
      poseCaption: "Match the pose shown above. Keep the gripper fully closed.",
      // Keyed by manifest id, then side — the VALUES are data. These are
      // per-id overrides of the manifest's own calibration.summary text
      // (which the server's calibration_summary() writes, and which a family
      // without an entry here renders as-is, in English) and of the first
      // step's text while the wizard runs. Wording mirrors it.
      instructionsFor: {
        maker: {
          leader:
            "Move the Star Arm 102 leader by hand to match the pose above: folded against the base, gripper fully closed. Its joints are unpowered, so the arm moves freely.",
          follower:
            "Move the arm by hand to match the pose above: folded against the base, gripper fully closed. Torque is off, so the arm moves freely.",
        },
        metal: {
          leader:
            "Move the Star Arm 102 leader by hand to match the pose above: folded against the base, gripper fully closed. Its joints are unpowered, so the arm moves freely.",
          // The Metal arm driven by a second Metal arm (leader_kind
          // "metal"): the leader IS a Metal arm, so its zero pose is the
          // follower's. Keyed `leader_<kind>`; the default leader keeps
          // the bare `leader` key above.
          leader_metal:
            "Move the leader Metal arm by hand to match the pose above: standing upright, all joints at 0°, gripper fully closed. Torque is off, so the arm moves freely.",
          follower:
            "Move the arm by hand to match the pose above: standing upright, all joints at 0°, gripper fully closed. Torque is off, so the arm moves freely.",
        },
      },
      // Caption on the reference-pose slot when a NON-default leader is
      // being calibrated (keyed by manifest id, then leader kind): the photo
      // shown is the family's follower photo, because that leader is the
      // family's own arm.
      leaderPoseImageFor: {
        metal: {
          metal: "Leader Metal arm zero pose: upright, gripper fully closed",
        },
      },
      liveAngles: "Live joint angles",
      start: "Set zero pose",
      confirm: "Set zero and save",
      saving: "Setting zero and saving the calibration…",
      // Caption on the reference-pose slot. The picture is what the user
      // matches the real arm against, so followers name their family-specific
      // pose and the shared Star leader names its own.
      poseImage: "Zero pose: folded, gripper fully closed",
      poseImageLeader:
        "Star Arm 102 leader zero pose: folded, gripper fully closed",
      poseImageMetal: "Zero pose: upright, gripper fully closed",
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
    // The one-column flow: mode first, then video, pose, advanced, Start.
    start: "Start",
    // Combined instruction + warning shown while sweeping. One alert, not
    // two: what to do and why the arm is limp belong in the same breath.
    sweepNote:
      "Move every joint through its full range, both ways. Torque is off, so the arm is limp. Keep it supported.",
    autoNote:
      "Each arm moves on its own to find its joint limits. Keep the area clear.",
    zeroNote:
      "Put the arm in the position shown above, then set zero. Torque stays off, so it moves freely.",
    // A family that calibrates through its extension's own served page
    // (manifest calibration.kind "panel"). Shown in place of the calibrate
    // controls until the panel is mounted here.
    panel: {
      notice:
        "This arm is calibrated through its extension's own panel. It will appear here once the extension provides it.",
    },
    videoAuto: "Auto-calibration demo",
    poseMiddle: "Start pose: middle position",
    poseAutoStart: "Start pose for auto-calibration",
    // Captions under the two start-pose photos. They carry the one thing the
    // picture cannot: that the arm has to be put there BEFORE Start.
    restingPoseCaption:
      "This is the SO-101's resting position — the pose auto-calibration starts from. Put the arm in it before you press Start.",
    middlePoseCaption:
      "Put the arm in this middle position — every joint near the centre of its range — before you press Start.",
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
      failedFallback: "Calibration failed. Try again.",
      notResponding: "Arm not responding. Check power and cable.",
      jointsNotResponding:
        "No response from {{joints}}. Check power and wiring.",
      disconnected: "Arm disconnected. Reconnect it and retry.",
      portDenied: "Port access denied. Check permissions or close other apps.",
      motorFault: "Motor reported a fault. Check the arm before retrying.",
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
    // Every arm with a detected port is ticked already (the user pressed
    // "Calibrate all"), so the only thing left to say is how to opt one out.
    pickerHint: "Untick any arm you want to skip. They run at the same time.",
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
      titleSingle: "Auto-calibrate {{arm}}",
      titleFallbackArm: "this arm",
      titleMulti: "Auto-calibrate {{count}} arms",
      // <0> bolds the "moves on its own" safety warning, which is the whole
      // point of the dialog. Everything else got cut: what the run does to
      // the existing calibration is recoverable and reads fine after the
      // fact, but an arm swinging at an unprepared bench does not.
      bodySingle:
        "The arm will <0>move on its own</0>. Put it at its resting position and clear the workspace.",
      bodyMulti:
        "{{count}} arms will <0>move on their own</0>. Put them at their resting position and clear the workspace.",
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
    step: "Cameras",
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
