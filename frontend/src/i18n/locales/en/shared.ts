export default {
  logPanel: {
    // Fallback heading; callers usually pass their own ("Recording log", …).
    defaultTitle: "Log",
    waiting: "Waiting for log output…",
  },
  // 409 session.held from POST /api/v1/sessions — rendered by
  // formatSessionHeld (lib/sessionApi.ts) in every flow's start-error path.
  // The activity labels are display twins of the backend's session-kind enum
  // (the values themselves are data, matched on and never translated).
  sessionBusy: {
    message: "The robot is busy — {{activity}} is running. Stop it first.",
    generic: "The robot is busy with another session. Stop it first.",
    activity: {
      teleoperation: "teleoperation",
      recording: "a recording session",
      inference: "an inference run",
      replay: "an episode replay",
      calibration: "a calibration",
      auto_calibration: "an auto-calibration",
      wiggle: "a gripper wiggle",
    },
  },
  update: {
    title: "MakerMods Lab update available",
    behind_one: "{{count}} commit behind",
    behind_other: "{{count}} commits behind",
    available: "A new version is available",
    // <1> is the "See what changed" link; it is omitted when the backend
    // sends no compare URL, so the sentence must read correctly without it.
    body: "You're {{behind}} 😱.",
    bodyLine2: "Update to get the latest fixes and features 🤗.",
    seeChanges: "See what changed",
    manual: "Or update manually",
    copyCommand: "Copy command",
    copiedTitle: "Copied",
    copiedDescription: "Update command copied to clipboard.",
    copyFailedTitle: "Copy failed",
    copyFailedDescription: "Select and copy the command manually.",
    updatedTitle: "Updated",
    failedTitle: "Update failed",
    dontAsk: "Don't ask me again",
    later: "Later",
    now: "Update now",
    updating: "Updating…",
  },
  mockHub: {
    banner:
      "MOCK HUB DATA — jobs, models, and Hugging Face auth on this page are fake (dev fixture).",
    turnOff: "Turn off",
  },
  camera: {
    retry: "Retry camera feeds",
    retryTitle: "Retry camera feeds (e.g. after reconnecting a camera)",
    loadingRobot: "Loading robot...",
    none: "No cameras configured for this robot. Add them during calibration to see live feeds here.",
    previewFailed: "Preview failed",
    clickToRetry: "Click to retry now",
    previewAlt: "Server camera preview",
  },
  visualizer: {
    heading: "Teleoperation",
    done: "Done",
    leftArm: "Left arm",
    rightArm: "Right arm",
    // Shown in the 3D viewer's place on the Metal arm, which has no URDF yet
    // (the SO-101 and Maker arm both drive the model).
    jointAngles: "Live joint angles",
    waitingForJoints: "Waiting for joint data…",
    // Kept arm-neutral: only the Metal arm falls back to the readout today,
    // but a future arm type without a URDF would land here too.
    noModel: "No 3D model is available for this arm yet.",
  },
  urdf: {
    switchedDefaultTitle: "Switched to default model",
    switchedDefaultDescription: "The default SO-101 robot model is now displayed.",
    loadingTitle: "Loading Urdf model...",
    loadingDescription: "Preparing 3D visualization",
    loadedTitle: "Urdf model loaded successfully",
    loadErrorTitle: "Error loading Urdf",
    processErrorTitle: "Error processing files",
    noUrdfTitle: "No Urdf file found",
    noUrdfDescription: "Please upload a folder containing a .urdf file.",
    // {{name}} is a user-supplied model name — data, rendered verbatim.
    loadingNamed: "Loading model: {{name}}",
  },
} as const;
