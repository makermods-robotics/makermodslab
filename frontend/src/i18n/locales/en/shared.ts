export default {
  logPanel: {
    // Fallback heading; callers usually pass their own ("Recording log", …).
    defaultTitle: "Log",
    waiting: "Waiting for log output…",
  },
  singleTab: {
    title: "MakerMods Lab is already open in another tab",
    body: "Only one tab can control the robot at a time. Switch back to the original tab, or take over here — the other tab will lock.",
    takeOver: "Use this tab",
  },
  teleopStopNotice: {
    title: "Teleoperation stopped",
    description:
      "Stopped when you left the page. The arm returns to its starting position, then goes limp.",
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
  },
  urdf: {
    switchedDefaultTitle: "Switched to default model",
    switchedDefaultDescription: "The default ARM100 robot model is now displayed.",
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
