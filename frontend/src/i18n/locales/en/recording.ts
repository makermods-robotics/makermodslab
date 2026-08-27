export default {
  // ---------------------------------------------------------------------
  // CameraConfiguration.tsx — the editable repeater (Robot settings) and the
  // read-only SessionCameraList shown before a session starts.
  //
  // NOT in this catalog, deliberately: the FOURCC codes ("MJPG", "YUYV", …),
  // the Cv2Backends names ("ANY", "V4L2", "AVFOUNDATION", …), and the camera
  // name presets ("wrist", "top", "front", "side"). All three are submitted to
  // the backend verbatim — the presets in particular ARE the camera name that
  // gets written into the robot record, so translating the label would store a
  // Chinese camera name. See the comments beside them in the component.
  // ---------------------------------------------------------------------
  cameras: {
    heading: "Cameras",
    addTitle: "Add a camera",
    availableLabel: "Available cameras",
    // Longer `title` tooltip and the short `aria-label` on the same button —
    // the two strings differ, so they stay two keys.
    rescanTooltip:
      "Rescan for cameras (e.g. after plugging in a new USB camera)",
    rescanLabel: "Rescan for cameras",
    loadingPlaceholder: "Loading cameras...",
    selectPlaceholder: "Select camera",
    // {{index}} is the cv2 device index — a number the recorder opens by.
    indexLabel: "Index {{index}}",
    // Suffixes appended after a dropdown item's own text; the leading space is
    // part of the separator.
    alreadyAddedSuffix: " · already added",
    alreadyUsedSuffix: " · already used",
    nameLabel: "Camera Name",
    namePlaceholder: "Select a name",
    customNameOption: "Custom name…",
    customNamePlaceholder: "e.g., workspace_cam",
    addButton: "Add camera",
    nameRequiredHint: "Name this camera to add it.",
    // {{total}} — not `count`: this is a bare tally in a heading, and naming it
    // `count` would make i18next look for a plural form that doesn't exist.
    configuredTitle: "Configured cameras ({{total}})",
    emptyState: "No cameras configured. Add a camera to get started.",
    noneSelected: "No camera selected",
    previewPaused: "Preview paused",
    disconnected: "Camera disconnected — reconnect it or rescan",
    disconnectedSettings:
      "Camera disconnected — reconnect it or check Robot settings",
    removeLabel: "Remove camera",
    configurationToggle: "Configuration",
    resolutionLabel: "Resolution:",
    fpsLabel: "FPS:",
    // The FOURCC / Backend field labels. The OPTIONS inside them are data.
    fourccLabel: "FOURCC:",
    backendLabel: "Backend:",
    // "leave unset" options for the two dropdowns above; the values behind
    // them are sentinels, not these labels.
    fourccAuto: "Auto",
    backendDefault: "Default",
    backendWarning:
      "Overriding the backend can reorder camera indices on macOS.",
    // {{type}} is the camera driver id ("opencv") and {{device}} a truncated
    // browser deviceId — both data, rendered verbatim.
    deviceInfo: "Type: {{type}} | Device: {{device}}...",
    sessionHint:
      "Cameras come from the selected robot. Add, remove, or adjust them in Robot settings.",
    sessionEmpty: "No cameras on this robot.",
    toast: {
      missingInfoTitle: "Missing Information",
      selectCameraFirst: "Select a camera first.",
      nameCameraFirst:
        "Give this camera a name before adding it (e.g. workspace_cam).",
      invalidTitle: "Invalid Camera",
      invalidBody: "Selected camera is not available.",
      duplicateTitle: "Camera Already Added",
      duplicateBody: "This camera is already in the configuration.",
      nameTakenTitle: "Name Already Used",
      // {{name}} is the name the user typed — data, echoed back verbatim.
      nameTakenBody:
        'Another camera on this robot is already named "{{name}}". Pick a different name.',
      addedTitle: "Camera Added",
      addedBody: "{{name}} has been added to the configuration.",
      removedTitle: "Camera Removed",
      removedBody: "Camera has been removed from the configuration.",
    },
  },

  // ---------------------------------------------------------------------
  // RecordingSessionDialog.tsx — the live session modal.
  // ---------------------------------------------------------------------
  session: {
    dialogTitle: "Recording session",
    connecting: "Connecting to recording session...",
    // Status pill. Written in sentence case and uppercased by CSS in Latin
    // scripts; the uppercase/tracking classes come off for CJK.
    status: {
      recordingEpisode: "Recording episode {{index}}",
      resetPaused: "Reset paused",
      resetGetReady: "Reset — get ready",
      connectingRobot: "Connecting arm & cameras…",
      // {{attempt}}/{{max}} come straight off the status payload.
      reconnectingRetry: "Camera hiccup, retrying ({{attempt}}/{{max}})…",
      reconnecting: "Camera hiccup, retrying…",
      connectingTeleop: "Connecting leader arm…",
      stopping: "Stopping…",
      error: "Session error — see log",
      preparing: "Preparing session",
      complete: "Session complete",
    },
    hud: {
      // <1> wraps the current episode number, which is emphasised.
      episodeCounter: "Episode <1>{{index}}</1> / {{total}}",
      // Screen-reader equivalent of the line above.
      episodeCounterLabel: "Episode {{index}} of {{total}}",
      // {{time}} arrives pre-formatted as mm:ss.
      sessionTimeLabel: "Total session time {{time}}",
      mute: "Mute",
      unmute: "Unmute",
    },
    button: {
      done: "Done",
      quit: "Quit",
      endEpisode: "End Episode",
      startNextEpisode: "Start Next Episode",
      advance: "Advance",
      pause: "Pause",
      resume: "Resume",
      rerecord: "Re-record",
      keepEpisodes: "Keep episodes & continue",
      discardExit: "Discard & exit",
      backHome: "Back to home",
    },
    ended: {
      complete: "Recording complete — returning home…",
      warnTitle:
        "Session finished with a cleanup warning — your episodes are safe",
      failedTitle: "Recording session failed",
    },
    toast: {
      startedWarningTitle: "Recording started with a warning",
      startedTitle: "Recording Started",
      startedBody_one: "Started recording {{count}} episode",
      startedBody_other: "Started recording {{count}} episodes",
      startFailedTitle: "Error Starting Recording",
      // Last-resort fallback only: the backend's `detail`/`message` wins.
      startFailedBody: "Failed to start recording session.",
      connectionErrorTitle: "Connection Error",
      connectionErrorBody: "Could not connect to the backend server.",
      // Generic title paired with the backend's own `message` as the body.
      errorTitle: "Error",
      pauseArmedTitle: "Pause armed",
      pauseArmedBody:
        "The episode keeps recording. It'll pause once the reset phase starts.",
      rerecordTitle: "Re-recording Episode",
      rerecordBody: "Episode {{index}} will be re-recorded.",
      quittingTitle: "Quitting",
      quittingBody: "Discarding the recording…",
      finishingTitle: "Finishing",
      finishingBody: "Finalizing dataset…",
      stopFailedBody: "Failed to end the recording session.",
    },
  },

  // ---------------------------------------------------------------------
  // lib/recordingExit.ts — copy for the two explicit session exits.
  // ---------------------------------------------------------------------
  exit: {
    done: {
      title: "Finish and save?",
      description:
        "Every episode saved so far is kept, and you'll go to the upload page. The arm returns to its starting position, then goes limp.",
    },
    quit: {
      title: "Quit without saving?",
      // A RESUME session keeps everything already committed to the
      // pre-existing dataset; a FRESH session's whole dataset is deleted.
      descriptionResume:
        "Episodes already saved remain in the dataset; only the in-progress take is discarded. The arm returns to its starting position, then goes limp.",
      descriptionFresh:
        "The recording and all its episodes will be deleted. The arm returns to its starting position, then goes limp.",
      confirm: "Quit without saving",
    },
    keepRecording: "Keep recording",
  },

  log: {
    title: "Recording log",
  },
} as const;
