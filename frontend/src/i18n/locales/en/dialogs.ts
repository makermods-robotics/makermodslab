/**
 * "dialogs" namespace — the standalone dialogs in `components/dialogs/`:
 * the dataset viewer (+ its joint chart and hardware-replay panel), the policy
 * detail / manage dialogs, and the floating teleoperation viewer.
 *
 * Key tree must match the other language exactly (see i18n/catalogs.test.ts).
 *
 * Everything a machine reads stays out of here: dataset repo ids, model names,
 * Hub namespaces, camera names, joint names, robot names and backend prose
 * (`status.hint`, `status.error`, `data.message`, an ApiError's detail) are
 * DATA and are rendered verbatim. Durations and byte sizes arrive
 * pre-formatted — the catalog only supplies the words around them.
 */
export default {
  datasetDetail: {
    // Eyebrow above the repo id. "MakerMods Lab" is a product name.
    eyebrow: "MakerMods Lab Dataset Viewer",
    loadingEpisodes: "Loading episodes…",
    // Empty states: a local dataset with zero episodes vs. one whose format
    // the viewer can't read (pre-v3.0 layout, or no video at all).
    noEpisodesTitle: "No episodes recorded yet",
    noEpisodesBody:
      "Record at least one episode into this dataset to view its camera footage here.",
    noFootageTitle: "No viewable footage yet",
    noFootageBody:
      "A Hub dataset with video streams on demand here — a first-time view of a new episode may take a moment to fetch. This message means the dataset predates the viewer's format, or has no video to show.",
    // Eyebrow over the episode list. `total` rather than `count`: it is a
    // plain tally in parentheses, not a pluralized sentence.
    episodesHeading: "episodes",
    episodesHeadingWithCount: "episodes ({{total}})",
    // One row in that list. {{index}} is the dataset's own episode index.
    episodeRow: "Episode {{index}}",
    // Multiplication sign + the number; no translatable words.
    weightTimes: "\u00d7{{weight}}",
    weightTitle: "Sampled {{weight}}\u00d7 as often during training",
    mixTitle: "Training mix",
    // {{weight}} is the sampling weight, {{count}} the episode count in that tier.
    mixTier_one: "\u00d7{{weight}} \u00b7 {{count}} episode",
    mixTier_other: "\u00d7{{weight}} \u00b7 {{count}} episodes",
    mixShare: "{{percent}}% of frames",
    episodesEmpty:
      "Episodes appear here once this dataset is downloaded to your machine.",
    noCameras:
      "This dataset has no camera footage — replay it on hardware or view the joint trace below.",
    // Shown in place of a camera tile whose stream the browser can't decode.
    videoDecodeError: "Can't decode this camera's video in this browser.",
    // Transport controls — icon-only buttons, so these are aria-labels.
    previousEpisode: "Previous episode",
    nextEpisode: "Next episode",
    play: "Play",
    pause: "Pause",
    trainPolicy: "Train a policy from this",
    // Episode curation: pick which episodes of this dataset a training run
    // uses. Nothing is deleted — excluded episodes stay on disk and on the
    // Hub, they are just left out of the next run's subset.
    curateEpisodes: "Curate episodes",
    curateDone: "Done",
    // Summary under the heading while some episodes are excluded. Plain
    // tallies, not pluralized sentences, hence `included`/`total`.
    includedCount: "{{included}} of {{total}} included",
    // aria-label on each row's checkbox. {{index}} is the dataset's own
    // episode index.
    includeEpisodeAria: "Include episode {{index}} in training",
    curateSaveFailedTitle: "Couldn't save episode selection",
    curateSaveFailedBody: "Your changes weren't saved — try again.",
    // title on the disabled Train button while curation is still open.
    finishCuratingFirst: "Finish selecting episodes first",
  },

  jointChart: {
    heading: "joint positions — synced to playhead",
    // Eyebrow on the right of the chart; {{index}} is the episode index.
    episode: "episode {{index}}",
    loading: "Loading joint data…",
    noEpisode: "No episode selected",
  },

  replay: {
    // Labels for ReplayPhase. The phase VALUES are backend data; only these
    // labels are display.
    phase: {
      idle: "Idle",
      easingIn: "Easing to start position…",
      playing: "Replaying",
      stopping: "Stopping…",
      done: "Done",
      error: "Error",
    },
    // {{gap}} is the localized diagnosis from formatRobotSetupGap().
    robotNotReady: "Select a robot ready to replay: this robot {{gap}}.",
    noRobot:
      "Select a robot with a connected follower arm to replay this episode on hardware.",
    start: "Replay on hardware",
    // {{robot}} is the robot's own name — data.
    movesArmWarning: "Moves {{robot}}'s arm — make sure the area is clear.",
    stop: "Stop",
    toast: {
      failedTitle: "Replay failed",
      // Last-resort fallback: the backend hint and error text win when present.
      seeLog: "See the server log for details.",
      lostConnectionTitle: "Lost connection to backend",
      startedWarningTitle: "Started with a warning",
      startFailedTitle: "Could not start replay",
      stopFailedTitle: "Could not stop replay",
    },
  },

  policyDetail: {
    // {{title}} is the policy's display title.
    previewAlt: "{{title}} rollout preview",
    previewPlaceholder: "rollout preview",
    // Pill on a policy that exists both locally and on the Hub. "Hub" is a
    // product name.
    localAndHub: "Local + Hub",
    // Byline. {{author}} is a Hub namespace — data, never translated.
    byAuthor: "by {{author}}",
    // `steps` arrives pre-formatted ("16k"), so it is deliberately not an
    // i18next `count`.
    steps: "{{steps}} steps",
    private: "private",
    // <0> is the emphasized lead-in, <1> the dataset repo id (mono, verbatim).
    trainedOn: "<0>Trained on</0> <1>{{dataset}}</1>",
    // How much of that dataset the policy actually trained on, when it was
    // curated down to a subset. The "of total" variant needs the dataset's
    // own episode count, which is unavailable for a Hub-only dataset.
    episodeSubset: "{{used}} episodes",
    episodeSubsetOfTotal: "{{used}} of {{total}} episodes",
    notTrained: "Not trained yet — this policy is still in development.",
    // {{robot}} is the selected robot's name, or `robotFallback` when none is
    // selected.
    run: "Run on {{robot}}",
    robotFallback: "robot",
    // Between run and fine-tune, because that is the real order: fine-tuning
    // needs data the operator does not have yet, and coaching is how they get it.
    coach: "Coach it — fix what it gets wrong",
    fineTune: "Fine-tune this policy",
    // The API exposes no like count and no like action, so this is a static
    // placeholder, not a button.
    likesUnavailable: "Likes unavailable",
    viewOnHub: "View on HF Hub",
  },

  policyManage: {
    runOnRobot: "Run on robot",
    toast: {
      // Unpin / hide — the row leaves the listing, nothing is deleted.
      removedFromList: "Removed from list",
      localCopyRemoved: "Local copy removed",
      modelDeleted: "Model deleted",
      deleteFailed: "Delete failed",
      removeFailed: "Couldn't remove",
    },
  },

  teleop: {
    // Used for both the window's aria-label and its visible heading.
    title: "Teleoperation",
    // {{robot}} is the selected robot's name — data.
    titleWithRobot: "Teleoperation — {{robot}}",
    done: "Done",
    leftArm: "Left arm",
    rightArm: "Right arm",
    // Inline banner when the session died under us.
    endedWithWarning: "Teleoperation ended with a cleanup warning",
    failed: "Teleoperation failed",
    toast: {
      stoppedCheckArm: "Teleoperation stopped — check the arm",
      stopped: "Teleoperation stopped",
      // Fallback beside the backend's own `message`, which wins when present.
      releasing: "The arm returns to its starting position, then goes limp.",
      checkArm: "Check the arm",
      disconnected: "The arm was disconnected cleanly.",
    },
  },
  // Station side of remote teleoperation — the status view of a station-mode
  // host. Robot names, operator identities and room names are data; the
  // phase VALUES (remote_host.PHASES) are data too, only the labels localize.
  hosting: {
    title: "Hosting for remote teleop",
    titleWithRobot: "Hosting {{robot}} for remote teleop",
    // The stop: hands the arm back to whoever is at the station.
    release: "Release for local use",
    releaseNow: "Release now",
    releasingBanner:
      "Returning the arm to rest before releasing torque. Press Release now to skip the return.",
    // Shown only when the descriptor says station_mode.
    stationModeNote:
      "Station mode: hosting re-arms itself a few seconds after any local session ends.",
    // The dialog is open but no hosting session is live right now.
    inactive:
      "Not hosting right now — a local session has the arm, or hosting stopped.",
    phaseLabel: "Arm",
    phase: {
      parked: "Parked",
      engaging: "Engaging…",
      engaged: "Engaged",
      parking: "Parking…",
    },
    operatorLabel: "Operator",
    waitingOperator: "Waiting for an operator…",
    roomLabel: "Room",
    leftArm: "Left arm",
    rightArm: "Right arm",
    endedWithWarning: "Hosting ended with a cleanup warning",
    failed: "Hosting failed",
    // Station mode only: opens the hosted-robot picker (StationRobotDialog).
    changeRobot: "Change hosted robot…",
    toast: {
      stoppedCheckArm: "Hosting stopped — check the arm",
      stopped: "Hosting stopped",
      releasing: "The arm returns to its starting position, then goes limp.",
      checkArm: "Check the arm",
      disconnected: "The arm was disconnected cleanly.",
    },
  },
  // Station side — which saved robot this station hosts (PUT
  // /api/v1/station/robot). Robot names are data; the list is the backend's
  // `hostable` (saved robots whose follower side is set up).
  stationRobot: {
    title: "Hosted robot",
    description:
      "The robot this station hosts for remote teleoperation. Hosting re-arms on your choice within a few seconds; a parked, unseated session of the previous robot yields on its own.",
    listLabel: "Robots this station can host",
    // The row of the robot hosted right now.
    hostedNow: "Hosted now",
    // The row of the chosen robot while its hosting is down (a local session
    // has the arm, or it is still re-arming).
    chosen: "Chosen",
    // No saved robot has its follower side set up.
    empty:
      "No robot on this station can be hosted yet. Set up a robot's follower arm and cameras first — hosting supports the SO-101.",
    openSettings: "Open Robot settings",
    createRobot: "Create robot",
    host: "Host this robot",
    applying: "Applying…",
    stopHosting: "Stop hosting",
    toast: {
      changedTitle: "Hosted robot changed",
      changedDescription: "This station now hosts {{robot}}.",
      stoppedTitle: "Hosting stopped",
      stoppedDescription: "This station no longer hosts a robot.",
    },
  },
  // Operator side — the station picker, then the live viewer. Station names,
  // hosted robot names, camera names and instance ids are data.
  remoteTeleop: {
    title: "Remote teleoperation",
    titleWithRobot: "Remote teleoperation — {{robot}}",
    stationsHeading: "Stations",
    refreshStations: "Refresh stations",
    stationsLoading: "Looking for stations…",
    stationsEmpty:
      "No station is hosting right now. Start one with makermodslab --sfu --host <robot>, then refresh.",
    // {{robot}} is the hosted robot's name.
    hostingRobot: "Hosting {{robot}}",
    // Why a station row is greyed out: its hosted arm's family differs from
    // the local record's, which the server would refuse as a schema mismatch.
    armMismatch: "Different arm family",
    // The station's seat state on a picker row. {{operator}} is the seat
    // holder's identity (data); a seated station is greyed out with the
    // second line.
    rowParked: "Parked",
    rowEngagedBy: "Engaged by {{operator}}",
    seatTaken: "Someone else is driving",
    start: "Start",
    starting: "Starting…",
    stop: "Stop",
    // Home parks the station's arm and holds it; Engage re-energizes it.
    home: "Home",
    homeHint: "Park the station's arm and hold it there.",
    engage: "Engage",
    engageHint: "Re-energize the station's arm with a soft start.",
    // The live phase of the station's arm, from station_phase.
    stationPhaseLabel: "Station arm",
    stationPhase: {
      parked: "Parked",
      engaging: "Engaging…",
      engaged: "Engaged",
      parking: "Parking…",
      // station_phase is null: the station could not be read.
      unknown: "Unknown",
    },
    softStart: "Soft start…",
    stationLabel: "Station",
    roomLabel: "Room",
    cameras: "Cameras",
    noCameras: "The station publishes no cameras.",
    // {{name}} is the camera's name.
    cameraAlt: "Remote camera {{name}}",
    cameraFailed: "Stream unavailable",
    latency: "Round trip",
    latencyLast: "last",
    latencyMean: "mean",
    latencyP95: "p95",
    latencyWaiting: "Waiting for the first sample…",
    observations: "observations",
    dropped: "dropped",
    leftArm: "Left arm",
    rightArm: "Right arm",
    endedWithWarning: "Remote teleoperation ended with a cleanup warning",
    failed: "Remote teleoperation failed",
    toast: {
      startedTitle: "Remote teleoperation started",
      // {{station}} is the station's display name.
      startedFallback: "Driving {{station}}.",
      startedWarningTitle: "Started with a warning",
      stoppedCheckArm: "Remote teleoperation stopped — check the arm",
      stopped: "Remote teleoperation stopped",
      checkArm: "Check the arm",
      disconnected: "The leader arm was disconnected cleanly.",
      // Home / Engage answered success=false; the reason is server prose.
      commandRefused: "The station refused the command",
    },
  },
  // Install flow for the `remote` optional extra (mirrors the training and
  // W&B install dialogs; the generic install copy lives in training.install).
  remoteExtra: {
    title: "Remote teleoperation extra not installed",
    srDescription: "Install the remote extra to enable remote teleoperation.",
    // <0> is the extra's name — a pip identifier, rendered verbatim.
    description:
      "Remote teleoperation needs the <0>remote</0> extra, which isn't installed in this environment. Install it to host a robot or drive one over the network.",
    ready:
      "Install complete — remote teleoperation is available immediately, no restart needed. Start again.",
  },
} as const;
