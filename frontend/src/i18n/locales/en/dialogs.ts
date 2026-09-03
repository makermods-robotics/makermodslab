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
  // Station side of remote teleoperation — the floating viewer while this
  // robot is hosted. Robot names, operator identities and room names are data.
  hosting: {
    title: "Hosting for remote teleop",
    titleWithRobot: "Hosting {{robot}} for remote teleop",
    stop: "Stop",
    releaseNow: "Release now",
    releasingBanner:
      "Returning the arm to rest before releasing torque. Press Release now to skip the return.",
    operatorLabel: "Operator",
    waitingOperator: "Waiting for an operator…",
    roomLabel: "Room",
    leftArm: "Left arm",
    rightArm: "Right arm",
    endedWithWarning: "Hosting ended with a cleanup warning",
    failed: "Hosting failed",
    toast: {
      stoppedCheckArm: "Hosting stopped — check the arm",
      stopped: "Hosting stopped",
      releasing: "The arm returns to its starting position, then goes limp.",
      checkArm: "Check the arm",
      disconnected: "The arm was disconnected cleanly.",
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
      "No station is hosting right now. On the station, press “Available for remote teleop”, then refresh.",
    // {{robot}} is the hosted robot's name.
    hostingRobot: "Hosting {{robot}}",
    start: "Start",
    starting: "Starting…",
    stop: "Stop",
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
