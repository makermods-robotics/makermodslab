/**
 * "landing" namespace — the dataset / model / robot dialogs, the two pickers,
 * the info cards, and the inference modal under `components/landing/`, plus the
 * client-side fallbacks of the two Hub transfer hooks they drive
 * (useDatasetUpload / useHubDownload).
 *
 * Repo ids, dataset and model names, Hub namespaces, usernames, tag strings,
 * file paths, policy-type ids and sample identifiers ("org/name", "my_robot")
 * are DATA: they are interpolated verbatim and never appear as catalog text.
 * Byte / count / duration / date formatting is likewise untouched — a
 * pre-formatted value arrives as a plain interpolation variable, never as
 * i18next's `count`.
 */
export default {
  // Shared by DatasetPicker and ModelPicker: the two popovers render the same
  // section headings and row chips, so one set of keys serves both.
  picker: {
    // Product name — same in every language, keyed so both sections have one
    // uniform shape.
    huggingFace: "Hugging Face",
    local: "Local",
    localAndHub: "local + hub",
    private: "private",
    // `title` on the per-row trash button (the aria-label names the row and
    // lives with each picker).
    deleteTitle: "Delete…",
  },
  datasetPicker: {
    searchPlaceholder: "Search datasets…",
    loading: "Loading datasets…",
    empty:
      "No datasets yet. Use “Add dataset” to record, download, or import one.",
    deleteAria: "Delete {{repoId}}",
  },
  modelPicker: {
    searchPlaceholder: "Search models…",
    loading: "Loading models…",
    empty: "No models yet. Use “Add model” to train, download, or import one.",
    deleteAria: "Delete {{name}}",
  },
  addDatasetFromHub: {
    title: "Add a dataset from Hugging Face",
    description:
      "Enter a Hub dataset id to add it to your list. It appears under “Hugging Face” and training fetches it on demand.",
    idLabel: "Hub dataset id",
    // <0> wraps a mono span holding the literal id format, which is a format
    // token and stays "org/name" in every language.
    idError: "Enter a Hub dataset id as <0>org/name</0>.",
    downloadNow: "Download to this machine now",
    downloadNowHint:
      "Fetches the dataset into the local cache in the background. It can be multi-GB.",
    submit: "Add dataset",
    submitWithDownload: "Add & download",
  },
  addModelFromHub: {
    title: "Add a model from Hugging Face",
    description:
      "Enter a Hub model id to add it to your list. It appears under “Hugging Face” and inference fetches it on demand.",
    idLabel: "Hub model id",
    idError: "Enter a Hub model id as <0>org/name</0>.",
    downloadNow: "Download to this machine now",
    downloadNowHint:
      "Fetches the checkpoint into the local models cache in the background, so inference works offline.",
    submit: "Add model",
    submitWithDownload: "Add & download",
  },
  createDataset: {
    title: "Create a new dataset",
    description:
      "Name the dataset you're about to record. You'll set the task and episode count in the next step.",
    nameLabel: "Name",
    duplicate: "A dataset with this name already exists.",
    // <0> is a mono span around the resulting repo id — data, interpolated.
    creates: "Creates <0>{{repoId}}</0>",
    submit: "Create",
  },
  createRobot: {
    title: "Create a new robot",
    description:
      "Choose a name and arm layout. The layout is fixed once created — a bimanual rig is a separate robot.",
    nameLabel: "Name",
    duplicate: "A robot with this name already exists.",
    // Label AND aria-label of the layout radiogroup — identical text, one key.
    armLayout: "Arm layout",
    // The DISPLAY half of MODE_OPTIONS. The submitted value ("single" /
    // "bimanual") is logic and stays in the component, untranslated.
    modes: {
      single: {
        label: "Single arm",
        description: "One leader + one follower",
      },
      bimanual: {
        label: "Bimanual",
        description: "Two leader/follower pairs (4 arms)",
      },
    },
    submit: "Create",
    submitting: "Creating…",
  },
  importDatasetFromDisk: {
    title: "Import a dataset from disk",
    description:
      "Point at a LeRobot dataset folder already on this machine. It's copied into your local cache — the original folder is left untouched.",
    pathLabel: "Dataset folder path",
    nameLabel: "Name (optional)",
    namePlaceholder: "Defaults to the folder name",
    submit: "Import",
    submitting: "Importing…",
  },
  importModelFromDisk: {
    title: "Import a model from disk",
    description:
      "Point at a policy checkpoint folder already on this machine. It's copied into your local models cache — the original folder is left untouched.",
    pathLabel: "Checkpoint folder path",
    nameLabel: "Name (optional)",
    namePlaceholder: "Defaults to the folder name",
    submit: "Import",
    submitting: "Importing…",
  },
  visibilityToggle: {
    public: "Public",
    private: "Private",
  },
  uploadDataset: {
    visibility: "Visibility",
    privateNote: "Only you can see this dataset.",
    publicNote:
      "Anyone can see this dataset — recordings include your camera footage.",
    tagsLabel: "Tags (optional, comma-separated)",
    submit: "Upload to Hub",
    starting: "Starting…",
    failedTitle: "Upload failed",
    startedTitle: "Upload started",
    startedBody: "{{repoId}} is uploading to the Hub in the background.",
  },
  hfAuthChip: {
    checking: "Checking HF…",
    loggedInTitle: "Logged in to Hugging Face as {{username}}",
    // Trailing space is intentional — it sits directly before the username.
    prefix: "Hugging Face: ",
    loginAria: "Not logged in to Hugging Face — show login instructions",
    login: "Log in to Hugging Face",
  },
  hfAuthDialog: {
    title: "Hugging Face CLI not configured",
    description:
      "Uploads, training, and replay-from-Hub require a logged-in HF CLI. Run this in a terminal:",
    copyAria: "Copy command",
    recheck: "I've logged in — recheck",
  },
  hfAuthBanner: {
    title: "Hugging Face access required for cloud training",
    // <0> is the link to the token settings page (its href and visible URL are
    // data; the external-link icon lives inside the link COMPONENT, not in this
    // string — <Trans> does not resolve nested slots). <1> is the mono span
    // holding "Write", the literal name of the HF permission, kept verbatim.
    tokenHint:
      "Create a token at <0>huggingface.co/settings/tokens</0> with <1>Write</1> access (so trained policies can upload to your account), then paste it below.",
    save: "Save token",
    saving: "Saving…",
  },
  manageCaches: {
    title: "Manage cached datasets",
    description:
      "Free disk space by clearing the local cache of datasets that also live on the Hugging Face Hub. The Hub copy stays — clearing only removes the local copy.",
    // <0> wraps a code span holding the env var name, which is a literal.
    offlineNote:
      "This backend is in offline mode (<0>HF_HUB_OFFLINE</0>). A cleared cache can't be re-downloaded from the Hub until offline mode is switched off.",
    empty: "No HF datasets are cached locally.",
    clear: "Clear cache",
    clearing: "Clearing…",
    // `{{n}}`, not i18next `count`: the number only sizes the button's badge,
    // it selects no plural form.
    clearAll: "Clear all ({{n}})",
    clearFailed: "Could not clear the cache for {{repoId}}.",
  },
  mergeDatasets: {
    title: "Merge datasets",
    description:
      "Combine episodes from two or more datasets into a new one. Sources must share the same robot, fps, and cameras.",
    sources: "Sources ({{n}} selected)",
    noDatasets: "No datasets found.",
    outputLabel: "Output dataset name",
    // <0> wraps a code span around the resolved repo id — data.
    willBeCreatedAs: "Will be created as <0>{{repoId}}</0>",
    starting: "Starting…",
    submit_one: "Merge {{count}} dataset",
    submit_other: "Merge {{count}} datasets",
    merging: "Merging into <0>{{repoId}}</0>…",
    created: "Created <0>{{repoId}}</0>",
    failed: "Merge failed",
    done: "Done",
  },
  usageInstructions: {
    title: "Get Started with MakerMods Lab",
    description:
      "MakerMods Lab runs on your machine. Click the command to copy it, then paste in a terminal:",
    copyAria: "Copy command to clipboard",
    copied: "Copied",
    copy: "Copy",
    afterRunning:
      "After running, your browser will open the local MakerMods Lab app.",
    open: "Open MakerMods Lab",
  },
  datasetInfo: {
    loadingAria: "Loading dataset details",
    loadError: "Couldn't load dataset details.",
    episodes_one: "{{count}} episode",
    episodes_other: "{{count}} episodes",
    // `frames` arrives pre-formatted ("16.7k") — deliberately NOT an i18next
    // `count` plural, which would try to re-derive a plural from a string.
    frames: "{{frames}} frames",
    noEpisodes: "No episodes recorded",
    // aria-label and title on the same button — one key, two attributes.
    renameAria: "Rename dataset",
    deleteAria: "Delete dataset",
    rowCameras: "Cameras",
    rowRobot: "Robot",
    rowTasks: "Tasks",
    rowSize: "Size",
    noCameras: "No camera data — unusable for vision training",
    hubOnlyNote:
      "Hub dataset — not downloaded locally. Training will fetch it from the Hub on demand.",
    tasks: {
      // `{{n}}`, not `count`: "ep" is an abbreviation with no plural form.
      episodeCount: "{{n}} ep",
      count_one: "{{count}} task",
      count_other: "{{count}} tasks",
    },
    notDownloaded: {
      checking: "Checking availability…",
      localUnreadable:
        "This dataset is on this machine, but its details couldn't be read — the local copy looks incomplete or corrupt. Re-record or re-download it.",
      absent:
        "This dataset couldn't be found — there's no local copy and it isn't on the Hugging Face Hub. It may have been deleted or renamed.",
      onHub:
        "Hub dataset — not downloaded locally. Training will fetch it from the Hub on demand; per-episode details show once it's cached.",
      unknown:
        "Not downloaded locally, and the Hub couldn't be reached to confirm. Training will try to fetch it from the Hub on demand.",
    },
    hubSync: {
      uploading: "Uploading to Hub…",
      onHub: "Uploaded to HuggingFace",
      absent: "Not on the Hub, no local copy",
      localOnly: "Local only",
      unknown: "Hub status unknown",
      upload: "Upload to Hub",
      uploadedTitle: "Uploaded to Hub",
      // <0> wraps the link to the Hub page; {{repoId}} is the dataset id.
      uploadedBody: "{{repoId}} is now on the Hub. <0>View dataset</0>",
      uploadFailedTitle: "Upload failed",
      // {{message}} is the backend's own (English) explanation; only the link
      // text around it is ours. <0> wraps the setup-guide link.
      uploadFailedWithGuide: "{{message}} <0>Open setup guide</0>",
    },
    hubSettings: {
      // aria-label, title, and the visible button text are identical.
      trigger: "Visibility & tags",
      loading: "Loading Hub settings…",
      loadError: "Couldn't load Hub settings.",
      visibility: "Visibility",
      privateNote: "Only you can see this dataset.",
      publicNote:
        "Anyone can see this dataset — recordings include your camera footage.",
      saveVisibility: "Save visibility",
      saving: "Saving…",
      tags: "Tags",
      tagPlaceholder: "Add a tag, then press Enter",
      // The three tag names are literals the backend always re-adds.
      requiredTagsNote:
        "The makermods, openbooth, and MakerModsLab tags are always kept.",
      lockedTag: "Always kept — can't be removed",
      // aria-label and title on the same chip button.
      removeTag: "Remove tag {{tag}}",
      saveTags: "Save tags",
      visibilityUpdatedTitle: "Visibility updated",
      // Two whole sentences rather than an interpolated "private"/"public"
      // word, which no translator could inflect.
      nowPrivate: "{{repoId}} is now private.",
      nowPublic: "{{repoId}} is now public.",
      tagsUpdatedTitle: "Tags updated",
    },
    rename: {
      title: "Rename dataset",
      description:
        "Renames the local dataset directory. If this dataset also has a copy on the Hub, it's renamed there too.",
      placeholder: "New name",
      submit: "Rename",
      submitting: "Renaming…",
      renamedTitle: "Dataset renamed",
      renamedHub: "{{repoId}} — the Hub copy was renamed too.",
      renamedLocallyTitle: "Dataset renamed locally",
      renamedLocally:
        "{{repoId}} — any copy on the Hub still has the old name.",
    },
    download: {
      notDownloaded: "Not downloaded",
      button: "Download to this machine",
      inProgress: "Downloading to this machine…",
      doneTitle: "Downloaded to this machine",
      doneBody: "{{repoId}} is now in your local cache.",
      failedTitle: "Download failed",
      startFailedTitle: "Couldn't start download",
    },
  },
  modelInfo: {
    loadingAria: "Loading model details",
    notFound: "Model not found — it may have been deleted.",
    loadError: "Couldn't load model details.",
    // `steps` arrives pre-formatted ("16k") — not an i18next `count` plural.
    steps: "{{steps}} steps",
    private: "private",
    // aria-label and title on the same button.
    deleteAria: "Delete model",
    rowDataset: "Dataset",
    rowSize: "Size",
    rowUpdated: "Updated",
    rowHub: "Hub",
    rowPath: "Path",
    upload: "Upload to Hub",
    uploading: "Uploading…",
    uploadedTitle: "Uploaded to Hub",
    // <0> wraps the link to the Hub page; {{repoId}} is the model id.
    uploadedBody: "{{repoId}} is now on the Hub. <0>View model</0>",
    uploadFailedTitle: "Upload failed",
    download: {
      notDownloaded: "Not downloaded",
      button: "Download to this machine",
      inProgress: "Downloading to this machine…",
      doneTitle: "Downloaded to this machine",
      doneBody: "{{repoId}} is now in your local models cache.",
      failedTitle: "Download failed",
      startFailedTitle: "Couldn't start download",
    },
  },
  inference: {
    title: "Configure Inference",
    description:
      "Pick a checkpoint and confirm hardware. The selected policy will drive the follower autonomously for the configured duration.",
    robotSection: "Robot Configuration",
    noRobot: "Select and configure a robot on the Landing page first.",
    // <0> bolds the robot name; {{gap}} is the rendered setup-gap phrase from
    // formatRobotSetupGap (robot.setupGap.* in the robot namespace).
    followerNotReady:
      "<0>{{name}}</0> {{gap}}. Open Robot settings before running inference. (Inference only uses the follower arm — leader setup isn't needed.)",
    runningOn: "Running on <0>{{name}}</0>",
    runningOnBimanual: "Running on <0>{{name}}</0> (bimanual — both followers)",
    checkpointSection: "Checkpoint",
    noCheckpoints: "No checkpoints available for this job yet.",
    // <0> bolds the checkpoint's layout, <1> the robot name. {{dim}} and
    // {{arms}} are raw numbers read off the checkpoint.
    mismatchBimanual:
      "This checkpoint was trained on a <0>bimanual robot</0> ({{dim}}-dim state, {{arms}} arms), but <1>{{name}}</1> is a single-arm robot. Pick a single-arm checkpoint, or select a bimanual robot on the Landing page.",
    mismatchSingle:
      "This checkpoint was trained on a <0>single-arm robot</0> ({{dim}}-dim state), but <1>{{name}}</1> is a bimanual robot. Pick a bimanual checkpoint, or select a single-arm robot on the Landing page.",
    paramsSection: "Run parameters",
    taskLabel: "Task description",
    taskPlaceholder: "e.g., pick up the red block",
    // {{policyType}} is the policy id from the checkpoint — data.
    languageConditioned: "This policy is language-conditioned ({{policyType}}).",
    durationLabel: "Max duration (seconds)",
    engineLabel: "Inference engine",
    // Only the option LABELS are translated — the submitted values stay
    // "sync" / "rtc" in the component.
    engineSync: "Sync (default)",
    engineRtc: "RTC — experimental, smoother control",
    engineRtcHint:
      "Real-Time Chunking overlaps inference with motion, removing the pause between action chunks. It also changes how actions are generated — compare against Sync before trusting a result.",
    engineSyncHint:
      "One policy forward per control step. The arm pauses briefly between action chunks.",
    camerasSection: "Cameras",
    policyConfigLoading: "Reading policy config…",
    // {{message}} is the raw error text (backend or JS) — not ours to translate.
    policyConfigError: "Couldn't load policy config: {{message}}",
    noCameras: "This policy doesn't use cameras.",
    bindHint:
      "Bind one of this robot's cameras to each name the policy was trained with. Which camera and how it's opened come from the robot (edit in Robot settings); the capture resolution comes from the checkpoint.",
    capturesAt: "Captures at {{width}}×{{height}} — the policy's resolution",
    robotCameraResolution:
      "({{name}} is set to {{width}}×{{height}} in Robot settings)",
    disconnected: "Disconnected — reconnect it before starting",
    selectCamera: "Select a camera",
    noRobotCameras: "This robot has no cameras — add them in Robot settings",
    thumbnailReleased: "Released",
    thumbnailNoPreview: "No preview",
    advancedSummary: "Temporal ensembling for ACT",
    actionSelection: "Action selection",
    temporalEnsemble: "Temporal ensembling",
    temporalEnsembleHint:
      "Averages the overlapping action chunks the policy predicts at each step instead of executing one chunk open-loop — smoother motion, but the policy runs every control step, so it is slower.",
    coeffLabel: "Ensemble coefficient",
    // {{coeff}} is the ACT paper's default, rendered as the literal number.
    coeffPlaceholder: "{{coeff}} (ACT paper default)",
    coeffInvalid: "Enter a number greater than 0.",
    coeffHint:
      "Weights are exp(-coeff × age): higher favours the newest prediction, lower averages more evenly. The ACT paper uses {{coeff}}.",
    start: "Start Inference",
    starting: "Starting…",
    startFailedTitle: "Couldn't start inference",
  },
  // useDatasetUpload / useHubDownload. These are the CLIENT-side fallbacks used
  // only when the backend sent no message of its own; a backend message is
  // surfaced verbatim (it is English server prose, not ours to translate).
  hubUpload: {
    failed: "Upload failed.",
    couldNotStart: "Upload could not be started.",
    unreachable: "Could not reach the backend to upload.",
  },
  hubDownload: {
    failed: "Download failed.",
    couldNotStart: "Download could not be started.",
    unreachable: "Could not reach the backend to download.",
  },
} as const;
