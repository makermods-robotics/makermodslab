/**
 * "jobs" namespace — the training-jobs and models area (components/jobs/*)
 * plus the labels lib/jobsApi.ts owns.
 *
 * Key tree must match the other language exactly (see i18n/catalogs.test.ts).
 *
 * Nothing a machine reads lives here: job ids, run ids, repo ids, dataset
 * names, policy types, flavors and file paths are DATA and are interpolated
 * verbatim. Backend job-state and Hub-stage enum VALUES are never translated —
 * only the label shown for them.
 */
export default {
  // Backend job-state enum → label, keyed by the wire value. Consumed through
  // JOB_STATE_LABELS (lib/jobsApi.ts), which holds these KEY PATHS rather than
  // resolved words so the module-level map can't freeze the first language
  // loaded. `interrupted` reads "Stopped" on purpose: it is the state the
  // Stop button produces, and "Interrupted" suggests something happened TO the
  // run.
  jobState: {
    running: "Running",
    done: "Done",
    failed: "Failed",
    interrupted: "Stopped",
  },
  // Hugging Face Jobs stage → label, keyed by the Hub's stage string. Two
  // words for SCHEDULING on purpose: the Hub job card says "Scheduling", the
  // run dropdown says "Starting". `unknown` is the fallback for a stage this
  // bundle has no word for — the raw stage string wins when it is non-empty.
  stage: {
    running: "Running",
    queued: "Queued",
    scheduling: "Scheduling",
    starting: "Starting",
    completed: "Done",
    failed: "Failed",
    cancelled: "Cancelled",
    unknown: "Unknown",
  },
  // Where a run/model lives. Shared by the job card, the model card and the
  // run dropdown so one run reads the same in all three.
  location: {
    local: "Local",
    cloud: "Cloud",
    imported: "Imported",
    // "Hub" is a product name and stays untranslated; it is the fallback
    // label when a Hub job names no flavor.
    hub: "Hub",
    localTitle: "Runs on this machine",
    cloudTitle: "Runs on Hugging Face cloud",
    fromHub: "from Hub",
    fromHubTitle: "Imported from a Hugging Face Hub repo",
  },
  // Left-hand labels of the cards' metadata rows. The values beside them are
  // data (policy type, dataset id, owner) or pre-formatted numbers/durations.
  meta: {
    policy: "Policy",
    dataset: "Dataset",
    steps: "Steps",
    flavor: "Flavor",
    created: "Created",
    owner: "Owner",
    image: "Image",
    updated: "Updated",
  },
  // Card action controls, shared between the job card, the model card and the
  // Hub model card. Where an aria-label and a title carry the same words they
  // share one key rather than two that could drift.
  actions: {
    run: "Run",
    runInferenceCheckpoint: "Run inference with this checkpoint",
    runInferenceModel: "Run inference with this model",
    runInference: "Run inference",
    fineTune: "Fine-tune",
    fineTuneHint: "Fine-tune a new run from this model's weights",
    download: "Download this checkpoint",
    rename: "Rename",
    renameAria: "Rename model",
    openHubJob: "Open Hub job page",
    viewOnHub: "View on Hub",
  },
  // The rename dialog, rendered identically by JobCard and ModelCard.
  rename: {
    title: "Rename model",
    // <0/> wraps the mono span holding the run id / repo id — data, rendered
    // verbatim. {{target}} is one of the two words below.
    description:
      "Sets a display name only — the underlying {{target}} (<0/>) is not moved or changed.",
    targetRun: "run",
    targetHubRepo: "Hub repo",
    placeholder: "New name",
    submit: "Rename",
    submitting: "Renaming…",
    empty: "Name cannot be empty.",
    toastTitle: "Model renamed",
    // {{from}} and {{to}} are the names themselves — user data.
    toastDescription: '"{{from}}" → "{{to}}".',
  },
  // Client-side substitutes raised by lib/jobsApi.ts. Only OUR sentences live
  // here — every other refusal shows the backend's own `detail` verbatim.
  errors: {
    trainingAlreadyRunning:
      "Another training is already running. Stop it first.",
  },
  // Hub-only jobs, shared by the Hub job card and the run dropdown.
  hubJob: {
    // {{id}} is a truncated job id — data.
    fallbackTitle: "Job {{id}}…",
    removeAria: "Remove job from list",
    removeTitle: "Remove from list",
  },
  progress: {
    // Lowercase on purpose: it stands in for a "started 5m ago" subtitle and
    // for the dropdown's step counter, both of which read as running text.
    starting: "starting…",
  },
  checkpointDropdown: {
    placeholder: "Select checkpoint",
    // step 0 is the sentinel for an imported single-model checkpoint, which
    // has no meaningful step number.
    latest: "latest",
    // {{step}} is the raw step number, stringified by the caller — no locale
    // number formatting is introduced here.
    step: "step {{step}}",
  },
  jobCard: {
    stopAria: "Stop job",
    deleteAria: "Delete job",
    trainingStarting: "Training starting…",
    // Each subtitle branch gets its own key — the card picks between them, it
    // never assembles one from parts.
    subtitle: {
      // {{when}} is a pre-formatted relative duration ("5m ago"); duration
      // formatting is deliberately left as-is.
      started: "started {{when}}",
      ended: "ended {{when}}",
    },
    // The final subtitle branch: the run's state as running text. Lowercase in
    // English by hand, never by calling .toLowerCase() on a translated word.
    subtitleState: {
      running: "running",
      done: "done",
      failed: "failed",
      interrupted: "stopped",
    },
    resumeLatest: "Resume from latest",
    // {{step}} arrives pre-formatted (toLocaleString) — deliberately not an
    // i18next `count`, which would try to re-derive a plural from a string.
    resumeStep: "Resume from step {{step}}",
    resumeHint:
      "Opens the training form to continue from this checkpoint. Compute defaults to where this checkpoint's run executed, and can be retargeted before you start.",
    // {{target}} is the pip install target the backend named — data.
    install: "Install {{target}}",
    downloadFailed: "Download failed",
  },
  hubModelCard: {
    uploaded: "Uploaded",
    deleteAria: "Delete model repo",
    deletedToast: "Model repo deleted",
    dialog: {
      title: "Delete model repo",
      description:
        "This permanently deletes the model repository and its files from the Hugging Face Hub. This cannot be undone.",
      // <0/> is the code element holding the repo id — data, typed back
      // verbatim by the user to arm the button.
      confirmPrompt: "Type <0/> to confirm.",
      submit: "Delete permanently",
      submitting: "Deleting…",
    },
  },
  importModal: {
    title: "Import a skill",
    description:
      "Point at a local directory or a Hugging Face repo. It appears in your skills, ready to run inference on.",
    sourceLabel: "Local path or Hugging Face repo id",
    // The example path and repo id are illustrative data — keep them verbatim
    // in every language; only the joining word is translated.
    sourcePlaceholder: "/path/to/pretrained_model  or  user/my-policy",
    nameLabel: "Display name (optional)",
    namePlaceholder: "My imported policy",
    submit: "Import",
    submitting: "Importing…",
    alreadyImportedTitle: "Already imported",
    // {{name}} is the model's display name — data.
    alreadyImportedDescription: '"{{name}}" is already in your models.',
  },
  // Toasts raised by the shared jobs/models data provider. The descriptions
  // beside these titles are backend error text and stay as the server wrote
  // them.
  jobsData: {
    stopping: "Job stopping",
    stopFailed: "Stop failed",
    removed: "Job removed",
    deleteFailed: "Delete failed",
    dismissed: "Job removed from list",
    dismissFailed: "Remove failed",
  },
  jobsDropdown: {
    triggerAria: "Select a training run",
    listAria: "Training runs",
    placeholder: "Select a run",
    stopAria: "Stop this run",
    resumeAria: "Resume from the newest usable checkpoint",
    openHubAria: "Open this job on the Hub",
    openHubTitle: "Open on the Hub",
    // {{owner}} is a Hugging Face username — data.
    hubJobTitleWithOwner: "Hugging Face job · {{owner}}",
    hubJobTitle: "Hugging Face job",
    // Section labels above each group of rows. "MakerMods Lab" is the product
    // name and stays untranslated.
    groups: {
      lab: "Launched from MakerMods Lab",
      hub: "Other Hub jobs",
      untrackedLab: "Untracked · launched from MakerMods Lab",
      untrackedHub: "Untracked · other Hub jobs",
    },
    hideUntracked: "Hide untracked",
    // {{total}} is a plain row count. Named `total` rather than `count` so
    // i18next does not treat it as a plural selector — English has one form
    // here.
    untracked: "Untracked ({{total}})",
  },
  jobsLibrary: {
    title: "Training jobs",
    refresh: "Refresh job list",
    searchPlaceholder: "Search jobs",
    // Filter pill labels. The `key` each one filters on is logic and lives in
    // the component; only these words are translated.
    filters: {
      all: "All",
      local: "Local",
      online: "Online",
    },
    empty: {
      search: "No jobs match your search.",
      local: "No local jobs.",
      online: "No online jobs.",
      none: "No training jobs yet.",
    },
    firstRun: "No training jobs yet. Start one above.",
    // Appended after `firstRun` as its own sentence when the cloud half of the
    // list is silent for a reason worth naming.
    signIn: "Sign in with Hugging Face to see your cloud jobs.",
    missingJobRead:
      "Your Hugging Face token is missing the job.read permission, so cloud jobs can't be listed.",
    // Same fact as `missingJobRead`, rendered where the permission name can be
    // a <code> element. <0/> is that element and holds "job.read" — an API
    // scope name, never translated.
    missingJobReadRich:
      "Your Hugging Face token is missing the <0/> permission, so cloud jobs can't be listed.",
    // {{error}} is backend/network error text — passed through as written.
    localError: "Couldn't load local jobs: {{error}}",
    cloudError: "Couldn't load cloud jobs: {{error}}",
    checkpointsError: "Couldn't load checkpoints",
    noResumeTitle: "Nothing to resume from",
    // One key per NoResumeReason the shared rule can return (resumeSeed.ts).
    // The reason VALUES are internal identifiers and never translated.
    noResume: {
      notResumable: "This run isn't in a state that can be continued.",
      noCheckpoints:
        "This run and the runs it continues from saved no checkpoint.",
      ownerDone:
        "Every checkpoint this run can continue from belongs to a run that already reached its target, so its learning-rate schedule is spent. Fine-tune from the final checkpoint instead.",
      atTarget:
        "Every checkpoint this run can continue from is already at its step target. Raise the target to continue, or fine-tune from the final checkpoint.",
      siblingCap:
        "Every checkpoint left in this run's lineage was saved past the step this run reached, so it belongs to another continuation sharing the same cloud output.",
      other: "No checkpoint in this run's lineage can be resumed from.",
    },
  },
  modelCard: {
    deleteAria: "Delete model",
    deleteTitle: "Remove",
    checkpointPlaceholder: "No checkpoints",
    // Subtitle branches — the card picks one, never assembles one.
    // {{when}} is a pre-formatted relative duration.
    trained: "trained {{when}}",
    created: "created {{when}}",
    // Every control renders even when its predicate is unmet; these are the
    // reasons shown on hover instead of hiding it.
    reason: {
      noCheckpointToRun: "No checkpoint available to run yet.",
      finetuneWhileRunning:
        "Can't fine-tune while this run is still in progress.",
      noCheckpointToFinetune: "No checkpoint to fine-tune from yet.",
      noCheckpointToDownload: "No checkpoint available to download yet.",
      hubImportWeights:
        "This model's weights live on the Hub, not on this machine.",
      // {{path}} is the on-disk location the user imported from — data.
      importedFromDisk:
        "Imported from disk — the checkpoint is already at {{path}}",
      importedNoExport:
        "Imported models aren't re-exported — open the original folder.",
      cloudCheckpoints:
        "Cloud runs keep their checkpoints on the Hub, not on this machine.",
      localOnly: "Only local training runs have a downloadable checkpoint.",
      noCheckpointsToChoose: "No checkpoints to choose from yet.",
      oneCheckpoint: "Only one checkpoint — nothing to choose.",
    },
  },
  modelsLibrary: {
    title: "Your skills",
    importSkill: "Import skill",
    searchPlaceholder: "Search skills",
    // Names the "Import skill" button above it — keep the two in step.
    empty:
      "No skills yet. Train one, or use Import skill to add one from the Hub or a local folder.",
    noMatch: "No models match.",
    filters: {
      all: "All",
      trained: "Trained",
      imported: "Imported",
      uploaded: "Uploaded",
    },
  },
} as const;
