/**
 * "library" namespace — the shared library primitives (`components/library/`:
 * the capped card grid, the dataset library), the launchpad's "My library"
 * sheet, and the delete-confirm copy resolved by `lib/deleteSemantics`.
 *
 * Key tree must match the other language exactly (see i18n/catalogs.test.ts).
 *
 * Everything a machine reads stays out of here: repo ids, dataset/model names,
 * Hub namespaces, camera names, robot types and backend messages are DATA and
 * are interpolated verbatim. Numbers, byte sizes and durations arrive
 * pre-formatted from `lib/datasetFormat` — the catalog only supplies the words
 * around them.
 */
export default {
  grid: {
    showLess: "Show less",
    // `total`, not `count`: the hidden-card tally is a plain number, and
    // naming it `count` would make i18next try to select a plural form for a
    // string that has none.
    showAll: "Show all {{total}}",
  },

  datasets: {
    // DatasetItem["source"] — the enum values are data, only their labels are
    // translated. "Hub" is a product name and stays as-is.
    source: {
      local: "Local",
      hub: "Hub",
      both: "Local · Hub",
    },
    // title attribute on the lock chip; `private` is the chip's own text.
    privateTitle: "Private on the Hub",
    private: "private",
    hubOnly: "On the Hub — not downloaded locally.",
    detailsError: "Couldn't read this dataset's details.",
    // aria-labels on the two skeleton placeholders.
    loadingDetails: "Loading dataset details",
    loadingList: "Loading datasets",
    empty: "No datasets yet. Record your first one above.",
    noMatch: "No datasets match.",
    // Used for both the search input's placeholder and its aria-label.
    searchPlaceholder: "Search datasets",
    // Pill labels only — the filter KEYS ("all"/"local"/"hub") are logic and
    // never translated.
    filters: {
      all: "All",
      local: "Local",
      hub: "Hub",
    },
    episodes_one: "{{count}} episode",
    episodes_other: "{{count}} episodes",
    // The frame tally arrives pre-formatted ("16.7k"), so it is deliberately
    // not an i18next `count` — that would try to derive a plural from a string.
    frames: "{{frames}} frames",
    meta: {
      cameras: "Cameras",
      robot: "Robot",
      // Row label: singular when the dataset has exactly one task.
      task_one: "Task",
      task_other: "Tasks",
      // Row value shown in place of the task text once there is more than one.
      taskCount_one: "{{count}} task",
      taskCount_other: "{{count}} tasks",
      size: "Size",
    },
    select: "Select",
    selected: "Selected",
  },

  // The launchpad's "My library" slide-over (launchpad/LibrarySheet).
  sheet: {
    title: "My library",
    close: "Close library",
    tabs: {
      skills: "My skills",
      datasets: "My datasets",
    },
    // Checkpoint step count, pre-formatted ("16k") — see `frames` above for
    // why this is not a `count` plural.
    steps: "{{steps}} steps",
    private: "private",
    skills: {
      loading: "Loading skills…",
      empty: "No skills of yours yet — create one below.",
      // aria-labels. {{name}} is the skill's display title.
      manage: "Manage {{name}}",
      run: "Run {{name}}",
    },
    datasets: {
      loading: "Loading datasets…",
      empty: "No datasets of yours yet — record one to get started.",
      // Lower-case on purpose: this is a mono subtitle line, not a chip.
      source: {
        both: "local + Hub",
        hub: "on Hub",
        local: "local only",
      },
    },
    actions: {
      addFromHub: "Add from Hub",
      importFromDisk: "Import from disk",
      manageCaches: "Manage caches",
      newSkill: "New Skill",
      mergeDatasets: "Merge datasets",
    },
    // Toast titles. Their descriptions are always a repo id or the backend's
    // own error text, both rendered verbatim.
    toast: {
      datasetSaved: "Dataset saved",
      datasetImported: "Dataset imported",
      modelSaved: "Model saved",
      modelImported: "Model imported",
      downloadStarted: "Download started",
      downloadFailed: "Couldn't start download",
    },
  },

  /**
   * Delete-confirm copy, one complete sentence per (action × kind).
   *
   * `lib/deleteSemantics` used to build these by splicing the noun "dataset" /
   * "model" into a shared template and by handing the caller a title PREFIX to
   * concatenate a name after. Neither survives translation — the noun inflects
   * nothing in English but sits in a different clause position elsewhere, and a
   * prefix is half a sentence. So every combination spells itself out, and the
   * title takes the item name as `{{label}}` wherever that language wants it.
   *
   * Action keys mirror `DeleteAction`; the action and kind values themselves
   * are data and never appear on screen.
   */
  delete: {
    // "both" rows: the first press removes only the local copy.
    localCopy: {
      dataset: {
        title: 'Remove local copy of "{{label}}"?',
        description:
          "This removes the local copy from disk — the Hub copy stays, and it remains listed as a Hub dataset.",
        confirm: "Remove local copy",
      },
      model: {
        title: 'Remove local copy of "{{label}}"?',
        description:
          "This removes the local copy from disk — the Hub copy stays, and it remains listed as a Hub model.",
        confirm: "Remove local copy",
      },
    },
    // Pinned custom Hub rows are un-pinned, not deleted.
    unpin: {
      dataset: {
        title: 'Remove "{{label}}"?',
        description:
          "This just removes the dataset from your list. The Hub repo and any local copy are untouched — you can re-add it any time from the Add dataset menu.",
        confirm: "Remove",
      },
      model: {
        title: 'Remove "{{label}}"?',
        description:
          "This just removes the model from your list. The Hub repo and any local copy are untouched — you can re-add it any time from the Add model menu.",
        confirm: "Remove",
      },
    },
    // Own-namespace Hub rows go on the persistent hidden-list.
    hide: {
      dataset: {
        title: 'Remove "{{label}}"?',
        description:
          "This hides the dataset from your list. The Hub repo is not deleted — you can re-add it any time from the Add dataset menu.",
        confirm: "Remove",
      },
      model: {
        title: 'Remove "{{label}}"?',
        description:
          "This hides the model from your list. The Hub repo is not deleted — you can re-add it any time from the Add model menu.",
        confirm: "Remove",
      },
    },
    // Local-only rows: a real, destructive file delete.
    local: {
      dataset: {
        title: 'Delete "{{label}}"?',
        description:
          "This permanently removes the dataset from local disk — including all recorded episodes and videos. You can't undo this.",
        confirm: "Delete",
      },
      model: {
        title: 'Delete "{{label}}"?',
        description:
          "This permanently removes the model's local files from disk — including its checkpoints. You can't undo this. A Hub copy, if any, is not affected.",
        confirm: "Delete",
      },
    },
  },
} as const;
