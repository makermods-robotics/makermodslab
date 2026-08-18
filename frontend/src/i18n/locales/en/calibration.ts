/**
 * "calibration" namespace — the per-side calibration config library.
 *
 * Everything a user or the backend NAMES stays out of this file: calibration
 * config file names, robot names, and the `teleop`/`robot` device vocabulary
 * are data. They arrive as interpolation values and render verbatim in every
 * language.
 */
export default {
  library: {
    // Dropdown placeholders. The options themselves are calibration file
    // names, rendered verbatim.
    placeholderEmpty: "No saved configs",
    placeholder: "Select a config",
    // Chips on a dropdown row. Authored lowercase and uppercased by CSS in
    // cased scripts; caseless scripts drop both `uppercase` and its tracking.
    inUse: "in use",
    otherArm: "other arm",
    // aria-label and tooltip on the same button deliberately differ: the
    // tooltip is the short verb, the aria-label names what it acts on.
    renameAria: "Rename selected config",
    renameTooltip: "Rename",
    deleteAria: "Delete selected config",
    deleteTooltip: "Delete",
    rename: {
      title: "Rename config",
      description:
        "Renames the calibration file. Robots using it are updated automatically. Won't overwrite an existing name.",
      placeholder: "New name",
      submit: "Rename",
      submitting: "Renaming…",
      // Client-side validation. A server rejection shows the backend's own
      // message instead; this is only the fallback when it carries none.
      emptyName: "Name cannot be empty.",
      failed: "Rename failed.",
    },
    delete: {
      // {{name}} is the calibration file name — data, never translated.
      title: 'Delete config "{{name}}"?',
      description:
        "This permanently deletes the calibration file — you'd have to recalibrate the arm to recreate it. Any robot using it will need calibration before its next use.",
      confirm: "Delete",
    },
    toast: {
      deletedTitle: "Config deleted",
      deleted: 'Removed "{{name}}".',
      // Whole sentences, not a stem plus an appended clause. {{robots}} is a
      // pre-joined list of backend robot names (data); `count` is the number
      // of robots in it and only drives verb agreement, so it is deliberately
      // not rendered — the list already names them.
      deletedUnassigned_one:
        'Removed "{{name}}". {{robots}} now needs calibration before use.',
      deletedUnassigned_other:
        'Removed "{{name}}". {{robots}} now need calibration before use.',
      // Separator for that list of robot names.
      robotJoin: ", ",
      deleteFailedTitle: "Delete failed",
      assignedTitle: "Config assigned",
      assigned: '"{{name}}" is now used for this robot.',
      swappedTitle: "Configs swapped",
      swapped:
        '"{{name}}" is now used for this arm; the other arm took "{{previous}}".',
      // Display placeholder for {{previous}} when this slot held no config.
      // Never a stored value — the record keeps an empty string.
      noConfig: "(none)",
      assignFailedTitle: "Assign failed",
      renamedTitle: "Config renamed",
      // Both names are calibration file names — data.
      renamed: '"{{from}}" → "{{to}}".',
    },
  },
  import: {
    // One phrase names the action in three places on the same flow: the
    // button's aria-label, its tooltip, and the dialog's title. Split per arm
    // side rather than interpolating a translated word into a sentence.
    labelLeader: "Import leader calibration",
    labelFollower: "Import follower calibration",
    descriptionLeader:
      "Saves the uploaded calibration as a new leader config. Won't overwrite an existing name — pick a different one if it's taken.",
    descriptionFollower:
      "Saves the uploaded calibration as a new follower config. Won't overwrite an existing name — pick a different one if it's taken.",
    placeholder: "Config name",
    submit: "Import",
    submitting: "Importing…",
    // Client-side; a server rejection renders the backend's message instead.
    emptyName: "Name cannot be empty.",
    failed: "Import failed.",
    invalidJsonTitle: "Not a valid JSON file",
    invalidJsonDescription: "The selected file could not be parsed as JSON.",
    importedTitle: "Calibration imported",
    // {{name}} is the saved config's file name — data.
    imported: 'Saved as "{{name}}".',
  },
} as const;
