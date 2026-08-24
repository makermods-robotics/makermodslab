export default {
  cancel: "Cancel",
  close: "Close",
  connectionError: {
    title: "Connection error",
    description: "Could not connect to the backend server.",
  },
  // Dataset-name validation. Each (code x subject) pair is a complete sentence:
  // validateDatasetRepoId used to derive the namespace variants by running
  // .replace("Dataset name", "Namespace") over its own English output, which no
  // translation survives.
  datasetName: {
    dataset: {
      empty: "Dataset name can't be empty.",
      spaces: "Dataset name can't have leading or trailing spaces.",
      slashes: "Dataset name can't contain slashes.",
      dots: "Dataset name can't be '.' or '..'.",
      tooLong: "Dataset name is too long (max {{max}} characters).",
    },
    namespace: {
      empty: "Namespace can't be empty.",
      spaces: "Namespace can't have leading or trailing spaces.",
      slashes: "Namespace can't contain slashes.",
      dots: "Namespace can't be '.' or '..'.",
      tooLong: "Namespace is too long (max {{max}} characters).",
    },
    charset:
      "Use letters, digits, '.', '_' and '-'; start and end with a letter or digit.",
    tooManySlashes:
      "Dataset name may contain at most one '/' (namespace/name).",
  },
} as const;
