// Client-side dataset-name validation, mirroring the backend
// `validate_dataset_name` in makermodslab/utils/config.py. The user types just the
// NAME (one repo-id segment); the namespace is prepended from HF auth. Rejecting
// a bad name here gives immediate feedback, but the backend re-validates since
// the UI can be bypassed.
const SEGMENT_RE = /^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$/;

/** Returns a human-readable error, or null when `name` is a valid dataset name. */
export function validateDatasetName(name: string): string | null {
  if (!name || !name.trim()) return "Dataset name can't be empty.";
  if (name !== name.trim())
    return "Dataset name can't have leading or trailing spaces.";
  if (name.includes("/") || name.includes("\\"))
    return "Dataset name can't contain slashes.";
  if (name === "." || name === "..") return "Dataset name can't be '.' or '..'.";
  if (name.length > 96) return "Dataset name is too long (max 96 characters).";
  if (!SEGMENT_RE.test(name))
    return "Use letters, digits, '.', '_' and '-'; start and end with a letter or digit.";
  return null;
}

/**
 * Validate a user-chosen MODEL name, mirroring backend `validate_model_name`
 * (makermodslab/utils/config.py). Same segment rules as a dataset name, because
 * the name lands in the same kind of place: one path segment of the repo the
 * run publishes to (`<name>_<timestamp>`). Only the wording differs, so the
 * message names the thing the user typed. Returns null when `name` is valid.
 */
export function validateModelName(name: string): string | null {
  const error = validateDatasetName(name);
  return error === null ? null : error.replace("Dataset name", "Model name");
}

/**
 * Validate a full dataset id: a bare name, or "namespace/name" (one slash).
 * Mirrors backend `validate_dataset_repo_id`. Use for fields that may carry a
 * namespace (e.g. the merge output). Returns an error message, or null if valid.
 */
export function validateDatasetRepoId(repoId: string): string | null {
  if (!repoId || !repoId.trim()) return "Dataset name can't be empty.";
  const parts = repoId.split("/");
  if (parts.length > 2)
    return "Dataset name may contain at most one '/' (namespace/name).";
  if (parts.length === 2) {
    const nsError = validateDatasetName(parts[0]);
    if (nsError) return nsError.replace("Dataset name", "Namespace");
    return validateDatasetName(parts[1]);
  }
  return validateDatasetName(parts[0]);
}
