import type { TFunction } from "i18next";

// Client-side dataset-name validation, mirroring the backend
// `validate_dataset_name` in makermodslab/utils/config.py. The user types just the
// NAME (one repo-id segment); the namespace is prepended from HF auth. Rejecting
// a bad name here gives immediate feedback, but the backend re-validates since
// the UI can be bypassed.
const SEGMENT_RE = /^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$/;
const MAX_LENGTH = 96;

/** Which noun the message is about. `validateDatasetRepoId` used to derive this
 * by running `.replace("Dataset name", "Namespace")` over its own output —
 * string surgery on English prose, which no translation survives. */
export type DatasetNameSubject = "dataset" | "namespace";

export type DatasetNameIssue = {
  code:
    | "empty"
    | "spaces"
    | "slashes"
    | "dots"
    | "charset"
    | "tooManySlashes"
    | "tooLong";
  subject: DatasetNameSubject;
  /** Only set for `tooLong`. */
  max?: number;
};

/** Structured validation result — null when `name` is valid. */
export function datasetNameIssue(
  name: string,
  subject: DatasetNameSubject = "dataset",
): DatasetNameIssue | null {
  if (!name || !name.trim()) return { code: "empty", subject };
  if (name !== name.trim()) return { code: "spaces", subject };
  if (name.includes("/") || name.includes("\\"))
    return { code: "slashes", subject };
  if (name === "." || name === "..") return { code: "dots", subject };
  if (name.length > MAX_LENGTH)
    return { code: "tooLong", subject, max: MAX_LENGTH };
  if (!SEGMENT_RE.test(name)) return { code: "charset", subject };
  return null;
}

/**
 * Validate a full dataset id: a bare name, or "namespace/name" (one slash).
 * Mirrors backend `validate_dataset_repo_id`. Use for fields that may carry a
 * namespace (e.g. the merge output).
 */
export function datasetRepoIdIssue(repoId: string): DatasetNameIssue | null {
  if (!repoId || !repoId.trim()) return { code: "empty", subject: "dataset" };
  const parts = repoId.split("/");
  if (parts.length > 2)
    return { code: "tooManySlashes", subject: "dataset" };
  if (parts.length === 2) {
    const nsIssue = datasetNameIssue(parts[0], "namespace");
    if (nsIssue) return nsIssue;
    return datasetNameIssue(parts[1]);
  }
  return datasetNameIssue(parts[0]);
}

/** Localized message for an issue. Each (code x subject) pair is a complete
 * sentence in the catalog rather than a noun interpolated into a template. */
export function formatDatasetNameIssue(
  t: TFunction,
  issue: DatasetNameIssue,
): string {
  if (issue.code === "charset") return t("common.datasetName.charset");
  if (issue.code === "tooManySlashes")
    return t("common.datasetName.tooManySlashes");
  const scope =
    issue.subject === "namespace"
      ? "common.datasetName.namespace"
      : "common.datasetName.dataset";
  return t(`${scope}.${issue.code}` as never, { max: issue.max });
}

/** English message, kept for non-React callers. Byte-identical to the
 * pre-i18n implementation — frozen by datasetName.test.ts. */
export function validateDatasetName(name: string): string | null {
  const issue = datasetNameIssue(name);
  return issue ? englishIssue(issue) : null;
}

/** English message for a full repo id. Byte-identical to the pre-i18n
 * implementation. */
export function validateDatasetRepoId(repoId: string): string | null {
  const issue = datasetRepoIdIssue(repoId);
  return issue ? englishIssue(issue) : null;
}

function englishIssue(issue: DatasetNameIssue): string {
  const noun = issue.subject === "namespace" ? "Namespace" : "Dataset name";
  switch (issue.code) {
    case "empty":
      return `${noun} can't be empty.`;
    case "spaces":
      return `${noun} can't have leading or trailing spaces.`;
    case "slashes":
      return `${noun} can't contain slashes.`;
    case "dots":
      return `${noun} can't be '.' or '..'.`;
    case "tooLong":
      return `${noun} is too long (max ${issue.max} characters).`;
    case "tooManySlashes":
      return "Dataset name may contain at most one '/' (namespace/name).";
    case "charset":
      return "Use letters, digits, '.', '_' and '-'; start and end with a letter or digit.";
  }
}
