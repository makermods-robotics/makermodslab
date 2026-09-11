import { describe, expect, it } from "vitest";
import {
  datasetNameIssue,
  datasetRepoIdIssue,
  validateDatasetName,
  validateDatasetRepoId,
} from "@/lib/datasetName";

/**
 * Freezes the ENGLISH output byte-for-byte. The validator was restructured to
 * return a code + subject (it used to build the namespace variants by running
 * .replace("Dataset name", "Namespace") over its own prose), and that
 * restructure must not change a single character an English user sees.
 */
describe("validateDatasetName (English frozen)", () => {
  it("accepts valid names", () => {
    expect(validateDatasetName("my_dataset")).toBeNull();
    expect(validateDatasetName("a")).toBeNull();
    expect(validateDatasetName("A1.b-c_d")).toBeNull();
  });

  it.each([
    ["", "Dataset name can't be empty."],
    ["   ", "Dataset name can't be empty."],
    [" lead", "Dataset name can't have leading or trailing spaces."],
    ["a/b", "Dataset name can't contain slashes."],
    ["a\\b", "Dataset name can't contain slashes."],
    [".", "Dataset name can't be '.' or '..'."],
    ["..", "Dataset name can't be '.' or '..'."],
    ["x".repeat(97), "Dataset name is too long (max 96 characters)."],
    [
      "-bad",
      "Use letters, digits, '.', '_' and '-'; start and end with a letter or digit.",
    ],
  ])("%j -> %j", (input, expected) => {
    expect(validateDatasetName(input)).toBe(expected);
  });
});

describe("validateDatasetRepoId (English frozen)", () => {
  it("accepts a bare name and one namespace segment", () => {
    expect(validateDatasetRepoId("my_dataset")).toBeNull();
    expect(validateDatasetRepoId("me/my_dataset")).toBeNull();
  });

  it("rejects more than one slash", () => {
    expect(validateDatasetRepoId("a/b/c")).toBe(
      "Dataset name may contain at most one '/' (namespace/name).",
    );
  });

  it("reports a bad namespace with the Namespace noun", () => {
    expect(validateDatasetRepoId("-bad/name")).toBe(
      "Use letters, digits, '.', '_' and '-'; start and end with a letter or digit.",
    );
    expect(validateDatasetRepoId(" /name")).toBe("Namespace can't be empty.");
  });

  it("reports a bad name segment with the Dataset name noun", () => {
    expect(validateDatasetRepoId("ns/-bad")).toBe(
      "Use letters, digits, '.', '_' and '-'; start and end with a letter or digit.",
    );
  });
});

describe("structured issues", () => {
  it("tags the subject so the renderer can pick a whole sentence", () => {
    expect(datasetNameIssue("")).toEqual({ code: "empty", subject: "dataset" });
    expect(datasetRepoIdIssue(" /name")).toEqual({
      code: "empty",
      subject: "namespace",
    });
    expect(datasetNameIssue("x".repeat(97))).toEqual({
      code: "tooLong",
      subject: "dataset",
      max: 96,
    });
  });
});
