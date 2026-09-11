import { describe, expect, it } from "vitest";
import { ApiError } from "@/lib/apiClient";
import type { DatasetInfo, DatasetTask } from "@/lib/replayApi";
import {
  classifyTaskLookup,
  defaultTaskFrom,
  effectiveTaskFor,
  loadingDots,
  loadingWordKey,
  rankDatasetTasks,
  TASK_LOADING_WORD_KEYS,
  TASK_LOADING_DOT_MS,
  TASK_LOADING_MAX_MS,
  taskFieldVisible,
  taskIsAmbiguous,
  taskPlaceholderKey,
  taskPrefillRepoId,
  tasksFrom,
  type TaskPrefillState,
} from "./deployTaskPrefill";

const task = (t: string, n: number | null): DatasetTask => ({
  task: t,
  num_episodes: n,
});

const info = (tasks: DatasetTask[]): DatasetInfo =>
  ({ repo_id: "u/d", tasks }) as DatasetInfo;

describe("rankDatasetTasks", () => {
  it("puts the most-represented task first when every count is known", () => {
    expect(
      rankDatasetTasks([task("rare", 3), task("common", 90), task("mid", 20)]),
    ).toEqual(["common", "mid", "rare"]);
  });

  it("does NOT reorder when any count is unknown", () => {
    // The server's order is task_index. A null is "we couldn't read the episode
    // metadata" — treating it as 0 would let an unreadable file decide the
    // ranking while every number still looked plausible.
    expect(
      rankDatasetTasks([task("first", null), task("second", 100)]),
    ).toEqual(["first", "second"]);
  });

  it("does not mutate its input", () => {
    const tasks = [task("a", 1), task("b", 2)];
    rankDatasetTasks(tasks);
    expect(tasks.map((t) => t.task)).toEqual(["a", "b"]);
  });

  it("drops empty task strings", () => {
    expect(rankDatasetTasks([task("", 5), task("real", 1)])).toEqual(["real"]);
  });
});

describe("taskPrefillRepoId", () => {
  it("prefers the checkpoint's own dataset over the job record's", () => {
    expect(taskPrefillRepoId("checkpoint/repo", "record/repo")).toBe(
      "checkpoint/repo",
    );
  });

  it("falls back to the job record for an older backend with no field", () => {
    expect(taskPrefillRepoId(undefined, "record/repo")).toBe("record/repo");
    expect(taskPrefillRepoId(null, "record/repo")).toBe("record/repo");
  });

  it("is null with nothing to fall back to", () => {
    expect(taskPrefillRepoId(undefined, undefined)).toBeNull();
    expect(taskPrefillRepoId(null, null)).toBeNull();
  });

  it("never resolves the import placeholder", () => {
    expect(taskPrefillRepoId(undefined, "(imported)")).toBeNull();
  });

  it("returns an EQUAL string across two different-identity checkpoints on the", () => {
    // same job — the property the Deploy panel's prefill effect relies on to
    // avoid re-fetching on every checkpoint step: every checkpoint of one job
    // shares its training dataset, so a fresh policyConfig object with the
    // same dataset_repo_id must resolve to the same lookup target.
    const stepA = taskPrefillRepoId("shared/repo", undefined);
    const stepB = taskPrefillRepoId("shared/repo", undefined);
    expect(stepA).toBe(stepB);
    expect(stepA).toBe("shared/repo");
  });
});

describe("classifyTaskLookup", () => {
  it("reports a dataset that lists no task as loaded-and-empty", () => {
    // The ONLY case that may claim "no task found on the training dataset".
    expect(classifyTaskLookup(info([]))).toEqual({ kind: "loaded", tasks: [] });
  });

  it("distinguishes a missing dataset from an unreachable one", () => {
    expect(classifyTaskLookup(new ApiError("gone", 404, null), true)).toEqual({
      kind: "unknown",
      reason: "not_found",
    });
    expect(classifyTaskLookup(new ApiError("boom", 500, null), true)).toEqual({
      kind: "unknown",
      reason: "unreachable",
    });
    expect(classifyTaskLookup(new TypeError("offline"), true)).toEqual({
      kind: "unknown",
      reason: "unreachable",
    });
  });

  it("treats a null hub task list as unknown, not as no-task", () => {
    // The server sends tasks: null when it could not read the task file at all
    // (a blip, an HTTP 5xx) — as opposed to [] for a dataset that genuinely
    // lists none. Rendering null as "no task found" is the exact false claim
    // this module exists to stop making.
    expect(
      classifyTaskLookup({ repo_id: "u/d", tasks: null } as unknown as DatasetInfo),
    ).toEqual({ kind: "unknown", reason: "unreachable" });
  });
});

describe("defaultTaskFrom", () => {
  it("offers the single task", () => {
    expect(defaultTaskFrom({ kind: "loaded", tasks: ["only"] })).toBe("only");
  });

  it("offers NOTHING when several tasks are on the table", () => {
    // Sending one silently is how a coaching dataset ends up labelled with a
    // sentence nobody chose — the measured margin between two near-identical
    // task strings on a real merged dataset is one episode.
    expect(defaultTaskFrom({ kind: "loaded", tasks: ["a", "b"] })).toBe("");
  });

  it("never contributes a default from a failed lookup", () => {
    expect(
      defaultTaskFrom({ kind: "unknown", reason: "not_found" }),
    ).toBe("");
    expect(defaultTaskFrom({ kind: "idle" })).toBe("");
  });
});

describe("taskIsAmbiguous", () => {
  it("is true only while several tasks are unresolved by the operator", () => {
    const many: TaskPrefillState = { kind: "loaded", tasks: ["a", "b"] };
    expect(taskIsAmbiguous(many, "")).toBe(true);
    expect(taskIsAmbiguous(many, "   ")).toBe(true);
    expect(taskIsAmbiguous(many, "a")).toBe(false);
  });

  it("is false for a single task or no tasks", () => {
    expect(taskIsAmbiguous({ kind: "loaded", tasks: ["only"] }, "")).toBe(false);
    expect(taskIsAmbiguous({ kind: "loaded", tasks: [] }, "")).toBe(false);
    expect(taskIsAmbiguous({ kind: "unknown", reason: "unreachable" }, "")).toBe(
      false,
    );
  });
});

describe("taskFieldVisible", () => {
  it("shows for a language-conditioned policy, and for coaching regardless", () => {
    expect(taskFieldVisible(true, "single")).toBe(true);
    expect(taskFieldVisible(true, "eval")).toBe(true);
    expect(taskFieldVisible(false, "coach")).toBe(true);
  });

  it("hides for a policy that does not read the task", () => {
    expect(taskFieldVisible(false, "single")).toBe(false);
    expect(taskFieldVisible(false, "eval")).toBe(false);
  });
});

describe("effectiveTaskFor", () => {
  const one: TaskPrefillState = { kind: "loaded", tasks: ["trained sentence"] };

  it("sends the typed sentence over the suggestion", () => {
    expect(effectiveTaskFor("mine", one, true, "single")).toBe("mine");
  });

  it("falls back to the suggestion when the box is empty", () => {
    expect(effectiveTaskFor("  ", one, true, "single")).toBe(
      "trained sentence",
    );
  });

  it("sends NOTHING when the field was never on screen", () => {
    // A value the operator could not see, confirm or correct has no business
    // reaching lerobot's `--task=`.
    expect(effectiveTaskFor("", one, false, "single")).toBe("");
    expect(effectiveTaskFor("", one, false, "eval")).toBe("");
    // ...but coaching always shows it, so it is sent there.
    expect(effectiveTaskFor("", one, false, "coach")).toBe("trained sentence");
  });

  it("sends nothing when several tasks are unresolved", () => {
    const many: TaskPrefillState = { kind: "loaded", tasks: ["a", "b"] };
    expect(effectiveTaskFor("", many, true, "single")).toBe("");
  });
});

describe("tasksFrom", () => {
  it("is empty for every state but loaded", () => {
    expect(tasksFrom({ kind: "idle" })).toEqual([]);
    expect(tasksFrom({ kind: "unknown", reason: "not_found" })).toEqual([]);
    expect(tasksFrom({ kind: "loaded", tasks: ["a"] })).toEqual(["a"]);
  });
});

describe("loadingDots", () => {
  it("cycles one, two, three and back to one", () => {
    expect([0, 1, 2, 3, 4, 5, 6].map(loadingDots)).toEqual([
      ".",
      "..",
      "...",
      ".",
      "..",
      "...",
      ".",
    ]);
  });

  it("keeps cycling far from zero, so a long wait never breaks the pattern", () => {
    expect(loadingDots(1000)).toBe(loadingDots(1000 % 3));
    expect(loadingDots(999)).toBe(".");
  });
});

describe("loading timings", () => {
  it("cycles all three dots several times within the animation's lifetime", () => {
    // The dots have to read as an animation, not a flicker or a freeze: three
    // steps per cycle, and enough cycles before it gives up that the field
    // clearly looks busy rather than stuck.
    const cycles = TASK_LOADING_MAX_MS / (TASK_LOADING_DOT_MS * 3);
    expect(cycles).toBeGreaterThanOrEqual(4);
    expect(TASK_LOADING_DOT_MS).toBeGreaterThanOrEqual(200);
  });
});

describe("a lookup in flight contributes nothing", () => {
  it("offers no tasks, no default, and no ambiguity while loading", () => {
    // The field says it is loading; it must not also be quietly resolving to a
    // suggestion, or blocking the launch on a choice it cannot show yet.
    expect(tasksFrom({ kind: "loading" })).toEqual([]);
    expect(defaultTaskFrom({ kind: "loading" })).toBe("");
    expect(taskIsAmbiguous({ kind: "loading" }, "")).toBe(false);
  });

  it("still sends what the operator typed", () => {
    expect(effectiveTaskFor("mine", { kind: "loading" }, true, "single")).toBe(
      "mine",
    );
  });
});

describe("loadingWordKey", () => {
  it("holds each word for one full dot cycle", () => {
    // Three ticks per word: the dots must visibly animate underneath a word
    // that stays put long enough to actually be read.
    expect([0, 1, 2].map(loadingWordKey)).toEqual([
      TASK_LOADING_WORD_KEYS[0],
      TASK_LOADING_WORD_KEYS[0],
      TASK_LOADING_WORD_KEYS[0],
    ]);
    expect(loadingWordKey(3)).toBe(TASK_LOADING_WORD_KEYS[1]);
    expect(loadingWordKey(6)).toBe(TASK_LOADING_WORD_KEYS[2]);
  });

  it("starts on the plain one, so the field says what it is doing first", () => {
    expect(loadingWordKey(0)).toBe("studio.deploy.task.loading.loading");
  });

  it("wraps rather than running out on a long wait", () => {
    const cycle = TASK_LOADING_WORD_KEYS.length * 3;
    expect(loadingWordKey(cycle)).toBe(TASK_LOADING_WORD_KEYS[0]);
    expect(loadingWordKey(cycle + 3)).toBe(TASK_LOADING_WORD_KEYS[1]);
    expect(loadingWordKey(99999)).toBeDefined();
  });

  it("has no duplicate keys", () => {
    expect(new Set(TASK_LOADING_WORD_KEYS).size).toBe(
      TASK_LOADING_WORD_KEYS.length,
    );
  });
});

describe("taskPlaceholderKey", () => {
  // Loading's cycling words are the caller's job (loadingWordKey needs a
  // tick); everything else is a pure function of the state.
  it("is its own key for idle — never claims a dataset was checked", () => {
    // idle means no checkpoint dataset was even resolvable to attempt a
    // lookup with. Falling through to placeholderNone would claim the server
    // looked and found nothing, which never happened.
    expect(taskPlaceholderKey({ kind: "idle" }, false)).toBe(
      "studio.deploy.task.placeholderUnresolved",
    );
  });

  it("distinguishes a missing dataset from an unreadable one", () => {
    expect(taskPlaceholderKey({ kind: "unknown", reason: "not_found" }, false)).toBe(
      "studio.deploy.task.placeholderMissing",
    );
    expect(
      taskPlaceholderKey({ kind: "unknown", reason: "unreachable" }, false),
    ).toBe("studio.deploy.task.placeholderUnreadable");
  });

  it("offers the choose prompt only while loaded-and-ambiguous", () => {
    expect(taskPlaceholderKey({ kind: "loaded", tasks: ["a", "b"] }, true)).toBe(
      "studio.deploy.task.placeholderChoose",
    );
  });

  it("falls back to placeholderNone only for a genuinely empty loaded list", () => {
    expect(taskPlaceholderKey({ kind: "loaded", tasks: [] }, false)).toBe(
      "studio.deploy.task.placeholderNone",
    );
  });

  it("uses placeholderSlow for a loading state (caller overrides while cycling)", () => {
    expect(taskPlaceholderKey({ kind: "loading" }, false)).toBe(
      "studio.deploy.task.placeholderSlow",
    );
  });
});
