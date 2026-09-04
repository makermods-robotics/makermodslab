import { describe, expect, it } from "vitest";
import { ApiError } from "@/lib/apiClient";
import type { DatasetInfo, DatasetTask } from "@/lib/replayApi";
import {
  classifyTaskLookup,
  defaultTaskFrom,
  effectiveTaskFor,
  rankDatasetTasks,
  taskFieldVisible,
  taskIsAmbiguous,
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
