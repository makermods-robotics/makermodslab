import type { MergeStatus, MergeStartResult } from "@/lib/replayApi";

/** The merge-status poll cadence, matched to MergeDatasetsDialog's POLL_MS. */
const POLL_MS = 1500;

/** Give up after this many status reads fail back-to-back (~30s at POLL_MS).
 * The poll sits inside an awaited handleStart with no user escape, so it must
 * not spin forever when the backend has gone away. */
const MAX_CONSECUTIVE_STATUS_FAILURES = 20;

export interface RunTemporaryMergeDeps {
  /** Kick the merge off (the caller binds baseUrl / sources / weights / the
   * `temporary` flag). */
  startMerge: () => Promise<MergeStartResult>;
  /** Read the current merge-status singleton. */
  getStatus: () => Promise<MergeStatus>;
  /** Called with every status snapshot the poll reads, so the caller can drive
   * a <MergeProgress> readout. */
  onStatus?: (status: MergeStatus) => void;
  /** Swapped out in tests; defaults to a real timer. */
  sleep?: (ms: number) => Promise<void>;
  pollMs?: number;
}

/**
 * Merge datasets into a throwaway output and resolve to the repo id the backend
 * minted for it. Used by the Train panel's "Combine multiple datasets" mode:
 * phase one of a two-phase "merge, then train on the result" launch.
 *
 * Rejects — submitting nothing downstream — when the merge is refused up front
 * (e.g. sources differ by a droppable column: that path stays in the full Merge
 * dialog) or when the job itself ends in error.
 */
export async function runTemporaryMerge(
  deps: RunTemporaryMergeDeps,
): Promise<string> {
  const sleep =
    deps.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)));
  const pollMs = deps.pollMs ?? POLL_MS;

  const res = await deps.startMerge();
  if (!res.started) {
    throw new Error(res.message || "The merge could not be started.");
  }

  let consecutiveFailures = 0;
  for (;;) {
    await sleep(pollMs);
    let status: MergeStatus;
    try {
      status = await deps.getStatus();
    } catch {
      // Transient — the merge is still running; retry on the next tick, but
      // give up if the backend stays unreachable.
      if (++consecutiveFailures >= MAX_CONSECUTIVE_STATUS_FAILURES) {
        throw new Error("Lost contact with the merge — check the server.");
      }
      continue;
    }
    consecutiveFailures = 0;
    deps.onStatus?.(status);
    if (status.state === "done") {
      if (!status.output_repo_id) {
        throw new Error("The merge finished without naming an output dataset.");
      }
      return status.output_repo_id;
    }
    if (status.state === "error") {
      throw new Error(status.error || "The merge failed.");
    }
  }
}
