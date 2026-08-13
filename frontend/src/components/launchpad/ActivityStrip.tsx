import React, { useCallback, useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";

import { useApi } from "@/contexts/ApiContext";
import { useStudio } from "@/contexts/StudioContext";
import { useJobsChangedSignal } from "@/hooks/useJobsChangedSignal";
import {
  JobProgressSnapshot,
  JobRecord,
  jobDisplayName,
  listJobs,
} from "@/lib/jobsApi";
import { cn } from "@/lib/utils";

/**
 * Compact strip of active (running) training jobs on the Launchpad. Live via
 * the same jobs polling + `useJobsChangedSignal` socket the Train panel's
 * JobsSection uses: `jobs_changed` refetches the list, `job_progress` patches
 * progress in place. Click a job → its monitor dialog over the studio's
 * Train panel. Renders nothing when no
 * job is active — queued/finished jobs live in the Train panel's full section.
 */
const ActivityStrip: React.FC = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { openJobMonitor } = useStudio();
  const [jobs, setJobs] = useState<JobRecord[]>([]);

  // Returns the fetch so callers can sequence on it — the heal in
  // applyProgress needs to know when its refetch has landed.
  const refresh = useCallback(() => {
    // Pull a generous slice so a running job isn't masked by newer records in
    // the started_at-desc ordering, then keep only what's live.
    return listJobs(baseUrl, fetchWithHeaders, 200)
      .then((j) => setJobs(j.filter((r) => r.state === "running")))
      .catch(() => setJobs([]));
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Guards the reconciling refetch below — same pair, and same rationale, as
  // JobsDataContext's: `healingRef` stops the ~1Hz ticks that land while a
  // refetch is in flight from stacking up identical fetches, and `healedRef`
  // caps each unknown id at one refetch so an id the refetch CAN'T produce
  // doesn't refetch forever. /jobs is a capped page here too (200), so a long
  // run buried under enough newer records is legitimately absent from the list
  // while still reporting progress.
  const healingRef = useRef(false);
  const healedRef = useRef<Set<string>>(new Set());

  // Patch progress in place from the watchdog's ~1Hz snapshots (no refetch),
  // and drop any job that just left the running state.
  const applyProgress = useCallback(
    (snapshots: JobProgressSnapshot[]) => {
      if (snapshots.length === 0) return;
      // A snapshot names a RUNNING job, so an id this strip doesn't hold is a
      // live run it never learned about: the `jobs_changed` that announced it
      // was missed (the broadcast is fire-and-forget, and a dead socket never
      // fires onclose). Dropping it left a running job off the strip until the
      // next mount. Refetch once instead — the watchdog re-announces every
      // running job each tick, so this heals within about a second.
      const known = new Set(jobs.map((j) => j.id));
      const unseen = snapshots
        .map((s) => s.id)
        .filter((id) => !known.has(id) && !healedRef.current.has(id));
      if (unseen.length > 0 && !healingRef.current) {
        for (const id of unseen) healedRef.current.add(id);
        healingRef.current = true;
        refresh().finally(() => {
          healingRef.current = false;
        });
      }
      setJobs((prev) => {
        if (prev.length === 0) return prev;
        const byId = new Map(snapshots.map((s) => [s.id, s]));
        return prev
          .map((j) => {
            const s = byId.get(j.id);
            return s ? { ...j, state: s.state, metrics: s.metrics } : j;
          })
          .filter((j) => j.state === "running");
      });
    },
    [jobs, refresh],
  );

  useJobsChangedSignal(refresh, applyProgress);

  if (jobs.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-2">
      {jobs.map((job) => {
        const { current_step, total_steps } = job.metrics;
        const pct =
          total_steps > 0
            ? Math.min(100, Math.round((current_step / total_steps) * 100))
            : null;
        return (
          <button
            key={job.id}
            type="button"
            onClick={() => openJobMonitor(job.id)}
            className={cn(
              "group flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5",
              "text-sm text-foreground transition-colors hover:bg-muted/60",
            )}
          >
            <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
            <span className="max-w-[12rem] truncate font-medium">
              {jobDisplayName(job)}
            </span>
            {pct != null ? (
              <span className="tabular-nums text-xs text-muted-foreground">
                {pct}%
              </span>
            ) : null}
            <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary">
              {job.state}
            </span>
          </button>
        );
      })}
    </div>
  );
};

export default ActivityStrip;
