import React, { useCallback, useMemo, useState } from "react";
import { ChevronRight, RefreshCw } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import LibraryToolbar from "@/components/library/LibraryToolbar";
import CappedGrid, {
  GRID_MIN_H,
  GRID_ROW_MIN_H,
} from "@/components/library/CappedGrid";
import LibraryHeader from "@/components/library/LibraryHeader";
import { SLIDE } from "@/components/studio/panel/primitives";
import { useStudio } from "@/contexts/StudioContext";
import JobCard from "./JobCard";
import HubJobCard from "./HubJobCard";
import { isJobActive, useJobsData } from "./JobsDataContext";
import { HubJob, JobRecord, isHubJobActive } from "@/lib/jobsApi";

/** Recency keys (ms) for the mixed local/cloud/hub grid — every library is
 * ordered newest-first regardless of where a run lives. */
const jobTime = (j: JobRecord) => (j.started_at ?? 0) * 1000;
const hubTime = (h: HubJob) =>
  h.created_at ? Date.parse(h.created_at) || 0 : 0;

/** Where a job runs: everything, this machine, or Hugging Face cloud/Hub. */
type JobsFilter = "all" | "local" | "online";

const FILTERS: Array<{ key: JobsFilter; label: string }> = [
  { key: "all", label: "All" },
  { key: "local", label: "Local" },
  { key: "online", label: "Online" },
];

interface JobsLibraryProps {
  /** Controlled fold state so the Train panel can collapse the library while
   * the new-training form is open (mirrors Collect's dataset library). */
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/**
 * Training-jobs library for the studio Train panel: search + location filter
 * over a three-up grid of active runs (each card carries its own Local/Cloud
 * chip), with inactive leftovers folded under Untracked. The models column
 * that used to sit beside this now lives in the Deploy panel (ModelsLibrary)
 * — a model artifact is deployed, not trained.
 */
const JobsLibrary: React.FC<JobsLibraryProps> = ({ open, onOpenChange }) => {
  const {
    localJobs,
    trackedCloudJobs,
    untrackedHubJobs,
    supersededIds,
    ancestorsOf,
    hubAuthenticated,
    hubJobsPermission,
    error,
    hubError,
    refresh,
    stop,
    remove,
    dismissHub,
  } = useJobsData();

  // Run on a job card doesn't open a dialog: it prefills the Deploy panel's
  // skill/checkpoint picker and focuses that panel.
  const { openStudio } = useStudio();
  const handlePlay = useCallback(
    (job: JobRecord, step: number) =>
      openStudio("deploy", { deploy: { source: "job", id: job.id, step } }),
    [openStudio],
  );

  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<JobsFilter>("all");
  // Lifted so the main grid can stop reserving its blank second row while
  // Untracked is expanded — the untracked cards flow right below the active
  // ones instead of after a gap.
  const [untrackedOpen, setUntrackedOpen] = useState(false);
  // Lifted from the active grid so Untracked can fold under "Show all" rather
  // than stacking a second dashed button beneath it: when the active runs
  // overflow one row, Untracked appears only once "Show all" is expanded.
  const [activeExpanded, setActiveExpanded] = useState(false);
  const [activeOverflow, setActiveOverflow] = useState(false);
  const query = search.trim().toLowerCase();
  const matchesQuery = useCallback(
    (text: string | null | undefined) =>
      !query || (text ?? "").toLowerCase().includes(query),
    [query],
  );

  const showLocal = filter !== "online";
  const showOnline = filter !== "local";

  // Match on the display alias as well as the original name, so a renamed
  // model is findable by either.
  const filteredLocal = useMemo(
    () =>
      showLocal
        ? localJobs.filter(
            (j) => matchesQuery(j.name) || matchesQuery(j.display_name),
          )
        : [],
    [localJobs, matchesQuery, showLocal],
  );
  const filteredCloud = useMemo(
    () =>
      showOnline
        ? trackedCloudJobs.filter(
            (j) => matchesQuery(j.name) || matchesQuery(j.display_name),
          )
        : [],
    [trackedCloudJobs, matchesQuery, showOnline],
  );
  const filteredHub = useMemo(
    () =>
      showOnline
        ? untrackedHubJobs.filter((h) =>
            matchesQuery(h.docker_image ?? h.space_id ?? h.id),
          )
        : [],
    [untrackedHubJobs, matchesQuery, showOnline],
  );

  // Active = running or has runnable checkpoints. Everything else collapses
  // under UNTRACKED so the eye lands on what's still relevant. Superseded runs
  // are dropped from both — they surface nested under their successor instead.
  const localActive = useMemo(
    () => filteredLocal.filter((j) => isJobActive(j) && !supersededIds.has(j.id)),
    [filteredLocal, supersededIds],
  );
  const localUntracked = useMemo(
    () =>
      filteredLocal.filter((j) => !isJobActive(j) && !supersededIds.has(j.id)),
    [filteredLocal, supersededIds],
  );
  const cloudActive = useMemo(
    () => filteredCloud.filter(isJobActive),
    [filteredCloud],
  );
  const cloudUntracked = useMemo(
    () => filteredCloud.filter((j) => !isJobActive(j)),
    [filteredCloud],
  );
  const hubActive = useMemo(
    () => filteredHub.filter(isHubJobActive),
    [filteredHub],
  );
  const hubInactive = useMemo(
    () => filteredHub.filter((h) => !isHubJobActive(h)),
    [filteredHub],
  );

  const activeCount = localActive.length + cloudActive.length + hubActive.length;
  const untrackedCount =
    localUntracked.length + cloudUntracked.length + hubInactive.length;

  // When the active runs spill past the single reserved row the grid shows its
  // own "Show all" toggle; Untracked then folds under it, revealed only once
  // "Show all" is open, so the collapsed library shows just one footer button.
  // With no overflow there is no "Show all", so Untracked stays the sole footer.
  const showUntrackedToggle =
    untrackedCount > 0 && (!activeOverflow || activeExpanded);

  const emptyMessage = query
    ? "No jobs match your search."
    : filter === "local"
      ? "No active local jobs."
      : filter === "online"
        ? "No active online jobs."
        : "No active training jobs.";

  // Nothing in the library at all, before any search or filter. The dataset and
  // skills libraries answer that with a single dashed GRID_MIN_H box and no
  // toolbar; jobs must too, or this panel's library is a toolbar taller than
  // theirs and its header, search bar, and Start button sit above the other
  // panels' (all three libraries are pinned to the panel foot, so extra height
  // pushes everything up).
  const isEmpty =
    localJobs.length === 0 &&
    trackedCloudJobs.length === 0 &&
    untrackedHubJobs.length === 0;

  // Why an empty library is empty, said inside the box instead of as a line
  // above it — an extra line would make the panel taller again.
  const emptyHint = !showOnline
    ? ""
    : hubError
      ? ""
      : !hubAuthenticated
        ? " Sign in with Hugging Face to see your cloud jobs."
        : !hubJobsPermission
          ? " Your Hugging Face token is missing the job.read permission, so cloud jobs can't be listed."
          : "";

  return (
    <Collapsible open={open} onOpenChange={onOpenChange} className="space-y-3">
      <LibraryHeader
        title="Training jobs"
        count={activeCount}
        open={open}
        actions={
          <button
            type="button"
            onClick={refresh}
            aria-label="Refresh job list"
            title="Refresh job list"
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded text-muted-foreground hover:text-foreground"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        }
      />

      <CollapsibleContent className={SLIDE}>
        <div className="space-y-3">
          {isEmpty ? null : (
            <LibraryToolbar
              query={search}
              onQueryChange={setSearch}
              searchPlaceholder="Search jobs"
              filters={FILTERS}
              filter={filter}
              onFilterChange={setFilter}
            />
          )}

          {error ? (
            <p className="text-sm text-destructive">
              Couldn't load local jobs: {error}
            </p>
          ) : null}
          {showOnline && hubError ? (
            <p className="text-sm text-destructive">
              Couldn't load cloud jobs: {hubError}
            </p>
          ) : null}
          {!isEmpty && showOnline && !hubError && !hubAuthenticated &&
          trackedCloudJobs.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Sign in with Hugging Face to see your cloud jobs.
            </p>
          ) : null}
          {!isEmpty && showOnline && hubAuthenticated && !hubJobsPermission ? (
            <p className="text-sm text-warn">
              Your Hugging Face token is missing the{" "}
              <code className="text-warn">job.read</code> permission, so cloud
              jobs can't be listed.
            </p>
          ) : null}

          {/* Active runs — local and online merged newest-first in one grid
              (two rows; the rest behind Show all); each card's location chip
              says where it runs. The grid and the Untracked footer row share
              one space-y-2 stack so the footer sits exactly where the other
              libraries' "Show all" row sits. */}
          {isEmpty ? (
            <div
              className={cn(
                "flex items-center justify-center rounded-md border border-dashed border-border px-4 py-6 text-center text-sm text-muted-foreground",
                GRID_MIN_H,
              )}
            >
              No training jobs yet. Start one above.{emptyHint}
            </div>
          ) : (
          <div className="space-y-2">
          {activeCount === 0 ? (
            // Only the card-row height when the Untracked toggle renders its
            // own footer row below — the two together then equal one
            // GRID_MIN_H block, so this library lines up with Collect's and
            // Deploy's instead of standing 2.375rem taller.
            <p
              className={cn(
                "flex items-center justify-center text-sm text-muted-foreground",
                showUntrackedToggle ? GRID_ROW_MIN_H : GRID_MIN_H,
              )}
            >
              {emptyMessage}
            </p>
          ) : (
            <CappedGrid
              reserveRows={!untrackedOpen}
              footerSpacer={untrackedCount === 0}
              expanded={activeExpanded}
              onExpandedChange={setActiveExpanded}
              onOverflowChange={setActiveOverflow}
              items={[
                ...[...localActive, ...cloudActive].map((job) => ({
                  time: jobTime(job),
                  node: (
                    <JobCard
                      key={job.id}
                      job={job}
                      onStop={stop}
                      onDelete={remove}
                      onPlay={handlePlay}
                      onRenamed={refresh}
                      ancestors={ancestorsOf(job)}
                    />
                  ),
                })),
                ...hubActive.map((job) => ({
                  time: hubTime(job),
                  node: (
                    <HubJobCard key={job.id} job={job} onDismiss={dismissHub} />
                  ),
                })),
              ]
                .sort((a, b) => b.time - a.time)
                .map((e) => e.node)}
            />
          )}

          {/* Inactive leftovers, folded away by default — the trigger is the
              jobs library's footer row, styled like "Show all". Hidden until
              the active grid's "Show all" is open when there is overflow, so
              the collapsed library never stacks two dashed footer buttons. */}
          {showUntrackedToggle ? (
            <Collapsible open={untrackedOpen} onOpenChange={setUntrackedOpen}>
              <CollapsibleTrigger className="group flex h-[1.875rem] w-full items-center justify-center gap-1 rounded-md border border-dashed border-border text-xs font-medium text-muted-foreground transition-colors hover:border-muted-foreground/40 hover:text-foreground">
                <ChevronRight className="h-3.5 w-3.5 transition-transform group-data-[state=open]:rotate-90" />
                Untracked ({untrackedCount})
              </CollapsibleTrigger>
              <CollapsibleContent className={cn(SLIDE, "pt-2")}>
                <CappedGrid
                  reserveRows={false}
                  items={[
                    ...[...localUntracked, ...cloudUntracked].map((job) => ({
                      time: jobTime(job),
                      node: (
                        <JobCard
                          key={job.id}
                          job={job}
                          onStop={stop}
                          onDelete={remove}
                          onPlay={handlePlay}
                          onRenamed={refresh}
                          ancestors={ancestorsOf(job)}
                        />
                      ),
                    })),
                    ...hubInactive.map((job) => ({
                      time: hubTime(job),
                      node: (
                        <HubJobCard
                          key={job.id}
                          job={job}
                          onDismiss={dismissHub}
                        />
                      ),
                    })),
                  ]
                    .sort((a, b) => b.time - a.time)
                    .map((e) => e.node)}
                />
              </CollapsibleContent>
            </Collapsible>
          ) : null}
          </div>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
};

export default JobsLibrary;
