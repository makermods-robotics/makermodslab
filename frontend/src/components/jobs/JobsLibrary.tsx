import React, { useCallback, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Collapsible, CollapsibleContent } from "@/components/ui/collapsible";
import LibraryToolbar from "@/components/library/LibraryToolbar";
import LibraryHeader from "@/components/library/LibraryHeader";
import { GRID_H } from "@/components/library/CappedGrid";
import { SLIDE } from "@/components/studio/panel/primitives";
import { useApi } from "@/contexts/ApiContext";
import { useStudio } from "@/contexts/StudioContext";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";
import {
  buildResumeSeed,
  loadLineageCheckpoints,
  noResumeReason,
  resumableCheckpoints,
} from "./resumeSeed";
import type { NoResumeReason } from "./resumeSeed";
import JobCard from "./JobCard";
import HubJobCard from "./HubJobCard";
import JobsDropdown, { JobsEntry } from "./JobsDropdown";
import { useJobsData } from "./JobsDataContext";
import { HubJob, JobRecord, isHubJobActive } from "@/lib/jobsApi";

/** Recency keys (ms) for the mixed local/cloud/hub list — every library is
 * ordered newest-first regardless of where a run lives. */
const jobTime = (j: JobRecord) => (j.started_at ?? 0) * 1000;
const hubTime = (h: HubJob) =>
  h.created_at ? Date.parse(h.created_at) || 0 : 0;

/** Selection keys — a local/cloud run and a Hub-only job can share an id, so
 * the kind is part of the key. */
const jobEntryKey = (j: JobRecord) => `job:${j.id}`;
const hubEntryKey = (h: HubJob) => `hub:${h.id}`;

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
 * over a compact run dropdown. The collapsed trigger carries the most relevant
 * run (a live one beats the latest finished one) with its live progress; the
 * list gives one row per run with its state, where it ran, when, and the
 * row-level primary actions; and the selected run's card below the dropdown
 * keeps every other affordance — monitor, rename, checkpoint picker,
 * Run / Resume-from-step / Download / delete. (Run and Download are
 * model-shaped actions and move to ModelCard once ModelsLibrary is rewired to
 * render it; they still live on JobCard on this stack.)
 *
 * The models column that used to sit beside this lives in the Deploy panel
 * (ModelsLibrary) — a model artifact is deployed, not trained.
 */
const JobsLibrary: React.FC<JobsLibraryProps> = ({ open, onOpenChange }) => {
  const {
    localJobs,
    trackedCloudJobs,
    untrackedHubJobs,
    supersededIds,
    ancestorsOf,
    chainCheckpointCount,
    isJobActive,
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
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const handlePlay = useCallback(
    (job: JobRecord, step: number) =>
      openStudio("deploy", { deploy: { source: "job", id: job.id, step } }),
    [openStudio],
  );

  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<JobsFilter>("all");
  // Null until the user picks a row: the dropdown then follows the most
  // relevant run on its own (a running job, else the newest one), so a launch
  // moves the trigger to the new run without a click.
  const [pickedKey, setPickedKey] = useState<string | null>(null);
  const [resumingId, setResumingId] = useState<string | null>(null);
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

  // Active = running, or the CHAIN has a checkpoint to resume from (see
  // isJobActive). Everything else folds under UNTRACKED inside the dropdown so
  // the trigger lands on what's still relevant. Superseded runs are dropped
  // from both — they surface nested under their successor's card instead.
  const toEntries = useCallback(
    (jobs: JobRecord[], hubs: HubJob[]): JobsEntry[] =>
      [
        ...jobs.map(
          (job): JobsEntry => ({
            kind: "job",
            key: jobEntryKey(job),
            time: jobTime(job),
            job,
            // The row stands for the whole chain, so its Resume gate counts
            // the chain's checkpoints, not just the tip's. Counted by the
            // provider, which owns the ancestor records — and which files a
            // chain whose ancestors haven't landed yet as active regardless,
            // so a row can't sit in the fold while its count reads low.
            chainCheckpointCount: chainCheckpointCount(job),
          }),
        ),
        ...hubs.map(
          (job): JobsEntry => ({
            kind: "hub",
            key: hubEntryKey(job),
            time: hubTime(job),
            job,
          }),
        ),
      ].sort((a, b) => b.time - a.time),
    [chainCheckpointCount],
  );

  const trackedRuns = useMemo(
    () =>
      [...filteredLocal, ...filteredCloud].filter(
        (j) => !supersededIds.has(j.id),
      ),
    [filteredLocal, filteredCloud, supersededIds],
  );
  const activeEntries = useMemo(
    () =>
      toEntries(
        trackedRuns.filter((j) => isJobActive(j)),
        filteredHub.filter(isHubJobActive),
      ),
    [toEntries, trackedRuns, filteredHub, isJobActive],
  );
  const untrackedEntries = useMemo(
    () =>
      toEntries(
        trackedRuns.filter((j) => !isJobActive(j)),
        filteredHub.filter((h) => !isHubJobActive(h)),
      ),
    [toEntries, trackedRuns, filteredHub, isJobActive],
  );

  const activeCount = activeEntries.length;

  // Selection: the user's pick wins while it still exists; otherwise the most
  // relevant run — a running one first (newest), then the newest active run,
  // then the newest untracked leftover.
  const allEntries = useMemo(
    () => [...activeEntries, ...untrackedEntries],
    [activeEntries, untrackedEntries],
  );
  const autoKey = useMemo(() => {
    const isLive = (e: JobsEntry) =>
      e.kind === "job" ? e.job.state === "running" : isHubJobActive(e.job);
    return (
      activeEntries.find(isLive)?.key ??
      activeEntries[0]?.key ??
      untrackedEntries[0]?.key ??
      null
    );
  }, [activeEntries, untrackedEntries]);
  const selectedKey =
    pickedKey != null && allEntries.some((e) => e.key === pickedKey)
      ? pickedKey
      : autoKey;
  const selected = allEntries.find((e) => e.key === selectedKey) ?? null;

  // Resume from the row: resolve the chain's newest resumable checkpoint and
  // open the Train panel's form in resume mode with exactly the seed the
  // card's Resume produces. Same shared loader AND same shared rule,
  // so the two entry points can offer neither a different checkpoint nor a
  // different verdict (this path used to walk its own, thinner logic).
  // Choosing a *specific* step stays on the selected run's card below.
  const handleResume = useCallback(
    async (job: JobRecord) => {
      setResumingId(job.id);
      try {
        const lineage = await loadLineageCheckpoints(
          baseUrl,
          fetchWithHeaders,
          job,
          ancestorsOf(job),
        );
        // Newest first, and each entry knows which run owns it — so the seed
        // resumes from the run that actually holds the checkpoint, which on a
        // chain is often an ancestor rather than this row's own run.
        const best = resumableCheckpoints(job, lineage)[0];
        if (!best) {
          // The row's button gates on the CHEAP chain-wide count, so landing
          // here is normal — this is where the exact rule gets to explain
          // itself. The cause comes from the rule itself (`noResumeReason`)
          // rather than being re-guessed from the lineage here: guessing is
          // what produced the "already at its step target" message for a run
          // whose checkpoints were below its target and had been dropped by a
          // different filter entirely.
          const reason = noResumeReason(job, lineage);
          const description: Record<NoResumeReason, string> = {
            "not-resumable":
              "This run isn't in a state that can be continued.",
            "no-checkpoints":
              "This run and the runs it continues from saved no checkpoint.",
            "owner-done":
              "Every checkpoint this run can continue from belongs to a run that already " +
              "reached its target, so its learning-rate schedule is spent. Fine-tune from " +
              "the final checkpoint instead.",
            "at-target":
              "Every checkpoint this run can continue from is already at its step target. " +
              "Raise the target to continue, or fine-tune from the final checkpoint.",
            "sibling-cap":
              "Every checkpoint left in this run's lineage was saved past the step this run " +
              "reached, so it belongs to another continuation sharing the same cloud output.",
            other: "No checkpoint in this run's lineage can be resumed from.",
          };
          toast({
            title: "Nothing to resume from",
            description: description[reason],
            variant: reason === "no-checkpoints" ? "destructive" : "default",
          });
          return;
        }
        // KNOWN LIMIT, CLOUD-OWNED checkpoints only and now narrowed: the runs
        // of a cloud chain all publish to the parent's Hub repo, so a FORKED
        // SIBLING's checkpoints are in that listing too and nothing in them
        // says who wrote them. resumableCheckpoints drops every step above this
        // run's OWN furthest step — those provably belong to a sibling — so
        // what can still slip through is a sibling that forked early, inside
        // this run's range. Per-run attribution of Hub checkpoints is a backend
        // change. Locally-owned checkpoints are exact (each run owns its output
        // dir). Note the limit follows the checkpoint's OWNER, not this run's
        // runner: post-F7 a local run continuing a cloud parent lists that
        // parent's Hub repo too — see `cloudSiblingStepCap`.
        openStudio("train", {
          train: { resume: buildResumeSeed(best.job, best.ckpt.step) },
        });
      } catch (e) {
        toast({
          title: "Couldn't load checkpoints",
          description: e instanceof Error ? e.message : String(e),
          variant: "destructive",
        });
      } finally {
        setResumingId(null);
      }
    },
    [ancestorsOf, baseUrl, fetchWithHeaders, openStudio, toast],
  );

  const emptyMessage = query
    ? "No jobs match your search."
    : filter === "local"
      ? "No local jobs."
      : filter === "online"
        ? "No online jobs."
        : "No training jobs yet.";

  // First run: nothing anywhere, before any filter or search narrows anything
  // down (main, #79). Distinct from `emptyMessage`, which answers "your current
  // view is empty" inside the picker — here there is no view to have, so the
  // toolbar and the cloud-auth notices are suppressed too and one instruction
  // stands on its own. Deliberately reads the unfiltered source lists, so a
  // filter that hides every run still gets the picker with its own message
  // rather than this.
  const isEmpty =
    localJobs.length === 0 &&
    trackedCloudJobs.length === 0 &&
    untrackedHubJobs.length === 0;

  // The one thing worth saying on a first run that the instruction can't: why
  // the cloud half might be silent. Blank when there's nothing to explain.
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
          {!isEmpty &&
          showOnline &&
          !hubError &&
          !hubAuthenticated &&
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

          {/* The run picker, then the selected run's detail card. Local and
              online runs share one list, newest-first inside their launched-by
              group; each row's Local/Cloud chip says where it runs.

              The block is held at the libraries' one reserved height (GRID_H —
              the same measurement the dataset and model grids floor themselves
              at). That reservation is what puts the three studio panels'
              `mt-auto` action rows on one visual row, and this library lost it
              when its card grid became a dropdown: the Train panel's Start
              button then rose and fell with whether a run was selected and how
              tall its card was. Held in BOTH directions, so an empty selection
              reserves the same height a tall card does — the card scrolls
              inside the box rather than growing it. */}
          <div className={cn(GRID_H, "flex flex-col gap-2 overflow-hidden")}>
            {isEmpty ? (
              // Inside the reserved block, not instead of it: the height is
              // what keeps the three studio panels' action rows on one row, so
              // a first run has to hold it exactly like a populated one does.
              <div className="flex h-full items-center justify-center rounded-md border border-dashed border-border px-4 py-6 text-center text-sm text-muted-foreground">
                No training jobs yet. Start one above.{emptyHint}
              </div>
            ) : (
              <>
                <div className="shrink-0">
                  <JobsDropdown
                    entries={activeEntries}
                    untracked={untrackedEntries}
                    selectedKey={selectedKey}
                    onSelect={(entry) => setPickedKey(entry.key)}
                    onStop={stop}
                    onResume={handleResume}
                    onDismissHub={dismissHub}
                    resumingId={resumingId}
                    emptyMessage={emptyMessage}
                  />
                </div>
                {/* Overflow is spent HERE, on the detail card, never on the page:
                the region takes whatever the dropdown leaves and scrolls its
                own content. The inner wrapper is load-bearing — it leaves the
                region's height indefinite for the card's own `h-full`, so a
                card taller than the box hugs its content and scrolls whole
                instead of being cut off at the box's edge. */}
                <div className="min-h-0 flex-1 overflow-y-auto">
                  <div>
                    {selected ? (
                      selected.kind === "job" ? (
                        <JobCard
                          // Remount on every run switch: JobCard holds per-run
                          // state (its lineage checkpoint list and the selected
                          // checkpoint ref) that its fetch effect only replaces
                          // once the new run's fetch resolves. Without a key the
                          // instance is reused and, in that window, Run /
                          // Resume / Fine-tune / Download would act on the
                          // PREVIOUS run while the header already shows the
                          // new one.
                          key={selected.key}
                          job={selected.job}
                          onStop={stop}
                          onDelete={remove}
                          onPlay={handlePlay}
                          onRenamed={refresh}
                          ancestors={ancestorsOf(selected.job)}
                        />
                      ) : (
                        <HubJobCard job={selected.job} onDismiss={dismissHub} />
                      )
                    ) : null}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
};

export default JobsLibrary;
