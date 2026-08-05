import React, { useCallback, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { RefreshCw } from "lucide-react";
import { Collapsible, CollapsibleContent } from "@/components/ui/collapsible";
import LibraryToolbar from "@/components/library/LibraryToolbar";
import LibraryHeader from "@/components/library/LibraryHeader";
import { SLIDE } from "@/components/studio/panel/primitives";
import { useApi } from "@/contexts/ApiContext";
import { useStudio } from "@/contexts/StudioContext";
import { useToast } from "@/hooks/use-toast";
import { listJobCheckpoints } from "@/lib/checkpointsApi";
import JobCard from "./JobCard";
import HubJobCard from "./HubJobCard";
import JobsDropdown, { JobsEntry } from "./JobsDropdown";
import { isJobActive, useJobsData } from "./JobsDataContext";
import { HubJob, JobRecord, isHubJobActive, jobDisplayName } from "@/lib/jobsApi";

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
 * Run / Continue / Resume-from-step / Download / delete.
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
  const navigate = useNavigate();
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

  // Active = running or has runnable checkpoints. Everything else folds under
  // UNTRACKED inside the dropdown so the trigger lands on what's still
  // relevant. Superseded runs are dropped from both — they surface nested under
  // their successor's card instead.
  const toEntries = useCallback(
    (jobs: JobRecord[], hubs: HubJob[]): JobsEntry[] =>
      [
        ...jobs.map(
          (job): JobsEntry => ({
            kind: "job",
            key: jobEntryKey(job),
            time: jobTime(job),
            job,
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
    [],
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
        trackedRuns.filter(isJobActive),
        filteredHub.filter(isHubJobActive),
      ),
    [toEntries, trackedRuns, filteredHub],
  );
  const untrackedEntries = useMemo(
    () =>
      toEntries(
        trackedRuns.filter((j) => !isJobActive(j)),
        filteredHub.filter((h) => !isHubJobActive(h)),
      ),
    [toEntries, trackedRuns, filteredHub],
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

  // Resume from the row: resolve the run's newest checkpoint and hand the
  // Training page the same resume state the card's Continue/Resume does.
  // Choosing a *specific* step stays on the selected run's card below.
  const handleResume = useCallback(
    async (job: JobRecord) => {
      setResumingId(job.id);
      try {
        const cks = await listJobCheckpoints(baseUrl, fetchWithHeaders, job.id);
        const step = cks.length > 0 ? cks[cks.length - 1].step : null;
        if (step == null) {
          toast({
            title: "Nothing to resume from",
            description: "This run saved no checkpoint.",
            variant: "destructive",
          });
          return;
        }
        const target = job.config?.steps ?? 0;
        if (target > 0 && step >= target) {
          toast({
            title: "Nothing to resume",
            description: "This run already reached its step target.",
          });
          return;
        }
        const runner = job.runner === "hf_cloud" ? "hf_cloud" : "local";
        navigate("/training", {
          state: {
            resume: {
              jobId: job.id,
              step,
              name: jobDisplayName(job),
              datasetRepoId: job.config.dataset_repo_id,
              policyType: job.config.policy_type,
              sourceSteps: job.config.steps,
              logFreq: job.config.log_freq,
              saveFreq: job.config.save_freq,
              runner,
              flavor:
                runner === "hf_cloud" ? (job.hf_flavor ?? undefined) : undefined,
            },
          },
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
    [baseUrl, fetchWithHeaders, navigate, toast],
  );

  const emptyMessage = query
    ? "No jobs match your search."
    : filter === "local"
      ? "No local jobs."
      : filter === "online"
        ? "No online jobs."
        : "No training jobs yet.";

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
          <LibraryToolbar
            query={search}
            onQueryChange={setSearch}
            searchPlaceholder="Search jobs"
            filters={FILTERS}
            filter={filter}
            onFilterChange={setFilter}
          />

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
          {showOnline && !hubError && !hubAuthenticated &&
          trackedCloudJobs.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Sign in with Hugging Face to see your cloud jobs.
            </p>
          ) : null}
          {showOnline && hubAuthenticated && !hubJobsPermission ? (
            <p className="text-sm text-warn">
              Your Hugging Face token is missing the{" "}
              <code className="text-warn">job.read</code> permission, so cloud
              jobs can't be listed.
            </p>
          ) : null}

          {/* The run picker, then the selected run's detail card. Local and
              online runs share one list, newest-first inside their launched-by
              group; each row's Local/Cloud chip says where it runs. */}
          <div className="space-y-2">
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
            {selected ? (
              selected.kind === "job" ? (
                <JobCard
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
      </CollapsibleContent>
    </Collapsible>
  );
};

export default JobsLibrary;
