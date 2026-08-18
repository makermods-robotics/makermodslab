import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useTranslation } from "react-i18next";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useJobsChangedSignal } from "@/hooks/useJobsChangedSignal";
import {
  HubJob,
  HubModel,
  JobProgressSnapshot,
  JobRecord,
  deleteJob,
  dismissHubJob,
  getJob,
  listHubJobs,
  listJobs,
  stopJob,
} from "@/lib/jobsApi";

const LIMIT = 10;

interface JobsDataValue {
  /** Local job registry page (trainings, cloud mirrors, imports). */
  jobs: JobRecord[];
  localJobs: JobRecord[];
  trackedCloudJobs: JobRecord[];
  importedJobs: JobRecord[];
  /** Hub jobs with no mirroring local record. */
  untrackedHubJobs: HubJob[];
  /** Uploaded hub model repos no job (cloud run or import) tracks. */
  untrackedHubModels: HubModel[];
  /** Parents hidden from top-level lists because a successor resumed them. */
  supersededIds: Set<string>;
  /** Resume lineage of a job, nearest parent first. */
  ancestorsOf: (job: JobRecord) => JobRecord[];
  /** Checkpoints reachable from a job: its own plus its loaded ancestors'.
   *
   * The libraries' Resume gate, because a run continues from the newest
   * checkpoint on its LINEAGE — so a tip that died before saving anything is
   * resumable on its ancestors' checkpoints, and its own `checkpoint_count`
   * of 0 would hide the button on the commonest resumable shape there is.
   * Lives on the context because only the provider holds the ancestor records. */
  chainCheckpointCount: (job: JobRecord) => number;
  /** Running, or with a checkpoint anywhere in its chain — i.e. worth showing
   * outside the libraries' UNTRACKED fold. Same chain-wide reading as the gate
   * above, and now for the same reason. */
  isJobActive: (job: JobRecord) => boolean;
  hubAuthenticated: boolean;
  hubJobsPermission: boolean;
  error: string | null;
  hubError: string | null;
  refresh: () => Promise<void>;
  stop: (id: string) => Promise<void>;
  remove: (id: string) => Promise<void>;
  dismissHub: (id: string) => Promise<void>;
}

const JobsDataContext = createContext<JobsDataValue | null>(null);

/**
 * Shared job/model registry state for the studio panels. Extracted from the
 * old JobsSection so the Train panel's jobs library and the Deploy panel's
 * model library read one fetch + one WS subscription instead of duplicating
 * the /jobs + Hub round-trips (the Hub listing is rate-limit sensitive).
 */
export const JobsDataProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { t } = useTranslation();

  const [jobs, setJobs] = useState<JobRecord[]>([]);
  // Ancestors referenced via resume_from_job_id but paged out of the list, so a
  // resumed run can still nest its source even when the source is old.
  const [ancestorCache, setAncestorCache] = useState<Record<string, JobRecord>>(
    {},
  );
  const [hubJobs, setHubJobs] = useState<HubJob[]>([]);
  const [hubModels, setHubModels] = useState<HubModel[]>([]);
  const [hubAuthenticated, setHubAuthenticated] = useState(false);
  const [hubJobsPermission, setHubJobsPermission] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hubError, setHubError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    // Settle the two fetches independently: a hub failure (network, HF outage,
    // missing scope) must never blank the local jobs, and vice versa.
    const [localRes, hubRes] = await Promise.allSettled([
      listJobs(baseUrl, fetchWithHeaders, LIMIT),
      listHubJobs(baseUrl, fetchWithHeaders),
    ]);
    if (localRes.status === "fulfilled") {
      setJobs(localRes.value);
      setError(null);
    } else {
      const r = localRes.reason;
      setError(r instanceof Error ? r.message : String(r));
    }
    if (hubRes.status === "fulfilled") {
      setHubJobs(hubRes.value.jobs);
      setHubModels(hubRes.value.models);
      setHubAuthenticated(hubRes.value.authenticated);
      setHubJobsPermission(hubRes.value.jobs_permission ?? true);
      setHubError(null);
    } else {
      const r = hubRes.reason;
      setHubError(r instanceof Error ? r.message : String(r));
    }
  }, [baseUrl, fetchWithHeaders]);

  // Initial fetch on mount + refetch when the tab regains focus. Backend
  // pushes a `jobs_changed` WS event on every registry mutation, which
  // covers any change originating on this machine. The focus refresh
  // catches changes originating elsewhere (e.g. a job submitted from
  // another tab or the HF dashboard) without burning the rate limit.
  useEffect(() => {
    refresh();
    const onVisible = () => {
      if (document.visibilityState === "visible") refresh();
    };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", refresh);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("focus", refresh);
    };
  }, [refresh]);

  // Guards the reconciling refetch below. `healingRef` keeps the ~1Hz ticks
  // that land while a refetch is in flight from stacking up identical fetches;
  // `healedRef` caps each unknown id at one refetch for the life of the
  // provider, so an id the refetch CAN'T produce doesn't refetch forever. That
  // is a real case, not a hypothetical: /jobs is a capped page (LIMIT), so a
  // long run with enough newer records after it is legitimately absent from
  // the list while still reporting progress.
  const healingRef = useRef(false);
  const healedRef = useRef<Set<string>>(new Set());

  const applyProgress = useCallback(
    (snapshots: JobProgressSnapshot[]) => {
      if (snapshots.length === 0) return;
      // A snapshot for a job this provider has never heard of means the
      // `jobs_changed` that announced it never landed — the broadcast is
      // fire-and-forget (the server drops it outright when no connection is
      // registered, and a socket can die without ever firing onclose), and
      // nothing else re-reads /jobs while the studio stays mounted. Patching
      // only known ids and silently dropping the rest is what made a run
      // started from the Train panel's in-place form invisible until a page
      // reload. Reconcile instead: the watchdog re-announces every running
      // job each tick, so one refetch heals the list within about a second of
      // the miss, whatever the miss's origin (this tab, another tab, the CLI).
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
        let mutated = false;
        const next = prev.map((j) => {
          const s = byId.get(j.id);
          if (!s) return j;
          mutated = true;
          return {
            ...j,
            state: s.state,
            metrics: s.metrics,
            wandb_run_url: s.wandb_run_url,
            checkpoint_count: s.checkpoint_count,
          };
        });
        return mutated ? next : prev;
      });
    },
    [jobs, refresh],
  );

  useJobsChangedSignal(refresh, applyProgress);

  // Backfill the resume ancestors that aren't in the loaded page (or already
  // cached), so a chain nests regardless of how old its source runs are.
  //
  // The server hands each record its whole `ancestor_ids` closure, so this is
  // one parallel fan-out over known-missing ids — not the old hop-by-hop walk,
  // which had to fetch a parent before it could learn about a grandparent and
  // so paid one serial round-trip per generation. Idempotent: only unseen ids
  // are fetched, so the frequent list refreshes during a run don't re-fetch.
  // A fetch that fails (the run was deleted between the list and this call) is
  // dropped, and that chain simply stops there.
  useEffect(() => {
    let cancelled = false;
    const loaded = new Set(jobs.map((j) => j.id));
    const missing = [
      ...new Set(jobs.flatMap((j) => j.ancestor_ids ?? [])),
    ].filter((id) => !loaded.has(id) && !ancestorCache[id]);
    if (missing.length === 0) return;
    (async () => {
      const settled = await Promise.all(
        missing.map((id) =>
          getJob(baseUrl, fetchWithHeaders, id)
            .then((rec) => [id, rec] as const)
            .catch(() => null),
        ),
      );
      if (cancelled) return;
      const fetched = Object.fromEntries(
        settled.filter((e): e is [string, JobRecord] => e !== null),
      );
      if (Object.keys(fetched).length > 0) {
        setAncestorCache((prev) => ({ ...prev, ...fetched }));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [jobs, ancestorCache, baseUrl, fetchWithHeaders]);

  const stop = useCallback(
    async (id: string) => {
      try {
        await stopJob(baseUrl, fetchWithHeaders, id);
        toast({ title: t("jobs.jobsData.stopping") });
        refresh();
      } catch (e) {
        toast({
          title: t("jobs.jobsData.stopFailed"),
          // Backend/network prose — shown as the server wrote it.
          description: e instanceof Error ? e.message : String(e),
          variant: "destructive",
        });
      }
    },
    [baseUrl, fetchWithHeaders, toast, refresh, t],
  );

  const remove = useCallback(
    async (id: string) => {
      try {
        await deleteJob(baseUrl, fetchWithHeaders, id);
        toast({ title: t("jobs.jobsData.removed") });
        refresh();
      } catch (e) {
        toast({
          title: t("jobs.jobsData.deleteFailed"),
          description: e instanceof Error ? e.message : String(e),
          variant: "destructive",
        });
      }
    },
    [baseUrl, fetchWithHeaders, toast, refresh, t],
  );

  // Untracked hub jobs aren't deletable on the Hub (the Jobs API has no
  // delete), so "remove" is a persisted backend-side dismissal.
  const dismissHub = useCallback(
    async (id: string) => {
      try {
        await dismissHubJob(baseUrl, fetchWithHeaders, id);
        toast({ title: t("jobs.jobsData.dismissed") });
        refresh();
      } catch (e) {
        toast({
          title: t("jobs.jobsData.dismissFailed"),
          description: e instanceof Error ? e.message : String(e),
          variant: "destructive",
        });
      }
    },
    [baseUrl, fetchWithHeaders, toast, refresh, t],
  );

  const localJobs = useMemo(
    () => jobs.filter((j) => j.runner === "local"),
    [jobs],
  );
  const trackedCloudJobs = useMemo(
    () => jobs.filter((j) => j.runner === "hf_cloud"),
    [jobs],
  );
  const importedJobs = useMemo(
    () => jobs.filter((j) => j.runner === "imported"),
    [jobs],
  );
  // Hub jobs already mirrored by a local JobRecord get their richer card via
  // trackedCloudJobs; everything else from the hub gets a plain HubJobCard.
  const trackedHfJobIds = useMemo(
    () =>
      new Set(
        trackedCloudJobs
          .map((j) => j.hf_job_id)
          .filter((id): id is string => !!id),
      ),
    [trackedCloudJobs],
  );
  const untrackedHubJobs = useMemo(
    () => hubJobs.filter((h) => !trackedHfJobIds.has(h.id)),
    [hubJobs, trackedHfJobIds],
  );
  // Hide model repos already claimed by a tracked job — a cloud run (shown via
  // JobCard) OR an imported model (also a JobCard, and the target a lazy
  // auto-import lands on). Repo ids are compared case-insensitively to match
  // the backend's find_imported dedup. The remainder are past trainings the
  // registry no longer remembers, rendered as untracked Hub cards.
  const trackedRepoIds = useMemo(
    () =>
      new Set(
        [...trackedCloudJobs, ...importedJobs]
          .map((j) => j.hf_repo_id?.toLowerCase())
          .filter((id): id is string => !!id),
      ),
    [trackedCloudJobs, importedJobs],
  );
  const untrackedHubModels = useMemo(
    () =>
      hubModels.filter((m) => !trackedRepoIds.has(m.repo_id.toLowerCase())),
    [hubModels, trackedRepoIds],
  );

  // Resume lineage, as the server computed it: a run with children has been
  // resumed, so it is hidden from the top level and reached through the
  // descendant that continued it — one row per LEAF, one row per chain.
  //
  // New lineages are CHAINS: jobs.py refuses a resume whose source already has
  // a child (sticks only, user decision 2026-08-07), so a run created from here
  // on can gain at most one. `child_ids` stays a LIST and everything below
  // stays fork-tolerant anyway, because registries written before that rule
  // hold real forks and must keep rendering exactly as they did — several
  // leaves off one trunk, each with its own row, no migration.
  const byId = useMemo(() => {
    const m = new Map(jobs.map((j) => [j.id, j]));
    // Cached ancestors fill in parents paged out of the list (never overriding
    // a loaded record).
    for (const rec of Object.values(ancestorCache)) {
      if (!m.has(rec.id)) m.set(rec.id, rec);
    }
    return m;
  }, [jobs, ancestorCache]);
  // Superseded is now a plain reading of the record: the server knows every
  // child, this page might not. That closes two holes in the client-side
  // approximation this replaces — a child sitting past the page limit failed
  // to hide its parent, and a child that died before its first checkpoint was
  // ruled "not legit" and left the chain rendering as two rows. A dead tip
  // still represents its chain; the row says so by being dead.
  const supersededIds = useMemo(
    () => new Set(jobs.filter((j) => j.child_ids.length > 0).map((j) => j.id)),
    [jobs],
  );
  // Nearest parent first, straight from the server's walk; ids it lists but
  // this client hasn't loaded yet are skipped until the backfill above lands.
  const ancestorsOf = useCallback(
    (job: JobRecord): JobRecord[] =>
      (job.ancestor_ids ?? [])
        .map((id) => byId.get(id))
        .filter((rec): rec is JobRecord => rec !== undefined),
    [byId],
  );

  // Checkpoints reachable from a run: its own plus every LOADED ancestor's.
  //
  // THE resume gate, and the fold gate, on the same reading: a run continues
  // from the newest checkpoint on its lineage, its own or an ancestor's, so
  // what decides both is what the CHAIN holds, not what this tip happened to
  // save. The commonest
  // resumable shape — a tip that died before its first checkpoint — has zero of
  // its own and a full chain behind it.
  const chainCheckpointCount = useCallback(
    (job: JobRecord): number =>
      [job, ...ancestorsOf(job)].reduce((n, j) => n + j.checkpoint_count, 0),
    [ancestorsOf],
  );

  // True while the server named an ancestor this client hasn't fetched yet, so
  // the count above is a known UNDERCOUNT rather than an answer. The backfill
  // effect above resolves these one render later; until it does, a chain whose
  // checkpoints all live in an unloaded ancestor would otherwise read as
  // having none.
  const ancestorsPending = useCallback(
    (job: JobRecord): boolean =>
      (job.ancestor_ids ?? []).some((id) => !byId.has(id)),
    [byId],
  );

  // Active = still running, or the CHAIN has a checkpoint. Everything else
  // folds under UNTRACKED in the libraries, so this decides whether a run is
  // reachable without opening the fold.
  //
  // Deliberately still chain-aware, where the resume gate no longer is: this
  // asks "is there anything here worth looking at", not "can this be
  // continued". A tip that died before its first save inherits a whole chain's
  // history and checkpoints — it charts, it can serve inference from an
  // inherited checkpoint, and deleting it is the move that frees its parent to
  // be resumed. Folding it away would hide the only handle on that.
  //
  // While the ancestor backfill is in flight the answer is INDETERMINATE, and
  // indeterminate resolves to active: a chain that flickered into the fold on
  // mount and back out a moment later would move rows under the user's cursor
  // for the sake of a value the client simply hasn't received yet.
  const isJobActive = useCallback(
    (job: JobRecord): boolean =>
      job.state === "running" ||
      chainCheckpointCount(job) > 0 ||
      ancestorsPending(job),
    [chainCheckpointCount, ancestorsPending],
  );

  const value = useMemo(
    () => ({
      jobs,
      localJobs,
      trackedCloudJobs,
      importedJobs,
      untrackedHubJobs,
      untrackedHubModels,
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
    }),
    [
      jobs,
      localJobs,
      trackedCloudJobs,
      importedJobs,
      untrackedHubJobs,
      untrackedHubModels,
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
    ],
  );

  return (
    <JobsDataContext.Provider value={value}>
      {children}
    </JobsDataContext.Provider>
  );
};

export function useJobsData(): JobsDataValue {
  const ctx = useContext(JobsDataContext);
  if (!ctx)
    throw new Error("useJobsData must be used within JobsDataProvider");
  return ctx;
}
