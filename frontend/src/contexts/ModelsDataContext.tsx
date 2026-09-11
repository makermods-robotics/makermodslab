import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import { useApi } from "@/contexts/ApiContext";
import { useJobsChangedSignal } from "@/hooks/useJobsChangedSignal";
import {
  ModelItem,
  SkillItem,
  SkillsHubStatus,
  getModels,
  getSkills,
} from "@/lib/modelsApi";

interface ModelsDataValue {
  /** The merged `/models` listing — model ARTIFACTS, the wider population the
   * fine-tune base picker needs (a foundation checkpoint is a valid base and
   * not a deployable skill). */
  models: ModelItem[];
  /** The `/skills` listing — deployable trained policies, each carrying why it
   * can or cannot run. The deploy picker and the models library both read this
   * one, which is what stops them disagreeing about what a skill is. */
  skills: SkillItem[];
  /** Reachability of the Hub half. Lets a caller say "the Hub is unreachable"
   * instead of rendering an outage as an empty shelf. */
  hub: SkillsHubStatus | null;
  /** True only while a fetch with nothing to show yet is in flight — a refresh
   * over an existing listing does not blank the UI back to a spinner. */
  loading: boolean;
  /** Set when the LAST fetch failed, whatever `models` holds. Callers must
   * treat this as distinct from an empty listing: "we could not ask" and
   * "you own nothing" are different sentences, and rendering the second for
   * the first is how a backend outage came to read as "no skills yet". */
  error: string | null;
  refresh: () => Promise<void>;
}

const ModelsDataContext = createContext<ModelsDataValue | null>(null);

/**
 * One owner for the `/models` listing, app-wide.
 *
 * Before this, four places fetched the same endpoint on four different refresh
 * policies: the Deploy panel's picker (on studio open, plus again after an
 * import), the Train panel's fine-tune Starting point (on mount), and
 * `useModels` for the launchpad's library sheet and skill slider (on mount,
 * once per consumer). None of them subscribed to anything, so a run that
 * finished while a surface was already mounted stayed missing from it until
 * that surface happened to remount — while the jobs-driven model library, which
 * rides the `jobs_changed` push, had already updated. Four snapshots of one
 * listing, taken at four different times, is most of what "the two skill lists
 * disagree" actually was.
 *
 * The freshness machinery here is deliberately the same as
 * `JobsDataContext`'s — the same WS signal, the same focus/visibility refetch —
 * because the two listings must move together to agree. The backend half is in
 * `server._on_jobs_changed`, which now drops the `/models` cache on every
 * registry mutation; without that this provider would refetch on the event and
 * be handed the same stale cache it already had.
 */
export const ModelsDataProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();

  const [models, setModels] = useState<ModelItem[]>([]);
  const [skills, setSkills] = useState<SkillItem[]>([]);
  const [hub, setHub] = useState<SkillsHubStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Read inside refresh without making it a dependency: refresh's identity is
  // what the WS subscription and every consumer effect key on, so it must not
  // change every time the listing does.
  const hasRowsRef = useRef(false);
  hasRowsRef.current = models.length > 0 || skills.length > 0;

  const refresh = useCallback(async () => {
    // Only show a spinner when there is nothing to show. A refresh triggered by
    // a background event (a run finishing, a tab regaining focus) must not
    // flash the picker back to "Loading skills…" underneath the user.
    if (!hasRowsRef.current) setLoading(true);
    try {
      // Two requests, one server-side build: /skills is a projection of the
      // same merged listing /models serves and rides its cache, so the second
      // call costs a round trip and nothing else. Fetched together so the two
      // populations are never a refresh apart — surfaces reading different
      // snapshots of one listing is the bug this whole context exists to end.
      const [nextModels, envelope] = await Promise.all([
        getModels(baseUrl, fetchWithHeaders),
        getSkills(baseUrl, fetchWithHeaders),
      ]);
      setModels(nextModels);
      setSkills(envelope.skills);
      setHub(envelope.hub);
      setError(null);
    } catch (e) {
      // Keep the last good listing. A transient failure blanking the picker is
      // indistinguishable, on screen, from the user's skills having been
      // deleted — and the rows we already hold are still very likely valid.
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [baseUrl, fetchWithHeaders]);

  // Initial fetch, plus a refetch when the tab regains focus — the same pair
  // JobsDataContext uses, and for the same reason: the WS push covers changes
  // originating on this machine, focus covers everything else (another tab, the
  // CLI, a repo pushed from the HF dashboard).
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

  // `jobs_changed` fires on submit, watchdog finalisation, import, rename and
  // delete — i.e. on every transition that can change whether a run is a
  // deployable skill. Progress ticks are ignored here: a step count moving does
  // not change the listing, and refetching at ~1Hz would hammer the Hub half.
  useJobsChangedSignal(refresh);

  return (
    <ModelsDataContext.Provider
      value={{ models, skills, hub, loading, error, refresh }}
    >
      {children}
    </ModelsDataContext.Provider>
  );
};

export function useModelsData(): ModelsDataValue {
  const ctx = useContext(ModelsDataContext);
  if (!ctx)
    throw new Error("useModelsData must be used within ModelsDataProvider");
  return ctx;
}
