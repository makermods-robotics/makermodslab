import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Trans, useTranslation } from "react-i18next";
import { Check, ChevronsUpDown, Loader2, Play, Plus, X } from "lucide-react";

import { useStudio } from "@/contexts/StudioContext";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useDatasets } from "@/hooks/useDatasets";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import { ModelItem } from "@/lib/modelsApi";
import { useModels } from "@/hooks/useModels";
import { JobRecord, getJob, importModel, jobDisplayName } from "@/lib/jobsApi";
import { listJobCheckpoints } from "@/lib/checkpointsApi";
import {
  DatasetItem,
  getDatasetInfo,
  saveCustomDataset,
} from "@/lib/replayApi";
import { HUB_REPO_ID_RE } from "@/lib/repoId";
import TrainingConfigurator, {
  FinetuneSeed,
  ResumeSeed,
} from "@/components/training/TrainingConfigurator";
import TrainingJobDialog from "@/components/training/TrainingJobDialog";
import JobsLibrary from "@/components/jobs/JobsLibrary";
import { useJobsData } from "@/components/jobs/JobsDataContext";
import DatasetPicker from "@/components/landing/DatasetPicker";
import {
  LibrarySection,
  PanelEntryControl,
  PanelHeader,
  SLIDE,
} from "@/components/studio/panel/primitives";
import MilestoneReveal from "@/components/onboarding/MilestoneReveal";
import { useOnceFlag } from "@/lib/onboarding/storage";

const NONE = "__none__";

/** jobs.register_imported's fallback when a checkpoint's config.json can't be
 * read — a display label, not an architecture. */
const UNKNOWN_POLICY_TYPE = "model";

/** Policies with no real "from scratch": each builds a pretrained backbone
 * (a vision-language model for smolvla, a PaliGemma+expert stack for
 * pi0/pi05/pi0_fast) that only receives real weights via an explicit
 * pretrained path, so leaving Starting point unset fine-tunes the matching
 * public foundation checkpoint instead of training with random weights (see
 * jobs.start's `_POLICY_FOUNDATION_BASE_REPO_ID`). Keep in sync with that
 * dict's keys. Every other policy's from-scratch run is normal. */
const FOUNDATION_POLICY_TYPES = new Set(["smolvla", "pi0", "pi05", "pi0_fast"]);

/** The architecture a job record trains (imported models record their
 * checkpoint's own `type`), or null when the record doesn't know it. Never
 * returns the "model" placeholder — pushing that into the form would select a
 * policy lerobot can't build. */
function recordPolicyType(record: JobRecord): string | null {
  const type = record.config?.policy_type?.trim();
  if (!type || type === UNKNOWN_POLICY_TYPE) return null;
  return type;
}

/** One search-result row: repo id + (local) episode count, lazily fetched, +
 * Hub marker for remote-only rows. Skips the network for Hub-only rows (a
 * remote meta.json read) — counts are shown "where available". */
const DatasetResultRow: React.FC<{
  item: DatasetItem;
  selected: boolean;
  onPick: () => void;
}> = ({ item, selected, onPick }) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const [episodes, setEpisodes] = useState<number | null>(null);

  useEffect(() => {
    if (item.source === "hub") return; // avoid a remote fetch just for a count
    let cancelled = false;
    getDatasetInfo(baseUrl, fetchWithHeaders, item.repo_id)
      .then((info) => {
        if (!cancelled) setEpisodes(info.total_episodes);
      })
      .catch(() => {
        /* count is optional — leave it blank */
      });
    return () => {
      cancelled = true;
    };
  }, [item.repo_id, item.source, baseUrl, fetchWithHeaders]);

  return (
    <button
      type="button"
      onClick={onPick}
      className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-muted/50"
    >
      <span className="min-w-0 flex-1 truncate font-mono text-foreground">
        {item.repo_id}
      </span>
      {episodes != null ? (
        <span className="shrink-0 text-xs text-muted-foreground">
          {t("studio.train.dataset.row.episodes", { episodes })}
        </span>
      ) : null}
      {/* A weighted merge and its unweighted sibling are otherwise
          indistinguishable here, and picking the wrong one silently trains the
          wrong mix. `=== true` on purpose: the flag is absent (unknown), not
          false, for Hub-only rows. */}
      {item.weighted === true ? (
        <span
          className="shrink-0 font-mono text-xs text-info"
          title={t("studio.train.dataset.row.weightedTitle")}
        >
          {t("studio.train.dataset.row.weighted")}
        </span>
      ) : null}
      {item.source === "hub" ? (
        <span className="shrink-0 text-xs text-muted-foreground">
          {t("studio.train.dataset.row.hub")}
        </span>
      ) : null}
      {selected ? (
        <Check className="h-3.5 w-3.5 shrink-0 text-primary" />
      ) : null}
    </button>
  );
};

/**
 * Studio panel 2 · Train. Mirrors the Collect panel's progressive disclosure:
 * a "Start a new training" button slides the full configuration open in place
 * (dataset → starting point → the shared training configurator), folding the
 * training-jobs library down to its header while the form is open.
 *
 * The form is flat — every control carries its own <Label> and there are no
 * eyebrow category headings above single fields, so nothing reads as a
 * category restating the parameter beneath it. Policy is picked in the
 * configurator's PolicyField, immediately after Starting point.
 */
const TrainPanel: React.FC = () => {
  const { t } = useTranslation();
  const { trainPrefill, clearTrainPrefill, monitorJobId, closeJobMonitor } =
    useStudio();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const {
    datasets,
    loading: datasetsLoading,
    refresh: refreshDatasets,
  } = useDatasets();
  const { selectedDataset, setSelectedDataset } = useSelectedDataset();
  // Same registry state the jobs library below renders. The panel only needs
  // the refetch: a launch from this form mutates the registry, and every other
  // mutation the studio performs (stop, delete, rename, hub dismiss) already
  // pulls the list afterwards rather than trusting the broadcast.
  const { refresh: refreshJobs } = useJobsData();

  const {
    seen: hasSeenTrainingMilestone,
    markSeen: markTrainingMilestoneSeen,
  } = useOnceFlag("makerlab:milestone-first-training");
  // Edge-triggered "consume once": onStarted sets the pending job id, and the
  // effect below latches it into showTrainingMilestone the first time the
  // monitor dialog (opened right after onStarted, see launchJob) closes, then
  // clears the pending id so it can't re-trigger — openJobMonitor is also
  // called from ActivityStrip, JobCard and the /training/:jobId deep link, so
  // monitorJobId cycling non-null→null again later (e.g. opening/closing the
  // monitor for an unrelated job) must not resurrect this banner.
  const [pendingMilestoneJobId, setPendingMilestoneJobId] = useState<
    string | null
  >(null);
  const [showTrainingMilestone, setShowTrainingMilestone] = useState(false);

  useEffect(() => {
    if (!monitorJobId && pendingMilestoneJobId) {
      setShowTrainingMilestone(true);
      setPendingMilestoneJobId(null);
    }
  }, [monitorJobId, pendingMilestoneJobId]);

  // The new-training form slides open in place; the jobs library folds to its
  // header while the form is open (still expandable by hand).
  const [formOpen, setFormOpen] = useState(false);
  const [jobsOpen, setJobsOpen] = useState(true);

  // Pinned actions slot above the jobs library. While the form is open the
  // configurator portals its fully-gated Start button here; closed, a
  // disabled stand-in keeps the action visible at the same spot.
  const [actionsEl, setActionsEl] = useState<HTMLDivElement | null>(null);

  // ── Starting point (fine-tune base) ───────────────────────────────────────
  // Shared with the Deploy picker and the launchpad slider (ModelsDataContext),
  // so the three cannot drift apart. Note this list is a WIDER population than
  // the Deploy picker's: a foundation checkpoint is a legitimate fine-tune base
  // and not a deployable skill. They read one endpoint today; if that ever
  // splits, this is the call site that keeps `/models`.
  const { models } = useModels();
  const [baseModelId, setBaseModelId] = useState<string>(NONE);
  const [finetuneSeed, setFinetuneSeed] = useState<FinetuneSeed | null>(null);
  const [resolvingBase, setResolvingBase] = useState(false);

  // ── Continue / Resume ─────────────────────────────────────────────────────
  // A resume arrives already resolved (see TrainPrefill.resume), so unlike a
  // fine-tune base there is nothing to look up — it goes straight into state
  // and the configurator switches to resume mode.
  const [resumeSeed, setResumeSeed] = useState<ResumeSeed | null>(null);

  const toggleForm = (open: boolean) => {
    setFormOpen(open);
    setJobsOpen(!open);
    // Closing the form is the way out of a resume: resume mode hides the
    // dataset and starting-point controls (both are the parent run's and not
    // editable), so unlike a fine-tune — which can be dropped by setting
    // Starting point back to its no-selection state (NONE) — it has no
    // in-form escape hatch. Reopening then gives a fresh form, which is also
    // what the user gets after a launch (onStarted folds the form the same
    // way).
    if (!open) setResumeSeed(null);
  };

  // ── Policy ────────────────────────────────────────────────────────────────
  // Chosen inside Run configuration (EssentialsCard's select); a base-skill or
  // prefill choice re-targets it so the fine-tune trains the matching policy.
  const [policyType, setPolicyType] = useState<string>("act");

  // Resolve a starting point into a fine-tune seed: fine-tuning needs a concrete
  // job id + checkpoint step (the exact contract Training's finetune flow
  // expects). A Hub-only model is registered as an imported job first (the
  // proven lazy-import path), then its latest checkpoint step seeds the run.
  // Sequence guard: only the LATEST resolution may write state — a slower,
  // older import/checkpoint lookup finishing last must not overwrite a newer
  // base-skill choice.
  const resolveSeqRef = useRef(0);
  const resolveFinetune = useCallback(
    async (opts: {
      jobId?: string;
      repoId?: string;
      name?: string;
      step?: number;
    }) => {
      const seq = ++resolveSeqRef.current;
      const current = () => resolveSeqRef.current === seq;
      setResolvingBase(true);
      try {
        let jobId = opts.jobId;
        let name = opts.name;
        let policy: string | null = null;
        if (!jobId && opts.repoId) {
          const rec = await importModel(baseUrl, fetchWithHeaders, opts.repoId);
          jobId = rec.id;
          name = name ?? jobDisplayName(rec);
          policy = recordPolicyType(rec);
        }
        if (!jobId || !current()) return;
        if (policy === null) {
          // A caller-supplied job id (the job card's Fine-tune button, a studio
          // prefill, a local model row) carries no policy type, so read it off
          // the registry record — the same value the import branch above gets.
          // Without this the policy stays on the "act" default while the form
          // LOCKS the picker ("set by the base skill"), and the run silently
          // trains ACT from e.g. smolvla weights: lerobot loads a checkpoint
          // non-strictly, so the mismatch never surfaces at runtime.
          const rec = await getJob(baseUrl, fetchWithHeaders, jobId).catch(
            () => null,
          );
          if (!current()) return;
          policy = rec ? recordPolicyType(rec) : null;
        }
        const cks = await listJobCheckpoints(baseUrl, fetchWithHeaders, jobId);
        if (!current()) return;
        // A caller-pinned step (the card's dropdown choice) wins when it still
        // exists; otherwise fall back to the latest checkpoint.
        const pinned =
          opts.step != null && cks.some((c) => c.step === opts.step)
            ? opts.step
            : null;
        const latest =
          pinned ?? (cks.length > 0 ? cks[cks.length - 1].step : null);
        if (latest == null) {
          toast({
            title: t("studio.train.toast.noCheckpointsTitle"),
            description: t("studio.train.toast.noCheckpointsBody"),
            variant: "destructive",
          });
          setBaseModelId(NONE);
          setFinetuneSeed(null);
          return;
        }
        // Carry the chosen checkpoint's own storage side along with its step:
        // a "local" one exists only on this machine, so fine-tuning it on the
        // cloud has to upload it first and the form has to say so.
        const chosen = cks.find((c) => c.step === latest);
        setFinetuneSeed({
          jobId,
          step: latest,
          name: name ?? jobId,
          policyType: policy ?? "act",
          checkpointSource: chosen?.source,
        });
        if (policy) setPolicyType(policy);
      } catch (e) {
        if (!current()) return;
        toast({
          title: t("studio.train.toast.baseFailedTitle"),
          // The thrown error's own text — shown exactly as raised.
          description: e instanceof Error ? e.message : String(e),
          variant: "destructive",
        });
        setBaseModelId(NONE);
        setFinetuneSeed(null);
      } finally {
        if (current()) setResolvingBase(false);
      }
    },
    [baseUrl, fetchWithHeaders, toast, t],
  );

  const handleBaseModelChange = (value: string) => {
    setBaseModelId(value);
    if (value === NONE) {
      setFinetuneSeed(null);
      return;
    }
    const model = models.find((m) => m.id === value);
    if (!model) {
      setFinetuneSeed(null);
      return;
    }
    // Local runs already have a job id; Hub-only models resolve via import.
    if (model.source === "hub") {
      resolveFinetune({
        repoId: model.hf_repo_id ?? model.id,
        name: model.name,
      });
    } else {
      if (model.policy_type) setPolicyType(model.policy_type);
      resolveFinetune({ jobId: model.id, name: model.name });
    }
  };

  // ── Dataset picker (single-select) ────────────────────────────────────────
  const [selectedId, setSelectedId] = useState<string | null>(
    () => selectedDataset,
  );
  const [query, setQuery] = useState("");
  // A prefill's episode subset, paired with the repo id it was seeded for —
  // dropped (via the repoId guard below) if the user then picks a different
  // dataset, so a stale subset can never silently apply to it.
  const [prefillEpisodes, setPrefillEpisodes] = useState<{
    repoId: string;
    indices: number[];
  } | null>(null);

  // Apply a studio prefill (fine-tune base / preselected dataset) once, then
  // clear it so reopening the studio fresh doesn't re-apply a stale one.
  // Local skills arrive as baseJobId (a job registry id), Hub skills as
  // baseModelRepoId — a job id must never be sent through the Hub import path.
  // A prefill is an intent to configure a run, so it slides the form open too.
  useEffect(() => {
    if (!trainPrefill) return;
    if (trainPrefill.datasetRepoId) {
      setSelectedId(trainPrefill.datasetRepoId);
      setPrefillEpisodes(
        trainPrefill.episodeIndices
          ? { repoId: trainPrefill.datasetRepoId, indices: trainPrefill.episodeIndices }
          : null,
      );
    }
    if (trainPrefill.resume) {
      // Continue / Resume-cloud. Mutually exclusive with a fine-tune base, so
      // clear that side rather than letting a stale one ride along.
      setResumeSeed(trainPrefill.resume);
      setBaseModelId(NONE);
      setFinetuneSeed(null);
      // The parent's architecture, same as the /training route seeds it with.
      setPolicyType(trainPrefill.resume.policyType);
    } else if (trainPrefill.baseJobId) {
      setBaseModelId(trainPrefill.baseJobId);
      resolveFinetune({
        jobId: trainPrefill.baseJobId,
        step: trainPrefill.baseStep,
        name: trainPrefill.baseName,
      });
    } else if (trainPrefill.baseModelRepoId) {
      setBaseModelId(trainPrefill.baseModelRepoId);
      resolveFinetune({
        repoId: trainPrefill.baseModelRepoId,
        step: trainPrefill.baseStep,
        name: trainPrefill.baseName,
      });
    }
    setFormOpen(true);
    setJobsOpen(false);
    clearTrainPrefill();
  }, [trainPrefill, clearTrainPrefill, resolveFinetune]);

  // Follow the shared selection while mounted: picking a dataset in Collect's
  // library (or the handoff banner) must re-target Train too — the panels are
  // mounted simultaneously, so mount-time seeding isn't enough.
  useEffect(() => {
    if (selectedDataset) setSelectedId(selectedDataset);
  }, [selectedDataset]);

  // Empty keeps the configurator's Start disabled until a dataset is picked.
  // A resume inherits the parent run's dataset and can't be re-pointed, so its
  // seed wins over the shared selection outright — same precedence the
  // /training route applies (`resumeSource?.datasetRepoId ?? selectedDataset`).
  // Deliberately NOT written back into selectedId/setSelectedDataset: pressing
  // Continue shouldn't silently re-target the studio-wide dataset selection
  // that Collect and Deploy also read.
  const trainingDatasetRepoId = resumeSeed
    ? resumeSeed.datasetRepoId
    : (selectedId ?? "");

  // A resume can't carry an episode subset (it inherits the parent run's
  // dataset wholesale), and the subset only applies to the exact dataset it
  // was computed against.
  const trainingEpisodeIndices =
    !resumeSeed && prefillEpisodes?.repoId === trainingDatasetRepoId
      ? prefillEpisodes.indices
      : undefined;

  // Total episode count for the "training on X of Y" note below — only
  // fetched while a subset is actually active, since that's the only time
  // the note renders.
  const [datasetTotalEpisodes, setDatasetTotalEpisodes] = useState<
    number | null
  >(null);
  useEffect(() => {
    if (!trainingEpisodeIndices || !trainingDatasetRepoId) {
      setDatasetTotalEpisodes(null);
      return;
    }
    const controller = new AbortController();
    getDatasetInfo(baseUrl, fetchWithHeaders, trainingDatasetRepoId, controller.signal)
      .then((info) => setDatasetTotalEpisodes(info.total_episodes))
      .catch(() => setDatasetTotalEpisodes(null));
    return () => controller.abort();
  }, [trainingEpisodeIndices, trainingDatasetRepoId, baseUrl, fetchWithHeaders]);

  // Keep the shared selection (Deploy panel, direct /training route) in step
  // with the dataset chosen here.
  useEffect(() => {
    if (selectedId) setSelectedDataset(selectedId);
  }, [selectedId, setSelectedDataset]);

  // Search-driven picker: results only exist while a query is typed — there is
  // no standing list.
  const trimmedQuery = query.trim();
  const matches = useMemo(() => {
    const q = trimmedQuery.toLowerCase();
    if (!q) return [];
    return datasets.filter((d) => d.repo_id.toLowerCase().includes(q));
  }, [datasets, trimmedQuery]);

  // A well-formed `org/name` id that isn't in the library yet is offered as a
  // public Hub dataset — the affordance that ANY public dataset can be trained
  // on, not just the user's own.
  const hubCandidate = useMemo(() => {
    if (!HUB_REPO_ID_RE.test(trimmedQuery)) return null;
    const q = trimmedQuery.toLowerCase();
    if (datasets.some((d) => d.repo_id.toLowerCase() === q)) return null;
    return trimmedQuery;
  }, [datasets, trimmedQuery]);

  // Picking a result replaces the selection and collapses the results by
  // clearing the query.
  const pickDataset = (repoId: string) => {
    setSelectedId(repoId);
    setQuery("");
  };

  // Selecting a not-yet-listed public Hub id also pins it (best-effort, same
  // path as the library's "Add from Hub") so it persists in dataset lists and
  // training can fetch it on demand.
  const addHubDataset = (repoId: string) => {
    setSelectedId(repoId);
    setQuery("");
    saveCustomDataset(baseUrl, fetchWithHeaders, repoId)
      .then(() => refreshDatasets())
      .catch(() => {
        /* non-fatal: the dataset is still selected for this run */
      });
  };

  // Both the <Select> placeholder and the "no base model" option's label. The
  // submitted option VALUE ("__none__") is untouched.
  const noStartingPointLabel = FOUNDATION_POLICY_TYPES.has(policyType)
    ? t("studio.train.startingPoint.fromBase")
    : t("studio.train.startingPoint.scratch");

  return (
    <div className="flex flex-1 flex-col gap-5 p-5">
      <PanelHeader
        step="2"
        title={t("studio.train.title")}
        dataTour="studio-train"
      />

      {/* Start a new training — the form slides open in place (no dialog),
          mirroring Collect's "Record new dataset". */}
      <Collapsible
        open={formOpen}
        onOpenChange={toggleForm}
        className="space-y-5"
      >
        <CollapsibleTrigger asChild>
          <PanelEntryControl open={formOpen} dotClassName="bg-emerald-500">
            {t("studio.train.entry")}
          </PanelEntryControl>
        </CollapsibleTrigger>
        <CollapsibleContent className={SLIDE}>
          <div className="space-y-6">
            <p className="text-sm leading-relaxed text-muted-foreground">
              {resumeSeed
                ? t("studio.train.intro.resume")
                : t("studio.train.intro.fresh")}
            </p>

            {/* Dataset picker — search-driven, no standing list. Flat: the
                label rides on the control, so no "Dataset" category heading
                sits above it restating the same word.

                A resume shows the parent's dataset read-only instead: a
                continuation trains on the dataset baked into the checkpoint,
                so a live picker here would offer an edit the run can't honour.
                (The seed also wins in trainingDatasetRepoId, so what's shown
                is exactly what launches.) */}
            {resumeSeed ? (
              <div className="space-y-2">
                <Label>{t("studio.train.dataset.resumeLabel")}</Label>
                <div className="flex flex-wrap gap-1.5">
                  <span className="inline-flex max-w-full items-center rounded-md border border-border bg-muted/40 px-2 py-1 font-mono text-xs text-muted-foreground">
                    <span className="truncate">{resumeSeed.datasetRepoId}</span>
                  </span>
                </div>
                <p className="text-xs text-muted-foreground">
                  {t("studio.train.dataset.resumeHint")}
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                <Label htmlFor="train-dataset-search">
                  {t("studio.train.dataset.label")}
                </Label>
                {selectedId ? (
                  <div className="flex flex-wrap gap-1.5">
                    <span className="inline-flex max-w-full items-center gap-1 rounded-md border border-border bg-muted/40 py-1 pl-2 pr-1 font-mono text-xs text-foreground">
                      <span className="truncate">{selectedId}</span>
                      <button
                        type="button"
                        aria-label={t("studio.train.dataset.remove", {
                          repoId: selectedId,
                        })}
                        onClick={() => setSelectedId(null)}
                        className="shrink-0 rounded p-0.5 text-muted-foreground hover:text-foreground"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </span>
                  </div>
                ) : null}
                {trainingEpisodeIndices && (
                  <p className="text-xs text-muted-foreground">
                    {datasetTotalEpisodes != null
                      ? t("studio.train.dataset.episodeSubsetOfTotal", {
                          used: trainingEpisodeIndices.length,
                          total: datasetTotalEpisodes,
                        })
                      : t("studio.train.dataset.episodeSubset", {
                          used: trainingEpisodeIndices.length,
                        })}
                  </p>
                )}
                <div className="relative">
                  <Input
                    id="train-dataset-search"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder={t("studio.train.dataset.searchPlaceholder")}
                    className="h-8 pr-8 text-sm"
                  />
                  {/* "Choose dataset" — browse the full Local/Hugging Face list
                    instead of typing a search term. Docked on the right edge
                    of the search bar itself (a sibling overlay, not nested in
                    the <input>) so it's reachable without typing anything. */}
                  <DatasetPicker
                    datasets={datasets}
                    loading={datasetsLoading}
                    onPickExisting={(item) => pickDataset(item.repo_id)}
                    hideSearch
                  >
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="absolute right-0.5 top-1/2 h-7 w-7 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                      title={t("studio.train.dataset.choose")}
                      aria-label={t("studio.train.dataset.choose")}
                    >
                      <ChevronsUpDown className="h-3.5 w-3.5" />
                    </Button>
                  </DatasetPicker>
                </div>
                {trimmedQuery ? (
                  <div className="max-h-56 divide-y divide-border overflow-auto rounded-md border border-border">
                    {matches.map((d) => (
                      <DatasetResultRow
                        key={d.repo_id}
                        item={d}
                        selected={selectedId === d.repo_id}
                        onPick={() => pickDataset(d.repo_id)}
                      />
                    ))}
                    {hubCandidate ? (
                      <button
                        type="button"
                        onClick={() => addHubDataset(hubCandidate)}
                        className="flex w-full items-start gap-2 px-3 py-2 text-left text-sm hover:bg-muted/50"
                      >
                        <Plus className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
                        <span className="min-w-0 flex-1">
                          <span className="block truncate">
                            {/* The typed repo id is DATA — one <0> slot, not a
                              sentence stitched around it. */}
                            <Trans
                              i18nKey="studio.train.dataset.useHub"
                              values={{ repoId: hubCandidate }}
                              components={[
                                <span key="0" className="font-mono" />,
                              ]}
                            />
                          </span>
                          <span className="block text-xs text-muted-foreground">
                            {t("studio.train.dataset.useHubHint")}
                          </span>
                        </span>
                      </button>
                    ) : null}
                    {matches.length === 0 && !hubCandidate ? (
                      <p className="px-3 py-4 text-sm text-muted-foreground">
                        {/* <0> holds the literal `org/name` id shape — syntax,
                          so it is not translated. */}
                        <Trans
                          i18nKey="studio.train.dataset.noMatches"
                          components={[<span key="0" className="font-mono" />]}
                        />
                      </p>
                    ) : null}
                  </div>
                ) : null}
                {!selectedId ? (
                  <p className="text-xs text-muted-foreground">
                    {t("studio.train.dataset.hint")}
                  </p>
                ) : null}
              </div>
            )}

            {/* Starting point — the optional fine-tune base. Sits directly
                under Dataset because both answer "what is this run built
                from"; Policy follows from inside the configurator.

                Hidden entirely while resuming: a continuation's starting point
                IS the checkpoint it resumes from — picking a different base
                would make it a fine-tune, not a resume — and the configurator's
                "Continuing <name> from step N" banner already names it. */}
            {resumeSeed ? null : (
              <div className="space-y-2">
                <Label htmlFor="train-starting-point">
                  {t("studio.train.startingPoint.label")}
                </Label>
                <Select
                  value={baseModelId}
                  onValueChange={handleBaseModelChange}
                >
                  <SelectTrigger id="train-starting-point" className="w-full">
                    {/* A prefilled base (job card's Fine-tune) may not exist as
                      an item in the models listing — render the resolved
                      seed's name so the trigger is never blank (same pattern
                      as the Run panel's skill picker). */}
                    {baseModelId !== NONE && finetuneSeed ? (
                      <span className="truncate">{finetuneSeed.name}</span>
                    ) : (
                      <SelectValue placeholder={noStartingPointLabel} />
                    )}
                  </SelectTrigger>
                  <SelectContent>
                    {/* The option VALUE stays `__none__` — only its label is
                      translated. */}
                    <SelectItem value={NONE}>{noStartingPointLabel}</SelectItem>
                    {models.map((m) => (
                      <SelectItem key={m.id} value={m.id}>
                        {m.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {resolvingBase ? (
                    <span className="inline-flex items-center gap-1">
                      <Loader2 className="h-3 w-3 animate-spin" />{" "}
                      {t("studio.train.startingPoint.loading")}
                    </span>
                  ) : finetuneSeed ? (
                    t("studio.train.startingPoint.finetuneHint")
                  ) : FOUNDATION_POLICY_TYPES.has(policyType) ? (
                    t("studio.train.startingPoint.foundationHint")
                  ) : (
                    t("studio.train.startingPoint.hint")
                  )}
                </p>
              </div>
            )}

            {/* Shared configuration form: compute target, run configuration,
                advanced, extras gates, and the Start button with all its
                gating. Policy is chosen inside Run configuration.

                Keyed on both seeds (same composite the /training route uses):
                switching between runs — or between resuming and fine-tuning —
                must rebuild the form, since the seeds are read once as initial
                state. Both null ⇒ a stable fresh key.

                The key carries the whole seed identity — run, step and (for a
                resume) the checkpoint's owner — not just the run id. The jobs
                library stays expandable while the form is open, so a second
                Continue / Fine-tune on the same run is reachable, and only the
                banner and the step validation read the seed live: everything
                derived at mount (resume_from_step, the checkpoint owner, steps,
                the optimizer/W&B prefills, the compute default) would stay
                frozen at the first press. That shipped a request continuing
                from the first checkpoint under a banner naming the second —
                which the backend cannot catch, both being internally valid.
                Resume no longer offers a step choice, so the resume half is
                belt-and-braces; the fine-tune half is live, since ModelCard's
                history selector really can re-fire at a different step. */}
            <TrainingConfigurator
              // The fine-tune STEP is deliberately absent from this key. It
              // used to be here so mount-derived values refreshed, but the
              // checkpoint picker edits that step — remounting on every pick
              // discarded whatever had already been typed into the form, and
              // (worse) reset the pick itself back to the source's latest. The
              // configurator now reads the fine-tune step live off the seed, so
              // it needs no remount to see a change. The resume half keeps its
              // step: resume is not step-selectable, so it only varies when the
              // user really did arrive from a different checkpoint.
              key={`${resumeSeed?.jobId ?? ""}@${resumeSeed?.step ?? ""}@${
                resumeSeed?.checkpointJobId ?? ""
              }::${finetuneSeed?.jobId ?? "fresh"}`}
              policyType={policyType}
              onPolicyTypeChange={setPolicyType}
              datasetRepoId={trainingDatasetRepoId}
              episodeIndices={trainingEpisodeIndices}
              finetuneSeed={finetuneSeed}
              // The seed owns the chosen base checkpoint, so the pick survives
              // a remount of the form below. `checkpointSource` moves with it:
              // it decides whether a cloud run must stage the weights first.
              onFinetuneCheckpointChange={(c) =>
                setFinetuneSeed((prev) =>
                  prev ? { ...prev, step: c.step, checkpointSource: c.source } : prev,
                )
              }
              resumeSeed={resumeSeed}
              // Launch opens the monitor dialog over this panel (via
              // openJobMonitor in the configurator); fold the form back so
              // closing the dialog lands on the jobs library, not a stale form.
              //
              // And pull the job list, because the new run has to be IN it:
              // the panel never unmounts across a launch, so nothing else
              // re-reads /jobs. A launch's only route into the list is
              // otherwise the fire-and-forget `jobs_changed` broadcast; miss
              // it once and the run stays invisible until the next mount
              // (page reload). This is the same after-mutation refetch every
              // other studio action (stop, delete, rename) already does.
              onStarted={(jobId) => {
                toggleForm(false);
                refreshJobs();
                if (!hasSeenTrainingMilestone) {
                  setPendingMilestoneJobId(jobId);
                  markTrainingMilestoneSeen();
                }
              }}
              actionsContainer={actionsEl}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>

      {/* Start training — pinned directly above the jobs library, level with
          Collect's and Deploy's actions. The configurator portals its real,
          fully-gated button into the slot while the form is open. */}
      <div className="mt-auto pt-2">
        {!formOpen ? (
          <Button disabled className="w-full gap-2">
            <Play className="h-4 w-4" />
            {t("studio.train.start")}
          </Button>
        ) : null}
        <div ref={setActionsEl} className={formOpen ? undefined : "hidden"} />
      </div>

      {/* Training jobs library — local + remote runs as cards, pinned to the
          panel foot like Collect's datasets and Deploy's models. mt-0 keeps it
          glued to the actions slot above, which carries the panel's mt-auto. */}
      <LibrarySection className="mt-0">
        <JobsLibrary open={jobsOpen} onOpenChange={setJobsOpen} />
      </LibrarySection>

      {/* Training-started milestone — launchJob opens the monitor dialog
          immediately after onStarted fires, so the effect above waits for it
          to close before latching showTrainingMilestone true, mirroring
          CollectHandoff's "show after returning from the session" pattern.
          Gated on the latched flag alone (not live on !monitorJobId) so a
          later, unrelated monitor open/close (another job's card, the
          activity strip, a deep link, or a second training run) can't
          resurrect an already-dismissed-or-shown banner. */}
      {showTrainingMilestone && (
        <MilestoneReveal
          title={t("studio.train.milestone.title")}
          description={t("studio.train.milestone.description")}
          onDismiss={() => setShowTrainingMilestone(false)}
        />
      )}

      {/* Job monitor as a dialog over the studio (same pattern as Collect's
          RecordingSessionDialog) — closing it lands back on this panel. */}
      {monitorJobId ? (
        <TrainingJobDialog jobId={monitorJobId} onExit={closeJobMonitor} />
      ) : null}
    </div>
  );
};

export default TrainPanel;
