import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
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

import { getModels, ModelItem } from "@/lib/modelsApi";
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
          {episodes} ep
        </span>
      ) : null}
      {item.source === "hub" ? (
        <span className="shrink-0 text-xs text-muted-foreground">Hub</span>
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

  const { seen: hasSeenTrainingMilestone, markSeen: markTrainingMilestoneSeen } =
    useOnceFlag("makerlab:milestone-first-training");
  const [pendingMilestoneJobId, setPendingMilestoneJobId] = useState<
    string | null
  >(null);

  // The new-training form slides open in place; the jobs library folds to its
  // header while the form is open (still expandable by hand).
  const [formOpen, setFormOpen] = useState(false);
  const [jobsOpen, setJobsOpen] = useState(true);

  // Pinned actions slot above the jobs library. While the form is open the
  // configurator portals its fully-gated Start button here; closed, a
  // disabled stand-in keeps the action visible at the same spot.
  const [actionsEl, setActionsEl] = useState<HTMLDivElement | null>(null);

  // ── Starting point (fine-tune base) ───────────────────────────────────────
  const [models, setModels] = useState<ModelItem[]>([]);
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
    // Starting point back to "Train from scratch" — it has no in-form escape
    // hatch. Reopening then gives a fresh form, which is also what the user
    // gets after a launch (onStarted folds the form the same way).
    if (!open) setResumeSeed(null);
  };

  useEffect(() => {
    let cancelled = false;
    getModels(baseUrl, fetchWithHeaders)
      .then((m) => {
        if (!cancelled) setModels(m);
      })
      .catch(() => {
        /* the starting point is optional — leave the list empty */
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

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
            title: "No checkpoints in this skill",
            description: "It has no saved checkpoint to fine-tune from.",
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
          title: "Couldn't load the starting point",
          description: e instanceof Error ? e.message : String(e),
          variant: "destructive",
        });
        setBaseModelId(NONE);
        setFinetuneSeed(null);
      } finally {
        if (current()) setResolvingBase(false);
      }
    },
    [baseUrl, fetchWithHeaders, toast],
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
      resolveFinetune({ repoId: model.hf_repo_id ?? model.id, name: model.name });
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

  // Apply a studio prefill (fine-tune base / preselected dataset) once, then
  // clear it so reopening the studio fresh doesn't re-apply a stale one.
  // Local skills arrive as baseJobId (a job registry id), Hub skills as
  // baseModelRepoId — a job id must never be sent through the Hub import path.
  // A prefill is an intent to configure a run, so it slides the form open too.
  useEffect(() => {
    if (!trainPrefill) return;
    if (trainPrefill.datasetRepoId) {
      setSelectedId(trainPrefill.datasetRepoId);
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

  return (
    <div className="flex flex-1 flex-col gap-5 p-5">
      <PanelHeader step="2" title="Train" />

      {/* Start a new training — the form slides open in place (no dialog),
          mirroring Collect's "Record new dataset". */}
      <Collapsible open={formOpen} onOpenChange={toggleForm} className="space-y-5">
        <CollapsibleTrigger asChild>
          <PanelEntryControl open={formOpen} dotClassName="bg-emerald-500">
            Start a new training
          </PanelEntryControl>
        </CollapsibleTrigger>
        <CollapsibleContent className={SLIDE}>
          <div className="space-y-6">
            <p className="text-sm leading-relaxed text-muted-foreground">
              {resumeSeed
                ? "Continuing an existing run — its dataset and weights are fixed. Set how much further it trains, then start."
                : "Choose what to train on, where the run executes, and how long it trains — then start."}
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
                <Label>Dataset</Label>
                <div className="flex flex-wrap gap-1.5">
                  <span className="inline-flex max-w-full items-center rounded-md border border-border bg-muted/40 px-2 py-1 font-mono text-xs text-muted-foreground">
                    <span className="truncate">{resumeSeed.datasetRepoId}</span>
                  </span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Inherited from the run being continued.
                </p>
              </div>
            ) : (
            <div className="space-y-2">
              <Label htmlFor="train-dataset-search">Dataset *</Label>
              {selectedId ? (
                <div className="flex flex-wrap gap-1.5">
                  <span className="inline-flex max-w-full items-center gap-1 rounded-md border border-border bg-muted/40 py-1 pl-2 pr-1 font-mono text-xs text-foreground">
                    <span className="truncate">{selectedId}</span>
                    <button
                      type="button"
                      aria-label={`Remove ${selectedId}`}
                      onClick={() => setSelectedId(null)}
                      className="shrink-0 rounded p-0.5 text-muted-foreground hover:text-foreground"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </span>
                </div>
              ) : null}
              <div className="relative">
                <Input
                  id="train-dataset-search"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search datasets, or type a public org/name Hub id"
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
                    title="Choose dataset"
                    aria-label="Choose dataset"
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
                          Use <span className="font-mono">{hubCandidate}</span>{" "}
                          from the Hub
                        </span>
                        <span className="block text-xs text-muted-foreground">
                          Public dataset — training fetches it on demand.
                        </span>
                      </span>
                    </button>
                  ) : null}
                  {matches.length === 0 && !hubCandidate ? (
                    <p className="px-3 py-4 text-sm text-muted-foreground">
                      No matching datasets. Type a full{" "}
                      <span className="font-mono">org/name</span> id to use any
                      public Hugging Face dataset.
                    </p>
                  ) : null}
                </div>
              ) : null}
              {!selectedId ? (
                <p className="text-xs text-muted-foreground">
                  Yours, or any public Hugging Face dataset.
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
              <Label htmlFor="train-starting-point">Starting point</Label>
              <Select value={baseModelId} onValueChange={handleBaseModelChange}>
                <SelectTrigger id="train-starting-point" className="w-full">
                  {/* A prefilled base (job card's Fine-tune) may not exist as
                      an item in the models listing — render the resolved
                      seed's name so the trigger is never blank (same pattern
                      as the Run panel's skill picker). */}
                  {baseModelId !== NONE && finetuneSeed ? (
                    <span className="truncate">{finetuneSeed.name}</span>
                  ) : (
                    <SelectValue placeholder="Train from scratch" />
                  )}
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NONE}>Train from scratch</SelectItem>
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
                    <Loader2 className="h-3 w-3 animate-spin" /> Loading
                    checkpoints…
                  </span>
                ) : finetuneSeed ? (
                  "Fine-tunes from this skill's latest checkpoint."
                ) : (
                  "Fine-tune an existing skill, or start fresh."
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
              key={`${resumeSeed?.jobId ?? ""}@${resumeSeed?.step ?? ""}@${
                resumeSeed?.checkpointJobId ?? ""
              }::${finetuneSeed?.jobId ?? "fresh"}@${finetuneSeed?.step ?? ""}`}
              policyType={policyType}
              onPolicyTypeChange={setPolicyType}
              datasetRepoId={trainingDatasetRepoId}
              finetuneSeed={finetuneSeed}
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
            Start training
          </Button>
        ) : null}
        <div ref={setActionsEl} className={formOpen ? undefined : "hidden"} />
      </div>

      {/* Training jobs library — local + online runs as cards, pinned to the
          panel foot like Collect's datasets and Deploy's models. mt-0 keeps it
          glued to the actions slot above, which carries the panel's mt-auto. */}
      <LibrarySection className="mt-0">
        <JobsLibrary open={jobsOpen} onOpenChange={setJobsOpen} />
      </LibrarySection>

      {/* Training-started milestone — launchJob opens the monitor dialog
          immediately after onStarted fires, so wait for it to close
          (!monitorJobId) before revealing this, mirroring CollectHandoff's
          "show after returning from the session" pattern. */}
      {!monitorJobId && pendingMilestoneJobId && (
        <MilestoneReveal
          title="Training started!"
          description="Watch progress from the jobs list above. Once it finishes, run it on your robot from the Deploy panel."
          onDismiss={() => setPendingMilestoneJobId(null)}
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
