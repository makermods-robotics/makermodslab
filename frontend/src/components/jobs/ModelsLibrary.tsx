import React, { useMemo, useState } from "react";
import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import LibraryToolbar from "@/components/library/LibraryToolbar";
import CappedGrid, { GRID_MIN_H } from "@/components/library/CappedGrid";
import LibraryHeader from "@/components/library/LibraryHeader";
import { SLIDE } from "@/components/studio/panel/primitives";
import { useStudio } from "@/contexts/StudioContext";
import { useInferenceLaunch } from "@/hooks/useInferenceLaunch";
import { JobRecord } from "@/lib/jobsApi";
import ModelCard from "./ModelCard";
import HubModelCard from "./HubModelCard";
import ImportModelModal from "./ImportModelModal";
import { useJobsData } from "./JobsDataContext";

/** Which shelf of the library to look at. The split is the seam the data
 * actually has — the component boundary: "yours" is everything backed by a job
 * record (trained here or imported), which renders a ModelCard with the full
 * affordance set; "hub" is the Hub repos with no local record, which render
 * the thinner HubModelCard. */
type ModelsFilter = "all" | "yours" | "hub";

const FILTERS: Array<{ key: ModelsFilter; label: string }> = [
  { key: "all", label: "All" },
  { key: "yours", label: "Yours" },
  { key: "hub", label: "On Hub" },
];

interface ModelsLibraryProps {
  /** Select this model (job record + optional checkpoint step) as the skill
   * to deploy — wired to the Deploy panel's picker state. */
  onPick: (job: JobRecord, step: number | null) => void;
}

/** When a model last did something, for the library's newest-first order.
 *
 * `ended_at` over `started_at`: a training that finished this morning is newer
 * to the user than one started last night and abandoned, and the shelf mixes
 * both. Import wrappers have no meaningful end, so they fall back to their
 * start — the moment the model appeared here, which is the right proxy.
 */
const latestActivity = (job: JobRecord) => job.ended_at ?? job.started_at ?? 0;

/** One card's worth of the owned shelf: the record that renders, plus the
 * names of the records it absorbed, so search still finds the card by any of
 * them. */
interface OwnedModel {
  record: JobRecord;
  /** Searchable handles contributed by the collapsed siblings (their names,
   * aliases and paths). The winner's own handles are matched separately. */
  aliases: string[];
}

/** Which of two records should represent a shared Hub repo.
 *
 * Same rule the rest of the stack ranks by (models.py `_job_outranks`, and
 * `findJobForModel` in lib/inferenceLaunch.ts), plus the tier those two don't
 * need: a TRAINED run beats an import wrapper of the same repo. The wrapper is
 * a pseudo-job minted by the lazy auto-import so a fine-tune has something to
 * resolve against; the run that actually produced the weights knows the policy,
 * the dataset and the step target, and the wrapper only knows a repo id.
 * Then: done beats not-done (it published the final policy at the repo root),
 * and among equals the later-started record wins (the deepest link of a resume
 * chain, which is the one whose checkpoints are current).
 */
function modelOutranks(candidate: JobRecord, incumbent: JobRecord): boolean {
  const candidateTrained = candidate.runner !== "imported";
  const incumbentTrained = incumbent.runner !== "imported";
  if (candidateTrained !== incumbentTrained) return candidateTrained;
  const candidateDone = candidate.state === "done";
  const incumbentDone = incumbent.state === "done";
  if (candidateDone !== incumbentDone) return candidateDone;
  return candidate.started_at > incumbent.started_at;
}

/** Collapse the owned shelf to one card per Hub repo.
 *
 * A cloud resume reuses its parent's output repo, and a fine-tune of such a
 * chain auto-imports that same repo as a base — so one set of weights can be
 * covered by several registry records at once (a trained run AND an import
 * wrapper of the identical repo), which rendered as duplicate cards claiming to
 * be different models. Repo ids are compared case-insensitively, matching the
 * backend's find_imported dedup.
 *
 * Losers are only hidden from the grid, never deleted: the import wrapper is
 * still what a fine-tune resolves its pretrained path through, and the parent
 * links of a resume chain still back the checkpoint lineage.
 *
 * A record with no hf_repo_id (a purely local run, an on-disk import) can't
 * collide by definition and is keyed by its own id, so it always gets a card.
 */
function collapseByRepo(records: JobRecord[]): OwnedModel[] {
  const order: string[] = [];
  const groups = new Map<string, JobRecord[]>();
  for (const record of records) {
    const key = record.hf_repo_id
      ? `repo:${record.hf_repo_id.toLowerCase()}`
      : `id:${record.id}`;
    const group = groups.get(key);
    if (group) {
      group.push(record);
    } else {
      groups.set(key, [record]);
      order.push(key);
    }
  }
  return order.map((key) => {
    const group = groups.get(key) as JobRecord[];
    const winner = group.reduce((best, r) => (modelOutranks(r, best) ? r : best));
    const aliases: string[] = [];
    for (const r of group) {
      if (r === winner) continue;
      for (const handle of [r.name, r.display_name, r.output_dir])
        if (handle) aliases.push(handle);
    }
    return { record: winner, aliases };
  });
}

/**
 * Model/policy library for the studio Deploy panel: search + shelf filter over
 * a grid of the models you can deploy — finished trainings (a successful run
 * IS the model it produced), imported models, and uploaded hub repos no job
 * tracks. Owns the Import button; rendered even when empty so the entry point
 * is always visible. Card Run actions select the model in the Deploy panel
 * instead of opening the legacy modal.
 *
 * Surfacing finished runs here is what makes a tracked cloud training
 * searchable as a model at all: before, it existed only as a run row in the
 * Train panel's jobs library, so the Deploy panel's own search couldn't find
 * it. Under "All" the two shelves stay legible without sub-headers — the
 * job-backed cards sort first, and each card's top-left origin chip says which
 * shelf it is on (ModelCard prints Local / Cloud / Imported (+ "from Hub"),
 * HubModelCard prints its own Hub badge).
 */
const ModelsLibrary: React.FC<ModelsLibraryProps> = ({ onPick }) => {
  const { openStudio } = useStudio();
  const {
    importedJobs,
    deployableModels,
    untrackedHubModels,
    ancestorsOf,
    refresh,
    remove,
  } = useJobsData();
  // Shared lazy-import (idempotent registration + husk-repo messaging) so an
  // untracked Hub repo resolves to a pseudo-job exactly as everywhere else.
  const { importSource } = useInferenceLaunch();

  const [libraryOpen, setLibraryOpen] = useState(true);
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<ModelsFilter>("all");

  const query = search.trim().toLowerCase();
  const matchesQuery = (text: string | null | undefined) =>
    !query || (text ?? "").toLowerCase().includes(query);

  // Finished trainings and imports are one shelf, collapsed to one card per
  // Hub repo (see collapseByRepo) before anything is filtered or counted.
  const ownedModels = useMemo(
    () => collapseByRepo([...deployableModels, ...importedJobs]),
    [deployableModels, importedJobs],
  );
  // A model is findable by any handle of ANY record behind its card: the
  // winner's alias, original name, repo id or path, plus the names of the
  // records it absorbed — so searching for the import wrapper's name still
  // lands on the trained run's card that replaced it.
  const matchesOwned = ({ record, aliases }: OwnedModel) =>
    matchesQuery(record.name) ||
    matchesQuery(record.display_name) ||
    matchesQuery(record.hf_repo_id) ||
    matchesQuery(record.output_dir) ||
    aliases.some(matchesQuery);
  const visibleOwned = useMemo(
    () =>
      filter === "hub"
        ? []
        : ownedModels
            .filter(matchesOwned)
            // Newest activity first across both sources — a just-finished
            // training lands above an older import.
            .sort((a, b) => latestActivity(b.record) - latestActivity(a.record)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [ownedModels, filter, query],
  );
  const visibleUploaded = useMemo(
    () =>
      filter === "yours"
        ? []
        : untrackedHubModels
            .filter((m) => matchesQuery(m.repo_id))
            .sort(
              (a, b) =>
                (b.last_modified ? Date.parse(b.last_modified) || 0 : 0) -
                (a.last_modified ? Date.parse(a.last_modified) || 0 : 0),
            ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [untrackedHubModels, filter, query],
  );

  // Counted after the collapse, so the badge matches the cards on screen.
  const count = ownedModels.length + untrackedHubModels.length;
  const visibleCount = visibleOwned.length + visibleUploaded.length;

  // Untracked hub model actions: register the repo as an imported pseudo-job
  // first (the proven lazy-import path), then either select it for deployment
  // right here or hand it to the Train panel as a fine-tune base.
  const handleHubAction = async (
    repoId: string,
    action: "inference" | "finetune",
  ) => {
    if (action === "finetune") {
      openStudio("train", { train: { baseModelRepoId: repoId } });
      return;
    }
    const record = await importSource(repoId);
    if (!record) return;
    refresh();
    // step null → the checkpoint loader picks the repo's latest.
    onPick(record, null);
  };

  return (
    <Collapsible
      open={libraryOpen}
      onOpenChange={setLibraryOpen}
      className="space-y-3"
    >
      <LibraryHeader
        title="Your skills"
        count={count}
        open={libraryOpen}
        actions={
          <Button
            variant="outline"
            size="sm"
            onClick={() => setImportModalOpen(true)}
            className="h-7 shrink-0 gap-1.5 text-xs text-muted-foreground hover:text-foreground"
          >
            <Download className="h-3.5 w-3.5" />
            Import skill
          </Button>
        }
      />

      <CollapsibleContent className={SLIDE}>
        {count === 0 ? (
          <div
            className={cn(
              "flex items-center justify-center rounded-md border border-dashed border-border px-4 py-6 text-center text-sm text-muted-foreground",
              GRID_MIN_H,
            )}
          >
            No skills yet. Train one, or use Import skill to add one from the
            Hub or a local folder.
          </div>
        ) : (
          <div className="space-y-3">
            <LibraryToolbar
              query={search}
              onQueryChange={setSearch}
              searchPlaceholder="Search skills"
              filters={FILTERS}
              filter={filter}
              onFilterChange={setFilter}
            />
            {visibleCount === 0 ? (
              <p
                className={cn(
                  "flex items-center justify-center px-1 py-4 text-center text-sm text-muted-foreground",
                  GRID_MIN_H,
                )}
              >
                No models match.
              </p>
            ) : (
              // One grid, two shelves. Each keeps its own sort (job records
              // newest-started first, Hub repos newest-modified first) and they
              // are concatenated rather than interleaved, so the actionable
              // models lead and the untracked Hub repos trail. One row by
              // default, the rest behind Show all.
              <CappedGrid
                items={[
                  ...visibleOwned.map(({ record: job }) => (
                    <ModelCard
                      key={job.id}
                      model={job}
                      onDelete={remove}
                      onPlay={(j, step) => onPick(j, step)}
                      onRenamed={refresh}
                      ancestors={ancestorsOf(job)}
                    />
                  )),
                  ...visibleUploaded.map((model) => (
                    <HubModelCard
                      key={model.repo_id}
                      model={model}
                      onDeleted={refresh}
                      onAction={handleHubAction}
                    />
                  )),
                ]}
              />
            )}
          </div>
        )}
      </CollapsibleContent>

      <ImportModelModal
        open={importModalOpen}
        onOpenChange={setImportModalOpen}
        onImported={refresh}
      />
    </Collapsible>
  );
};

export default ModelsLibrary;
