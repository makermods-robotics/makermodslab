import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
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
import JobCard from "./JobCard";
import HubModelCard from "./HubModelCard";
import ImportModelModal from "./ImportModelModal";
import { useJobsData } from "./JobsDataContext";
import { useSkills } from "@/hooks/useSkills";

/** How a model got here: everything, trained on this machine or in the cloud,
 * imported (local folder or Hub pull), or uploaded Hub repos no job tracks.
 *
 * These map onto the `origin` the SERVER stamps on every skill row. The library
 * used to decide membership itself, by filtering the job registry to
 * `runner === "imported"` — a different question from the one the deploy picker
 * asked of /models, which is why the two lists disagreed about what a skill is.
 * Both now read /skills and differ only in which origins they show. */
type ModelsFilter = "all" | "trained" | "imported" | "uploaded";

/** `key` is LOGIC — it is what the grid filters on and never changes. `label`
 * is a translation KEY, not a word: this array is built at import time, so a
 * resolved label would freeze whichever language loaded first. It is resolved
 * where the toolbar is rendered. */
const FILTERS = [
  { key: "all", label: "jobs.modelsLibrary.filters.all" },
  { key: "trained", label: "jobs.modelsLibrary.filters.trained" },
  { key: "imported", label: "jobs.modelsLibrary.filters.imported" },
  { key: "uploaded", label: "jobs.modelsLibrary.filters.uploaded" },
] as const satisfies ReadonlyArray<{ key: ModelsFilter; label: string }>;

interface ModelsLibraryProps {
  /** Select this model (job record + optional checkpoint step) as the policy
   * to deploy — wired to the Deploy panel's picker state. */
  onPick: (job: JobRecord, step: number | null) => void;
  /** Controlled disclosure, so the Deploy panel can fold this shelf down to
   * its header while its run form is open — the same wiring JobsLibrary takes
   * from Train and Collect's dataset library takes from Collect. Optional:
   * left out, the library owns its own open state and starts expanded. */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

/**
 * Model/policy library for the studio Deploy panel: search + origin filter
 * over a three-up grid of imported models plus uploaded hub repos no job
 * tracks (a model artifact, not a run — so it lives here, not under the Train
 * panel's jobs). Owns the Import button; rendered even when empty so the
 * entry point is always visible. Card Run actions select the model in the
 * Deploy panel instead of opening the legacy modal.
 */
const ModelsLibrary: React.FC<ModelsLibraryProps> = ({
  onPick,
  open: openProp,
  onOpenChange,
}) => {
  const { t } = useTranslation();
  const { openStudio } = useStudio();
  const { jobs, untrackedHubModels, refresh, stop, remove } = useJobsData();
  // The DEFINITION of what belongs here. The job registry still supplies the
  // record each card renders from (progress, lineage, stop/delete), but it no
  // longer decides membership — a run is in this library because /skills says
  // it is a skill, which is the same sentence the deploy picker reads.
  const { skills } = useSkills();
  // Shared lazy-import (idempotent registration + husk-repo messaging) so an
  // untracked Hub repo resolves to a pseudo-job exactly as everywhere else.
  const { importSource } = useInferenceLaunch();

  // Uncontrolled fallback: only used when the caller passes neither prop.
  const [ownOpen, setOwnOpen] = useState(true);
  const libraryOpen = openProp ?? ownOpen;
  const setLibraryOpen = onOpenChange ?? setOwnOpen;
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<ModelsFilter>("all");

  // Skill rows that a registry record backs, keyed by that record's id.
  const skillByJobId = useMemo(() => {
    const out = new Map<string, (typeof skills)[number]>();
    for (const skill of skills) if (skill.job_id) out.set(skill.job_id, skill);
    return out;
  }, [skills]);

  const query = search.trim().toLowerCase();
  const matchesQuery = (text: string | null | undefined) =>
    !query || (text ?? "").toLowerCase().includes(query);

  // Every loaded run that /skills calls a skill, narrowed by the chosen origin.
  // NOTE this is still bounded by the jobs page JobsDataContext loads — a skill
  // whose run has scrolled off that page has no record to render a card from.
  // That bound is unchanged by this rewiring; it is the pagination limit the
  // library always had, not a new one.
  const skillJobs = useMemo(
    () =>
      jobs.filter((j) => {
        const skill = skillByJobId.get(j.id);
        if (!skill) return false;
        // A superseded run is a LINK in a chain, not a skill of its own — the
        // tip stands for the whole chain. /skills returns it so a caller can
        // explain where a run went; a library of skills is not that caller, and
        // listing it here would show one trained model as several.
        if (skill.superseded_by) return false;
        if (filter === "uploaded") return false;
        if (filter === "trained")
          return (
            skill.origin === "trained-local" || skill.origin === "trained-cloud"
          );
        if (filter === "imported")
          return skill.origin === "imported" || skill.origin === "downloaded";
        return true;
      }),
    [jobs, skillByJobId, filter],
  );

  // A renamed import is findable by alias, original name, repo id, or path.
  const visibleImported = useMemo(
    () =>
      skillJobs.filter(
        (j) =>
          matchesQuery(j.name) ||
          matchesQuery(j.display_name) ||
          matchesQuery(j.hf_repo_id) ||
          matchesQuery(j.output_dir),
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [skillJobs, query],
  );
  const visibleUploaded = useMemo(
    () =>
      filter === "imported"
        ? []
        : untrackedHubModels.filter((m) => matchesQuery(m.repo_id)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [untrackedHubModels, filter, query],
  );

  const count = skillJobs.length + untrackedHubModels.length;
  const visibleCount = visibleImported.length + visibleUploaded.length;

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
      // flex-1 + min-h-0 down the whole chain (see LibrarySection): the
      // library fills the column's spare height and its footer sits at the
      // foot, instead of the slack pooling above the section's rule.
      className="flex min-h-0 flex-1 flex-col space-y-3"
    >
      <LibraryHeader
        title={t("jobs.modelsLibrary.title")}
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
            {t("jobs.modelsLibrary.importPolicy")}
          </Button>
        }
      />

      <CollapsibleContent className={cn(SLIDE, "flex min-h-0 flex-1 flex-col")}>
        {count === 0 ? (
          <div
            className={cn(
              "flex items-center justify-center rounded-md border border-dashed border-border px-4 py-6 text-center text-sm text-muted-foreground",
              GRID_MIN_H,
            )}
          >
            {t("jobs.modelsLibrary.empty")}
          </div>
        ) : (
          <div className="flex min-h-0 flex-1 flex-col space-y-3">
            <LibraryToolbar
              query={search}
              onQueryChange={setSearch}
              searchPlaceholder={t("jobs.modelsLibrary.searchPlaceholder")}
              // Only the LABEL is translated — `key` is what the grid filters
              // on and is passed through untouched.
              filters={FILTERS.map((f) => ({
                key: f.key,
                label: t(f.label),
              }))}
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
                {t("jobs.modelsLibrary.noMatch")}
              </p>
            ) : (
              // Imported and uploaded cards merged newest-first (import time
              // vs Hub last-modified); two rows by default, rest behind
              // Show all.
              <CappedGrid
                items={[
                  ...visibleImported.map((job) => ({
                    time: (job.started_at ?? 0) * 1000,
                    node: (
                      <JobCard
                        key={job.id}
                        job={job}
                        onStop={stop}
                        onDelete={remove}
                        onPlay={(j, step) => onPick(j, step)}
                        onRenamed={refresh}
                      />
                    ),
                  })),
                  ...visibleUploaded.map((model) => ({
                    time: model.last_modified
                      ? Date.parse(model.last_modified) || 0
                      : 0,
                    node: (
                      <HubModelCard
                        key={model.repo_id}
                        model={model}
                        onDeleted={refresh}
                        onAction={handleHubAction}
                      />
                    ),
                  })),
                ]
                  .sort((a, b) => b.time - a.time)
                  .map((e) => e.node)}
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
