import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import {
  Archive,
  CloudDownload,
  FolderInput,
  GitMerge,
  Play,
  Plus,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useStudio } from "@/contexts/StudioContext";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useModels } from "@/hooks/useModels";
import { useDatasets } from "@/hooks/useDatasets";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";
import { useHubVideoFilter } from "@/hooks/useHubVideoFilter";
import { policyTypeDisplayName } from "@/components/training/types";
import { ModelItem, downloadModel, saveCustomModel } from "@/lib/modelsApi";
import {
  DatasetItem,
  downloadDataset,
  saveCustomDataset,
} from "@/lib/replayApi";
import AddDatasetFromHubDialog from "@/components/landing/AddDatasetFromHubDialog";
import ImportDatasetFromDiskDialog from "@/components/landing/ImportDatasetFromDiskDialog";
import ManageCachesDialog from "@/components/landing/ManageCachesDialog";
import AddModelFromHubDialog from "@/components/landing/AddModelFromHubDialog";
import ImportModelFromDiskDialog from "@/components/landing/ImportModelFromDiskDialog";
import {
  formatCount,
  isMineSkill,
  skillDisplayTitle,
} from "@/components/launchpad/SkillCard";
import MergeDatasetsDialog from "@/components/landing/MergeDatasetsDialog";
import DatasetDetailDialog from "@/components/dialogs/DatasetDetailDialog";
import SkillManageDialog from "@/components/dialogs/SkillManageDialog";

/** The backend's dataset `source` enum → its label KEY for the mono subtitle
 * line. The enum values are data; the map holds keys, not resolved copy, so it
 * doesn't freeze whichever language loaded first. */
const DATASET_SOURCE_KEY = {
  both: "library.sheet.datasets.source.both",
  hub: "library.sheet.datasets.source.hub",
  local: "library.sheet.datasets.source.local",
} as const;

export interface LibrarySheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

type Tab = "skills" | "datasets";

const SegButton: React.FC<{
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}> = ({ active, onClick, children }) => (
  <button
    type="button"
    onClick={onClick}
    className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
      active
        ? "bg-background text-foreground shadow-1"
        : "text-muted-foreground hover:text-foreground"
    }`}
  >
    {children}
  </button>
);

/**
 * "My library" slide-over — a right-anchored sheet (Radix Dialog primitive) with
 * My skills / My datasets tabs. Skill rows Run on the corner robot (→ Deploy,
 * prefilled); dataset rows open the dataset detail dialog. Footer offers a new
 * skill (→ studio Collect) and Merge datasets (the existing MergeDatasetsDialog,
 * reused unmodified).
 */
const LibrarySheet: React.FC<LibrarySheetProps> = ({ open, onOpenChange }) => {
  const { t } = useTranslation();
  const { openStudio } = useStudio();
  const { auth } = useHfAuth();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { setSelectedDataset } = useSelectedDataset();
  const { models, loading: modelsLoading, refresh: refreshModels } =
    useModels();
  const { datasets, loading: datasetsLoading, refresh: refreshDatasets } =
    useDatasets();
  const [tab, setTab] = useState<Tab>("skills");
  const [mergeOpen, setMergeOpen] = useState(false);
  const [detailRepo, setDetailRepo] = useState<string | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const [manageSkill, setManageSkill] = useState<ModelItem | null>(null);
  const [manageOpen, setManageOpen] = useState(false);
  const [addDatasetOpen, setAddDatasetOpen] = useState(false);
  const [importDatasetOpen, setImportDatasetOpen] = useState(false);
  const [manageCachesOpen, setManageCachesOpen] = useState(false);
  const [addModelOpen, setAddModelOpen] = useState(false);
  const [importModelOpen, setImportModelOpen] = useState(false);

  const username = auth.status === "authenticated" ? auth.username : null;

  // "Add a dataset from Hugging Face": pin + select the typed Hub id, and
  // optionally kick off a background download into the local cache. Ported
  // from the old DatasetsPanel (handleAddFromHub / handleOpenCustom) — the
  // pin is best-effort, selection still works if the save call fails.
  const handleAddDatasetFromHub = async (repoId: string, download: boolean) => {
    setSelectedDataset(repoId);
    // Every toast here describes itself with a repo id — data, rendered
    // verbatim; only the title is ours to translate.
    toast({ title: t("library.sheet.toast.datasetSaved"), description: repoId });
    try {
      await saveCustomDataset(baseUrl, fetchWithHeaders, repoId);
      refreshDatasets();
    } catch {
      // Non-fatal: the dataset is still selected for training this session.
    }
    if (!download) return;
    try {
      await downloadDataset(baseUrl, fetchWithHeaders, repoId);
      toast({
        title: t("library.sheet.toast.downloadStarted"),
        description: repoId,
      });
    } catch (e) {
      toast({
        title: t("library.sheet.toast.downloadFailed"),
        // The backend's own explanation — left in its original language.
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  // "Import a dataset from disk": the dialog copied a local folder into the
  // cache and returns the new repo id — select it and refresh the list.
  const handleDatasetImported = (repoId: string) => {
    setSelectedDataset(repoId);
    refreshDatasets();
    toast({
      title: t("library.sheet.toast.datasetImported"),
      description: repoId,
    });
  };

  // Models twins of the two handlers above (ported from ModelsPanel).
  const handleAddModelFromHub = async (repoId: string, download: boolean) => {
    toast({ title: t("library.sheet.toast.modelSaved"), description: repoId });
    try {
      await saveCustomModel(baseUrl, fetchWithHeaders, repoId);
      refreshModels();
    } catch {
      // Non-fatal — the pin is best-effort.
    }
    if (!download) return;
    try {
      await downloadModel(baseUrl, fetchWithHeaders, repoId);
      toast({
        title: t("library.sheet.toast.downloadStarted"),
        description: repoId,
      });
    } catch (e) {
      toast({
        title: t("library.sheet.toast.downloadFailed"),
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handleModelImported = (repoId: string) => {
    refreshModels();
    toast({
      title: t("library.sheet.toast.modelImported"),
      description: repoId,
    });
  };

  const mySkills = useMemo(
    () => models.filter((m) => isMineSkill(m, username)),
    [models, username],
  );

  // Hides a Hub-only row once it's confirmed to have no video — this tab
  // opens DatasetDetailDialog on click, so a row without video would just
  // open to an empty state. See useHubVideoFilter for why the Train picker
  // must NOT do the same.
  const videoFilteredDatasets = useHubVideoFilter(datasets);
  const myDatasets = useMemo(
    () =>
      videoFilteredDatasets.filter((d) => {
        if (d.source === "local" || d.source === "both") return true;
        const ns = d.repo_id.includes("/") ? d.repo_id.split("/")[0] : null;
        return !!ns && !!username && ns.toLowerCase() === username.toLowerCase();
      }),
    [videoFilteredDatasets, username],
  );

  const runSkill = (model: ModelItem) => {
    // Only a Hub-ONLY model goes through the repo-id lazy-import path; a model
    // with a local copy (`local`/`both`) deploys through its existing job
    // registry entry (the run id is its job id) — re-importing would duplicate
    // the record and break offline runs.
    if (model.source === "hub" && model.hf_repo_id) {
      openStudio("deploy", { deploy: { source: "hub", id: model.hf_repo_id } });
    } else {
      openStudio("deploy", { deploy: { source: "job", id: model.id } });
    }
    onOpenChange(false);
  };

  const openDatasetDetail = (d: DatasetItem) => {
    setDetailRepo(d.repo_id);
    setDetailOpen(true);
  };

  return (
    <>
      <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
        <DialogPrimitive.Portal>
          <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-foreground/20 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0" />
          <DialogPrimitive.Content className="fixed inset-y-0 right-0 z-50 flex w-full max-w-sm flex-col border-l border-border bg-background shadow-2 duration-300 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:slide-out-to-right data-[state=open]:slide-in-from-right">
            <div className="flex items-center justify-between border-b border-border px-4 py-3">
              <DialogPrimitive.Title className="font-display text-lg font-semibold tracking-tight">
                {t("library.sheet.title")}
              </DialogPrimitive.Title>
              <DialogPrimitive.Close
                aria-label={t("library.sheet.close")}
                className="rounded-sm p-1 text-muted-foreground opacity-70 transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring"
              >
                <X className="h-4 w-4" />
              </DialogPrimitive.Close>
            </div>

            <div className="flex-1 overflow-y-auto p-4">
              <div className="mb-4 flex gap-1 rounded-lg border border-border bg-muted p-1">
                <SegButton
                  active={tab === "skills"}
                  onClick={() => setTab("skills")}
                >
                  {t("library.sheet.tabs.skills")}
                </SegButton>
                <SegButton
                  active={tab === "datasets"}
                  onClick={() => setTab("datasets")}
                >
                  {t("library.sheet.tabs.datasets")}
                </SegButton>
              </div>

              {tab === "skills" ? (
                <div className="flex flex-col gap-2">
                  {modelsLoading ? (
                    <p className="px-1 py-6 text-center text-sm text-muted-foreground">
                      {t("library.sheet.skills.loading")}
                    </p>
                  ) : mySkills.length === 0 ? (
                    <p className="px-1 py-6 text-center text-sm text-muted-foreground">
                      {t("library.sheet.skills.empty")}
                    </p>
                  ) : (
                    mySkills.map((m) => (
                      <div
                        key={m.id}
                        className="flex items-center gap-2 rounded-md border border-border bg-card p-3 transition-colors hover:border-ring"
                      >
                        <button
                          type="button"
                          onClick={() => {
                            setManageSkill(m);
                            setManageOpen(true);
                          }}
                          className="min-w-0 flex-1 text-left"
                          aria-label={t("library.sheet.skills.manage", {
                            name: skillDisplayTitle(t, m),
                          })}
                        >
                          <div className="flex items-center gap-2">
                            <span className="truncate font-medium">
                              {skillDisplayTitle(t, m)}
                            </span>
                          </div>
                          <p className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
                            {/* Policy type is data (an enum name the backend
                                owns); the step count arrives pre-formatted
                                ("16k"), so it goes in as `steps` rather than
                                i18next's `count`. The `m.source` fallback is
                                the raw enum — data too. */}
                            {[
                              m.policy_type
                                ? policyTypeDisplayName(m.policy_type)
                                : null,
                              m.steps != null
                                ? t("library.sheet.steps", {
                                    steps: formatCount(m.steps),
                                  })
                                : null,
                              m.private ? t("library.sheet.private") : null,
                            ]
                              .filter(Boolean)
                              .join(" · ") || m.source}
                          </p>
                        </button>
                        <Button
                          size="sm"
                          onClick={() => runSkill(m)}
                          className="shrink-0"
                          aria-label={t("library.sheet.skills.run", {
                            name: skillDisplayTitle(t, m),
                          })}
                        >
                          <Play className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    ))
                  )}
                </div>
              ) : (
                <div className="flex flex-col gap-2">
                  {datasetsLoading ? (
                    <p className="px-1 py-6 text-center text-sm text-muted-foreground">
                      {t("library.sheet.datasets.loading")}
                    </p>
                  ) : myDatasets.length === 0 ? (
                    <p className="px-1 py-6 text-center text-sm text-muted-foreground">
                      {t("library.sheet.datasets.empty")}
                    </p>
                  ) : (
                    myDatasets.map((d) => (
                      <button
                        key={d.repo_id}
                        type="button"
                        onClick={() => openDatasetDetail(d)}
                        className="flex items-center gap-2 rounded-md border border-border bg-card p-3 text-left transition-colors hover:border-ring"
                      >
                        <div className="min-w-0 flex-1">
                          <span className="block truncate font-medium">
                            {d.repo_id}
                          </span>
                          <p className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
                            {[
                              t(DATASET_SOURCE_KEY[d.source]),
                              d.private ? t("library.sheet.private") : null,
                            ]
                              .filter(Boolean)
                              .join(" · ")}
                          </p>
                        </div>
                      </button>
                    ))
                  )}
                </div>
              )}
            </div>

            <div className="flex flex-col gap-2 border-t border-border p-4">
              {tab === "skills" ? (
                <div className="flex gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setAddModelOpen(true)}
                    className="flex-1 gap-1.5"
                  >
                    <CloudDownload className="h-3.5 w-3.5" />
                    {t("library.sheet.actions.addFromHub")}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setImportModelOpen(true)}
                    className="flex-1 gap-1.5"
                  >
                    <FolderInput className="h-3.5 w-3.5" />
                    {t("library.sheet.actions.importFromDisk")}
                  </Button>
                </div>
              ) : (
                <>
                  <div className="flex gap-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setAddDatasetOpen(true)}
                      className="flex-1 gap-1.5"
                    >
                      <CloudDownload className="h-3.5 w-3.5" />
                      {t("library.sheet.actions.addFromHub")}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setImportDatasetOpen(true)}
                      className="flex-1 gap-1.5"
                    >
                      <FolderInput className="h-3.5 w-3.5" />
                      {t("library.sheet.actions.importFromDisk")}
                    </Button>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setManageCachesOpen(true)}
                    className="w-full gap-1.5"
                  >
                    <Archive className="h-3.5 w-3.5" />
                    {t("library.sheet.actions.manageCaches")}
                  </Button>
                </>
              )}
              <Button
                variant="outline"
                onClick={() => {
                  openStudio("collect");
                  onOpenChange(false);
                }}
                className="w-full gap-2"
              >
                <Plus className="h-4 w-4" />
                {t("library.sheet.actions.newSkill")}
              </Button>
              <Button
                variant="ghost"
                onClick={() => setMergeOpen(true)}
                className="w-full gap-2"
              >
                <GitMerge className="h-4 w-4" />
                {t("library.sheet.actions.mergeDatasets")}
              </Button>
            </div>
          </DialogPrimitive.Content>
        </DialogPrimitive.Portal>
      </DialogPrimitive.Root>

      <MergeDatasetsDialog
        open={mergeOpen}
        onOpenChange={setMergeOpen}
        datasets={datasets}
        onMerged={refreshDatasets}
      />

      <DatasetDetailDialog
        repoId={detailRepo}
        open={detailOpen}
        onOpenChange={setDetailOpen}
        // The sheet sits above the studio overlay — close it when a dataset
        // action opens a studio panel, or the panel appears "behind" it.
        onStudioAction={() => onOpenChange(false)}
      />

      <SkillManageDialog
        model={manageSkill}
        open={manageOpen}
        onOpenChange={setManageOpen}
        onChanged={refreshModels}
        onRun={(m) => {
          setManageOpen(false);
          runSkill(m);
        }}
      />

      <AddDatasetFromHubDialog
        open={addDatasetOpen}
        onOpenChange={setAddDatasetOpen}
        onAdd={handleAddDatasetFromHub}
      />
      <ImportDatasetFromDiskDialog
        open={importDatasetOpen}
        onOpenChange={setImportDatasetOpen}
        onImported={handleDatasetImported}
      />
      <ManageCachesDialog
        open={manageCachesOpen}
        onOpenChange={setManageCachesOpen}
        datasets={datasets}
        onCleared={refreshDatasets}
      />
      <AddModelFromHubDialog
        open={addModelOpen}
        onOpenChange={setAddModelOpen}
        onAdd={handleAddModelFromHub}
      />
      <ImportModelFromDiskDialog
        open={importModelOpen}
        onOpenChange={setImportModelOpen}
        onImported={handleModelImported}
      />
    </>
  );
};

export default LibrarySheet;
