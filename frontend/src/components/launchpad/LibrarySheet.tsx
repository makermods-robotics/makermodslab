import React, { useEffect, useMemo, useState } from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import {
  Archive,
  CloudDownload,
  FolderInput,
  GitMerge,
  Play,
  Plus,
  Trash2,
  Undo2,
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
  DeletedDatasetEntry,
  deleteDataset,
  downloadDataset,
  hideDataset,
  listDeletedDatasets,
  removeCustomDataset,
  saveCustomDataset,
  undoDatasetDelete,
} from "@/lib/replayApi";
import AddDatasetFromHubDialog from "@/components/landing/AddDatasetFromHubDialog";
import ImportDatasetFromDiskDialog from "@/components/landing/ImportDatasetFromDiskDialog";
import ManageCachesDialog from "@/components/landing/ManageCachesDialog";
import AddModelFromHubDialog from "@/components/landing/AddModelFromHubDialog";
import ImportModelFromDiskDialog from "@/components/landing/ImportModelFromDiskDialog";
import {
  formatCount,
  isMineSkill,
  skillTitle,
} from "@/components/launchpad/SkillCard";
import MergeDatasetsDialog from "@/components/landing/MergeDatasetsDialog";
import DatasetDetailDialog from "@/components/dialogs/DatasetDetailDialog";
import DeleteConfirmDialog from "@/components/dialogs/DeleteConfirmDialog";
import SkillManageDialog from "@/components/dialogs/SkillManageDialog";
import { DeleteResolution } from "@/lib/deleteSemantics";

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
  const [pendingDeleteDataset, setPendingDeleteDataset] =
    useState<DatasetItem | null>(null);
  const [showRecentlyDeleted, setShowRecentlyDeleted] = useState(false);
  const [deletedDatasets, setDeletedDatasets] = useState<DeletedDatasetEntry[]>([]);
  const [undoingTrashId, setUndoingTrashId] = useState<string | null>(null);

  const username = auth.status === "authenticated" ? auth.username : null;

  useEffect(() => {
    if (!showRecentlyDeleted) {
      setDeletedDatasets([]);
      return;
    }
    let cancelled = false;
    listDeletedDatasets(baseUrl, fetchWithHeaders)
      .then((entries) => {
        if (!cancelled) setDeletedDatasets(entries);
      })
      .catch(() => {
        if (!cancelled) setDeletedDatasets([]);
      });
    return () => {
      cancelled = true;
    };
  }, [showRecentlyDeleted, baseUrl, fetchWithHeaders]);

  // "Add a dataset from Hugging Face": pin + select the typed Hub id, and
  // optionally kick off a background download into the local cache. Ported
  // from the old DatasetsPanel (handleAddFromHub / handleOpenCustom) — the
  // pin is best-effort, selection still works if the save call fails.
  const handleAddDatasetFromHub = async (repoId: string, download: boolean) => {
    setSelectedDataset(repoId);
    toast({ title: "Dataset saved", description: repoId });
    try {
      await saveCustomDataset(baseUrl, fetchWithHeaders, repoId);
      refreshDatasets();
    } catch {
      // Non-fatal: the dataset is still selected for training this session.
    }
    if (!download) return;
    try {
      await downloadDataset(baseUrl, fetchWithHeaders, repoId);
      toast({ title: "Download started", description: repoId });
    } catch (e) {
      toast({
        title: "Couldn't start download",
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
    toast({ title: "Dataset imported", description: repoId });
  };

  const confirmDeleteDataset = async (resolution: DeleteResolution) => {
    const item = pendingDeleteDataset;
    if (!item) return;
    setPendingDeleteDataset(null);
    try {
      let result: { success: boolean; message?: string };
      if (resolution.action === "unpin") {
        result = await removeCustomDataset(baseUrl, fetchWithHeaders, item.repo_id);
      } else if (resolution.action === "hide") {
        result = await hideDataset(baseUrl, fetchWithHeaders, item.repo_id);
      } else {
        result = await deleteDataset(baseUrl, fetchWithHeaders, item.repo_id);
      }
      if (!result.success) {
        toast({
          title:
            resolution.action === "delete-local" ? "Delete failed" : "Couldn't remove",
          description: result.message ?? "Something went wrong",
          variant: "destructive",
        });
        return;
      }
      toast({
        title:
          resolution.action === "delete-local-copy"
            ? "Local copy removed"
            : resolution.action === "delete-local"
              ? "Dataset deleted"
              : "Removed from list",
        description: item.repo_id,
      });
      refreshDatasets();
    } catch (e) {
      toast({
        title:
          resolution.action === "delete-local" ? "Delete failed" : "Couldn't remove",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handleUndoDatasetDelete = async (trashId: string) => {
    const entry = deletedDatasets.find((d) => d.trash_id === trashId);
    if (!entry) return;
    setUndoingTrashId(trashId);
    try {
      const result = await undoDatasetDelete(
        baseUrl,
        fetchWithHeaders,
        entry.repo_id,
        trashId,
      );
      if (!result.success) {
        toast({
          title: "Undo failed",
          description: result.message || "Something went wrong",
          variant: "destructive",
        });
        return;
      }
      setDeletedDatasets((prev) => prev.filter((d) => d.trash_id !== trashId));
      refreshDatasets();
      toast({ title: "Dataset restored", description: entry.repo_id });
    } catch (e) {
      toast({
        title: "Undo failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setUndoingTrashId(null);
    }
  };

  // Models twins of the two handlers above (ported from ModelsPanel).
  const handleAddModelFromHub = async (repoId: string, download: boolean) => {
    toast({ title: "Model saved", description: repoId });
    try {
      await saveCustomModel(baseUrl, fetchWithHeaders, repoId);
      refreshModels();
    } catch {
      // Non-fatal — the pin is best-effort.
    }
    if (!download) return;
    try {
      await downloadModel(baseUrl, fetchWithHeaders, repoId);
      toast({ title: "Download started", description: repoId });
    } catch (e) {
      toast({
        title: "Couldn't start download",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handleModelImported = (repoId: string) => {
    refreshModels();
    toast({ title: "Model imported", description: repoId });
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
                My library
              </DialogPrimitive.Title>
              <DialogPrimitive.Close
                aria-label="Close library"
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
                  My skills
                </SegButton>
                <SegButton
                  active={tab === "datasets"}
                  onClick={() => setTab("datasets")}
                >
                  My datasets
                </SegButton>
              </div>

              {tab === "skills" ? (
                <div className="flex flex-col gap-2">
                  {modelsLoading ? (
                    <p className="px-1 py-6 text-center text-sm text-muted-foreground">
                      Loading skills…
                    </p>
                  ) : mySkills.length === 0 ? (
                    <p className="px-1 py-6 text-center text-sm text-muted-foreground">
                      No skills of yours yet — create one below.
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
                          aria-label={`Manage ${skillTitle(m)}`}
                        >
                          <div className="flex items-center gap-2">
                            <span className="truncate font-medium">
                              {skillTitle(m)}
                            </span>
                          </div>
                          <p className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
                            {[
                              m.policy_type
                                ? policyTypeDisplayName(m.policy_type)
                                : null,
                              m.steps != null
                                ? `${formatCount(m.steps)} steps`
                                : null,
                              m.private ? "private" : null,
                            ]
                              .filter(Boolean)
                              .join(" · ") || m.source}
                          </p>
                        </button>
                        <Button
                          size="sm"
                          onClick={() => runSkill(m)}
                          className="shrink-0"
                          aria-label={`Run ${skillTitle(m)}`}
                        >
                          <Play className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    ))
                  )}
                </div>
              ) : (
                <div className="flex flex-col gap-2">
                  <button
                    type="button"
                    onClick={() => setShowRecentlyDeleted((v) => !v)}
                    className="self-start rounded-md px-1 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
                  >
                    {showRecentlyDeleted
                      ? "Hide recently deleted"
                      : "Recently deleted"}
                  </button>

                  {showRecentlyDeleted && (
                    <div className="flex flex-col gap-2 border-b border-border pb-2">
                      {deletedDatasets.length === 0 ? (
                        <p className="px-1 py-2 text-center text-xs text-muted-foreground">
                          No recently deleted datasets.
                        </p>
                      ) : (
                        deletedDatasets.map((entry) => {
                          const hoursLeft = Math.max(
                            0,
                            (entry.expires_at * 1000 - Date.now()) / 3_600_000,
                          );
                          return (
                            <div
                              key={entry.trash_id}
                              className="flex items-center gap-2 rounded-md border border-border bg-card p-3 transition-colors hover:border-ring"
                            >
                              <div className="min-w-0 flex-1">
                                <span className="block truncate font-medium">
                                  {entry.repo_id}
                                </span>
                                <p className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
                                  expires{" "}
                                  {hoursLeft < 1
                                    ? "<1h"
                                    : `${Math.floor(hoursLeft)}h`}
                                </p>
                              </div>
                              <button
                                type="button"
                                onClick={() =>
                                  handleUndoDatasetDelete(entry.trash_id)
                                }
                                disabled={undoingTrashId !== null}
                                aria-label={`Undo delete of ${entry.repo_id}`}
                                title="Undo delete"
                                className="shrink-0 rounded p-1.5 text-muted-foreground hover:text-foreground disabled:opacity-50"
                              >
                                <Undo2 className="h-3.5 w-3.5" />
                              </button>
                            </div>
                          );
                        })
                      )}
                    </div>
                  )}

                  {datasetsLoading ? (
                    <p className="px-1 py-6 text-center text-sm text-muted-foreground">
                      Loading datasets…
                    </p>
                  ) : myDatasets.length === 0 ? (
                    <p className="px-1 py-6 text-center text-sm text-muted-foreground">
                      No datasets of yours yet — record one to get started.
                    </p>
                  ) : (
                    myDatasets.map((d) => (
                      <div
                        key={d.repo_id}
                        className="flex items-center gap-2 rounded-md border border-border bg-card p-3 transition-colors hover:border-ring"
                      >
                        <button
                          type="button"
                          onClick={() => openDatasetDetail(d)}
                          className="min-w-0 flex-1 text-left"
                        >
                          <div className="min-w-0 flex-1">
                            <span className="block truncate font-medium">
                              {d.repo_id}
                            </span>
                            <p className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
                              {d.source === "both"
                                ? "local + Hub"
                                : d.source === "hub"
                                  ? "on Hub"
                                  : "local only"}
                              {d.private ? " · private" : ""}
                            </p>
                          </div>
                        </button>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setPendingDeleteDataset(d);
                          }}
                          aria-label={`Delete ${d.repo_id}`}
                          title="Delete dataset"
                          className="shrink-0 rounded p-1.5 text-muted-foreground hover:text-destructive"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
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
                    Add from Hub
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setImportModelOpen(true)}
                    className="flex-1 gap-1.5"
                  >
                    <FolderInput className="h-3.5 w-3.5" />
                    Import from disk
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
                      Add from Hub
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setImportDatasetOpen(true)}
                      className="flex-1 gap-1.5"
                    >
                      <FolderInput className="h-3.5 w-3.5" />
                      Import from disk
                    </Button>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setManageCachesOpen(true)}
                    className="w-full gap-1.5"
                  >
                    <Archive className="h-3.5 w-3.5" />
                    Manage caches
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
                New Skill
              </Button>
              <Button
                variant="ghost"
                onClick={() => setMergeOpen(true)}
                className="w-full gap-2"
              >
                <GitMerge className="h-4 w-4" />
                Merge datasets
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
        onDeleted={refreshDatasets}
      />

      <DeleteConfirmDialog
        kind="dataset"
        item={pendingDeleteDataset}
        label={pendingDeleteDataset?.repo_id}
        onOpenChange={(o) => !o && setPendingDeleteDataset(null)}
        onConfirm={confirmDeleteDataset}
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
