import React, { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  JOB_STATE_LABELS,
  JobRecord,
  jobDisplayName,
  renameJob,
} from "@/lib/jobsApi";
import { middleEllipsis, runTaskTitle } from "@/lib/modelNames";
import {
  Square,
  Trash2,
  AlertTriangle,
  CheckCircle2,
  Globe,
  HardDrive,
  Loader2,
  XCircle,
  ExternalLink,
  Pencil,
  FastForward,
  Download,
  Upload,
} from "lucide-react";
import MetaRows from "@/components/library/MetaRows";
import DisplayName from "@/components/library/DisplayName";
import { useApi } from "@/contexts/ApiContext";
import { useStudio } from "@/contexts/StudioContext";
import { useToast } from "@/hooks/use-toast";
import { useTruncationTitle } from "@/hooks/useTruncationTitle";
import {
  JobCheckpoint,
  dedupeCheckpointEntries,
  listJobCheckpoints,
} from "@/lib/checkpointsApi";
import { buildResumeSeed } from "./resumeSeed";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import PolicyExtraDialog from "@/components/training/PolicyExtraDialog";

interface Props {
  job: JobRecord;
  onStop: (id: string) => void;
  onDelete: (id: string) => void;
  // Called after a successful rename so the parent can refetch the list.
  onRenamed?: () => void;
  // Runs this job was resumed from, nearest-parent first. Rendered nested and
  // hidden from the top-level list so a resumed lineage reads as one entry.
  ancestors?: JobRecord[];
}

function relativeTime(epochSec: number): string {
  const diff = Math.max(0, Date.now() / 1000 - epochSec);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

const statePresentation: Record<
  JobRecord["state"],
  {
    label: string;
    color: string;
    Icon: React.ComponentType<{ className?: string }>;
  }
> = {
  running: { label: JOB_STATE_LABELS.running, color: "text-ok", Icon: Loader2 },
  done: {
    label: JOB_STATE_LABELS.done,
    color: "text-muted-foreground",
    Icon: CheckCircle2,
  },
  failed: {
    label: JOB_STATE_LABELS.failed,
    color: "text-destructive",
    Icon: XCircle,
  },
  interrupted: {
    label: JOB_STATE_LABELS.interrupted,
    color: "text-warn",
    Icon: AlertTriangle,
  },
};

/**
 * Run-centric card for the jobs history: what a training is doing (state,
 * progress, logs) and the run-shaped actions — stop, Continue / Resume,
 * rename, delete.
 *
 * The model-shaped actions (Run, Fine-tune, Download) are NOT here: they
 * operate on the weights a run produced, so they live on ModelCard in the
 * model library. Continue / Resume stay because they are the opposite case —
 * they need a run that stopped SHORT of its target, which by construction
 * never gets a model card.
 */
const JobCard: React.FC<Props> = ({
  job,
  onStop,
  onDelete,
  onRenamed,
  ancestors = [],
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { openJobMonitor, openStudio } = useStudio();
  const present = statePresentation[job.state];
  const Icon = present.Icon;
  const isRunning = job.state === "running";
  const isImported = job.runner === "imported";
  // A Hub-backed import (vs a local-folder import) — provenance stays visible
  // after an untracked Hub repo is unified into a tracked imported card.
  const isHubImport = isImported && !!job.hf_repo_id;
  // Alias-aware display name; the true identity (run id / hub repo id) stays
  // visible as muted subtext when an alias is set.
  const displayName = jobDisplayName(job);
  const importedSource = job.hf_repo_id || job.output_dir;
  // The title line's shortenings: the task peel on a generated run name
  // (`runTaskTitle` — the policy is already on the Policy meta row below and
  // the dataset on its own, so the widest line stops repeating them),
  // middleEllipsis on an imported name (JS — the caller knows), and the div's
  // own `truncate` (CSS — measured on hover). Any of them earns a title; none
  // means the name is whole on screen and a tooltip would only echo it. Same
  // identity as ModelCard's: the full name, plus the source for an import (the
  // thing that actually locates the weights). `displayName` stays the full name
  // everywhere else on this card — the rename dialog prefills and compares
  // against what the run is really called, never the peeled label.
  const taskTitle = runTaskTitle(displayName);
  const titleText = isImported ? middleEllipsis(taskTitle) : taskTitle;
  const titleHover = useTruncationTitle(
    isImported && importedSource
      ? `${displayName}\n${importedSource}`
      : displayName,
    titleText !== displayName,
  );
  const stateLabel = isImported ? "Imported" : present.label;
  const isStarting = isRunning && job.metrics.total_steps === 0;
  const progressPct =
    job.metrics.total_steps > 0
      ? Math.min(
          100,
          (job.metrics.current_step / job.metrics.total_steps) * 100,
        )
      : 0;

  const subtitle = isImported
    ? importedSource
    : isStarting
      ? "starting…"
      : isRunning
        ? `started ${relativeTime(job.started_at)}`
        : job.ended_at != null
          ? `ended ${relativeTime(job.ended_at)}`
          : present.label.toLowerCase();

  // Checkpoints across the resume lineage (this run + the runs it resumed
  // from), each tagged with its owning job so inference/continue route to the
  // right run. Sorted newest-step-first so the current run sits above inherited
  // source checkpoints in the dropdown.
  const [lineageCheckpoints, setLineageCheckpoints] = useState<
    { job: JobRecord; ckpt: JobCheckpoint }[]
  >([]);
  // Selection is keyed on the checkpoint `ref` (its unique identity), not the
  // step — a lineage can hold two distinct checkpoints with the same step.
  const [selectedRef, setSelectedRef] = useState<string | null>(null);
  // Set on a failed run whose policy needs a lerobot extra that's still missing
  // — the likely cause. Offers the same one-click install as the training form.
  const [missingExtra, setMissingExtra] = useState<{
    policyType: string;
    packageName: string;
    installTarget: string;
    installHint: string;
  } | null>(null);
  const [extraDialogOpen, setExtraDialogOpen] = useState(false);

  // Rename dialog (mirrors CalibrationLibrary's rename UI). Sets a display
  // alias only — the run id / output dir / hub repo id never change.
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameValue, setRenameValue] = useState("");
  const [renameError, setRenameError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);

  const openRename = (e: React.MouseEvent) => {
    e.stopPropagation();
    setRenameValue(displayName);
    setRenameError(null);
    setRenameOpen(true);
  };

  const doRename = async () => {
    const next = renameValue.trim();
    if (!next) {
      setRenameError("Name cannot be empty.");
      return;
    }
    if (next === displayName) {
      setRenameOpen(false);
      return;
    }
    setRenaming(true);
    setRenameError(null);
    try {
      await renameJob(baseUrl, fetchWithHeaders, job.id, next);
      toast({
        title: "Model renamed",
        description: `"${displayName}" → "${next}".`,
      });
      setRenameOpen(false);
      onRenamed?.();
    } catch (e) {
      // 400/404 keep the dialog open with the message for a retry.
      setRenameError(e instanceof Error ? e.message : String(e));
    } finally {
      setRenaming(false);
    }
  };

  // Key ancestors by id+count so the frequent list refreshes (which hand us new
  // array refs) don't refetch unless the lineage actually changed.
  const ancestorKey = ancestors
    .map((a) => `${a.id}:${a.checkpoint_count}`)
    .join("|");

  useEffect(() => {
    const lineage = [job, ...ancestors].filter((j) => j.checkpoint_count > 0);
    if (lineage.length === 0) {
      setLineageCheckpoints([]);
      setSelectedRef(null);
      return;
    }
    let cancelled = false;
    Promise.all(
      lineage.map((j) =>
        listJobCheckpoints(baseUrl, fetchWithHeaders, j.id)
          .then((cks) => cks.map((ckpt) => ({ job: j, ckpt })))
          .catch(() => [] as { job: JobRecord; ckpt: JobCheckpoint }[]),
      ),
    ).then((results) => {
      if (cancelled) return;
      // Flat-merge, then collapse duplicates: a cloud resume reuses its
      // parent's output repo, so parent and child both enumerate the same Hub
      // checkpoint tree and every inherited step would otherwise appear
      // twice. Lineage order puts this run's list before its ancestors' and
      // the sort is stable for equal steps, so the surviving entry of a
      // duplicate pair is tagged with the nearest (child) run.
      const combined = dedupeCheckpointEntries(
        results.flat().sort((a, b) => b.ckpt.step - a.ckpt.step),
      );
      setLineageCheckpoints(combined);
      setSelectedRef((prev) =>
        prev != null && combined.some((c) => c.ckpt.ref === prev)
          ? prev
          : (combined[0]?.ckpt.ref ?? null),
      );
    });
    return () => {
      cancelled = true;
    };
    // job/ancestors captured via id+count keys above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseUrl, fetchWithHeaders, job.id, job.checkpoint_count, ancestorKey]);

  // A failed local training whose policy needs a lerobot extra that's still not
  // installed almost certainly died on that ImportError — surface the install.
  useEffect(() => {
    const policyType = job.config?.policy_type;
    if (job.state !== "failed" || job.runner !== "local" || !policyType) {
      setMissingExtra(null);
      return;
    }
    let cancelled = false;
    fetchWithHeaders(`${baseUrl}/system/policy-extra/${policyType}`)
      .then((r) => r.json())
      .then(
        (d: {
          policy_type: string;
          needs_extra: boolean;
          available: boolean;
          package: string;
          install_target: string;
          install_hint: string;
        }) => {
          if (cancelled) return;
          setMissingExtra(
            d.needs_extra && !d.available
              ? {
                  policyType: d.policy_type,
                  packageName: d.package,
                  installTarget: d.install_target,
                  installHint: d.install_hint,
                }
              : null,
          );
        },
      )
      .catch(() => {
        if (!cancelled) setMissingExtra(null);
      });
    return () => {
      cancelled = true;
    };
  }, [
    baseUrl,
    fetchWithHeaders,
    job.state,
    job.runner,
    job.config?.policy_type,
  ]);

  const handleAction = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (isRunning) {
      if (window.confirm("Stop this run?")) onStop(job.id);
    } else if (isImported) {
      if (
        window.confirm(
          "Remove this imported model? The source files are left untouched.",
        )
      )
        onDelete(job.id);
    } else if (job.runner === "hf_cloud") {
      // Cloud runs live on the Hub: deleting the record only removes it (and
      // its local logs) from this list — uploaded model repos are untouched.
      if (
        window.confirm(
          "Remove this cloud run from the list? Model repos on the Hub are not deleted.",
        )
      )
        onDelete(job.id);
    } else if (
      window.confirm("Delete this run? This wipes the output directory.")
    ) {
      onDelete(job.id);
    }
  };

  // The selected checkpoint may belong to this run or an inherited source run;
  // route inference/continue to whichever run owns it. Resolved by ref, so
  // same-step checkpoints from different runs can't be confused.
  const selected =
    lineageCheckpoints.find((c) => c.ckpt.ref === selectedRef) ?? null;
  const selectedJob = selected?.job ?? job;
  const selectedStep = selected?.ckpt.step ?? null;
  // Flat list for the dropdown (already newest-first).
  const checkpoints = lineageCheckpoints.map((c) => c.ckpt);

  // Resume — local Continue and cloud Resume alike — is for a run that stopped
  // SHORT of what it was configured to do: failed, interrupted or cancelled,
  // with a saved checkpoint below the step target.
  //
  // A `done` run is deliberately excluded. Resuming restores the optimizer AND
  // the LR schedule's position, and a completed run's schedule is spent: the
  // SmolVLA preset cosine-decays to a 2.5e-6 floor over a fixed 30k-step
  // horizon, so continuing past a reached target trains at floor LR — the loss
  // curve flattens and reads as convergence while the run is barely learning.
  // Fine-tuning from the final checkpoint is the intended way to build on a
  // completed run: it starts a FRESH schedule from those weights. Blanket rule,
  // no per-policy exceptions.
  //
  // Asked of a lineage ENTRY (its owning run + that checkpoint's step), not of
  // the card, so the buttons and the row's visibility below share one
  // definition: on a mixed lineage a done child can still offer Continue for a
  // checkpoint inherited from a run that stopped short.
  const canResumeEntry = (j: JobRecord, step: number) =>
    (j.runner === "local" || j.runner === "hf_cloud") &&
    !isRunning &&
    (j.state === "failed" || j.state === "interrupted") &&
    (j.config.steps === 0 || step < j.config.steps);

  // Continue (local resume) additionally needs the optimizer/step state to be
  // on THIS machine — i.e. a local run's own checkpoint dir.
  const canContinue =
    selectedStep != null &&
    selectedJob.runner === "local" &&
    canResumeEntry(selectedJob, selectedStep);

  // Resume (cloud): an HF Job is immutable once ended, so this launches a NEW
  // cloud job that continues from the parent's Hub checkpoint (restoring
  // optimizer + step, unlike Fine-tune).
  const canResumeCloud =
    selectedStep != null &&
    selectedJob.runner === "hf_cloud" &&
    canResumeEntry(selectedJob, selectedStep);

  // No dialog and no route jump: continuing opens the Train panel's
  // "Start a new training" form in resume mode, seeded from this run and the
  // dropdown's checkpoint — the same in-place flow ModelCard's Fine-tune uses,
  // rather than navigating away to /training and losing the studio.
  //
  // The payload itself comes from the ONE shared builder (buildResumeSeed), so
  // this and the library's row-level quick-resume can no longer drift. The
  // runner is derived there from the job: local Continue is gated on
  // canContinue and cloud Resume on canResumeCloud, both of which already
  // require selectedJob.runner to match, so there is nothing to pass.
  const goToResume = () => {
    if (selectedStep == null) return;
    openStudio("train", { train: { resume: buildResumeSeed(selectedJob, selectedStep) } });
  };

  const handleContinue = (e: React.MouseEvent) => {
    e.stopPropagation();
    goToResume();
  };

  const handleResumeCloud = (e: React.MouseEvent) => {
    e.stopPropagation();
    goToResume();
  };

  // Fine-tune, Run and Download used to live here too. They act on the WEIGHTS
  // a run produced, not on the run, so they moved to ModelCard in the model
  // library — where a finished run is surfaced as the model it is. Continue /
  // Resume stayed: they need a run that ended short of its target, and such a
  // run never gets a model card.

  const showProgressBar = isRunning;
  // The checkpoint row exists to serve Continue / Resume: it says which
  // checkpoint they will start from. Rendered whenever SOME checkpoint in the
  // lineage has such an action — never as a lone dropdown wired to nothing.
  // The gate is lineage-wide, not selection-wide, on purpose: a done run whose
  // ancestor stopped short must still show the dropdown, or the checkpoint that
  // lights Continue could never be reached (the default selection is the
  // newest, which is the done run's own).
  const showResumeRow =
    selectedStep != null &&
    lineageCheckpoints.some((c) => canResumeEntry(c.job, c.ckpt.step));

  // Unified metadata rows (same format as the dataset/model cards). Imported
  // models keep their source path in the subtitle; trainings surface what they
  // ran on. Rows are omitted when the fact is absent.
  const metaRows: Array<[string, string]> = [];
  if (job.config?.policy_type) metaRows.push(["Policy", job.config.policy_type]);
  // Imported pseudo-jobs carry the "(imported)" sentinel, not a real dataset.
  if (job.config?.dataset_repo_id && job.config.dataset_repo_id !== "(imported)")
    metaRows.push(["Dataset", job.config.dataset_repo_id]);
  if (!isImported && (job.config?.steps ?? 0) > 0)
    metaRows.push([
      "Steps",
      isRunning
        ? `${job.metrics.current_step.toLocaleString()} / ${job.config.steps.toLocaleString()}`
        : job.config.steps.toLocaleString(),
    ]);

  return (
    <Card
      onClick={() => {
        if (!isImported) openJobMonitor(job.id);
      }}
      className={`@container bg-card border-border rounded-md transition-colors h-full ${
        isImported ? "" : "cursor-pointer hover:border-ring/50 hover:bg-muted/40"
      }`}
    >
      <CardContent className="flex h-full flex-col gap-2.5 p-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <div
              className={`flex items-center gap-1.5 text-xs font-semibold ${present.color}`}
            >
              <Icon
                className={`w-3.5 h-3.5 ${isRunning ? "animate-spin" : ""}`}
              />
              {stateLabel}
            </div>
            {/* Location chip — with local and cloud runs mixed in one grid,
                each card says where it runs (same family as the dataset
                card's Local/Hub source badge). */}
            {!isImported ? (
              <div
                className="flex items-center gap-1 text-[11px] font-medium text-muted-foreground"
                title={
                  job.runner === "hf_cloud"
                    ? "Runs on Hugging Face cloud"
                    : "Runs on this machine"
                }
              >
                {job.runner === "hf_cloud" ? (
                  <Globe className="w-3 h-3" />
                ) : (
                  <HardDrive className="w-3 h-3" />
                )}
                {job.runner === "hf_cloud" ? "Cloud" : "Local"}
              </div>
            ) : null}
            {isHubImport ? (
              <div
                className="flex items-center gap-1 text-[11px] font-medium text-info"
                title="Imported from a Hugging Face Hub repo"
              >
                <Upload className="w-3 h-3" />
                from Hub
              </div>
            ) : null}
          </div>
          <div className="flex items-center gap-0.5">
            <Button
              variant="ghost"
              size="icon"
              onClick={openRename}
              className="h-7 w-7 text-muted-foreground hover:text-foreground"
              aria-label="Rename model"
              title="Rename"
            >
              <Pencil className="w-3.5 h-3.5" />
            </Button>
            {job.runner === "hf_cloud" && job.hf_job_url ? (
              <Button
                variant="ghost"
                size="icon"
                asChild
                className="h-7 w-7 text-muted-foreground hover:text-foreground"
                aria-label="Open Hub job page"
              >
                <a
                  href={job.hf_job_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                </a>
              </Button>
            ) : null}
            {/* A running cloud run is steered from its Hub page (the link
                above), so it gets no local action button. Everything else —
                including a FINISHED cloud run — gets stop/delete, so dead
                cloud runs are removable instead of link-only. */}
            {!(job.runner === "hf_cloud" && job.hf_job_url && isRunning) ? (
              <Button
                variant="ghost"
                size="icon"
                onClick={handleAction}
                className={`h-7 w-7 text-muted-foreground ${
                  isRunning ? "hover:text-foreground" : "hover:text-destructive"
                }`}
                aria-label={isRunning ? "Stop job" : "Delete job"}
              >
                {isRunning ? (
                  <Square className="w-3.5 h-3.5" />
                ) : (
                  <Trash2 className="w-3.5 h-3.5" />
                )}
              </Button>
            ) : null}
          </div>
        </div>
        <div>
          {/* An imported title is already peeled down to its task by the
              backend (utils/naming.derive_imported_title), so it normally fits;
              middleEllipsis only bites on the fallback — an unparseable
              community repo name — where both ends carry meaning and `truncate`
              would keep only the head. Whenever either shortening bites, the
              hover title is the full name (plus the repo id / path for an
              import), so the exact identity is one hover away; a name that
              fits gets no title, since it is already all there. */}
          <div className="text-foreground font-semibold truncate" {...titleHover}>
            {titleText}
          </div>
          {/* When aliased, keep the true identity visible: the run id for
              trainings (imported models already show their repo id / path in
              the subtitle below). */}
          {!isImported && job.display_name ? (
            <div className="text-[11px] text-muted-foreground truncate" title={job.id}>
              {job.id}
            </div>
          ) : null}
          {/* Imported subtitles are file paths — truncate the *start* (rtl
              flips the ellipsis to the left) so the more useful tail stays
              visible. The leading LRM keeps the path's first "/" from being
              bidi-reordered to the wrong end. */}
          <div
            className="text-xs text-muted-foreground truncate"
            title={subtitle}
            style={
              isImported ? { direction: "rtl", textAlign: "left" } : undefined
            }
          >
            {isImported ? "\u200e" + subtitle : subtitle}
          </div>
        </div>
        <MetaRows rows={metaRows} />
        {showProgressBar ? (
          <div className="relative h-5 w-full overflow-hidden rounded-md bg-muted border border-border">
            <div
              className="h-full bg-info transition-[width] duration-500"
              style={{ width: `${progressPct}%` }}
            />
            <div className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-white tabular-nums drop-shadow">
              {isStarting ? "Training starting…" : `${progressPct.toFixed(1)}%`}
            </div>
          </div>
        ) : null}
        {showResumeRow ? (
          // Single-line action row: the checkpoint dropdown flexes and the
          // buttons never wrap. Continue / Resume are icon-only so the row fits
          // a narrow grid card.
          <div className="mt-auto flex items-center gap-1.5 pt-1">
            {/* A single checkpoint offers no choice — skip the dropdown and
                free the row for the buttons. */}
            {checkpoints.length > 1 ? (
              <div className="min-w-0 flex-1">
                <CheckpointDropdown
                  checkpoints={checkpoints}
                  selectedRef={selectedRef}
                  onChange={(c) => setSelectedRef(c.ref)}
                  className="w-full min-w-0"
                />
              </div>
            ) : null}
            {canContinue ? (
              <Button
                size="sm"
                variant="outline"
                onClick={handleContinue}
                className="h-8 w-8 shrink-0 p-0 border-info/50 text-info hover:bg-info/10"
                aria-label="Continue training from this checkpoint"
                title="Continue training from this checkpoint"
              >
                <FastForward className="w-3.5 h-3.5" />
              </Button>
            ) : null}
            {canResumeCloud ? (
              <Button
                size="sm"
                variant="outline"
                onClick={handleResumeCloud}
                className="h-8 w-8 shrink-0 p-0 border-info/50 text-info hover:bg-info/10"
                aria-label="Resume this cloud run from its last checkpoint"
                title="Resume: launch a new cloud job continuing from this checkpoint"
              >
                <FastForward className="w-3.5 h-3.5" />
              </Button>
            ) : null}
          </div>
        ) : null}
        {missingExtra ? (
          // When there's no resume row this is the card's only CTA — pin it
          // to the footer like every other card's action row.
          <div
            className={`flex items-center ${showResumeRow ? "" : "mt-auto pt-1"}`}
          >
            <Button
              size="sm"
              variant="outline"
              onClick={(e) => {
                e.stopPropagation();
                setExtraDialogOpen(true);
              }}
              className="h-8 gap-1.5 border-warn/50 text-warn hover:bg-warn/10"
            >
              <Download className="w-3.5 h-3.5" /> Install{" "}
              {missingExtra.installTarget}
            </Button>
          </div>
        ) : null}
      </CardContent>
      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent
          className="bg-background border-border"
          onClick={(e) => e.stopPropagation()}
        >
          <DialogHeader>
            <DialogTitle>Rename model</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              Sets a display name only — the underlying{" "}
              {isImported && job.hf_repo_id ? "Hub repo" : "run"} (
              <span className="font-mono text-muted-foreground">
                {isImported ? importedSource : job.id}
              </span>
              ) is not moved or changed.
            </DialogDescription>
          </DialogHeader>
          <Input
            value={renameValue}
            onChange={(e) => {
              setRenameValue(e.target.value);
              setRenameError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void doRename();
              }
            }}
            autoFocus
            placeholder="New name"
            className="bg-background border-input"
          />
          {renameError && <p className="text-sm text-destructive">{renameError}</p>}
          <DialogFooter className="flex gap-2 justify-end">
            <Button
              variant="outline"
              className="border-border text-muted-foreground"
              onClick={() => setRenameOpen(false)}
            >
              Cancel
            </Button>
            <Button
              className="bg-primary hover:bg-primary/90 text-primary-foreground"
              disabled={
                renaming ||
                !renameValue.trim() ||
                renameValue.trim() === displayName
              }
              onClick={doRename}
            >
              {renaming ? "Renaming…" : "Rename"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      {missingExtra ? (
        <PolicyExtraDialog
          open={extraDialogOpen}
          onOpenChange={setExtraDialogOpen}
          policyType={missingExtra.policyType}
          packageName={missingExtra.packageName}
          installTarget={missingExtra.installTarget}
          installHint={missingExtra.installHint}
        />
      ) : null}
    </Card>
  );
};

export default JobCard;
