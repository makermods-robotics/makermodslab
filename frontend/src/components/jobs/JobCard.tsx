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
import { runTaskTitle } from "@/lib/modelNames";
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
  Play,
  FastForward,
  Download,
  Sparkles,
  Upload,
} from "lucide-react";
import MetaRows from "@/components/library/MetaRows";
import DisplayName from "@/components/library/DisplayName";
import { useApi } from "@/contexts/ApiContext";
import { useStudio } from "@/contexts/StudioContext";
import { useToast } from "@/hooks/use-toast";
import {
  LineageCheckpoint,
  blockedByContinuedOwner,
  buildResumeSeed,
  loadLineageCheckpoints,
  resumableCheckpoints,
} from "./resumeSeed";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import PolicyExtraDialog from "@/components/training/PolicyExtraDialog";

interface Props {
  job: JobRecord;
  onStop: (id: string) => void;
  onDelete: (id: string) => void;
  onPlay: (job: JobRecord, step: number) => void;
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
 * Card for the jobs history: what a training is doing (state, progress, logs)
 * and the run-shaped actions — stop, Resume, rename, delete.
 *
 * Resume is the one this change owns, and it is now a SINGLE verb decided by
 * the shared rule in resumeSeed (`resumableCheckpoints`), so this card and the
 * library row's one-click resume cannot disagree.
 *
 * The model-shaped actions (Run, Fine-tune, Download) are still rendered here.
 * They act on the WEIGHTS a run produced rather than on the run, and upstream
 * they move to ModelCard in the model library — but that rewiring (ModelsLibrary
 * collapsing runs to one card per Hub repo, ModelCard actually being rendered)
 * is deliberately not on this stack yet, and ModelsLibrary still mounts THIS
 * card with `onPlay` for imported and uploaded models. Removing them here would
 * make Run/Fine-tune/Download unreachable, so they stay until the card swap
 * lands.
 */
const JobCard: React.FC<Props> = ({
  job,
  onStop,
  onDelete,
  onPlay,
  onRenamed,
  ancestors = [],
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { openStudio, openJobMonitor } = useStudio();
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
  // What the title line RENDERS: a generated run name peeled to the task it
  // learned. The policy is already on the Policy meta row below and the dataset
  // on its own, so the widest line stops repeating them. Everything else on
  // this card keeps `displayName` — the rename dialog prefills and compares
  // against what the run is really called, never the peeled label — and the
  // title's hover reveals it too (DisplayName's `full`).
  const taskTitle = runTaskTitle(displayName);
  const importedSource = job.hf_repo_id || job.output_dir;
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
    LineageCheckpoint[]
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
    let cancelled = false;
    // The shared loader — this card and the library row's one-click resume
    // read the SAME ancestor-path list, so they can't offer different
    // checkpoints for the same run.
    loadLineageCheckpoints(baseUrl, fetchWithHeaders, job, ancestors).then(
      (combined) => {
        if (cancelled) return;
        setLineageCheckpoints(combined);
        // Default to the newest RESUMABLE checkpoint, not the newest one in
        // the lineage. They differ exactly when the newest is excluded by the
        // rule — most often an ancestor checkpoint saved past this run's step
        // target — and defaulting to it opened the card with the Resume button
        // hidden and no hint that picking an older step would bring it back.
        // The dropdown still lists the whole lineage; only the starting
        // selection is narrowed. Falls back to the newest checkpoint when
        // nothing is resumable, so the row still reads as a checkpoint list.
        setSelectedRef((prev) =>
          prev != null && combined.some((c) => c.ckpt.ref === prev)
            ? prev
            : (resumableCheckpoints(job, combined)[0]?.ckpt.ref ??
              combined[0]?.ckpt.ref ??
              null),
        );
      },
    );
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

  const handlePlay = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedStep == null) return;
    onPlay(selectedJob, selectedStep);
  };

  // Resume — one verb, whichever runner owns the checkpoint — is decided by
  // the ONE shared rule (resumableCheckpoints), so this card and the library
  // row's one-click resume can't disagree about whether a run can continue.
  //
  // The question is asked of the LEAF, not of each lineage entry: this card
  // always renders the tip of a chain (the libraries give a row to leaves
  // only), and it is the tip's state and step target that say whether the
  // TRAINING is unfinished. The rule then picks which checkpoints along the
  // leaf's ancestor path may serve as the source, and each one carries its
  // owning run so the seed resumes from the run that actually holds it.
  //
  // Behaviour change worth naming: a `done` leaf no longer offers Resume for
  // a checkpoint inherited from an ancestor that stopped short. That chain
  // reached its target — the way to build on it is Fine-tune, which starts a
  // fresh LR schedule instead of restoring a spent one.
  //
  // Sticks (user decision 2026-08-07), the second and larger narrowing: an
  // INHERITED checkpoint is no longer a resume source at all. Its owner has a
  // child by construction — the run below it on this card's own path — and
  // jobs.py refuses a second continuation of a run that already has one, so
  // the button would only buy a 409. The dropdown still lists the whole
  // lineage, and an inherited checkpoint remains selectable for inference; it
  // is Resume alone that dims, via the existing `selectedIsResumable` gate.
  // The empty-handed case (this tip saved nothing, so no selection lights the
  // button) is recoverable by deleting this run and resuming its parent —
  // which the library row's toast spells out.
  const resumable = resumableCheckpoints(job, lineageCheckpoints);
  const selectedIsResumable =
    !isRunning &&
    selected != null &&
    resumable.some((c) => c.ckpt.ref === selected.ckpt.ref);

  // ONE verb. This used to fork into "Continue" (a local-owned checkpoint) and
  // "Resume" (a cloud-owned one) — two buttons that differed only in wording,
  // because pre-F7 the owner's runner really was a constraint on where the
  // continuation could run. It no longer is: jobs.py continues on EITHER
  // runner, moving the checkpoint as needed, and the Train form merely opens
  // with Compute DEFAULTED to where the owner ran, toggle live. So the
  // distinction had become dead information at the button level — it named the
  // form's default, which the form itself already shows and lets you change.
  // Asking a novice to tell two verbs apart for one action is the cost; there
  // is no benefit left to pay it with.
  const canResume = selectedIsResumable;

  // Step 0 is the whole-repo/single-model sentinel (see CheckpointDropdown),
  // which has no meaningful number to name — matching its "latest" label.
  const resumeLabel =
    selectedStep == null || selectedStep === 0
      ? "Resume from latest"
      : `Resume from step ${selectedStep.toLocaleString()}`;

  // No dialog and no route jump: continuing opens the Train panel's
  // "Start a new training" form in resume mode, seeded from this run and the
  // dropdown's checkpoint — the same in-place flow Fine-tune already uses,
  // rather than navigating away to /training and losing the studio.
  //
  // The configurator PREFILLS from this seed, then renders read-only the
  // settings lerobot rebuilds from the checkpoint anyway (batch size, seed,
  // device, optimizer, AMP). Steps, the log/save cadence, the worker count,
  // the cloud flavor and the timeout stay editable — those a continuation can
  // really change, and so is the runner it continues ON (F7's cross-runner
  // resume).
  //
  // The payload itself comes from the ONE shared builder (buildResumeSeed), so
  // this and the library's row-level quick-resume can no longer drift. The
  // runner is derived there from the OWNING run — the seed's record of where
  // the parent actually ran, which the form reads both as its default Compute
  // and as the fact that decides whether continuing elsewhere has to move the
  // checkpoint first (F7).
  const goToResume = () => {
    if (selectedStep == null) return;
    openStudio("train", {
      train: { resume: buildResumeSeed(selectedJob, selectedStep) },
    });
  };

  const handleResume = (e: React.MouseEvent) => {
    e.stopPropagation();
    goToResume();
  };

  // Fine-tune: start a FRESH run whose weights are initialized from this
  // model's checkpoint. Unlike Continue (which needs optimizer/step state and
  // is local-only), fine-tuning is weights-only, so it works from ANY source
  // that has a checkpoint — the user's own local and cloud runs included.
  //
  // The old gate also required runner === "imported". That was a workaround for
  // MT2, where a hub step-ref was truncated to the bare repo id and a fine-tune
  // of a cloud run silently trained from ROOT weights instead of the step the
  // user picked; excluding cloud runs (and, collaterally, local ones) hid the
  // bug rather than fixing it. jobs._resolve_finetune_pretrained_path now
  // branches on all three runners and keeps a step-suffixed hub ref verbatim
  // for the trainer's host to materialize, so there is nothing left to guard:
  // gate on state and checkpoints only.
  const canFinetune =
    !isRunning && lineageCheckpoints.length > 0 && selectedStep != null;

  // No dialog and no route jump: fine-tuning opens the Train panel's
  // "Start a new training" form with the base skill (and the dropdown's
  // checkpoint step) prefilled.
  const handleFinetune = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedStep == null) return;
    openStudio("train", {
      train: {
        baseJobId: selectedJob.id,
        baseStep: selectedStep,
        baseName: jobDisplayName(selectedJob),
      },
    });
  };

  // A local checkpoint can be exported as a zip while training continues, so
  // (unlike Continue) this doesn't gate on !isRunning.
  const canDownload =
    selectedJob.runner === "local" &&
    lineageCheckpoints.length > 0 &&
    selectedStep != null;

  const handleDownload = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedStep == null) return;
    try {
      const res = await fetchWithHeaders(
        `${baseUrl}/jobs/${selectedJob.id}/checkpoints/${selectedStep}/download`,
      );
      if (!res.ok) {
        toast({ title: "Download failed", variant: "destructive" });
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${jobDisplayName(selectedJob)}_step_${selectedStep}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      toast({
        title: "Download failed",
        description: String(err),
        variant: "destructive",
      });
    }
  };

  const showProgressBar = isRunning;
  // The action row carries Run / Resume / Fine-tune / Download, so it is gated
  // on having a checkpoint at all rather than on Resume's own rule. (Upstream
  // of this branch the row exists to serve Resume alone and narrows to
  // `resumable.length > 0`; the model-shaped actions only move off this card
  // when ModelsLibrary is rewired to render ModelCard — see the header note.)
  const showInferenceRow =
    lineageCheckpoints.length > 0 && selectedStep != null;
  // The empty-handed tip: this run saved nothing of its own, and everything it
  // inherited belongs to a run that has already been continued — so under the
  // sticks rule nothing here is resumable, and the recovery is to delete THIS
  // run and resume the one behind it. Without this the card just dropped its
  // whole action row and said nothing, which is the one case where silence is
  // worst: the checkpoints are visibly there, and the reason they can't be
  // used is a rule the card never mentions.
  //
  // Keyed on `resumable.length === 0`, not on `canResume`: when this run does
  // have resumable checkpoints of its own, a selection that lands on an
  // inherited one is a choice the user just made (the default is the newest
  // resumable), and the answer there is to pick another step, not to delete
  // the run.
  const continuedOwnerHint =
    resumable.length === 0
      ? blockedByContinuedOwner(job, lineageCheckpoints)
      : null;

  // (Upstream the row is gated on `resumable.length > 0 || continuedOwnerHint`,
  // because there it carries Resume alone. Here it also carries Run /
  // Fine-tune / Download, so `showInferenceRow` above — "there is a checkpoint
  // at all" — is already the wider gate and covers the empty-handed tip.)

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
          <DisplayName
            name={taskTitle}
            full={displayName}
            className="text-foreground font-semibold"
          />
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
        {showInferenceRow ? (
          // Single-line action row: the checkpoint dropdown flexes and the
          // buttons never wrap. Resume now carries its step in the label — one
          // verb, so the row says what it will do without a hover — and takes
          // the width it needs; the dropdown yields, which fits now that there
          // is one resume button here instead of two. Download stays icon-only
          // so the row still fits a narrow grid card.
          <div className="mt-auto flex items-center gap-1.5 pt-1">
            {/* A single checkpoint offers no choice — skip the dropdown and
                free the row for the buttons (imported models are the common
                case: one "latest" entry). */}
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
            <Button
              size="sm"
              onClick={handlePlay}
              className="h-8 shrink-0 gap-1 bg-primary hover:bg-primary/90 text-primary-foreground"
              aria-label="Run inference with this checkpoint"
            >
              <Play className="w-3.5 h-3.5" /> Run
            </Button>
            {canResume ? (
              <Button
                size="sm"
                variant="outline"
                onClick={handleResume}
                className="h-8 shrink-0 gap-1.5 px-2.5 border-info/50 text-info hover:bg-info/10"
                aria-label={resumeLabel}
                title="Opens the training form to continue from this checkpoint. Compute defaults to where this checkpoint's run executed, and can be retargeted before you start."
              >
                <FastForward className="w-3.5 h-3.5 shrink-0" />
                <span className="truncate">{resumeLabel}</span>
              </Button>
            ) : continuedOwnerHint ? (
              // Where the button would be, the two-step way to get it back.
              // Names the run and the step so it can be acted on without
              // opening anything else — same guidance, same numbers, as the
              // library row's toast (both read the rule, not a re-derivation).
              <div className="min-w-0 flex-1 text-[11px] leading-tight text-muted-foreground">
                Saved no checkpoints of its own, and its parent was already
                continued. Delete this run, then resume{" "}
                <span className="text-foreground font-medium">
                  {jobDisplayName(continuedOwnerHint.job)}
                </span>{" "}
                from step {continuedOwnerHint.ckpt.step.toLocaleString()}.
              </div>
            ) : null}
            {canFinetune ? (
              <Button
                size="sm"
                variant="outline"
                onClick={handleFinetune}
                className="h-8 shrink-0 gap-1 border-primary/40 text-primary hover:bg-primary/10"
                aria-label="Fine-tune a new run from this model's weights"
                title="Fine-tune a new run from this model's weights"
              >
                <Sparkles className="w-3.5 h-3.5" />
                {/* Label only when the card is wide enough for the whole row
                    to stay on one line; the tooltip covers the narrow case. */}
                <span className="hidden @[13rem]:inline">Fine-tune</span>
              </Button>
            ) : null}
            {canDownload ? (
              <Button
                size="sm"
                variant="outline"
                onClick={handleDownload}
                className="h-8 w-8 shrink-0 p-0 border-border text-muted-foreground hover:bg-muted"
                aria-label="Download this checkpoint"
                title="Download this checkpoint"
              >
                <Download className="w-3.5 h-3.5" />
              </Button>
            ) : null}
          </div>
        ) : null}
        {missingExtra ? (
          // When there's no inference row this is the card's only CTA — pin it
          // to the footer like every other card's action row.
          <div
            className={`flex items-center ${showInferenceRow ? "" : "mt-auto pt-1"}`}
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
