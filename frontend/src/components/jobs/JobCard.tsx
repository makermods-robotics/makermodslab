import React, { useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
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
  FOUNDATION_BASE_REPO_IDS,
  JOB_STATE_LABELS,
  JobRecord,
  RunKind,
  formatBaseModel,
  jobDisplayName,
  renameJob,
  splitCheckpointRef,
} from "@/lib/jobsApi";
import { jobRunStamp, runTaskTitle } from "@/lib/modelNames";
import {
  Square,
  Trash2,
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  CheckCircle2,
  Clock,
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
import RunKindChip from "@/components/jobs/RunKindChip";
import NodeLocationChip from "@/components/jobs/NodeLocationChip";
import DisplayName from "@/components/library/DisplayName";
import { useJobsData } from "@/components/jobs/JobsDataContext";
import { useApi } from "@/contexts/ApiContext";
import { useStudio } from "@/contexts/StudioContext";
import { useToast } from "@/hooks/use-toast";
import {
  LineageCheckpoint,
  buildResumeSeed,
  checkpointOwners,
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

/**
 * State → badge presentation. `labelKey` is a translation KEY (JOB_STATE_LABELS
 * holds key paths, not words) because this map is evaluated at import time —
 * resolved copy here would freeze whichever language loaded first. The colour
 * and icon are not copy and stay put.
 */
const statePresentation = {
  // The same Clock + warn pairing the Hub's QUEUED stage wears in the run
  // dropdown, so one word means one look everywhere.
  queued: {
    labelKey: JOB_STATE_LABELS.queued,
    color: "text-warn",
    Icon: Clock,
  },
  running: {
    labelKey: JOB_STATE_LABELS.running,
    color: "text-ok",
    Icon: Loader2,
  },
  done: {
    labelKey: JOB_STATE_LABELS.done,
    color: "text-muted-foreground",
    Icon: CheckCircle2,
  },
  failed: {
    labelKey: JOB_STATE_LABELS.failed,
    color: "text-destructive",
    Icon: XCircle,
  },
  interrupted: {
    labelKey: JOB_STATE_LABELS.interrupted,
    color: "text-warn",
    Icon: AlertTriangle,
  },
} as const;

/** The subtitle's last branch says the state as running text. One key per
 * state rather than .toLowerCase() on a translated word — case is a property of
 * a script, not of a string. */
const SUBTITLE_STATE_KEYS = {
  queued: "jobs.jobCard.subtitleState.queued",
  running: "jobs.jobCard.subtitleState.running",
  done: "jobs.jobCard.subtitleState.done",
  failed: "jobs.jobCard.subtitleState.failed",
  interrupted: "jobs.jobCard.subtitleState.interrupted",
} as const;

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
  const { t } = useTranslation();
  const { openStudio, openJobMonitor } = useStudio();
  // Queue plumbing comes from the shared provider (this card is only ever
  // mounted under it): the uncapped queue list is what the up/down controls
  // reorder against, and cancelQueued carries the expect_state precondition.
  // `jobs` is only to put a NAME on a fine-tune's source run; the lineage this
  // card renders still comes from the `ancestors` prop.
  const { jobs, queue, cancelQueued, moveQueued } = useJobsData();
  const present = statePresentation[job.state];
  const Icon = present.Icon;
  const isRunning = job.state === "running";
  const isQueued = job.state === "queued";
  const isImported = job.runner === "imported";

  // What this run started FROM, and what to call it.
  //
  // Read straight off the run's own config — the local registry already knows,
  // so unlike the Hub card there is nothing to parse. The kind mirrors
  // _hub_job_provenance's so the two cards classify identically: a base that is
  // one of the VLA foundation checkpoints was DEFAULTED there by jobs.py when
  // the user chose no starting point, and is not a fine-tune.
  const runKind: RunKind = job.config?.resume
    ? "resume"
    : job.config?.finetune_from_job_id
      ? "finetune"
      : job.config?.policy_pretrained_path
        ? FOUNDATION_BASE_REPO_IDS.has(job.config.policy_pretrained_path)
          ? "foundation"
          : "finetune"
        : "scratch";
  // A fine-tune of a run this machine still has is named by that run — its
  // display alias when it has one, since that is what the user calls it
  // everywhere else. Falls back to the job id, which is readable by
  // construction ("act_cube_2026-08-01_12-00-00").
  const sourceRecord = job.config?.finetune_from_job_id
    ? jobs.find((j) => j.id === job.config.finetune_from_job_id)
    : undefined;
  const baseModel = isImported
    ? null
    : job.config?.finetune_from_job_id
      ? formatBaseModel({
          base_job_id:
            sourceRecord?.display_name ??
            sourceRecord?.name ??
            job.config.finetune_from_job_id,
          base_step:
            job.config.finetune_from_step != null
              ? String(job.config.finetune_from_step)
              : null,
        })
      : formatBaseModel(splitCheckpointRef(job.config?.policy_pretrained_path));
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
  // A queued badge carries its 1-based queue position ("Queued · #2") — the
  // position is derived per response server-side, never a frozen copy.
  const queuePosition = isQueued ? (job.queue_position ?? 0) : 0;
  // Where this run sits in the provider's uncapped queue list — what the
  // up/down controls swap against. -1 while the two fetches disagree for a
  // moment; both buttons then disable rather than reorder blind.
  const queueIndex = isQueued
    ? queue.findIndex((q) => q.id === job.id)
    : -1;
  const stateLabel = isImported
    ? t("jobs.location.imported")
    : isQueued && queuePosition > 0
      ? t("jobs.jobState.queuedAt", { position: queuePosition })
      : t(present.labelKey);
  const isStarting = isRunning && job.metrics.total_steps === 0;
  const progressPct =
    job.metrics.total_steps > 0
      ? Math.min(
          100,
          (job.metrics.current_step / job.metrics.total_steps) * 100,
        )
      : 0;

  // One key per branch — the card picks a whole sentence, it never assembles
  // one. `relativeTime` output is passed in pre-formatted: duration formatting
  // is deliberately left exactly as it was.
  const subtitle = isImported
    ? importedSource
    : isStarting
      ? t("jobs.progress.starting")
      : isRunning
        ? t("jobs.jobCard.subtitle.started", {
            when: relativeTime(job.started_at),
          })
        : job.ended_at != null
          ? t("jobs.jobCard.subtitle.ended", {
              when: relativeTime(job.ended_at),
            })
          : t(SUBTITLE_STATE_KEYS[job.state]);

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
      setRenameError(t("jobs.rename.empty"));
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
        title: t("jobs.rename.toastTitle"),
        // Both names are user data — interpolated, never translated.
        description: t("jobs.rename.toastDescription", {
          from: displayName,
          to: next,
        }),
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
    fetchWithHeaders(`${baseUrl}/api/v1/system/policy-extra/${policyType}`)
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

  // The four window.confirm() questions below stay in ENGLISH on purpose: a
  // native confirm draws its OK/Cancel from the BROWSER's locale, so a
  // translated question over English buttons reads worse than an English one.
  // Replacing them with AlertDialogs is a separate UX change.
  const handleAction = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (isQueued) {
      // Cancel, not stop: the run never started, and the record is removed
      // outright. cancelQueued sends the expect_state precondition so a click
      // against a stale queue refuses instead of killing a promoted run.
      if (window.confirm("Cancel this queued run? It hasn't started training."))
        cancelQueued(job.id);
    } else if (isRunning) {
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
  // Continue is deliberately NOT step-selectable (user decision 2026-08-10):
  // it always takes the newest resumable checkpoint, `resumableCheckpoints`
  // being newest-first. That is the one-click behaviour the jobs library row
  // has always had, so the two entry points now differ in nothing at all.
  //
  // The dropdown beside it still drives Run / Fine-tune / Download, where
  // picking an older checkpoint is a real choice. Resuming from one is not:
  // it re-trains steps the chain already covered, and it blocks the intended
  // end state, where a continuation ABSORBS its parent (inheriting its
  // checkpoints outright, so there is no ancestor left to reach back to). A
  // rewound child re-writes steps its parent still holds, so absorbing it
  // would have to either collide or silently discard the superseded ones;
  // continuing from the newest checkpoint is a pure append and does neither.
  //
  // What stays is the REACH, invisibly: the newest resumable checkpoint may
  // be owned by an ANCESTOR, since a tip that died before saving anything has
  // none of its own. The user presses one button and never learns which run
  // held the bytes — but that reach is what keeps every row in the library
  // done-or-resumable, and buildResumeSeed still records the owner separately
  // from the lineage edge so the chain stays linear.
  const resumable = resumableCheckpoints(job, lineageCheckpoints);
  const resumeSource = isRunning ? null : (resumable[0] ?? null);

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
  const canResume = resumeSource != null;

  // Names the step it WILL use, not one the user chose — the button is now the
  // whole decision. Step 0 is the whole-repo/single-model sentinel (see
  // CheckpointDropdown), which has no meaningful number to name, matching its
  // "latest" label.
  const resumeStep = resumeSource?.ckpt.step ?? null;
  const resumeLabel =
    resumeStep == null || resumeStep === 0
      ? t("jobs.jobCard.resumeLatest")
      // The step arrives ALREADY formatted (toLocaleString, untouched) — passed
      // under its own name so i18next never tries to re-derive a plural from it.
      : t("jobs.jobCard.resumeStep", { step: resumeStep.toLocaleString() });

  // No dialog and no route jump: continuing opens the Train panel's
  // "Start a new training" form in resume mode, seeded from this run and the
  // newest resumable checkpoint — the same in-place flow Fine-tune already uses,
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
  // this and the library's row-level quick-resume can no longer drift — they
  // now pass the same entry, the top of the same rule's list. It is handed
  // THIS card's run (the leaf being continued — the lineage edge) plus that
  // entry, which carries its own owner; the builder keeps those two apart.
  // Passing the owner as the run is precisely the fork bug chain rewind fixes.
  // The runner still follows the owner there, since it says where the bytes
  // live and therefore whether they must move first (F7).
  const goToResume = () => {
    if (resumeSource == null) return;
    openStudio("train", {
      train: { resume: buildResumeSeed(job, resumeSource) },
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
        `${baseUrl}/api/v1/jobs/${selectedJob.id}/checkpoints/${selectedStep}/download`,
      );
      if (!res.ok) {
        toast({
          title: t("jobs.jobCard.downloadFailed"),
          variant: "destructive",
        });
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
        title: t("jobs.jobCard.downloadFailed"),
        description: String(err),
        variant: "destructive",
      });
    }
  };

  const showProgressBar = isRunning;
  // The action row carries Run / Resume / Fine-tune / Download, so it is gated
  // on having a checkpoint at all rather than on Resume's own rule. A QUEUED
  // run suppresses it wholesale: it has trained nothing yet, and the lineage
  // checkpoints it may inherit belong to the run it will continue — offering
  // Run/Fine-tune off a card that says "Queued" reads as progress it hasn't
  // made. Progress affordances (the bar above) are likewise running-only. (Upstream
  // of this branch the row exists to serve Resume alone and narrows to
  // `resumable.length > 0`; the model-shaped actions only move off this card
  // when ModelsLibrary is rewired to render ModelCard — see the header note.)
  const showInferenceRow =
    !isQueued && lineageCheckpoints.length > 0 && selectedStep != null;
  // The previous commit's delete-first hint is gone with the rule that needed
  // it: an empty-handed tip is simply resumable — it continues ITSELF from
  // the newest thing its ancestors saved — so there is no longer
  // a state where visible inherited checkpoints are unusable for a reason the
  // card never says. What is left is a genuinely dead chain (nothing saved
  // anywhere, or everything owned by finished runs), and the library row's
  // toast explains that on click.
  //
  // (Upstream the row is gated on `resumable.length > 0`, because there it
  // carries Resume alone. Here it also carries Run / Fine-tune / Download, so
  // `showInferenceRow` above — "there is a checkpoint at all" — is the wider
  // gate and stays.)

  // Unified metadata rows (same format as the dataset/model cards). Imported
  // models keep their source path in the subtitle; trainings surface what they
  // ran on. Rows are omitted when the fact is absent.
  // Only the LABELS are translated; every value beside them is data (policy
  // type, dataset repo id) or a pre-formatted number left exactly as it was.
  const metaRows: Array<[string, string]> = [];
  if (baseModel)
    metaRows.push([t("jobs.meta.base"), baseModel]);
  if (job.config?.policy_type)
    metaRows.push([t("jobs.meta.policy"), job.config.policy_type]);
  // Imported pseudo-jobs carry the "(imported)" sentinel, not a real dataset.
  if (job.config?.dataset_repo_id && job.config.dataset_repo_id !== "(imported)")
    metaRows.push([t("jobs.meta.dataset"), job.config.dataset_repo_id]);
  if (!isImported && (job.config?.steps ?? 0) > 0)
    metaRows.push([
      t("jobs.meta.steps"),
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
            {/* Location chip — with local, cloud and node runs mixed in one
                grid, each card says where it runs (same family as the dataset
                card's Local/Hub source badge). A lan_node run names its NODE
                (falling back to the short instance id once the node has left
                the registry) — its own component, so the registry lookup only
                mounts when there is a node to name. */}
            {!isImported ? (
              job.runner === "lan_node" ? (
                <NodeLocationChip job={job} />
              ) : (
                <div
                  className="flex items-center gap-1 text-[11px] font-medium text-muted-foreground"
                  title={
                    job.runner === "hf_cloud"
                      ? t("jobs.location.cloudTitle")
                      : t("jobs.location.localTitle")
                  }
                >
                  {job.runner === "hf_cloud" ? (
                    <Globe className="w-3 h-3" />
                  ) : (
                    <HardDrive className="w-3 h-3" />
                  )}
                  {job.runner === "hf_cloud"
                    ? t("jobs.location.cloud")
                    : t("jobs.location.local")}
                </div>
              )
            ) : null}
            {/* What the run IS, beside where it runs. An import has no starting
                point of its own — its weights came from elsewhere entirely. */}
            {!isImported ? <RunKindChip kind={runKind} /> : null}
            {isHubImport ? (
              <div
                className="flex items-center gap-1 text-[11px] font-medium text-info"
                title={t("jobs.location.fromHubTitle")}
              >
                <Upload className="w-3 h-3" />
                {t("jobs.location.fromHub")}
              </div>
            ) : null}
          </div>
          <div className="flex items-center gap-0.5">
            {/* MINIMAL reorder: one slot up / one slot down, driving the
                whole-list reorder endpoint (no drag-and-drop). Only on queued
                cards, and only while there is something to reorder past. */}
            {isQueued && queue.length > 1 ? (
              <>
                <Button
                  variant="ghost"
                  size="icon"
                  disabled={queueIndex <= 0}
                  onClick={(e) => {
                    e.stopPropagation();
                    moveQueued(job.id, -1);
                  }}
                  className="h-7 w-7 text-muted-foreground hover:text-foreground"
                  aria-label={t("jobs.jobCard.queueMoveUpAria")}
                  title={t("jobs.jobCard.queueMoveUpAria")}
                >
                  <ArrowUp className="w-3.5 h-3.5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  disabled={queueIndex < 0 || queueIndex >= queue.length - 1}
                  onClick={(e) => {
                    e.stopPropagation();
                    moveQueued(job.id, 1);
                  }}
                  className="h-7 w-7 text-muted-foreground hover:text-foreground"
                  aria-label={t("jobs.jobCard.queueMoveDownAria")}
                  title={t("jobs.jobCard.queueMoveDownAria")}
                >
                  <ArrowDown className="w-3.5 h-3.5" />
                </Button>
              </>
            ) : null}
            <Button
              variant="ghost"
              size="icon"
              onClick={openRename}
              className="h-7 w-7 text-muted-foreground hover:text-foreground"
              aria-label={t("jobs.actions.renameAria")}
              title={t("jobs.actions.rename")}
            >
              <Pencil className="w-3.5 h-3.5" />
            </Button>
            {job.runner === "hf_cloud" && job.hf_job_url ? (
              <Button
                variant="ghost"
                size="icon"
                asChild
                className="h-7 w-7 text-muted-foreground hover:text-foreground"
                aria-label={t("jobs.actions.openHubJob")}
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
                aria-label={
                  isQueued
                    ? t("jobs.jobCard.cancelQueuedAria")
                    : isRunning
                      ? t("jobs.jobCard.stopAria")
                      : t("jobs.jobCard.deleteAria")
                }
              >
                {isQueued ? (
                  <XCircle className="w-3.5 h-3.5" />
                ) : isRunning ? (
                  <Square className="w-3.5 h-3.5" />
                ) : (
                  <Trash2 className="w-3.5 h-3.5" />
                )}
              </Button>
            ) : null}
          </div>
        </div>
        <div>
          {/* The run NUMBER rides OUTSIDE the truncating title, so a long name
              can never eat the one token that identifies the run — every run on
              a resume chain shares this title, and the number is the same
              handle the backend's refusals lead with (a 409 naming #46 points
              at a row the user can find). Its hover carries the run stamp and
              the full id. */}
          <div className="flex min-w-0 items-baseline gap-1.5">
            {job.job_number > 0 ? (
              <span
                className="shrink-0 font-mono text-muted-foreground"
                title={`${jobRunStamp(job.id)} · ${job.id}`}
              >
                #{job.job_number}
              </span>
            ) : null}
            <DisplayName
              name={taskTitle}
              full={displayName}
              className="min-w-0 text-foreground font-semibold"
            />
            {/* Continuation marker. A resume hides the parent and shows the
                successor in its place, which reads as "my run vanished and a
                new card appeared" unless the new card says what it is. Naming
                the parent's number makes the chain legible, and explains why
                the row the user was watching is no longer in the list.
                `ancestors` is nearest-parent-first. */}
            {ancestors.length > 0 && ancestors[0].job_number > 0 ? (
              <span
                className="shrink-0 whitespace-nowrap font-mono text-[11px] text-muted-foreground"
                title={t("jobs.jobCard.continuesTitle", {
                  chain: ancestors
                    .filter((a) => a.job_number > 0)
                    .map((a) => `#${a.job_number}`)
                    .join(" ← "),
                })}
              >
                {t("jobs.jobCard.continues", {
                  parent: `#${ancestors[0].job_number}`,
                })}
              </span>
            ) : null}
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
          {/* Why it failed, on the card itself. The reason was already on the
              record but only the job dialog rendered it, so a run that died on
              something actionable (out of memory) looked, from the list the
              user actually lands on, like it had failed for no reason. */}
          {job.state === "failed" && job.error_message ? (
            <div
              className="text-destructive mt-0.5 line-clamp-2 text-[11px]"
              title={job.error_message}
            >
              {job.error_message}
            </div>
          ) : null}
        </div>
        <MetaRows rows={metaRows} />
        {showProgressBar ? (
          <div className="relative h-5 w-full overflow-hidden rounded-md bg-muted border border-border">
            <div
              className="h-full bg-info transition-[width] duration-500"
              style={{ width: `${progressPct}%` }}
            />
            <div className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-white tabular-nums drop-shadow">
              {isStarting
                ? t("jobs.jobCard.trainingStarting")
                : `${progressPct.toFixed(1)}%`}
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
                  // This list is a whole lineage, so two entries can both read
                  // "step 2000" and belong to different runs — and the runs
                  // share a display name, because a continuation continues the
                  // same model. The dropdown renders this only when the list
                  // really does span runs.
                  owners={checkpointOwners(lineageCheckpoints)}
                />
              </div>
            ) : null}
            <Button
              size="sm"
              onClick={handlePlay}
              className="h-8 shrink-0 gap-1 bg-primary hover:bg-primary/90 text-primary-foreground"
              aria-label={t("jobs.actions.runInferenceCheckpoint")}
            >
              <Play className="w-3.5 h-3.5" /> {t("jobs.actions.run")}
            </Button>
            {canResume ? (
              <Button
                size="sm"
                variant="outline"
                onClick={handleResume}
                className="h-8 shrink-0 gap-1.5 px-2.5 border-info/50 text-info hover:bg-info/10"
                aria-label={resumeLabel}
                title={t("jobs.jobCard.resumeHint")}
              >
                <FastForward className="w-3.5 h-3.5 shrink-0" />
                <span className="truncate">{resumeLabel}</span>
              </Button>
            ) : null}
            {canFinetune ? (
              <Button
                size="sm"
                variant="outline"
                onClick={handleFinetune}
                className="h-8 shrink-0 gap-1 border-primary/40 text-primary hover:bg-primary/10"
                // Same words on both, so one key rather than two that could
                // drift apart.
                aria-label={t("jobs.actions.fineTuneHint")}
                title={t("jobs.actions.fineTuneHint")}
              >
                <Sparkles className="w-3.5 h-3.5" />
                {/* Label only when the card is wide enough for the whole row
                    to stay on one line; the tooltip covers the narrow case. */}
                <span className="hidden @[13rem]:inline">
                  {t("jobs.actions.fineTune")}
                </span>
              </Button>
            ) : null}
            {canDownload ? (
              <Button
                size="sm"
                variant="outline"
                onClick={handleDownload}
                className="h-8 w-8 shrink-0 p-0 border-border text-muted-foreground hover:bg-muted"
                // One key: the hover text and the accessible name are the same
                // sentence on the same control.
                aria-label={t("jobs.actions.download")}
                title={t("jobs.actions.download")}
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
              <Download className="w-3.5 h-3.5" />{" "}
              {/* The install target is the backend's own package spec — data. */}
              {t("jobs.jobCard.install", { target: missingExtra.installTarget })}
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
            <DialogTitle>{t("jobs.rename.title")}</DialogTitle>
            <DialogDescription className="text-muted-foreground">
              {/* One sentence with the identity embedded in it, not three
                  concatenated fragments — <0/> is the mono span below and its
                  contents (run id / repo id) are data. */}
              <Trans
                i18nKey="jobs.rename.description"
                values={{
                  target: t(
                    isImported && job.hf_repo_id
                      ? "jobs.rename.targetHubRepo"
                      : "jobs.rename.targetRun",
                  ),
                }}
                components={[
                  <span key="0" className="font-mono text-muted-foreground">
                    {isImported ? importedSource : job.id}
                  </span>,
                ]}
              />
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
            placeholder={t("jobs.rename.placeholder")}
            className="bg-background border-input"
          />
          {renameError && <p className="text-sm text-destructive">{renameError}</p>}
          <DialogFooter className="flex gap-2 justify-end">
            <Button
              variant="outline"
              className="border-border text-muted-foreground"
              onClick={() => setRenameOpen(false)}
            >
              {t("common.cancel")}
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
              {renaming ? t("jobs.rename.submitting") : t("jobs.rename.submit")}
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
          purpose="training"
        />
      ) : null}
    </Card>
  );
};

export default JobCard;
