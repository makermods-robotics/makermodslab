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
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { JobRecord, jobDisplayName, renameJob } from "@/lib/jobsApi";
import {
  Box,
  Download,
  ExternalLink,
  Globe,
  HardDrive,
  History,
  Pencil,
  Play,
  Sparkles,
  Trash2,
  Upload,
} from "lucide-react";
import MetaRows from "@/components/library/MetaRows";
import { useApi } from "@/contexts/ApiContext";
import { useStudio } from "@/contexts/StudioContext";
import { useToast } from "@/hooks/use-toast";
import { useTruncationTitle } from "@/hooks/useTruncationTitle";
import {
  JobCheckpoint,
  dedupeCheckpointEntries,
  listJobCheckpoints,
} from "@/lib/checkpointsApi";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import { displayDedupeSuffix, splitDedupeSuffix } from "@/lib/modelNames";
import { checkpointOwners } from "./resumeSeed";

interface Props {
  /** The model, represented by its backing job record: an import, or a
   * finished local/cloud training (the run IS the model it produced). */
  model: JobRecord;
  onDelete: (id: string) => void;
  onPlay: (job: JobRecord, step: number) => void;
  /** Called after a successful rename so the parent can refetch the list. */
  onRenamed?: () => void;
  /** Runs this model descends from, nearest-parent first. Their checkpoints
   * join this model's own in the checkpoint dropdown, so a resumed lineage
   * reads as one model with its run history folded in behind it. */
  ancestors?: JobRecord[];
}

/** A trained run's auto-generated name peeled down to the TASK it learned.
 *
 * jobs.start names an unnamed run "{POLICY} · {dataset_repo_id}" —
 * "SMOLVLA · makermods/eraser_place". The card prints the policy in its own
 * Policy meta row and the full dataset id (namespace included) in its Dataset
 * row, so both halves on the title line are duplication eating the width the
 * identity needs. Peel both, leaving "eraser_place" — the same shape imported
 * titles already have, and the same reasoning that retired the "Imported · "
 * prefix. This peels the JOB RECORD's name; the Deploy picker reads the models
 * listing instead, which the backend peels identically in
 * models._run_identity_name.
 *
 * Conservative by construction: the head must equal this record's OWN policy
 * type, which is exactly the generated shape. A job name the user typed that
 * happens to contain " · " keeps every word — and the namespace peel is gated
 * behind that same check, so it can never eat a slash out of a human's name.
 */
function runIdentityTitle(name: string, policyType?: string | null): string {
  if (!policyType) return name;
  const separator = name.indexOf(" · ");
  if (separator < 0) return name;
  const head = name.slice(0, separator);
  const tail = name.slice(separator + 3);
  if (!tail || head.toLowerCase() !== policyType.toLowerCase()) return name;
  // The tail is a dataset repo id: keep the segment after the namespace. A
  // bare name with no slash is already the task.
  const slash = tail.indexOf("/");
  return slash >= 0 && slash < tail.length - 1 ? tail.slice(slash + 1) : tail;
}

function relativeTime(epochSec: number): string {
  const diff = Math.max(0, Date.now() / 1000 - epochSec);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

/**
 * Model-centric card for the studio's model library. The primary unit is the
 * MODEL, not the run that produced it: the actions that operate on weights —
 * Run, Fine-tune, Download — live here, and the runs behind the model collapse
 * into the checkpoint dropdown above them.
 *
 * Deliberately NOT a JobCard: the run-shaped affordances (stop, Continue /
 * Resume, progress, monitor) stay on JobCard in the jobs history, because a
 * model card only ever describes a run that FINISHED (or an import) — a failed
 * or interrupted run never becomes a model, so there is nothing here to resume.
 *
 * Every affordance always renders. One whose backing predicate is unmet renders
 * DISABLED with a specific, human-readable reason on hover, rather than
 * vanishing — a control that disappears teaches the user nothing about why.
 */
const ModelCard: React.FC<Props> = ({
  model,
  onDelete,
  onPlay,
  onRenamed,
  ancestors = [],
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { t } = useTranslation();
  const { openStudio } = useStudio();

  const isRunning = model.state === "running";
  const isImported = model.runner === "imported";
  // A Hub-backed import (vs a local-folder import) — provenance stays visible
  // after an untracked Hub repo is unified into a tracked imported card.
  const isHubImport = isImported && !!model.hf_repo_id;
  // Alias-aware display name; the true identity (run id / hub repo id) stays
  // visible as muted subtext when an alias is set. A user's alias is rendered
  // exactly as typed — only the auto-generated name is peeled.
  const displayName = model.display_name?.trim()
    ? jobDisplayName(model)
    : runIdentityTitle(model.name, model.config?.policy_type);
  // Only an auto-derived imported title carries a dedupe suffix; a trained
  // run's name (or a user's own alias) is rendered whole, so don't split it.
  const { base: titleBase, suffix: titleSuffix } = isImported
    ? splitDedupeSuffix(displayName)
    : { base: displayName, suffix: null };
  // The suffix is rendered a year shorter than it is stored, so the two spans
  // below no longer spell the full name between them — see the render comment.
  // `splitDedupeSuffix` hands back the suffix with its parentheses already
  // stripped (lib/modelNames, shared with components/library/DisplayName), so
  // they go back on here, where the pair is rendered.
  const titleSuffixText = titleSuffix && `(${displayDedupeSuffix(titleSuffix)})`;
  const importedSource = model.hf_repo_id || model.output_dir;

  // Hover title for the name — only when the name is actually shortened (see
  // useTruncationTitle). The base+suffix pair below is measured as ONE unit:
  // either span truncating means the visible name is incomplete, so the title
  // hangs on their container and carries the whole thing. An import adds its
  // source, which is the identity that actually locates the weights.
  //
  // The suffix's dropped year is the OTHER kind of shortening, the one the DOM
  // can't see: nothing is clipped, yet what's on screen is not the whole name.
  // The caller knows it, so it passes it in — comparing what the pair actually
  // renders against the full name, which stays true however the suffix's
  // display form changes later.
  const renderedTitle = titleSuffixText
    ? `${titleBase} ${titleSuffixText}`
    : titleBase;
  const titleHover = useTruncationTitle(
    isImported && importedSource
      ? `${displayName}\n${importedSource}`
      : displayName,
    renderedTitle !== displayName,
  );

  // Where the model came from — the header chip mirrors the dataset/job cards'
  // source badges (Imported / from Hub / Local / Cloud).
  const originLabel = isImported
    ? t("jobs.location.imported")
    : model.runner === "hf_cloud"
      ? t("jobs.location.cloud")
      : t("jobs.location.local");
  const OriginIcon = isImported
    ? Box
    : model.runner === "hf_cloud"
      ? Globe
      : HardDrive;

  // One key per branch. `relativeTime` output is passed in pre-formatted —
  // duration formatting is deliberately left exactly as it was.
  const subtitle = isImported
    ? importedSource
    : model.ended_at != null
      ? t("jobs.modelCard.trained", { when: relativeTime(model.ended_at) })
      : t("jobs.modelCard.created", { when: relativeTime(model.started_at) });

  // Checkpoints across the lineage (this model + the runs it descends from),
  // each tagged with its owning run so Run/Fine-tune/Download route to the
  // right record. Sorted newest-step-first.
  const [lineageCheckpoints, setLineageCheckpoints] = useState<
    { job: JobRecord; ckpt: JobCheckpoint }[]
  >([]);
  // Selection is keyed on the checkpoint `ref` (its unique identity), not the
  // step — a lineage can hold two distinct checkpoints with the same step.
  const [selectedRef, setSelectedRef] = useState<string | null>(null);

  // Rename dialog (mirrors JobCard's / CalibrationLibrary's rename UI). Sets a
  // display alias only — the run id / output dir / hub repo id never change.
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
      await renameJob(baseUrl, fetchWithHeaders, model.id, next);
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

  // The three window.confirm() questions below stay in ENGLISH on purpose: a
  // native confirm draws its OK/Cancel from the BROWSER's locale, so a
  // translated question over English buttons reads worse than an English one.
  // Replacing them with AlertDialogs is a separate UX change.
  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (isImported) {
      if (
        window.confirm(
          "Remove this imported model? The source files are left untouched.",
        )
      )
        onDelete(model.id);
    } else if (model.runner === "hf_cloud") {
      // Cloud runs live on the Hub: deleting the record only removes it (and
      // its local logs) from this list — uploaded model repos are untouched.
      if (
        window.confirm(
          "Remove this cloud model from the list? Model repos on the Hub are not deleted.",
        )
      )
        onDelete(model.id);
    } else if (
      window.confirm("Delete this model? This wipes the output directory.")
    ) {
      onDelete(model.id);
    }
  };

  // Key ancestors by id+count so the frequent list refreshes (which hand us new
  // array refs) don't refetch unless the lineage actually changed.
  const ancestorKey = ancestors
    .map((a) => `${a.id}:${a.checkpoint_count}`)
    .join("|");

  useEffect(() => {
    const lineage = [model, ...ancestors].filter((j) => j.checkpoint_count > 0);
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
      // twice. Lineage order puts this model's list before its ancestors' and
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
    // model/ancestors captured via id+count keys above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseUrl, fetchWithHeaders, model.id, model.checkpoint_count, ancestorKey]);

  // The selected checkpoint may belong to this model or an inherited source
  // run; route actions to whichever run owns it. Resolved by ref, so same-step
  // checkpoints from different runs can't be confused.
  const selected =
    lineageCheckpoints.find((c) => c.ckpt.ref === selectedRef) ?? null;
  const selectedJob = selected?.job ?? model;
  const selectedStep = selected?.ckpt.step ?? null;
  // Flat list for the dropdown (already newest-first).
  const checkpoints = lineageCheckpoints.map((c) => c.ckpt);
  const hasCheckpoints = lineageCheckpoints.length > 0 && selectedStep != null;

  const handlePlay = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedStep == null) return;
    onPlay(selectedJob, selectedStep);
  };

  // Fine-tune: start a FRESH run whose weights are initialized from this
  // model's checkpoint. Unlike Continue/Resume (which need optimizer + step
  // state and live on the run's JobCard), fine-tuning is weights-only, so it
  // works from ANY source that has a checkpoint — imports and the user's own
  // finished local and cloud runs alike.
  const canFinetune = !isRunning && hasCheckpoints;

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
  // (unlike the run-side resume actions) this doesn't gate on !isRunning.
  const canDownload = selectedJob.runner === "local" && hasCheckpoints;

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

  // Presentation model for the affordances: every control always renders, but
  // one whose backing predicate is unmet renders DISABLED with a specific
  // reason (surfaced on hover). The enabled decisions reuse the predicates
  // above unchanged — only the presentation flips from hidden→disabled. Each
  // reason maps to *why* that particular predicate failed.

  // Run: needs a checkpoint to load.
  const runEnabled = hasCheckpoints;
  const runReason = runEnabled
    ? undefined
    : t("jobs.modelCard.reason.noCheckpointToRun");

  const finetuneEnabled = canFinetune;
  let finetuneReason: string | undefined;
  if (!finetuneEnabled) {
    if (isRunning)
      finetuneReason = t("jobs.modelCard.reason.finetuneWhileRunning");
    else finetuneReason = t("jobs.modelCard.reason.noCheckpointToFinetune");
  }

  // A hub-imported base whose weights live on the Hub, not on disk here.
  const selectedIsHubImport =
    selectedJob.runner === "imported" && !!selectedJob.hf_repo_id;

  // Download: exports a local on-disk checkpoint as a zip. Gated to
  // runner === "local" on BOTH sides — server.py's download_checkpoint rejects
  // anything else — because a local run's output_dir is server-generated under
  // the jobs root while an imported one is a path the user typed, so serving it
  // would make this "zip any directory on the server", which --lan mode exposes
  // to the network. The gate is deliberate; the wording below explains why
  // rather than implying the checkpoint doesn't exist (for a disk import it
  // plainly does — it's just not ours to re-serve).
  const downloadEnabled = canDownload;
  let downloadReason: string | undefined;
  if (!downloadEnabled) {
    if (!hasCheckpoints) {
      downloadReason = t("jobs.modelCard.reason.noCheckpointToDownload");
    } else if (selectedIsHubImport) {
      downloadReason = t("jobs.modelCard.reason.hubImportWeights");
    } else if (selectedJob.runner === "imported") {
      downloadReason = selectedJob.output_dir
        ? // The path is data — interpolated, never translated.
          t("jobs.modelCard.reason.importedFromDisk", {
            path: selectedJob.output_dir,
          })
        : t("jobs.modelCard.reason.importedNoExport");
    } else if (selectedJob.runner === "hf_cloud") {
      downloadReason = t("jobs.modelCard.reason.cloudCheckpoints");
    } else {
      downloadReason = t("jobs.modelCard.reason.localOnly");
    }
  }

  // Checkpoint picker: only meaningful when the lineage holds more than one
  // checkpoint to choose between (a flat import is the common single case).
  const selectorEnabled = hasCheckpoints && lineageCheckpoints.length > 1;
  let selectorReason: string | undefined;
  if (!selectorEnabled) {
    if (!hasCheckpoints)
      selectorReason = t("jobs.modelCard.reason.noCheckpointsToChoose");
    else selectorReason = t("jobs.modelCard.reason.oneCheckpoint");
  }

  // Wraps a control so its disabled reason is reachable on hover. A natively
  // `disabled` button emits no pointer events, so the tooltip trigger is a
  // `span` wrapper (the pattern used in RobotCorner). When there's no reason
  // (the control is enabled) the child renders bare, with its own handlers.
  const withHint = (
    reason: string | undefined,
    node: React.ReactNode,
    wrapperClassName?: string,
  ) =>
    reason ? (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className={cn("inline-flex", wrapperClassName)}>{node}</span>
        </TooltipTrigger>
        <TooltipContent side="top">{reason}</TooltipContent>
      </Tooltip>
    ) : (
      <>{node}</>
    );

  // Metadata block (same format as the dataset/job cards). Steps pairs the
  // selected checkpoint with the run's step TARGET — the dropdown alone says
  // where the model stopped but never what it was aiming at, so a card can't
  // otherwise tell "20,000 of 20,000" from "20,000 of 200,000". Step and target
  // are read off the run that OWNS the selected checkpoint (which may be an
  // ancestor), so the two halves of the ratio always belong together. Imported
  // models carry a step-0 sentinel and no target, so they get no row rather
  // than a dishonest "0 / 0".
  // Only the LABELS are translated; every value beside them is data (policy
  // type, dataset repo id) or a pre-formatted number left exactly as it was.
  const metaRows: Array<[string, string]> = [];
  if (model.config?.policy_type)
    metaRows.push([t("jobs.meta.policy"), model.config.policy_type]);
  // Imported pseudo-jobs carry the "(imported)" sentinel, not a real dataset.
  if (
    model.config?.dataset_repo_id &&
    model.config.dataset_repo_id !== "(imported)"
  )
    metaRows.push([t("jobs.meta.dataset"), model.config.dataset_repo_id]);
  const stepTarget = selectedJob.config?.steps ?? 0;
  if (isImported) {
    // An import has no step TARGET to pair against: register_imported fills
    // `config` with placeholders (dataset_repo_id "(imported)", the training
    // form's default step count), so "20,000 / 10,000" was the checkpoint
    // measured against a number nobody chose — and one the model actually
    // exceeded, which reads as a broken progress ratio. The backend distrusts
    // the same placeholders when it ranks records (models._jobs_by_hub_repo
    // skips imports for exactly this reason). Show the checkpoint alone — bare,
    // in the same style as a trained card's numerator; the row's own "Steps"
    // label is what keeps a lone number from reading as a total.
    if (selectedStep != null && selectedStep > 0)
      metaRows.push([t("jobs.meta.steps"), selectedStep.toLocaleString()]);
  } else if (stepTarget > 0) {
    metaRows.push([
      t("jobs.meta.steps"),
      selectedStep != null && selectedStep > 0
        ? `${selectedStep.toLocaleString()} / ${stepTarget.toLocaleString()}`
        : stepTarget.toLocaleString(),
    ]);
  }

  return (
    <Card className="@container bg-card border-border rounded-md transition-colors h-full">
      {/* gap-2 rather than the job card's gap-2.5: this card carries one row
          more (checkpoint picker + action row) inside the library's fixed
          16.5rem grid row. Everything except the metadata block is shrink-0,
          and the metadata block is the single min-h-0/overflow-hidden child —
          so when a card is unusually tall (aliased run id + three meta rows)
          it is the least load-bearing text that clips, and the action row can
          never be pushed out through the bottom of the card. */}
      <CardContent className="flex h-full flex-col gap-2 p-3">
        <div className="flex shrink-0 items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
              <OriginIcon className="w-3.5 h-3.5" />
              {originLabel}
            </div>
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
            {model.runner === "hf_cloud" && model.hf_job_url ? (
              <Button
                variant="ghost"
                size="icon"
                asChild
                className="h-7 w-7 text-muted-foreground hover:text-foreground"
                aria-label={t("jobs.actions.openHubJob")}
              >
                <a
                  href={model.hf_job_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                </a>
              </Button>
            ) : null}
            <Button
              variant="ghost"
              size="icon"
              onClick={handleDelete}
              className="h-7 w-7 text-muted-foreground hover:text-destructive"
              aria-label={t("jobs.modelCard.deleteAria")}
              title={t("jobs.modelCard.deleteTitle")}
            >
              <Trash2 className="w-3.5 h-3.5" />
            </Button>
          </div>
        </div>

        <div className="shrink-0">
          {/* Middle truncation done by CSS, not by counting characters.

              An imported title is peeled down to its task by the backend
              (utils/naming.derive_imported_title) and then, when two imports
              derive the SAME task, given a trailing " (date time)"
              disambiguator by dedupe_display_names. That suffix is the only
              thing telling two such cards apart, so it must never be the part
              that disappears — which rules out plain `truncate` (eats the tail)
              and, as two rounds of testing showed, also rules out a
              character-budget middleEllipsis: a char budget cannot track the
              real pixel width of a card that is 2-up on one column width and
              3-up on another, so it either double-ellipsed (with `truncate`) or
              silently hard-clipped (without it).

              So: split the structured name and let flexbox do the shortening.
              The base shrinks and takes ONE CSS ellipsis of its own; the suffix
              is shrink-0 and always renders whole. Adapts to any card width —
              "eraser_place… (07-31 12:22)". A trained model's name has no
              such suffix and takes plain truncation. When either half is
              shortened the hover title carries the full identity; when the
              whole name fits, there is nothing to reveal and no title. */}
          <div
            className="flex items-baseline gap-1 text-foreground font-semibold leading-tight"
            {...titleHover}
          >
            {/* The base is the only half that degrades. It has `flex-1`
                (basis 0), so it takes whatever the suffix leaves, down to the
                floor below.

                The suffix never degrades: `shrink-0`, no `truncate`. Letting it
                ellipsize was the earlier arrangement and it produced exactly
                the failure this element exists to prevent — "eraser_plac…
                (2026-0…", where the disambiguator itself is the thing cut off
                and the two colliding cards read identically again. What buys
                the room instead is the year (displayDedupeSuffix): ~5
                characters off a form whose remaining fields are the ones that
                actually separate two same-day runs. No max-width either: the
                base's floor is what bounds this, and a percentage cap would
                truncate a timestamp that still had room to render whole.

                A floor is still needed — with none, the base starves to "era…"
                beside a shrink-0 suffix — but because that suffix can no longer
                give, the floor is now what decides whether the pair OVERFLOWS
                the card, so it is sized against the narrowest interior rather
                than the roomiest. 6ch (~53px) + the 4px gap + a whole
                "(07-31 17:35)" (~98px at the inherited 16px semibold) ≈ 155px:
                inside the ~170px interior of a 2-up card in a 1280px window,
                and far inside the ~239px of CappedGrid's documented 263px card.
                At the retired 10ch the pair needed ~190px and spilled past the
                card edge below a ~1400px viewport. */}
            <span className="min-w-[6ch] flex-1 truncate">{titleBase}</span>
            {titleSuffixText ? (
              <span className="shrink-0">{titleSuffixText}</span>
            ) : null}
          </div>
          {/* When aliased, keep the true identity visible: the run id for
              trained models (imports already show their repo id / path in the
              subtitle below). */}
          {!isImported && model.display_name ? (
            <div
              className="text-[11px] leading-4 text-muted-foreground truncate"
              title={model.id}
            >
              {model.id}
            </div>
          ) : null}
          {/* Which end of the subtitle to keep depends on what the subtitle IS.
              A disk import's subtitle is a filesystem path, whose useful half
              is the tail — rtl flips the ellipsis to the left, and the leading
              LRM keeps the path's first "/" from being bidi-reordered to the
              wrong end. A HUB import's subtitle is a repo id, whose useful half
              is the HEAD (the namespace, which appears nowhere else on the
              card, and the policy token); rtl there hid the namespace and left
              only the timestamp the title already carries. So: rtl for paths
              only. */}
          <div
            className="text-xs text-muted-foreground truncate"
            title={subtitle}
            style={
              isImported && !isHubImport
                ? { direction: "rtl", textAlign: "left" }
                : undefined
            }
          >
            {isImported && !isHubImport ? "‎" + subtitle : subtitle}
          </div>
        </div>

        <div className="min-h-0 overflow-hidden">
          <MetaRows rows={metaRows} />
        </div>

        {/* Bottom-pinned control cluster: the checkpoint row and the action
            row are ONE block (mt-auto on the cluster, tight gap inside), not
            two separately-placed rows. The checkpoint picker chooses what Run /
            Fine-tune / Download act on, so it belongs with them — and pinning
            them together means a sparse card's slack collects between the
            metadata and this cluster (breathing space under the facts) instead
            of opening a hole in the middle of the controls, which is what
            mt-auto on the action row alone produced. */}
        <div className="mt-auto flex shrink-0 flex-col gap-1.5">
          {/* Checkpoint row — which weights the actions below operate on, plus
              the runs behind this model, collapsed into one dropdown. Download
              sits at its right: it acts on the SELECTED checkpoint, so it reads
              as part of this row rather than of the launch actions. Both always
              render — greyed with their reason on hover when the backing record
              has nothing to offer. */}
          <div className="flex items-center gap-1.5">
            <History className="w-3.5 h-3.5 shrink-0 text-muted-foreground" />
            {withHint(
              selectorReason,
              <CheckpointDropdown
                checkpoints={checkpoints}
                selectedRef={selectedRef}
                onChange={(c) => setSelectedRef(c.ref)}
                disabled={!selectorEnabled}
                placeholder={t("jobs.modelCard.checkpointPlaceholder")}
                className="w-36"
                // Same ambiguity as JobCard's resume row, same fix: this list
                // merges a lineage, so "step 2000" can appear once per run and
                // the runs share a display name. It matters more here — the
                // selection routes Run / Fine-tune / Download to the OWNING
                // record, so picking the wrong one silently acts on a
                // different model's weights.
                owners={checkpointOwners(lineageCheckpoints)}
              />,
              "shrink-0",
            )}
            <div className="min-w-0 flex-1" />
            {withHint(
              downloadReason,
              <Button
                size="icon"
                variant="outline"
                onClick={handleDownload}
                disabled={!downloadEnabled}
                className="h-7 w-7 shrink-0 p-0 border-border text-muted-foreground hover:bg-muted"
                aria-label={t("jobs.actions.download")}
                title={downloadEnabled ? t("jobs.actions.download") : undefined}
              >
                <Download className="w-3.5 h-3.5" />
              </Button>,
              "shrink-0",
            )}
          </div>

          {/* Action footer. Run is the model's primary affordance; Fine-tune is
              the other thing you do with weights. Continue / Resume are
              deliberately absent — they belong to a run that stopped short, and
              such a run never appears here (see the component doc). Neither label
              ever collapses to an icon: an unlabelled Fine-tune button is
              unrecognisable, and a card narrow enough to need that is narrow
              enough to wrap instead. */}
          <div className="flex items-center gap-1.5">
            {withHint(
              runReason,
              <Button
                size="sm"
                onClick={handlePlay}
                disabled={!runEnabled}
                className="h-8 w-full gap-1 bg-primary hover:bg-primary/90 text-primary-foreground"
                aria-label={t("jobs.actions.runInferenceModel")}
              >
                <Play className="w-3.5 h-3.5" /> {t("jobs.actions.run")}
              </Button>,
              "min-w-0 flex-1",
            )}
            {withHint(
              finetuneReason,
              <Button
                size="sm"
                variant="outline"
                onClick={handleFinetune}
                disabled={!finetuneEnabled}
                className="h-8 shrink-0 gap-1 border-primary/40 text-primary hover:bg-primary/10"
                // Same words on both, so one key rather than two that could
                // drift apart.
                aria-label={t("jobs.actions.fineTuneHint")}
                title={finetuneEnabled ? t("jobs.actions.fineTuneHint") : undefined}
              >
                <Sparkles className="w-3.5 h-3.5" />
                {t("jobs.actions.fineTune")}
              </Button>,
              "shrink-0",
            )}
          </div>
        </div>
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
                    isImported && model.hf_repo_id
                      ? "jobs.rename.targetHubRepo"
                      : "jobs.rename.targetRun",
                  ),
                }}
                components={[
                  <span key="0" className="font-mono text-muted-foreground">
                    {isImported ? importedSource : model.id}
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
          {renameError && (
            <p className="text-sm text-destructive">{renameError}</p>
          )}
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
    </Card>
  );
};

export default ModelCard;
