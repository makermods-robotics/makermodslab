import React from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  Clock,
  ExternalLink,
  FastForward,
  Globe,
  HardDrive,
  HelpCircle,
  Loader2,
  Router,
  Square,
  Trash2,
  XCircle,
} from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useLanguage } from "@/contexts/LanguageContext";
import { isCaselessScript } from "@/i18n/config";
import { cn } from "@/lib/utils";
import { runTaskTitle } from "@/lib/modelNames";
import {
  policyTypeDisplayName,
  policyTypeShortLabel,
} from "@/components/training/types";
import {
  HubJob,
  JOB_STATE_LABELS,
  JobRecord,
  isHubJobActive,
  jobDisplayName,
} from "@/lib/jobsApi";
import { isResumableLeaf } from "./resumeSeed";

/**
 * One selectable run in the jobs dropdown. Local/cloud runs (`job`) and
 * Hub-only jobs (`hub`) carry genuinely different facts, so they keep their
 * own shapes and render into their own section rather than being flattened
 * into a row of em dashes.
 */
export type JobsEntry =
  | {
      kind: "job";
      key: string;
      time: number;
      job: JobRecord;
      /** Checkpoints reachable from this row's run — its own plus those of the
       * runs it continues. A row is a whole CHAIN (the list shows one row per
       * leaf), and a resume takes the newest checkpoint on that LINEAGE, which
       * may be an ancestor's, so the run's own `checkpoint_count` is the wrong
       * number to gate Resume on: the commonest resumable shape is a tip that
       * died before saving anything, whose checkpoints are all inherited.
       * Counted by
       * JobsDataContext, which holds the ancestor records — and which files a
       * chain whose ancestors are still being backfilled as active, so a row is
       * never hidden away in the UNTRACKED fold on the strength of a count that
       * hasn't settled. */
      chainCheckpointCount: number;
    }
  | { kind: "hub"; key: string; time: number; job: HubJob };

function relativeTime(ms: number): string {
  if (!ms) return "—";
  const diff = Math.max(0, (Date.now() - ms) / 1000);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

/** The `t` a component gets from `useTranslation()`. Passed into the pure
 * describe helpers below, which run outside any component. */
type Translate = TFunction;

interface Presentation {
  label: string;
  color: string;
  Icon: React.ComponentType<{ className?: string }>;
  spin?: boolean;
}

/** State/stage → presentation, keyed by the WIRE value (never translated).
 * These maps hold translation KEYS: they are built at import time, so a
 * resolved word here would freeze whichever language loaded first. Colour, icon
 * and spin are not copy and stay as they are. */
const statePresentation = {
  running: {
    labelKey: JOB_STATE_LABELS.running,
    color: "text-ok",
    Icon: Loader2,
    spin: true,
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

// SCHEDULING reads "Starting" here and "Scheduling" on the Hub job card — two
// deliberate wordings, so they point at two keys.
const stagePresentation = {
  RUNNING: {
    labelKey: "jobs.stage.running",
    color: "text-ok",
    Icon: Loader2,
    spin: true,
  },
  QUEUED: { labelKey: "jobs.stage.queued", color: "text-warn", Icon: Clock },
  SCHEDULING: {
    labelKey: "jobs.stage.starting",
    color: "text-warn",
    Icon: Clock,
  },
  COMPLETED: {
    labelKey: "jobs.stage.completed",
    color: "text-muted-foreground",
    Icon: CheckCircle2,
  },
  FAILED: {
    labelKey: "jobs.stage.failed",
    color: "text-destructive",
    Icon: XCircle,
  },
  // HF API uses "CANCELED" (single L); accept both spellings.
  CANCELED: {
    labelKey: "jobs.stage.cancelled",
    color: "text-warn",
    Icon: AlertTriangle,
  },
  CANCELLED: {
    labelKey: "jobs.stage.cancelled",
    color: "text-warn",
    Icon: AlertTriangle,
  },
} as const;

interface Described {
  /** What the title line RENDERS: a generated run name peeled to its task. */
  name: string;
  /** What the title line MEANS: the untouched name, for the hover title. Equal
   * to `name` whenever nothing was peeled. */
  fullName: string;
  /** The run's number, rendered beside the name. Neither `name` nor `fullName`
   * identifies a run on a resume chain — every run on one shares them, because
   * a continuation continues the same model. 0 ⇒ not assigned (a record the
   * backend hasn't backfilled yet); render nothing rather than "#0". */
  number: number;
  /** The run's policy, as a chip label — read off the record's own
   * `config.policy_type`, never inferred from the name. Null only when the
   * record states no policy (a Hub-only job). */
  policyLabel: string | null;
  /** Hover text for that chip: the policy's full display name. */
  policyTitle: string;
  present: Presentation;
  when: string;
  whereLabel: string;
  WhereIcon: React.ComponentType<{ className?: string }>;
  whereTitle: string;
  running: boolean;
  /** Only set while running: live step counters for the trigger and the row's
   * progress bar. `starting` covers the window before the first step lands. */
  progress: { pct: number; text: string; starting: boolean } | null;
}

/** Normalize a local/cloud run or a Hub-only job into the one shape the
 * trigger and the rows both render, so a run reads the same in either place. */
function describeEntry(entry: JobsEntry, t: Translate): Described {
  if (entry.kind === "hub") {
    const job = entry.job;
    const stage = job.status?.stage?.toUpperCase() ?? "";
    const known = stagePresentation[stage];
    // An unmapped stage keeps the RAW Hub string — it is data, and showing it
    // beats showing nothing; only the last-resort word is translated.
    const present: Presentation = known
      ? { ...known, label: t(known.labelKey) }
      : {
          label: stage || t("jobs.stage.unknown"),
          color: "text-muted-foreground",
          Icon: HelpCircle,
        };
    // A Hub-only job is named by its image/space, never by the "{POLICY} · {ds}"
    // shape, and nothing here knows what policy it trains — so no peel and no
    // chip.
    const hubName =
      job.docker_image ??
      job.space_id ??
      t("jobs.hubJob.fallbackTitle", { id: job.id.slice(0, 12) });
    return {
      name: hubName,
      fullName: hubName,
      // A Hub-only job has no local record, so it has no run number — the
      // sequence numbers this registry's own runs. 0 renders nothing.
      number: 0,
      policyLabel: null,
      policyTitle: "",
      present,
      when: relativeTime(entry.time),
      // The flavor is the Hub's own hardware name — data.
      whereLabel: job.flavor ?? t("jobs.location.hub"),
      WhereIcon: Globe,
      whereTitle: job.owner
        ? t("jobs.jobsDropdown.hubJobTitleWithOwner", { owner: job.owner })
        : t("jobs.jobsDropdown.hubJobTitle"),
      running: isHubJobActive(job),
      // A HubJob reports a stage, never a step count — there is no progress to
      // render, so the row shows its stage alone rather than a fake bar.
      progress: null,
    };
  }
  const job = entry.job;
  const state = statePresentation[job.state];
  const present: Presentation = { ...state, label: t(state.labelKey) };
  const isRunning = job.state === "running";
  const isCloud = job.runner === "hf_cloud";
  const isNode = job.runner === "lan_node";
  const target = job.config?.steps || job.metrics.total_steps || 0;
  const current = job.metrics.current_step;
  const pct = target > 0 ? Math.min(100, (current / target) * 100) : 0;
  // Peel the generated "{POLICY} · {namespace}/{task}" down to the task. A
  // rename, an import or a human-typed name is returned untouched.
  const fullName = jobDisplayName(job);
  const name = runTaskTitle(fullName);
  // The policy chip reads the record's OWN config, not the peel: a run names
  // its policy in `config.policy_type` whatever it is called, so a human-named
  // run ({name}_{timestamp}, the only shape new runs have) and an import state
  // their policy here exactly like a generated name does. Gating this on "did
  // the title lose a policy token?" left every named run's policy column blank
  // while the detail card below it (JobCard's Policy meta row, same field) had
  // the value all along.
  const policyType = job.config?.policy_type;
  return {
    name,
    fullName,
    // The run number, rendered beside the name because the name does not
    // identify a run on its own: every run on a resume chain carries the same
    // one. 0 ⇒ a record the backend hasn't backfilled; render nothing.
    number: job.job_number,
    policyLabel: policyType ? policyTypeShortLabel(policyType) : null,
    policyTitle: policyType ? policyTypeDisplayName(policyType) : "",
    present,
    when: relativeTime(
      job.ended_at != null ? job.ended_at * 1000 : (job.started_at ?? 0) * 1000,
    ),
    // The flavor is the Hub's own hardware name, and a node's short instance
    // id is the run's routing key — both data. (This helper is pure — no
    // registry lookup — so the node column shows the id; the detail card's
    // chip resolves the node's display name.)
    whereLabel: isCloud
      ? (job.hf_flavor ?? t("jobs.location.cloud"))
      : isNode
        ? (job.node_instance_id?.slice(0, 8) ?? t("jobs.location.node"))
        : t("jobs.location.local"),
    WhereIcon: isCloud ? Globe : isNode ? Router : HardDrive,
    whereTitle: isCloud
      ? t("jobs.location.cloudTitle")
      : isNode
        ? t("jobs.location.nodeTitle")
        : t("jobs.location.localTitle"),
    running: isRunning,
    progress: isRunning
      ? {
          pct,
          text:
            // Numbers stay exactly as they were formatted; only the
            // no-numbers-yet word is translated.
            target > 0
              ? `${current.toLocaleString()} / ${target.toLocaleString()} · ${pct.toFixed(1)}%`
              : t("jobs.progress.starting"),
          starting: target === 0,
        }
      : null,
  };
}

/** Resume is offered on a chain whose tip ended before its target with
 * something, anywhere in the chain, to continue from — the tip continues from
 * the newest checkpoint on its lineage, its own or an ancestor's. The state
 * half is the ONE
 * shared leaf rule (`isResumableLeaf`), so this row's button and the detail
 * card's Resume can't disagree; the checkpoint half is deliberately the cheap
 * chain-wide count, because the exact per-step answer needs a fetch. The click
 * resolves the real list and says so if it comes back empty.
 *
 * ONE verb across both levels of control, and by now the same action with
 * nothing added: Continue is not step-selectable on either (user decision
 * 2026-08-10), so this row and the detail card below both take the newest
 * resumable checkpoint. The card keeps a checkpoint dropdown, but it drives
 * only Run / Fine-tune / Download — picking an older checkpoint is a real
 * choice for those and not for a continuation. */
const canResumeEntry = (entry: JobsEntry): boolean =>
  entry.kind === "job" &&
  isResumableLeaf(entry.job) &&
  entry.chainCheckpointCount > 0;

const COL_STATE = "w-[4.75rem] shrink-0";
// The run's policy. Always occupies its column — a row whose record names no
// policy (a Hub-only job) leaves it empty rather than shifting where/when out
// of alignment with the rows around it.
const COL_POLICY = "w-[4rem] shrink-0";
const COL_WHERE = "w-[4.5rem] shrink-0";
const COL_WHEN = "w-[3.5rem] shrink-0 text-right";

/** `uppercase` is a no-op on a caseless script but the letter-spacing beside it
 * is not — left on, a Chinese label renders visibly over-spaced. Both are
 * dropped together there. */
const SectionLabel: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const { language } = useLanguage();
  return (
    <div
      className={cn(
        "px-2 pb-0.5 pt-1.5 text-[10px] text-muted-foreground",
        isCaselessScript(language) ? "" : "uppercase tracking-wide",
      )}
    >
      {children}
    </div>
  );
};

interface RowProps {
  entry: JobsEntry;
  selected: boolean;
  onSelect: (entry: JobsEntry) => void;
  onStop: (id: string) => void;
  onResume: (job: JobRecord) => void;
  onDismissHub: (id: string) => void;
  resuming: boolean;
}

/**
 * One run: state · task · policy · where · when, plus the row-level primary
 * actions. The name column carries the TASK alone (the policy moved to its own
 * chip, the dataset namespace is dropped) so rows of one policy+namespace stop
 * reading identically once the column truncates.
 * Everything else about the run (rename, monitor, checkpoint picker, Run,
 * Resume-from-step, Download, delete) lives in the detail card the
 * selection drives.
 */
const JobsRow: React.FC<RowProps> = ({
  entry,
  selected,
  onSelect,
  onStop,
  onResume,
  onDismissHub,
  resuming,
}) => {
  const { t } = useTranslation();
  const d = describeEntry(entry, t);
  const Icon = d.present.Icon;
  // Narrow once so the row's actions read off the concrete record.
  const record = entry.kind === "job" ? entry.job : null;
  const hub = entry.kind === "hub" ? entry.job : null;
  const showStop = record?.state === "running";
  const showResume = canResumeEntry(entry);
  const showDismiss = hub != null && !isHubJobActive(hub);

  return (
    <div className={cn("rounded-md", selected && "bg-muted")}>
      {/* A row is a toggle, not a listbox option: the section labels mean the
          rows aren't direct children of one list, so `button`/`aria-pressed`
          describes it honestly. */}
      <div
        role="button"
        aria-pressed={selected}
        tabIndex={0}
        onClick={() => onSelect(entry)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onSelect(entry);
          }
        }}
        className="flex h-7 cursor-pointer items-center gap-1.5 rounded-md px-2 text-[11px] hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
      >
        <span
          className={cn(
            COL_STATE,
            "flex items-center gap-1 font-medium",
            d.present.color,
          )}
        >
          <Icon
            className={cn("h-3 w-3 shrink-0", d.present.spin && "animate-spin")}
          />
          <span className="truncate">{d.present.label}</span>
        </span>
        {/* The hover title is the FULL name — the peel is a display shortening,
            so the exact identity stays one hover away. The number sits outside
            that span so truncation can never eat it: it is the shortest thing
            in the row and the only one that identifies the run. */}
        {d.number > 0 ? (
          <span className="shrink-0 font-mono text-muted-foreground">
            #{d.number}
          </span>
        ) : null}
        <span
          className="min-w-0 flex-1 truncate text-foreground"
          title={d.fullName}
        >
          {d.name}
        </span>
        <span
          className={cn(COL_POLICY, "truncate text-muted-foreground")}
          title={d.policyTitle || undefined}
        >
          {d.policyLabel}
        </span>
        <span
          className={cn(
            COL_WHERE,
            "flex items-center gap-1 text-muted-foreground",
          )}
          title={d.whereTitle}
        >
          <d.WhereIcon className="h-3 w-3 shrink-0" />
          <span className="truncate">{d.whereLabel}</span>
        </span>
        <span className={cn(COL_WHEN, "text-muted-foreground")}>{d.when}</span>
        <span className="flex w-10 shrink-0 items-center justify-end gap-0.5">
          {showStop && record ? (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                // English on purpose: a native confirm's OK/Cancel come from
                // the BROWSER's locale, so a translated question over English
                // buttons reads worse than an English one. Replacing it with an
                // AlertDialog is a separate UX change.
                if (window.confirm("Stop this run?")) onStop(record.id);
              }}
              // Identical words on the accessible name and the hover text, so
              // one key rather than two that could drift apart.
              aria-label={t("jobs.jobsDropdown.stopAria")}
              title={t("jobs.jobsDropdown.stopAria")}
              className="flex h-5 w-5 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <Square className="h-3 w-3" />
            </button>
          ) : null}
          {showResume && record ? (
            <button
              type="button"
              disabled={resuming}
              onClick={(e) => {
                e.stopPropagation();
                onResume(record);
              }}
              aria-label={t("jobs.jobsDropdown.resumeAria")}
              title={t("jobs.jobsDropdown.resumeAria")}
              className="flex h-5 w-5 items-center justify-center rounded text-info transition-colors hover:bg-info/10 disabled:opacity-50"
            >
              {resuming ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <FastForward className="h-3 w-3" />
              )}
            </button>
          ) : null}
          {hub ? (
            <a
              href={hub.url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              aria-label={t("jobs.jobsDropdown.openHubAria")}
              title={t("jobs.jobsDropdown.openHubTitle")}
              className="flex h-5 w-5 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <ExternalLink className="h-3 w-3" />
            </a>
          ) : null}
          {showDismiss && hub ? (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                // English on purpose — see the Stop confirm above.
                if (
                  window.confirm(
                    "Remove this job from the list? The job record on Hugging Face is unaffected.",
                  )
                )
                  onDismissHub(hub.id);
              }}
              aria-label={t("jobs.hubJob.removeAria")}
              title={t("jobs.hubJob.removeTitle")}
              className="flex h-5 w-5 items-center justify-center rounded text-muted-foreground transition-colors hover:bg-muted hover:text-destructive"
            >
              <Trash2 className="h-3 w-3" />
            </button>
          ) : null}
        </span>
      </div>
      {/* Live progress stays visible in the list for a running row — the one
          thing a collapsed grid would otherwise lose. */}
      {d.progress ? (
        <div className="mx-2 mb-1 h-1 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full bg-info transition-[width] duration-500"
            style={{ width: `${d.progress.pct}%` }}
          />
        </div>
      ) : null}
    </div>
  );
};

interface JobsDropdownProps {
  /** Active runs (running, or finished with runnable checkpoints), newest-first. */
  entries: JobsEntry[];
  /** Inactive leftovers, newest-first — folded behind the list's footer. */
  untracked: JobsEntry[];
  /** The run whose detail card is shown beneath the dropdown. */
  selectedKey: string | null;
  onSelect: (entry: JobsEntry) => void;
  onStop: (id: string) => void;
  onResume: (job: JobRecord) => void;
  onDismissHub: (id: string) => void;
  resumingId: string | null;
  /** Shown on the trigger when nothing matches the search/filter. */
  emptyMessage: string;
}

/**
 * The Train panel's run picker: one compact row per run instead of a grid of
 * cards. The collapsed trigger shows the most relevant run — a live one beats
 * the latest finished one — with its state and, while running, its step
 * counter and a progress bar, so a training in flight is readable without
 * opening anything.
 *
 * Rows are grouped by who launched the run (PR #26's split, kept to a single
 * label line): runs this install has a record of, then Hub jobs nothing here
 * tracks. Within a group they run newest-first.
 */
const JobsDropdown: React.FC<JobsDropdownProps> = ({
  entries,
  untracked,
  selectedKey,
  onSelect,
  onStop,
  onResume,
  onDismissHub,
  resumingId,
  emptyMessage,
}) => {
  const { t } = useTranslation();
  const [open, setOpen] = React.useState(false);
  const [untrackedOpen, setUntrackedOpen] = React.useState(false);

  const all = React.useMemo(
    () => [...entries, ...untracked],
    [entries, untracked],
  );
  const selected = all.find((e) => e.key === selectedKey) ?? null;
  const isEmpty = all.length === 0;

  const select = (entry: JobsEntry) => {
    onSelect(entry);
    setOpen(false);
  };

  // `id` is the React key (stable across a language switch); `label` is the
  // rendered copy.
  const renderGroup = (id: string, label: string, group: JobsEntry[]) =>
    group.length === 0 ? null : (
      <div key={id}>
        <SectionLabel>{label}</SectionLabel>
        {group.map((entry) => (
          <JobsRow
            key={entry.key}
            entry={entry}
            selected={entry.key === selectedKey}
            onSelect={select}
            onStop={onStop}
            onResume={onResume}
            onDismissHub={onDismissHub}
            resuming={
              entry.kind === "job" && resumingId === entry.job.id
            }
          />
        ))}
      </div>
    );

  const groupsFor = (list: JobsEntry[]) => ({
    lab: list.filter((e) => e.kind === "job"),
    hub: list.filter((e) => e.kind === "hub"),
  });
  const active = groupsFor(entries);
  const rest = groupsFor(untracked);

  const d = selected ? describeEntry(selected, t) : null;
  const TriggerIcon = d?.present.Icon;

  if (isEmpty) {
    return (
      <div className="flex h-9 items-center justify-center rounded-md border border-dashed border-border px-3 text-xs text-muted-foreground">
        {emptyMessage}
      </div>
    );
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={t("jobs.jobsDropdown.triggerAria")}
          className="relative w-full overflow-hidden rounded-md border border-border bg-background px-2 py-1.5 text-left transition-colors hover:border-ring/50 hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          <div className="flex items-center gap-2">
            {d && TriggerIcon ? (
              <>
                <span
                  className={cn(
                    "flex shrink-0 items-center gap-1 text-xs font-semibold",
                    d.present.color,
                  )}
                >
                  <TriggerIcon
                    className={cn(
                      "h-3.5 w-3.5",
                      d.present.spin && "animate-spin",
                    )}
                  />
                  {d.present.label}
                </span>
                {d.number > 0 ? (
                  <span className="shrink-0 font-mono text-sm text-muted-foreground">
                    #{d.number}
                  </span>
                ) : null}
                <span
                  className="min-w-0 flex-1 truncate text-sm font-medium text-foreground"
                  title={d.fullName}
                >
                  {d.name}
                </span>
                {/* Running runs trade "when" for the live step counter — the
                    collapsed trigger is where a training in flight has to be
                    readable. */}
                <span className="shrink-0 whitespace-nowrap text-[11px] tabular-nums text-muted-foreground">
                  {d.progress ? d.progress.text : d.when}
                </span>
                <span
                  className="flex shrink-0 items-center gap-1 text-[11px] text-muted-foreground"
                  title={d.whereTitle}
                >
                  <d.WhereIcon className="h-3 w-3" />
                  {d.whereLabel}
                </span>
              </>
            ) : (
              <span className="min-w-0 flex-1 truncate text-sm text-muted-foreground">
                {t("jobs.jobsDropdown.placeholder")}
              </span>
            )}
            <ChevronDown
              className={cn(
                "h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform",
                open && "rotate-180",
              )}
            />
          </div>
          {d?.progress ? (
            <div className="absolute inset-x-0 bottom-0 h-0.5 bg-muted">
              <div
                className="h-full bg-info transition-[width] duration-500"
                style={{ width: `${d.progress.pct}%` }}
              />
            </div>
          ) : null}
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        className="w-[var(--radix-popover-trigger-width)] max-w-none p-1"
      >
        <div
          aria-label={t("jobs.jobsDropdown.listAria")}
          className="max-h-72 overflow-y-auto"
        >
          {renderGroup("lab", t("jobs.jobsDropdown.groups.lab"), active.lab)}
          {renderGroup("hub", t("jobs.jobsDropdown.groups.hub"), active.hub)}
          {untracked.length > 0 && untrackedOpen ? (
            <>
              {renderGroup(
                "untracked-lab",
                t("jobs.jobsDropdown.groups.untrackedLab"),
                rest.lab,
              )}
              {renderGroup(
                "untracked-hub",
                t("jobs.jobsDropdown.groups.untrackedHub"),
                rest.hub,
              )}
            </>
          ) : null}
        </div>
        {/* Inactive leftovers stay one click away instead of padding the list
            — same fold the card grid kept, moved inside the dropdown. */}
        {untracked.length > 0 ? (
          <button
            type="button"
            onClick={() => setUntrackedOpen((v) => !v)}
            className="mt-1 flex h-7 w-full items-center justify-center gap-1 rounded-md border border-dashed border-border text-[11px] font-medium text-muted-foreground transition-colors hover:border-muted-foreground/40 hover:text-foreground"
          >
            <ChevronDown
              className={cn(
                "h-3.5 w-3.5 transition-transform",
                untrackedOpen && "rotate-180",
              )}
            />
            {untrackedOpen
              ? t("jobs.jobsDropdown.hideUntracked")
              : // A plain row count, passed as `total` rather than `count` so
                // i18next does not read it as a plural selector.
                t("jobs.jobsDropdown.untracked", {
                  total: untracked.length,
                })}
          </button>
        ) : null}
      </PopoverContent>
    </Popover>
  );
};

export default JobsDropdown;
