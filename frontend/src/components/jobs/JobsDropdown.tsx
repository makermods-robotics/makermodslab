import React from "react";
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
  Square,
  Trash2,
  XCircle,
} from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { runTaskTitle } from "@/lib/modelNames";
import {
  policyTypeDisplayName,
  policyTypeShortLabel,
} from "@/components/training/types";
import {
  HubJob,
  JobRecord,
  isHubJobActive,
  jobDisplayName,
} from "@/lib/jobsApi";

/**
 * One selectable run in the jobs dropdown. Local/cloud runs (`job`) and
 * Hub-only jobs (`hub`) carry genuinely different facts, so they keep their
 * own shapes and render into their own section rather than being flattened
 * into a row of em dashes.
 */
export type JobsEntry =
  | { kind: "job"; key: string; time: number; job: JobRecord }
  | { kind: "hub"; key: string; time: number; job: HubJob };

function relativeTime(ms: number): string {
  if (!ms) return "—";
  const diff = Math.max(0, (Date.now() - ms) / 1000);
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

interface Presentation {
  label: string;
  color: string;
  Icon: React.ComponentType<{ className?: string }>;
  spin?: boolean;
}

const statePresentation: Record<JobRecord["state"], Presentation> = {
  running: { label: "Running", color: "text-ok", Icon: Loader2, spin: true },
  done: { label: "Done", color: "text-muted-foreground", Icon: CheckCircle2 },
  failed: { label: "Failed", color: "text-destructive", Icon: XCircle },
  interrupted: { label: "Stopped", color: "text-warn", Icon: AlertTriangle },
};

const stagePresentation: Record<string, Presentation> = {
  RUNNING: { label: "Running", color: "text-ok", Icon: Loader2, spin: true },
  QUEUED: { label: "Queued", color: "text-warn", Icon: Clock },
  SCHEDULING: { label: "Starting", color: "text-warn", Icon: Clock },
  COMPLETED: {
    label: "Done",
    color: "text-muted-foreground",
    Icon: CheckCircle2,
  },
  FAILED: { label: "Failed", color: "text-destructive", Icon: XCircle },
  // HF API uses "CANCELED" (single L); accept both spellings.
  CANCELED: { label: "Cancelled", color: "text-warn", Icon: AlertTriangle },
  CANCELLED: { label: "Cancelled", color: "text-warn", Icon: AlertTriangle },
};

interface Described {
  /** What the title line RENDERS: a generated run name peeled to its task. */
  name: string;
  /** What the title line MEANS: the untouched name, for the hover title. Equal
   * to `name` whenever nothing was peeled. */
  fullName: string;
  /** The policy the peel took out of `name`, as a chip label — null when
   * nothing was peeled, so a human-named or imported row gains no chip. */
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
function describeEntry(entry: JobsEntry): Described {
  if (entry.kind === "hub") {
    const job = entry.job;
    const stage = job.status?.stage?.toUpperCase() ?? "";
    const present = stagePresentation[stage] ?? {
      label: stage || "Unknown",
      color: "text-muted-foreground",
      Icon: HelpCircle,
    };
    // A Hub-only job is named by its image/space, never by the "{POLICY} · {ds}"
    // shape, and nothing here knows what policy it trains — so no peel and no
    // chip.
    const hubName =
      job.docker_image ?? job.space_id ?? `Job ${job.id.slice(0, 12)}…`;
    return {
      name: hubName,
      fullName: hubName,
      policyLabel: null,
      policyTitle: "",
      present,
      when: relativeTime(entry.time),
      whereLabel: job.flavor ?? "Hub",
      WhereIcon: Globe,
      whereTitle: job.owner
        ? `Hugging Face job · ${job.owner}`
        : "Hugging Face job",
      running: isHubJobActive(job),
      // A HubJob reports a stage, never a step count — there is no progress to
      // render, so the row shows its stage alone rather than a fake bar.
      progress: null,
    };
  }
  const job = entry.job;
  const present = statePresentation[job.state];
  const isRunning = job.state === "running";
  const isCloud = job.runner === "hf_cloud";
  const target = job.config?.steps || job.metrics.total_steps || 0;
  const current = job.metrics.current_step;
  const pct = target > 0 ? Math.min(100, (current / target) * 100) : 0;
  // Peel the generated "{POLICY} · {namespace}/{task}" down to the task, and
  // hand the policy it removed to a chip of its own. A rename, an import or a
  // human-typed name is returned untouched by runTaskTitle, and `peeled` is
  // then false — those rows render exactly as they did before, chip included
  // (i.e. absent).
  const fullName = jobDisplayName(job);
  const name = runTaskTitle(fullName);
  const peeled = name !== fullName;
  const policyType = job.config?.policy_type;
  return {
    name,
    fullName,
    policyLabel: peeled && policyType ? policyTypeShortLabel(policyType) : null,
    policyTitle: peeled && policyType ? policyTypeDisplayName(policyType) : "",
    present,
    when: relativeTime(
      job.ended_at != null ? job.ended_at * 1000 : (job.started_at ?? 0) * 1000,
    ),
    whereLabel: isCloud ? (job.hf_flavor ?? "Cloud") : "Local",
    WhereIcon: isCloud ? Globe : HardDrive,
    whereTitle: isCloud ? "Runs on Hugging Face cloud" : "Runs on this machine",
    running: isRunning,
    progress: isRunning
      ? {
          pct,
          text:
            target > 0
              ? `${current.toLocaleString()} / ${target.toLocaleString()} · ${pct.toFixed(1)}%`
              : "starting…",
          starting: target === 0,
        }
      : null,
  };
}

/** Resume is offered on a run that ended before its target with something to
 * resume from. The row's button takes the newest checkpoint; picking a
 * specific step stays in the selected run's detail card below the dropdown. */
const canResumeEntry = (entry: JobsEntry): boolean =>
  entry.kind === "job" &&
  (entry.job.state === "failed" || entry.job.state === "interrupted") &&
  entry.job.checkpoint_count > 0;

const COL_STATE = "w-[4.75rem] shrink-0";
// The policy the title no longer carries. Always occupies its column — an
// imported or human-named row leaves it empty rather than shifting where/when
// out of alignment with the rows around it.
const COL_POLICY = "w-[4rem] shrink-0";
const COL_WHERE = "w-[4.5rem] shrink-0";
const COL_WHEN = "w-[3.5rem] shrink-0 text-right";

const SectionLabel: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => (
  <div className="px-2 pb-0.5 pt-1.5 text-[10px] uppercase tracking-wide text-muted-foreground">
    {children}
  </div>
);

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
 * Continue/Resume-from-step, Download, delete) lives in the detail card the
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
  const d = describeEntry(entry);
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
            so the exact identity stays one hover away. */}
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
                if (window.confirm("Stop this run?")) onStop(record.id);
              }}
              aria-label="Stop this run"
              title="Stop this run"
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
              aria-label="Resume this run from its last checkpoint"
              title="Resume from the last checkpoint (pick a specific step below)"
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
              aria-label="Open this job on the Hub"
              title="Open on the Hub"
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
                if (
                  window.confirm(
                    "Remove this job from the list? The job record on Hugging Face is unaffected.",
                  )
                )
                  onDismissHub(hub.id);
              }}
              aria-label="Remove job from list"
              title="Remove from list"
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

  const renderGroup = (label: string, group: JobsEntry[]) =>
    group.length === 0 ? null : (
      <div key={label}>
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

  const d = selected ? describeEntry(selected) : null;
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
          aria-label="Select a training run"
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
                Select a run
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
        <div aria-label="Training runs" className="max-h-72 overflow-y-auto">
          {renderGroup("Launched from MakerMods Lab", active.lab)}
          {renderGroup("Other Hub jobs", active.hub)}
          {untracked.length > 0 && untrackedOpen ? (
            <>
              {renderGroup("Untracked · launched from MakerMods Lab", rest.lab)}
              {renderGroup("Untracked · other Hub jobs", rest.hub)}
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
              ? "Hide untracked"
              : `Untracked (${untracked.length})`}
          </button>
        ) : null}
      </PopoverContent>
    </Popover>
  );
};

export default JobsDropdown;
