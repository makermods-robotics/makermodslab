import React, { useEffect, useMemo, useState } from "react";
import { Check, Globe, HardDrive, Lock } from "lucide-react";
import { Button } from "@/components/ui/button";
import LibraryToolbar from "@/components/library/LibraryToolbar";
import CappedGrid, { GRID_MIN_H } from "@/components/library/CappedGrid";
import MetaRows from "@/components/library/MetaRows";
import { cn } from "@/lib/utils";
import { useApi } from "@/contexts/ApiContext";
import { DatasetInfo, DatasetItem, getDatasetInfo } from "@/lib/replayApi";
import { formatBytes, formatCount, formatDuration } from "@/lib/datasetFormat";
import { useHubVideoFilter } from "@/hooks/useHubVideoFilter";

/** Where a dataset lives: local cache, the Hub, or both. Styled like the job
 * card's status chip (icon + muted bold text, no pill) so the dataset, job,
 * and model cards read as one family. */
const SourceBadge: React.FC<{ source: DatasetItem["source"] }> = ({
  source,
}) => (
  <span className="flex shrink-0 items-center gap-1.5 text-xs font-semibold text-muted-foreground">
    {source !== "hub" && <HardDrive className="h-3.5 w-3.5" />}
    {source !== "local" && <Globe className="h-3.5 w-3.5" />}
    {source === "local" ? "Local" : source === "hub" ? "Hub" : "Local · Hub"}
  </span>
);

/** Session-lifetime cache of /datasets/info summaries so collapsing and
 * re-expanding the library (which unmounts the cards) doesn't refetch every
 * dataset. "error" marks an unreadable local copy so it isn't retried on every
 * mount. Cleared by the library's refresh button. */
const infoCache = new Map<string, DatasetInfo | "error">();

/** Drop every cached /datasets/info summary — call before a refresh so cards
 * refetch (the Collect panel's refresh button and post-merge hook). */
export const clearDatasetInfoCache = () => infoCache.clear();

/** Metadata lines for one library card, lazily fetched. Hub-only datasets are
 * never fetched (a remote meta.json read per card — the same skip rule as the
 * Train panel's rows); their card says where the dataset lives instead. */
const DatasetCardDetails: React.FC<{ item: DatasetItem }> = ({ item }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [info, setInfo] = useState<DatasetInfo | "error" | null>(
    () => infoCache.get(item.repo_id) ?? null,
  );

  useEffect(() => {
    if (item.source === "hub" || infoCache.has(item.repo_id)) return;
    let cancelled = false;
    getDatasetInfo(baseUrl, fetchWithHeaders, item.repo_id)
      .then((data) => {
        infoCache.set(item.repo_id, data);
        if (!cancelled) setInfo(data);
      })
      .catch(() => {
        infoCache.set(item.repo_id, "error");
        if (!cancelled) setInfo("error");
      });
    return () => {
      cancelled = true;
    };
  }, [item.repo_id, item.source, baseUrl, fetchWithHeaders]);

  if (item.source === "hub") {
    return (
      <p className="text-[11px] text-muted-foreground">
        On the Hub — not downloaded locally.
      </p>
    );
  }

  if (info === "error") {
    return (
      <p className="text-[11px] text-muted-foreground">
        Couldn't read this dataset's details.
      </p>
    );
  }

  if (info === null) {
    return (
      <div
        className="animate-pulse space-y-1.5"
        aria-label="Loading dataset details"
      >
        <div className="h-3 w-2/3 rounded bg-muted" />
        <div className="h-3 w-1/2 rounded bg-muted" />
      </div>
    );
  }

  const duration = formatDuration(info.total_frames, info.fps);
  const rows: Array<[string, string]> = [];
  if (info.cameras.length > 0) rows.push(["Cameras", info.cameras.join(", ")]);
  if (info.robot_type) rows.push(["Robot", info.robot_type]);
  if (info.tasks.length === 1) {
    rows.push(["Task", info.tasks[0].task]);
  } else if (info.tasks.length > 1) {
    rows.push(["Tasks", `${info.tasks.length} tasks`]);
  }
  if (info.size_bytes != null) rows.push(["Size", formatBytes(info.size_bytes)]);

  return (
    <div className="space-y-1.5">
      {/* Muted one-liner right under the bold name — the dataset's "ended 3h
          ago": its volume at a glance. */}
      <p className="text-xs text-muted-foreground">
        {info.total_episodes} episode{info.total_episodes === 1 ? "" : "s"}
        {" · "}
        {formatCount(info.total_frames)} frames
        {duration ? ` · ${duration}` : ""}
      </p>
      <MetaRows rows={rows} />
    </div>
  );
};

/** One selectable dataset card, in the job/model cards' design language:
 * status-style source chip top-left, selection check in the top-right action
 * slot, bold name, muted stats subtitle, then the detail rows and a footer
 * Select button (the card's call-to-action; the whole card stays clickable).
 * Stretches to its grid row's height so a row of cards lines up evenly. */
const DatasetCard: React.FC<{
  item: DatasetItem;
  selected: boolean;
  onSelect: () => void;
  /** Opens the episode viewer for this dataset. The card itself opens the
   * viewer on click; selecting is only done via the footer Select button.
   * Optional: only wired up where the viewer dialog is actually rendered —
   * falls back to select-on-click if absent. */
  onView?: (item: DatasetItem) => void;
}> = ({ item, selected, onSelect, onView }) => (
  <div
    onClick={() => (onView ? onView(item) : onSelect())}
    className={cn(
      "flex w-full cursor-pointer flex-col gap-2 overflow-hidden rounded-md border bg-card p-3 text-left transition-colors",
      selected
        ? "border-ring bg-primary/5"
        : "border-border hover:border-muted-foreground/40",
    )}
  >
    <div className="flex w-full items-center justify-between gap-2">
      <div className="flex min-w-0 items-center gap-2">
        <SourceBadge source={item.source} />
        {item.private && (
          <span
            className="flex shrink-0 items-center gap-1 text-[11px] font-medium text-muted-foreground"
            title="Private on the Hub"
          >
            <Lock className="h-3 w-3" />
            private
          </span>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-0.5">
        <Check
          className={cn(
            "h-4 w-4 shrink-0 text-primary",
            selected ? "opacity-100" : "opacity-0",
          )}
        />
      </div>
    </div>
    <div className="w-full">
      <div
        className="truncate text-sm font-semibold text-foreground"
        title={item.repo_id}
      >
        {item.repo_id.split("/").pop()}
      </div>
      <div
        className="truncate text-[11px] text-muted-foreground"
        title={item.repo_id}
      >
        {item.repo_id}
      </div>
    </div>
    <DatasetCardDetails item={item} />
    <div className="mt-auto flex items-center pt-1">
      <Button
        size="sm"
        variant={selected ? "outline" : "default"}
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
        aria-pressed={selected}
        className={cn(
          "h-8 gap-1",
          selected
            ? "border-ring text-primary hover:bg-primary/10"
            : "bg-primary text-primary-foreground hover:bg-primary/90",
        )}
      >
        {selected ? (
          <>
            <Check className="h-3.5 w-3.5" /> Selected
          </>
        ) : (
          "Select"
        )}
      </Button>
    </div>
  </div>
);

/** Where in the library to look: everything, local copies, or Hub uploads. */
type LibraryFilter = "all" | "local" | "hub";

const FILTERS: Array<{ key: LibraryFilter; label: string }> = [
  { key: "all", label: "All" },
  { key: "local", label: "Local" },
  { key: "hub", label: "Hub" },
];

/** The library body: search + location filter over a three-up grid of dataset
 * cards. Clicking a card opens its episode viewer; the footer Select button
 * is the only way to select/deselect it (feeding the Collect panel's header
 * chip + the Train panel via the shared useSelectedDataset store). */
export const DatasetLibraryList: React.FC<{
  datasets: DatasetItem[];
  loading: boolean;
  selectedRepoId: string | null;
  onSelect: (item: DatasetItem) => void;
  /** Opens the episode viewer dialog for a dataset; omit where the caller
   * doesn't render that dialog. */
  onView?: (item: DatasetItem) => void;
}> = ({ datasets, loading, selectedRepoId, onSelect, onView }) => {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<LibraryFilter>("all");
  // Hides a Hub-only row once it's confirmed to have no video — this surface
  // wires onView (opens the episode viewer), so a row without video would
  // just open to an empty state. See useHubVideoFilter for why the Train
  // picker (which doesn't use this component) must NOT do the same.
  const videoFilteredDatasets = useHubVideoFilter(datasets);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return videoFilteredDatasets.filter((d) => {
      if (filter === "local" && d.source === "hub") return false;
      if (filter === "hub" && d.source === "local") return false;
      return q === "" || d.repo_id.toLowerCase().includes(q);
    });
  }, [videoFilteredDatasets, query, filter]);

  if (loading && videoFilteredDatasets.length === 0) {
    return (
      <div
        className={cn(
          "animate-pulse space-y-2 rounded-md border border-border p-3",
          GRID_MIN_H,
        )}
        aria-label="Loading datasets"
      >
        <div className="h-4 w-3/4 rounded bg-muted" />
        <div className="h-4 w-1/2 rounded bg-muted" />
        <div className="h-4 w-2/3 rounded bg-muted" />
      </div>
    );
  }

  if (videoFilteredDatasets.length === 0) {
    return (
      <div
        className={cn(
          "flex items-center justify-center rounded-md border border-dashed border-border px-4 py-6 text-center text-sm text-muted-foreground",
          GRID_MIN_H,
        )}
      >
        No datasets yet. Record your first one above.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <LibraryToolbar<LibraryFilter>
        query={query}
        onQueryChange={setQuery}
        searchPlaceholder="Search datasets"
        filters={FILTERS}
        filter={filter}
        onFilterChange={setFilter}
      />

      {visible.length === 0 ? (
        <p
          className={cn(
            "flex items-center justify-center px-1 py-4 text-center text-sm text-muted-foreground",
            GRID_MIN_H,
          )}
        >
          No datasets match.
        </p>
      ) : (
        // Two rows by default; anything past that stays behind Show all
        // (replaces the old scrolling max-height frame).
        <CappedGrid
          items={visible.map((item) => (
            <DatasetCard
              key={item.repo_id}
              item={item}
              selected={item.repo_id === selectedRepoId}
              onSelect={() => onSelect(item)}
              onView={onView}
            />
          ))}
        />
      )}
    </div>
  );
};

export default DatasetLibraryList;
