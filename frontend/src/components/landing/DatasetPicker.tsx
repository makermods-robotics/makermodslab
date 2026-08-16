import React, { useMemo, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Plus, Trash2 } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { DatasetItem } from "@/lib/replayApi";
import { sortDatasets } from "@/lib/sortDatasets";
import { HUB_REPO_ID_RE } from "@/lib/repoId";
import { useHfAuth } from "@/contexts/HfAuthContext";

interface DatasetPickerProps {
  datasets: DatasetItem[];
  loading: boolean;
  onPickExisting: (item: DatasetItem) => void;
  /** Per-row trash affordance. Invoked with the row's item; the parent routes
   * it through the shared delete confirm dialog (resolveDeleteAction decides
   * the semantics: local delete / local-copy removal / unpin / hide). The
   * picker closes so the Landing-scoped dialog is visible. */
  onDeleteItem?: (item: DatasetItem) => void;
  /** Omit the in-popover search box — for callers that already sit next to
   * their own bigger search field (e.g. Train's dataset picker), where a
   * second search input would just duplicate it. The list still shows every
   * dataset, unfiltered. */
  hideSearch?: boolean;
  /** Offer a typed, well-formed `org/name` that ISN'T in the listing as a
   * public Hub dataset — one extra row, "Use org/name from the Hub".
   *
   * Opt-in, because it is a capability and not a decoration: the caller has to
   * be somewhere that can act on an arbitrary public repo (Train fetches it on
   * demand and pins it). Left out, the picker only selects what it lists, as
   * it always has. */
  onPickHubId?: (repoId: string) => void;
  children: React.ReactNode;
}

/**
 * Search-only dataset selector. The input filters the existing Local /
 * Hugging Face lists — it does not create new names, and creation lives in the
 * "Add dataset" menu on the Landing page (Record / Add from Hugging Face /
 * Import from disk).
 *
 * The one thing it can do beyond selecting what it lists is opt-in: pass
 * `onPickHubId` and a typed, unlisted `org/name` is offered as a public Hub
 * dataset (see below). Train uses it; nothing else does.
 */
const DatasetPicker: React.FC<DatasetPickerProps> = ({
  datasets,
  loading,
  onPickExisting,
  onDeleteItem,
  hideSearch = false,
  onPickHubId,
  children,
}) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  // Namespace-first alphabetical ordering: datasets under the logged-in HF
  // account's namespace float to the top of each section. Falls back to plain
  // alphabetical when not authenticated / still loading.
  const { auth } = useHfAuth();
  const username = auth.status === "authenticated" ? auth.username : null;

  // Strict partition by Hub status (user-decided): Local = not yet on the
  // Hub; Hugging Face = on the Hub, whether or not a local copy also exists.
  // Clearing the local cache of a "both" dataset lives in the "Manage cached
  // datasets" dialog, not inline here.
  const localDatasets = useMemo(
    () => sortDatasets(datasets.filter((d) => d.source === "local"), username),
    [datasets, username],
  );
  const hubDatasets = useMemo(
    () =>
      sortDatasets(
        datasets.filter((d) => d.source === "hub" || d.source === "both"),
        username,
      ),
    [datasets, username],
  );

  // A well-formed `org/name` the listing doesn't have, offered as a public Hub
  // dataset — the affordance that ANY public dataset is usable, not just the
  // user's own. (Lived in Train's own results list until its typed search was
  // folded into this popover.)
  const trimmedQuery = query.trim();
  const hubCandidate = useMemo(() => {
    if (!onPickHubId || !HUB_REPO_ID_RE.test(trimmedQuery)) return null;
    const q = trimmedQuery.toLowerCase();
    if (datasets.some((d) => d.repo_id.toLowerCase() === q)) return null;
    return trimmedQuery;
  }, [onPickHubId, datasets, trimmedQuery]);

  const reset = () => {
    setQuery("");
    setOpen(false);
  };

  const handlePick = (item: DatasetItem) => {
    onPickExisting(item);
    reset();
  };

  const renderItem = (d: DatasetItem) => (
    <CommandItem
      key={d.repo_id}
      value={d.repo_id}
      onSelect={() => handlePick(d)}
      className="group items-start aria-selected:bg-accent"
    >
      <span className="min-w-0 flex-1 break-all">{d.repo_id}</span>
      {d.source === "both" && (
        <span className="shrink-0 text-xs text-muted-foreground">
          {t("landing.picker.localAndHub")}
        </span>
      )}
      {d.private && (
        <span className="shrink-0 text-xs text-amber-600 dark:text-amber-400">
          {t("landing.picker.private")}
        </span>
      )}
      {onDeleteItem && (
        <button
          type="button"
          aria-label={t("landing.datasetPicker.deleteAria", {
            repoId: d.repo_id,
          })}
          title={t("landing.picker.deleteTitle")}
          // cmdk/Radix act on pointerdown AND the click would bubble to the
          // CommandItem's onSelect — guard both so the trash never also
          // selects the row or closes the popover on its own.
          onPointerDown={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onDeleteItem(d);
            // Close the picker so the Landing-scoped confirm dialog is visible.
            reset();
          }}
          // Hover-revealed on pointer devices (with keyboard-focus fallback),
          // always visible on touch (no hover to reveal it with).
          className="shrink-0 rounded p-0.5 text-muted-foreground hover:text-destructive focus:opacity-100 sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      )}
    </CommandItem>
  );

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>{children}</PopoverTrigger>
      <PopoverContent
        className="w-[320px] p-0"
        align="end"
      >
        <Command>
          {!hideSearch && (
            <CommandInput
              placeholder={t("landing.datasetPicker.searchPlaceholder")}
              value={query}
              onValueChange={setQuery}
            />
          )}
          <CommandList>
            {datasets.length === 0 && (
              <CommandEmpty className="py-4 text-sm text-muted-foreground text-center">
                {loading
                  ? t("landing.datasetPicker.loading")
                  : t("landing.datasetPicker.empty")}
              </CommandEmpty>
            )}
            {hubDatasets.length > 0 && (
              <CommandGroup heading={t("landing.picker.huggingFace")}>
                {hubDatasets.map(renderItem)}
              </CommandGroup>
            )}
            {localDatasets.length > 0 && (
              <CommandGroup heading={t("landing.picker.local")}>
                {localDatasets.map(renderItem)}
              </CommandGroup>
            )}
            {hubCandidate && (
              // Last, under whatever the query did match: reaching for a repo
              // by its full id is the rarer intent, and cmdk keeps this row
              // visible because its value IS the query.
              <CommandGroup>
                <CommandItem
                  value={hubCandidate}
                  onSelect={() => {
                    onPickHubId?.(hubCandidate);
                    reset();
                  }}
                  className="items-start"
                >
                  <Plus className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate">
                      {/* The typed repo id is DATA — one <0> slot, not a
                          sentence stitched around it. (Same two strings the
                          Train panel rendered before its typed search folded
                          into this popover.) */}
                      <Trans
                        i18nKey="landing.datasetPicker.useHub"
                        values={{ repoId: hubCandidate }}
                        components={[<span key="0" className="font-mono" />]}
                      />
                    </span>
                    <span className="block text-xs text-muted-foreground">
                      {t("landing.datasetPicker.useHubHint")}
                    </span>
                  </span>
                </CommandItem>
              </CommandGroup>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
};

export default DatasetPicker;
