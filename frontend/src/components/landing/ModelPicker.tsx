import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Trash2 } from "lucide-react";
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
import { ModelItem } from "@/lib/modelsApi";
import { sortByNamespaceFirst } from "@/lib/sortByNamespaceFirst";
import { policyTypeShortLabel } from "@/components/training/types";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useLanguage } from "@/contexts/LanguageContext";
import { isCaselessScript } from "@/i18n/config";
import { cn } from "@/lib/utils";

/** A picker row. `state` is the terminal state of the run behind the model,
 * which /skills reports (SkillItem) and the wider /models listing does not —
 * optional here so both listings can be passed. */
type PickerModel = ModelItem & { state?: string | null };

interface ModelPickerProps {
  models: PickerModel[];
  loading: boolean;
  onPickExisting: (item: PickerModel) => void;
  /** Per-row trash affordance. Invoked with the row's item; the parent routes
   * it through the shared delete confirm dialog (resolveDeleteAction decides
   * the semantics: local delete / local-copy removal / unpin / hide). The
   * picker closes so the Landing-scoped dialog is visible. */
  onDeleteItem?: (item: ModelItem) => void;
  children: React.ReactNode;
}

/** Namespace key: the hub repo id when the model has one, else the local run
 * id (a slash-less run id only counts as "mine" when it equals the username). */
const modelKey = (m: ModelItem): string => m.hf_repo_id ?? m.id ?? "";

/** Namespace-first alphabetical ordering via the shared sortByNamespaceFirst
 * (the same helper sortDatasets wraps). Sorts on the display label so the
 * ordering matches what's shown. */
const sortModels = (
  items: PickerModel[],
  username: string | null,
): PickerModel[] =>
  sortByNamespaceFirst(items, username, modelKey, (m) => m.name || modelKey(m));

/**
 * Model selector mirroring DatasetPicker: a searchable popover split into
 * "Hugging Face" (hub / both) and "Local" (local-only) sections, each row
 * showing the model's display name. Creation is NOT offered here — new models
 * are made on the Create-a-model page — so this picker only selects an existing
 * model (unlike DatasetPicker, which also creates / opens custom repos).
 */
const ModelPicker: React.FC<ModelPickerProps> = ({
  models,
  loading,
  onPickExisting,
  onDeleteItem,
  children,
}) => {
  const { t } = useTranslation();
  const { language } = useLanguage();
  const [open, setOpen] = useState(false);

  const { auth } = useHfAuth();
  const username = auth.status === "authenticated" ? auth.username : null;

  // Strict partition by Hub status, same rule as DatasetPicker: Local = not on
  // the Hub; Hugging Face = on the Hub, whether or not a local copy also exists.
  const localModels = useMemo(
    () => sortModels(models.filter((m) => m.source === "local"), username),
    [models, username],
  );
  const hubModels = useMemo(
    () =>
      sortModels(
        models.filter((m) => m.source === "hub" || m.source === "both"),
        username,
      ),
    [models, username],
  );

  const handlePick = (item: PickerModel) => {
    onPickExisting(item);
    setOpen(false);
  };

  const renderItem = (m: PickerModel) => (
    <CommandItem
      key={m.id || m.hf_repo_id || m.name}
      // cmdk filters/matches on `value`; include the name so a search over the
      // visible label works even when the id is an opaque run id.
      value={`${m.name} ${m.id} ${m.hf_repo_id ?? ""}`}
      onSelect={() => handlePick(m)}
      className="group items-start aria-selected:bg-accent"
    >
      <span className="min-w-0 flex-1 break-all">{m.name}</span>
      {/* Policy type — from the local checkpoint's config.json, or inferred
          from the Hub repo's tags/name (backend _hub_policy_type). Omitted
          when unknown. */}
      {m.policy_type && (
        <span className="shrink-0 text-xs text-muted-foreground">
          {policyTypeShortLabel(m.policy_type)}
        </span>
      )}
      {m.source === "both" && (
        <span className="shrink-0 text-xs text-muted-foreground">
          {t("landing.picker.localAndHub")}
        </span>
      )}
      {m.private && (
        <span className="shrink-0 text-xs text-amber-600 dark:text-amber-400">
          {t("landing.picker.private")}
        </span>
      )}
      {/* A failed run that saved weights IS runnable, and the Train panel's
          card has always run one. It is offered here rather than silently
          withheld — but it says so, because a non-zero exit is a fact about
          the run the user should weigh before deploying it.
          `uppercase` is a no-op on Chinese but the tracking is not — drop both
          together on a caseless script. */}
      {m.state === "failed" && (
        <span
          className={cn(
            "shrink-0 text-[10px] text-amber-600 dark:text-amber-500",
            isCaselessScript(language) ? "" : "uppercase tracking-wide",
          )}
        >
          {t("landing.modelPicker.failedBadge")}
        </span>
      )}
      {onDeleteItem && (
        <button
          type="button"
          aria-label={t("landing.modelPicker.deleteAria", { name: m.name })}
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
            onDeleteItem(m);
            // Close the picker so the Landing-scoped confirm dialog is visible.
            setOpen(false);
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
          <CommandInput
            placeholder={t("landing.modelPicker.searchPlaceholder")}
          />
          <CommandList>
            {models.length === 0 && (
              <CommandEmpty className="py-4 text-sm text-muted-foreground text-center">
                {loading
                  ? t("landing.modelPicker.loading")
                  : t("landing.modelPicker.empty")}
              </CommandEmpty>
            )}
            {hubModels.length > 0 && (
              <CommandGroup heading={t("landing.picker.huggingFace")}>
                {hubModels.map(renderItem)}
              </CommandGroup>
            )}
            {localModels.length > 0 && (
              <CommandGroup heading={t("landing.picker.local")}>
                {localModels.map(renderItem)}
              </CommandGroup>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
};

export default ModelPicker;
