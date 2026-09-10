import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { Minus, Plus } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { NumberInput } from "@/components/ui/number-input";

/** One source's contribution to a weighted merge. `baseEpisodes` is null when
 * the dataset's episode count could not be read (Hub-only, offline) — the mix
 * preview then degrades to "unavailable" rather than guessing a share. */
export interface DatasetWeightRow {
  repo_id: string;
  weight: number;
  baseEpisodes: number | null;
}

interface Props {
  value: DatasetWeightRow[];
  onChange: (rows: DatasetWeightRow[]) => void;
  /** Largest per-source weight (mirrors the backend cap — pass MAX_SOURCE_WEIGHT). */
  maxWeight: number;
  /** Freeze the steppers while a merge this feeds is already running. */
  disabled?: boolean;
}

interface Computed extends DatasetWeightRow {
  episodes: number | null;
  share: number | null;
}

/** Post-weight episodes each source contributes, and that as a percentage of
 * the merged total. Shares are the actionable number ("corrections is 23% of
 * training data"), not the raw weight. Formula kept exact:
 *   episodes = baseEpisodes * weight;  share = episodes / sum(episodes). */
function computeMix(rows: DatasetWeightRow[]): Computed[] {
  const withEpisodes = rows.map((r) => ({
    ...r,
    episodes: r.baseEpisodes === null ? null : r.baseEpisodes * r.weight,
  }));
  const complete =
    withEpisodes.length > 0 && withEpisodes.every((r) => r.episodes !== null);
  const total = withEpisodes.reduce((sum, r) => sum + (r.episodes ?? 0), 0);
  return withEpisodes.map((r) => ({
    ...r,
    share:
      complete && total > 0
        ? Math.round(((r.episodes ?? 0) / total) * 100)
        : null,
  }));
}

/**
 * Per-source weight steppers plus a live "resulting mix" readout — each
 * source's share of the merged dataset after its weight is applied. Shared by
 * the Merge dialog and the Train panel's "Combine multiple datasets" mode.
 */
export const DatasetWeightPicker: React.FC<Props> = ({
  value,
  onChange,
  maxWeight,
  disabled = false,
}) => {
  const { t } = useTranslation();
  const mix = useMemo(() => computeMix(value), [value]);

  const setWeight = (repoId: string, next: number | undefined) => {
    const clamped = Math.min(
      maxWeight,
      Math.max(1, Math.round(next ?? 1)),
    );
    onChange(
      value.map((r) => (r.repo_id === repoId ? { ...r, weight: clamped } : r)),
    );
  };

  return (
    <div className="min-w-0 space-y-3">
      {mix.map((row) => (
        <div key={row.repo_id} className="space-y-1">
          <div className="flex min-w-0 items-center gap-2 text-xs">
            <span className="flex min-w-0 flex-1 items-center gap-1.5">
              <span className="min-w-0 truncate text-foreground">
                {row.repo_id}
              </span>
              {row.weight > 1 && (
                <Badge
                  variant="secondary"
                  className="shrink-0 px-1 py-0 text-[10px]"
                >
                  {t("landing.mergeDatasets.weightTimes", { weight: row.weight })}
                </Badge>
              )}
            </span>
            <div className="flex shrink-0 items-center gap-1">
              <Button
                type="button"
                variant="outline"
                size="icon"
                className="h-7 w-7"
                disabled={disabled || row.weight <= 1}
                aria-label={t("landing.mergeDatasets.decreaseWeight")}
                onClick={() => setWeight(row.repo_id, row.weight - 1)}
              >
                <Minus className="h-3 w-3" />
              </Button>
              <NumberInput
                value={row.weight}
                onChange={(v) => setWeight(row.repo_id, v)}
                min={1}
                max={maxWeight}
                disabled={disabled}
                aria-label={t("landing.mergeDatasets.weightAria", {
                  repoId: row.repo_id,
                })}
                className="h-7 w-12 min-w-0 px-1 text-center tabular-nums"
              />
              <Button
                type="button"
                variant="outline"
                size="icon"
                className="h-7 w-7"
                disabled={disabled || row.weight >= maxWeight}
                aria-label={t("landing.mergeDatasets.increaseWeight")}
                onClick={() => setWeight(row.repo_id, row.weight + 1)}
              >
                <Plus className="h-3 w-3" />
              </Button>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-info transition-all"
                style={{ width: `${row.share ?? 0}%` }}
              />
            </div>
            <span className="shrink-0 text-right text-xs tabular-nums text-muted-foreground">
              {row.episodes === null
                ? t("landing.mergeDatasets.episodesUnknown")
                : row.weight > 1
                  ? t("landing.mergeDatasets.mixEpisodesWeighted", {
                      count: row.episodes,
                      base: row.baseEpisodes ?? 0,
                    })
                  : t("landing.mergeDatasets.mixEpisodesPlain", {
                      count: row.episodes,
                    })}
            </span>
            <span className="w-9 shrink-0 text-right text-xs tabular-nums text-muted-foreground">
              {row.share === null
                ? "—"
                : t("landing.mergeDatasets.sharePercent", {
                    percent: row.share,
                  })}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
};
