import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { NumberInput } from "@/components/ui/number-input";
import {
  Loader2,
  CheckCircle2,
  XCircle,
  GitMerge,
  Minus,
  Plus,
  Scale,
} from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import {
  datasetRepoIdIssue,
  formatDatasetNameIssue,
} from "@/lib/datasetName";
import { formatBytes } from "@/lib/datasetFormat";
import { cn } from "@/lib/utils";
import {
  DatasetInfo,
  DatasetItem,
  MAX_SOURCE_WEIGHT,
  MergeStatus,
  getDatasetInfo,
  getDatasetMergeStatus,
  startDatasetMerge,
} from "@/lib/replayApi";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasets: DatasetItem[];
  onMerged: () => void;
}

const POLL_MS = 1500;

/** One selected source's contribution to the merged dataset. `baseEpisodes` is
 * null when the dataset's info couldn't be read (Hub-only, offline, corrupt) —
 * the mix preview then degrades to "unavailable" rather than guessing. */
interface MixRow {
  repoId: string;
  weight: number;
  baseEpisodes: number | null;
  episodes: number | null;
  bytes: number | null;
  share: number | null;
}

const MergeDatasetsDialog: React.FC<Props> = ({
  open,
  onOpenChange,
  datasets,
  onMerged,
}) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  // Per-source repeat count, keyed by repo id. A repo absent from this map (or
  // not selected) weighs 1 — the map only ever holds deliberate overrides.
  const [weights, setWeights] = useState<Record<string, number>>({});
  // Episode counts / sizes for the SELECTED sources only, so opening the dialog
  // with 50 datasets doesn't fan out 50 requests. null = lookup failed.
  const [infos, setInfos] = useState<Record<string, DatasetInfo | null>>({});
  const [output, setOutput] = useState("");
  const [status, setStatus] = useState<MergeStatus | null>(null);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const logBoxRef = useRef<HTMLDivElement>(null);
  const notifiedDone = useRef(false);

  // Reset on open, and re-attach to an already-running merge (survives closing
  // the dialog / reloading the page) by seeding from the backend.
  useEffect(() => {
    if (!open) return;
    setSelected(new Set());
    setWeights({});
    // Dropped rather than kept across opens: recording or deleting episodes
    // between opens would leave a stale mix preview on screen.
    setInfos({});
    setOutput("");
    setStartError(null);
    notifiedDone.current = false;
    getDatasetMergeStatus(baseUrl, fetchWithHeaders)
      .then((s) => setStatus(s.state === "running" ? s : null))
      .catch(() => setStatus(null));
  }, [open, baseUrl, fetchWithHeaders]);

  // Fetch dataset info: for the mix preview's episode counts, and — once an
  // anchor is picked — for every listed dataset, so incompatible ones can be
  // greyed out before the user wastes a click on them.
  //
  // Nothing is fetched until the FIRST selection: opening the dialog must not
  // fan out a request per dataset in the library. After that the sweep runs in
  // small batches and rows stay enabled until their own info lands, so a slow
  // Hub lookup never blocks the list — it just marks progressively.
  useEffect(() => {
    if (!open) return;
    const wanted =
      selected.size === 0 ? [] : datasets.map((d) => d.repo_id);
    const missing = wanted.filter((repoId) => !(repoId in infos));
    if (missing.length === 0) return;
    const controller = new AbortController();
    let cancelled = false;
    void (async () => {
      const BATCH = 8;
      for (let i = 0; i < missing.length; i += BATCH) {
        if (cancelled) return;
        const entries = await Promise.all(
          missing.slice(i, i + BATCH).map(async (repoId) => {
            try {
              const info = await getDatasetInfo(
                baseUrl,
                fetchWithHeaders,
                repoId,
                controller.signal,
              );
              return [repoId, info] as const;
            } catch {
              return [repoId, null] as const;
            }
          }),
        );
        if (cancelled) return;
        setInfos((prev) => ({ ...prev, ...Object.fromEntries(entries) }));
      }
    })();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [open, selected, datasets, infos, baseUrl, fetchWithHeaders]);

  // Poll while a merge runs; accumulate the drained log lines.
  useEffect(() => {
    if (!open || status?.state !== "running") return;
    const id = setInterval(async () => {
      try {
        const s = await getDatasetMergeStatus(baseUrl, fetchWithHeaders);
        setStatus((prev) =>
          prev ? { ...s, logs: [...prev.logs, ...s.logs] } : s,
        );
        if (s.state === "done" && !notifiedDone.current) {
          notifiedDone.current = true;
          onMerged();
        }
      } catch {
        // transient — retry next tick
      }
    }, POLL_MS);
    return () => clearInterval(id);
  }, [open, status?.state, baseUrl, fetchWithHeaders, onMerged]);

  useEffect(() => {
    if (logBoxRef.current)
      logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
  }, [status?.logs]);

  const toggle = (repoId: string) => {
    const wasSelected = selected.has(repoId);
    setSelected((prev) => {
      const next = new Set(prev);
      if (wasSelected) next.delete(repoId);
      else next.add(repoId);
      return next;
    });
    // Forget the weight too, so re-selecting starts from 1 instead of silently
    // reviving a number the user can no longer see. Kept OUT of the setSelected
    // updater — updaters must stay pure (React may re-run them).
    if (wasSelected)
      setWeights(({ [repoId]: _dropped, ...rest }) => rest);
  };

  const weightOf = useCallback(
    (repoId: string) => weights[repoId] ?? 1,
    [weights],
  );

  // Clamp on write so invalid state is unrepresentable — the backend cap is
  // mirrored here (MAX_SOURCE_WEIGHT) and an emptied field falls back to 1.
  const setWeight = (repoId: string, next: number | undefined) =>
    setWeights((prev) => ({
      ...prev,
      [repoId]: Math.min(
        MAX_SOURCE_WEIGHT,
        Math.max(1, Math.round(next ?? 1)),
      ),
    }));

  const resetWeights = () => setWeights({});

  // A bare output name (no "/") inherits the sources' namespace when they all
  // share one. Without this, typing "merged" created a namespace-less dataset
  // at the cache root — inconsistent with every other dataset, and rename can
  // never fix it (rename only touches the final path segment). Mixed-namespace
  // sources make no single answer right, so a bare name then stays bare and
  // the user can type the full id explicitly.
  const sourceNamespaces = [...selected].map((id) =>
    id.includes("/") ? id.split("/")[0] : null,
  );
  const commonNamespace =
    sourceNamespaces.length > 0 &&
    sourceNamespaces[0] !== null &&
    sourceNamespaces.every((ns) => ns === sourceNamespaces[0])
      ? sourceNamespaces[0]
      : null;
  const trimmedOutput = output.trim();
  const effectiveOutput =
    trimmedOutput && !trimmedOutput.includes("/") && commonNamespace
      ? `${commonNamespace}/${trimmedOutput}`
      : trimmedOutput;
  const outputIssue = effectiveOutput
    ? datasetRepoIdIssue(effectiveOutput)
    : null;
  const outputError = outputIssue
    ? formatDatasetNameIssue(t, outputIssue)
    : null;

  // Ordered by the LIST, not by when each was clicked. `selected` is a Set, so
  // spreading it yields insertion order — which meant clicking a lower dataset
  // and then a higher one put the lower one at the top of the mix table, the
  // reverse of what is on screen. Anything selected but no longer listed is
  // appended rather than dropped, so a stale id can never vanish silently from
  // the merge it is part of.
  const selectedIds = useMemo(() => {
    const listed = datasets
      .map((d) => d.repo_id)
      .filter((id) => selected.has(id));
    const strays = [...selected].filter((id) => !listed.includes(id));
    return [...listed, ...strays];
  }, [datasets, selected]);
  const anyWeighted = selectedIds.some((repoId) => weightOf(repoId) > 1);

  // Compatibility is judged against the FIRST selected dataset, mirroring the
  // backend, which compares every source to the first readable one. Returns a
  // reason string for an unmergeable dataset, or null.
  //
  // Deliberately permissive: a dataset whose info has not arrived (or could not
  // be read — Hub-only, offline) returns null and stays clickable. Greying out
  // on missing information would hide mergeable datasets, and the backend's own
  // refusal is the real gate. This only spares the user an obvious wasted click.
  //
  // Checks fps, camera set and robot type — the three the frontend can see.
  // The backend additionally compares feature keys/shapes, so passing here is
  // not a promise the merge will be accepted.
  const anchorId = selectedIds[0] ?? null;
  const anchorInfo = anchorId ? infos[anchorId] : null;
  const incompatibilityOf = (repoId: string): string | null => {
    if (!anchorInfo || repoId === anchorId || selected.has(repoId)) return null;
    const info = infos[repoId];
    if (!info) return null; // unknown -> never block
    if (
      info.fps != null &&
      anchorInfo.fps != null &&
      info.fps !== anchorInfo.fps
    ) {
      return t("landing.mergeDatasets.incompatibleFps", {
        theirs: String(info.fps),
        anchor: String(anchorInfo.fps),
      });
    }
    const camsOf = (i: DatasetInfo) => [...i.cameras].sort().join(", ");
    if (camsOf(info) !== camsOf(anchorInfo)) {
      return t("landing.mergeDatasets.incompatibleCameras", {
        theirs: camsOf(info) || "—",
        anchor: camsOf(anchorInfo) || "—",
      });
    }
    if (
      info.robot_type &&
      anchorInfo.robot_type &&
      info.robot_type !== anchorInfo.robot_type
    ) {
      return t("landing.mergeDatasets.incompatibleRobot", {
        theirs: info.robot_type,
        anchor: anchorInfo.robot_type,
      });
    }
    return null;
  };

  // Resulting mix: episodes each source contributes AFTER its weight, and that
  // as a share of the merged total. Shares are what the user is really tuning —
  // "corrections is 23% of training data" is the actionable number, not "x3".
  const mix = useMemo<MixRow[]>(() => {
    const rows = selectedIds.map((repoId) => {
      const info = infos[repoId];
      const weight = weights[repoId] ?? 1;
      const baseEpisodes = info ? info.total_episodes : null;
      return {
        repoId,
        weight,
        baseEpisodes,
        episodes: baseEpisodes === null ? null : baseEpisodes * weight,
        // Weight never multiplies bytes: the merge writes one copy of each
        // source's episodes plus a per-episode `sampling_weight` column, so the
        // merged dataset is the sum of the source sizes regardless of weights.
        bytes: info?.size_bytes ?? null,
        share: null as number | null,
      };
    });
    const total = rows.reduce((sum, r) => sum + (r.episodes ?? 0), 0);
    const complete = rows.length > 0 && rows.every((r) => r.episodes !== null);
    return rows.map((r) => ({
      ...r,
      share:
        complete && total > 0 ? Math.round(((r.episodes ?? 0) / total) * 100) : null,
    }));
  }, [selectedIds, infos, weights]);

  const mixComplete = mix.length > 0 && mix.every((r) => r.episodes !== null);
  const totalEpisodes = mixComplete
    ? mix.reduce((sum, r) => sum + (r.episodes ?? 0), 0)
    : null;
  const totalBytes =
    mix.length > 0 && mix.every((r) => r.bytes !== null)
      ? mix.reduce((sum, r) => sum + (r.bytes ?? 0), 0)
      : null;

  const canMerge =
    selected.size >= 2 &&
    effectiveOutput.length > 0 &&
    outputError === null &&
    !selected.has(effectiveOutput) &&
    status?.state !== "running";

  const handleMerge = async () => {
    setStarting(true);
    setStartError(null);
    try {
      const res = await startDatasetMerge(
        baseUrl,
        fetchWithHeaders,
        selectedIds,
        effectiveOutput,
        selectedIds.map((repoId) => weightOf(repoId)),
      );
      if (!res.started) {
        setStartError(res.message);
        return;
      }
      // Seed a running status so the poll effect attaches immediately.
      setStatus({
        state: "running",
        error: null,
        output_repo_id: effectiveOutput,
        logs: [],
      });
    } catch (e) {
      setStartError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  };

  const state = status?.state ?? "idle";

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {/* Wider than the default max-w-lg: each source row now carries a weight
          stepper. Clamped to the viewport so it can never exceed the screen. */}
      <DialogContent className="max-h-[calc(100vh-2rem)] max-w-[min(36rem,calc(100vw-2rem))]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <GitMerge className="w-5 h-5" /> {t("landing.mergeDatasets.title")}
          </DialogTitle>
          <DialogDescription className="text-muted-foreground">
            {t("landing.mergeDatasets.description")}
          </DialogDescription>
        </DialogHeader>

        {state === "idle" ? (
          // min-w-0: DialogContent is a grid, and a grid item's default
          // min-width:auto makes it refuse to shrink below its min-content
          // width — the content then spills past the dialog's background.
          <div className="min-h-0 min-w-0 space-y-4 overflow-y-auto">
            <div>
              <div className="flex items-baseline justify-between gap-2">
                <Label className="text-foreground">
                  {t("landing.mergeDatasets.sources", { n: selected.size })}
                </Label>
                {selected.size > 0 && (
                  <span className="text-xs text-muted-foreground">
                    {t("landing.mergeDatasets.weightColumn")}
                  </span>
                )}
              </div>
              <div className="mt-1 max-h-56 overflow-auto rounded-md border border-border divide-y divide-border">
                {datasets.length === 0 ? (
                  <p className="p-3 text-sm text-muted-foreground">
                    {t("landing.mergeDatasets.noDatasets")}
                  </p>
                ) : (
                  datasets.map((d) => {
                    const isSelected = selected.has(d.repo_id);
                    const weight = weightOf(d.repo_id);
                    const blockedReason = incompatibilityOf(d.repo_id);
                    return (
                      <div
                        key={d.repo_id}
                        // The reason rides on the ROW, so it is reachable by
                        // hovering anywhere on it — a disabled checkbox emits no
                        // pointer events of its own.
                        title={blockedReason ?? undefined}
                        className={cn(
                          "flex min-w-0 items-center gap-2 p-2 text-sm",
                          blockedReason
                            ? "opacity-40"
                            : "hover:bg-accent",
                        )}
                      >
                        {/* The label wraps only the checkbox + name: the weight
                            stepper must sit outside it, or clicking + would
                            also toggle the selection. */}
                        <label
                          className={cn(
                            "flex min-w-0 flex-1 items-start gap-2",
                            blockedReason ? "cursor-not-allowed" : "cursor-pointer",
                          )}
                        >
                          <Checkbox
                            className="shrink-0 mt-0.5"
                            checked={isSelected}
                            disabled={blockedReason !== null}
                            onCheckedChange={() => toggle(d.repo_id)}
                          />
                          <span className="min-w-0 break-all">{d.repo_id}</span>
                        </label>
                        {isSelected && (
                          <div className="flex shrink-0 items-center gap-1">
                            <Button
                              type="button"
                              variant="outline"
                              size="icon"
                              className="h-7 w-7"
                              disabled={weight <= 1}
                              aria-label={t(
                                "landing.mergeDatasets.decreaseWeight",
                              )}
                              onClick={() => setWeight(d.repo_id, weight - 1)}
                            >
                              <Minus className="h-3 w-3" />
                            </Button>
                            <NumberInput
                              value={weight}
                              onChange={(v) => setWeight(d.repo_id, v)}
                              min={1}
                              max={MAX_SOURCE_WEIGHT}
                              aria-label={t(
                                "landing.mergeDatasets.weightAria",
                                { repoId: d.repo_id },
                              )}
                              // min-w-0 is load-bearing: `w-12` sets width but
                              // not min-width, and a number input's intrinsic
                              // minimum would otherwise pin the row wide.
                              className="h-7 w-12 min-w-0 px-1 text-center tabular-nums"
                            />
                            <Button
                              type="button"
                              variant="outline"
                              size="icon"
                              className="h-7 w-7"
                              disabled={weight >= MAX_SOURCE_WEIGHT}
                              aria-label={t(
                                "landing.mergeDatasets.increaseWeight",
                              )}
                              onClick={() => setWeight(d.repo_id, weight + 1)}
                            >
                              <Plus className="h-3 w-3" />
                            </Button>
                          </div>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            {selected.size >= 2 && (
              <div className="min-w-0 rounded-md border border-border p-3">
                <div className="flex items-center justify-between gap-2">
                  <Label className="flex items-center gap-1.5 text-foreground">
                    <Scale className="h-3.5 w-3.5" />
                    {t("landing.mergeDatasets.mixTitle")}
                  </Label>
                  {anyWeighted && (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-6 px-2 text-xs"
                      onClick={resetWeights}
                    >
                      {t("landing.mergeDatasets.resetWeights")}
                    </Button>
                  )}
                </div>

                <div className="mt-2 max-h-48 min-w-0 space-y-2 overflow-y-auto pr-1">
                  {mix.map((row) => (
                    <div key={row.repoId} className="space-y-1">
                      <div className="flex min-w-0 items-baseline justify-between gap-2 text-xs">
                        <span className="flex min-w-0 flex-1 items-center gap-1.5">
                          <span className="min-w-0 truncate text-foreground">
                            {row.repoId}
                          </span>
                          {row.weight > 1 && (
                            <Badge
                              variant="secondary"
                              className="shrink-0 px-1 py-0 text-[10px]"
                            >
                              {t("landing.mergeDatasets.weightTimes", {
                                weight: row.weight,
                              })}
                            </Badge>
                          )}
                        </span>
                        <span className="shrink-0 text-muted-foreground tabular-nums">
                          {row.episodes === null
                            ? t("landing.mergeDatasets.episodesUnknown")
                            : row.weight > 1
                              ? t("landing.mergeDatasets.mixEpisodesWeighted", {
                                  count: row.episodes,
                                  // Non-null whenever `episodes` is: both come
                                  // from the same info lookup.
                                  base: row.baseEpisodes ?? 0,
                                })
                              : t("landing.mergeDatasets.mixEpisodesPlain", {
                                  count: row.episodes,
                                })}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                          <div
                            className="h-full rounded-full bg-info transition-all"
                            style={{ width: `${row.share ?? 0}%` }}
                          />
                        </div>
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

                <Separator className="my-2" />
                <div className="flex items-baseline justify-between gap-2 text-xs text-muted-foreground">
                  <span>
                    {totalEpisodes === null
                      ? t("landing.mergeDatasets.episodesUnknown")
                      : t("landing.mergeDatasets.mixTotal", {
                          count: totalEpisodes,
                        })}
                  </span>
                  {totalBytes !== null && (
                    <span className="tabular-nums">
                      {t("landing.mergeDatasets.diskEstimate", {
                        size: formatBytes(totalBytes),
                      })}
                    </span>
                  )}
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  {anyWeighted
                    ? t("landing.mergeDatasets.weightedHint")
                    : t("landing.mergeDatasets.weightHint")}
                </p>
              </div>
            )}

            <div>
              <Label htmlFor="merge-output" className="text-foreground">
                {t("landing.mergeDatasets.outputLabel")}
              </Label>
              <Input
                id="merge-output"
                value={output}
                onChange={(e) => setOutput(e.target.value)}
                placeholder="user/merged_dataset"
                aria-invalid={outputError !== null}
                className="mt-1 aria-[invalid=true]:border-destructive/70"
              />
              {outputError && (
                <p className="mt-1 text-xs text-destructive">{outputError}</p>
              )}
              {!outputError && effectiveOutput !== trimmedOutput && (
                <p className="mt-1 text-xs text-muted-foreground">
                  <Trans
                    i18nKey="landing.mergeDatasets.willBeCreatedAs"
                    values={{ repoId: effectiveOutput }}
                    components={[<code key="0" className="text-info" />]}
                  />
                </p>
              )}
            </div>
            {startError ? (
              <p className="text-sm text-destructive">{startError}</p>
            ) : null}
            <div className="flex justify-end">
              <Button
                onClick={handleMerge}
                disabled={!canMerge || starting}
                className=""
              >
                {starting ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />{" "}
                    {t("landing.mergeDatasets.starting")}
                  </>
                ) : (
                  <>
                    <GitMerge className="w-4 h-4 mr-2" />{" "}
                    {t("landing.mergeDatasets.submit", {
                      count: selected.size,
                    })}
                  </>
                )}
              </Button>
            </div>
          </div>
        ) : (
          <div className="min-h-0 min-w-0 space-y-3 overflow-y-auto">
            <div className="flex items-center gap-2 text-sm text-foreground">
              {state === "running" ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin text-info" />
                  <Trans
                    i18nKey="landing.mergeDatasets.merging"
                    values={{ repoId: status?.output_repo_id ?? "" }}
                    components={[<code key="0" className="text-info" />]}
                  />
                </>
              ) : state === "done" ? (
                <>
                  <CheckCircle2 className="w-4 h-4 text-ok" />
                  <Trans
                    i18nKey="landing.mergeDatasets.created"
                    values={{ repoId: status?.output_repo_id ?? "" }}
                    components={[<code key="0" className="text-ok" />]}
                  />
                </>
              ) : (
                <>
                  <XCircle className="w-4 h-4 text-destructive" />{" "}
                  {t("landing.mergeDatasets.failed")}
                </>
              )}
            </div>
            <div
              ref={logBoxRef}
              className="max-h-56 overflow-auto rounded-md border border-border bg-muted p-2 font-mono text-xs text-foreground whitespace-pre-wrap"
            >
              {(status?.logs ?? []).map((l, i) => (
                <div key={i}>{l.message}</div>
              ))}
            </div>
            {status?.error ? (
              <p className="text-sm text-destructive">{status.error}</p>
            ) : null}
            <div className="flex justify-end">
              <Button
                variant="outline"
                onClick={() => onOpenChange(false)}
              >
                {state === "done"
                  ? t("landing.mergeDatasets.done")
                  : t("common.close")}
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default MergeDatasetsDialog;
