import React, { useEffect, useRef, useState } from "react";
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
import { Loader2, CheckCircle2, XCircle, GitMerge } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { validateDatasetRepoId } from "@/lib/datasetName";
import {
  DatasetItem,
  MergeStatus,
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

const MergeDatasetsDialog: React.FC<Props> = ({
  open,
  onOpenChange,
  datasets,
  onMerged,
}) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const [selected, setSelected] = useState<Set<string>>(new Set());
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
    setOutput("");
    setStartError(null);
    notifiedDone.current = false;
    getDatasetMergeStatus(baseUrl, fetchWithHeaders)
      .then((s) => setStatus(s.state === "running" ? s : null))
      .catch(() => setStatus(null));
  }, [open, baseUrl, fetchWithHeaders]);

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

  const toggle = (repoId: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(repoId)) next.delete(repoId);
      else next.add(repoId);
      return next;
    });

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
  const outputError = effectiveOutput ? validateDatasetRepoId(effectiveOutput) : null;
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
        [...selected],
        effectiveOutput,
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
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <GitMerge className="w-5 h-5" /> {t("landing.mergeDatasets.title")}
          </DialogTitle>
          <DialogDescription className="text-muted-foreground">
            {t("landing.mergeDatasets.description")}
          </DialogDescription>
        </DialogHeader>

        {state === "idle" ? (
          <div className="space-y-4">
            <div>
              <Label className="text-foreground">
                {t("landing.mergeDatasets.sources", { n: selected.size })}
              </Label>
              <div className="mt-1 max-h-56 overflow-auto rounded-md border border-border divide-y divide-border">
                {datasets.length === 0 ? (
                  <p className="p-3 text-sm text-muted-foreground">
                    {t("landing.mergeDatasets.noDatasets")}
                  </p>
                ) : (
                  datasets.map((d) => (
                    <label
                      key={d.repo_id}
                      className="flex items-start gap-2 p-2 hover:bg-accent cursor-pointer text-sm"
                    >
                      <Checkbox
                        className="shrink-0 mt-0.5"
                        checked={selected.has(d.repo_id)}
                        onCheckedChange={() => toggle(d.repo_id)}
                      />
                      <span className="min-w-0 break-all">{d.repo_id}</span>
                    </label>
                  ))
                )}
              </div>
            </div>
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
          <div className="space-y-3">
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
