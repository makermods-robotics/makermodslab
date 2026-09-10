import React, { useEffect, useRef } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Loader2, CheckCircle2, XCircle } from "lucide-react";

import type { MergeState } from "@/lib/replayApi";

interface Props {
  /** Drained merge-log lines, oldest first. */
  logs: string[];
  state: MergeState;
  error: string | null;
  /** The name the merge is writing to (running) or wrote (done). */
  outputRepoId?: string | null;
}

/**
 * The status line + scrolling log panel for a running / finished merge. Shared
 * by the Merge dialog and the Train panel's "Combine multiple datasets" mode
 * (which shows it inline while phase one — the merge — runs).
 */
export const MergeProgress: React.FC<Props> = ({
  logs,
  state,
  error,
  outputRepoId,
}) => {
  const { t } = useTranslation();
  const logBoxRef = useRef<HTMLDivElement>(null);
  // `logs` is a fresh array on every parent render; only chase the scroll when
  // a line was actually appended.
  const seenCountRef = useRef(0);

  useEffect(() => {
    if (logs.length === seenCountRef.current) return;
    seenCountRef.current = logs.length;
    if (logBoxRef.current)
      logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
  }, [logs]);

  if (state === "idle") return null;

  return (
    <div className="min-w-0 space-y-3">
      <div className="flex items-center gap-2 text-sm text-foreground">
        {state === "running" ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin text-info" />
            <Trans
              i18nKey="landing.mergeDatasets.merging"
              values={{ repoId: outputRepoId ?? "" }}
              components={[<code key="0" className="text-info" />]}
            />
          </>
        ) : state === "done" ? (
          <>
            <CheckCircle2 className="w-4 h-4 text-ok" />
            <Trans
              i18nKey="landing.mergeDatasets.created"
              values={{ repoId: outputRepoId ?? "" }}
              components={[<code key="0" className="text-ok" />]}
            />
          </>
        ) : state === "cancelled" ? (
          <>
            <XCircle className="w-4 h-4 text-muted-foreground" />{" "}
            {t("landing.mergeDatasets.cancelled")}
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
        {logs.map((l, i) => (
          <div key={i}>{l}</div>
        ))}
      </div>
      {error ? <p className="text-sm text-destructive">{error}</p> : null}
    </div>
  );
};
