import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { JobCheckpoint } from "@/lib/checkpointsApi";
import { cn } from "@/lib/utils";

interface Props {
  checkpoints: JobCheckpoint[];
  /** `ref` of the selected checkpoint. The ref is the checkpoint's unique
   * identity (it encodes the owning repo/path plus the step); the step alone
   * is NOT unique when a card merges checkpoints across a resume lineage —
   * two runs can each have a step 2000. */
  selectedRef: string | null;
  onChange: (ckpt: JobCheckpoint) => void;
  disabled?: boolean;
  placeholder?: string;
  /** Extra classes for the trigger (e.g. `w-full min-w-0` when the dropdown
   * flexes inside a card's single-line action row). */
  className?: string;
  /** Trigger id, so a <Label htmlFor> can be attached when the dropdown is
   * rendered as a labelled form field (the Run panel). */
  id?: string;
  /** Which run each checkpoint came from, keyed by `ref`, for the callers that
   * merge a whole resume lineage into one list (JobCard, ModelCard).
   *
   * Optional because most callers show a single run's checkpoints, where the
   * attribution would be the same word on every row. Even when supplied it is
   * only RENDERED if the list actually spans more than one run — see below.
   *
   * Each value carries the run NUMBER as the visible distinguisher — runs on
   * one chain share a name by design — plus a `detail` string surfaced on
   * hover for matching a row against a log line or an API message. */
  owners?: Record<
    string,
    { name: string; number: number; detail: string }
  >;
}

export const CheckpointDropdown: React.FC<Props> = ({
  checkpoints,
  selectedRef,
  onChange,
  disabled,
  placeholder = "Select checkpoint",
  className,
  id,
  owners,
}) => {
  // step 0 is the sentinel for an imported single-model checkpoint (lerobot
  // never saves at step 0), so it has no meaningful step number — show
  // "latest" instead. Real training checkpoints keep their step label.
  const labelFor = (step: number) => (step === 0 ? "latest" : `step ${step}`);
  // Render newest-first regardless of the caller's order (the backend lists
  // ascending; JobCard pre-sorts descending — this is the one authoritative
  // display order). The step-0 "latest" sentinel sorts to the top, not the
  // bottom. Sorting a copy keeps the callers' own arrays in backend order,
  // which their `cks[cks.length - 1]` "latest" defaults rely on; the sort is
  // stable, so same-step entries keep the caller's relative order.
  // (Finite sentinel key: Infinity - Infinity would be NaN if a list ever
  // held two "latest" entries, making the comparator inconsistent.)
  const sortKey = (c: JobCheckpoint) =>
    c.step === 0 ? Number.MAX_SAFE_INTEGER : c.step;
  const ordered = [...checkpoints].sort((a, b) => sortKey(b) - sortKey(a));
  // Attribution earns its space only when the list MERGES runs. A single run's
  // checkpoints all carry the same owner, so printing it on every row would be
  // noise on the common case — and on a lineage it is the opposite of noise,
  // because "step 2000" appears once per run that saved one and the step alone
  // cannot say which is which (the `selectedRef` note above is the same fact
  // from the selection side).
  const attributed =
    owners !== undefined &&
    new Set(ordered.map((c) => owners[c.ref]?.detail).filter(Boolean)).size > 1;
  // Resolved, not assumed: a `selectedRef` naming a checkpoint that is no
  // longer in the list (a refresh dropped it) must fall back to the
  // placeholder, never to a step number invented from a missing entry — step 0
  // is the "latest" sentinel, so a `?? 0` would have quietly labelled a stale
  // selection "latest".
  const selected =
    selectedRef === null
      ? undefined
      : checkpoints.find((c) => c.ref === selectedRef);
  return (
    <Select
      value={selectedRef ?? undefined}
      onValueChange={(ref) => {
        const picked = checkpoints.find((c) => c.ref === ref);
        if (picked) onChange(picked);
      }}
      disabled={disabled || checkpoints.length === 0}
    >
      <SelectTrigger
        id={id}
        className={cn(
          "bg-background border-input h-8 text-xs px-2 w-auto min-w-[110px]",
          className,
        )}
        onClick={(e) => e.stopPropagation()}
      >
        {/* The TRIGGER stays the step alone even when the list is attributed.
            Radix would otherwise mirror the whole selected row into it, and the
            trigger is the tightest space on the card (ModelCard pins it to
            w-36, JobCard shares a single action line with the Resume button) —
            an owner + timestamp there would push the step out of view, losing
            the one fact the collapsed control exists to show. The disambiguator
            belongs where the ambiguity is visible: the open list. */}
        {attributed && selected !== undefined ? (
          <SelectValue placeholder={placeholder}>
            {labelFor(selected.step)}
          </SelectValue>
        ) : (
          <SelectValue placeholder={placeholder} />
        )}
      </SelectTrigger>
      <SelectContent className="bg-popover border-border">
        {ordered.map((c) => {
          const owner = owners?.[c.ref];
          return (
            <SelectItem
              key={c.ref}
              value={c.ref}
              onClick={(e) => e.stopPropagation()}
            >
              {attributed && owner ? (
                // Two lines rather than one: the step is what the user picks
                // by, so it keeps the readable weight, and the run it belongs
                // to sits under it. The RUN NUMBER leads that second line — it
                // is short enough for a dense row and is the same handle the
                // backend's refusals lead with, so a 409 naming #46 points at
                // a row the user can find. The timestamp and full id stay one
                // hover away rather than spending width here.
                <span className="flex flex-col gap-0.5" title={owner.detail}>
                  <span>{labelFor(c.step)}</span>
                  <span className="whitespace-nowrap text-[10px] leading-none text-muted-foreground">
                    {owner.number > 0 ? (
                      <span className="font-mono">#{owner.number} </span>
                    ) : null}
                    {owner.name}
                  </span>
                </span>
              ) : (
                labelFor(c.step)
              )}
            </SelectItem>
          );
        })}
      </SelectContent>
    </Select>
  );
};

export default CheckpointDropdown;
