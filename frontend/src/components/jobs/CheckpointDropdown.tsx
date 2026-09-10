import React from "react";
import { useTranslation } from "react-i18next";
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
  placeholder,
  className,
  id,
  owners,
}) => {
  const { t } = useTranslation();
  // Resolved in the body rather than as a default parameter: a default is
  // evaluated before `t` exists, and a module-level default would freeze the
  // first language loaded.
  const placeholderText = placeholder ?? t("jobs.checkpointDropdown.placeholder");
  // step 0 is the sentinel for an imported single-model checkpoint (lerobot
  // never saves at step 0), so it has no meaningful step number — show
  // "latest" instead. Real training checkpoints keep their step label. The
  // step is stringified by hand, not run through a locale number formatter.
  const labelFor = (step: number) =>
    step === 0
      ? t("jobs.checkpointDropdown.latest")
      : t("jobs.checkpointDropdown.step", { step: String(step) });
  // A checkpoint label is an identifier, not prose: JetBrains Mono + tabular
  // figures is the house treatment for that everywhere else in the app, and
  // tabular figures also stop the step column jittering as the digit count
  // changes down a list (step 1,000 / step 20,000 / step 100,000).
  // Applied to the whole label rather than just the numeral so the string
  // stays one translatable unit.
  const CheckpointLabel: React.FC<{ step: number }> = ({ step }) => (
    <span className="font-mono tabular-nums">{labelFor(step)}</span>
  );
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
          // A picker CHIP — deliberately between the two things this control
          // has been. The original was styled as a text input (h-8 rectangle,
          // `border-input`, a 110px min-width padding it out), which fought the
          // fine-tune banner's prose box: a bordered box inside a bordered box.
          // Stripping all the chrome fixed that but went too far — with only a
          // hover background, nothing at rest said "you can change this".
          //
          // So: a permanent boundary (it must read as interactive without being
          // hovered) in a pill radius with a `bg-background` surface, which
          // reads as a token rather than a field and contrasts against the
          // banner's own `bg-muted/50`. `border-border` rather than
          // `border-input` — the latter is the form-field token and is what
          // made it look like somewhere you type.
          //
          // justify-between is inherited from the base on purpose: at w-auto the box
          // shrinks to fit so it behaves like justify-start with the gap, while
          // JobCard's w-full override still parks the caret at the right edge.
          "h-7 w-auto min-w-0 gap-1.5 rounded-full border border-border bg-background px-2.5 py-0 text-xs shadow-sm",
          "hover:border-ring hover:bg-accent hover:text-accent-foreground",
          "data-[state=open]:border-ring data-[state=open]:bg-accent",
          "focus:ring-1 focus:ring-offset-0",
          // The shared caret is sized for a full-height field; scale it down so
          // it sits with inline text instead of towering over it.
          "[&_svg]:h-3.5 [&_svg]:w-3.5 [&_svg]:shrink-0 [&_svg]:text-foreground/70",
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
          <SelectValue placeholder={placeholderText}>
            <CheckpointLabel step={selected.step} />
          </SelectValue>
        ) : selected !== undefined ? (
          // Always render our own label, not Radix's mirror of the item, so the
          // trigger gets the same mono treatment as the list.
          <SelectValue placeholder={placeholderText}>
            <CheckpointLabel step={selected.step} />
          </SelectValue>
        ) : (
          <SelectValue placeholder={placeholderText} />
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
                  <CheckpointLabel step={c.step} />
                  <span className="whitespace-nowrap text-[10px] leading-none text-muted-foreground">
                    {owner.number > 0 ? (
                      <span className="font-mono">#{owner.number} </span>
                    ) : null}
                    {owner.name}
                  </span>
                </span>
              ) : (
                <CheckpointLabel step={c.step} />
              )}
            </SelectItem>
          );
        })}
      </SelectContent>
    </Select>
  );
};

export default CheckpointDropdown;
