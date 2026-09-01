import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { LIBRARY_GRID, LIBRARY_GRID_COLS } from "./LibraryToolbar";

/** Column width (px) at or above which a library row fits three cards instead
 * of two. Each library sits in a narrow one-third studio column, so what
 * decides the count is that column's width, not the display's: a 16" MacBook
 * (1728px viewport) gives the column 535px, where three cards are 173px each
 * and their icons and buttons squeeze past the card edges. Two cards there are
 * 263px. Three cards only reach that same 263px once the column is
 * 263*3 + 8*2 ≈ 805px wide, so that is the crossover. A large external display
 * (3008px viewport → 962px column) clears it and gets three. */
const WIDE_COLUMN_MIN_PX = 800;

/** Total height of the reserved grid block: the 16.5rem row + the 1.875rem
 * footer slot ("Show all" / Untracked / spacer) and its gutter. Empty/no-match
 * states adopt it so a library never shrinks below a populated one. Keep in
 * sync with the literal classes below — Tailwind only generates classes it
 * can see as literals. */
export const GRID_MIN_H = "min-h-[18.875rem]";

/** The same reservation as a FIXED height, for a library whose content is not
 * a fixed-height card row. A grid can floor its height and stop there — its
 * rows are 16.5rem whatever they hold. Jobs renders a dropdown plus one
 * variable-height detail card, so flooring alone would let a tall card push the
 * Train panel's action row back out of line with its siblings; capping too, and
 * scrolling inside the box, keeps the block one exact height in both
 * directions. Same measurement as GRID_MIN_H — change them together (and keep
 * both literal: Tailwind only generates classes it can see). */
export const GRID_H = "h-[18.875rem]";
/** Just the 16.5rem card row, without the footer slot. An empty/no-match state
 * that renders its OWN footer row beneath it (jobs' Untracked toggle) must use
 * this instead of GRID_MIN_H — otherwise message + footer stack to 21.25rem and
 * that library sits 2.375rem taller than the other panels'. */
export const GRID_ROW_MIN_H = "min-h-[16.5rem]";

/**
 * The shared library grid, capped at one row. Every studio library (datasets,
 * training jobs, models) renders one row of cards by default — three where the
 * column is wide enough (see WIDE_COLUMN_MIN_PX), two otherwise; anything past
 * that stays hidden behind a "Show all" toggle. The row is always reserved
 * (fixed row height, blank cells when there aren't enough cards) so the three
 * panels' libraries keep one uniform height however large a collection grows.
 */
const CappedGrid: React.FC<{
  /** Pre-keyed cards, already sorted newest-first by the caller. */
  items: React.ReactNode[];
  /** Reserve the fixed row even when items don't fill it (the main
   * libraries). False for nested grids (e.g. Untracked) that should hug
   * their content. */
  reserveRows?: boolean;
  /** Hold the 30px footer slot with an invisible spacer when there's no
   * "Show all" — keeps every library's total height identical. False when
   * the caller renders its own footer row in that slot (jobs' Untracked). */
  footerSpacer?: boolean;
  /** Optionally drive the expanded state from the parent. Jobs lifts it so its
   * Untracked toggle can fold *under* "Show all" — appearing only once "Show
   * all" is open. Omit for the self-managed default (datasets, models). */
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
  /** Reports whether this grid currently spills past its one reserved row —
   * i.e. whether it is showing a "Show all" toggle. The row cap is measured
   * from the column, not a constant, so a caller that needs to know (jobs,
   * to decide where its Untracked toggle goes) cannot derive it from the item
   * count alone. */
  onOverflowChange?: (overflow: boolean) => void;
}> = ({
  items,
  reserveRows = true,
  footerSpacer = true,
  expanded: expandedProp,
  onExpandedChange,
  onOverflowChange,
}) => {
  const { t } = useTranslation();
  const [expandedState, setExpandedState] = useState(false);
  const expanded = expandedProp ?? expandedState;
  const setExpanded = (value: boolean) => {
    onExpandedChange?.(value);
    if (expandedProp === undefined) setExpandedState(value);
  };
  const columnRef = useRef<HTMLDivElement>(null);
  // How many cards one row holds here. Measured rather than keyed off a media
  // query so the cap below always matches the number of cards actually laid
  // out — one threshold, not a CSS breakpoint and a JS constant that can drift
  // apart. useLayoutEffect so the first measurement lands before paint and the
  // row never flashes at the wrong count.
  const [cap, setCap] = useState<keyof typeof LIBRARY_GRID_COLS>(2);
  useLayoutEffect(() => {
    const el = columnRef.current;
    if (!el) return;
    const apply = (width: number) =>
      setCap(width >= WIDE_COLUMN_MIN_PX ? 3 : 2);
    apply(el.getBoundingClientRect().width);
    const observer = new ResizeObserver(([entry]) =>
      apply(entry.contentRect.width),
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  const overflow = items.length - cap;
  useEffect(() => {
    onOverflowChange?.(overflow > 0);
  }, [overflow, onOverflowChange]);
  const shown = expanded || overflow <= 0 ? items : items.slice(0, cap);
  return (
    <div ref={columnRef} className="space-y-2">
      <div
        className={cn(
          LIBRARY_GRID,
          LIBRARY_GRID_COLS[cap],
          "auto-rows-[16.5rem]",
          reserveRows && "grid-rows-[16.5rem]",
        )}
      >
        {shown}
      </div>
      {overflow > 0 ? (
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="flex h-[1.875rem] w-full items-center justify-center gap-1 rounded-md border border-dashed border-border text-xs font-medium text-muted-foreground transition-colors hover:border-muted-foreground/40 hover:text-foreground"
        >
          <ChevronDown
            className={cn(
              "h-3.5 w-3.5 transition-transform",
              expanded && "rotate-180",
            )}
          />
          {expanded
            ? t("library.grid.showLess")
            : /* `total`, not i18next's `count`: this is a tally, not a plural
                 switch, and the row cap it counts against is measured, not
                 fixed. */
              t("library.grid.showAll", { total: items.length })}
        </button>
      ) : reserveRows && footerSpacer ? (
        <div aria-hidden className="h-[1.875rem]" />
      ) : null}
    </div>
  );
};

export default CappedGrid;
