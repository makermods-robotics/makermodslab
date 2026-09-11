import React from "react";
import { ChevronDown } from "lucide-react";
import { CollapsibleTrigger } from "@/components/ui/collapsible";
import { useEyebrowClass } from "@/components/studio/panel/primitives";
import { cn } from "@/lib/utils";

/**
 * The unified library header row: a collapsible trigger (eyebrow title +
 * count badge + fold chevron) on the left, library-specific actions on the
 * right. Used by the dataset, training-jobs, and model libraries so all
 * three read as the same component. Must render inside a <Collapsible>.
 */
const LibraryHeader: React.FC<{
  title: string;
  count: number;
  open: boolean;
  /** Right-side actions: buttons, chips — truncation-safe (min-w-0). */
  actions?: React.ReactNode;
}> = ({ title, count, open, actions }) => {
  // `title` arrives already translated from the caller. `.eyebrow` bundles
  // `uppercase` with letter-spacing, and on a caseless script the uppercase is
  // a no-op while the spacing renders the heading visibly over-spaced — the
  // shared hook drops both together.
  const eyebrow = useEyebrowClass();
  return (
    <div className="flex items-center justify-between gap-2">
      <CollapsibleTrigger className="flex shrink-0 items-center gap-2">
        <h3 className={eyebrow}>{title}</h3>
        <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
          {count}
        </span>
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 text-muted-foreground transition-transform",
            open && "rotate-180",
          )}
        />
      </CollapsibleTrigger>
      {actions ? (
        <div className="flex min-w-0 items-center gap-1">{actions}</div>
      ) : null}
    </div>
  );
};

export default LibraryHeader;
