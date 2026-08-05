import React from "react";
import { cn } from "@/lib/utils";
import { useTruncationTitle } from "@/hooks/useTruncationTitle";

interface TruncateWithTitleProps {
  /** The text as rendered — already shortened, if the caller shortened it. */
  text: string;
  /**
   * The complete text `text` was shortened FROM. Defaults to `text` (nothing
   * was shortened in JS, so only CSS truncation can raise the title).
   */
  full?: string;
  /**
   * What the tooltip reveals, when that differs from `full` — e.g. a name
   * shortened from its own base but revealed together with its source.
   */
  title?: string;
  className?: string;
}

/**
 * A truncating span that reveals its full text on hover, and only then.
 *
 * The rule and the measurement both live in `useTruncationTitle` (see there for
 * why the title is conditional at all); this is that hook packaged for the
 * places that render a bare name, and for lists, where a hook can't be called
 * per row. Where an element with the right classes already exists, spread the
 * hook onto it rather than wrapping it in this.
 */
export const TruncateWithTitle: React.FC<TruncateWithTitleProps> = ({
  text,
  full,
  title,
  className,
}) => {
  const complete = full ?? text;
  const hover = useTruncationTitle(title ?? complete, text !== complete);
  return (
    <span className={cn("truncate", className)} {...hover}>
      {text}
    </span>
  );
};

export default TruncateWithTitle;
