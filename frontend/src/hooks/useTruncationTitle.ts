import React, { useCallback, useState } from "react";

/**
 * Conditional hover titles for names that may have been shortened.
 *
 * A name displayed WHOLE needs no tooltip: a native `title` on text that
 * already fits just repeats it under the cursor, and a grid full of those
 * teaches the user that hovering says nothing. A name that was SHORTENED does
 * need one, because the hidden half is usually the identifying one. So the
 * title has to be conditional — and the two ways this app shortens a name need
 * two different detections:
 *
 *   * **CSS `truncate`.** The browser decides at layout time, per card width,
 *     so only the DOM knows: measure `scrollWidth > clientWidth` on the element
 *     that clips. Measured on every hover rather than once at mount, which is
 *     what makes a window resize (or a grid re-flowing 3-up to 2-up) correct
 *     for free — no ResizeObserver, no layout effect, and nothing to keep in
 *     sync with the CSS.
 *   * **JS shortening** (`middleEllipsis`, a peeled namespace, a dedupe suffix
 *     rendered a year shorter than it is stored). The caller already knows this
 *     one, because the string it renders differs from the full one — passed in
 *     as `shortened`.
 *
 * Native `title` rather than a Radix tooltip on purpose: these sit on small,
 * tightly-measured spans inside cards and a Select trigger, and an attribute
 * carries zero layout risk where an extra wrapper element would not.
 *
 * Spread the returned props onto the element the name is IN — the truncating
 * span itself, or, where the name is rendered as a base+suffix pair, their
 * container (`components/library/DisplayName`), which the sweep below measures
 * as one unit.
 */

/**
 * Is `el` — or any of the (few) elements nested inside it — visually clipped?
 *
 * The descendant sweep is what lets the title sit on a CONTAINER whose
 * children are the things that truncate: DisplayName renders a name as a
 * base+suffix span pair, and the container itself never overflows (its flex
 * children shrink instead), so measuring the container alone would always
 * report "fits". Non-HTML descendants (a lucide `<svg>` and its paths) have no
 * scroll/client width and are skipped rather than compared as NaN.
 *
 * The 1px slack absorbs sub-pixel rounding: both widths are integers, so text
 * that fits at a fractional width can read one pixel over. Real truncation
 * overflows by far more than a pixel.
 */
function isClipped(el: HTMLElement): boolean {
  if (el.scrollWidth - el.clientWidth > 1) return true;
  for (const child of el.querySelectorAll("*")) {
    if (
      child instanceof HTMLElement &&
      child.scrollWidth - child.clientWidth > 1
    )
      return true;
  }
  return false;
}

/**
 * Props to spread onto the element the user hovers: the `title` — undefined
 * while the name fits, so no tooltip appears at all — and the handler that
 * re-measures on entry.
 *
 * @param full      what the tooltip should reveal: the complete name, plus
 *                  whatever else identifies the thing (an import's source).
 * @param shortened set when the CALLER already shortened the string it renders
 *                  (middleEllipsis, a dropped namespace, a dropped year); CSS
 *                  truncation is detected here and needs no flag.
 */
export function useTruncationTitle(
  full: string | null | undefined,
  shortened = false,
): {
  title: string | undefined;
  onMouseEnter: React.MouseEventHandler<HTMLElement>;
} {
  const [clipped, setClipped] = useState(false);
  // Setting state on hover re-renders long before the browser's ~1s tooltip
  // delay elapses, so the attribute is in place by the time the tooltip is due.
  const onMouseEnter = useCallback<React.MouseEventHandler<HTMLElement>>(
    (e) => setClipped(isClipped(e.currentTarget)),
    [],
  );
  return {
    title: full && (shortened || clipped) ? full : undefined,
    onMouseEnter,
  };
}

export default useTruncationTitle;
