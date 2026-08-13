import { cn } from "@/lib/utils";
import { displayDedupeSuffix, splitDedupeSuffix } from "@/lib/modelNames";
import { useTruncationTitle } from "@/hooks/useTruncationTitle";

interface Props {
  /** The display name as the backend resolved it, collision suffix included. */
  name: string;
  /**
   * What that name MEANS, when the caller is rendering a name it already
   * shortened itself — a jobs row peeling a generated `"{POLICY} · {ns}/{task}"`
   * down to its task (`runTaskTitle`). The hover then reveals the untouched
   * name instead of echoing the peeled one, and it is the peel, not the DOM,
   * that earns the tooltip: nothing is clipped, yet what's on screen is not the
   * whole name — the same kind of invisible shortening as the dropped year
   * below. Omit it (the normal case) and the name is its own meaning.
   */
  full?: string;
  className?: string;
}

/**
 * A model/run title line whose collision suffix survives truncation.
 *
 * The backend hands us names like `eraser_place (2026-08-02)`, where the
 * parenthesized tail is the ONLY thing separating that row from the one above
 * it (see `dedupe_display_names`). A single `truncate` span eats the tail
 * first, so in the narrow places these titles live — a card header, a select
 * item — the disambiguator is exactly what disappears, and two rows the backend
 * went to some trouble to distinguish arrive at the user identical, or worse,
 * distinguished only by a stump like "(2026-0…" that no longer says which.
 *
 * So the two parts get their own spans: the base truncates, the suffix does
 * not. The suffix is short and bounded by construction (a date, a date and
 * time, or a small ordinal), so protecting it costs a fixed sliver of the line
 * and the base absorbs the rest. When the base IS clipped, the full name is one
 * hover away — and only then, since a title on a name that already fits whole
 * just repeats it (see `useTruncationTitle`).
 *
 * The one thing the suffix does give up is its year (`displayDedupeSuffix`),
 * which buys the base ~5 characters at no cost to the fields that actually
 * separate two same-day runs. Two runs exactly a year apart do read identically
 * until hovered — that is the trade, and the hover title is what makes it
 * survivable.
 */
const DisplayName: React.FC<Props> = ({ name, full, className }) => {
  const { base, suffix } = splitDedupeSuffix(name);
  // Rendered a year shorter than it is stored, so the two spans below no longer
  // spell the full name between them.
  const displaySuffix = suffix === null ? null : displayDedupeSuffix(suffix);
  // The title hangs on the CONTAINER, and the hook sweeps its descendants —
  // the pair is one name, so either span clipping means the visible name is
  // incomplete, and the container itself never overflows (its flex children
  // shrink instead), which measuring it alone would misread as "fits".
  //
  // The dropped year is the OTHER kind of shortening, the one the DOM can't
  // see: nothing is clipped, yet what's on screen is not the whole name. Only
  // this component knows, so it says so — and the comparison stays true however
  // the suffix's display form changes later. A caller-side peel (`full`) is the
  // same case one layer up, so it feeds the same flag.
  const peeled = full != null && full !== name;
  const hover = useTruncationTitle(
    full ?? name,
    displaySuffix !== suffix || peeled,
  );
  // A flex row in both cases: `truncate` needs a block-level box to clip
  // against, and keeping one shape means the suffixed and unsuffixed titles
  // share a baseline wherever they sit side by side in a grid.
  return (
    <div className={cn("flex min-w-0 items-baseline", className)} {...hover}>
      {/* A floor on the base, because the suffix beside it can't give: with
          none, the base starves to "e…" next to a whole timestamp. Because that
          suffix can't give either, the floor is also what decides whether the
          pair OVERFLOWS, so it is sized against the narrowest line these titles
          sit on: 6ch (~53px) + a non-breaking space + a whole "(07-31 17:35)"
          (~98px at the inherited 16px semibold) ≈ 155px, well inside the ~239px
          interior of CappedGrid's documented 263px card. Squeezed below that,
          the pair overflows rather than shrinking further — deliberate, since
          what it would shrink is the disambiguator. */}
      <span className="min-w-[6ch] truncate">{base}</span>
      {/* shrink-0, and no `truncate`: the base gives up its characters first,
          the disambiguator never does. Letting it ellipsize is the failure this
          element exists to prevent — "eraser_plac… (2026-0…", where the thing
          cut off is the only thing telling two cards apart. */}
      {displaySuffix === null ? null : (
        <span className="shrink-0 whitespace-nowrap">&nbsp;({displaySuffix})</span>
      )}
    </div>
  );
};

export default DisplayName;
