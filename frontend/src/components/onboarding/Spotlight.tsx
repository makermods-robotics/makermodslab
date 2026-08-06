import React, { useEffect, useMemo, useRef } from "react";
import * as Popover from "@radix-ui/react-popover";
import { useOnboarding } from "@/contexts/OnboardingContext";
import { useSpotlightTarget } from "@/hooks/useSpotlightTarget";
import { Button } from "@/components/ui/button";
import type { Placement } from "@/lib/onboarding/types";

const prefersReducedMotion = (): boolean =>
  typeof window !== "undefined" &&
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

const PLACEMENT_TO_SIDE: Record<
  Placement,
  "top" | "bottom" | "left" | "right"
> = { top: "top", bottom: "bottom", left: "left", right: "right" };

/**
 * Renders the current step of the active tour: a backdrop with a cutout over
 * the target (a huge box-shadow, not a DOM hole), a pulsing ring, and a
 * Popover callout anchored to the target's live rect via a virtual element.
 * Renders nothing when there's no active tour; auto-advances past a step
 * whose target can't be measured (unmounted, zero-size, or not on this page).
 */
const Spotlight: React.FC = () => {
  const { activeTour, stepIndex, advance, back, skip } = useOnboarding();
  const step = activeTour?.steps[stepIndex] ?? null;
  const rect = useSpotlightTarget(step?.target ?? "[data-tour=__none__]");

  // Auto-skip a step whose target isn't actually on screen right now.
  useEffect(() => {
    if (step && !rect) advance();
  }, [step, rect, advance]);

  useEffect(() => {
    if (!step) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented) return;
      const target = e.target as HTMLElement | null;
      // Same guard as StudioOverlay's Escape handler — don't steal Escape
      // from a dialog layered above the tour.
      if (target?.closest('[role="dialog"], [role="alertdialog"]')) return;
      skip();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, skip]);

  useEffect(() => {
    if (!step || !rect) return;
    const el = document.querySelector(step.target);
    if (!(el instanceof HTMLElement)) return;
    const bounds = el.getBoundingClientRect();
    if (bounds.bottom < 0 || bounds.top > window.innerHeight) {
      el.scrollIntoView({
        block: "center",
        behavior: prefersReducedMotion() ? "auto" : "smooth",
      });
    }
  }, [rect, step]);

  // A virtual Popper anchor whose getBoundingClientRect reflects the live
  // measured rect. Radix's virtualRef type wants a RefObject<Measurable |
  // null>; useRef gives us a stable object identity across renders while the
  // rect it reads stays live via the mutable `.current` reassignment below.
  const virtualAnchorRef = useRef<{ getBoundingClientRect: () => DOMRect }>({
    getBoundingClientRect: () =>
      ({
        x: 0,
        y: 0,
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        width: 0,
        height: 0,
        toJSON() {
          return this;
        },
      }) as DOMRect,
  });

  const virtualAnchor = useMemo(() => {
    if (!rect) return virtualAnchorRef.current;
    virtualAnchorRef.current = {
      getBoundingClientRect: () =>
        ({
          x: rect.left,
          y: rect.top,
          top: rect.top,
          left: rect.left,
          right: rect.left + rect.width,
          bottom: rect.top + rect.height,
          width: rect.width,
          height: rect.height,
          toJSON() {
            return this;
          },
        }) as DOMRect,
    };
    return virtualAnchorRef.current;
  }, [rect]);

  if (!step || !rect || !activeTour) return null;

  const isLast = stepIndex === activeTour.steps.length - 1;

  return (
    <div role="dialog" aria-label="Feature tour" className="fixed inset-0 z-[60]">
      {/* Cutout backdrop: a huge box-shadow punches a hole over the target
          rect instead of an SVG mask. Not interactive — clicks pass through
          to the app so the tour never blocks normal use. */}
      <div
        aria-hidden
        className="pointer-events-none fixed rounded-md transition-all duration-300"
        style={{
          top: rect.top - 4,
          left: rect.left - 4,
          width: rect.width + 8,
          height: rect.height + 8,
          boxShadow: "0 0 0 9999px rgba(0, 0, 0, 0.5)",
        }}
      />
      {/* Pulsing ring — a separate layer so the grow/shrink animation never
          distorts the cutout's edge. */}
      <div
        aria-hidden
        className="pointer-events-none fixed rounded-md ring-2 ring-ring animate-spotlight-pulse motion-reduce:animate-none"
        style={{
          top: rect.top - 4,
          left: rect.left - 4,
          width: rect.width + 8,
          height: rect.height + 8,
        }}
      />
      <Popover.Root open>
        <Popover.Anchor virtualRef={virtualAnchorRef} />
        <Popover.Portal>
          <Popover.Content
            side={PLACEMENT_TO_SIDE[step.placement ?? "bottom"]}
            sideOffset={12}
            className="z-[61] w-80 rounded-md border border-border bg-popover p-4 text-popover-foreground shadow-2"
            onOpenAutoFocus={(e) => e.preventDefault()}
          >
            <div aria-live="polite">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Step {stepIndex + 1} of {activeTour.steps.length}
              </p>
              <h2 className="mt-1 font-display text-base font-semibold">
                {step.title}
              </h2>
              <p className="mt-1 text-sm text-muted-foreground">
                {step.description}
              </p>
            </div>
            <div className="mt-4 flex items-center justify-between gap-2">
              <Button variant="ghost" size="sm" onClick={skip}>
                Skip
              </Button>
              <div className="flex gap-2">
                {stepIndex > 0 && (
                  <Button variant="outline" size="sm" onClick={back}>
                    Back
                  </Button>
                )}
                <Button size="sm" autoFocus onClick={advance}>
                  {isLast ? "Done" : "Next"}
                </Button>
              </div>
            </div>
          </Popover.Content>
        </Popover.Portal>
      </Popover.Root>
    </div>
  );
};

export default Spotlight;
