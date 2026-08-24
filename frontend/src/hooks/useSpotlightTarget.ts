import { useLayoutEffect, useRef, useState } from "react";

export interface SpotlightRect {
  top: number;
  left: number;
  width: number;
  height: number;
  /** The target's own border-radius, so the cutout/ring match its actual
   * shape instead of assuming every target looks like a rounded button (a
   * hardcoded radius leaves visible corner gaps against square-cornered
   * targets like the Studio panels). */
  radius: string;
}

function measure(el: Element): SpotlightRect {
  const r = el.getBoundingClientRect();
  const radius = getComputedStyle(el).borderRadius || "0px";
  return { top: r.top, left: r.left, width: r.width, height: r.height, radius };
}

function measureSelector(selector: string): SpotlightRect | null {
  const el = document.querySelector(selector);
  if (!el) return null;
  const next = measure(el);
  return next.width > 0 && next.height > 0 ? next : null;
}

/**
 * Resolves a CSS selector to its live position, re-measuring on resize/scroll
 * and whenever the DOM changes (the target may not exist yet on first paint —
 * e.g. a Studio panel still loading its data). Returns null while the target
 * can't be found or has zero size, so callers can auto-advance past a step
 * whose element isn't actually visible.
 */
export function useSpotlightTarget(selector: string): SpotlightRect | null {
  const [rect, setRect] = useState<SpotlightRect | null>(() =>
    measureSelector(selector),
  );
  const lastSelector = useRef(selector);

  // The `rect` state belongs to `lastSelector`, not necessarily to this
  // render's `selector` — the effect below only re-measures for a new
  // selector *after* this render commits. Without this reset, a caller like
  // Spotlight's auto-skip effect would see the previous step's rect (or
  // null) on the render right after the selector changes, wrongly judging
  // the new step's target as visible or missing based on stale data. Resetting
  // synchronously during render (React's documented pattern for clearing
  // state on a prop change) means this render already reflects the new
  // selector, with no stale value ever observable.
  if (lastSelector.current !== selector) {
    lastSelector.current = selector;
    setRect(measureSelector(selector));
  }

  useLayoutEffect(() => {
    let frame: number | null = null;
    let observedElement: Element | null = null;
    let resizeObserver: ResizeObserver;

    const measureNow = () => {
      const el = document.querySelector(selector);
      if (!el) {
        setRect(null);
        // Detach ResizeObserver if target disappeared
        if (observedElement) {
          resizeObserver.unobserve(observedElement);
          observedElement = null;
        }
        return;
      }
      // Lazily attach ResizeObserver when we first find the target
      if (observedElement !== el) {
        if (observedElement) {
          resizeObserver.unobserve(observedElement);
        }
        observedElement = el;
        resizeObserver.observe(el);
      }
      const next = measure(el);
      setRect(next.width > 0 && next.height > 0 ? next : null);
    };

    const scheduleMeasure = () => {
      if (frame != null) return;
      frame = requestAnimationFrame(() => {
        frame = null;
        measureNow();
      });
    };

    resizeObserver = new ResizeObserver(scheduleMeasure);

    measureNow();

    const mutationObserver = new MutationObserver(scheduleMeasure);
    mutationObserver.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
    });

    window.addEventListener("resize", scheduleMeasure);
    window.addEventListener("scroll", scheduleMeasure, true);

    return () => {
      if (frame != null) cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      mutationObserver.disconnect();
      window.removeEventListener("resize", scheduleMeasure);
      window.removeEventListener("scroll", scheduleMeasure, true);
    };
  }, [selector]);

  return rect;
}
