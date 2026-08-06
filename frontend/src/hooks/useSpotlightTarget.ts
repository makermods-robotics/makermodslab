import { useEffect, useState } from "react";

export interface SpotlightRect {
  top: number;
  left: number;
  width: number;
  height: number;
}

function measure(el: Element): SpotlightRect {
  const r = el.getBoundingClientRect();
  return { top: r.top, left: r.left, width: r.width, height: r.height };
}

/**
 * Resolves a CSS selector to its live position, re-measuring on resize/scroll
 * and whenever the DOM changes (the target may not exist yet on first paint —
 * e.g. a Studio panel still loading its data). Returns null while the target
 * can't be found or has zero size, so callers can auto-advance past a step
 * whose element isn't actually visible.
 */
export function useSpotlightTarget(selector: string): SpotlightRect | null {
  const [rect, setRect] = useState<SpotlightRect | null>(null);

  useEffect(() => {
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
