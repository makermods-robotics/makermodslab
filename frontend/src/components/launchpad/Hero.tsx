import React, { useEffect, useRef, useState } from "react";
import { Search } from "lucide-react";
import BrandMark from "@/components/BrandMark";

export interface HeroProps {
  search: string;
  onSearchChange: (value: string) => void;
}

const WORDS = ["Run", "Train", "Share"] as const;
const HOLD_MS = 1800;
const FADE_MS = 190;

/** True when the OS asks for reduced motion — we then hold a static word. */
const prefersReducedMotion = (): boolean =>
  typeof window !== "undefined" &&
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/**
 * Launchpad hero — brand block (MakerMods mark + MakerMods Lab), the cycling
 * "<word> robot skills" slogan (Run → Train → Share, ~1.8s each, ~190ms opacity
 * fade; static when the user prefers reduced motion), and the search box that
 * live-filters the slider below.
 */
const Hero: React.FC<HeroProps> = ({ search, onSearchChange }) => {
  const [index, setIndex] = useState(0);
  const [visible, setVisible] = useState(true);
  const reduced = useRef(prefersReducedMotion());

  useEffect(() => {
    if (reduced.current) return;
    let fadeTimer: number;
    const holdTimer = window.setInterval(() => {
      setVisible(false);
      fadeTimer = window.setTimeout(() => {
        setIndex((i) => (i + 1) % WORDS.length);
        setVisible(true);
      }, FADE_MS);
    }, HOLD_MS);
    return () => {
      window.clearInterval(holdTimer);
      window.clearTimeout(fadeTimer);
    };
  }, []);

  return (
    <div className="flex w-full flex-col items-center gap-8">
      <BrandMark size="lg" />

      <h1 className="text-center font-display text-4xl font-semibold tracking-tight sm:text-5xl">
        {/* All words share one grid cell so the slot is as wide as the widest
            word — "robot skills" never shifts as the words cycle. */}
        <span className="inline-grid text-right align-bottom" aria-live="polite">
          {WORDS.map((word, i) => (
            <span
              key={word}
              className="col-start-1 row-start-1 transition-opacity"
              style={{
                opacity: i === index && visible ? 1 : 0,
                transitionDuration: `${FADE_MS}ms`,
              }}
              aria-hidden={i !== index}
            >
              {word}
            </span>
          ))}
        </span>{" "}
        robot skills
      </h1>

      <label
        data-tour="launchpad-search"
        className="flex w-full max-w-xl items-center gap-2 rounded-full border border-border bg-card px-4 py-2.5 shadow-1 focus-within:border-ring"
      >
        <Search className="h-4 w-4 shrink-0 text-muted-foreground" />
        <input
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Clean my desk…"
          className="w-full bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          aria-label="Search skills"
        />
      </label>
    </div>
  );
};

export default Hero;
