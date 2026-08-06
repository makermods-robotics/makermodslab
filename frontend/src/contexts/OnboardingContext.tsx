import React, {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
} from "react";
import type { Tour } from "@/lib/onboarding/types";

interface OnboardingContextValue {
  /** The tour currently being shown, or null when no tour is active. */
  activeTour: Tour | null;
  /** Index into activeTour.steps of the step currently shown. */
  stepIndex: number;
  /** Starts `tour` from its first step. `onDone` fires exactly once, when the
   * tour finishes (advance() past the last step) or is skipped — callers pass
   * their own useOnceFlag(...).markSeen so this context never touches
   * localStorage directly. */
  start: (tour: Tour, onDone: () => void) => void;
  advance: () => void;
  back: () => void;
  skip: () => void;
}

const OnboardingContext = createContext<OnboardingContextValue | null>(null);

export const OnboardingProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [tour, setTour] = useState<Tour | null>(null);
  const [stepIndex, setStepIndex] = useState(0);
  const onDoneRef = useRef<() => void>(() => {});

  const finish = useCallback(() => {
    onDoneRef.current();
    setTour(null);
    setStepIndex(0);
  }, []);

  const start = useCallback((nextTour: Tour, onDone: () => void) => {
    onDoneRef.current = onDone;
    setTour(nextTour);
    setStepIndex(0);
  }, []);

  const advance = useCallback(() => {
    if (!tour) return;
    if (stepIndex + 1 >= tour.steps.length) {
      finish();
    } else {
      setStepIndex(stepIndex + 1);
    }
  }, [tour, stepIndex, finish]);

  const back = useCallback(() => {
    setStepIndex((i) => Math.max(0, i - 1));
  }, []);

  const skip = useCallback(() => {
    if (!tour) return;
    finish();
  }, [tour, finish]);

  const value = useMemo(
    () => ({ activeTour: tour, stepIndex, start, advance, back, skip }),
    [tour, stepIndex, start, advance, back, skip],
  );

  return (
    <OnboardingContext.Provider value={value}>
      {children}
    </OnboardingContext.Provider>
  );
};

export function useOnboarding(): OnboardingContextValue {
  const ctx = useContext(OnboardingContext);
  if (!ctx) {
    throw new Error("useOnboarding must be used within OnboardingProvider");
  }
  return ctx;
}
