import { useCallback, useState } from "react";

/**
 * Reads a boolean "seen" flag from localStorage once on mount, and exposes a
 * setter that persists it. Mirrors the try/catch-wrapped pattern in
 * useUpdateCheck.ts's DISMISS_KEY — storage errors (private mode, quota) fail
 * silently rather than crashing the onboarding flow.
 */
export function useOnceFlag(key: string): {
  seen: boolean;
  markSeen: () => void;
} {
  const [seen, setSeen] = useState<boolean>(() => {
    try {
      return localStorage.getItem(key) === "1";
    } catch {
      return false;
    }
  });

  const markSeen = useCallback(() => {
    try {
      localStorage.setItem(key, "1");
    } catch {
      /* localStorage unavailable — the flag just won't persist across reloads */
    }
    setSeen(true);
  }, [key]);

  return { seen, markSeen };
}
