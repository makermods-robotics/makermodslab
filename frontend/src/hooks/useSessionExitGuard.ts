import { useCallback, useEffect, useRef } from "react";

/**
 * Page-leave safety net for CALIBRATION ONLY (RobotConfigDialog's manual
 * calibration flow). Recording, inference and replay no longer use this:
 * their sessions go through /api/v1/sessions and carry a server-side lease
 * (useSessionHeartbeat) that safety-stops an abandoned session — those
 * surfaces keep only a courtesy beforeunload confirm (useUnloadWarning).
 * Calibration is not yet on the sessions surface, has no lease, and still
 * needs the beacon; this hook retires when calibration migrates.
 *
 * While a session is `active`, an unintentional exit must stop / discard the
 * session so the arm can't keep driving (or the backend singleton stay
 * latched) with nobody on the page.
 *
 * It covers every leave vector with one mechanism:
 *  - **browser-level leave** (reload, tab/window close, typed URL): `beforeunload`
 *    raises the browser's native "Leave site?" confirm, and if the page really
 *    goes away `pagehide` fires a best-effort keepalive POST to `beaconUrl` (a
 *    bare "simple request" — no JSON body/header — so it survives unload without
 *    tripping a CORS preflight), optionally stamping `beaconFlagKey` in
 *    sessionStorage for the next page to read.
 *  - **in-app back button** (popstate): a blocking `window.confirm(confirmMessage)`;
 *    cancel keeps you on the page, confirm runs `onLeave` and lets the nav
 *    proceed.
 *  - **any other in-app navigation** (route change → unmount): `onLeave` runs on
 *    cleanup.
 *
 * Per-surface semantics live in the CALLER (what "leaving" does): Recording
 * quits-without-saving (discard), Inference/Calibration stop/abort. The hook is
 * mechanism-only.
 *
 * The caller marks the deliberate exits it handles itself — the explicit
 * Done/Quit/Stop buttons and the natural end-of-session navigation — via the
 * returned `markHandled()`, so the guard doesn't fire a second, spurious
 * stop/discard on the imminent unmount. After a session ends (`active` flips
 * false) the guard disarms and navigation is free.
 */
export interface SessionExitGuardOptions {
  /** True only while a live session should be protected (any phase). */
  active: boolean;
  /** Copy for the in-app back-button confirm and the native unload prompt. */
  confirmMessage: string;
  /**
   * URL the browser-unload beacon POSTs to (keepalive). Include any flag as a
   * query param (e.g. `/stop-recording?discard=true`) so the request stays a
   * simple request. Null disables the beacon.
   */
  beaconUrl: string | null;
  /** Best-effort in-app stop, run on confirmed back / route-change unmount. */
  onLeave?: () => void;
  /** Optional sessionStorage key stamped on unload (so the next page can react). */
  beaconFlagKey?: string;
}

export interface SessionExitGuard {
  /**
   * Mark the exit as already handled by the caller (explicit Done/Quit/Stop, or
   * the natural end-of-session navigation), so the guard won't fire again on the
   * imminent unmount / unload.
   */
  markHandled: () => void;
}

export function useSessionExitGuard({
  active,
  confirmMessage,
  beaconUrl,
  onLeave,
  beaconFlagKey,
}: SessionExitGuardOptions): SessionExitGuard {
  // One-way latch: once the caller (or a leave vector) has handled the exit, no
  // other vector should fire a second stop/discard.
  const handledRef = useRef(false);
  const markHandled = useCallback(() => {
    handledRef.current = true;
  }, []);

  // Keep the latest option values readable from stable event handlers without
  // resubscribing (and re-seeding the popstate sentinel) on every render.
  const onLeaveRef = useRef(onLeave);
  onLeaveRef.current = onLeave;
  const beaconUrlRef = useRef(beaconUrl);
  beaconUrlRef.current = beaconUrl;
  const confirmRef = useRef(confirmMessage);
  confirmRef.current = confirmMessage;
  const flagKeyRef = useRef(beaconFlagKey);
  flagKeyRef.current = beaconFlagKey;

  // Re-arm the one-way latch whenever a NEW session becomes active. Pages that
  // run several sessions without unmounting (e.g. Calibration: calibrate, then
  // calibrate again) would otherwise stay latched-handled from the first exit
  // and never guard the next session.
  useEffect(() => {
    if (active) handledRef.current = false;
  }, [active]);

  // Browser-level leave: native confirm + a keepalive beacon stop on the actual
  // unload. React cleanup never runs on a browser unload, so the beacon is the
  // only stop that fires for tab-close / reload / typed-URL.
  useEffect(() => {
    if (!active) return;

    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      if (handledRef.current) return;
      e.preventDefault();
      // Legacy requirement for the native prompt in some browsers.
      e.returnValue = "";
    };
    const onPageHide = () => {
      if (handledRef.current) return;
      handledRef.current = true;
      const key = flagKeyRef.current;
      if (key) {
        try {
          sessionStorage.setItem(key, "1");
        } catch {
          /* sessionStorage may be unavailable; the beacon below still runs */
        }
      }
      const url = beaconUrlRef.current;
      if (url) {
        // Bare keepalive fetch (no JSON body/header) so it stays a CORS simple
        // request and isn't dropped to a preflight mid-unload.
        fetch(url, { method: "POST", keepalive: true }).catch(() => {});
      }
    };

    window.addEventListener("beforeunload", onBeforeUnload);
    window.addEventListener("pagehide", onPageHide);
    return () => {
      window.removeEventListener("beforeunload", onBeforeUnload);
      window.removeEventListener("pagehide", onPageHide);
    };
  }, [active]);

  // In-app back button (popstate): seed a sentinel history entry while active so
  // the first Back lands on it (not off the page), then confirm on it. Cancel
  // re-seeds the sentinel to stay put; confirm runs onLeave and steps back past
  // the sentinel to actually leave.
  useEffect(() => {
    if (!active) return;

    window.history.pushState({ __sessionGuard: true }, "");
    const onPopState = () => {
      if (handledRef.current) return;
      if (window.confirm(confirmRef.current)) {
        handledRef.current = true;
        onLeaveRef.current?.();
        // Step back past our sentinel to the entry the user actually wanted.
        window.history.back();
      } else {
        // Stay: replace the sentinel the Back just consumed.
        window.history.pushState({ __sessionGuard: true }, "");
      }
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [active]);

  // Any other in-app navigation unmounts this component: stop on cleanup unless
  // the caller already handled the exit (explicit action / natural end) or a
  // leave vector above did.
  const activeRef = useRef(active);
  activeRef.current = active;
  useEffect(() => {
    return () => {
      if (activeRef.current && !handledRef.current) {
        handledRef.current = true;
        onLeaveRef.current?.();
      }
    };
  }, []);

  return { markHandled };
}
