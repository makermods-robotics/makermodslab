import { useEffect } from "react";

/**
 * Courtesy browser-unload confirm while a live session is on screen: closing
 * the tab / reloading raises the native "Leave site?" prompt so an accidental
 * ⌘W doesn't silently walk away from a running robot.
 *
 * Deliberately NOTHING more — no stop beacon, no popstate sentinel, no
 * unmount-stop. Those belonged to the retired useSessionExitGuard era, when
 * the browser was the only thing standing between an abandoned page and an
 * arm that kept driving. Sessions now carry a server-side lease
 * (useSessionHeartbeat): if the page really goes away, the missed heartbeats
 * make the SERVER stop the session — authoritative, and immune to the beacon
 * crossfire where one tab's unload used to kill another tab's session.
 *
 * (The prompt's text is the browser's own; custom copy is ignored by modern
 * browsers, so there is nothing to localize.)
 */
export function useUnloadWarning(active: boolean): void {
  useEffect(() => {
    if (!active) return;
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      // Legacy requirement for the native prompt in some browsers.
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [active]);
}
