import { ReactNode } from "react";
import { ActiveSessionContext, useActiveSession } from "@/hooks/useActiveSession";

/**
 * Provides the latest backend `session_changed` event app-wide over ONE
 * shared WebSocket subscription (see hooks/useActiveSession.ts, which also
 * hosts the context object and the `useSessionEvent` consumer hook).
 *
 * Infrastructure only for now: nothing visible consumes it yet — a later
 * change gives the Launchpad a busy indicator. Consumers must treat the
 * event as a refetch hint, never as state.
 */
export const SessionProvider = ({ children }: { children: ReactNode }) => {
  const lastEvent = useActiveSession();
  return <ActiveSessionContext.Provider value={lastEvent}>{children}</ActiveSessionContext.Provider>;
};
