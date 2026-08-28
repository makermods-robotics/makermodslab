import React, {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import InferenceSessionDialog from "@/components/inference/InferenceSessionDialog";

/**
 * Hosts the live-inference session dialog above the router — /inference is no
 * longer a route. Both launch flows (the studio Deploy panel and the legacy
 * InferenceModal) call `openInferenceSession(sessionId)` right after POST
 * /api/v1/sessions succeeds; the dialog then owns status polling, the lease
 * heartbeat, and the stop flow, and closing it lands back on whatever surface
 * launched the run.
 */
interface InferenceSessionContextValue {
  /** `sessionId` is the identity POST /api/v1/sessions returned — the dialog
   * heartbeats its lease and stops it by id. */
  openInferenceSession: (sessionId: string) => void;
  sessionOpen: boolean;
}

const InferenceSessionContext =
  createContext<InferenceSessionContextValue | null>(null);

export const InferenceSessionProvider: React.FC<{
  children: React.ReactNode;
}> = ({ children }) => {
  const [sessionOpen, setSessionOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const openInferenceSession = useCallback((id: string) => {
    setSessionId(id);
    setSessionOpen(true);
  }, []);
  const handleExit = useCallback(() => setSessionOpen(false), []);

  const value = useMemo(
    () => ({ openInferenceSession, sessionOpen }),
    [openInferenceSession, sessionOpen],
  );

  return (
    <InferenceSessionContext.Provider value={value}>
      {children}
      {sessionOpen ? (
        <InferenceSessionDialog sessionId={sessionId} onExit={handleExit} />
      ) : null}
    </InferenceSessionContext.Provider>
  );
};

export const useInferenceSession = (): InferenceSessionContextValue => {
  const ctx = useContext(InferenceSessionContext);
  if (!ctx) {
    throw new Error(
      "useInferenceSession must be used within InferenceSessionProvider",
    );
  }
  return ctx;
};
