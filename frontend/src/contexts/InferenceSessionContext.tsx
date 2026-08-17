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
 * InferenceModal) call `openInferenceSession()` right after POST
 * /inference/start succeeds; the dialog then owns status polling, the stop
 * flow, and the exit guard, and closing it lands back on whatever surface
 * launched the run.
 */
interface InferenceSessionContextValue {
  openInferenceSession: () => void;
  sessionOpen: boolean;
}

const InferenceSessionContext =
  createContext<InferenceSessionContextValue | null>(null);

export const InferenceSessionProvider: React.FC<{
  children: React.ReactNode;
}> = ({ children }) => {
  const [sessionOpen, setSessionOpen] = useState(false);

  const openInferenceSession = useCallback(() => setSessionOpen(true), []);
  const handleExit = useCallback(() => setSessionOpen(false), []);

  const value = useMemo(
    () => ({ openInferenceSession, sessionOpen }),
    [openInferenceSession, sessionOpen],
  );

  return (
    <InferenceSessionContext.Provider value={value}>
      {children}
      {sessionOpen ? <InferenceSessionDialog onExit={handleExit} /> : null}
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
