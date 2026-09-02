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
/**
 * What a coaching session needs to know to hand the operator onward when it
 * ends: which skill was being coached, and what it was trained on.
 *
 * Corrections are worth nothing on their own. They have to be merged with the
 * demonstrations the policy was trained on and the policy fine-tuned on the
 * result — `TrainPanel` takes exactly one dataset, so merging is mandatory, not
 * an optimisation. Until now the end-of-session summary explained that in prose
 * and offered no buttons, which put the entire payoff of the feature behind a
 * manual chore the UI did not help with.
 *
 * Carried from the launch site (DeployPanel knows the job it launched from)
 * rather than resolved server-side, because it is display and navigation state,
 * not session state. The cost is that it does not survive a page reload: the
 * corrections are still on disk and still mergeable by hand, but the one-click
 * path is gone. Worth it to avoid teaching the rollout module about the job
 * registry.
 */
export interface CoachingLineage {
  /** Job registry id of the skill being coached — the fine-tune base. */
  jobId: string;
  /** Display name for that skill, for the summary's own copy. */
  jobName?: string;
  /** The dataset the skill was trained on: the other half of the merge. */
  trainingDatasetRepoId?: string;
}

interface InferenceSessionContextValue {
  /** `sessionId` is the identity POST /api/v1/sessions returned — the dialog
   * heartbeats its lease and stops it by id. `lineage` is coaching-only: what
   * the summary needs to offer the merge and the fine-tune. */
  openInferenceSession: (
    sessionId: string,
    lineage?: CoachingLineage | null,
  ) => void;
  sessionOpen: boolean;
}

const InferenceSessionContext =
  createContext<InferenceSessionContextValue | null>(null);

export const InferenceSessionProvider: React.FC<{
  children: React.ReactNode;
}> = ({ children }) => {
  const [sessionOpen, setSessionOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [lineage, setLineage] = useState<CoachingLineage | null>(null);

  const openInferenceSession = useCallback(
    (id: string, next?: CoachingLineage | null) => {
      setSessionId(id);
      setLineage(next ?? null);
      setSessionOpen(true);
    },
    [],
  );
  // Deliberately NOT cleared on exit: the summary that uses it is rendered by
  // the dialog itself, and clearing here would blank the handoff at the exact
  // moment the operator reaches for it.
  const handleExit = useCallback(() => setSessionOpen(false), []);

  const value = useMemo(
    () => ({ openInferenceSession, sessionOpen }),
    [openInferenceSession, sessionOpen],
  );

  return (
    <InferenceSessionContext.Provider value={value}>
      {children}
      {sessionOpen ? (
        <InferenceSessionDialog
          sessionId={sessionId}
          onExit={handleExit}
          coachingLineage={lineage}
        />
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
