import { useModelsData } from "@/contexts/ModelsDataContext";

/** The `/skills` listing — deployable trained policies, plus the Hub's
 * reachability.
 *
 * One definition of "skill", served by one endpoint, read by both surfaces that
 * show skills: the Deploy panel's picker and the models library. They used to
 * answer the question from different sources — `/models` filtered by
 * usable-checkpoint on one side, the `/jobs` registry filtered by runner type
 * on the other — which is why they disagreed about what existed. */
export const useSkills = () => {
  const { skills, hub, loading, error, refresh } = useModelsData();
  return { skills, hub, loading, error, refresh };
};
