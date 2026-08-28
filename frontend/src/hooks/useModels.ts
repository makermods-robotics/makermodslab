import { useModelsData } from "@/contexts/ModelsDataContext";

/** The merged `/models` listing (local runs + imports + Hub repos).
 *
 * A thin read of `ModelsDataProvider`, kept as a hook because every consumer
 * already imports it under this name. It used to own a fetch per caller, which
 * is exactly how two surfaces reading one endpoint ended up showing different
 * listings — the provider now owns the single fetch, its cache, and its
 * `jobs_changed` subscription. `refresh` stays for the callers that want to
 * re-pull after a mutation of their own. */
export const useModels = () => useModelsData();
