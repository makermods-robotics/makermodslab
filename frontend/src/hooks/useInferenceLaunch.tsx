import { useCallback } from "react";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { ApiError } from "@/lib/apiClient";
import { JobRecord, importModel } from "@/lib/jobsApi";

/**
 * The Jobs cards' LAZY AUTO-IMPORT for a model that isn't a tracked job yet:
 * registers the repo id / local path as an imported pseudo-job (idempotent — a
 * re-import returns the existing record) with husk-repo messaging (a cloud run
 * that died before its first checkpoint save 400s and gets the plain "no
 * checkpoints" answer rather than a broken launch). Returns null on failure,
 * having already toasted.
 *
 * This hook used to also own `play` and `modal`, which opened the legacy
 * InferenceModal on a checkpoint. DeployPanel replaced that surface — it was
 * ported from the modal and every caller now routes through the studio's deploy
 * prefill — leaving `play` and `modal` with no consumers and the modal itself
 * never mounted. Both, and the modal, are gone; only the import path survived,
 * and it stays a hook because it needs the api client and the toaster.
 */
export const useInferenceLaunch = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

  const importSource = useCallback(
    async (source: string): Promise<JobRecord | null> => {
      try {
        return await importModel(baseUrl, fetchWithHeaders, source);
      } catch (e) {
        const isHusk =
          e instanceof ApiError && (e.status === 400 || e.status === 404);
        toast({
          title: isHusk ? "No checkpoints in this repo" : "Import failed",
          description: isHusk
            ? "The run likely died before its first checkpoint save."
            : e instanceof Error
              ? e.message
              : String(e),
          variant: "destructive",
        });
        return null;
      }
    },
    [baseUrl, fetchWithHeaders, toast],
  );

  return { importSource };
};

export default useInferenceLaunch;
