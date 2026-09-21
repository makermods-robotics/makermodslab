import { useHfAuth } from "@/contexts/HfAuthContext";

/** True when the logged-in user can push to their own namespace — the gate
 * for offering a Hub upload affordance on a local model. Mirrors
 * DatasetInfoCard's useCanEditHub for a bare (own-namespace) target: false
 * while loading / unauthenticated. Shared by ModelInfoCard's Launchpad
 * upload button and TrainingJobDialog's Publish-to-Hub row. */
export const useCanUpload = (): boolean => {
  const { auth } = useHfAuth();
  if (auth.status !== "authenticated") return false;
  return auth.username != null && auth.writableNamespaces.length > 0;
};
