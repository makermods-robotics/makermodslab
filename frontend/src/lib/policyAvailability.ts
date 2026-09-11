// Which policy types the backend's pinned lerobot can actually construct
// (GET /policy-optimizer-defaults -> `available`). Types mapping to false
// would fail at policy construction the moment training starts, so the
// landing page disables their buttons.
//
// Cached at module level: the answer can only change with a backend
// restart/upgrade, so one fetch per page load is plenty and remounting the
// landing page never re-polls.

export type PolicyAvailability = Record<string, boolean>;

let cached: PolicyAvailability | null = null;
let inflight: Promise<PolicyAvailability> | null = null;

export function fetchPolicyAvailability(
  baseUrl: string,
  fetchWithHeaders: (url: string, options?: RequestInit) => Promise<Response>,
): Promise<PolicyAvailability> {
  if (cached) return Promise.resolve(cached);
  if (!inflight) {
    inflight = (async () => {
      const r = await fetchWithHeaders(`${baseUrl}/api/v1/policy-optimizer-defaults`);
      if (!r.ok) throw new Error(`policy-optimizer-defaults: HTTP ${r.status}`);
      const data: { available?: PolicyAvailability } = await r.json();
      cached = data.available ?? {};
      return cached;
    })().catch((e) => {
      // Failed fetch must not poison the cache — allow a retry on next mount.
      inflight = null;
      throw e;
    });
  }
  return inflight;
}
