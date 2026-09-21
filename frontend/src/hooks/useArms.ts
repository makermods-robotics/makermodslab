import { createContext, useContext } from "react";
import type { ArmFamilyInfo } from "@/lib/armsApi";

export interface ArmsValue {
  /** The GET /api/v1/arms manifest, in the order served (default family
   * first). Empty until the first fetch resolves — consumers render the
   * SO-101 fallback shape through lib/armTypes' predicates meanwhile. */
  arms: ArmFamilyInfo[];
  /** The manifest entry for a record's `arm_type`, or undefined when the
   * manifest has not loaded yet OR the id is not installed (the record's
   * `arm_available` is false). Callers gate the second case on the record. */
  byId: (id: string | undefined) => ArmFamilyInfo | undefined;
  /** True only while the first fetch is in flight — a refresh over a manifest
   * we already hold does not blank every arm back to the fallback shape. */
  loading: boolean;
  /** Set when the LAST fetch failed, whatever `arms` holds. */
  error: string | null;
  refresh: () => Promise<void>;
}

/** The context object lives here, beside its hook, rather than in
 * contexts/ArmsContext.tsx: that file then exports only the provider
 * component, which is what keeps Vite's fast refresh working for it. */
export const ArmsContext = createContext<ArmsValue | null>(null);

/** The GET /api/v1/arms manifest — `arms`, `byId`, `loading`, `error`,
 * `refresh`. A thin read of `ArmsProvider`, which owns the single fetch;
 * pair `byId(record.arm_type)` with the predicates in lib/armTypes. */
export function useArms(): ArmsValue {
  const ctx = useContext(ArmsContext);
  if (!ctx) throw new Error("useArms must be used within ArmsProvider");
  return ctx;
}
