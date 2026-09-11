import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useApi } from "@/contexts/ApiContext";
import { ArmsContext } from "@/hooks/useArms";
import { fetchArmFamilies, type ArmFamilyInfo } from "@/lib/armsApi";

/** Backoff after a failed manifest fetch: 2s, 4s, 8s, then every 15s until
 * one succeeds. Bounded so a server that is down for a while is polled, not
 * hammered, and a --dev backend that comes up late is caught within 15s. */
const RETRY_DELAYS_MS = [2000, 4000, 8000, 15000];

/**
 * One owner for the arms manifest, app-wide.
 *
 * The manifest is what every arm-aware surface reads its capabilities from —
 * the robot corner (labels), the create dialog (the option list), the config
 * dialog (calibration flow, port detection, the servo-register UI), the
 * teleop page (URDF vs numeric readout), and the merge / train / deploy /
 * inference surfaces (joint widths and cross-arm warnings): eight-odd
 * consumers of one small, effectively static list. Fetching it once here and
 * handing out the same array is what keeps them agreeing, and what keeps a
 * still-loading page cheap: one request, not one per consumer. It is a
 * context rather than a module-level store (useRobots' shape) because it
 * needs the ApiContext's baseUrl and has no cross-instance mutations to share.
 *
 * A failed fetch keeps the last good manifest (an empty one on first load) and
 * reports through `error`; it never yields an empty list as if the server had
 * said "no arms" — that would render every robot as unavailable.
 */
export const ArmsProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [arms, setArms] = useState<ArmFamilyInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Retry bookkeeping. A manifest that never arrives is worse than a slow
  // one: every predicate then answers the SO-101 shape for a CAN record —
  // Detect on the Feetech endpoint, the sweep UI, an unsuffixed calibration
  // name landing in the SHARED Star-leader directory. So a failed fetch (the
  // backend still starting under --dev, a restart with the tab open) retries
  // on a bounded backoff until one succeeds; the dialog disables its actions
  // meanwhile. Refs, not state: the schedule must not re-create `refresh`.
  const failuresRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unmountedRef = useRef(false);

  const refresh = useCallback(async () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    try {
      const next = await fetchArmFamilies(baseUrl, fetchWithHeaders);
      if (unmountedRef.current) return;
      failuresRef.current = 0;
      setArms(next);
      setError(null);
    } catch (e) {
      if (unmountedRef.current) return;
      setError(e instanceof Error ? e.message : String(e));
      failuresRef.current += 1;
      const delay = RETRY_DELAYS_MS[
        Math.min(failuresRef.current - 1, RETRY_DELAYS_MS.length - 1)
      ];
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        void refresh();
      }, delay);
    } finally {
      if (!unmountedRef.current) setLoading(false);
    }
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    unmountedRef.current = false;
    refresh();
    return () => {
      unmountedRef.current = true;
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = null;
    };
  }, [refresh]);

  const byId = useCallback(
    (id: string | undefined) =>
      id === undefined ? undefined : arms.find((a) => a.id === id),
    [arms],
  );

  const value = useMemo(
    () => ({ arms, byId, loading, error, refresh }),
    [arms, byId, loading, error, refresh],
  );

  return <ArmsContext.Provider value={value}>{children}</ArmsContext.Provider>;
};
