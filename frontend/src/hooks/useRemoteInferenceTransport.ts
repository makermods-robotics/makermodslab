import { useCallback, useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { apiRequest } from "@/lib/apiClient";
import type { Fetcher } from "@/lib/apiClient";

/**
 * What transport a remote-inference (DRTC) child would resolve RIGHT NOW, and
 * whether anything is answering on it.
 *
 * Read-only and on demand: `GET /api/v1/remote-inference/transport` touches no
 * hardware and starts nothing, but it does open a short-lived
 * `list_participants` call against the SFU, so it is fetched when the panel
 * asks rather than on a timer.
 *
 * The one mutation on this surface is `clearLocalOverride()`. It deletes
 * `livekit.local.env`, sending the robot side back to LiveKit Cloud — the fix
 * for the documented top footgun of the local-SFU path, where that file
 * OUTLIVES the script that wrote it and the robot keeps dialing a dead
 * ws://127.0.0.1:7880. It is idempotent (an absent file is a 200 with
 * `removed: false`), and it deliberately does NOT touch livekit.local.yaml —
 * deleting that would rotate the local SFU's own credentials.
 */

/** Which layer of the env chain supplied LIVEKIT_URL. Wider than the RUNNING
 * session's `transport.source`, because a pre-launch panel has remedies to
 * offer that a running session does not ("your shell exported LIVEKIT_URL" and
 * "livekit.env says so" are different problems). */
export type TransportSource =
  | "cloud"
  | "local_override"
  | "cwd"
  | "process_env"
  | "none";

export interface RemoteInferenceTransportStatus {
  /** The optional `[drtc]` extra. False ⇒ the probe did not run and the four
   * probe-shaped fields below are null. */
  extra_installed: boolean;
  /** All four LIVEKIT_* vars resolved. */
  configured: boolean;
  /** Empty when `configured`. Variable NAMES — data, never translated. */
  missing_vars: string[];
  /** "" when unresolved — never null. Data. */
  url: string;
  /** Room NAME — data. */
  room: string;
  source: TransportSource;
  sfu_config_exists: boolean;
  local_env_exists: boolean;
  /** Always the path, whether the file exists or not. */
  local_env_path: string;
  /** NULL (not false) when the probe did not run: no extra, or not
   * configured. A third state, and the panel must render it as one. */
  endpoint_reachable: boolean | null;
  operator_present: boolean | null;
  /** The probe's coded failure ("transport.unreachable" /
   * "transport.unauthorized"); null on success and null when it did not run.
   * Backend data — matched on, shown verbatim, never translated. */
  error_code: string | null;
  message: string | null;
}

export interface ClearLocalOverrideResult {
  success: boolean;
  /** False when the file was already absent — not an error. */
  removed: boolean;
  path: string;
}

export async function getRemoteInferenceTransport(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<RemoteInferenceTransportStatus> {
  return apiRequest<RemoteInferenceTransportStatus>(
    baseUrl,
    fetcher,
    "/api/v1/remote-inference/transport",
    { signal, action: "Read remote transport" },
  );
}

export async function clearRemoteLocalOverride(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<ClearLocalOverrideResult> {
  return apiRequest<ClearLocalOverrideResult>(
    baseUrl,
    fetcher,
    "/api/v1/remote-inference/clear-local-override",
    { method: "POST", action: "Clear the local LiveKit override" },
  );
}

/**
 * True when a remote run could actually be launched: the extra is installed,
 * the credentials resolve, the SFU answers, AND an operator (the Modal
 * container) is already in the room.
 *
 * The operator check is not pedantry. Without it the room matches, the arm
 * energizes, and nothing ever drives it — the failure the backend's preflight
 * exists to convert into a refusal BEFORE torque.
 */
export function transportIsReady(
  transport: RemoteInferenceTransportStatus | null,
): boolean {
  if (!transport) return false;
  return (
    transport.extra_installed &&
    transport.configured &&
    transport.endpoint_reachable === true &&
    transport.operator_present === true
  );
}

export interface UseRemoteInferenceTransport {
  transport: RemoteInferenceTransportStatus | null;
  loading: boolean;
  /** The thrown error's own text, or null. Backend prose — shown as raised. */
  error: string | null;
  refresh: () => void;
  clearLocalOverride: () => Promise<ClearLocalOverrideResult>;
}

/**
 * @param enabled fetch while true (and re-fetch on `refresh()`). False leaves
 * the last answer in place rather than blanking it, so collapsing the form
 * does not throw away what the operator just read.
 */
export function useRemoteInferenceTransport(
  enabled: boolean,
): UseRemoteInferenceTransport {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [transport, setTransport] =
    useState<RemoteInferenceTransportStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    setLoading(true);
    getRemoteInferenceTransport(baseUrl, fetchWithHeaders)
      .then((next) => {
        if (cancelled) return;
        setTransport(next);
        setError(null);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [enabled, baseUrl, fetchWithHeaders, nonce]);

  const clearLocalOverride = useCallback(async () => {
    const result = await clearRemoteLocalOverride(baseUrl, fetchWithHeaders);
    // The whole point of clearing the override is that the NEXT resolution
    // differs, so re-read rather than patching the cached answer.
    refresh();
    return result;
  }, [baseUrl, fetchWithHeaders, refresh]);

  return { transport, loading, error, refresh, clearLocalOverride };
}
