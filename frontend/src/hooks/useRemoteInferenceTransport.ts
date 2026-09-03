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
 * Two transports arrive through one shape. When the Lab was started with
 * `--sfu` it hosts the LiveKit server itself and mints the url, the room and
 * the robot child's token in-process; otherwise it falls back to LiveKit Cloud
 * credentials in `livekit.env`. `sfu_enabled` selects which, and the whole
 * `sfu_*` block is null/false in the Cloud case.
 *
 * There is NO mutation on this surface any more. `clearLocalOverride()` went
 * with the shell SFU scripts in S3.6 — the dotenv override it deleted was
 * written by a script that no longer exists.
 */

/** Which layer supplied LIVEKIT_URL. `sfu` means the Lab's own server, where
 * nothing on disk is consulted at all; the other three are the LiveKit Cloud
 * fallback, and they are three different remedies ("your shell exported
 * LIVEKIT_URL" and "livekit.env says so" are not the same problem). */
export type TransportSource = "sfu" | "cloud" | "process_env" | "none";

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
  /** Whether this Lab process runs its own LiveKit server (`--sfu`). Everything
   * below is null/false when it does not, so the panel renders the block from
   * this one flag. */
  sfu_enabled: boolean;
  /** The loopback signalling URL a local child dials. Data. */
  sfu_url: string | null;
  /** The URL a MODAL container should dial: ws://<tailnet ipv4>:7880. Null when
   * tailscale is absent or not logged in — a container has no route to loopback
   * or to a LAN address, so there is nothing honest to offer then. Data. */
  sfu_modal_url: string | null;
  /** Whether the SFU advertises its STUN-discovered public IP (the launcher's
   * --sfu-external-ip). Without it a container can reach signalling over the
   * tailnet and still never punch a media path. */
  sfu_external_ip: boolean;
  /** The key's NAME — the `--livekit-api-key` half of the Modal line. It
   * identifies rather than authorizes. THE SECRET IS NEVER RETURNED. Data. */
  sfu_key_id: string | null;
  /** Where the secret actually lives, for a human to read. Data. */
  sfu_key_file: string | null;
  /** The per-OS install line, present ONLY when `livekit-server` is missing
   * from PATH. Backend prose — shown verbatim. */
  sfu_install_hint: string | null;
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

  return { transport, loading, error, refresh };
}
