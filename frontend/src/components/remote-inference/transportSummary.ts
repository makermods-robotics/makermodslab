import type { RemoteInferenceTransportStatus } from "@/hooks/useRemoteInferenceTransport";

/**
 * The transport read-out as ONE sentence, chosen by the first thing that is
 * wrong — the order is the order an operator has to fix things in.
 *
 * Pure, and returns a translation KEY plus its values rather than prose, for
 * the same reason deployGuards does: the assertions stay about which state was
 * detected, not about wording. The full row-by-row read-out is still there
 * behind "Details"; this is what the Remote tab shows by default.
 *
 * `endpoint_reachable` / `operator_present` are `boolean | null`, and null
 * means THE PROBE DID NOT RUN — a third state, reported as "not checked"
 * rather than collapsed into a failure nothing observed.
 */
export type TransportSummaryTone = "ok" | "warn" | "error" | "muted";

/**
 * The one summary case that carries more than a sentence.
 *
 * "The Lab isn't running a LiveKit server" is the only verdict whose remedy is
 * a command line, and a command is data — it cannot live in a catalog. So the
 * caller renders the literal `makermodslab --sfu --sfu-external-ip` (and the
 * backend's own per-OS install hint, when it sent one) beneath this sentence,
 * and needs to recognise the case to do it. Exported rather than re-derived so
 * the panel and this module cannot disagree about when it fired.
 */
export const SFU_OFF_SUMMARY_KEY = "remoteInference.transport.sfuNotRunning";

export interface TransportSummary {
  key: string;
  /** Interpolations: every value is DATA (a url, a room name, variable names,
   * the backend's own error text) and is shown verbatim. */
  values?: Record<string, string>;
  tone: TransportSummaryTone;
}

export function summarizeTransport(
  transport: RemoteInferenceTransportStatus | null,
  loading: boolean,
  error: string | null,
): TransportSummary {
  if (error) {
    return {
      key: "remoteInference.transport.summary.fetchFailed",
      values: { error },
      tone: "error",
    };
  }
  if (!transport) {
    return {
      key: loading
        ? "remoteInference.transport.summary.checking"
        : "remoteInference.transport.summary.notChecked",
      tone: "muted",
    };
  }
  if (!transport.extra_installed) {
    return { key: "remoteInference.transport.extraMissing", tone: "warn" };
  }
  // Ahead of the credentials line, because when the Lab's own SFU is the
  // intended source its being off is WHY there are no credentials — and
  // "LIVEKIT_URL is missing" sends the operator hunting for a file to write
  // instead of a server to start.
  //
  // Guarded on `configured` in the same breath: credentials from livekit.env
  // or the process environment are a complete transport on their own, and for
  // an operator using LiveKit Cloud the Lab not hosting a server is the normal
  // state, not a fault. So this fires only when the SFU is off AND nothing
  // else supplied credentials.
  if (!transport.sfu_enabled && !transport.configured) {
    return { key: SFU_OFF_SUMMARY_KEY, tone: "error" };
  }
  if (!transport.configured) {
    return {
      key: "remoteInference.transport.summary.missingVars",
      values: { vars: transport.missing_vars.join(", ") },
      tone: "error",
    };
  }
  if (transport.endpoint_reachable === false) {
    return {
      key: "remoteInference.transport.summary.unreachable",
      values: { url: transport.url },
      tone: "error",
    };
  }
  if (transport.endpoint_reachable === null) {
    return {
      key: "remoteInference.transport.summary.notProbed",
      tone: "muted",
    };
  }
  if (transport.operator_present === true) {
    return {
      key: "remoteInference.transport.summary.ready",
      values: { room: transport.room },
      tone: "ok",
    };
  }
  if (transport.operator_present === false) {
    return {
      key: "remoteInference.transport.summary.operatorAbsent",
      values: { room: transport.room },
      tone: "warn",
    };
  }
  return { key: "remoteInference.transport.summary.notProbed", tone: "muted" };
}
