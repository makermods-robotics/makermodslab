import React from "react";
import { useTranslation } from "react-i18next";
import { CheckCircle2, Loader2, RefreshCw, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { UseRemoteInferenceTransport } from "@/hooks/useRemoteInferenceTransport";

/**
 * What the remote run would dial, and whether anything is there.
 *
 * Three things here are worth more than they look:
 *
 *  - `endpoint_reachable` / `operator_present` are `boolean | null`, and null
 *    means THE PROBE DID NOT RUN (no `[drtc]` extra, or no credentials). That
 *    is a third state, and collapsing it into "false" would tell an operator
 *    the SFU is down when nothing ever asked it.
 *  - `source` distinguishes the Lab's OWN SFU from LiveKit Cloud, and within
 *    Cloud, "livekit.env says so" from "your shell exported LIVEKIT_URL" —
 *    three different problems with three different remedies.
 *  - the SFU block. When the Lab hosts the server, everything the GPU side
 *    needs is minted here: the key ID, the file its secret is in, and the
 *    TAILNET url a Modal container can actually dial. That block is what makes
 *    the generated `modal run` line above complete.
 *
 * There is no clear-override button any more. It deleted a dotenv file the
 * retired `tools/drtc/local_sfu*.sh` scripts wrote; with the Lab hosting the
 * SFU there is no file outliving a script to clear.
 *
 * Every value rendered here (url, room, key id, paths, variable names, error
 * codes, the backend's message and install hint) is data and appears verbatim.
 */
const Row: React.FC<{ label: string; children: React.ReactNode }> = ({
  label,
  children,
}) => (
  <div className="flex items-start justify-between gap-3 text-xs">
    <span className="shrink-0 text-muted-foreground">{label}</span>
    <span className="min-w-0 text-right break-all">{children}</span>
  </div>
);

const Verdict: React.FC<{ state: boolean | null; label: string }> = ({
  state,
  label,
}) => (
  <span
    className={cn(
      "inline-flex items-center gap-1",
      state === true && "text-emerald-600 dark:text-emerald-500",
      state === false && "text-destructive",
      state === null && "text-muted-foreground",
    )}
  >
    {state === true ? (
      <CheckCircle2 className="h-3 w-3" />
    ) : state === false ? (
      <XCircle className="h-3 w-3" />
    ) : null}
    {label}
  </span>
);

const TransportSection: React.FC<{
  transportState: UseRemoteInferenceTransport;
}> = ({ transportState }) => {
  const { t } = useTranslation();
  const { transport, loading, error, refresh } = transportState;

  return (
    <div className="space-y-2 rounded-lg border border-border p-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-semibold text-foreground">
          {t("remoteInference.transport.title")}
        </p>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={refresh}
          disabled={loading}
          className="h-7 gap-1.5 px-2 text-xs"
        >
          {loading ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <RefreshCw className="h-3 w-3" />
          )}
          {loading
            ? t("remoteInference.transport.checking")
            : t("remoteInference.transport.refresh")}
        </Button>
      </div>

      {error ? (
        // The thrown error's own text — the backend's, shown as raised.
        <p className="text-xs text-destructive">{error}</p>
      ) : null}

      {transport == null ? (
        !error ? (
          <p className="text-xs text-muted-foreground">
            {t("remoteInference.transport.notCheckedYet")}
          </p>
        ) : null
      ) : (
        <div className="space-y-1.5">
          {!transport.extra_installed ? (
            <p className="text-xs leading-relaxed text-warn">
              {t("remoteInference.transport.extraMissing")}
            </p>
          ) : null}

          <Row label={t("remoteInference.transport.sourceLabel")}>
            {/* The enum VALUE is data; only its label is translated, with the
                raw value as the fallback for one this build doesn't know. */}
            {t(
              `remoteInference.transport.source.${transport.source}` as never,
              { defaultValue: transport.source },
            )}
          </Row>
          <Row label={t("remoteInference.transport.urlLabel")}>
            <span className="font-mono">
              {transport.url || t("remoteInference.transport.unresolved")}
            </span>
          </Row>
          <Row label={t("remoteInference.transport.roomLabel")}>
            <span className="font-mono">
              {transport.room || t("remoteInference.transport.unresolved")}
            </span>
          </Row>
          <Row label={t("remoteInference.transport.credentialsLabel")}>
            <Verdict
              state={transport.configured}
              label={
                transport.configured
                  ? t("remoteInference.transport.configured")
                  : // Variable NAMES — identifiers, joined and shown verbatim.
                    t("remoteInference.transport.missingVars", {
                      vars: transport.missing_vars.join(", "),
                    })
              }
            />
          </Row>
          <Row label={t("remoteInference.transport.reachableLabel")}>
            <Verdict
              state={transport.endpoint_reachable}
              label={
                transport.endpoint_reachable === true
                  ? t("remoteInference.transport.reachable")
                  : transport.endpoint_reachable === false
                    ? t("remoteInference.transport.unreachable")
                    : t("remoteInference.transport.notProbed")
              }
            />
          </Row>
          <Row label={t("remoteInference.transport.operatorLabel")}>
            <Verdict
              state={transport.operator_present}
              label={
                transport.operator_present === true
                  ? t("remoteInference.transport.operatorPresent")
                  : transport.operator_present === false
                    ? t("remoteInference.transport.operatorAbsent")
                    : t("remoteInference.transport.notProbed")
              }
            />
          </Row>

          {transport.error_code ? (
            <p className="text-xs leading-relaxed text-destructive">
              {/* Code and message are both backend data — verbatim. */}
              <span className="font-mono">{transport.error_code}</span>
              {transport.message ? ` — ${transport.message}` : null}
            </p>
          ) : transport.message ? (
            <p className="text-xs leading-relaxed text-muted-foreground">
              {transport.message}
            </p>
          ) : null}

          {transport.sfu_enabled ? (
            <div className="space-y-1.5 rounded-md border border-border p-2">
              <p className="text-xs font-semibold text-foreground">
                {t("remoteInference.transport.sfuRunningTitle")}
              </p>
              <Row label={t("remoteInference.transport.sfuModalUrlLabel")}>
                {transport.sfu_modal_url ? (
                  <span className="font-mono">{transport.sfu_modal_url}</span>
                ) : (
                  <span className="text-warn">
                    {t("remoteInference.transport.sfuNoTailnet")}
                  </span>
                )}
              </Row>
              <Row label={t("remoteInference.transport.sfuKeyIdLabel")}>
                {/* The key NAME. The secret is never sent here — the file
                    below is where a human reads it. */}
                <span className="font-mono">
                  {transport.sfu_key_id ??
                    t("remoteInference.transport.unresolved")}
                </span>
              </Row>
              {transport.sfu_key_file ? (
                <Row label={t("remoteInference.transport.sfuKeyFileLabel")}>
                  <span className="font-mono">{transport.sfu_key_file}</span>
                </Row>
              ) : null}
              <Row label={t("remoteInference.transport.sfuExternalIpLabel")}>
                <Verdict
                  state={transport.sfu_external_ip}
                  label={
                    transport.sfu_external_ip
                      ? t("remoteInference.transport.sfuExternalIpOn")
                      : t("remoteInference.transport.sfuExternalIpOff")
                  }
                />
              </Row>
              {!transport.sfu_external_ip ? (
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  {t("remoteInference.transport.sfuExternalIpHint")}
                </p>
              ) : null}
            </div>
          ) : (
            <div className="space-y-1.5 rounded-md border border-border p-2">
              <p className="text-xs leading-relaxed text-muted-foreground">
                {t("remoteInference.transport.sfuNotRunning")}
              </p>
              {/* The start command is DATA — a shell line, shown verbatim. */}
              <pre className="overflow-x-auto rounded bg-muted/60 p-2 font-mono text-[11px] break-words whitespace-pre-wrap">
                makermodslab --sfu --sfu-external-ip
              </pre>
              {transport.sfu_install_hint ? (
                // The backend's per-OS install line, shown as raised.
                <p className="text-[11px] leading-relaxed text-warn">
                  {transport.sfu_install_hint}
                </p>
              ) : null}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TransportSection;
