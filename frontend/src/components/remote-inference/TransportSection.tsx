import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { CheckCircle2, Loader2, RefreshCw, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
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
 *  - `source` distinguishes "livekit.env says so" from "your shell exported
 *    LIVEKIT_URL" — different problems with different remedies.
 *  - the clear-override button. `livekit.local.env` is written by the local-SFU
 *    script and OUTLIVES it, so after a Ctrl-C the robot keeps dialing a dead
 *    ws://127.0.0.1:7880 forever. Deleting that one file is the documented fix
 *    for the top footgun of the local path. It is idempotent and touches
 *    nothing else — livekit.local.yaml, which holds the SFU's own credentials,
 *    is deliberately left alone.
 *
 * Every value rendered here (url, room, variable names, error codes, the
 * backend's message) is data and appears verbatim.
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
  const { toast } = useToast();
  const { transport, loading, error, refresh, clearLocalOverride } =
    transportState;
  const [clearing, setClearing] = useState(false);

  const onClear = async () => {
    setClearing(true);
    try {
      const result = await clearLocalOverride();
      toast({
        title: result.removed
          ? t("remoteInference.transport.clearedTitle")
          : t("remoteInference.transport.alreadyClearTitle"),
        // The path is data — echoed exactly as the backend reported it.
        description: result.path,
      });
    } catch (e) {
      toast({
        title: t("remoteInference.transport.clearFailedTitle"),
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setClearing(false);
    }
  };

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

          {transport.local_env_exists ? (
            <div className="space-y-1.5 rounded-md border border-warn/40 p-2">
              <p className="text-xs leading-relaxed text-warn">
                {t("remoteInference.transport.overrideActive")}
              </p>
              {/* The path — data. */}
              <p className="font-mono text-[11px] break-all text-muted-foreground">
                {transport.local_env_path}
              </p>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => void onClear()}
                disabled={clearing}
                className="h-7 px-2 text-xs"
              >
                {clearing
                  ? t("remoteInference.transport.clearing")
                  : t("remoteInference.transport.clearOverride")}
              </Button>
              <p className="text-[11px] leading-relaxed text-muted-foreground">
                {t("remoteInference.transport.clearOverrideHint")}
              </p>
              {transport.sfu_config_exists ? (
                // Says where the GPU side's --livekit-api-key/-secret come
                // from. Deleting THIS file would rotate the SFU's own
                // credentials, so the button above never touches it.
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  {t("remoteInference.transport.sfuConfigPresent")}
                </p>
              ) : null}
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
};

export default TransportSection;
