import React from "react";
import { useTranslation } from "react-i18next";
import { Cpu, Loader2, Play, Square, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { UseGpuLauncher } from "@/hooks/useGpuLauncher";
import { MODAL_WRAPPERS } from "./modalCommand";
import type { RemoteRunConfig } from "./remoteRunConfig";

/**
 * Start and stop the policy server on Modal, from here.
 *
 * The Lab owns the robot side and, since S3.6, the SFU; since S3.8 it can also
 * launch the GPU. Three things about this block are deliberate:
 *
 *  - **It does not gate the run.** "Run it remotely" still unblocks off the
 *    transport probe's `operator_present`, never off `gpu.state`. Two
 *    independent signals, and the one that gates the ARM is the one that
 *    observes the room — a log line saying "connected" is a hint.
 *  - **It says the GPU is billing.** A ready A100 costs real money whether or
 *    not an arm is moving, so the ready state says so and shows the countdown
 *    to the automatic idle stop. Visibility is the cheapest cost control there
 *    is.
 *  - **The manual command stays** (below, under "Run it yourself instead"). It
 *    is the only route when `modal` is missing or unauthenticated, the only
 *    route to a hand-tuned flag, and the ground truth an operator compares
 *    against when the fingerprint watchdog fires.
 *
 * Everything the backend sends — the message, the hint, the last log line, the
 * room, the log path — is DATA and is rendered verbatim. Only the frame around
 * it is localized.
 */
const GpuLaunchSection: React.FC<{
  launcher: UseGpuLauncher;
  config: RemoteRunConfig;
  /** The Hub id to launch with when the field is left empty. */
  hubIdDefault: string;
  /** The effective task — the same string the robot side is started with, so
   * a language-conditioned policy is steered by the same sentence. */
  task: string;
}> = ({ launcher, config, hubIdDefault, task }) => {
  const { t } = useTranslation();
  const { status, pending, error, start, stop } = launcher;

  const state = status?.state ?? "idle";
  const hubId = config.policyHubId.trim() || hubIdDefault;
  const busy = pending || state === "stopping";

  const launch = () =>
    void start({
      engine: config.engine,
      policy_hub_id: hubId,
      task,
      horizon: config.horizon,
      fps: config.fps,
      video_codec: config.videoCodec,
      s_min: config.sMin,
    });

  return (
    <div className="space-y-2 rounded-lg border border-border p-3">
      <div className="flex items-center justify-between gap-2">
        <p className="flex items-center gap-1.5 text-xs font-semibold text-foreground">
          <Cpu className="h-3.5 w-3.5" />
          {t("remoteInference.gpu.title")}
        </p>
        {state === "idle" || state === "failed" ? (
          <Button
            type="button"
            size="sm"
            onClick={launch}
            disabled={busy || !hubId}
            className="h-7 gap-1.5 px-2 text-xs"
          >
            <Play className="h-3 w-3" />
            {state === "failed"
              ? t("remoteInference.gpu.retry")
              : t("remoteInference.gpu.start")}
          </Button>
        ) : (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => void stop()}
            disabled={busy}
            className="h-7 gap-1.5 px-2 text-xs"
          >
            {busy ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <Square className="h-3 w-3" />
            )}
            {state === "starting"
              ? t("remoteInference.gpu.cancel")
              : t("remoteInference.gpu.stop")}
          </Button>
        )}
      </div>

      {state === "idle" ? (
        <>
          {/* Backend prose, and the one case where an IDLE panel has something
              to say: the idle auto-stop leaves its reason behind so "the GPU is
              gone" does not read as a crash. */}
          {status?.message ? (
            <p className="text-xs leading-relaxed text-muted-foreground">
              {status.message}
            </p>
          ) : null}
          <p className="text-xs leading-relaxed text-muted-foreground">
            {/* The wrapper PATH is data — it is what `modal run` is handed, and
                it is how the operator knows which of the two servers this is. */}
            {t("remoteInference.gpu.idleHint", {
              wrapper: MODAL_WRAPPERS[config.engine],
            })}
          </p>
        </>
      ) : null}

      {state === "starting" ? (
        <div className="space-y-1.5">
          <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" />
            {/* The phase VALUE is a backend identifier; only its prose is
                translated, with the raw value as the fallback for one this
                build doesn't know. */}
            {status?.phase
              ? t(`remoteInference.gpu.phase.${status.phase}` as never, {
                  defaultValue: status.phase,
                })
              : t("remoteInference.gpu.phase.pending")}
            {" · "}
            {t("remoteInference.gpu.elapsed", {
              seconds: Math.round(status?.elapsed_s ?? 0),
            })}
          </p>
          {status?.last_line ? (
            // The container's own output, verbatim.
            <pre className="overflow-x-auto rounded border border-border bg-muted/60 px-2 py-1 font-mono text-[10px] leading-relaxed break-all whitespace-pre-wrap text-muted-foreground">
              {status.last_line}
            </pre>
          ) : null}
        </div>
      ) : null}

      {state === "ready" ? (
        <div className="space-y-1">
          <p className="text-xs font-medium text-emerald-600 dark:text-emerald-500">
            {t("remoteInference.gpu.running")}
          </p>
          <p className="text-xs leading-relaxed text-muted-foreground">
            {status?.idle_stop_in_s == null
              ? t("remoteInference.gpu.idleStopPaused")
              : t("remoteInference.gpu.idleStopIn", {
                  minutes: Math.ceil(status.idle_stop_in_s / 60),
                })}
          </p>
          {status?.room ? (
            <p className="text-xs text-muted-foreground">
              {t("remoteInference.gpu.roomLabel")}{" "}
              <span className="font-mono break-all">{status.room}</span>
            </p>
          ) : null}
        </div>
      ) : null}

      {state === "failed" && status ? (
        <div className="space-y-1">
          {/* Backend prose, shown as raised. */}
          <p className="flex items-start gap-1.5 text-xs leading-relaxed text-destructive">
            <XCircle className="mt-0.5 h-3 w-3 shrink-0" />
            {status.message}
          </p>
          {status.hint ? (
            <p className="text-xs leading-relaxed text-muted-foreground">
              {status.hint}
            </p>
          ) : null}
          {status.code ? (
            // The machine-readable code, verbatim. Deliberately NOT a
            // localized hint keyed off it: the backend already sends the
            // remedy as prose, and a second, translated copy of the same
            // sentence is exactly the drift the localization rule forbids
            // ("the Python backend is never localized"). Shown so an operator
            // can quote it in a bug report.
            <p className="font-mono text-[10px] text-muted-foreground">
              {status.code}
            </p>
          ) : null}
        </div>
      ) : null}

      {error ? (
        // The refusal the request itself raised (a missing `modal` binary, an
        // empty Hub id, no tailnet address) — the backend's own text.
        <p className="text-xs leading-relaxed text-destructive">{error}</p>
      ) : null}

      {status?.log_path && state !== "idle" ? (
        <p className="text-xs break-all text-muted-foreground">
          {t("remoteInference.gpu.logLabel")}{" "}
          <span className="font-mono">{status.log_path}</span>
        </p>
      ) : null}
    </div>
  );
};

export default GpuLaunchSection;
