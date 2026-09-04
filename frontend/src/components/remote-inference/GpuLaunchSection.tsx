import React from "react";
import { useTranslation } from "react-i18next";
import {
  AlertTriangle,
  Cpu,
  Loader2,
  Play,
  RefreshCw,
  Square,
  XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { UseGpuLauncher, UseGpuTargets } from "@/hooks/useGpuLauncher";
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
 *  - **It says the GPU is billing, and WHO PAYS.** A ready A100 costs real
 *    money whether or not an arm is moving, so the ready state says so, names
 *    the profile and workspace it was launched against, and shows the countdown
 *    to the automatic idle stop. Visibility is the cheapest cost control there
 *    is. The two pickers above the button are the same idea one step earlier:
 *    a machine with seven Modal profiles should not bill whichever one happens
 *    to be active in some other terminal.
 *  - **A failed listing never blocks a launch.** The pickers come from two
 *    `modal … list --json` calls; when they fail, the backend's own message
 *    replaces them and Start GPU stays live, because with no selection the CLI
 *    resolves the profile and environment itself — exactly as it did before
 *    the pickers existed.
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
  /** This machine's Modal profiles + the selected profile's environments, and
   * the remembered selection. */
  targets: UseGpuTargets;
  config: RemoteRunConfig;
  /** The Hub id to launch with when the field is left empty. */
  hubIdDefault: string;
  /** The effective task — the same string the robot side is started with, so
   * a language-conditioned policy is steered by the same sentence. */
  task: string;
  /** The policy is language-conditioned: an empty task must not reach the
   * GPU (its server refuses to start, and the failure used to be misread as
   * a tailnet problem). */
  taskRequired: boolean;
}> = ({ launcher, targets, config, hubIdDefault, task, taskRequired }) => {
  const { t } = useTranslation();
  const { status, pending, error, start, stop, launched } = launcher;
  const {
    targets: listing,
    profile,
    environment,
    setProfile,
    setEnvironment,
  } = targets;

  const state = status?.state ?? "idle";
  const hubId = config.policyHubId.trim() || hubIdDefault;
  const busy = pending || state === "stopping";
  // Nothing to pick from is not an error state: it is the CLI deciding, which
  // is what happened before these pickers existed.
  const canPick = listing != null && listing.error == null;
  const running = state !== "idle" && state !== "failed";

  const startBody = {
    engine: config.engine,
    policy_hub_id: hubId,
    task,
    horizon: config.horizon,
    fps: config.fps,
    video_codec: config.videoCodec,
    s_min: config.sMin,
    // Whatever is selected, always — the backend's "empty means the CLI
    // decides" is for API clients; the panel is explicit about who pays.
    profile,
    environment,
  };
  const launch = () => void start(startBody);
  const taskMissing = taskRequired && task.trim() === "";

  // The running server holds the values it was started with; the form can
  // move on without it. Everything below is part of the Portal wire schema
  // (a disagreement drops every packet in silence) or steers the policy
  // itself, so any drift while the GPU is up is worth a loud line and a
  // one-press restart.
  //
  // The reference is the SERVER's own echo of what it launched with, and this
  // tab's memory of its last start only as the fallback. That ordering is the
  // whole point: `launched` is null after a page reload and for a GPU another
  // tab (or an SDK client) started, which is precisely when a stale form is
  // most likely — and a partial comparison would say "matches" about knobs it
  // never checked, so the echo is used only when the whole tuple is there.
  const up = state === "starting" || state === "ready";
  const echoed =
    status &&
    status.engine != null &&
    status.policy_hub_id != null &&
    status.horizon != null &&
    status.fps != null &&
    status.video_codec != null
      ? {
          engine: status.engine,
          policy_hub_id: status.policy_hub_id,
          task: status.task ?? "",
          horizon: status.horizon,
          fps: status.fps,
          video_codec: status.video_codec,
          s_min: status.s_min,
        }
      : null;
  const reference: {
    engine: string;
    policy_hub_id: string;
    task: string;
    horizon: number;
    fps: number;
    video_codec: string;
    s_min: number | null;
  } | null = echoed ?? launched;
  const drifted: string[] = [];
  if (reference && up) {
    if (reference.engine !== startBody.engine) drifted.push("engine");
    if (reference.policy_hub_id !== startBody.policy_hub_id)
      drifted.push("policy");
    if (reference.task !== startBody.task) drifted.push("task");
    if (reference.horizon !== startBody.horizon) drifted.push("horizon");
    if (reference.fps !== startBody.fps) drifted.push("fps");
    if (reference.video_codec !== startBody.video_codec) drifted.push("codec");
    // s_min only exists on the wire for rtc; sync ignores it on both sides.
    // Null (an older server that echoes no tuple) is "unknown", not "differs".
    if (
      startBody.engine === "rtc" &&
      reference.s_min != null &&
      reference.s_min !== startBody.s_min
    )
      drifted.push("s_min");
  }
  const restart = async () => {
    await stop();
    await start(startBody);
  };

  // The workspace behind the profile a RUNNING GPU was launched with. The
  // status echoes the profile name only, so the workspace is looked up in the
  // listing; a profile that has since disappeared just shows its name.
  const launchedWorkspace = listing?.profiles.find(
    (p) => p.name === status?.profile,
  )?.workspace;

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
            disabled={busy || !hubId || taskMissing}
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

      {/* WHO PAYS. Above the button because it is a property of the launch,
          and disabled while one is in flight or a GPU is up — the selection
          describes what WAS launched until that GPU is stopped. */}
      {canPick && listing.profiles.length > 0 ? (
        <div className="grid gap-2 sm:grid-cols-2">
          <div className="space-y-1">
            <Label htmlFor="gpu-modal-profile" className="text-xs">
              {t("remoteInference.gpu.profileLabel")}
            </Label>
            <Select
              value={profile}
              disabled={busy || running}
              onValueChange={setProfile}
            >
              <SelectTrigger id="gpu-modal-profile" className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {listing.profiles.map((p) => (
                  // BOTH halves are data: the profile name is what
                  // MODAL_PROFILE is set to, and the workspace is the thing an
                  // operator actually recognizes as "the account this bills".
                  <SelectItem key={p.name} value={p.name} className="text-xs">
                    <span className="font-mono">{p.name}</span>
                    {p.workspace ? (
                      <span className="text-muted-foreground">
                        {" · "}
                        {p.workspace}
                      </span>
                    ) : null}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {listing.environments.length > 0 ? (
            <div className="space-y-1">
              <Label htmlFor="gpu-modal-environment" className="text-xs">
                {t("remoteInference.gpu.environmentLabel")}
              </Label>
              <Select
                value={environment}
                disabled={busy || running}
                onValueChange={setEnvironment}
              >
                <SelectTrigger
                  id="gpu-modal-environment"
                  className="h-8 text-xs"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {listing.environments.map((e) => (
                    <SelectItem key={e.name} value={e.name} className="text-xs">
                      <span className="font-mono">{e.name}</span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          ) : null}
        </div>
      ) : null}

      {listing?.error ? (
        // The backend's own text, verbatim. NOT a blocker: Start GPU stays
        // live above, because with no selection the CLI resolves the target
        // itself — a failed listing is not a failed launch.
        <p className="text-xs leading-relaxed text-muted-foreground">
          {listing.error.message}
        </p>
      ) : null}

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
            {taskMissing ? `${t("remoteInference.gpu.taskRequired")} ` : ""}
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
            {/* WHICH WORKSPACE PAYS, from the status echo — the whole point of
                letting the target be chosen. Absent when the CLI resolved it,
                which is honest: the Lab genuinely does not know which one it
                picked. Profile, workspace and environment names are DATA. */}
            {status?.profile ? (
              <>
                {" "}
                <span className="font-normal text-muted-foreground">
                  {launchedWorkspace
                    ? t("remoteInference.gpu.billingToWorkspace", {
                        profile: status.profile,
                        workspace: launchedWorkspace,
                      })
                    : t("remoteInference.gpu.billingTo", {
                        profile: status.profile,
                      })}
                  {status.environment
                    ? ` ${t("remoteInference.gpu.billingEnvironment", {
                        environment: status.environment,
                      })}`
                    : ""}
                </span>
              </>
            ) : null}
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

      {drifted.length > 0 && reference ? (
        <div className="space-y-1.5 rounded-md border border-warn/40 p-2">
          <p className="flex items-start gap-1.5 text-xs leading-relaxed text-warn">
            <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
            <span>
              {/* Field NAMES are identifiers (they are the flags the server
                  was started with) and the values are data — both verbatim. */}
              {t("remoteInference.gpu.driftBody", {
                fields: drifted.join(", "),
              })}{" "}
              <span className="font-mono">
                {reference.engine} · h{reference.horizon} · {reference.fps} fps
                · {reference.video_codec}
                {reference.engine === "rtc" && reference.s_min != null
                  ? ` · s_min ${reference.s_min}`
                  : ""}
              </span>
            </span>
          </p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => void restart()}
            disabled={busy}
            className="h-7 gap-1.5 px-2 text-xs"
          >
            <RefreshCw className="h-3 w-3" />
            {t("remoteInference.gpu.restart")}
          </Button>
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
