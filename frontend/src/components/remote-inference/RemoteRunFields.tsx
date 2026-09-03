import React from "react";
import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  DEFAULT_HORIZON,
  VIDEO_CODECS,
  type RemoteEngine,
  type RemoteRunConfig,
} from "./remoteRunConfig";
import { POLICY_PATH_PLACEHOLDER } from "./modalCommand";

/**
 * The remote run's own fields: the Hub id the GPU loads, the engine, plus the
 * compact transport group that MUST match the GPU side.
 *
 * Everything else a remote run needs — robot, checkpoint, cameras, task — is
 * the Deploy panel's existing form, reused verbatim. That is the point: a
 * remote run is the same run with the policy somewhere else.
 */
const RemoteRunFields: React.FC<{
  config: RemoteRunConfig;
  onChange: (next: RemoteRunConfig) => void;
  /** The selected run's own Hub repo, offered as a placeholder default the
   * same way the task and coaching-dataset fields offer theirs: visibly not
   * the operator's input, and restored the moment they clear the box. */
  hubIdDefault: string;
  /** Whether this checkpoint's policy family can be in-painted at all. False
   * also for an UNKNOWN policy type — the rtc option stays selectable so the
   * refusal explains itself rather than vanishing, and `deployGuards` blocks
   * the launch. */
  rtcSupported: boolean;
  disabled?: boolean;
}> = ({ config, onChange, hubIdDefault, rtcSupported, disabled }) => {
  const { t } = useTranslation();
  const set = <K extends keyof RemoteRunConfig>(
    key: K,
    value: RemoteRunConfig[K],
  ) => onChange({ ...config, [key]: value });

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="remote-hub-id">
          {t("remoteInference.form.hubIdLabel")}
        </Label>
        <Input
          id="remote-hub-id"
          value={config.policyHubId}
          disabled={disabled}
          onChange={(e) => set("policyHubId", e.target.value)}
          // A repo id shape, not prose — the literal the operator must match.
          placeholder={hubIdDefault || POLICY_PATH_PLACEHOLDER}
          className="font-mono"
        />
        <p className="text-xs text-muted-foreground">
          {t("remoteInference.form.hubIdHint")}
          {config.policyHubId.trim() === "" && hubIdDefault
            ? ` ${t("remoteInference.form.hubIdInherited")}`
            : ""}
        </p>
      </div>

      {/* The engine. It picks the robot-side chunk player AND the GPU wrapper
          in the command below, so it sits above the transport group that has
          to match it. */}
      <div className="space-y-2">
        <Label htmlFor="remote-engine">
          {t("remoteInference.form.engineLabel")}
        </Label>
        <Select
          value={config.engine}
          disabled={disabled}
          onValueChange={(v) => {
            // Switching engines re-seeds the horizon, because the two regimes
            // want different ones (one open-loop ACT block vs the flow
            // families' full chunk_size) and a horizon carried over from the
            // other engine is the mismatch Portal drops packets over. An
            // operator who has already typed their own keeps it.
            const engine = v as RemoteEngine;
            const kept =
              config.horizon !== DEFAULT_HORIZON[config.engine]
                ? config.horizon
                : DEFAULT_HORIZON[engine];
            onChange({ ...config, engine, horizon: kept });
          }}
        >
          <SelectTrigger id="remote-engine">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {/* Option VALUES ("sync" / "rtc") are what the backend parses —
                only the labels are translated. */}
            <SelectItem value="sync">
              {t("remoteInference.form.engine.sync")}
            </SelectItem>
            <SelectItem value="rtc">
              {t("remoteInference.form.engine.rtc")}
            </SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs leading-relaxed text-muted-foreground">
          {config.engine === "rtc"
            ? t("remoteInference.form.engine.rtcHint")
            : t("remoteInference.form.engine.syncHint")}
        </p>
        {config.engine === "rtc" && !rtcSupported ? (
          <p className="text-xs leading-relaxed text-warn">
            {t("remoteInference.form.engine.rtcUnsupported")}
          </p>
        ) : null}
      </div>

      <div className="space-y-3 rounded-lg border border-border bg-muted/40 p-3">
        <p className="text-xs font-semibold text-foreground">
          {t("remoteInference.form.transportGroup")}
        </p>
        <p className="text-xs leading-relaxed text-muted-foreground">
          {t("remoteInference.form.transportGroupHint")}
        </p>
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <Label htmlFor="remote-horizon" className="text-xs">
              {t("remoteInference.form.horizonLabel")}
            </Label>
            <NumberInput
              id="remote-horizon"
              min={1}
              value={config.horizon}
              disabled={disabled}
              onChange={(v) => {
                if (v !== undefined) set("horizon", v);
              }}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="remote-fps" className="text-xs">
              {t("remoteInference.form.fpsLabel")}
            </Label>
            <NumberInput
              id="remote-fps"
              min={1}
              value={config.fps}
              disabled={disabled}
              onChange={(v) => {
                if (v !== undefined) set("fps", v);
              }}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="remote-codec" className="text-xs">
              {t("remoteInference.form.codecLabel")}
            </Label>
            <Select
              value={config.videoCodec}
              disabled={disabled}
              onValueChange={(v) =>
                set("videoCodec", v as RemoteRunConfig["videoCodec"])
              }
            >
              <SelectTrigger id="remote-codec">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {/* Codec ids are wire values AND their own labels — there is
                    nothing here to translate. */}
                {VIDEO_CODECS.map((codec) => (
                  <SelectItem key={codec} value={codec}>
                    {codec}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="remote-duration" className="text-xs">
              {t("remoteInference.form.durationLabel")}
            </Label>
            <NumberInput
              id="remote-duration"
              min={0}
              value={config.durationS}
              disabled={disabled}
              onChange={(v) => {
                if (v !== undefined) set("durationS", v);
              }}
            />
          </div>
          {/* RTC only. The sync wrapper has no --s-min flag at all, and the
              sync robot's own s_min is left at its default with the rest of
              the scheduler knobs. */}
          {config.engine === "rtc" ? (
            <div className="space-y-1.5">
              <Label htmlFor="remote-s-min" className="text-xs">
                {t("remoteInference.form.sMinLabel")}
              </Label>
              <NumberInput
                id="remote-s-min"
                min={1}
                value={config.sMin}
                disabled={disabled}
                onChange={(v) => {
                  if (v !== undefined) set("sMin", v);
                }}
              />
            </div>
          ) : null}
        </div>
        <p className="text-xs text-muted-foreground">
          {config.durationS === 0
            ? t("remoteInference.form.durationUnbounded")
            : t("remoteInference.form.durationHint")}
        </p>
        {config.engine === "rtc" ? (
          <p className="text-xs leading-relaxed text-muted-foreground">
            {t("remoteInference.form.sMinHint")}
          </p>
        ) : null}
      </div>
    </div>
  );
};

export default RemoteRunFields;
