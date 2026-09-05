import React from "react";
import { useTranslation } from "react-i18next";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AdvancedSection } from "@/components/studio/panel/primitives";
import { VIDEO_CODECS, type RemoteRunConfig } from "./remoteRunConfig";

/**
 * The TRANSPORT knobs behind Advanced — and only those.
 *
 * They are not defaults anyone should tune casually: all four MUST match the
 * GPU side (Portal fingerprints the wire schema, so a disagreement silently
 * drops every packet instead of raising). That is exactly why they are
 * collapsed with their values on the summary line: readable at a glance,
 * without inviting a fiddle.
 *
 * What is NOT here is the point of the split. Precision, GPU type and flow
 * steps live on the "Policy server on Modal" card, beside Start GPU and beside
 * who pays, because they describe what that ONE side loads and what it loads
 * onto — nothing on the robot has to agree with them. The engine is not here
 * either: it is one shared field beside the task, because it is the one choice
 * on this screen that changes how the policy behaves rather than how the two
 * sides agree to talk.
 */

const RemoteAdvancedSection: React.FC<{
  config: RemoteRunConfig;
  onChange: (next: RemoteRunConfig) => void;
  /** The checkpoint's `n_action_steps` — how many steps it actually returns
   * per chunk, and therefore the CEILING on the horizon. Null when its config
   * doesn't say, which is when the engine defaults stand alone. */
  checkpointHorizon: number | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  disabled?: boolean;
}> = ({
  config,
  onChange,
  checkpointHorizon,
  open,
  onOpenChange,
  disabled,
}) => {
  const { t } = useTranslation();
  const set = <K extends keyof RemoteRunConfig>(
    key: K,
    value: RemoteRunConfig[K],
  ) => onChange({ ...config, [key]: value });

  // Every value on the line is live and is DATA — the codec id is the wire
  // value, the numbers are the ones that go out. `s_min` only reaches the wire
  // for rtc, so it is only claimed there. The GPU-side knobs used to be
  // appended here; they are on their own card now, and a summary claiming
  // settings this section no longer owns would be the worst of both.
  const summary =
    config.engine === "rtc"
      ? t("remoteInference.form.advancedSummaryRtc", {
          horizon: config.horizon,
          fps: config.fps,
          codec: config.videoCodec,
          sMin: config.sMin,
        })
      : t("remoteInference.form.advancedSummary", {
          horizon: config.horizon,
          fps: config.fps,
          codec: config.videoCodec,
        });

  return (
    <AdvancedSection open={open} onOpenChange={onOpenChange} summary={summary}>
      <div className="space-y-4">
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
        </div>
        {/* Said once, under the group, rather than beside the field: the
            number is only ever a problem in relation to the GPU side, which is
            what this whole group is about. Shown whenever the checkpoint
            declares a chunk width, and sharpened when the operator has typed
            past it — that is the state that ends in a connected session
            receiving nothing at all. */}
        {checkpointHorizon != null ? (
          <p
            className={`text-xs leading-relaxed ${
              config.horizon > checkpointHorizon
                ? "text-warn"
                : "text-muted-foreground"
            }`}
          >
            {config.horizon > checkpointHorizon
              ? t("remoteInference.form.horizonOverCeiling", {
                  steps: checkpointHorizon,
                })
              : t("remoteInference.form.horizonFromCheckpoint", {
                  steps: checkpointHorizon,
                })}
          </p>
        ) : null}
        {config.engine === "rtc" ? (
          <p className="text-xs leading-relaxed text-muted-foreground">
            {t("remoteInference.form.sMinHint")}
          </p>
        ) : null}
      </div>
    </AdvancedSection>
  );
};

export default RemoteAdvancedSection;
