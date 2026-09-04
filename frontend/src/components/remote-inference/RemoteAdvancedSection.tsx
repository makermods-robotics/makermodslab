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
import {
  GPU_TYPES,
  MODEL_DTYPES,
  type GpuType,
  type ModelDtype,
  type UseGpuKnobs,
} from "@/hooks/useGpuLauncher";
import { VIDEO_CODECS, type RemoteRunConfig } from "./remoteRunConfig";

/**
 * The transport knobs — and, since S3.8e, the two GPU-side ones — behind
 * Advanced.
 *
 * They are not defaults anyone should tune casually. The transport four MUST
 * match the GPU side (Portal fingerprints the wire schema, so a disagreement
 * silently drops every packet instead of raising); the two below them change
 * what the GPU loads and what it loads onto, which is money and latency rather
 * than a broken stream. That is exactly why all of them are collapsed with
 * their values on the summary line: readable at a glance, without inviting a
 * fiddle.
 *
 * The engine is NOT here. It is one shared field beside the task, because it is
 * the one choice on this screen that changes how the policy behaves rather than
 * how the two sides agree to talk.
 */

/** Radix refuses a `SelectItem` with an empty value, and "" is exactly what
 * "as the checkpoint saved it" IS on the wire. So the option carries this
 * sentinel and the mapping happens at the boundary — it never leaves this
 * file, and it is never sent. */
const CHECKPOINT_DTYPE = "__checkpoint__";

const RemoteAdvancedSection: React.FC<{
  config: RemoteRunConfig;
  onChange: (next: RemoteRunConfig) => void;
  /** Precision + GPU type. Beside the config rather than on it: everything on
   * `RemoteRunConfig` goes to BOTH halves of the run, and these two go only to
   * the GPU. */
  knobs: UseGpuKnobs;
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
  knobs,
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

  // The GPU-side half of the summary, built from DATA alone — a Modal GPU spec
  // and a torch dtype name, with the separator the rest of the line uses. It is
  // interpolated rather than concatenated with translated words so a translator
  // still owns the whole sentence. The precision appears only when it was
  // chosen: unset is the checkpoint's own, which is not a value to display.
  const extra =
    ` · ${knobs.gpu}` + (knobs.modelDtype ? ` · ${knobs.modelDtype}` : "");
  // Every other value on the line is live and is DATA too — the codec id is the
  // wire value, the numbers are the ones that go out. `s_min` only reaches the
  // wire for rtc, so it is only claimed there.
  const summary =
    config.engine === "rtc"
      ? t("remoteInference.form.advancedSummaryRtc", {
          horizon: config.horizon,
          fps: config.fps,
          codec: config.videoCodec,
          sMin: config.sMin,
          extra,
        })
      : t("remoteInference.form.advancedSummary", {
          horizon: config.horizon,
          fps: config.fps,
          codec: config.videoCodec,
          extra,
        });

  return (
    <AdvancedSection open={open} onOpenChange={onOpenChange} summary={summary}>
      <div className="space-y-4">
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
        {/* The GPU side's own two. Separated from the group above because they
            are the opposite kind of setting: nothing here has to MATCH the
            robot — these decide what the container loads and what it loads
            onto, i.e. whether the policy fits its latency budget at all, and
            what the hour costs. --------------------------------------- */}
        <div className="space-y-3 border-t border-border pt-3">
          <p className="text-xs leading-relaxed text-muted-foreground">
            {t("remoteInference.form.gpuGroupHint")}
          </p>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label htmlFor="remote-precision" className="text-xs">
                {t("remoteInference.form.precisionLabel")}
              </Label>
              <Select
                value={knobs.modelDtype || CHECKPOINT_DTYPE}
                disabled={disabled}
                onValueChange={(v) =>
                  knobs.setModelDtype(
                    v === CHECKPOINT_DTYPE ? "" : (v as ModelDtype),
                  )
                }
              >
                <SelectTrigger id="remote-precision">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {/* The one option that is prose rather than a value: it
                      stands for sending no flag at all. */}
                  <SelectItem value={CHECKPOINT_DTYPE}>
                    {t("remoteInference.form.precisionCheckpoint")}
                  </SelectItem>
                  {/* torch dtype names — wire values AND their own labels. */}
                  {MODEL_DTYPES.map((dtype) => (
                    <SelectItem key={dtype} value={dtype}>
                      {dtype}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="remote-gpu" className="text-xs">
                {t("remoteInference.form.gpuLabel")}
              </Label>
              <Select
                value={knobs.gpu}
                disabled={disabled}
                onValueChange={(v) => knobs.setGpu(v as GpuType)}
              >
                <SelectTrigger id="remote-gpu">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {/* Modal's own GPU specs — identifiers, never translated. */}
                  {GPU_TYPES.map((gpu) => (
                    <SelectItem key={gpu} value={gpu}>
                      {gpu}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <p className="text-xs leading-relaxed text-muted-foreground">
            {t("remoteInference.form.precisionHint")}
          </p>
        </div>
      </div>
    </AdvancedSection>
  );
};

export default RemoteAdvancedSection;
