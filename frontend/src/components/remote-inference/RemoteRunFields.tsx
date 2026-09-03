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
  VIDEO_CODECS,
  type RemoteRunConfig,
} from "./remoteRunConfig";
import { POLICY_PATH_PLACEHOLDER } from "./modalCommand";

/**
 * The remote run's own fields: the Hub id the GPU loads, plus the compact
 * transport group that MUST match the GPU side.
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
  disabled?: boolean;
}> = ({ config, onChange, hubIdDefault, disabled }) => {
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
        </div>
        <p className="text-xs text-muted-foreground">
          {config.durationS === 0
            ? t("remoteInference.form.durationUnbounded")
            : t("remoteInference.form.durationHint")}
        </p>
      </div>
    </div>
  );
};

export default RemoteRunFields;
