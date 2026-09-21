import { useState } from "react";
import { useTranslation } from "react-i18next";
import { AdvancedSection } from "@/components/studio/panel/primitives";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { VIDEO_CODECS, type RemoteRunConfig } from "./remoteRunConfig";

const REGIONS = ["auto", "us-west", "us-east", "us-central", "eu", "ap", "uk", "ca", "me", "sa", "af", "mx"];

export default function RemoteNetworkSection({ config, onChange, disabled }: {
  config: RemoteRunConfig;
  onChange: (next: RemoteRunConfig) => void;
  disabled?: boolean;
}) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const set = <K extends keyof RemoteRunConfig>(key: K, value: RemoteRunConfig[K]) => onChange({ ...config, [key]: value });
  const number = (key: "videoQuality" | "videoBitrateKbps" | "cameraSendHz" | "latencyK" | "tolerance", min: number, max: number, step = 1) => (
    <div className="space-y-1.5">
      <Label htmlFor={`remote-${key}`} className="text-xs">{t(`remoteInference.network.${key}`)}</Label>
      <NumberInput id={`remote-${key}`} value={config[key]} min={min} max={max} step={step} integer={key === "videoQuality" || key === "videoBitrateKbps"} disabled={disabled}
        onChange={(value) => { if (value !== undefined) set(key, value); }} />
      <p className="text-xs text-muted-foreground">{t(`remoteInference.network.${key}Hint`)}</p>
    </div>
  );
  return (
    <AdvancedSection title={t("remoteInference.network.title")} summary={`${config.region === "auto" ? t("remoteInference.form.autoShort") : config.region} · ${config.videoCodec}`} open={open} onOpenChange={setOpen}>
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <Label htmlFor="remote-region" className="text-xs">{t("remoteInference.network.region")}</Label>
            <Select value={config.region} disabled={disabled} onValueChange={(value) => set("region", value)}>
              <SelectTrigger id="remote-region"><SelectValue /></SelectTrigger>
              <SelectContent>{REGIONS.map((region) => <SelectItem key={region} value={region}>{region === "auto" ? t("remoteInference.form.autoAssignment") : region}</SelectItem>)}</SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">{t(config.region === "auto" ? "remoteInference.network.autoRegionHint" : "remoteInference.network.regionHint")}</p>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="remote-codec" className="text-xs">{t("remoteInference.form.codecLabel")}</Label>
            <Select value={config.videoCodec} disabled={disabled} onValueChange={(value) => set("videoCodec", value as RemoteRunConfig["videoCodec"])}>
              <SelectTrigger id="remote-codec"><SelectValue /></SelectTrigger>
              <SelectContent>{VIDEO_CODECS.map((codec) => <SelectItem key={codec} value={codec}>{codec}</SelectItem>)}</SelectContent>
            </Select>
          </div>
          {config.videoCodec === "MJPEG" ? number("videoQuality", 1, 100) : number("videoBitrateKbps", 256, 20000, 256)}
          {config.engine === "rtc" ? number("cameraSendHz", 0, config.fps, 0.5) : null}
        </div>
        <details className="text-xs">
          <summary className="cursor-pointer text-muted-foreground">{t("remoteInference.network.advanced")}</summary>
          <div className="mt-3 grid grid-cols-2 gap-3">
            {config.engine === "rtc" ? number("latencyK", 0, 5, 0.1) : null}
            {number("tolerance", 0.1, 5, 0.1)}
          </div>
        </details>
      </div>
    </AdvancedSection>
  );
}
