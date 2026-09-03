import React from "react";
import { Trans, useTranslation } from "react-i18next";
import { Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import type { RemoteInferenceTransportStatus } from "@/hooks/useRemoteInferenceTransport";
import {
  buildModalRunLine,
  LOCAL_SECRET_PLACEHOLDER,
  LOCAL_SFU_CONFIG_PATH,
} from "./modalCommand";
import type { RemoteRunConfig } from "./remoteRunConfig";

/**
 * The command the operator runs in the OTHER terminal.
 *
 * The Lab owns only the robot side (docs/drtc/SLICE3.md §2, lifecycle A): it
 * verifies the SFU and that an operator is present, and it never launches the
 * GPU. So the single most useful thing this panel can do is hand over a line
 * that is guaranteed to agree with what the robot side is about to start with
 * — same horizon, same fps, same codec, same room — because a disagreement
 * there is invisible by construction.
 *
 * The command text itself is DATA: never translated, never reflowed, never
 * case-folded. Only the prose around it is localized.
 */
const ModalRunLine: React.FC<{
  config: RemoteRunConfig;
  transport: RemoteInferenceTransportStatus | null;
  hubIdDefault: string;
  /** The task the robot side will be started with — DATA, forwarded so both
   * sides steer the policy with the same sentence. */
  task: string;
}> = ({ config, transport, hubIdDefault, task }) => {
  const { t } = useTranslation();
  const { toast } = useToast();

  const line = buildModalRunLine({
    policyHubId: config.policyHubId.trim() || hubIdDefault,
    task,
    horizon: config.horizon,
    fps: config.fps,
    videoCodec: config.videoCodec,
    room: transport?.room ?? "",
    url: transport?.url ?? "",
    source: transport?.source ?? "none",
  });

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(line);
      toast({ title: t("remoteInference.modalRun.copiedTitle") });
    } catch {
      toast({
        title: t("remoteInference.modalRun.copyFailedTitle"),
        description: t("remoteInference.modalRun.copyFailedBody"),
        variant: "destructive",
      });
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-semibold text-foreground">
          {t("remoteInference.modalRun.title")}
        </p>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => void copy()}
          className="h-7 gap-1.5 px-2 text-xs"
        >
          <Copy className="h-3 w-3" />
          {t("remoteInference.modalRun.copy")}
        </Button>
      </div>
      <p className="text-xs leading-relaxed text-muted-foreground">
        {t("remoteInference.modalRun.intro")}
      </p>
      {/* Wraps rather than scrolls: this is meant to be READ before it is
          pasted — a mismatched --horizon is the whole failure mode. */}
      <pre className="overflow-x-auto rounded-md border border-border bg-muted/60 p-3 font-mono text-[11px] leading-relaxed break-words whitespace-pre-wrap">
        {line}
      </pre>
      {transport == null || !transport.room ? (
        <p className="text-xs text-warn">
          {t("remoteInference.modalRun.noRoomYet")}
        </p>
      ) : null}
      {transport?.source === "local_override" ? (
        <p className="text-xs leading-relaxed text-muted-foreground">
          {/* <0> and <1> hold a literal placeholder and a literal path —
              identifiers, so they stay in the Latin script inside whatever
              sentence a translator writes around them. */}
          <Trans
            i18nKey="remoteInference.modalRun.secretsHint"
            values={{
              placeholder: LOCAL_SECRET_PLACEHOLDER,
              path: LOCAL_SFU_CONFIG_PATH,
            }}
            components={[
              <code key="0" className="font-mono" />,
              <code key="1" className="font-mono" />,
            ]}
          />
        </p>
      ) : null}
    </div>
  );
};

export default ModalRunLine;
