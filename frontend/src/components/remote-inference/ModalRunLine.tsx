import React from "react";
import { Trans, useTranslation } from "react-i18next";
import { Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import type { RemoteInferenceTransportStatus } from "@/hooks/useRemoteInferenceTransport";
import {
  buildModalRunLine,
  LOCAL_SECRET_PLACEHOLDER,
  LOCAL_SFU_KEY_FILE,
} from "./modalCommand";
import type { RemoteRunConfig } from "./remoteRunConfig";

/**
 * The command the operator runs in the OTHER terminal.
 *
 * The Lab owns the robot side and, since S3.6, the SFU (`makermodslab --sfu`);
 * it still never launches the GPU. So the single most useful thing this panel
 * can do is hand over a line that is guaranteed to agree with what the robot
 * side is about to start with — same horizon, same fps, same codec, same room
 * — because a disagreement there is invisible by construction.
 *
 * Under the Lab's own SFU the line also carries the whole transport: the
 * TAILNET url (`sfu_modal_url` — never the loopback one a local child dials,
 * which a Modal container has no route to), the real key ID, and a placeholder
 * for the secret the API deliberately never returns.
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
  /** The Modal target selected above, so the pasted line bills the SAME
   * workspace the Lab's own Start GPU would. Empty ⇒ the CLI resolves it. */
  profile: string;
  environment: string;
}> = ({ config, transport, hubIdDefault, task, profile, environment }) => {
  const { t } = useTranslation();
  const { toast } = useToast();

  const line = buildModalRunLine({
    policyHubId: config.policyHubId.trim() || hubIdDefault,
    // The engine picks WHICH wrapper this line runs. The two GPU servers
    // publish different state schemas, so pairing the wrong one with the robot
    // side is the same silent zero-chunk failure a wrong horizon is.
    engine: config.engine,
    task,
    horizon: config.horizon,
    fps: config.fps,
    videoCodec: config.videoCodec,
    sMin: config.sMin,
    room: transport?.room ?? "",
    // The url a CONTAINER dials, which is not the one this machine's child
    // dials. Empty when tailscale reported no address — the line then omits
    // --livekit-url rather than offering a loopback one that cannot work.
    url: transport?.sfu_modal_url ?? "",
    source: transport?.source ?? "none",
    sfuKeyId: transport?.sfu_key_id ?? "",
    // The same two values Start GPU sends, rendered the way the CLI takes
    // them: MODAL_PROFILE= in front, --env as a `modal run` option.
    profile,
    environment,
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
      {transport?.source === "sfu" ? (
        <>
          <p className="text-xs leading-relaxed text-muted-foreground">
            {/* <0> and <1> hold a literal placeholder and a literal path —
                identifiers, so they stay in the Latin script inside whatever
                sentence a translator writes around them. */}
            <Trans
              i18nKey="remoteInference.modalRun.secretsHint"
              values={{
                placeholder: LOCAL_SECRET_PLACEHOLDER,
                path: transport.sfu_key_file ?? LOCAL_SFU_KEY_FILE,
              }}
              components={[
                <code key="0" className="font-mono" />,
                <code key="1" className="font-mono" />,
              ]}
            />
          </p>
          {!transport.sfu_modal_url ? (
            <p className="text-xs text-warn">
              {t("remoteInference.modalRun.noTailnetUrl")}
            </p>
          ) : null}
        </>
      ) : null}
    </div>
  );
};

export default ModalRunLine;
