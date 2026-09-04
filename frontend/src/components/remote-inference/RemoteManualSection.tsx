import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import type { RemoteInferenceTransportStatus } from "@/hooks/useRemoteInferenceTransport";
import ModalRunLine from "./ModalRunLine";
import type { RemoteRunConfig } from "./remoteRunConfig";

/**
 * "Run it yourself instead" — the hand-typed route, and the crib sheet that
 * goes with it.
 *
 * Kept even though the Lab launches the GPU itself: it is the only route when
 * `modal` is missing or unauthenticated, the only route to a hand-tuned flag,
 * and the ground truth an operator compares against when a run connects and
 * receives nothing.
 *
 * The four rows under the command are what the retired Transport section was
 * actually FOR. Everything else it showed (source, reachability, the operator
 * verdict) is now one sentence under Start, re-probed on its own; these four
 * are the values a human has to read with their eyes and retype somewhere else,
 * so they stay — beside the command they belong to rather than in a panel of
 * their own. Every one of them is data and appears verbatim.
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

const RemoteManualSection: React.FC<{
  config: RemoteRunConfig;
  transport: RemoteInferenceTransportStatus | null;
  hubIdDefault: string;
  task: string;
  /** The Modal target selected on the GPU card, so the pasted line bills the
   * SAME workspace Start GPU would. */
  profile: string;
  environment: string;
}> = ({ config, transport, hubIdDefault, task, profile, environment }) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
        <ChevronDown
          className={`h-3 w-3 transition-transform ${open ? "" : "-rotate-90"}`}
        />
        {t("remoteInference.modalRun.manualToggle")}
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-3 pt-2">
        <ModalRunLine
          config={config}
          transport={transport}
          hubIdDefault={hubIdDefault}
          task={task}
          profile={profile}
          environment={environment}
        />
        {transport?.sfu_enabled ? (
          <div className="space-y-1.5 rounded-md border border-border p-2">
            <Row label={t("remoteInference.transport.sfuModalUrlLabel")}>
              {transport.sfu_modal_url ? (
                <span className="font-mono">{transport.sfu_modal_url}</span>
              ) : (
                <span className="text-warn">
                  {t("remoteInference.transport.sfuNoTailnet")}
                </span>
              )}
            </Row>
            <Row label={t("remoteInference.transport.roomLabel")}>
              <span className="font-mono">
                {transport.room || t("remoteInference.transport.unresolved")}
              </span>
            </Row>
            <Row label={t("remoteInference.transport.sfuKeyIdLabel")}>
              {/* The key NAME. The secret is never sent here — the file below
                  is where a human reads it. */}
              <span className="font-mono">
                {transport.sfu_key_id ??
                  t("remoteInference.transport.unresolved")}
              </span>
            </Row>
            {transport.sfu_key_file ? (
              <Row label={t("remoteInference.transport.sfuKeyFileLabel")}>
                <span className="font-mono">{transport.sfu_key_file}</span>
              </Row>
            ) : null}
          </div>
        ) : null}
      </CollapsibleContent>
    </Collapsible>
  );
};

export default RemoteManualSection;
