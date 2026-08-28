import React from "react";
import { Trans, useTranslation } from "react-i18next";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useNodes } from "@/hooks/useNodes";
import { listableNodes, nodeDisplayName } from "@/lib/nodesApi";
import { ConfigComponentProps, RESUME_INHERITED_SHORT_KEY } from "../types";
import { RunnerFlavor } from "@/lib/jobsApi";
import ComputeSelector from "./ComputeSelector";
import NodeDetailPanel from "./NodeDetailPanel";

interface TargetCardProps extends ConfigComponentProps {
  authenticated: boolean;
  flavors: RunnerFlavor[];
  loading: boolean;
}

// Currency and number formatting are deliberately NOT locale-aware: HF Jobs
// prices are quoted in USD, and the figure is the vendor's, not ours.
const formatHourly = (unitCostUsd: number, unitLabel: string): string => {
  const hourly = unitLabel === "minute" ? unitCostUsd * 60 : unitCostUsd;
  return `$${hourly.toFixed(2)}/hr`;
};

// VRAM is included because it is the one spec that decides whether a run
// starts at all: the big VLA policies OOM on a small card at the standard
// batch size, and the failure lands on the first training step, minutes after
// the user has already paid for the box.
const formatFlavorLine = (f: RunnerFlavor): string => {
  const accel = f.accelerator
    ? f.vram
      ? `${f.accelerator} · ${f.vram} VRAM`
      : f.accelerator
    : f.cpu;
  return `${f.pretty_name} · ${accel} · ${formatHourly(f.unit_cost_usd, f.unit_label)}`;
};

/** Where the run executes — the Compute radio-row list (ComputeSelector: this
 * machine / HF Cloud / registered LAN nodes) plus whichever contextual control
 * the chosen runner needs: the Device picker for local, the Hardware picker
 * for the cloud, the node detail panel (identity + live workload) for a node.
 *
 * Both the runner and the hardware are genuinely chosen per launch, including
 * on a resume: a continuation may cross runners in either direction (F7 — the
 * parent's checkpoint is fetched from the Hub for a cloud→local one, and
 * uploaded to it for a local→cloud one), so the selector stays live and merely
 * DEFAULTS to the parent's runner. `policy_device` is still locked on a resume,
 * for an unrelated reason: the resume branch emits no --policy.device, so
 * lerobot uses whatever the checkpoint's train_config.json recorded. */
const TargetCard: React.FC<TargetCardProps> = ({
  config,
  updateConfig,
  authenticated,
  flavors,
  loading,
  resumeLocked,
}) => {
  const { t } = useTranslation();
  const target = config.target;
  const {
    nodes,
    sources: nodeSources,
    loading: nodesLoading,
    refresh: refreshNodes,
    forceRefresh: forceRefreshNodes,
  } = useNodes();
  // Manual refresh: a FORCED registry pass (the server probes everything now,
  // TTL notwithstanding) plus a token bump so NodeDetailPanel refetches the
  // workload in the same gesture.
  const [nodesRefreshToken, setNodesRefreshToken] = React.useState(0);
  const refreshNodesAndWorkload = React.useCallback(() => {
    forceRefreshNodes();
    setNodesRefreshToken((v) => v + 1);
  }, [forceRefreshNodes]);

  const selectedNode =
    target.runner === "lan_node" && target.node_instance_id
      ? (listableNodes(nodes).find(
          (n) => n.instance_id === target.node_instance_id,
        ) ?? null)
      : null;

  // The live "Run on: X" summary in the section header. The node's name is
  // data; a vanished node degrades to its short instance id.
  const runOnName =
    target.runner === "local"
      ? t("training.target.thisMachine")
      : target.runner === "hf_cloud"
        ? t("training.target.runnerCloud")
        : selectedNode
          ? nodeDisplayName(selectedNode)
          : (target.node_instance_id?.slice(0, 8) ?? "?");

  return (
    <section className="space-y-4">
      <div className="space-y-2">
        <div className="flex items-baseline justify-between gap-2">
          <Label>{t("training.target.computeLabel")}</Label>
          <span className="text-xs text-muted-foreground">
            <Trans
              i18nKey="training.target.runOn"
              values={{ name: runOnName }}
              components={[
                <strong key="0" className="font-semibold text-foreground" />,
              ]}
            />
          </span>
        </div>
        <ComputeSelector
          target={target}
          nodes={nodes}
          sources={nodeSources}
          nodesLoading={nodesLoading}
          onSelect={(next) => updateConfig("target", next)}
          onNodeAdded={() => void refreshNodes()}
          onRefresh={refreshNodesAndWorkload}
        />
        <p className="text-xs text-muted-foreground">
          {resumeLocked
            ? t("training.target.resumeRunnerHint")
            : t("training.target.selectorHint")}
        </p>
      </div>

      {target.runner === "local" ? (
        <div className="space-y-2">
          <Label htmlFor="policy_device">{t("training.target.deviceLabel")}</Label>
          <Select
            value={config.policy_device === "cpu" ? "cpu" : "auto"}
            onValueChange={(value) => updateConfig("policy_device", value)}
            disabled={resumeLocked}
          >
            <SelectTrigger id="policy_device">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {/* Values are wire settings; only the labels are copy. */}
              <SelectItem value="auto">
                {t("training.target.deviceAuto")}
              </SelectItem>
              <SelectItem value="cpu">{t("training.target.deviceCpu")}</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            {resumeLocked
              ? t(RESUME_INHERITED_SHORT_KEY)
              : t("training.target.deviceHint")}
          </p>
        </div>
      ) : target.runner === "hf_cloud" ? (
        <div className="space-y-2">
          <Label>{t("training.target.hardwareLabel")}</Label>
          <Select
            value={target.flavor ?? ""}
            onValueChange={(flavor) =>
              updateConfig("target", { runner: "hf_cloud", flavor })
            }
          >
            <SelectTrigger>
              <SelectValue
                placeholder={
                  loading
                    ? t("training.target.hardwareLoading")
                    : t("training.target.hardwarePlaceholder")
                }
              />
            </SelectTrigger>
            <SelectContent>
              {flavors.map((f) => (
                <SelectItem
                  key={f.name}
                  value={f.name}
                  disabled={!authenticated}
                >
                  {formatFlavorLine(f)}
                  {!authenticated && (
                    <span className="text-warn ml-2 text-xs">
                      {t("training.target.loginToHf")}
                    </span>
                  )}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            {t("training.target.costHint")}
          </p>
        </div>
      ) : target.node_instance_id ? (
        <NodeDetailPanel
          node={selectedNode}
          instanceId={target.node_instance_id}
          refreshToken={nodesRefreshToken}
        />
      ) : null}
    </section>
  );
};

export default TargetCard;
