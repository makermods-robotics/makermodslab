import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Loader2, Plus, RefreshCw } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { useEyebrowClass } from "@/hooks/useEyebrowClass";
import { ApiError } from "@/lib/apiClient";
import {
  NodeEntry,
  addNode,
  isSelectableNode,
  listableNodes,
  nodeDisplayName,
  nodeGpuLabel,
} from "@/lib/nodesApi";
import { relativeTimeAgo } from "@/lib/relativeTime";
import { TrainingConfig } from "../types";
import { cn } from "@/lib/utils";

type Target = TrainingConfig["target"];

interface ComputeSelectorProps {
  target: Target;
  /** The raw registry listing (self entry included — filtered here). */
  nodes: NodeEntry[];
  nodesLoading: boolean;
  onSelect: (target: Target) => void;
  /** Fired after a successful Add so the owner can refetch the registry. */
  onNodeAdded: (node: NodeEntry) => void;
  /** Manual refetch of the registry AND the selected node's workload. */
  onRefresh: () => void;
}

/** The little radio glyph. Purely presentational — the row is the control. */
const RadioDot: React.FC<{ checked: boolean; dimmed?: boolean }> = ({
  checked,
  dimmed,
}) => (
  <span
    aria-hidden
    className={cn(
      "relative h-3.5 w-3.5 shrink-0 rounded-full border",
      checked ? "border-primary" : "border-muted-foreground/60",
      dimmed && "opacity-40",
    )}
  >
    {checked ? (
      <span className="absolute inset-[2.5px] rounded-full bg-primary" />
    ) : null}
  </span>
);

interface OptionRowProps {
  checked: boolean;
  disabled?: boolean;
  title?: string;
  onPick: () => void;
  /** ok | bad | pending — the node rows' status dot; undefined ⇒ no dot. */
  dot?: "ok" | "bad" | "pending";
  children: React.ReactNode;
  chips?: React.ReactNode;
}

/** One radio row: dot · title/sub body · chips. Disabled rows stay VISIBLE
 * and focusable-for-hover (aria-disabled, reason in `title`) — an unreachable
 * node reads as unreachable, never as deleted. */
const OptionRow: React.FC<OptionRowProps> = ({
  checked,
  disabled,
  title,
  onPick,
  dot,
  children,
  chips,
}) => (
  <button
    type="button"
    role="radio"
    aria-checked={checked}
    aria-disabled={disabled || undefined}
    title={title}
    onClick={() => {
      if (!disabled) onPick();
    }}
    className={cn(
      "flex w-full items-center gap-2.5 bg-background px-3 py-2 text-left transition-colors",
      checked && "bg-accent/60",
      disabled
        ? "cursor-not-allowed"
        : "text-muted-foreground hover:text-foreground",
    )}
  >
    <RadioDot checked={checked} dimmed={disabled} />
    {dot ? (
      <span
        aria-hidden
        className={cn(
          "h-[7px] w-[7px] shrink-0 rounded-full",
          dot === "ok" && "bg-ok",
          dot === "bad" && "bg-warn",
          dot === "pending" && "animate-pulse bg-muted-foreground/60",
        )}
      />
    ) : null}
    <span className="flex min-w-0 flex-1 flex-col gap-0.5">{children}</span>
    {chips ? (
      <span
        className={cn(
          "flex shrink-0 items-center gap-1.5",
          disabled && "opacity-50",
        )}
      >
        {chips}
      </span>
    ) : null}
  </button>
);

const Chip: React.FC<{
  children: React.ReactNode;
  mono?: boolean;
  accent?: boolean;
  title?: string;
}> = ({ children, mono, accent, title }) => (
  <span
    title={title}
    className={cn(
      "whitespace-nowrap rounded border px-1.5 py-px text-[11px] font-medium",
      mono && "font-mono",
      accent
        ? "border-info/35 text-info"
        : "border-border text-muted-foreground",
    )}
  >
    {children}
  </span>
);

/**
 * The Compute control: a radio-row list — This machine, Hugging Face Cloud,
 * then the LAN NODES group with an inline Add-node row. Replaces the old
 * 2-segment grid, which had no room for N nodes with status/GPU metadata.
 *
 * Honesty rules the rows follow:
 *  - the browser NEVER switches servers — a node runs the job, driven
 *    server-to-server (the hint under the list says so);
 *  - an unreachable node is disabled-but-visible with its last-seen age;
 *  - a still-verifying discovery candidate shows as such, unselectable;
 *  - a SELECTED node that vanishes from the registry keeps its (flagged) row
 *    rather than the form silently flipping back to local — submission then
 *    surfaces the server's own coded refusal.
 *
 * Node names, URLs, versions, instance ids and GPU strings are DATA and render
 * verbatim; only the framing copy is translated.
 */
const ComputeSelector: React.FC<ComputeSelectorProps> = ({
  target,
  nodes,
  nodesLoading,
  onSelect,
  onNodeAdded,
  onRefresh,
}) => {
  const { t } = useTranslation();
  const { baseUrl, fetchWithHeaders } = useApi();
  const eyebrow = useEyebrowClass();

  const [addOpen, setAddOpen] = useState(false);
  const [addUrl, setAddUrl] = useState("");
  const [adding, setAdding] = useState(false);
  const [addError, setAddError] = useState<string | null>(null);

  const peers = listableNodes(nodes);
  const selectedNodeId =
    target.runner === "lan_node" ? (target.node_instance_id ?? null) : null;
  // The selection outliving the registry: keep a flagged row for it.
  const selectedGone =
    selectedNodeId != null &&
    !peers.some((n) => n.instance_id === selectedNodeId);

  const pickNode = (node: NodeEntry) => {
    if (!node.instance_id) return;
    onSelect({ runner: "lan_node", node_instance_id: node.instance_id });
  };

  const submitAdd = async () => {
    const url = addUrl.trim();
    if (!url || adding) return;
    setAdding(true);
    setAddError(null);
    try {
      const added = await addNode(baseUrl, fetchWithHeaders, url);
      setAddUrl("");
      setAddOpen(false);
      onNodeAdded(added);
      // A node added by hand was added to be used — select it when it can be.
      if (isSelectableNode(added)) pickNode(added);
    } catch (e) {
      // Branch on the CODE, never the prose (apiClient.ts). Uncoded refusals
      // show the server's own detail verbatim.
      const code = e instanceof ApiError ? e.code : null;
      setAddError(
        code === "node.self"
          ? t("training.target.addNode.errorSelf")
          : code === "node.duplicate"
            ? t("training.target.addNode.errorDuplicate")
            : code === "node.unreachable"
              ? t("training.target.addNode.errorUnreachable")
              : e instanceof ApiError && e.detail
                ? e.detail
                : e instanceof Error
                  ? e.message
                  : String(e),
      );
    } finally {
      setAdding(false);
    }
  };

  const nodeRow = (node: NodeEntry) => {
    const name = nodeDisplayName(node);
    const gpu = nodeGpuLabel(node);
    const chips =
      gpu || node.source === "tailscale" ? (
        <>
          {gpu ? <Chip mono>{gpu}</Chip> : null}
          {node.source === "tailscale" ? (
            // "tailscale" is the discovery source's id (and a product name) —
            // shown as-is inside a translated framing word.
            <Chip accent>{t("training.target.viaTailscale")}</Chip>
          ) : null}
        </>
      ) : null;
    const checked = node.instance_id != null && node.instance_id === selectedNodeId;
    const lastSeen =
      node.last_seen_at != null
        ? relativeTimeAgo(node.last_seen_at * 1000)
        : null;

    if (node.status === "pending") {
      return (
        <OptionRow
          key={node.url ?? name}
          checked={false}
          disabled
          dot="pending"
          title={t("training.target.verifyingTitle")}
          onPick={() => {}}
          chips={chips}
        >
          <span className="truncate text-sm font-medium">{name}</span>
          <span className="text-xs text-muted-foreground">
            {t("training.target.verifying")}
          </span>
        </OptionRow>
      );
    }

    const unreachable = node.status !== "ok";
    const selectable = isSelectableNode(node);
    return (
      <OptionRow
        key={node.instance_id ?? node.url ?? name}
        checked={checked}
        // A selected node that turns unreachable keeps its checked row —
        // disabled against RE-picking, never silently dropped.
        disabled={!selectable}
        dot={unreachable ? "bad" : "ok"}
        title={
          unreachable
            ? lastSeen
              ? t("training.target.unreachableTitleLastSeen", { when: lastSeen })
              : t("training.target.unreachableTitle")
            : undefined
        }
        onPick={() => pickNode(node)}
      chips={chips}
      >
        <span
          className={cn(
            "truncate text-sm font-medium",
            checked && "text-foreground",
            !selectable && "opacity-50",
          )}
        >
          {name}
        </span>
        <span className="truncate text-xs text-muted-foreground">
          {unreachable ? (
            <>
              <span className="text-warn">
                {t("training.target.unreachable")}
              </span>
              {lastSeen ? (
                <span>
                  {" · "}
                  {t("training.target.lastSeen", { when: lastSeen })}
                </span>
              ) : null}
            </>
          ) : node.version ? (
            // "makermodslab" is the product/package name; the version is data.
            t("training.target.nodeVersion", { version: node.version })
          ) : (
            node.url
          )}
        </span>
      </OptionRow>
    );
  };

  return (
    <div
      role="radiogroup"
      aria-label={t("training.target.computeLabel")}
      className="divide-y divide-border overflow-hidden rounded-md border border-border"
    >
      <OptionRow
        checked={target.runner === "local"}
        onPick={() => onSelect({ runner: "local" })}
      >
        <span
          className={cn(
            "text-sm font-medium",
            target.runner === "local" && "text-foreground",
          )}
        >
          {t("training.target.thisMachine")}
        </span>
        <span className="text-xs text-muted-foreground">
          {t("training.target.thisMachineSub")}
        </span>
      </OptionRow>
      <OptionRow
        checked={target.runner === "hf_cloud"}
        // Preserve any previously-chosen flavor (may be undefined until picked).
        onPick={() => onSelect({ runner: "hf_cloud", flavor: target.flavor })}
      >
        <span
          className={cn(
            "text-sm font-medium",
            target.runner === "hf_cloud" && "text-foreground",
          )}
        >
          {t("training.target.runnerCloud")}
        </span>
        <span className="text-xs text-muted-foreground">
          {t("training.target.cloudSub")}
        </span>
      </OptionRow>

      <div className="flex items-center justify-between gap-2 bg-muted/30 px-3 pb-1 pt-1.5">
        <span className={eyebrow}>
          {t("training.target.lanNodes")}
        </span>
        <button
          type="button"
          onClick={onRefresh}
          title={t("training.target.refreshNodes")}
          aria-label={t("training.target.refreshNodes")}
          className="flex items-center rounded px-1.5 py-0.5 text-xs text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground"
        >
          <RefreshCw className="h-3 w-3" />
        </button>
        <button
          type="button"
          onClick={() => {
            setAddOpen((v) => !v);
            setAddError(null);
          }}
          title={t("training.target.addNode.title")}
          className="flex items-center gap-1 rounded px-1.5 py-0.5 text-xs text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground"
        >
          <Plus className="h-3 w-3" />
          {t("training.target.addNode.button")}
        </button>
      </div>

      {nodesLoading && peers.length === 0 ? (
        <div className="flex items-center gap-2 bg-background px-3 py-2.5 text-xs text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" />
          {t("training.target.nodesLoading")}
        </div>
      ) : peers.length === 0 && !selectedGone ? (
        <div className="bg-background px-3 py-2.5 text-xs text-muted-foreground">
          {t("training.target.nodesEmpty")}
        </div>
      ) : (
        <>
          {peers.map(nodeRow)}
          {selectedGone && selectedNodeId ? (
            // The chosen node left the registry: keep the selection VISIBLE
            // and flagged. Submission surfaces the server's coded refusal —
            // this form never silently flips a choice back to local.
            <OptionRow
              checked
              disabled
              dot="bad"
              title={t("training.target.nodeGoneTitle")}
              onPick={() => {}}
            >
              <span className="truncate font-mono text-sm font-medium opacity-70">
                {selectedNodeId.slice(0, 8)}
              </span>
              <span className="truncate text-xs text-warn">
                {t("training.target.nodeGone")}
              </span>
            </OptionRow>
          ) : null}
        </>
      )}

      {addOpen ? (
        <div className="space-y-1.5 bg-muted/20 px-3 py-2.5">
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={addUrl}
              onChange={(e) => {
                setAddUrl(e.target.value);
                setAddError(null);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void submitAdd();
                }
              }}
              // The placeholder is a literal URL shape the user must match.
              placeholder="http://bench-rig.local:8000"
              aria-label={t("training.target.addNode.urlLabel")}
              className="h-8 min-w-0 flex-1 rounded-md border border-input bg-background px-2.5 font-mono text-xs text-foreground placeholder:text-muted-foreground/70 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            />
            <button
              type="button"
              onClick={() => void submitAdd()}
              disabled={adding || !addUrl.trim()}
              className="flex h-8 shrink-0 items-center gap-1.5 rounded-md border border-border px-3 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground disabled:cursor-not-allowed disabled:opacity-60"
            >
              {adding ? <Loader2 className="h-3 w-3 animate-spin" /> : null}
              {adding
                ? t("training.target.addNode.adding")
                : t("training.target.addNode.submit")}
            </button>
          </div>
          {addError ? (
            <p className="text-xs text-destructive">{addError}</p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
};

export default ComputeSelector;
