import React from "react";
import { useTranslation } from "react-i18next";
import { Router } from "lucide-react";
import { useNodes } from "@/hooks/useNodes";
import { JobRecord } from "@/lib/jobsApi";
import { nodeDisplayName } from "@/lib/nodesApi";

/** The job card's location chip for a `lan_node` run: the node's NAME, looked
 * up live in the registry by the record's instance id. A node that has since
 * left the registry degrades to the short instance id — the run still says
 * where it happened. Mounted only for lan_node runs, so the registry poll the
 * lookup rides on only runs when there is a node to name. */
const NodeLocationChip: React.FC<{ job: JobRecord }> = ({ job }) => {
  const { t } = useTranslation();
  const { nodes } = useNodes();
  const entry = job.node_instance_id
    ? (nodes.find((n) => n.instance_id === job.node_instance_id) ?? null)
    : null;
  // Name / id are data — rendered verbatim; only the fallback word and the
  // hover sentence are copy.
  const label = entry
    ? nodeDisplayName(entry)
    : (job.node_instance_id?.slice(0, 8) ?? t("jobs.location.node"));
  return (
    <div
      className="flex items-center gap-1 text-[11px] font-medium text-muted-foreground"
      title={
        entry
          ? t("jobs.location.nodeTitleNamed", { name: label })
          : t("jobs.location.nodeTitle")
      }
    >
      <Router className="w-3 h-3" />
      {label}
    </div>
  );
};

export default NodeLocationChip;
