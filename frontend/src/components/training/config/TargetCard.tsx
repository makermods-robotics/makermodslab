import React from "react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { ConfigComponentProps, RESUME_INHERITED_SHORT } from "../types";
import { RunnerFlavor } from "@/lib/jobsApi";

interface TargetCardProps extends ConfigComponentProps {
  authenticated: boolean;
  flavors: RunnerFlavor[];
  loading: boolean;
}

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

/** Where the run executes — the runner toggle plus whichever hardware control
 * that runner needs. Flat: the controls carry their own <Label>s and there is
 * no "Compute target" eyebrow above them, which used to restate the "Run
 * training on" label directly beneath it.
 *
 * Both the runner and the hardware are genuinely chosen per launch, including
 * on a resume: a continuation may cross runners in either direction (F7 — the
 * parent's checkpoint is fetched from the Hub for a cloud→local one, and
 * uploaded to it for a local→cloud one), so the toggle stays live and merely
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
  const target = config.target;

  const setRunner = (runner: "local" | "hf_cloud") => {
    if (runner === target.runner) return;
    if (runner === "local") {
      updateConfig("target", { runner: "local" });
    } else {
      // Preserve any previously-chosen flavor (may be undefined until picked).
      updateConfig("target", { runner: "hf_cloud", flavor: target.flavor });
    }
  };

  return (
    <section className="space-y-4">
      <div className="space-y-2">
        <Label>Compute</Label>
        <div className="grid grid-cols-2 overflow-hidden rounded-md border border-border text-sm">
          {(["local", "hf_cloud"] as const).map((r) => (
            <button
              key={r}
              type="button"
              onClick={() => setRunner(r)}
              className={cn(
                "px-3 py-1.5 transition-colors",
                target.runner === r
                  ? "bg-primary text-primary-foreground"
                  : "bg-background text-muted-foreground hover:text-foreground",
              )}
            >
              {r === "local" ? "Local — your machine" : "Hugging Face Cloud"}
            </button>
          ))}
        </div>
        {resumeLocked ? (
          <p className="text-xs text-muted-foreground">
            Defaults to the runner this run started on — switch it to continue
            somewhere else.
          </p>
        ) : null}
      </div>

      {target.runner === "local" ? (
        <div className="space-y-2">
          <Label htmlFor="policy_device">Device</Label>
          <Select
            value={config.policy_device === "cpu" ? "cpu" : "auto"}
            onValueChange={(value) => updateConfig("policy_device", value)}
            disabled={resumeLocked}
          >
            <SelectTrigger id="policy_device">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">
                Automatic (use GPU if available)
              </SelectItem>
              <SelectItem value="cpu">CPU</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            {resumeLocked
              ? RESUME_INHERITED_SHORT
              : "lerobot auto-detects your GPU (CUDA/MPS); only CPU is forced."}
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          <Label>Hardware</Label>
          <Select
            value={target.flavor ?? ""}
            onValueChange={(flavor) =>
              updateConfig("target", { runner: "hf_cloud", flavor })
            }
          >
            <SelectTrigger>
              <SelectValue
                placeholder={loading ? "Loading…" : "Select hardware"}
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
                      log in to HF
                    </span>
                  )}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            Cost shown is per running hour. Final policy uploads to your HF
            account when training completes.
          </p>
        </div>
      )}
    </section>
  );
};

export default TargetCard;
