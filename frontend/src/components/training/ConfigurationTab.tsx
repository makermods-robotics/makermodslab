import React from "react";
import EssentialsCard from "./config/EssentialsCard";
import AdvancedCard from "./config/AdvancedCard";
import PolicyField from "./config/PolicyField";
import TargetCard from "./config/TargetCard";
import { ConfigComponentProps } from "./types";
import { RunnerFlavor } from "@/lib/jobsApi";

interface ConfigurationTabProps extends ConfigComponentProps {
  authenticated: boolean;
  flavors: RunnerFlavor[];
  hardwareLoading: boolean;
  /** True when a base skill (fine-tune) or resume seed fixes the policy —
   * the run must train the source checkpoint's architecture. */
  policyLocked?: boolean;
  /** True when a resume seed fixes the compute target — a resume can only
   * continue on the parent run's runner (F7). Narrower than `resumeLocked`:
   * that one covers the hyperparameters lerobot rebuilds from the checkpoint,
   * this one is only about WHERE the continuation executes, and it leaves the
   * cloud flavor editable. */
  runnerLocked?: boolean;
}

const ConfigurationTab: React.FC<ConfigurationTabProps> = ({
  config,
  updateConfig,
  authenticated,
  flavors,
  hardwareLoading,
  policyLocked,
  resumeLocked,
  runnerLocked,
}) => {
  return (
    // Order matters: Policy answers "what am I training" and so belongs with
    // the Train panel's Dataset / Starting point above it, before the form
    // moves on to where the run executes and how long it trains.
    <div className="space-y-6">
      <PolicyField
        config={config}
        updateConfig={updateConfig}
        policyLocked={policyLocked}
      />
      <TargetCard
        config={config}
        updateConfig={updateConfig}
        authenticated={authenticated}
        flavors={flavors}
        loading={hardwareLoading}
        resumeLocked={resumeLocked}
        runnerLocked={runnerLocked}
      />
      <EssentialsCard
        config={config}
        updateConfig={updateConfig}
        resumeLocked={resumeLocked}
      />
      <AdvancedCard
        config={config}
        updateConfig={updateConfig}
        resumeLocked={resumeLocked}
      />
    </div>
  );
};

export default ConfigurationTab;
