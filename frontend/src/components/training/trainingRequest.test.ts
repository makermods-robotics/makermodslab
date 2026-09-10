import { describe, it, expect } from "vitest";

import { configToRequest } from "./trainingRequest";
import type { TrainingConfig } from "@/components/training/types";

const baseConfig: TrainingConfig = {
  target: { runner: "local" },
  dataset_repo_id: "",
  policy_type: "act",
  steps: 10000,
  batch_size: 8,
  num_workers: 4,
  log_freq: 50,
  save_freq: 1000,
  save_checkpoint: true,
  resume: false,
  wandb_enable: false,
  wandb_disable_artifact: false,
  policy_use_amp: false,
  use_policy_training_preset: true,
};

describe("configToRequest", () => {
  it("carries the config's dataset_repo_id through to the request", () => {
    const req = configToRequest(
      { ...baseConfig, dataset_repo_id: "ns/plain" },
      null,
    );
    expect(req.dataset_repo_id).toBe("ns/plain");
  });

  it("submits the combine-mode merged output when launchJob overrides it", () => {
    // The controlled dataset_repo_id is blank in combine mode; launchJob spreads
    // the minted temporary-merge id over it before submitting. Mirror that
    // exact expression here.
    const config: TrainingConfig = { ...baseConfig, dataset_repo_id: "" };
    const datasetOverride = "ns/mix-ab12";

    const req = configToRequest(
      datasetOverride
        ? { ...config, dataset_repo_id: datasetOverride }
        : config,
      null,
    );

    expect(req.dataset_repo_id).toBe("ns/mix-ab12");
  });
});
