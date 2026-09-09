import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

import { ApiProvider } from "@/contexts/ApiContext";
import { StudioProvider } from "@/contexts/StudioContext";
import { TooltipProvider } from "@/components/ui/tooltip";
import type { JobRecord, TrainingRequest } from "@/lib/jobsApi";
import ModelCard from "./ModelCard";

// A finished LOCAL training run, reduced to the fields ModelCard reads. Its
// lineage is empty (checkpoint_count 0), so the checkpoint-list effect
// early-returns and nothing hits the network.
const makeJobRecord = (
  over: Partial<Omit<JobRecord, "config">> & {
    config?: Partial<TrainingRequest>;
  } = {},
): JobRecord =>
  ({
    id: "run_1",
    job_number: 1,
    name: "act · ns/whatever",
    display_name: null,
    state: "done",
    config: { policy_type: "act", dataset_repo_id: "ns/plain", steps: 20000 },
    output_dir: "/jobs/run_1",
    started_at: 1_700_000_000,
    ended_at: 1_700_000_500,
    exit_code: 0,
    error_message: null,
    metrics: {},
    runner: "local",
    hf_job_id: null,
    hf_flavor: null,
    hf_repo_id: null,
    hf_job_url: null,
    wandb_run_url: null,
    checkpoint_count: 0,
    child_ids: [],
    ancestor_ids: [],
    ...over,
  }) as unknown as JobRecord;

const renderCard = (model: JobRecord) =>
  render(
    <ApiProvider>
      <StudioProvider>
        <TooltipProvider>
          <ModelCard model={model} onDelete={vi.fn()} onPlay={vi.fn()} />
        </TooltipProvider>
      </StudioProvider>
    </ApiProvider>,
  );

describe("ModelCard training-data block", () => {
  it("renders the training-data recipe from merge_provenance", () => {
    const model = makeJobRecord({
      config: { policy_type: "act", dataset_repo_id: "ns/merged-sentinel", steps: 20000 },
      merge_provenance: {
        merged_repo_id: "ns/sock-mix",
        weighted: true,
        temporary: true,
        sources: [
          { repo_id: "ns/sock-v1", weight: 1, episodes: 200 },
          { repo_id: "ns/sock-corrections", weight: 3, episodes: 30 },
          { repo_id: "ns/sock-demo", weight: 2, episodes: 100 },
        ],
      },
    });
    renderCard(model);

    expect(screen.getByText("ns/sock-corrections")).toBeInTheDocument();
    expect(screen.getByText(/×3/)).toBeInTheDocument();
    // units 200 : 90 : 200, total 490 => 41% / 18% / 41%
    expect(screen.getByText(/18%/)).toBeInTheDocument();
    expect(screen.getByText(/temporary merge/i)).toBeInTheDocument();

    // The single dataset row is REPLACED by the block, not shown alongside it.
    expect(screen.queryByText("ns/merged-sentinel")).not.toBeInTheDocument();
  });

  it("falls back to the single dataset row without merge_provenance", () => {
    renderCard(makeJobRecord({ merge_provenance: null }));
    expect(screen.getByText("ns/plain")).toBeInTheDocument();
    expect(screen.queryByText(/temporary merge/i)).not.toBeInTheDocument();
  });
});
