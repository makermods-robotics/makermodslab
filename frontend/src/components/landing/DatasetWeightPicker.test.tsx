import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";

import { DatasetWeightPicker } from "./DatasetWeightPicker";

describe("DatasetWeightPicker", () => {
  it("shows each source's post-weight share", () => {
    render(
      <DatasetWeightPicker
        maxWeight={20}
        value={[
          { repo_id: "ns/a", weight: 1, baseEpisodes: 200 },
          { repo_id: "ns/b", weight: 3, baseEpisodes: 30 },
        ]}
        onChange={() => {}}
      />,
    );
    // 200 vs 90 => 69% / 31%
    expect(screen.getByText(/69%/)).toBeInTheDocument();
    expect(screen.getByText(/31%/)).toBeInTheDocument();
  });

  it("leaves the share unavailable when a source's episode count is unknown", () => {
    render(
      <DatasetWeightPicker
        maxWeight={20}
        value={[
          { repo_id: "ns/a", weight: 1, baseEpisodes: 200 },
          { repo_id: "ns/b", weight: 2, baseEpisodes: null },
        ]}
        onChange={() => {}}
      />,
    );
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
  });

  it("raises a weight through the stepper", () => {
    const onChange = vi.fn();
    render(
      <DatasetWeightPicker
        maxWeight={20}
        value={[{ repo_id: "ns/a", weight: 2, baseEpisodes: 10 }]}
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByLabelText(/increase weight/i));
    expect(onChange).toHaveBeenCalledWith([
      { repo_id: "ns/a", weight: 3, baseEpisodes: 10 },
    ]);
  });

  it("clamps weight to maxWeight", () => {
    const onChange = vi.fn();
    render(
      <DatasetWeightPicker
        maxWeight={5}
        value={[{ repo_id: "ns/a", weight: 5, baseEpisodes: 10 }]}
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByLabelText(/increase weight/i));
    expect(onChange).not.toHaveBeenCalledWith([
      expect.objectContaining({ weight: 6 }),
    ]);
  });
});
