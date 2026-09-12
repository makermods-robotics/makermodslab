import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { EpisodeSummary } from "@/lib/replayApi";

const mocks = vi.hoisted(() => ({
  fetch: vi.fn(),
  toast: vi.fn(),
  listEpisodes: vi.fn(),
  getDatasetInfo: vi.fn(async () => ({ cameras: [] })),
  getExcludedEpisodes: vi.fn(async () => []),
  setExcludedEpisodes: vi.fn(async () => {}),
  deleteEpisodes: vi.fn(),
}));

vi.mock("@/contexts/ApiContext", () => ({
  useApi: () => ({ baseUrl: "http://test", fetchWithHeaders: mocks.fetch }),
}));
vi.mock("@/contexts/LanguageContext", () => ({
  useLanguage: () => ({ language: "en" }),
}));
vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({ toast: mocks.toast }),
}));
vi.mock("@/contexts/StudioContext", () => ({
  useStudio: () => ({ openStudio: vi.fn() }),
}));
vi.mock("@/hooks/useSelectedDataset", () => ({
  useSelectedDataset: () => ({ setSelectedDataset: vi.fn() }),
}));
vi.mock("@/components/landing/DatasetInfoCard", () => ({
  default: ({ canDelete, onDelete }: { canDelete?: boolean; onDelete?: () => void }) =>
    canDelete ? (
      <button type="button" onClick={onDelete}>
        info-card-delete
      </button>
    ) : null,
}));
vi.mock("@/components/dialogs/JointPositionChart", () => ({ default: () => null }));
vi.mock("@/components/dialogs/EpisodeReplayPanel", () => ({ default: () => null }));
vi.mock("@/lib/replayApi", async () => {
  const actual = await vi.importActual<typeof import("@/lib/replayApi")>("@/lib/replayApi");
  return {
    ...actual,
    listEpisodes: mocks.listEpisodes,
    getDatasetInfo: mocks.getDatasetInfo,
    getExcludedEpisodes: mocks.getExcludedEpisodes,
    setExcludedEpisodes: mocks.setExcludedEpisodes,
    deleteEpisodes: mocks.deleteEpisodes,
    episodeVideoUrl: () => "http://test/video.mp4",
    getEpisodeJoints: vi.fn(async () => null),
  };
});

import DatasetDetailDialog from "./DatasetDetailDialog";
import type { DatasetItem } from "@/lib/replayApi";

const LOCAL_ITEM: DatasetItem = {
  repo_id: "makermods/ds",
  source: "local",
  last_modified: null,
  private: false,
};

const episode = (index: number): EpisodeSummary => ({
  episode_index: index,
  length: 100,
  duration: 10,
  tasks: ["pick"],
  video_offsets: {},
});

beforeEach(() => {
  vi.clearAllMocks();
  mocks.getDatasetInfo.mockResolvedValue({ cameras: [] });
  mocks.getExcludedEpisodes.mockResolvedValue([]);
});

afterEach(() => cleanup());

const setup = (props: Partial<React.ComponentProps<typeof DatasetDetailDialog>> = {}) => {
  const onOpenChange = vi.fn();
  const onDeleted = vi.fn();
  render(
    <DatasetDetailDialog
      repoId="makermods/ds"
      open
      onOpenChange={onOpenChange}
      onDeleted={onDeleted}
      {...props}
    />,
  );
  return { onOpenChange, onDeleted };
};

describe("per-episode delete", () => {
  it("deletes the episode and refreshes the list after confirming", async () => {
    mocks.listEpisodes.mockResolvedValue([episode(0), episode(1)]);
    mocks.deleteEpisodes.mockResolvedValue({ success: true, whole_dataset_deleted: false });
    setup();

    await screen.findByText("Episode 1");
    fireEvent.click(screen.getByRole("button", { name: /delete episode 1/i }));
    fireEvent.click(screen.getByRole("button", { name: /^delete episode$/i }));

    await waitFor(() =>
      expect(mocks.deleteEpisodes).toHaveBeenCalledWith(
        "http://test",
        mocks.fetch,
        "makermods/ds",
        [1],
      ),
    );
    // Refetches the episode list rather than relabeling client-side.
    await waitFor(() => expect(mocks.listEpisodes).toHaveBeenCalledTimes(2));
  });

  it("warns that the Hub copy is untouched when the dataset is also on the Hub", async () => {
    mocks.listEpisodes.mockResolvedValue([episode(0), episode(1)]);
    setup({ item: { ...LOCAL_ITEM, source: "both" } });

    await screen.findByText("Episode 1");
    fireEvent.click(screen.getByRole("button", { name: /delete episode 1/i }));
    expect(screen.getByText(/hub/i)).toBeInTheDocument();
  });

  it("says nothing about the Hub for a local-only dataset", async () => {
    mocks.listEpisodes.mockResolvedValue([episode(0), episode(1)]);
    setup({ item: LOCAL_ITEM });

    await screen.findByText("Episode 1");
    fireEvent.click(screen.getByRole("button", { name: /delete episode 1/i }));
    expect(screen.queryByText(/hub/i)).not.toBeInTheDocument();
  });

  it("is not offered while curating episodes for training", async () => {
    mocks.listEpisodes.mockResolvedValue([episode(0)]);
    setup();

    await screen.findByText("Episode 0");
    fireEvent.click(screen.getByRole("button", { name: /curate episodes/i }));
    expect(screen.queryByRole("button", { name: /delete episode 0/i })).not.toBeInTheDocument();
  });
});

describe("whole-dataset delete via the info card", () => {
  it("is hidden without an `item` prop (unknown Hub/local status)", async () => {
    mocks.listEpisodes.mockResolvedValue([episode(0)]);
    setup();
    await screen.findByText("Episode 0");
    expect(screen.queryByText("info-card-delete")).not.toBeInTheDocument();
  });

  it("closes the dialog and reports onDeleted for a local-only dataset", async () => {
    mocks.listEpisodes.mockResolvedValue([episode(0)]);
    mocks.fetch.mockResolvedValue({
      ok: true,
      json: async () => ({ success: true }),
    });
    const { onOpenChange, onDeleted } = setup({
      item: LOCAL_ITEM,
    });
    await screen.findByText("Episode 0");

    fireEvent.click(screen.getByText("info-card-delete"));
    fireEvent.click(screen.getByRole("button", { name: /^delete$/i }));

    await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false));
    expect(onDeleted).toHaveBeenCalled();
  });
});

describe("Finalize review mode", () => {
  const finalize = () => ({ onFinalize: vi.fn(), onDiscarded: vi.fn() });

  it("starts every episode checked and shows the Finalize button instead of Train", async () => {
    mocks.listEpisodes.mockResolvedValue([episode(0), episode(1)]);
    setup({ finalize: finalize() });

    await screen.findByText("Episode 1");
    expect(screen.queryByRole("button", { name: /train a policy/i })).not.toBeInTheDocument();
    const finalizeBtn = screen.getByRole("button", { name: /^finalize$/i });
    expect(finalizeBtn).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /keep episode 0/i })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: /keep episode 1/i })).toBeChecked();
  });

  it("hides the training-curation toggle and the info card's delete affordance", async () => {
    mocks.listEpisodes.mockResolvedValue([episode(0)]);
    setup({ finalize: finalize(), item: LOCAL_ITEM });

    await screen.findByText("Episode 0");
    expect(screen.queryByRole("button", { name: /curate episodes/i })).not.toBeInTheDocument();
    expect(screen.queryByText("info-card-delete")).not.toBeInTheDocument();
  });

  it("deletes only the unchecked episodes and calls onFinalize, leaving the checked ones untouched", async () => {
    mocks.listEpisodes.mockResolvedValue([episode(0), episode(1), episode(2)]);
    mocks.deleteEpisodes.mockResolvedValue({ success: true, whole_dataset_deleted: false });
    const f = finalize();
    setup({ finalize: f });

    await screen.findByText("Episode 1");
    fireEvent.click(screen.getByRole("checkbox", { name: /keep episode 1/i }));
    fireEvent.click(screen.getByRole("button", { name: /^finalize$/i }));

    await waitFor(() =>
      expect(mocks.deleteEpisodes).toHaveBeenCalledWith(
        "http://test",
        mocks.fetch,
        "makermods/ds",
        [1],
      ),
    );
    await waitFor(() => expect(f.onFinalize).toHaveBeenCalled());
    expect(f.onDiscarded).not.toHaveBeenCalled();
  });

  it("skips the delete call entirely when every episode stays checked", async () => {
    mocks.listEpisodes.mockResolvedValue([episode(0)]);
    const f = finalize();
    setup({ finalize: f });

    await screen.findByText("Episode 0");
    fireEvent.click(screen.getByRole("button", { name: /^finalize$/i }));

    await waitFor(() => expect(f.onFinalize).toHaveBeenCalled());
    expect(mocks.deleteEpisodes).not.toHaveBeenCalled();
  });

  it("reports onDiscarded instead of onFinalize when every episode is unchecked", async () => {
    mocks.listEpisodes.mockResolvedValue([episode(0), episode(1)]);
    mocks.deleteEpisodes.mockResolvedValue({ success: true, whole_dataset_deleted: true });
    const f = finalize();
    const { onOpenChange, onDeleted } = setup({ finalize: f });

    await screen.findByText("Episode 1");
    fireEvent.click(screen.getByRole("checkbox", { name: /keep episode 0/i }));
    fireEvent.click(screen.getByRole("checkbox", { name: /keep episode 1/i }));
    fireEvent.click(screen.getByRole("button", { name: /^finalize$/i }));

    await waitFor(() => expect(f.onDiscarded).toHaveBeenCalled());
    expect(f.onFinalize).not.toHaveBeenCalled();
    expect(onOpenChange).toHaveBeenCalledWith(false);
    expect(onDeleted).toHaveBeenCalled();
  });

  it("cannot be dismissed by escape or outside click", async () => {
    mocks.listEpisodes.mockResolvedValue([episode(0)]);
    const { onOpenChange } = setup({ finalize: finalize() });
    await screen.findByText("Episode 0");

    fireEvent.keyDown(document, { key: "Escape" });
    expect(onOpenChange).not.toHaveBeenCalled();
  });
});
