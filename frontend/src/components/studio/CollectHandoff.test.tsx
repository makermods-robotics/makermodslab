import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import { StudioProvider } from "@/contexts/StudioContext";

// The upload hook talks to the backend and polls; the handoff's contract here
// is only that it ASKS for the push, so the transport is stubbed out.
const start = vi.fn(async () => null);
vi.mock("@/hooks/useDatasetUpload", () => ({
  useDatasetUpload: () => ({ uploading: false, start }),
}));

const setSelectedDataset = vi.fn();
vi.mock("@/hooks/useSelectedDataset", () => ({
  useSelectedDataset: () => ({ selectedDataset: null, setSelectedDataset }),
}));

import CollectHandoff from "./CollectHandoff";

const setup = (
  recorded: React.ComponentProps<typeof CollectHandoff>["recorded"],
) => {
  const onDismiss = vi.fn();
  render(
    <StudioProvider>
      <CollectHandoff recorded={recorded} onDismiss={onDismiss} />
    </StudioProvider>,
  );
  return { onDismiss };
};

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
});

describe("the handoff banner", () => {
  it("renders nothing before a session has finished", () => {
    const { container } = render(
      <StudioProvider>
        <CollectHandoff recorded={null} onDismiss={vi.fn()} />
      </StudioProvider>,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("names the saved dataset and offers the next step", () => {
    setup({ repo_id: "makermods/sock_sort", saved_episodes: 5 });
    expect(screen.getByText(/makermods\/sock_sort/)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /train on this/i }),
    ).toBeInTheDocument();
  });

  it("hands dismissal back to the owner of the payload", () => {
    const { onDismiss } = setup({ repo_id: "makermods/sock_sort" });
    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
    expect(onDismiss).toHaveBeenCalled();
  });
});

// This is the reason the banner is rendered unconditionally in the
// always-mounted Collect panel rather than only when someone is looking at it.
// It used to run because a finished session NAVIGATED to the Launchpad,
// mounting the banner there; the studio now stays open, so nothing else would
// trigger it.
describe("the side effect that used to ride on the navigation home", () => {
  it("preselects the fresh dataset so Train opens onto it", () => {
    setup({ repo_id: "makermods/preselect", saved_episodes: 5 });
    expect(setSelectedDataset).toHaveBeenCalledWith("makermods/preselect");
  });
});

// The automatic first Hub push moved to CollectPanel's Finalize action (fired
// once, before this banner ever mounts) — this component only re-attaches to
// whatever useDatasetUpload finds already running, via its own mount-time
// poll. It must never start a second, independent push itself, regardless of
// the repo id shape.
describe("the Hub push", () => {
  it("is never started from this banner", () => {
    setup({ repo_id: "makermods/autopush", saved_episodes: 5 });
    expect(start).not.toHaveBeenCalled();
  });

  it("stays manual for a repo with no namespace too", () => {
    setup({ repo_id: "sock_sort", saved_episodes: 5 });
    expect(start).not.toHaveBeenCalled();
  });
});

describe("a session that saved nothing", () => {
  // Nothing is on disk, so there is no repo to train on, preselect or upload.
  const empty = { repo_id: "makermods/gone", discarded_empty: true };

  it("says so instead of offering a dataset", () => {
    setup(empty);
    expect(
      screen.queryByRole("button", { name: /train on this/i }),
    ).not.toBeInTheDocument();
  });

  it("neither preselects nor uploads", () => {
    setup(empty);
    expect(setSelectedDataset).not.toHaveBeenCalled();
    expect(start).not.toHaveBeenCalled();
  });
});
