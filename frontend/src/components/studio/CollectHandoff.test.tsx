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

// These two are the reason the banner is rendered unconditionally in the
// always-mounted Collect panel rather than only when someone is looking at it.
// They used to run because a finished session NAVIGATED to the Launchpad,
// mounting the banner there; the studio now stays open, so nothing else would
// trigger them.
describe("the side effects that used to ride on the navigation home", () => {
  // Each case needs its OWN repo id: the auto-push is guarded by a
  // module-level set of already-pushed ids, so reusing one here would report a
  // missing upload that the guard had merely deduplicated.
  it("preselects the fresh dataset so Train opens onto it", () => {
    setup({ repo_id: "makermods/preselect", saved_episodes: 5 });
    expect(setSelectedDataset).toHaveBeenCalledWith("makermods/preselect");
  });

  it("kicks off the Hub push for a namespaced repo", () => {
    setup({ repo_id: "makermods/autopush", saved_episodes: 5 });
    expect(start).toHaveBeenCalled();
  });

  // That guard is itself load-bearing — a remount must not fire a second,
  // redundant upload of the same dataset.
  it("does not push the same dataset twice", () => {
    setup({ repo_id: "makermods/pushed_once", saved_episodes: 5 });
    expect(start).toHaveBeenCalledTimes(1);
    setup({ repo_id: "makermods/pushed_once", saved_episodes: 5 });
    expect(start).toHaveBeenCalledTimes(1);
  });

  // A repo id with no namespace means the user wasn't logged in at record
  // time, so a push could only 401.
  it("stays manual when the repo has no namespace", () => {
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
