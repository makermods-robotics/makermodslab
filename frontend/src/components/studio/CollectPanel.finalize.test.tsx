import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { StudioProvider, useStudio } from "@/contexts/StudioContext";

// CollectPanel's own orchestration is what's under test here — the
// heavy-lifting children (the actual recording UI, the dataset viewer, the
// post-session banner) are already covered by their own test files, so they
// render as trivial stubs that expose just enough to drive/observe the wiring
// between "a session ended" -> "Finalize review opens" -> "Finalize/Discard
// hands off to the library + (maybe) the Hub upload".

const mocks = vi.hoisted(() => ({
  fetch: vi.fn(),
  toast: vi.fn(),
  uploadDataset: vi.fn(async () => ({ started: true, repo_id: "x", message: "" })),
}));

vi.mock("@/contexts/ApiContext", () => ({
  useApi: () => ({ baseUrl: "http://test", fetchWithHeaders: mocks.fetch }),
}));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ toast: mocks.toast }) }));
vi.mock("@/contexts/HfAuthContext", () => ({
  useHfAuth: () => ({
    auth: { status: "authenticated", username: "alice", writableNamespaces: [] },
  }),
}));
vi.mock("@/hooks/useRobots", () => ({
  useRobots: () => ({
    selectedRecord: { name: "bench", is_clean: true, cameras: [] },
  }),
}));
vi.mock("@/hooks/useDatasets", () => ({
  useDatasets: () => ({ datasets: [], loading: false, refresh: vi.fn() }),
}));
vi.mock("@/hooks/useSelectedDataset", () => ({
  useSelectedDataset: () => ({ selectedDataset: null, setSelectedDataset: vi.fn() }),
}));
vi.mock("@/lib/replayApi", async () => {
  const actual = await vi.importActual<typeof import("@/lib/replayApi")>("@/lib/replayApi");
  return { ...actual, uploadDataset: mocks.uploadDataset };
});

// Seeds a valid form (name + task) through the REAL StudioContext on mount,
// the same way filling in RecordingForm's fields would — the panel's Start
// button is gated on this, and pushToHub is exactly the flag under test.
vi.mock("@/components/studio/RecordingForm", () => ({
  default: function RecordingFormStub({
    pushToHub,
    setPushToHub,
  }: {
    pushToHub: boolean;
    setPushToHub: (v: boolean) => void;
  }) {
    const { updateCollectForm } = useStudio();
    React.useEffect(() => {
      updateCollectForm({ datasetName: "sock_sort", singleTask: "pick the sock" });
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);
    return (
      <button type="button" onClick={() => setPushToHub(!pushToHub)}>
        toggle-push-to-hub ({String(pushToHub)})
      </button>
    );
  },
}));

vi.mock("@/components/recording/RecordingSessionDialog", () => ({
  default: ({ onExit }: { onExit: (recorded?: unknown) => void }) => (
    <div>
      <button type="button" onClick={() => onExit({ repo_id: "alice/sock_sort", saved_episodes: 3 })}>
        end-session-with-episodes
      </button>
      <button type="button" onClick={() => onExit({ repo_id: "alice/empty", discarded_empty: true })}>
        end-session-discarded-empty
      </button>
    </div>
  ),
}));

vi.mock("@/components/dialogs/DatasetDetailDialog", () => ({
  default: ({
    open,
    repoId,
    finalize,
  }: {
    open: boolean;
    repoId: string | null;
    finalize?: { onFinalize: () => void; onDiscarded: () => void };
  }) =>
    open ? (
      <div>
        <span>viewing {repoId}</span>
        {finalize && (
          <>
            <span>finalize-mode</span>
            <button type="button" onClick={finalize.onFinalize}>
              do-finalize
            </button>
            <button type="button" onClick={finalize.onDiscarded}>
              do-discard
            </button>
          </>
        )}
      </div>
    ) : null,
}));

vi.mock("@/components/studio/CollectHandoff", () => ({
  default: ({ recorded }: { recorded: { repo_id: string } | null }) =>
    recorded ? <div>handoff: {recorded.repo_id}</div> : null,
}));

vi.mock("@/components/library/LibraryHeader", () => ({ default: () => null }));
vi.mock("@/components/library/DatasetLibrary", () => ({
  DatasetLibraryList: () => null,
  clearDatasetInfoCache: vi.fn(),
}));
vi.mock("@/components/landing/MergeDatasetsDialog", () => ({ default: () => null }));

import CollectPanel from "./CollectPanel";

beforeEach(() => vi.clearAllMocks());
afterEach(() => cleanup());

const setup = () =>
  render(
    <StudioProvider>
      <CollectPanel />
    </StudioProvider>,
  );

const startAndEndSession = async (endButtonName: RegExp) => {
  setup();
  // The form (and the stub inside it that seeds a valid name/task) only
  // mounts once "Record new dataset" is expanded.
  fireEvent.click(await screen.findByRole("button", { name: /record new dataset/i }));
  const startButton = await screen.findByRole("button", { name: /start recording/i });
  await waitFor(() => expect(startButton).toBeEnabled());
  fireEvent.click(startButton);
  fireEvent.click(await screen.findByRole("button", { name: endButtonName }));
};

describe("a session that saved episodes", () => {
  it("opens the dataset viewer in Finalize mode instead of the old immediate handoff", async () => {
    await startAndEndSession(/end-session-with-episodes/i);

    expect(await screen.findByText("viewing alice/sock_sort")).toBeInTheDocument();
    expect(screen.getByText("finalize-mode")).toBeInTheDocument();
    // No handoff banner yet — Finalize hasn't been clicked.
    expect(screen.queryByText(/handoff:/)).not.toBeInTheDocument();
  });

  it("clicking Finalize hands off to the library banner and starts the Hub upload when Push to Hub was on", async () => {
    // Push to Hub defaults to true (StudioContext's initial collectForm).
    await startAndEndSession(/end-session-with-episodes/i);
    fireEvent.click(await screen.findByRole("button", { name: /do-finalize/i }));

    expect(await screen.findByText("handoff: alice/sock_sort")).toBeInTheDocument();
    await waitFor(() =>
      expect(mocks.uploadDataset).toHaveBeenCalledWith(
        "http://test",
        mocks.fetch,
        "alice/sock_sort",
        [],
        false,
      ),
    );
  });

  it("does not start an upload when Push to Hub was off", async () => {
    await startAndEndSession(/end-session-with-episodes/i);
    fireEvent.click(screen.getByRole("button", { name: /toggle-push-to-hub \(true\)/i }));
    fireEvent.click(await screen.findByRole("button", { name: /do-finalize/i }));

    expect(await screen.findByText("handoff: alice/sock_sort")).toBeInTheDocument();
    expect(mocks.uploadDataset).not.toHaveBeenCalled();
  });

  it("clicking Discard hands off nothing — no banner, no upload", async () => {
    await startAndEndSession(/end-session-with-episodes/i);
    fireEvent.click(await screen.findByRole("button", { name: /do-discard/i }));

    await waitFor(() =>
      expect(screen.queryByText("viewing alice/sock_sort")).not.toBeInTheDocument(),
    );
    expect(screen.queryByText(/handoff:/)).not.toBeInTheDocument();
    expect(mocks.uploadDataset).not.toHaveBeenCalled();
  });
});

describe("a session that saved nothing", () => {
  it("skips Finalize entirely (nothing to review)", async () => {
    await startAndEndSession(/end-session-discarded-empty/i);
    expect(screen.queryByText(/viewing/)).not.toBeInTheDocument();
    expect(screen.queryByText("finalize-mode")).not.toBeInTheDocument();
  });
});
