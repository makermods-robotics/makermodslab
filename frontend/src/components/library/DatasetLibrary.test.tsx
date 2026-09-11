import React from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DatasetLibraryList } from "./DatasetLibrary";
import type { DatasetItem } from "@/lib/replayApi";

vi.mock("@/contexts/ApiContext", () => ({
  useApi: () => ({ baseUrl: "http://test", fetchWithHeaders: vi.fn() }),
}));
vi.mock("@/hooks/useHubVideoFilter", () => ({
  useHubVideoFilter: (datasets: DatasetItem[]) => datasets,
}));
vi.mock("./CappedGrid", () => ({
  default: ({ items }: { items: React.ReactNode[] }) => <div>{items}</div>,
  GRID_MIN_H: "",
}));

afterEach(cleanup);

const item: DatasetItem = { repo_id: "test/example", source: "hub", last_modified: null, private: false };

describe("dataset card keyboard actions", () => {
  it.each(["Enter", " "])("keeps %j on View available to the native button", (key) => {
    const onSelect = vi.fn();
    const onView = vi.fn();
    render(<DatasetLibraryList datasets={[item]} loading={false}
      selectedRepoId={null} onSelect={onSelect} onView={onView} />);
    const view = screen.getByRole("button", { name: "View dataset" });
    // jsdom does not synthesize a click from a key; uncancelled keydown is
    // the contract that lets the browser perform that native activation.
    expect(fireEvent.keyDown(view, { key })).toBe(true);
    expect(onSelect).not.toHaveBeenCalled();
    fireEvent.click(view);
    expect(onView).toHaveBeenCalledWith(item);
    expect(onSelect).not.toHaveBeenCalled();
  });

  it.each(["Enter", " "])("still selects the card itself with %j", (key) => {
    const onSelect = vi.fn();
    render(<DatasetLibraryList datasets={[item]} loading={false}
      selectedRepoId={null} onSelect={onSelect} onView={vi.fn()} />);
    const card = screen.getAllByRole("button").find((node) => node.tagName === "DIV" && node.hasAttribute("aria-pressed"));
    expect(card).toBeTruthy();
    fireEvent.keyDown(card!, { key });
    expect(onSelect).toHaveBeenCalledWith(item);
  });
});
