import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import EpisodeTaskPrompt from "./EpisodeTaskPrompt";

describe("the per-episode task prompt", () => {
  it("prefills the box with the previous episode's task", () => {
    render(
      <EpisodeTaskPrompt
        episode={2}
        defaultTask="pick the cube"
        submitting={false}
        onSubmit={vi.fn()}
      />,
    );
    expect(screen.getByRole("textbox")).toHaveValue("pick the cube");
  });

  it("submits the (edited) task", () => {
    const onSubmit = vi.fn();
    render(
      <EpisodeTaskPrompt
        episode={2}
        defaultTask="pick the cube"
        submitting={false}
        onSubmit={onSubmit}
      />,
    );
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "fold the towel" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save task/i }));
    expect(onSubmit).toHaveBeenCalledWith("fold the towel");
  });

  it("will not submit an empty task — naming is mandatory", () => {
    const onSubmit = vi.fn();
    render(
      <EpisodeTaskPrompt
        episode={1}
        defaultTask=""
        submitting={false}
        onSubmit={onSubmit}
      />,
    );
    const save = screen.getByRole("button", { name: /save task/i });
    expect(save).toBeDisabled();
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "   " },
    });
    expect(save).toBeDisabled();
  });

  it("locks the button while a submission is in flight", () => {
    render(
      <EpisodeTaskPrompt
        episode={2}
        defaultTask="pick the cube"
        submitting={true}
        onSubmit={vi.fn()}
      />,
    );
    expect(screen.getByRole("button", { name: /save task/i })).toBeDisabled();
  });
});
