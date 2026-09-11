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
    fireEvent.click(screen.getByRole("button", { name: /start recording/i }));
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
    const save = screen.getByRole("button", { name: /start recording/i });
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
    expect(screen.getByRole("button", { name: /start recording/i })).toBeDisabled();
  });
});

it("keeps Space for typing, then starts on Space after Enter finishes editing", () => {
  const onSubmit = vi.fn();
  render(<EpisodeTaskPrompt episode={1} defaultTask="pick cube" submitting={false} onSubmit={onSubmit} />);
  const box = screen.getByRole("textbox");
  fireEvent.keyDown(box, { key: " " });
  expect(onSubmit).not.toHaveBeenCalled();
  fireEvent.keyDown(box, { key: "Enter" });
  expect(onSubmit).not.toHaveBeenCalled();
  const start = screen.getByRole("button", { name: /start recording/i });
  expect(start).toHaveFocus();
  fireEvent.keyDown(start, { key: " " });
  expect(onSubmit).toHaveBeenCalledExactlyOnceWith("pick cube");
});

it("can discard and re-record the previous take without submitting the next task", () => {
  const onSubmit = vi.fn();
  const onRerecord = vi.fn();
  render(<EpisodeTaskPrompt episode={2} defaultTask="pick cube" submitting={false} onSubmit={onSubmit} onRerecord={onRerecord} />);
  fireEvent.click(screen.getByRole("button", { name: /re-record previous task/i }));
  expect(onRerecord).toHaveBeenCalledOnce();
  expect(onSubmit).not.toHaveBeenCalled();
});

it("starts on Space after focus moves out of the description", () => {
  const onSubmit = vi.fn();
  render(<EpisodeTaskPrompt episode={1} defaultTask="pick cube" submitting={false} onSubmit={onSubmit} />);
  screen.getByRole("textbox").blur();
  fireEvent.keyDown(document.body, { key: " " });
  expect(onSubmit).toHaveBeenCalledExactlyOnceWith("pick cube");
});
