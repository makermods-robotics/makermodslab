import { afterEach, describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import SingleTabGuard from "./SingleTabGuard";

describe("SingleTabGuard", () => {
  afterEach(() => {
    document.body.style.pointerEvents = "";
  });

  it("stays clickable when a modal dialog elsewhere on the page has disabled pointer events on <body> (regression: Radix's modal Dialog sets body { pointer-events: none } while open — e.g. the update-available popup — and since pointer-events is inherited, the visually-topmost tab-conflict overlay silently became unclickable, leaving the dialog underneath as the only interactive element)", async () => {
    // A peer tab that opened first wins the election, making this instance
    // non-primary and showing the "already open in another tab" overlay.
    const peer = new BroadcastChannel("makermodslab-tabs-v1");
    render(<SingleTabGuard>{null}</SingleTabGuard>);

    peer.postMessage({ type: "HEARTBEAT", id: "peer-1", openedAt: 0 });

    const overlay = await screen.findByRole("dialog");

    // Mirrors what @radix-ui/react-dismissable-layer does to <body> for the
    // duration any modal Dialog (like UpdateNotice) is open.
    document.body.style.pointerEvents = "none";

    expect(overlay).toHaveClass("pointer-events-auto");

    peer.close();
  });
});
