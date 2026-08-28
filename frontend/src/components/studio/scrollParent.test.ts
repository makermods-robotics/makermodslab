import { afterEach, describe, expect, it } from "vitest";

import { scrollParent } from "./DeployPanel";

/**
 * The scroll anchor that keeps the verb row still while the form above it
 * grows and shrinks is only as good as the box it nudges. Which ancestor
 * scrolls is a media query in the studio (each panel below `lg`, the whole
 * grid above it), so this picks it at call time rather than assuming.
 */
const build = (overflows: (string | null)[]) => {
  const root = document.createElement("div");
  let node = root;
  for (const overflow of overflows) {
    const child = document.createElement("div");
    if (overflow) child.style.overflowY = overflow;
    node.appendChild(child);
    node = child;
  }
  const leaf = document.createElement("div");
  node.appendChild(leaf);
  document.body.appendChild(root);
  return leaf;
};

afterEach(() => {
  document.body.innerHTML = "";
});

describe("scrollParent finds the box that actually scrolls", () => {
  it("takes the nearest scrolling ancestor, not the outermost", () => {
    const leaf = build(["auto", "auto"]);
    expect(scrollParent(leaf)).toBe(leaf.parentElement);
  });

  it("walks past non-scrolling ancestors", () => {
    const leaf = build(["scroll", null, null]);
    expect(scrollParent(leaf)).toBe(
      leaf.parentElement?.parentElement?.parentElement,
    );
  });

  it("ignores overflow-y: hidden — a clipped box cannot be nudged", () => {
    const leaf = build(["auto", "hidden"]);
    expect(scrollParent(leaf)).toBe(leaf.parentElement?.parentElement);
  });

  it("falls back to the document when nothing between scrolls", () => {
    const leaf = build([null, null]);
    expect(scrollParent(leaf)).toBe(document.scrollingElement);
  });
});
