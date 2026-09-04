import { describe, expect, it } from "vitest";

import {
  PREVIEW_OFFSET_S,
  frameDuration,
  lastFramePosition,
  previewPosition,
} from "./episodePreview";

// Real numbers from sock_2_orange_color_only (30fps, v3.0 packed mp4):
// episode 6 spans 73.4 -> 85.5s and episode 7 starts at exactly 85.5s.
const FPS = 30;
const EP7_LENGTH = 388;
const EP7_DURATION = EP7_LENGTH / FPS;

describe("a preview position never sits on an episode boundary", () => {
  // The bug this exists to prevent: v3.0 packs episodes into one mp4, so
  // episode 7's start IS episode 6's end, to the bit. Seeking there is a
  // coin-flip between the two frames either side of it, and browsers do pick
  // the earlier one — painting episode 6's last frame as episode 7's still.
  it("lands strictly inside the episode, never at 0", () => {
    const t = previewPosition(EP7_DURATION);
    expect(t).toBeGreaterThan(0);
    expect(t).toBeLessThan(EP7_DURATION);
    expect(t).toBe(PREVIEW_OFFSET_S);
  });

  it("clamps to the midpoint of an episode shorter than the offset", () => {
    // 12 frames at 30fps = 0.4s, so a flat 0.5s would seek past the end.
    expect(previewPosition(0.4)).toBeCloseTo(0.2, 10);
  });

  it("stays strictly inside even a single-frame episode", () => {
    // Regression: a previous one-frame floor returned exactly `duration`
    // here — the boundary shared with the next episode, which is the one
    // position this function exists to avoid. Half a frame is inside frame 0.
    const duration = 1 / FPS;
    const t = previewPosition(duration);
    expect(t).toBeGreaterThan(0);
    expect(t).toBeLessThan(duration);
  });

  it("degrades to 0 rather than NaN for a malformed episode", () => {
    expect(frameDuration(0, 0)).toBe(0);
    expect(previewPosition(0)).toBe(0);
    expect(lastFramePosition(0, 0)).toBe(0);
  });
});

describe("playback stops on the episode's own last frame", () => {
  // timeupdate fires at ~4Hz, so the end-of-episode check can overshoot by
  // ~250ms — several frames into the NEXT episode. Freezing there shows a
  // frame that isn't this episode's, which reads as a contaminated recording.
  it("is one frame short of the boundary, not on it", () => {
    const t = lastFramePosition(EP7_LENGTH, EP7_DURATION);
    expect(t).toBeCloseTo(EP7_DURATION - 1 / FPS, 10);
    expect(t).toBeLessThan(EP7_DURATION);
  });

  it("recovers the frame rate from length and duration", () => {
    expect(frameDuration(EP7_LENGTH, EP7_DURATION)).toBeCloseTo(1 / FPS, 10);
    // A 50fps dataset would work without touching this code.
    expect(frameDuration(100, 2)).toBeCloseTo(1 / 50, 10);
  });

  it("never precedes the preview position for a normal episode", () => {
    expect(lastFramePosition(EP7_LENGTH, EP7_DURATION)).toBeGreaterThan(
      previewPosition(EP7_DURATION),
    );
  });

  it("stays strictly inside the episode, off the shared boundary", () => {
    // handleSeek clamps to this, so dragging the scrubber fully right must
    // not land on `duration` — that timestamp belongs to the next episode.
    expect(lastFramePosition(EP7_LENGTH, EP7_DURATION)).toBeLessThan(EP7_DURATION);
  });
});
