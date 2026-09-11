/** Playback positions within a v3.0 packed episode video.
 *
 * LeRobot v3.0 packs every episode for a camera into ONE mp4, so episode N's
 * end timestamp is bit-for-bit equal to episode N+1's start. Seeking to either
 * lands exactly on a frame boundary, and the HTML spec leaves it
 * implementation-defined which side a seek resolves to — so a browser may
 * legitimately paint the neighbouring episode's frame. These helpers pick
 * positions that are strictly INSIDE the episode, which makes that ambiguity
 * unreachable instead of something to compensate for.
 *
 * All values here are episode-relative seconds; callers add the camera's
 * `video_offsets[camera].from` to get a position in the packed file.
 */

/** Where the viewer parks when an episode is selected, in seconds from its
 * start. Deliberately not 0 — see previewPosition. */
export const PREVIEW_OFFSET_S = 0.5;

/** One frame, in seconds.
 *
 * `duration` is length/fps server-side (makermodslab/datasets.py), so this
 * recovers 1/fps without plumbing fps through the component tree, and stays
 * correct if a dataset is ever recorded at a rate other than 30. Returns 0 for
 * a malformed (zero-length) episode so callers degrade to the raw boundary
 * rather than producing NaN seeks. */
export function frameDuration(length: number, duration: number): number {
  return length > 0 ? duration / length : 0;
}

/** The preview ("thumbnail") position: the frame shown before playback starts.
 *
 * Half a second in, for two reasons. Frame 0 of a manipulation episode is the
 * arm at rest at home position — near-identical across every episode in a
 * dataset — so a frame slightly later is what makes episodes visually
 * distinguishable in the viewer. And being strictly inside the episode avoids
 * the start-boundary ambiguity described above.
 *
 * Clamped to the episode's midpoint for episodes shorter than a second. That
 * midpoint is the safe floor on its own — for a single-frame episode it is
 * half a frame, which is inside frame 0. An earlier version floored this at a
 * whole frame instead, which for a single-frame episode returned exactly
 * `duration` — the boundary shared with the next episode, i.e. the one
 * position this function exists to avoid. */
export function previewPosition(duration: number): number {
  if (!(duration > 0)) return 0;
  return Math.min(PREVIEW_OFFSET_S, duration / 2);
}

/** The episode's own final frame.
 *
 * Playback is stopped by a `timeupdate` handler firing at ~4Hz, so it
 * overshoots into the next episode's footage by up to ~250ms before anything
 * notices; freezing there would leave the viewer showing a frame that belongs
 * to the next episode. Snapping here puts it back on this episode's last
 * frame. This IS a frame boundary, so an engine may resolve it to the last or
 * second-to-last frame — both are inside the episode, which is the only
 * property that matters. */
export function lastFramePosition(length: number, duration: number): number {
  return Math.max(0, duration - frameDuration(length, duration));
}
