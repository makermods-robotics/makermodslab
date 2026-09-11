# Calibration demo media

The clips and posters behind the SO-101 calibration panel in
`components/dialogs/RobotConfigDialog.tsx` (see its `CalibrationClip`).

| file                  | shown for                        |
| --------------------- | -------------------------------- |
| `autocal-so101.mp4`   | Auto-calibrate, before Start     |
| `autocal-so101.jpg`   | its poster frame                 |
| `manualcal-so101.mp4` | Calibrate manually, before Start |
| `manualcal-so101.jpg` | its poster frame                 |

These live in `public/` rather than `src/assets/` on purpose: Vite hashes and
graphs imported assets, and a multi-megabyte video does not belong in the
module graph. They are referenced by absolute path, so they land in
`dist/media/calibration/` byte-for-byte and stream with range requests — which
is what makes the scrub slider work.

Spec:

- **960 × 540 at 24fps**, h264 High, `yuv420p`, `-movflags +faststart` so the
  browser can start playing before the whole file lands. The slot renders at
  roughly 640 CSS px wide, so this is already ~1.5x for HiDPI; 720p60 doubles
  the bytes for pixels the dialog never shows.
- **No audio track.** The clips autoplay, and autoplay with sound is blocked
  everywhere; stripping the track makes that unambiguous rather than leaning on
  the `muted` attribute alone.
- Keep each clip **under ~2 MB** and roughly a minute. They ship inside
  `frontend/dist/`, which is committed, so every byte lands in the repo twice.
  They are also the one path excluded from the `check-added-large-files`
  pre-commit hook — that exemption is for clips encoded to this spec, not a
  licence to drop a raw phone capture here.
- Poster: one representative frame, 960 px wide, JPEG. Without it the slot is
  an empty black box until the first frame decodes.

Re-encode from a source clip with:

```bash
ffmpeg -ss <start> -i <source> -t <seconds> -an \
  -vf "scale=960:-2,fps=24" \
  -c:v libx264 -profile:v high -pix_fmt yuv420p -crf 30 -preset veryslow \
  -movflags +faststart out.mp4
```
