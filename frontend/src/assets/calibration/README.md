# Calibration start-pose photos

Start-pose stills for the SO-101 calibration panel in
`components/dialogs/RobotConfigDialog.tsx`.

| file                          | shown for                                |
| ----------------------------- | ---------------------------------------- |
| `so101-manual-start-pose.jpg` | Calibrate manually — the middle position |

The AUTO-calibration start pose is not here: it is the SO-101's folded resting
pose, which `assets/arms/so101.jpg` already shows, so that file is reused
rather than duplicated.

Spec:

- **960 × 540** (16:9). The frame is `aspect-video` + `object-cover`, so a 4:3
  source has to be padded to 16:9 here — otherwise the card shows muted bars
  down both sides.
- Plain **white** background, arm centred with a small margin, no hands or
  bench in shot. The panel puts this directly under a video of a real bench;
  the still's job is to be unambiguous about the POSE, not the setting.
- Keep under ~50 KB (JPEG quality ~85). Bundled into `frontend/dist/`, which
  is committed.
