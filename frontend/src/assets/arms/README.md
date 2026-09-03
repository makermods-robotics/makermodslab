# Arm photos

One product picture per hardware family, shown on the arm-type cards in the
"Create a new robot" dialog (`components/landing/CreateRobotDialog.tsx`).

| file        | arm type     |
| ----------- | ------------ |
| `so101.jpg` | SO-101       |
| `maker.jpg` | Maker Arm v1 |
| `metal.jpg` | Metal arm    |

Spec:

- **800 × 600** (4:3). The slot renders at roughly 170 × 128 CSS px, so this is
  ~2x for HiDPI; the frame is `object-cover`, so anything 4:3 crops cleanly.
- Shoot the **follower** arm, whole, on a plain background — the card is small
  and a busy bench turns to mush at 170px wide.
- Keep each file under ~150 KB (JPEG quality ~80). They are bundled into
  `frontend/dist/`, which is committed.

The current photos are imported by `CreateRobotDialog.tsx`. Future hardware can
use the same filenames/spec and the existing placeholder fallback until its
photo is ready.
