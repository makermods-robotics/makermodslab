# Maker Arm v1 — 3D viewer URDF

Loaded by `UrdfViewer` (via `src/lib/urdfConfigs.ts`, `maker` entry) to
animate the teleop panel's 3D model for a Maker-arm session.

## Provenance

Vendored from the official SDK,
[`makermods-robotics/maker-arm-sdk`](https://github.com/makermods-robotics/maker-arm-sdk),
`urdf/maker_arm/` at commit `b30d05a` ("Add Maker Arm URDF with symmetric
sliding gripper", 2026-09-05), content unchanged (only a trailing newline
added to `robot.urdf`, per this repo's end-of-file hook):

- `robot.urdf` — six revolute arm joints (`link_002_joint`..`link_007_joint`)
  plus a symmetric sliding gripper: `gripper_left_joint` (prismatic, 0 –
  0.0524125 m) drives it, `gripper_right_joint` mimics it, and the fixed
  `grasp_center` frame sits midway between the jaws.
- `meshes/` — 52 binary STL parts, `meshes/part_NNN_solid_NNN.stl`, mm, scale
  `0.001` in the URDF. All 52 are referenced by `robot.urdf`.
- `revision_report.json` — the SDK's provenance record: mesh assignments,
  inertia estimates, the gripper motor-endpoint capture, and the remaining
  validation checks.

Keep these in sync with the SDK on its next URDF revision rather than editing
them here.

## What the app does with it, without touching the file

- **Up axis.** The SDK retains the CAD's Y-up geometry and deliberately does
  **not** encode the inspection viewer's display rotation (see the SDK
  README). `urdfConfigs.ts` mounts the `maker` model with `up: "+Y"` so it
  stands upright in the viewer's Z-up scene.
- **Joint limits ignored in the viewer.** `robot.urdf` carries the designer's
  unverified arm-joint limits, and the SDK flags physical zero alignment and
  actuator limits as still-to-validate. Until that hardware pass, the teleop
  broadcast feeds raw motor angles (`teleoperate._MAKER_URDF_JOINTS`, neutral
  sign/offset), which can land outside those limits and freeze a joint on
  screen. `urdfConfigs.ts` sets `ignoreLimits` for the `maker` entry so the
  model tracks the arm across its full travel; the gripper value is clamped
  to its real range in the broadcast instead.

## Gripper

`teleoperate._MAKER_URDF_GRIPPER` maps the Maker gripper motor angle to
`gripper_left_joint` travel using the two endpoints the SDK's
`revision_report.json` records (`motor_calibration`: closed ≈ 0.0067 rad,
commanded-open ≈ −2.079 rad → jaw gap 0 – 104.825 mm). The SDK marks that a
**visual-preview** interpolation, not a calibrated transmission — the on-screen
jaw opening is indicative, not measured.
