# Maker Arm v1 — 3D viewer URDF

Loaded by `UrdfViewer` (via `src/lib/urdfConfigs.ts`, `maker` entry) to
animate the teleop panel's 3D model for a Maker-arm session.

## Provenance

Exported from the CAD assembly `Maker Arm (RS02 打印版)_总装.step` with a
STEP→URDF converter (coordinate frame `z_up_x_forward`, `Rz*Ry*Rx`,
2026-09-02). 62 per-part STL meshes (`meshes/part_NNN_solid_NNN.stl`, scale
`0.001` mm→m). The one unreferenced part the exporter warned about
(`part_003`) is dropped.

## Local edits

- **Joint limits widened** to the follower's real motor ranges
  (`MakerFollowerConfig.joint_limits`, degrees → radians). The exporter wrote
  rounded placeholder limits (e.g. `link_004_joint` capped at 3.14 rad where
  the elbow motor travels to ~4.12); `urdf-loader` clamps `setJointValue` to
  these, so a too-narrow limit would freeze the on-screen joint short of the
  real one.

## Joint mapping

Generic exporter joint names, base → wrist, one per Maker motor. The
motor → joint map and each joint's sign/offset live in
`makermodslab/teleoperate.py` `_MAKER_URDF_JOINTS`:

| URDF joint       | Maker motor     |
| ---------------- | --------------- |
| `link_002_joint` | `shoulder_pan`  |
| `link_003_joint` | `shoulder_lift` |
| `link_004_joint` | `elbow_flex`    |
| `link_005_joint` | `wrist_flex`    |
| `link_006_joint` | `wrist_yaw`     |
| `link_007_joint` | `wrist_roll`    |

There is no gripper joint — the gripper geometry is rigid on `link_007`. The
gripper angle is still broadcast under `joints_deg` for the numeric value.
