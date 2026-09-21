# Metal Arm — teleop viewer model

Vendored from [makermods-robotics/metal-python-ros](https://github.com/makermods-robotics/metal-python-ros/tree/ef4181f1305cbcfc63431d3bcfb96f5fb7f72763/metal_ros2/src/metal_description) at `ef4181f1305cbcfc63431d3bcfb96f5fb7f72763`:

- `urdf/metal_with_gripper.urdf` → `metal_with_gripper.urdf` (only a trailing newline added).
- `meshes/` → the ten original binary STL files, unchanged. Case is significant on Linux.
- `LICENSE` retains the SDK's MIT notice (Copyright 2025 MakerMods).

This is the ROS description with articulated jaws, not the SDK example's older six-joint model with a rigid gripper. The frontend resolves `package://metal_description/meshes/…` to this directory and mounts the model Z-up.

`arms/urdf.py` maps the six motor angles in degrees to `joint1`–`joint6` in radians, matching the vendor ROS controller's direct motor mapping. The gripper uses the SDK's measured nonlinear angle/stroke table from `metal_sdk/native/can_manager.h`, with linear interpolation between samples for display. `joint7 = -opening_mm / 2000` moves one jaw; `joint8` mimics it with multiplier -1. Opening is clamped to the URDF's 100 mm maximum, including motor angles beyond the table's measured range. This affects visualization only.

Keep the geometry and transmission table in sync with the vendor source. Software checks cover assets, mappings, and simulated telemetry; physical pose alignment still needs a hardware comparison.
