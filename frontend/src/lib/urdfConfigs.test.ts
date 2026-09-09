import { describe, expect, it } from "vitest";
import { URDF_CONFIGS, urdfConfigFor } from "./urdfConfigs";

// One entry per arm type that ships a URDF (the families whose
// telemetry_kind is "urdf"). Each entry has to resolve its own mesh paths — a
// shared rewrite would send one arm's loader at the other's mesh folder.

describe("URDF_CONFIGS", () => {
  it("ships a model for every built-in arm", () => {
    expect(URDF_CONFIGS.so101).toBeDefined();
    expect(URDF_CONFIGS.maker).toBeDefined();
    expect(URDF_CONFIGS.metal).toBeDefined();
  });

  it("points each arm at its own public URDF directory", () => {
    expect(URDF_CONFIGS.so101?.urdfPath).toBe(
      "/so-101-urdf/urdf/so101_new_calib.urdf",
    );
    expect(URDF_CONFIGS.maker?.urdfPath).toBe("/maker-urdf/robot.urdf");
  });

  it("carries each URDF's up-axis — Z for the SO-101, Y for the Maker export", () => {
    expect(URDF_CONFIGS.so101?.up).toBe("Z");
    expect(URDF_CONFIGS.maker?.up).toBe("+Y");
  });

  it("ignores joint limits for the Maker model but not the SO-101", () => {
    expect(URDF_CONFIGS.maker?.ignoreLimits).toBe(true);
    expect(URDF_CONFIGS.so101?.ignoreLimits).toBeFalsy();
  });
});

describe("urdfConfigFor", () => {
  it("returns the Maker config for a Maker arm", () => {
    expect(urdfConfigFor("maker").urdfPath).toBe("/maker-urdf/robot.urdf");
  });

  it("falls back to the SO-101 config for an unknown or absent arm type", () => {
    expect(urdfConfigFor(undefined).urdfPath).toBe(
      "/so-101-urdf/urdf/so101_new_calib.urdf",
    );
    expect(urdfConfigFor("unknown_extension").urdfPath).toBe(
      "/so-101-urdf/urdf/so101_new_calib.urdf",
    );
  });
});

describe("rewriteMeshUrl", () => {
  it("maps the SO-101's package:// mesh refs into /so-101-urdf/meshes/", () => {
    expect(
      urdfConfigFor("so101").rewriteMeshUrl(
        "package://so_arm_description/meshes/base_so101_v2.stl",
      ),
    ).toBe("/so-101-urdf/meshes/base_so101_v2.stl");
  });

  it("keeps the Maker arm's mesh refs inside /maker-urdf/meshes/", () => {
    expect(
      urdfConfigFor("maker").rewriteMeshUrl("meshes/part_012_solid_012.stl"),
    ).toBe("/maker-urdf/meshes/part_012_solid_012.stl");
    // An already-resolved absolute path is left alone.
    expect(
      urdfConfigFor("maker").rewriteMeshUrl(
        "/maker-urdf/meshes/part_012_solid_012.stl",
      ),
    ).toBe("/maker-urdf/meshes/part_012_solid_012.stl");
  });
});

describe("Metal model", () => {
  it("loads the Z-up articulated gripper model", () => {
    expect(urdfConfigFor("metal").urdfPath).toBe("/metal-urdf/metal_with_gripper.urdf");
    expect(urdfConfigFor("metal").up).toBe("Z");
  });

  it.each([
    "package://metal_description/meshes/link1.STL",
    "/metal_description/meshes/link1.STL",
    "meshes/link1.STL",
    "/metal-urdf/meshes/link1.STL",
  ])("resolves %s with case preserved", (url) => {
    expect(urdfConfigFor("metal").rewriteMeshUrl(url)).toBe("/metal-urdf/meshes/link1.STL");
  });
});
