/**
 * Per-arm-type 3D model wiring for `UrdfViewer`.
 *
 * There is one entry per arm type that ships a URDF (`frontend/public/…`);
 * the arm family's `telemetry_kind` (`"urdf"` vs `"degrees"`, from
 * `GET /api/v1/arms`) decides which arms have one. Each entry owns its own
 * URDF path, `package` attribute, and mesh-URL rewrite, because a shared
 * rewrite would send one arm's mesh loader at the other's folder.
 */
import type { ArmType } from "./armTypes";

export interface UrdfConfig {
  /** Public path to the URDF file the viewer loads. */
  urdfPath: string;
  /** Value for the viewer element's `package` attribute (the `package://` base). */
  packagePath: string;
  /**
   * Which model axis is "up" — the viewer element's `up` attribute. The SO-101
   * URDF is authored Z-up; the Maker CAD export is Y-up, and mounting that in
   * the default Z-up scene lays the arm flat on its side.
   */
  up: string;
  /** Minimum framing sphere radius (metres), leaving room for the arm to unfold. */
  minViewRadius?: number;
  /**
   * Let the model ignore the URDF's own joint limits (`ignore-limits` on the
   * viewer element). Set for the Maker arm: its shipped URDF carries the
   * designer's unverified limits, and the teleop broadcast feeds raw motor
   * angles with a still-unvalidated sign/offset — clamping to a too-narrow
   * limit would freeze a joint short of the real one. The gripper value is
   * range-clamped in the broadcast instead.
   */
  ignoreLimits?: boolean;
  /**
   * Rewrites a mesh URL the urdf-loader asks for into a real public path.
   * Called for every `<mesh>` in the URDF.
   */
  rewriteMeshUrl: (url: string) => string;
}

const SO101: UrdfConfig = {
  urdfPath: "/so-101-urdf/urdf/so101_new_calib.urdf",
  // Root, so the rewrite below does the full path resolution.
  packagePath: "/",
  up: "Z",
  rewriteMeshUrl: (url) => {
    // `package://so_arm_description/meshes/foo.stl` → `/so-101-urdf/meshes/foo.stl`
    if (url.startsWith("package://so_arm_description/meshes/")) {
      return url.replace(
        "package://so_arm_description/meshes/",
        "/so-101-urdf/meshes/",
      );
    }
    // A partially-resolved package path.
    if (url.includes("/so-101-urdf/so_arm_description/meshes/")) {
      return url.replace(
        "/so-101-urdf/so_arm_description/meshes/",
        "/so-101-urdf/meshes/",
      );
    }
    if (url.includes("so_arm_description/meshes/")) {
      return url.replace(
        /.*so_arm_description\/meshes\//,
        "/so-101-urdf/meshes/",
      );
    }
    // A bare relative `foo.stl`.
    if (url.endsWith(".stl") && !url.startsWith("/") && !url.startsWith("http")) {
      return `/so-101-urdf/meshes/${url}`;
    }
    return url;
  },
};

const MAKER: UrdfConfig = {
  urdfPath: "/maker-urdf/robot.urdf",
  packagePath: "/maker-urdf",
  // The SDK keeps the CAD's Y-up geometry and does not encode a display
  // rotation, so mount it Y-up (base plate in the XZ plane).
  up: "+Y",
  minViewRadius: 0.7,
  ignoreLimits: true,
  rewriteMeshUrl: (url) => {
    // The CAD export writes relative `meshes/part_XXX.stl`; urdf-loader
    // already resolves those against the URDF's own directory. Pin any mesh
    // reference to the public folder anyway so a stray `package://` or
    // resolved-from-root request still lands.
    const tail = url.match(/meshes\/[^/]+\.stl$/i);
    if (tail && !url.startsWith("/maker-urdf/")) {
      return `/maker-urdf/${tail[0]}`;
    }
    return url;
  },
};

const METAL: UrdfConfig = {
  urdfPath: "/metal-urdf/metal_with_gripper.urdf",
  packagePath: "/",
  up: "Z",
  minViewRadius: 0.75,
  rewriteMeshUrl: (url) => {
    const tail = url.match(/meshes\/[^/]+\.stl$/i);
    return tail ? `/metal-urdf/${tail[0]}` : url;
  },
};

export const URDF_CONFIGS: Partial<Record<ArmType, UrdfConfig>> = {
  so101: SO101,
  maker: MAKER,
  metal: METAL,
};

/** The URDF wiring for an arm type, or the SO-101's as a safe default. */
export function urdfConfigFor(armType: ArmType | undefined): UrdfConfig {
  return (armType && URDF_CONFIGS[armType]) || SO101;
}
