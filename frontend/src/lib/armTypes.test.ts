import { describe, expect, it } from "vitest";
import i18n from "@/i18n";
import type { TFunction } from "i18next";
import type { ArmFamilyInfo } from "./armsApi";
import {
  armLabel,
  armTypeFromRobotType,
  jointsPerArm,
  supportsAutoCalibration,
  supportsDagger,
  supportsPortProbe,
  telemetryKind,
  calibrationKind,
  usesFeetechBus,
} from "./armTypes";

// The client mirror of the backend's arm_capabilities.py predicates, now read
// from the GET /api/v1/arms manifest instead of a closed id union. These
// fixtures are the three built-ins EXACTLY as makermodslab/arms/manifest.py
// serves them (mirroring arms/so101.py, maker.py, metal.py), plus three
// families the core does not ship — the shape an extension registers — so
// every predicate is proven to read the manifest's fields, never the id.

const SO101: ArmFamilyInfo = {
  id: "so101",
  label: "SO-101",
  short_label: "SO-101",
  provided_by: "builtin",
  image_url: null,
  joints_per_arm: 6,
  calibration_name_suffix: "",
  supports_bimanual: true,
  calibration: { kind: "range_sweep", summary: null, panel_url: null },
  telemetry_kind: "urdf",
  capabilities: {
    uses_feetech_bus: true,
    supports_auto_calibration: true,
    supports_dagger: true,
    supports_port_probe: false,
    motion_identify_energizes_follower: false,
  },
  robot_types: ["so101_follower", "bi_so_follower"],
  robot_type_markers: [
    "so100",
    "so101",
    "so-100",
    "so-101",
    "so_follower",
    "so_leader",
  ],
};

const MAKER: ArmFamilyInfo = {
  id: "maker",
  label: "Maker Arm v1",
  short_label: "Maker",
  provided_by: "builtin",
  image_url: null,
  joints_per_arm: 7,
  calibration_name_suffix: "_maker",
  supports_bimanual: true,
  calibration: {
    kind: "steps",
    summary: {
      leader: {
        text: "Move the Star Arm 102 leader by hand to its ZERO POSE — folded against the base, gripper closed — then confirm.",
        image_url: null,
      },
      follower: {
        text: "Move the arm by hand to its ZERO POSE — folded against the base, gripper fully open — then confirm.",
        image_url: null,
      },
    },
    panel_url: null,
  },
  telemetry_kind: "urdf",
  capabilities: {
    uses_feetech_bus: false,
    supports_auto_calibration: false,
    supports_dagger: false,
    supports_port_probe: true,
    motion_identify_energizes_follower: false,
  },
  robot_types: ["maker_follower", "bi_maker_follower"],
  robot_type_markers: ["maker"],
};

const METAL: ArmFamilyInfo = {
  id: "metal",
  label: "Metal Arm",
  short_label: "Metal",
  provided_by: "builtin",
  image_url: null,
  joints_per_arm: 7,
  calibration_name_suffix: "_metal",
  supports_bimanual: true,
  calibration: {
    kind: "steps",
    summary: {
      leader: {
        text: "Move the Star Arm 102 leader by hand to its ZERO POSE — folded against the base, gripper closed — then confirm.",
        image_url: null,
      },
      follower: {
        text: "Move the arm by hand to its ZERO POSE — standing upright, all joints at 0 degrees, gripper closed — then confirm.",
        image_url: null,
      },
    },
    panel_url: null,
  },
  telemetry_kind: "degrees",
  capabilities: {
    uses_feetech_bus: false,
    supports_auto_calibration: false,
    supports_dagger: false,
    supports_port_probe: true,
    motion_identify_energizes_follower: true,
  },
  robot_types: ["metal_follower", "bi_metal_follower"],
  robot_type_markers: ["metal"],
};

/** An extension's SO-101-shaped family: same hardware answers, different id. */
const SO101_TWIN: ArmFamilyInfo = {
  id: "so101_twin",
  label: "SO-101 Twin",
  short_label: "Twin",
  provided_by: "hello",
  image_url: null,
  joints_per_arm: 6,
  calibration_name_suffix: "_so101_twin",
  supports_bimanual: true,
  calibration: { kind: "range_sweep", summary: null, panel_url: null },
  telemetry_kind: "urdf",
  capabilities: {
    uses_feetech_bus: true,
    supports_auto_calibration: true,
    supports_dagger: true,
    supports_port_probe: false,
    motion_identify_energizes_follower: false,
  },
  robot_types: ["so101_twin_follower", "bi_so101_twin_follower"],
  robot_type_markers: ["twin"],
};

/** An extension's family with a joint count no built-in has. */
const NINE: ArmFamilyInfo = {
  id: "nine",
  label: "Nine Arm",
  short_label: "Nine",
  provided_by: "nine-ext",
  image_url: null,
  joints_per_arm: 9,
  calibration_name_suffix: "_nine",
  supports_bimanual: false,
  calibration: {
    kind: "steps",
    summary: {
      leader: { text: "Fold the nine leader.", image_url: null },
      follower: { text: "Fold the nine.", image_url: null },
    },
    panel_url: null,
  },
  telemetry_kind: "degrees",
  capabilities: {
    uses_feetech_bus: false,
    supports_auto_calibration: false,
    supports_dagger: false,
    supports_port_probe: true,
    motion_identify_energizes_follower: false,
  },
  robot_types: ["nine_follower", "bi_nine_follower"],
  robot_type_markers: ["nine"],
};

/**
 * An extension's family that calibrates through its own served page (kind
 * "panel") and ships its own photo — the two served URLs the manifest can
 * carry for a family the frontend bundles nothing for.
 */
const PANELED: ArmFamilyInfo = {
  id: "paneled",
  label: "Paneled Arm",
  short_label: "Paneled",
  provided_by: "paneled",
  image_url: "/api/v1/ext/paneled/static/arm.jpg",
  joints_per_arm: 7,
  calibration_name_suffix: "_paneled",
  supports_bimanual: false,
  calibration: {
    kind: "panel",
    summary: null,
    panel_url: "/api/v1/ext/paneled/static/calibrate.html",
  },
  telemetry_kind: "degrees",
  capabilities: {
    uses_feetech_bus: false,
    supports_auto_calibration: false,
    supports_dagger: false,
    supports_port_probe: true,
    motion_identify_energizes_follower: false,
  },
  robot_types: ["paneled_follower", "bi_paneled_follower"],
  robot_type_markers: ["paneled"],
};

/** Manifest order: registry order, the default (SO-101) family first. */
const ARMS: ArmFamilyInfo[] = [SO101, MAKER, METAL, SO101_TWIN, NINE, PANELED];

describe("calibrationKind", () => {
  // The dialog picks its whole calibration UI off this one value (the sweep
  // wizard, the step wizard, or the extension's panel notice), so it must
  // come from the manifest's `calibration.kind` and never from the id.
  it("is range_sweep for the SO-101 sweep families (built-in and twin)", () => {
    expect(calibrationKind(SO101)).toBe("range_sweep");
    expect(calibrationKind(SO101_TWIN)).toBe("range_sweep");
  });

  it("is steps for every step-wizard family, built-in or not", () => {
    expect(calibrationKind(MAKER)).toBe("steps");
    expect(calibrationKind(METAL)).toBe("steps");
    expect(calibrationKind(NINE)).toBe("steps");
  });

  it("is panel for a family that calibrates through its own served page", () => {
    expect(calibrationKind(PANELED)).toBe("panel");
  });

  it("is range_sweep for undefined — the SO-101 shape, like every other fallback", () => {
    expect(calibrationKind(undefined)).toBe("range_sweep");
  });
});

describe("supportsAutoCalibration", () => {
  it("follows the manifest flag, not the id", () => {
    expect(supportsAutoCalibration(SO101)).toBe(true);
    expect(supportsAutoCalibration(SO101_TWIN)).toBe(true);
    expect(supportsAutoCalibration(MAKER)).toBe(false);
    expect(supportsAutoCalibration(METAL)).toBe(false);
    expect(supportsAutoCalibration(NINE)).toBe(false);
    expect(supportsAutoCalibration(PANELED)).toBe(false);
  });
});

describe("supportsPortProbe", () => {
  it("is true only where the follower answers a protocol probe", () => {
    expect(supportsPortProbe(SO101)).toBe(false);
    expect(supportsPortProbe(SO101_TWIN)).toBe(false);
    expect(supportsPortProbe(MAKER)).toBe(true);
    expect(supportsPortProbe(METAL)).toBe(true);
    expect(supportsPortProbe(NINE)).toBe(true);
    expect(supportsPortProbe(PANELED)).toBe(true);
  });
});

describe("usesFeetechBus", () => {
  it("gates the servo-register UI (wiggle, identity, motor power) by the manifest flag", () => {
    expect(usesFeetechBus(SO101)).toBe(true);
    expect(usesFeetechBus(SO101_TWIN)).toBe(true);
    expect(usesFeetechBus(MAKER)).toBe(false);
    expect(usesFeetechBus(METAL)).toBe(false);
    expect(usesFeetechBus(NINE)).toBe(false);
    expect(usesFeetechBus(PANELED)).toBe(false);
  });
});

describe("supportsDagger", () => {
  it("is a hardware fact carried by the manifest (the Star leader has no motors to back-drive)", () => {
    expect(supportsDagger(SO101)).toBe(true);
    expect(supportsDagger(SO101_TWIN)).toBe(true);
    expect(supportsDagger(MAKER)).toBe(false);
    expect(supportsDagger(METAL)).toBe(false);
    expect(supportsDagger(NINE)).toBe(false);
    expect(supportsDagger(PANELED)).toBe(false);
  });
});

describe("jointsPerArm", () => {
  it("reads the width from the manifest — 6, 7, and a 9 no built-in has", () => {
    expect(jointsPerArm(SO101)).toBe(6);
    expect(jointsPerArm(SO101_TWIN)).toBe(6);
    expect(jointsPerArm(MAKER)).toBe(7);
    expect(jointsPerArm(METAL)).toBe(7);
    expect(jointsPerArm(NINE)).toBe(9);
    expect(jointsPerArm(PANELED)).toBe(7);
  });
});

describe("telemetryKind", () => {
  it("is urdf for the families that ship a model and degrees for the readout-only ones", () => {
    expect(telemetryKind(SO101)).toBe("urdf");
    expect(telemetryKind(SO101_TWIN)).toBe("urdf");
    expect(telemetryKind(MAKER)).toBe("urdf");
    expect(telemetryKind(METAL)).toBe("degrees");
    expect(telemetryKind(NINE)).toBe("degrees");
    expect(telemetryKind(PANELED)).toBe("degrees");
  });
});

describe("the undefined fallback", () => {
  // The manifest has not loaded yet, or the record's arm type is not
  // installed. Every predicate answers with the SO-101 shape — what the
  // pre-manifest `?? "so101"` renders drew — so a loading page looks exactly
  // as it did. Callers gate an UNAVAILABLE arm on record.arm_available first.
  it("answers every predicate with the SO-101 shape", () => {
    expect(calibrationKind(undefined)).toBe("range_sweep");
    expect(supportsAutoCalibration(undefined)).toBe(true);
    expect(supportsPortProbe(undefined)).toBe(false);
    expect(usesFeetechBus(undefined)).toBe(true);
    expect(supportsDagger(undefined)).toBe(true);
    expect(jointsPerArm(undefined)).toBe(6);
    expect(telemetryKind(undefined)).toBe("urdf");
  });
});

describe("armTypeFromRobotType", () => {
  // Mirrors arm_capabilities.arm_type_from_robot_type — a robot_type STRING
  // (from a dataset's meta/info.json), scanned against every family's
  // manifest markers.
  it("maps the names this app records, for built-ins and extensions alike", () => {
    expect(armTypeFromRobotType(ARMS, "so101_follower")).toBe("so101");
    expect(armTypeFromRobotType(ARMS, "bi_so_follower")).toBe("so101");
    expect(armTypeFromRobotType(ARMS, "maker_follower")).toBe("maker");
    expect(armTypeFromRobotType(ARMS, "bi_maker_follower")).toBe("maker");
    expect(armTypeFromRobotType(ARMS, "metal_follower")).toBe("metal");
    expect(armTypeFromRobotType(ARMS, "bi_metal_follower")).toBe("metal");
    expect(armTypeFromRobotType(ARMS, "nine_follower")).toBe("nine");
    expect(armTypeFromRobotType(ARMS, "paneled_follower")).toBe("paneled");
  });

  it("maps legacy / differently-cased / padded strings", () => {
    expect(armTypeFromRobotType(ARMS, "so100_follower")).toBe("so101");
    expect(armTypeFromRobotType(ARMS, "  Maker_Follower ")).toBe("maker");
    expect(armTypeFromRobotType(ARMS, "SO-101")).toBe("so101");
  });

  it("matches a marker anywhere in the string (greedy substring scan)", () => {
    expect(armTypeFromRobotType(ARMS, "experimental_metal_rig")).toBe("metal");
    expect(armTypeFromRobotType(ARMS, "bi_so100_follower")).toBe("so101");
    expect(armTypeFromRobotType(ARMS, "so_leader")).toBe("so101");
  });

  it("checks the default (first) family LAST, so a tighter marker wins over its loose ones", () => {
    // "so101_twin_follower" contains the default's "so101" marker AND the
    // twin's "twin" marker. Scanning the default first would swallow every
    // extension whose name embeds a built-in's; the default is the fallback,
    // not the first guess.
    expect(armTypeFromRobotType(ARMS, "so101_twin_follower")).toBe(
      "so101_twin",
    );
    expect(armTypeFromRobotType([SO101, SO101_TWIN], "bi_so101_twin_follower")).toBe(
      "so101_twin",
    );
    // …but the default still catches what only its markers match.
    expect(armTypeFromRobotType([SO101, SO101_TWIN], "so101_follower")).toBe(
      "so101",
    );
  });

  it("scans non-default families in manifest order", () => {
    // Two non-default families both match: the earlier manifest entry wins.
    const twinFirst: ArmFamilyInfo[] = [SO101, SO101_TWIN, NINE];
    const nineFirst: ArmFamilyInfo[] = [SO101, NINE, SO101_TWIN];
    expect(armTypeFromRobotType(twinFirst, "twin_nine")).toBe("so101_twin");
    expect(armTypeFromRobotType(nineFirst, "twin_nine")).toBe("nine");
  });

  it("returns null — not a default — when the arm can't be established", () => {
    expect(armTypeFromRobotType(ARMS, null)).toBeNull();
    expect(armTypeFromRobotType(ARMS, undefined)).toBeNull();
    expect(armTypeFromRobotType(ARMS, "")).toBeNull();
    expect(armTypeFromRobotType(ARMS, "   ")).toBeNull();
    expect(armTypeFromRobotType(ARMS, "aloha")).toBeNull();
  });

  it("returns null when the manifest is empty (not loaded yet)", () => {
    expect(armTypeFromRobotType([], "so101_follower")).toBeNull();
  });
});

describe("armLabel", () => {
  // i18next's `t(key, { defaultValue })` returns the catalog string when the
  // key exists and the defaultValue otherwise. This fake pins both branches
  // without depending on the catalog's contents.
  const KNOWN: Record<string, string> = {
    "robot.corner.armType.so101": "SO-101 (catalog)",
    "robot.corner.armType.maker": "Maker (catalog)",
    "robot.corner.armType.metal": "Metal (catalog)",
  };
  const fakeT = ((key: string, opts?: { defaultValue?: string }) =>
    KNOWN[key] ?? opts?.defaultValue ?? key) as unknown as TFunction;

  it("prefers the catalog's per-id label when it has one", () => {
    expect(armLabel(SO101, "so101", fakeT)).toBe("SO-101 (catalog)");
    expect(armLabel(MAKER, "maker", fakeT)).toBe("Maker (catalog)");
    expect(armLabel(METAL, "metal", fakeT)).toBe("Metal (catalog)");
  });

  it("falls back to the manifest label for an id the catalog does not know", () => {
    expect(armLabel(NINE, "nine", fakeT)).toBe("Nine Arm");
    expect(armLabel(SO101_TWIN, "so101_twin", fakeT)).toBe("SO-101 Twin");
    expect(armLabel(PANELED, "paneled", fakeT)).toBe("Paneled Arm");
  });

  it("falls back to the bare id when the manifest has no entry either (arm not installed)", () => {
    expect(armLabel(undefined, "nope", fakeT)).toBe("nope");
  });

  it("looks the built-ins up under robot.corner.armType.<id> in the real catalog", () => {
    // Pins the key PATH: the fake above would pass with any path.
    const t = i18n.getFixedT("en") as unknown as TFunction;
    expect(armLabel(SO101, "so101", t)).toBe("SO-101");
    expect(armLabel(MAKER, "maker", t)).toBe("Maker");
    expect(armLabel(METAL, "metal", t)).toBe("Metal");
    expect(armLabel(NINE, "nine", t)).toBe("Nine Arm");
  });
});
