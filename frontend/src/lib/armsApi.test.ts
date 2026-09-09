import { describe, expect, it } from "vitest";
import { ApiError, type Fetcher } from "./apiClient";
import { fetchArmFamilies, servedUrl, type ArmFamilyInfo } from "./armsApi";

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

/** An extension family with both served URLs set: the fetch must hand them
 * through verbatim — the dialog renders them, the client never rewrites. */
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

/** A Fetcher that records the URL it was given and answers with `body`. */
function fetcherReturning(
  status: number,
  body: unknown,
): { fetcher: Fetcher; calls: string[] } {
  const calls: string[] = [];
  const fetcher: Fetcher = async (url) => {
    calls.push(url);
    return new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  };
  return { fetcher, calls };
}

describe("fetchArmFamilies", () => {
  it("returns the manifest's `arms` array, in the order served", async () => {
    const { fetcher } = fetcherReturning(200, {
      arms: [SO101, NINE, PANELED],
    });
    await expect(
      fetchArmFamilies("http://localhost:8000", fetcher),
    ).resolves.toEqual([SO101, NINE, PANELED]);
  });

  it("hits GET ${baseUrl}/api/v1/arms — the v1 mount, never the flat one", async () => {
    const { fetcher, calls } = fetcherReturning(200, { arms: [] });
    await fetchArmFamilies("http://bench.local:8000", fetcher);
    expect(calls).toEqual(["http://bench.local:8000/api/v1/arms"]);
  });

  it("throws on a non-OK response instead of yielding an empty manifest", async () => {
    // An empty list would make every arm render as unavailable; a thrown
    // error lets the provider keep its last good manifest and report.
    const { fetcher } = fetcherReturning(500, { detail: "boom" });
    const err = await fetchArmFamilies("http://localhost:8000", fetcher).then(
      () => null,
      (e: unknown) => e,
    );
    // The same coded shape every other lib fetcher throws (apiRequest's
    // ApiError), so the provider can show status + detail, not a bare throw.
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).status).toBe(500);
    expect((err as ApiError).detail).toBe("boom");
  });
});

describe("servedUrl", () => {
  // The manifest's served URLs are root-relative backend paths. Under --dev
  // the page is served by Vite, so an <img src> must be built on baseUrl the
  // way every camera and episode src is.
  it("resolves a root-relative served path against the API origin", () => {
    expect(servedUrl("http://bench.local:8000", PANELED.image_url)).toBe(
      "http://bench.local:8000/api/v1/ext/paneled/static/arm.jpg",
    );
  });

  it("passes an absolute URL through and keeps null as null", () => {
    expect(
      servedUrl("http://bench.local:8000", "https://cdn.example/a.jpg"),
    ).toBe("https://cdn.example/a.jpg");
    expect(servedUrl("http://bench.local:8000", null)).toBeNull();
    expect(servedUrl("http://bench.local:8000", "")).toBeNull();
  });
});
