import { describe, expect, it } from "vitest";
import {
  NodeEntry,
  hostingNodes,
  isSelectableNode,
  listableNodes,
  nodeDisplayName,
  nodeGpuLabel,
} from "./nodesApi";

const base: NodeEntry = {
  name: "bench-rig",
  url: "http://bench-rig.local:8000",
  instance_id: "a1f3c9d2e4b6a8c0a1f3c9d2e4b6a8c0",
  version: "0.9.2",
  capabilities: { serves_ui: true, accepts_jobs: true },
  status: "ok",
  source: "manual",
  is_self: false,
  last_seen_at: 1_700_000_000,
  last_verified_at: 123.4,
};

describe("isSelectableNode", () => {
  it("accepts a verified, reachable, job-accepting peer", () => {
    expect(isSelectableNode(base)).toBe(true);
  });

  it("refuses an unreachable peer", () => {
    expect(isSelectableNode({ ...base, status: "unreachable" })).toBe(false);
  });

  it("refuses a pending candidate (no identity to route by)", () => {
    expect(
      isSelectableNode({
        ...base,
        status: "pending",
        instance_id: null,
        capabilities: null,
      }),
    ).toBe(false);
  });

  it("refuses a peer that does not accept jobs", () => {
    expect(
      isSelectableNode({
        ...base,
        capabilities: { serves_ui: true, accepts_jobs: false },
      }),
    ).toBe(false);
  });
});

describe("listableNodes", () => {
  it("drops the self entry and declared non-workers, keeps the unreachable", () => {
    const self: NodeEntry = { ...base, is_self: true, url: null };
    const dead: NodeEntry = {
      ...base,
      name: "printer-pi",
      status: "unreachable",
      capabilities: null,
    };
    const uiOnly: NodeEntry = {
      ...base,
      name: "kiosk",
      capabilities: { serves_ui: true, accepts_jobs: false },
    };
    expect(listableNodes([self, base, dead, uiOnly])).toEqual([base, dead]);
  });
});

describe("nodeDisplayName", () => {
  it("prefers the name", () => {
    expect(nodeDisplayName(base)).toBe("bench-rig");
  });

  it("falls back to the URL host, then a short instance id", () => {
    expect(nodeDisplayName({ ...base, name: null })).toBe(
      "bench-rig.local:8000",
    );
    expect(nodeDisplayName({ ...base, name: null, url: null })).toBe(
      "a1f3c9d2",
    );
  });
});

describe("nodeGpuLabel", () => {
  it("renders a string capability verbatim", () => {
    expect(
      nodeGpuLabel({
        ...base,
        capabilities: { accepts_jobs: true, gpu: "RTX 4090 · 24GB" },
      }),
    ).toBe("RTX 4090 · 24GB");
  });

  it("joins an object capability's name and vram", () => {
    expect(
      nodeGpuLabel({
        ...base,
        capabilities: {
          accepts_jobs: true,
          gpu: { name: "RTX 4090", vram: "24GB" },
        },
      }),
    ).toBe("RTX 4090 · 24GB");
  });

  it("returns null when absent or empty", () => {
    expect(nodeGpuLabel(base)).toBeNull();
    expect(
      nodeGpuLabel({ ...base, capabilities: { accepts_jobs: true, gpu: {} } }),
    ).toBeNull();
    expect(nodeGpuLabel({ ...base, capabilities: null })).toBeNull();
  });
});

describe("hostingNodes", () => {
  const hosting = { robot: "bench", arm_type: "so101" };

  it("lists reachable, verified peers that advertise a hosted robot", () => {
    const station = {
      ...base,
      capabilities: { ...base.capabilities, hosting },
    };
    expect(hostingNodes([station])).toEqual([station]);
  });

  it("never lists the self entry, an unreachable peer, or one not hosting", () => {
    const station = {
      ...base,
      capabilities: { ...base.capabilities, hosting },
    };
    expect(
      hostingNodes([
        { ...station, is_self: true, url: null },
        { ...station, status: "unreachable" },
        { ...station, status: "pending", instance_id: null, capabilities: null },
        base,
      ]),
    ).toEqual([]);
  });
});
