import { beforeEach, describe, expect, it } from "vitest";

import { readCamerasActive, writeCamerasActive } from "./cameraPrefs";

const KEY = "makermodslab.camerasActive";

beforeEach(() => {
  localStorage.clear();
});

describe("the camera switch is remembered per robot", () => {
  // The point of the whole module: Robot settings mounts fresh on every open,
  // so without this the switch snapped back to off and previews had to be
  // reopened by hand after every visit.
  it("replays a robot's switch-on across a remount", () => {
    writeCamerasActive("desk_isaac", true);
    expect(readCamerasActive("desk_isaac")).toBe(true);
  });

  // Cameras are a per-robot fact — a three-camera rig and a bare arm on the
  // next bench must not inherit each other's answer.
  it("does not leak one robot's answer to another", () => {
    writeCamerasActive("desk_isaac", true);
    expect(readCamerasActive("openbooth_lab_left")).toBe(false);
  });

  // Off is the default, so switching off has to restore the default rather
  // than persist a second kind of "on".
  it("forgets a robot switched back off", () => {
    writeCamerasActive("desk_isaac", true);
    writeCamerasActive("desk_isaac", false);
    expect(readCamerasActive("desk_isaac")).toBe(false);
    // Stored as an absence, not as `false` — the map only holds robots the
    // user actually uses cameras with.
    expect(JSON.parse(localStorage.getItem(KEY) ?? "{}")).toEqual({});
  });

  it("defaults a never-seen robot to off", () => {
    expect(readCamerasActive("brand_new")).toBe(false);
  });
});

describe("a corrupt or foreign value never breaks the dialog", () => {
  // This key is read during the settings window's first render. A throw here
  // would take the whole window down, so every bad shape has to read as "off"
  // rather than propagate.
  it.each([
    ["unparsable", "{not json"],
    ["an array", "[1,2,3]"],
    ["a bare string", '"on"'],
    ["null", "null"],
  ])("reads %s as off", (_label, raw) => {
    localStorage.setItem(KEY, raw);
    expect(() => readCamerasActive("desk_isaac")).not.toThrow();
    expect(readCamerasActive("desk_isaac")).toBe(false);
  });

  it("recovers by overwriting the bad value on the next write", () => {
    localStorage.setItem(KEY, "{not json");
    writeCamerasActive("desk_isaac", true);
    expect(readCamerasActive("desk_isaac")).toBe(true);
  });
});
