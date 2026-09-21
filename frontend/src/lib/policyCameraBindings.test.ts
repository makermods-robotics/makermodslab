import { describe, expect, it } from "vitest";
import { policyCameraBindings } from "./policyCameraBindings";

const inputs = (names: string[]) => names.map((name) => ({ requestKey: name, display: name }));

describe("camera cards to policy inputs", () => {
  it("maps the displayed top, side, wrist order and declares the third view", () => {
    expect(policyCameraBindings(inputs(["cam0", "cam1"]), ["top", "side", "wrist"], true))
      .toEqual({ roles: { cam0: "top", cam1: "side", cam2: "wrist" }, extra: ["cam2"] });
  });

  it("uses the numeric slot, regardless of the checkpoint's key order", () => {
    expect(policyCameraBindings(inputs(["cam1", "cam0"]), ["top", "side", "wrist"], true))
      .toEqual({ roles: { cam0: "top", cam1: "side", cam2: "wrist" }, extra: ["cam2"] });
  });

  it("keeps meaningful camera names and their original spelling", () => {
    expect(policyCameraBindings(inputs(["wrist", "top"]), ["Top", "side", "Wrist"], false))
      .toEqual({ roles: { top: "Top", wrist: "Wrist" }, extra: [] });
  });

  it("does not add a third input to fixed-view models or local runs", () => {
    expect(policyCameraBindings(inputs(["cam0", "cam1"]), ["top", "side", "wrist"], false))
      .toEqual({ roles: { cam0: "top", cam1: "side" }, extra: [] });
  });

  it("does not guess missing named inputs or shift a missing numbered slot", () => {
    expect(policyCameraBindings(inputs(["front", "wrist"]), ["top", "side", "wrist"], true))
      .toEqual({ roles: { wrist: "wrist" }, extra: [] });
    expect(policyCameraBindings(inputs(["cam2"]), ["top", "side"], true))
      .toEqual({ roles: {}, extra: [] });
  });

  it("waits for checkpoint inputs and respects the extra-view limit", () => {
    expect(policyCameraBindings([], ["top", "side"], true)).toEqual({ roles: {}, extra: [] });
    expect(policyCameraBindings(inputs(["cam0"]), ["top", "side", "wrist", "back"], true))
      .toEqual({ roles: { cam0: "top", cam1: "side", cam2: "wrist" }, extra: ["cam1", "cam2"] });
  });

  it("does not give a camera already matched by name to another input", () => {
    expect(policyCameraBindings(inputs(["cam0", "cam1"]), ["cam1", "top"], true))
      .toEqual({ roles: { cam1: "cam1" }, extra: [] });
  });
});
