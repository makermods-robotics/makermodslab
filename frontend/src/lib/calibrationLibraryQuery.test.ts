import { describe, expect, it } from "vitest";
import { libraryQuery } from "./calibrationLibraryQuery";

describe("libraryQuery", () => {
  it("is byte-identical to the pre-leader-kind query when no kind is given", () => {
    expect(libraryQuery("so101")).toBe("?arm_type=so101");
    expect(libraryQuery("metal", undefined)).toBe("?arm_type=metal");
    expect(libraryQuery("metal", "")).toBe("?arm_type=metal");
  });

  it("appends the leader kind, encoded, when one is given", () => {
    expect(libraryQuery("metal", "metal")).toBe(
      "?arm_type=metal&leader_kind=metal",
    );
    expect(libraryQuery("a b", "x/y")).toBe("?arm_type=a%20b&leader_kind=x%2Fy");
  });
});
