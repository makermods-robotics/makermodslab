import { describe, expect, it } from "vitest";
import {
  mintOwnerId,
  readOrMintOwner,
  type StringStore,
} from "@/lib/sessionOwner";

const OWNER_KEY = "makermodslab:session-owner";

/** Minimal in-memory Storage stand-in. */
function memoryStore(initial: Record<string, string> = {}): StringStore & {
  data: Record<string, string>;
} {
  const data = { ...initial };
  return {
    data,
    getItem: (k) => (k in data ? data[k] : null),
    setItem: (k, v) => {
      data[k] = v;
    },
  };
}

describe("mintOwnerId", () => {
  it("mints a ui:-prefixed id well under the server's 128-char owner cap", () => {
    const id = mintOwnerId();
    expect(id.startsWith("ui:")).toBe(true);
    expect(id.length).toBeGreaterThan(3);
    expect(id.length).toBeLessThanOrEqual(128);
  });

  it("differs for different random draws", () => {
    const a = mintOwnerId(() => 0.1234567);
    const b = mintOwnerId(() => 0.7654321);
    expect(a).not.toBe(b);
  });
});

describe("readOrMintOwner", () => {
  it("reuses a stored id so a reload keeps renewing the same lease", () => {
    const store = memoryStore({ [OWNER_KEY]: "ui:stored" });
    expect(readOrMintOwner(store)).toBe("ui:stored");
  });

  it("mints and persists on first use", () => {
    const store = memoryStore();
    const id = readOrMintOwner(store, () => "ui:fresh");
    expect(id).toBe("ui:fresh");
    expect(store.data[OWNER_KEY]).toBe("ui:fresh");
  });

  it("returns a usable id when the store is missing entirely", () => {
    expect(readOrMintOwner(null, () => "ui:ephemeral")).toBe("ui:ephemeral");
  });

  it("survives a store whose reads and writes throw", () => {
    const throwing: StringStore = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
    };
    expect(readOrMintOwner(throwing, () => "ui:fallback")).toBe("ui:fallback");
  });

  it("ignores an empty stored value instead of returning it", () => {
    const store = memoryStore({ [OWNER_KEY]: "" });
    expect(readOrMintOwner(store, () => "ui:replacement")).toBe(
      "ui:replacement"
    );
  });
});
