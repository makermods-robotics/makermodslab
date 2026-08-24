import { afterEach, describe, expect, it, vi } from "vitest";
import {
  LANGUAGE_STORAGE_KEY,
  detectLanguage,
  isSupportedLanguage,
  matchLanguage,
  readStoredLanguage,
  storeLanguage,
} from "@/i18n/config";

/**
 * localStorage is stubbed explicitly rather than using the ambient global:
 * some Node versions expose their own partial `localStorage` that shadows
 * jsdom's and lacks methods (it is what breaks lib/onboarding/storage.test.ts
 * on Node 25), so relying on the ambient one makes these tests environment
 * dependent.
 */
function stubStorage(initial: Record<string, string> = {}, opts: { throws?: boolean } = {}) {
  const store = new Map(Object.entries(initial));
  const storage = {
    getItem: vi.fn((k: string) => {
      if (opts.throws) throw new Error("denied");
      return store.get(k) ?? null;
    }),
    setItem: vi.fn((k: string, v: string) => {
      if (opts.throws) throw new Error("denied");
      store.set(k, v);
    }),
    removeItem: vi.fn((k: string) => void store.delete(k)),
    clear: vi.fn(() => store.clear()),
    key: vi.fn(),
    length: 0,
  };
  vi.stubGlobal("localStorage", storage);
  return { storage, store };
}

function stubNavigator(languages: string[]) {
  vi.stubGlobal("navigator", { language: languages[0], languages });
}

afterEach(() => vi.unstubAllGlobals());

describe("matchLanguage", () => {
  it.each(["zh", "zh-CN", "zh-Hans", "zh-TW", "ZH-sg"])(
    "maps %s onto the Simplified Chinese catalog",
    (tag) => expect(matchLanguage(tag)).toBe("zh-CN"),
  );

  it.each(["en", "en-US", "en-GB"])("maps %s onto English", (tag) =>
    expect(matchLanguage(tag)).toBe("en"),
  );

  it.each(["fr", "de-DE", "ja", "", null, undefined])(
    "returns null for unshipped tag %s",
    (tag) => expect(matchLanguage(tag as string)).toBeNull(),
  );
});

describe("isSupportedLanguage", () => {
  it("accepts shipped codes and rejects anything else", () => {
    expect(isSupportedLanguage("en")).toBe(true);
    expect(isSupportedLanguage("zh-CN")).toBe(true);
    expect(isSupportedLanguage("zh")).toBe(false);
    expect(isSupportedLanguage(null)).toBe(false);
  });
});

describe("readStoredLanguage / storeLanguage", () => {
  it("round-trips a supported code", () => {
    const { store } = stubStorage();
    storeLanguage("zh-CN");
    expect(store.get(LANGUAGE_STORAGE_KEY)).toBe("zh-CN");
    expect(readStoredLanguage()).toBe("zh-CN");
  });

  it("ignores a stored value that is no longer supported", () => {
    stubStorage({ [LANGUAGE_STORAGE_KEY]: "klingon" });
    expect(readStoredLanguage()).toBeNull();
  });

  it("survives a throwing localStorage", () => {
    stubStorage({}, { throws: true });
    expect(() => storeLanguage("zh-CN")).not.toThrow();
    expect(readStoredLanguage()).toBeNull();
  });
});

describe("detectLanguage", () => {
  it("prefers an explicit stored choice over the browser language", () => {
    stubStorage({ [LANGUAGE_STORAGE_KEY]: "en" });
    stubNavigator(["zh-CN"]);
    expect(detectLanguage()).toBe("en");
  });

  it("falls back to the browser language when nothing is stored", () => {
    stubStorage();
    stubNavigator(["zh-CN", "en-US"]);
    expect(detectLanguage()).toBe("zh-CN");
  });

  it("skips unshipped browser languages and takes the first match", () => {
    stubStorage();
    stubNavigator(["fr-FR", "de", "zh-Hans"]);
    expect(detectLanguage()).toBe("zh-CN");
  });

  it("defaults to English when no browser language matches", () => {
    stubStorage();
    stubNavigator(["fr-FR", "de"]);
    expect(detectLanguage()).toBe("en");
  });

  it("defaults to English when storage throws and navigator is absent", () => {
    stubStorage({}, { throws: true });
    vi.stubGlobal("navigator", undefined);
    expect(detectLanguage()).toBe("en");
  });
});
