import { describe, expect, it } from "vitest";
import { resources } from "@/i18n";
import { SUPPORTED_LANGUAGES } from "@/i18n/config";

type Json = Record<string, unknown>;

/** i18next plural suffixes. `en` needs _one/_other where `zh-CN` needs only
 * _other, so keys are compared with the suffix stripped. */
const PLURAL_SUFFIX = /_(zero|one|two|few|many|other)$/;

function flatten(obj: Json, prefix = ""): Map<string, string> {
  const out = new Map<string, string>();
  for (const [k, v] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object") {
      for (const [ck, cv] of flatten(v as Json, path)) out.set(ck, cv);
    } else {
      out.set(path, String(v));
    }
  }
  return out;
}

const base = (key: string) => key.replace(PLURAL_SUFFIX, "");
const catalogs = Object.fromEntries(
  Object.entries(resources).map(([lng, r]) => [
    lng,
    flatten(r.translation as unknown as Json),
  ]),
) as Record<string, Map<string, string>>;

const EN = "en";
const OTHERS = SUPPORTED_LANGUAGES.map((l) => l.code).filter((c) => c !== EN);

describe("translation catalogs", () => {
  it("ships a catalog for every supported language", () => {
    for (const { code } of SUPPORTED_LANGUAGES) {
      expect(catalogs[code], `missing catalog for ${code}`).toBeDefined();
    }
  });

  describe.each(OTHERS)("%s", (lng) => {
    // The whole point of this file: a translation PR that forgets a key, or
    // leaves a stale one behind after an English rename, fails CI instead of
    // rendering the raw key path to a user.
    it("has no keys missing relative to en", () => {
      const enKeys = new Set([...catalogs[EN].keys()].map(base));
      const theirs = new Set([...catalogs[lng].keys()].map(base));
      const missing = [...enKeys].filter((k) => !theirs.has(k)).sort();
      expect(missing, `missing in ${lng}`).toEqual([]);
    });

    it("has no orphan keys that en does not define", () => {
      const enKeys = new Set([...catalogs[EN].keys()].map(base));
      const theirs = new Set([...catalogs[lng].keys()].map(base));
      const orphans = [...theirs].filter((k) => !enKeys.has(k)).sort();
      expect(orphans, `orphaned in ${lng}`).toEqual([]);
    });
  });

  describe.each(SUPPORTED_LANGUAGES.map((l) => l.code))("%s values", (lng) => {
    it("has no empty strings", () => {
      const empty = [...catalogs[lng].entries()]
        .filter(([, v]) => v.trim() === "")
        .map(([k]) => k);
      expect(empty).toEqual([]);
    });

    it("keeps every interpolation placeholder that en declares", () => {
      const placeholders = (v: string) =>
        (v.match(/\{\{(\w+)\}\}/g) ?? []).sort().join(",");
      const mismatched: string[] = [];
      for (const [key, enValue] of catalogs[EN]) {
        const theirValue = catalogs[lng].get(key);
        // Plural variants legitimately differ per language; compare only keys
        // present verbatim in both.
        if (theirValue === undefined) continue;
        if (placeholders(enValue) !== placeholders(theirValue)) {
          mismatched.push(
            `${key}: en(${placeholders(enValue)}) vs ${lng}(${placeholders(theirValue)})`,
          );
        }
      }
      expect(mismatched).toEqual([]);
    });
  });
});
