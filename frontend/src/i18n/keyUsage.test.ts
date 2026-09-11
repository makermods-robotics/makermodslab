import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import i18n, { resources } from "@/i18n";
import en from "@/i18n/locales/en";

/**
 * Guards the keys that `t()`'s TypeScript augmentation cannot check.
 *
 * Most call sites are typed, so a typo is a compile error. But ~17 sites index
 * a map or build a template literal and cast `as never` (backend enums, badge
 * maps, phase maps, delete-action keys). A wrong key there compiles fine and
 * silently renders the key path to the user. This scans the source for
 * key-shaped literals and asserts every one resolves in every language.
 */
const NAMESPACES = Object.keys(en);
// Only single/double-quoted literals: backticks in comments are prose (a
// comment saying `robot.cameras` means the RECORD's field, not a key).
const KEY_RE = new RegExp(
  `["'](${NAMESPACES.join("|")})\\.[A-Za-z0-9_.]+["']`,
  "g",
);

/** i18next plural suffixes — a stem resolves through one of these. */
const PLURAL_SUFFIXES = ["_one", "_other", "_zero", "_two", "_few", "_many"];

/** Strips // line and block comments so prose can't masquerade as a key. */
function stripComments(text: string): string {
  return text
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:])\/\/.*$/gm, "$1");
}

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) return walk(full);
    return /\.tsx?$/.test(entry) && !/\.test\.tsx?$/.test(entry) ? [full] : [];
  });
}

/** Literal keys referenced anywhere in src, minus the catalogs themselves. */
function collectKeys(): Map<string, string[]> {
  const found = new Map<string, string[]>();
  for (const file of walk("src")) {
    if (file.includes(join("i18n", "locales"))) continue;
    const text = stripComments(readFileSync(file, "utf8"));
    for (const m of text.matchAll(KEY_RE)) {
      const key = m[0].slice(1, -1);
      // Skip plural stems — i18next resolves those via `count`.
      if (/_(one|other|zero|two|few|many)$/.test(key)) continue;
      const at = found.get(key) ?? [];
      at.push(file);
      found.set(key, at);
    }
  }
  return found;
}

const KEYS = collectKeys();
const LANGUAGES = Object.keys(resources);

describe("translation key usage", () => {
  it("finds key literals to check", () => {
    expect(KEYS.size).toBeGreaterThan(50);
  });

  it.each(LANGUAGES)("every referenced key resolves in %s", (lng) => {
    // Deliberately untyped: this test's whole purpose is checking keys that
    // are runtime strings, which the typed `t()` signature rejects by design.
    const fixed = i18n.getFixedT(lng) as unknown as (
      key: string,
      opts?: Record<string, unknown>,
    ) => string;
    const unresolved: string[] = [];
    for (const [key, files] of KEYS) {
      // A key with a plural or a dynamic suffix resolves via its variants;
      // `exists` covers both.
      // A stem like `foo.bar` may exist only as `foo.bar_one`/`_other`.
      const resolves =
        i18n.exists(key, { lng }) ||
        PLURAL_SUFFIXES.some((sfx) => i18n.exists(key + sfx, { lng }));
      if (!resolves) {
        unresolved.push(`${key} (${files[0]})`);
        continue;
      }
      // A plural stem only resolves once i18next has a `count` to pick a
      // variant with; without one it hands back the key itself.
      let value = fixed(key);
      if (value === key) value = fixed(key, { count: 2 });
      if (typeof value === "string" && value === key) {
        unresolved.push(`${key} rendered as its own key (${files[0]})`);
      }
    }
    expect(unresolved).toEqual([]);
  });
});
