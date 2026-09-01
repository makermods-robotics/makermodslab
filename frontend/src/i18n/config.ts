/**
 * Language identity, persistence, and detection.
 *
 * Kept free of any i18next import so tests (and `LanguageContext`) can exercise
 * detection without booting the whole translation stack.
 */

/** Locale tags the app ships catalogs for. `code` is always ASCII — it is
 * persisted verbatim, so it must never become localized text. */
export const SUPPORTED_LANGUAGES = [
  { code: "en", label: "English" },
  // Endonym, not "Simplified Chinese" — a language picker is the one place a
  // language must NOT be named in the language the user is trying to leave.
  { code: "zh-CN", label: "简体中文" },
] as const;

export type Language = (typeof SUPPORTED_LANGUAGES)[number]["code"];

export const DEFAULT_LANGUAGE: Language = "en";

/** Matches the `makerlab:*` prefix the onboarding flags use. Storage keys are
 * inconsistent repo-wide (`makermodslab.*`, `vite-ui-theme`); this follows the
 * newest convention. */
export const LANGUAGE_STORAGE_KEY = "makerlab:language";

export function isSupportedLanguage(value: unknown): value is Language {
  return SUPPORTED_LANGUAGES.some((l) => l.code === value);
}

/** The persisted choice, or null when unset/unreadable/no longer supported. */
export function readStoredLanguage(): Language | null {
  try {
    const raw = localStorage.getItem(LANGUAGE_STORAGE_KEY);
    return isSupportedLanguage(raw) ? raw : null;
  } catch {
    // Storage may be unavailable (private mode, quota). Non-fatal.
    return null;
  }
}

export function storeLanguage(language: Language): void {
  try {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
  } catch {
    // Non-fatal: the choice just won't survive a reload.
  }
}

/** Best-effort match of a BCP-47 tag onto a shipped catalog. Any `zh-*` tag
 * (zh, zh-CN, zh-Hans, zh-SG…) resolves to zh-CN — we ship one Chinese
 * catalog, and Simplified is the closest fit for all of them. */
export function matchLanguage(tag: string | undefined | null): Language | null {
  if (!tag) return null;
  const lower = tag.toLowerCase();
  if (lower === "zh" || lower.startsWith("zh-")) return "zh-CN";
  if (lower === "en" || lower.startsWith("en-")) return "en";
  return null;
}

/**
 * Resolution order: an explicit stored choice, then the browser's languages,
 * then English. Called once at startup.
 */
export function detectLanguage(): Language {
  const stored = readStoredLanguage();
  if (stored) return stored;
  try {
    const tags =
      navigator.languages?.length ? navigator.languages : [navigator.language];
    for (const tag of tags) {
      const matched = matchLanguage(tag);
      if (matched) return matched;
    }
  } catch {
    // navigator may be absent (SSR, odd test envs).
  }
  return DEFAULT_LANGUAGE;
}

/**
 * True for languages whose script has no letter case. CSS `uppercase` is a
 * no-op on CJK text, but the `tracking-*` that usually accompanies it is not —
 * leaving it on renders visibly over-spaced pills. Callers drop both together.
 */
export function isCaselessScript(language: Language): boolean {
  return language === "zh-CN";
}
