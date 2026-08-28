import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

import i18n from "@/i18n";
import {
  DEFAULT_LANGUAGE,
  Language,
  detectLanguage,
  storeLanguage,
} from "@/i18n/config";

export interface LanguageProviderState {
  language: Language;
  setLanguage: (language: Language) => void;
}

const initialState: LanguageProviderState = {
  language: DEFAULT_LANGUAGE,
  setLanguage: () => null,
};

export const LanguageProviderContext =
  createContext<LanguageProviderState>(initialState);

/**
 * App language, persisted to localStorage and mirrored onto i18next.
 *
 * Shaped after ThemeContext (lazy initializer, DOM-syncing effect, memoized
 * value) with one deliberate difference: every storage access is wrapped in
 * try/catch — ThemeContext's is not, but `useOnceFlag` and `useUpdateCheck`
 * both guard theirs, and a throwing localStorage (private mode, quota) should
 * not take down the whole app.
 *
 * The selection is display-only. It never reaches the backend and the stored
 * value is always an ASCII locale tag.
 */
export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<Language>(() =>
    detectLanguage(),
  );

  useEffect(() => {
    if (i18n.language !== language) void i18n.changeLanguage(language);
    // Keeps the static lang="en" in index.html honest, which matters for
    // screen readers and for the browser's own font fallback on CJK text.
    document.documentElement.lang = language;
  }, [language]);

  const setLanguage = useCallback((next: Language) => {
    storeLanguage(next);
    setLanguageState(next);
  }, []);

  const value = useMemo(
    () => ({ language, setLanguage }),
    [language, setLanguage],
  );

  return (
    <LanguageProviderContext.Provider value={value}>
      {children}
    </LanguageProviderContext.Provider>
  );
}

export function useLanguage(): LanguageProviderState {
  return useContext(LanguageProviderContext);
}
