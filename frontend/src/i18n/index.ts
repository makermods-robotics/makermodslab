import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import en from "./locales/en";
import zhCN from "./locales/zh-CN";
import { DEFAULT_LANGUAGE, detectLanguage } from "./config";

/**
 * Catalogs are imported statically rather than fetched at runtime: the built
 * bundle in `frontend/dist/` is committed and served by FastAPI as plain
 * StaticFiles, so there is no place to serve locale JSON from and no loading
 * state to design around.
 */
export const resources = {
  en: { translation: en },
  "zh-CN": { translation: zhCN },
} as const;

i18n.use(initReactI18next).init({
  resources,
  lng: detectLanguage(),
  fallbackLng: DEFAULT_LANGUAGE,
  interpolation: {
    // React already escapes interpolated values; double-escaping mangles
    // names that contain quotes or ampersands.
    escapeValue: false,
  },
  // Resources are bundled, so nothing is ever loaded asynchronously and there
  // is no need to suspend.
  react: { useSuspense: false },
});

export default i18n;
