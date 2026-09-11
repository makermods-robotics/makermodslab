import type en from "./locales/en";

/**
 * Makes the English catalog the source of truth for `t()` keys, so a typo is a
 * compile error rather than a string that silently renders as its own key.
 * `tsconfig.app.json` runs with `strict: false`, so this augmentation is the
 * main guard we have — keep it.
 */
declare module "i18next" {
  interface CustomTypeOptions {
    defaultNS: "translation";
    resources: {
      translation: typeof en;
    };
  }
}
