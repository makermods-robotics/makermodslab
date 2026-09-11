import type { ParseKeys } from "i18next";

export type Placement = "top" | "bottom" | "left" | "right";

/** Any valid key in the bundled catalogs, checked at compile time. */
export type TranslationKey = ParseKeys;

export interface TourStep {
  /** CSS selector for the target element, e.g. '[data-tour="launchpad-search"]'. */
  target: string;
  /**
   * Tours are module-level constants, evaluated at import time — long before
   * any React or i18n context exists — so they hold translation KEYS and the
   * renderer resolves them. Storing resolved strings here would freeze the
   * copy in whichever language happened to load first.
   */
  titleKey: TranslationKey;
  descriptionKey: TranslationKey;
  placement?: Placement;
}

export interface Tour {
  id: string;
  steps: TourStep[];
}
