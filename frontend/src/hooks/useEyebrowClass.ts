import { useLanguage } from "@/contexts/LanguageContext";
import { isCaselessScript } from "@/i18n/config";

/**
 * The class an eyebrow heading wears, given the active language.
 *
 * `.eyebrow` (src/index.css) bundles `uppercase` with `tracking-[0.08em]`. On a
 * caseless script the uppercase is a no-op but the letter-spacing is not — it
 * renders CJK headings visibly over-spaced. The class sits in Tailwind's
 * utilities layer, so a `tracking-normal` override beside it would be a
 * source-order coin flip; drop the whole utility instead and keep only its
 * size/weight/colour.
 *
 * Lives here rather than in the studio's panel primitives because four areas
 * (studio, library, training, recording) need the same decision.
 */
export function useEyebrowClass(): string {
  const { language } = useLanguage();
  return isCaselessScript(language)
    ? "text-[11px] font-semibold text-muted-foreground"
    : "eyebrow";
}
