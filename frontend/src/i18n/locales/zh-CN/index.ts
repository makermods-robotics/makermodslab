import common from "./common";
import launchpad from "./launchpad";
import onboarding from "./onboarding";
import robot from "./robot";

/** Simplified Chinese catalog. Key tree must match `locales/en` exactly —
 * enforced by `src/i18n/catalogs.test.ts`. */
export default { common, launchpad, onboarding, robot } as const;
