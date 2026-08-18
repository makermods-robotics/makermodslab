import common from "./common";
import launchpad from "./launchpad";
import onboarding from "./onboarding";
import robot from "./robot";

/**
 * The English catalog — the source of truth for the key tree. `types.d.ts`
 * derives i18next's key types from this object, so a typo in a `t()` call is a
 * compile error.
 *
 * Namespaces are added as areas get migrated. Still English-only, each landing
 * in its own follow-up: studio, recording, calibration, jobs, library, dialogs,
 * inference, training, errors.
 */
export default { common, launchpad, onboarding, robot } as const;
