import common from "./common";
import launchpad from "./launchpad";
import onboarding from "./onboarding";
import robot from "./robot";
import landing from "./landing";
import studio from "./studio";
import recording from "./recording";
import calibration from "./calibration";
import jobs from "./jobs";
import library from "./library";
import dialogs from "./dialogs";
import robotConfig from "./robotConfig";
import inference from "./inference";
import remoteInference from "./remoteInference";
import training from "./training";
import pages from "./pages";
import shared from "./shared";

/** Simplified Chinese catalog. Key tree must match `locales/en` exactly —
 * enforced by `src/i18n/catalogs.test.ts`. */
export default {
  common,
  launchpad,
  onboarding,
  robot,
  landing,
  studio,
  recording,
  calibration,
  jobs,
  library,
  dialogs,
  robotConfig,
  inference,
  remoteInference,
  training,
  pages,
  shared,
} as const;
