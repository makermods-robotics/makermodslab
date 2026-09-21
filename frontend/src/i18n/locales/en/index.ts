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

/**
 * The English catalog — the source of truth for the key tree. `types.d.ts`
 * derives i18next's key types from this object, so a typo in a `t()` call is a
 * compile error.
 *
 * One namespace per feature area, mirroring the component directories, so a
 * migration PR touches exactly one catalog file per language.
 */
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
