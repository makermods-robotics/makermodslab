// Copy helpers for the recording session's two explicit exits, kept out of
// the component so the fresh-vs-resume wording is testable/greppable in one place.
//
// The two exits and what they do to the episodes:
//   Done — end now, KEEP everything saved so far, go to the upload page.
//   Quit — end WITHOUT saving. A FRESH session's whole dataset (this session's
//          own creation) is deleted; a RESUME session keeps every episode
//          already committed to the pre-existing dataset and only drops the
//          in-progress take. (An abandoned page no longer discards anything:
//          the session's server-side lease expires and the safety stop KEEPS
//          the saved episodes — quit-without-saving is explicit-buttons-only.)
//
// These return STRUCTURE, not a resolved sentence assembled at module scope:
// `t` is injected by the caller so the copy tracks the live language (the same
// shape `formatRobotSetupGap` uses in lib/robotSetupGap.ts).

import type { TFunction } from "i18next";

export interface ExitConfirmCopy {
  title: string;
  description: string;
}

export function doneConfirmCopy(t: TFunction): ExitConfirmCopy {
  return {
    title: t("recording.exit.done.title"),
    description: t("recording.exit.done.description"),
  };
}

export function quitConfirmCopy(t: TFunction, resume: boolean): ExitConfirmCopy {
  return {
    title: t("recording.exit.quit.title"),
    description: resume
      ? t("recording.exit.quit.descriptionResume")
      : t("recording.exit.quit.descriptionFresh"),
  };
}
