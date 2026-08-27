// Copy helpers for the recording session's two explicit exits, kept out of
// the component so the fresh-vs-resume wording is testable/greppable in one place.
//
// The two exits and what they do to the episodes:
//   Done — end now, KEEP everything saved so far, go to the upload page.
//   Quit — end WITHOUT saving. A FRESH session's whole dataset (this session's
//          own creation) is deleted; a RESUME session keeps every episode
//          already committed to the pre-existing dataset and only drops the
//          in-progress take. An unintentional page exit is treated as Quit.
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

/**
 * Toast/confirm line for an UNINTENTIONAL leave (back button, tab close), which
 * is treated as Quit. Mirrors quitConfirmCopy's fresh-vs-resume distinction.
 *
 * ENGLISH ONLY, deliberately. This feeds the shared exit guard's native
 * `window.confirm()`, whose chrome (the OK/Cancel buttons, the origin line) is
 * rendered by the browser in the BROWSER's language — a translated body inside
 * an English dialog frame reads worse than an English one, and the native
 * prompt is not ours to style. `formatLeaveDiscardMessage` below is the
 * localized twin used everywhere the message is rendered by React.
 */
export function leaveDiscardMessage(resume: boolean): string {
  return resume
    ? "Leaving quits the recording without saving — episodes already saved stay in the dataset."
    : "Leaving quits the recording without saving — the recording and all its episodes will be deleted.";
}

/** Localized equivalent of `leaveDiscardMessage`, for React-rendered surfaces
 * (the discard toast). Same fresh-vs-resume split. */
export function formatLeaveDiscardMessage(
  t: TFunction,
  resume: boolean,
): string {
  return resume
    ? t("recording.exit.leaveResume")
    : t("recording.exit.leaveFresh");
}
