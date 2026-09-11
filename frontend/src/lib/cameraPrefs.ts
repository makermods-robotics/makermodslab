/**
 * Remembers, per robot, whether the user had the Robot settings camera section
 * switched on.
 *
 * The settings window mounts fresh on every open and unmounts on close, which
 * is what makes its drafts reset — but it also meant the camera switch snapped
 * back to off every time, so a user who had already configured and previewed
 * their cameras had to turn them on again on each visit.
 *
 * Off is still the default for a robot the user has never turned cameras on
 * for: opening the window must not grab a camera on its own. What is stored
 * here is the user's own explicit switch-on, replayed — not a new decision made
 * on their behalf.
 *
 * Keyed by robot because cameras are a per-robot fact: a rig with three USB
 * cameras and a bare arm on the next bench should not inherit each other's
 * answer. Stale entries for renamed or deleted robots are harmless — they are
 * just booleans nothing reads.
 */

const KEY = "makermodslab.camerasActive";

type PrefMap = Record<string, boolean>;

const readAll = (): PrefMap => {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    // Anything but a plain object means the key was written by something else
    // (or an older shape); treat it as absent rather than throwing on read.
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    return parsed as PrefMap;
  } catch {
    // Unavailable (private mode, quota) or unparsable — default everyone off.
    return {};
  }
};

/** Whether this robot's camera section should come up already switched on. */
export const readCamerasActive = (robotName: string): boolean =>
  readAll()[robotName] === true;

/** Record the user's switch position for this robot. */
export const writeCamerasActive = (robotName: string, active: boolean): void => {
  try {
    const all = readAll();
    // Delete rather than store `false`: off is the default, so an absent key
    // says the same thing and the map stays as small as the set of robots the
    // user actually uses cameras with.
    if (active) all[robotName] = true;
    else delete all[robotName];
    localStorage.setItem(KEY, JSON.stringify(all));
  } catch {
    // storage unavailable — the in-memory switch still works for this visit
  }
};
