import { useCallback, useEffect, useMemo, useState } from "react";

/**
 * Remembers which robot camera a checkpoint's camera ROLE was bound to, per
 * (checkpoint, robot).
 *
 * The design decision behind this, spelled out because the storage shape only
 * makes sense with it: a camera's NAME in the robot record is its IDENTITY and
 * is never renamed to suit a checkpoint. A checkpoint's camera name is a ROLE,
 * bound per run. Name-match remains the default and needs no control — this map
 * only ever holds the roles that matched NOTHING by name (`cam0` / `cam1` on
 * `lerobot/MolmoAct2-SO100_101-LeRobot`, say), which without a pick would leave
 * Start permanently blocked on a binding the operator has no way to make.
 *
 * Keyed by both halves because both halves matter: a role belongs to the
 * checkpoint (a different checkpoint names different cameras) and the answer
 * belongs to the robot (the bench with three USB cameras and the bare arm next
 * to it must not inherit each other's). The checkpoint half is the panel's own
 * identity for the thing it fetched a policy config for — the owning job id
 * plus the checkpoint ref, which is what addresses `/policy-config` — joined
 * with a separator. A collision there is not a correctness problem: every
 * remembered pick is re-checked against the robot's CURRENT camera names on
 * read, and one naming a camera the robot no longer has is dropped silently.
 *
 * Storage is best-effort in the same way `useSelectedModel` and `cameraPrefs`
 * are: every read and write is wrapped, and an unavailable store (private mode,
 * quota) costs the operator only the memory, never the run.
 */

const STORAGE_KEY = "makermodslab.remoteCameraRoles";

/** role (the request key, i.e. the checkpoint's camera name) → robot camera name. */
export type CameraRoleMap = Record<string, string>;

/** checkpoint key → robot name → roles. */
type Stored = Record<string, Record<string, CameraRoleMap>>;

const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === "object" && !Array.isArray(v);

/**
 * The checkpoint half of the key. `null` whenever the panel has no checkpoint
 * resolved, which is also when there is nothing to remember.
 */
export function remoteCameraRoleKey(
  policyConfigJobId: string | null | undefined,
  checkpointRef: string | null | undefined,
): string | null {
  if (!checkpointRef) return null;
  return `${policyConfigJobId ?? "-"}::${checkpointRef}`;
}

function readAll(): Stored {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    return isPlainObject(parsed) ? (parsed as Stored) : {};
  } catch {
    // Unavailable or written by something else — start from nothing rather
    // than throwing on a read that only ever offers a convenience.
    return {};
  }
}

function readEntry(
  checkpointKey: string | null,
  robotName: string | null,
): CameraRoleMap {
  if (!checkpointKey || !robotName) return {};
  const forCheckpoint = readAll()[checkpointKey];
  const entry = isPlainObject(forCheckpoint) ? forCheckpoint[robotName] : null;
  if (!isPlainObject(entry)) return {};
  // Values are camera NAMES; anything else came from a different writer.
  const out: CameraRoleMap = {};
  for (const [role, camera] of Object.entries(entry)) {
    if (typeof camera === "string" && camera) out[role] = camera;
  }
  return out;
}

function writeEntry(
  checkpointKey: string,
  robotName: string,
  roles: CameraRoleMap,
): void {
  try {
    const all = readAll();
    const forCheckpoint = isPlainObject(all[checkpointKey])
      ? { ...all[checkpointKey] }
      : {};
    // Delete rather than store an empty map, so clearing every pick leaves the
    // store the size it was before the operator ever opened the panel.
    if (Object.keys(roles).length > 0) forCheckpoint[robotName] = roles;
    else delete forCheckpoint[robotName];
    if (Object.keys(forCheckpoint).length > 0)
      all[checkpointKey] = forCheckpoint;
    else delete all[checkpointKey];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
  } catch {
    // storage unavailable — the in-memory picks still drive this visit's run
  }
}

export interface UseRemoteCameraRoles {
  /** The remembered picks, filtered to cameras the robot still has. */
  roles: CameraRoleMap;
  /** Bind a role to a robot camera by name, or `null` to leave it unbound. */
  setRole: (role: string, cameraName: string | null) => void;
}

/**
 * @param checkpointKey `remoteCameraRoleKey(...)`; null disables persistence.
 * @param robotName the selected robot record's name; null disables persistence.
 * @param cameraNames the robot's CURRENT camera names — a remembered pick
 *   naming anything else is dropped, because a binding the record cannot back
 *   is not a binding and must not count towards "every camera is bound".
 */
export function useRemoteCameraRoles(
  checkpointKey: string | null,
  robotName: string | null,
  cameraNames: string[],
): UseRemoteCameraRoles {
  const [stored, setStored] = useState<CameraRoleMap>(() =>
    readEntry(checkpointKey, robotName),
  );

  // Re-read on every change of either half of the key: this is the moment a
  // different checkpoint's (or robot's) answers become the right ones.
  useEffect(() => {
    setStored(readEntry(checkpointKey, robotName));
  }, [checkpointKey, robotName]);

  // Serialised rather than the array itself: `robot.cameras` is a fresh array
  // on every render of its parent, and a dependency on it would rebuild the set
  // (and the memo) on every one of them. JSON rather than a join because a
  // camera name is free text — any separator could appear inside one.
  const namesKey = JSON.stringify(cameraNames);

  const roles = useMemo(() => {
    const known = new Set(JSON.parse(namesKey) as string[]);
    const out: CameraRoleMap = {};
    for (const [role, camera] of Object.entries(stored)) {
      if (known.has(camera)) out[role] = camera;
    }
    return out;
  }, [stored, namesKey]);

  const setRole = useCallback(
    (role: string, cameraName: string | null) => {
      const next = { ...stored };
      if (cameraName) next[role] = cameraName;
      else delete next[role];
      setStored(next);
      if (checkpointKey && robotName)
        writeEntry(checkpointKey, robotName, next);
    },
    [stored, checkpointKey, robotName],
  );

  return { roles, setRole };
}
