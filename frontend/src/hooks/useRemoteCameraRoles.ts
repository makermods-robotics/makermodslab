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
 *
 * Since S3.8g the same record also holds the EXTRA roles — views the checkpoint
 * does not declare at all, which the GPU side can be asked to add before the
 * weights load. They are a separate list rather than more entries in the map
 * for one reason: an extra role exists before it is bound to anything, and it
 * has to keep existing (and keep blocking Start) while it is unbound, which a
 * role → camera map has no way to say.
 */

const STORAGE_KEY = "makermodslab.remoteCameraRoles";

/** role (the request key, i.e. the checkpoint's camera name) → robot camera name. */
export type CameraRoleMap = Record<string, string>;

/** One (checkpoint, robot) pair's remembered answers. */
interface StoredEntry {
  roles: CameraRoleMap;
  /** Roles the CHECKPOINT does not declare, added by the operator. Ordered, and
   * the order is what `cam<N>` numbering and the request's role list follow. */
  extra: string[];
}

/** checkpoint key → robot name → answers. The value is `StoredEntry` since
 * S3.8g and was a bare `CameraRoleMap` before it; `readEntry` reads both,
 * because a browser that used the panel last week has the old shape. */
type Stored = Record<string, Record<string, StoredEntry | CameraRoleMap>>;

/** The role names this stack will carry end to end — the same rule as the
 * backend's `utils.system.is_valid_image_role`, and it has to be checked on
 * READ because localStorage is written by other versions of this app. A role
 * becomes a policy feature key, a Portal video track name and a
 * `--robot.cameras` key, and anything else breaks one of those silently. */
const ROLE_RE = /^[a-z][a-z0-9_]{0,31}$/;

/** How many extra roles one launch may add — `utils.system.MAX_EXTRA_IMAGE_ROLES`.
 * A LATENCY ceiling, not a model limit: each view is another ~196 image tokens
 * through the policy's prefill. */
export const MAX_EXTRA_CAMERA_ROLES = 2;

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

const EMPTY_ENTRY: StoredEntry = { roles: {}, extra: [] };

function readEntry(
  checkpointKey: string | null,
  robotName: string | null,
): StoredEntry {
  if (!checkpointKey || !robotName) return EMPTY_ENTRY;
  const forCheckpoint = readAll()[checkpointKey];
  const stored = isPlainObject(forCheckpoint) ? forCheckpoint[robotName] : null;
  if (!isPlainObject(stored)) return EMPTY_ENTRY;
  // Two shapes live here. The S3.8g one nests the picks under `roles`; before
  // it, the entry WAS the map. Discriminated on the value's type rather than on
  // key presence, because "roles" is itself a legal role name — in the old
  // shape its value would be a camera-name string, never an object.
  const legacy = !isPlainObject(stored.roles);
  const rawRoles = legacy ? stored : stored.roles;
  // Values are camera NAMES; anything else came from a different writer.
  const roles: CameraRoleMap = {};
  for (const [role, camera] of Object.entries(rawRoles)) {
    if (typeof camera === "string" && camera) roles[role] = camera;
  }
  const rawExtra: unknown = legacy ? [] : stored.extra;
  const extra: string[] = Array.isArray(rawExtra)
    ? rawExtra.filter(
        (r): r is string => typeof r === "string" && ROLE_RE.test(r),
      )
    : [];
  // De-duplicated and capped on READ as well as on write: the cap is a latency
  // budget the backend enforces too, and a store written by another version
  // (or edited by hand) must not be able to push a launch past it.
  return { roles, extra: [...new Set(extra)].slice(0, MAX_EXTRA_CAMERA_ROLES) };
}

function writeEntry(
  checkpointKey: string,
  robotName: string,
  entry: StoredEntry,
): void {
  try {
    const all = readAll();
    const forCheckpoint = isPlainObject(all[checkpointKey])
      ? { ...all[checkpointKey] }
      : {};
    // Delete rather than store an empty entry, so clearing every pick leaves
    // the store the size it was before the operator ever opened the panel.
    const empty =
      Object.keys(entry.roles).length === 0 && entry.extra.length === 0;
    if (!empty) forCheckpoint[robotName] = entry;
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
  /** Roles the CHECKPOINT does not declare, added by the operator (S3.8g), in
   * the order they were added. Empty for every checkpoint nobody has added one
   * to, which is every checkpoint by default. */
  extraRoles: string[];
  /** Append the next free `cam<N>`. A no-op at the cap, or when the name it
   * would mint is already taken — `taken` is the roles the CHECKPOINT declares,
   * which the caller has and this hook does not. */
  addExtraRole: (taken: string[]) => void;
  /** Drop one extra role, and any binding made for it: the binding was only
   * ever an answer to a question this removes. */
  removeExtraRole: (role: string) => void;
}

/** The name an "add a role" press would mint: `cam<N>` for the lowest N that
 * collides with nothing. Exported for the test, and for the label. */
export function nextExtraRoleName(taken: Iterable<string>): string {
  const used = new Set(taken);
  // Bounded rather than `while (true)`: with the cap at 2 this can never run
  // far, and an unbounded loop over a set someone else fills is not worth the
  // one line it saves.
  for (let n = 0; n < 64; n += 1) {
    const name = `cam${n}`;
    if (!used.has(name)) return name;
  }
  return `cam${Date.now()}`;
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
  const [stored, setStored] = useState<StoredEntry>(() =>
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
    for (const [role, camera] of Object.entries(stored.roles)) {
      if (known.has(camera)) out[role] = camera;
    }
    return out;
  }, [stored, namesKey]);

  // One writer for all three mutations, so persistence cannot be wired for one
  // of them and forgotten for another.
  const commit = useCallback(
    (next: StoredEntry) => {
      setStored(next);
      if (checkpointKey && robotName)
        writeEntry(checkpointKey, robotName, next);
    },
    [checkpointKey, robotName],
  );

  const setRole = useCallback(
    (role: string, cameraName: string | null) => {
      const nextRoles = { ...stored.roles };
      if (cameraName) nextRoles[role] = cameraName;
      else delete nextRoles[role];
      commit({ ...stored, roles: nextRoles });
    },
    [stored, commit],
  );

  const addExtraRole = useCallback(
    (taken: string[]) => {
      if (stored.extra.length >= MAX_EXTRA_CAMERA_ROLES) return;
      const name = nextExtraRoleName([...taken, ...stored.extra]);
      if (stored.extra.includes(name)) return;
      commit({ ...stored, extra: [...stored.extra, name] });
    },
    [stored, commit],
  );

  const removeExtraRole = useCallback(
    (role: string) => {
      // The binding goes with it: it was an answer to "which camera plays this
      // role?", and the role is what is being removed.
      const nextRoles = { ...stored.roles };
      delete nextRoles[role];
      commit({
        roles: nextRoles,
        extra: stored.extra.filter((r) => r !== role),
      });
    },
    [stored, commit],
  );

  return {
    roles,
    setRole,
    extraRoles: stored.extra,
    addExtraRole,
    removeExtraRole,
  };
}
