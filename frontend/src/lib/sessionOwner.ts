// The per-tab owner identity for /api/v1/sessions leases.
//
// A session started with an `owner` gets a lease the owner must renew by
// heartbeat (useSessionHeartbeat), or the server safety-stops it — the
// server-side replacement for the retired browser unload guards. The owner is
// an opaque identity string, minted ONCE per tab and kept in sessionStorage so
// a reload of the same tab keeps renewing the session it started, while a
// different tab (its own sessionStorage) can never heartbeat someone else's
// lease by accident.
//
// Like the old SingleTabGuard's tab id, this only needs uniqueness, not
// unguessability — crypto.randomUUID is unavailable on plain-HTTP LAN hosts,
// so a Math.random-based id is deliberate. Stops are never owner-gated, so the
// identity grants nothing beyond "my heartbeats renew my lease".

const STORAGE_KEY = "makermodslab:session-owner";

/** Mint a fresh `ui:<random>` owner id. Well under the server's 128-char cap. */
export function mintOwnerId(random: () => number = Math.random): string {
  return `ui:${Date.now().toString(36)}-${random().toString(36).slice(2, 10)}`;
}

/** The subset of the Web Storage API this module needs (testable). */
export interface StringStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

/**
 * Read the tab's owner id from `store`, minting and persisting one on first
 * use. A missing or throwing store still returns a usable (ephemeral) id —
 * the lease machinery must work even where sessionStorage doesn't.
 */
export function readOrMintOwner(
  store: StringStore | null,
  mint: () => string = mintOwnerId
): string {
  try {
    const existing = store?.getItem(STORAGE_KEY);
    if (existing) return existing;
  } catch {
    /* storage unreadable — fall through to a fresh id */
  }
  const minted = mint();
  try {
    store?.setItem(STORAGE_KEY, minted);
  } catch {
    /* storage unwritable — the id lives only as long as this document */
  }
  return minted;
}

// Cache the resolved id so a throwing/unavailable sessionStorage still yields
// ONE stable identity for the lifetime of this document.
let cachedOwner: string | null = null;

/** This tab's session-owner identity (stable for the life of the tab). */
export function tabOwnerId(): string {
  if (cachedOwner) return cachedOwner;
  let store: StringStore | null = null;
  try {
    store = window.sessionStorage;
  } catch {
    /* sessionStorage itself can throw (privacy modes) — mint ephemeral */
  }
  cachedOwner = readOrMintOwner(store);
  return cachedOwner;
}
