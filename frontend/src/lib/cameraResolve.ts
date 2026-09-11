import type { AvailableCamera } from "@/hooks/useAvailableCameras";
import type { CameraConfig } from "@/components/recording/CameraConfiguration";

// A stored camera_index is only a position in cv2's device list — it silently
// rebinds to a different physical camera when devices come and go (a robot cam
// unplugging can leave its index pointing at the laptop's built-in camera).
// These helpers re-anchor a configured camera to its unique_id against a fresh
// /available-cameras enumeration before the index is trusted.
//
// unique_id is the best anchor available, NOT a stable one: on macOS it is the
// USB locationID plus a per-model constant, so it tracks (model, port), not the
// unit — see the note on CameraConfig.unique_id. A camera moved to another port
// reads as absent here, which is the safe direction (we refuse to trust an
// index we can't confirm) but is not the same as "this device is unplugged".
//
// Verification is only possible when the record stores a unique_id AND the
// enumeration reports uniqueIds (macOS). Otherwise fall back to legacy
// trust-the-stored-index behavior rather than falsely flagging disconnects.

const canVerify = (cam: CameraConfig, available: AvailableCamera[]) =>
  Boolean(cam.unique_id) && available.some((m) => m.uniqueId);

/**
 * Is `candidate` (a row of a live enumeration) the camera `saved` (a camera
 * already on a record) refers to?
 *
 * Identity, never position. `camera_index` is a POSITION in an enumeration the
 * backend re-sorts whenever the device set changes, so a saved index that has
 * gone stale matches whichever camera occupies that slot now — a different
 * physical device. That aliasing is what rejected a camera a record didn't hold
 * ("Camera Already Added" with one camera configured) and greyed out the row
 * the user actually wanted.
 *
 * Matching an id CONFIRMS; only a unique_id pair may DENY. That asymmetry is
 * the whole design. A unique_id mismatch is the one signal strong enough to
 * rule a camera out and stop the stale index having a say. It is strong, not
 * airtight — a camera moved to another port also mismatches (see
 * CameraConfig.unique_id), so the verdict there is "treat as different", which
 * leaves the stale row on the record for the user to delete rather than
 * stranding them with a record they cannot re-bind.
 *
 * Every other signal is one-directional. A `device_id` equality really is the
 * same device, but a mismatch proves nothing: the browser handle is per-origin
 * and rotates (site data cleared, a different origin), and its pairing to an
 * index is a COIN FLIP for twin identically-named cameras. Letting a mismatch
 * there decide would hand back the opposite bug — the same camera added twice
 * under two names.
 *
 * No platform check, on purpose. macOS supplies an AVFoundation uniqueID and
 * settles on the first rule; off Darwin camera_identity returns None, so no
 * unique_id pair exists, nothing can deny, and the index decides exactly as it
 * always has. Note `device_id` is NOT mac-only — the browser enumeration fills
 * it everywhere — which is why it must never be what rules a camera out.
 */
export function isSameCamera(
  saved: Pick<CameraConfig, "camera_index" | "device_id" | "unique_id">,
  candidate: Pick<AvailableCamera, "index" | "deviceId" | "uniqueId">,
): boolean {
  if (saved.unique_id && candidate.uniqueId)
    return saved.unique_id === candidate.uniqueId;
  // Confirms only — never falls through to a "not a duplicate" verdict.
  if (saved.device_id && saved.device_id === candidate.deviceId) return true;
  return saved.camera_index === candidate.index;
}

/** True when `cam` is verifiably attached, or when we can't verify. */
export function isCameraConnected(
  cam: CameraConfig,
  available: AvailableCamera[],
): boolean {
  if (!canVerify(cam, available)) return true;
  return available.some((m) => m.uniqueId === cam.unique_id);
}

/** The cv2 index to open for `cam` right now: the enumerated index of its
 * unique_id when verifiable (stored indices go stale on replug), undefined
 * when the camera is verifiably disconnected, else the stored index. */
export function resolveCameraIndex(
  cam: CameraConfig,
  available: AvailableCamera[],
): number | undefined {
  if (!canVerify(cam, available)) return cam.camera_index;
  return available.find((m) => m.uniqueId === cam.unique_id)?.index;
}
