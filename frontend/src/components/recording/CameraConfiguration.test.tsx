import React, { useEffect, useState } from "react";
import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import CameraConfiguration, { CameraConfig } from "./CameraConfiguration";
import type { AvailableCamera } from "@/hooks/useAvailableCameras";

// The live enumeration this machine actually reports: two identically-named
// USB cameras, ordered by AVFoundation uniqueID.
//   index 0 -> 0x11300002c7f4a60   ("top" in the saved records)
//   index 1 -> 0x1322002c7f4a60    ("front")
const MAC_CAMERAS: AvailableCamera[] = [
  {
    index: 0,
    name: "KD-USB Cameras",
    deviceId: "browser-device-id-for-top",
    available: true,
    uniqueId: "0x11300002c7f4a60",
  },
  {
    index: 1,
    name: "KD-USB Cameras",
    deviceId: "browser-device-id-for-front",
    available: true,
    uniqueId: "0x1322002c7f4a60",
  },
];

// Off macOS the backend reports no uniqueID at all (camera_identity returns
// None when platform.system() != "Darwin"), so the frontend has nothing but
// the index to go on. Same two cameras, stripped of the mac-only identity.
const LINUX_CAMERAS: AvailableCamera[] = MAC_CAMERAS.map(
  ({ uniqueId: _uniqueId, ...cam }) => cam,
);

// One camera on the record — "front" — carrying a camera_index that is STALE:
// it was written when a third camera was plugged in and front sat at index 0.
const SAVED_FRONT: CameraConfig = {
  id: "camera_repro_0001",
  name: "front",
  type: "opencv",
  camera_index: 0,
  device_id: "stale-browser-device-id",
  unique_id: "0x1322002c7f4a60",
  width: 640,
  height: 480,
  fps: 30,
};

const toastSpy = vi.fn();
let liveCameras: AvailableCamera[] = MAC_CAMERAS;

vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({ toast: toastSpy }),
}));

vi.mock("@/hooks/useAvailableCameras", () => ({
  useAvailableCameras: () => ({
    cameras: liveCameras,
    isLoading: false,
    refresh: vi.fn(),
  }),
}));

// The preview tile opens a real MJPEG stream against the API; irrelevant here.
vi.mock("@/components/BackendCameraStream", () => ({
  default: () => <div data-testid="camera-stream" />,
}));

beforeAll(() => {
  // Radix Select drives its popup through Pointer Events APIs jsdom lacks.
  Element.prototype.hasPointerCapture = vi.fn(() => false);
  Element.prototype.setPointerCapture = vi.fn();
  Element.prototype.releasePointerCapture = vi.fn();
  Element.prototype.scrollIntoView = vi.fn();
});

beforeEach(() => {
  toastSpy.mockClear();
  liveCameras = MAC_CAMERAS;
});

/**
 * Mirrors RobotConfigDialog: the saved record is fetched, so `cameras` lands
 * one tick AFTER the camera enumeration has already settled. `seedFirst`
 * flips that order — before the fix, the order alone decided the outcome.
 */
const Harness: React.FC<{
  seedFirst?: boolean;
  saved?: CameraConfig;
  onChange?: (cameras: CameraConfig[]) => void;
  /** Drives the record explicitly, to land it mid-interaction on rerender. */
  seed?: CameraConfig[];
}> = ({ seedFirst = false, saved = SAVED_FRONT, onChange, seed }) => {
  const [cameras, setCameras] = useState<CameraConfig[]>(
    seed ?? (seedFirst ? [saved] : []),
  );
  useEffect(() => {
    if (seed) setCameras(seed);
    else if (!seedFirst) setCameras([saved]);
  }, [seedFirst, saved, seed]);
  return (
    <CameraConfiguration
      cameras={cameras}
      onCamerasChange={(next) => {
        onChange?.(next);
        setCameras(next);
      }}
    />
  );
};

const openDropdown = async () => {
  const trigger = screen.getAllByRole("combobox")[0];
  fireEvent.keyDown(trigger, { key: "Enter" });
  await waitFor(() => expect(screen.getAllByRole("option")).toHaveLength(2));
  return screen.getAllByRole("option");
};

/** Picks a name from the preset select that appears once a camera is chosen. */
const chooseName = async (name: string) => {
  const triggers = await waitFor(() => {
    const found = screen.getAllByRole("combobox");
    expect(found.length).toBeGreaterThan(1);
    return found;
  });
  fireEvent.keyDown(triggers[triggers.length - 1], { key: "Enter" });
  fireEvent.click(await screen.findByRole("option", { name }));
};

/** Selects the dropdown row at `optionIndex`, names it, and clicks Add. */
const addCameraAt = async (optionIndex: number, name: string) => {
  const options = await openDropdown();
  fireEvent.click(options[optionIndex]);
  await chooseName(name);
  fireEvent.click(screen.getByRole("button", { name: /add camera/i }));
};

describe("CameraConfiguration — 'Camera Already Added' false positive", () => {
  // The regression. A stale camera_index used to alias a DIFFERENT physical
  // camera: index 0 was greyed out (it matched front's stale index) while the
  // only selectable row was front itself, which Add then refused by unique_id.
  it("adds the second camera when the record's saved index is stale", async () => {
    render(<Harness />);
    await screen.findByText("front");

    const options = await openDropdown();
    // Identity decides: index 0 is a different camera and stays available;
    // index 1 IS front and is the row correctly greyed out.
    expect(options[0]).not.toHaveAttribute("aria-disabled", "true");
    expect(options[1]).toHaveAttribute("aria-disabled", "true");

    fireEvent.click(options[0]);
    await chooseName("top");
    fireEvent.click(screen.getByRole("button", { name: /add camera/i }));

    expect(toastSpy).toHaveBeenCalledTimes(1);
    expect(toastSpy.mock.calls[0][0]).toMatchObject({ title: "Camera Added" });
    await screen.findByText("top");
  });

  it("behaves identically when the record arrives before the enumeration", async () => {
    render(<Harness seedFirst />);
    await screen.findByText("front");

    await addCameraAt(0, "top");

    expect(toastSpy).toHaveBeenCalledTimes(1);
    expect(toastSpy.mock.calls[0][0]).toMatchObject({ title: "Camera Added" });
    await screen.findByText("top");
  });

  it("re-anchors a stale index onto the camera's unique_id", async () => {
    const seen: CameraConfig[][] = [];
    render(<Harness onChange={(next) => seen.push(next)} />);
    await screen.findByText("front");

    // front is live at index 1; the record said 0. Without the re-anchor the
    // recorder would open index 0 — a different physical camera.
    await waitFor(() => {
      const latest = seen[seen.length - 1];
      expect(latest?.[0]).toMatchObject({
        name: "front",
        camera_index: 1,
        unique_id: "0x1322002c7f4a60",
      });
    });
  });

  // Both weaker ids decay (a browser deviceId rotates; unique_id tracks the USB
  // port), and a record can only heal while something still matches. If the
  // match refreshed the index alone, the stale id would poison every later
  // comparison for the life of the record.
  it("heals the stale ids, not just the index, on a confirmed match", async () => {
    const seen: CameraConfig[][] = [];
    render(<Harness onChange={(next) => seen.push(next)} />);
    await screen.findByText("front");

    await waitFor(() => {
      expect(seen[seen.length - 1]?.[0]).toMatchObject({
        camera_index: 1,
        unique_id: "0x1322002c7f4a60",
        // was "stale-browser-device-id" on the record
        device_id: "browser-device-id-for-front",
      });
    });
  });

  // The effect writes through onCamerasChange, which feeds back into `cameras`,
  // which is one of its own deps. The `changed` guard is what stops that being
  // a render loop — this pins that it settles rather than merely converging
  // fast enough not to be noticed.
  it("settles: the re-anchor writes back exactly once", async () => {
    const seen: CameraConfig[][] = [];
    render(<Harness onChange={(next) => seen.push(next)} />);
    await screen.findByText("front");
    await waitFor(() => expect(seen.length).toBe(1));

    // Give any further passes a chance to fire before declaring it settled.
    await new Promise((r) => setTimeout(r, 50));
    expect(seen).toHaveLength(1);
  });

  // The unique_id on disk is not forever: these cameras re-enumerate with a
  // different AVFoundation uniqueID across sessions (0x1323… became 0x1322…),
  // so re-anchoring can legitimately find no match and the index stays stale.
  // Identity must still win over that stale position.
  it("does not alias a stale index onto a camera whose unique_id is gone", async () => {
    const vanished = { ...SAVED_FRONT, unique_id: "0x1323002c7f4a60" };
    render(<Harness saved={vanished} />);
    await screen.findByText("front");

    // Nothing live matches, so camera_index is still 0 — but index 0 is a
    // different camera by unique_id, so it must remain addable.
    await addCameraAt(0, "top");

    expect(toastSpy).toHaveBeenCalledTimes(1);
    expect(toastSpy.mock.calls[0][0]).toMatchObject({ title: "Camera Added" });
  });

  it("still greys out the camera already on the record", async () => {
    render(<Harness />);
    await screen.findByText("front");

    // Re-adding front (live index 1, already on the record) is a real
    // duplicate, and the picker refuses to offer it.
    const options = await openDropdown();
    expect(options[1]).toHaveAttribute("aria-disabled", "true");
  });

  // The picker and Add now share one predicate, so Add's guard is unreachable
  // through a greyed row — but it is still load-bearing for the race that made
  // the two disagree in the first place: the user picks while the record is
  // still loading, and the record arrives holding that very camera.
  it("refuses on Add when the record loads after the pick", async () => {
    const { rerender } = render(<Harness seed={[]} />);

    const options = await openDropdown();
    expect(options[1]).not.toHaveAttribute("aria-disabled", "true");
    fireEvent.click(options[1]);
    await chooseName("top");

    // The fetch lands now, and it already contains the camera just picked.
    rerender(<Harness seed={[SAVED_FRONT]} />);
    await screen.findByText("front");

    fireEvent.click(screen.getByRole("button", { name: /add camera/i }));

    expect(toastSpy).toHaveBeenCalledTimes(1);
    expect(toastSpy.mock.calls[0][0]).toMatchObject({
      title: "Camera Already Added",
      variant: "destructive",
    });
  });
});

describe("CameraConfiguration — platforms without a stable device id", () => {
  // Linux/Windows report no uniqueID, so the index is the only identity there.
  // The tiers must degrade to exactly the old index behaviour, with no
  // platform check anywhere in the component.
  beforeEach(() => {
    liveCameras = LINUX_CAMERAS;
  });

  // A rotated device_id must never RULE A CAMERA OUT. The browser handle is
  // populated on every platform and rotates when site data is cleared, so if a
  // mismatch there decided "different camera", the same physical device could
  // be added twice under two names — the mirror image of the bug being fixed.
  it("still refuses a duplicate when the saved device_id has rotated", async () => {
    const rotated: CameraConfig = {
      ...SAVED_FRONT,
      unique_id: undefined,
      device_id: "device-id-from-before-site-data-was-cleared",
      camera_index: 0,
    };
    render(<Harness saved={rotated} />);
    await screen.findByText("front");

    // Nothing identifies it any more, so the index is the only signal left —
    // and it must still be consulted.
    const options = await openDropdown();
    expect(options[0]).toHaveAttribute("aria-disabled", "true");
  });

  it("falls back to the index when neither side has a unique id", async () => {
    const noIdentity: CameraConfig = {
      ...SAVED_FRONT,
      unique_id: undefined,
      device_id: "browser-device-id-for-front",
      camera_index: 1,
    };
    render(<Harness saved={noIdentity} />);
    await screen.findByText("front");

    const options = await openDropdown();
    expect(options[0]).not.toHaveAttribute("aria-disabled", "true");
    expect(options[1]).toHaveAttribute("aria-disabled", "true");

    fireEvent.click(options[0]);
    await chooseName("top");
    fireEvent.click(screen.getByRole("button", { name: /add camera/i }));

    expect(toastSpy).toHaveBeenCalledTimes(1);
    expect(toastSpy.mock.calls[0][0]).toMatchObject({ title: "Camera Added" });
  });

  it("refuses a genuine duplicate by index alone", async () => {
    const noIdentity: CameraConfig = {
      ...SAVED_FRONT,
      unique_id: undefined,
      device_id: "browser-device-id-for-top",
      camera_index: 0,
    };
    render(<Harness saved={noIdentity} />);
    await screen.findByText("front");

    const options = await openDropdown();
    expect(options[0]).toHaveAttribute("aria-disabled", "true");
  });
});
