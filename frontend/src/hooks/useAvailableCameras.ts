import { useCallback, useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";

export interface AvailableCamera {
  index: number;
  name: string;
  deviceId: string;
  available: boolean;
  /** Stable OS-level device identity (AVFoundation uniqueID on macOS).
   * Unlike the cv2 index, it survives replugs and identically-named devices. */
  uniqueId?: string;
}

const norm = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();

interface UseAvailableCamerasOptions {
  /** When false, do nothing. Use to gate on modal open. */
  enabled?: boolean;
  /** When false, skip browser device matching entirely (no getUserMedia
   * permission probe, deviceId stays ""). Previews that stream from the
   * backend by cv2 index don't need a browser match at all. */
  matchBrowser?: boolean;
}

/**
 * Enumerates cv2 camera indices from `/available-cameras` and merges each
 * with the matching browser deviceId (by AVFoundation localizedName) so
 * callers can render a preview alongside the bound dropdowns. Refreshes on
 * USB hotplug.
 */
export function useAvailableCameras({
  enabled = true,
  matchBrowser = true,
}: UseAvailableCamerasOptions = {}) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [cameras, setCameras] = useState<AvailableCamera[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  // Latest result, readable without making `refresh` depend on `cameras`
  // (which would re-fire the effect and its listener every enumeration).
  const camerasRef = useRef<AvailableCamera[]>([]);
  // Drop overlapping refreshes so a devicechange burst can't stack getUserMedia
  // probes / backend enumerations on top of each other.
  const refreshingRef = useRef(false);
  // Opening a camera stream is only needed ONCE, to unlock enumerateDevices()
  // labels. Re-opening it on every refresh — and re-entrantly from the
  // devicechange handler — is what made the preview flash and macOS reshuffle
  // camera indices (waking Continuity Camera / Desk View shifts the uniqueID
  // sort). After the first grant, labels persist, so we never probe again.
  const permissionProbedRef = useRef(false);

  const refresh = useCallback(async (): Promise<AvailableCamera[]> => {
    if (refreshingRef.current) return camerasRef.current;
    refreshingRef.current = true;
    setIsLoading(true);
    try {
      // navigator.mediaDevices only exists in secure contexts (https or
      // localhost). Without it, skip browser matching — backend cv2 indices
      // still list and recording works, there are just no live previews.
      let browserDevices: { deviceId: string; label: string }[] = [];
      const md = navigator.mediaDevices;
      if (md && matchBrowser) {
        let devices = await md.enumerateDevices();
        const hasLabels = devices.some((d) => d.kind === "videoinput" && d.label);
        // Only open a real stream if labels are still hidden (permission not yet
        // granted) AND we've never probed. Never from a devicechange re-entry —
        // that feedback loop was the root of the flashing/index churn.
        if (!hasLabels && !permissionProbedRef.current) {
          permissionProbedRef.current = true;
          try {
            const probe = await md.getUserMedia({ video: true });
            probe.getTracks().forEach((t) => t.stop());
            devices = await md.enumerateDevices();
          } catch {
            // ignore — we'll still try to enumerate, just without labels
          }
        }
        browserDevices = devices
          .filter((d) => d.kind === "videoinput")
          .map((d) => ({ deviceId: d.deviceId, label: d.label }));
      }

      const r = await fetchWithHeaders(`${baseUrl}/api/v1/available-cameras`);
      if (!r.ok) {
        setCameras([]);
        return [];
      }
      const data = await r.json();
      const backendCams: {
        index: number;
        name?: string;
        available: boolean;
        unique_id?: string;
      }[] = data.cameras ?? [];

      // Browser's MediaDeviceInfo.label starts with AVFoundation's localizedName
      // but Chrome often appends "(vendorId:productId)". Match by exact, then
      // prefix, then either-contains.
      const used = new Set<string>();
      const merged: AvailableCamera[] = backendCams.map((cam) => {
        const label = cam.name || `Camera ${cam.index}`;
        const target = norm(label);
        const candidates = browserDevices.filter(
          (d) => !used.has(d.deviceId) && d.label
        );
        const match =
          candidates.find((d) => norm(d.label) === target) ||
          candidates.find((d) => norm(d.label).startsWith(target)) ||
          candidates.find(
            (d) => norm(d.label).includes(target) || target.includes(norm(d.label))
          );
        if (match) used.add(match.deviceId);
        return {
          index: cam.index,
          name: label,
          deviceId: match?.deviceId ?? "",
          available: cam.available,
          uniqueId: cam.unique_id,
        };
      });
      setCameras(merged);
      camerasRef.current = merged;
      return merged;
    } catch {
      setCameras([]);
      camerasRef.current = [];
      return [];
    } finally {
      setIsLoading(false);
      refreshingRef.current = false;
    }
  }, [baseUrl, fetchWithHeaders, matchBrowser]);

  useEffect(() => {
    if (!enabled) return;
    refresh();
    const md = navigator.mediaDevices;
    if (!md) return; // insecure context: no hotplug events, refresh() ran once
    // Debounce hotplug bursts (a single plug event can fire several
    // devicechanges, and any getUserMedia churn elsewhere adds more) into one
    // refresh instead of a storm that keeps re-opening cameras.
    let debounce: ReturnType<typeof setTimeout> | null = null;
    const handler = () => {
      if (debounce) clearTimeout(debounce);
      debounce = setTimeout(() => refresh(), 500);
    };
    md.addEventListener("devicechange", handler);
    return () => {
      if (debounce) clearTimeout(debounce);
      md.removeEventListener("devicechange", handler);
    };
  }, [enabled, refresh]);

  return { cameras, isLoading, refresh };
}
