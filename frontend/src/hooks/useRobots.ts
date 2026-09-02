import { useCallback, useEffect, useMemo, useSyncExternalStore } from "react";
import { useLocation } from "react-router-dom";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import type { CameraConfig } from "@/components/recording/CameraConfiguration";

export type RobotMode = "single" | "bimanual";

// ArmType and its capability predicates moved to lib/armTypes.ts so they stay
// pure and independently testable (this module pulls in the API context on
// import). Re-exported here because ArmType's importers already use this path.
import type { ArmType } from "@/lib/armTypes";
export { isCanArmType, jointsPerArm } from "@/lib/armTypes";
export type { ArmType } from "@/lib/armTypes";

export interface RobotRecord {
  name: string;
  mode: RobotMode;
  arm_type: ArmType;
  // Primary pair (single mode), or the LEFT arm pair (bimanual mode).
  leader_port: string;
  follower_port: string;
  leader_config: string;
  follower_config: string;
  // Right arm pair — populated only in bimanual mode.
  right_leader_port: string;
  right_follower_port: string;
  right_leader_config: string;
  right_follower_config: string;
  cameras: CameraConfig[];
  // Auto-calibration drive torque as a percentage of full power (10-100,
  // default 38 = the vendored script's stock 380). Sessions (teleop/record/
  // skill runs) use stock LeRobot torque and ignore this value.
  motor_power: number;
  is_clean: boolean;
  // Follower-side readiness only (ports + calibrations for the follower arm(s)).
  // Follower-only activities (inference, replay) gate on this instead of
  // is_clean so a missing LEADER setup — which they never touch — can't block
  // them. Mirrors the backend's is_robot_record_clean(record, arms="follower").
  follower_ready: boolean;
}

// The setup-gap diagnosis moved to lib/robotSetupGap.ts so it stays a pure,
// independently testable function (this module pulls in the API context and
// localStorage on import). Re-exported here because four call sites already
// import it from useRobots.
export {
  robotSetupGap,
  robotSetupGaps,
  formatRobotSetupGap,
} from "@/lib/robotSetupGap";
export type { ArmKey, RobotSetupGaps } from "@/lib/robotSetupGap";

const SELECTED_KEY = "makermodslab.selectedRobot";

const readSelected = (): string | null => {
  try {
    const raw = localStorage.getItem(SELECTED_KEY);
    return raw && typeof raw === "string" ? raw : null;
  } catch {
    return null;
  }
};

const writeSelected = (name: string | null) => {
  try {
    if (name) localStorage.setItem(SELECTED_KEY, name);
    else localStorage.removeItem(SELECTED_KEY);
  } catch {
    // Storage may be unavailable (private mode, quota). Failures here are non-fatal.
  }
};

// Module-level store shared by every useRobots() instance. The Landing card,
// JobsSection, and Training page mount simultaneously, so per-instance
// useState copies drift: selecting a robot on Landing left the inference
// modal (mounted under JobsSection) holding the stale previous selection.
// One store, one truth — instances subscribe via useSyncExternalStore.
interface RobotsState {
  records: Record<string, RobotRecord>;
  selectedName: string | null;
  isLoading: boolean;
}

let state: RobotsState = {
  records: {},
  selectedName: readSelected(),
  isLoading: false,
};
const listeners = new Set<() => void>();

const setState = (patch: Partial<RobotsState>) => {
  state = { ...state, ...patch };
  listeners.forEach((l) => l());
};

const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
};

const getSnapshot = (): RobotsState => state;

const setSelectedShared = (name: string | null) => {
  writeSelected(name);
  setState({ selectedName: name });
};

const patchRecords = (
  updater: (prev: Record<string, RobotRecord>) => Record<string, RobotRecord>
) => {
  setState({ records: updater(state.records) });
};

// Several instances can fetch concurrently (each mount refreshes); isLoading
// is true while ANY fetch is in flight, not just the last one to finish.
let pendingFetches = 0;

export const useRobots = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const location = useLocation();

  const { records, selectedName, isLoading } = useSyncExternalStore(
    subscribe,
    getSnapshot
  );

  // Re-fetch the shared store from the backend. Exposed as `refresh` so
  // surfaces that mutate records without a route change (e.g. closing the
  // Robot settings dialog, which may have saved ports/cameras/calibrations)
  // can update every subscriber. Writes go to the module-level store, so a
  // late response after unmount is harmless.
  const refresh = useCallback(async () => {
    pendingFetches += 1;
    setState({ isLoading: true });
    try {
      const res = await fetchWithHeaders(`${baseUrl}/api/v1/robots`);
      const data = await res.json();
      const next: Record<string, RobotRecord> = {};
      for (const r of data.robots ?? []) next[r.name] = r;
      setState({ records: next });
      // Drop the selection if the underlying record vanished (deleted from another tab)
      if (state.selectedName && !(state.selectedName in next)) {
        setSelectedShared(null);
      }
    } catch (e) {
      console.error("Failed to fetch robots:", e);
    } finally {
      pendingFetches -= 1;
      setState({ isLoading: pendingFetches > 0 });
    }
  }, [baseUrl, fetchWithHeaders]);

  // Re-fetch records when location changes (fires on initial mount and on
  // back-navigation to a page that mounts a useRobots consumer).
  useEffect(() => {
    refresh();
  }, [refresh, location.key]);

  const selectRobot = useCallback((name: string) => {
    setSelectedShared(name);
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedShared(null);
  }, []);

  const createRobot = useCallback(
    async (
      rawName: string,
      mode: RobotMode = "single",
      armType: ArmType = "so101"
    ): Promise<boolean> => {
      const name = rawName.trim();
      if (!name) {
        toast({ title: "Missing name", description: "Robot name cannot be empty.", variant: "destructive" });
        return false;
      }
      if (/[/\\]|\.\./.test(name)) {
        toast({ title: "Invalid name", description: "Robot names cannot contain '/', '\\', or '..'", variant: "destructive" });
        return false;
      }
      try {
        const res = await fetchWithHeaders(`${baseUrl}/api/v1/robots/${encodeURIComponent(name)}?create=true`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode, arm_type: armType }),
        });
        if (res.status === 409) {
          toast({
            title: "Already exists",
            description: `A robot named "${name}" already exists. Pick it from the dropdown or choose a different name.`,
            variant: "destructive",
          });
          return false;
        }
        if (!res.ok) {
          const text = await res.text();
          toast({ title: "Failed to create", description: text, variant: "destructive" });
          return false;
        }
        const data = await res.json();
        if (data.robot) {
          patchRecords((prev) => ({ ...prev, [name]: data.robot }));
          setSelectedShared(name);
        }
        return true;
      } catch (e) {
        toast({ title: "Network error", description: String(e), variant: "destructive" });
        return false;
      }
    },
    [baseUrl, fetchWithHeaders, toast]
  );

  const deleteRobot = useCallback(
    async (name: string): Promise<boolean> => {
      try {
        const res = await fetchWithHeaders(`${baseUrl}/api/v1/robots/${encodeURIComponent(name)}`, {
          method: "DELETE",
        });
        // 404 = the record is already gone (deleted elsewhere, or removed on
        // disk out-of-band). The user's intent is fulfilled either way — drop
        // it from the local list instead of showing a scary failure.
        if (!res.ok && res.status !== 404) {
          const text = await res.text();
          toast({ title: "Delete failed", description: text, variant: "destructive" });
          return false;
        }
        patchRecords((prev) => {
          const { [name]: _omit, ...rest } = prev;
          return rest;
        });
        if (state.selectedName === name) setSelectedShared(null);
        toast({
          title: "Robot deleted",
          description:
            res.status === 404
              ? `"${name}" was already removed — updated the list.`
              : `Removed "${name}". Calibration files are kept in the library.`,
        });
        return true;
      } catch (e) {
        toast({ title: "Delete failed", description: String(e), variant: "destructive" });
        return false;
      }
    },
    [baseUrl, fetchWithHeaders, toast]
  );

  const renameRobot = useCallback(
    async (oldName: string, rawNew: string): Promise<boolean> => {
      const newName = rawNew.trim();
      if (!newName) {
        toast({ title: "Missing name", description: "Robot name cannot be empty.", variant: "destructive" });
        return false;
      }
      if (newName === oldName) return true; // no-op
      if (/[/\\]|\.\./.test(newName)) {
        toast({ title: "Invalid name", description: "Robot names cannot contain '/', '\\', or '..'", variant: "destructive" });
        return false;
      }
      try {
        const res = await fetchWithHeaders(`${baseUrl}/api/v1/robots/${encodeURIComponent(oldName)}/rename`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ new_name: newName }),
        });
        if (res.status === 409) {
          toast({
            title: "Already exists",
            description: `A robot named "${newName}" already exists. Choose a different name.`,
            variant: "destructive",
          });
          return false;
        }
        if (!res.ok) {
          const text = await res.text();
          toast({ title: "Failed to rename", description: text, variant: "destructive" });
          return false;
        }
        const data = await res.json();
        // Swap the key oldName → newName in the local map, preserving order roughly.
        patchRecords((prev) => {
          const { [oldName]: _omit, ...rest } = prev;
          return data.robot ? { ...rest, [newName]: data.robot } : rest;
        });
        if (state.selectedName === oldName) setSelectedShared(newName);
        return true;
      } catch (e) {
        toast({ title: "Network error", description: String(e), variant: "destructive" });
        return false;
      }
    },
    [baseUrl, fetchWithHeaders, toast]
  );

  const selectedRecord = useMemo(
    () => (selectedName ? records[selectedName] ?? null : null),
    [selectedName, records]
  );

  const availableNames = useMemo(
    () => Object.keys(records).sort(),
    [records]
  );

  return {
    records,
    selectedName,
    selectedRecord,
    availableNames,
    isLoading,
    refresh,
    selectRobot,
    clearSelection,
    createRobot,
    renameRobot,
    deleteRobot,
  };
};
