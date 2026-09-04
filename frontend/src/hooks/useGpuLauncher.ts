import { useCallback, useEffect, useRef, useState } from "react";
import { useApi } from "@/contexts/ApiContext";
import { apiRequest } from "@/lib/apiClient";
import type { Fetcher } from "@/lib/apiClient";
import type { RemoteEngine } from "@/components/remote-inference/remoteRunConfig";

/**
 * The GPU half of a remote-inference run, which the Lab now launches itself
 * (`modal run makermodslab/drtc/modal_policy*.py`).
 *
 * A LAB-LEVEL RESOURCE, not a session: it holds no hardware, it has no lease,
 * and stopping it is not a safety action. That is why it lives on its own three
 * routes rather than as a field on the remote-run options — a 1-3 minute cold
 * start inside the session start would hold the robot-busy discriminant while
 * the arm sat completely free.
 *
 * It is also NOT what unblocks "Run it remotely". That gate stays the transport
 * probe's `operator_present`, which observes the ROOM; `state: "ready"` here is
 * derived from the container's own stdout and is a hint, not an authority.
 */

/** idle | starting | ready | failed | stopping. Backend enum values — matched
 * on, never displayed raw. */
export type GpuState = "idle" | "starting" | "ready" | "failed" | "stopping";

export interface GpuStatus {
  state: GpuState;
  /** The container's progress: tailscale_up | loading | warmup | connecting |
   * connected | claimed. Null before the first recognizable line. Backend enum
   * values. `connected` is what flips `state` to ready — `claimed` is a display
   * refinement, since the policy claims control in a background task and a
   * healthy run may never print it. */
  phase: string | null;
  /** Which wrapper is running. Backend identifiers ("sync" / "rtc"). */
  engine: string | null;
  policy_hub_id: string | null;
  /** The room the launcher pinned with --livekit-room. Data. */
  room: string | null;
  /** WHICH WORKSPACE IS PAYING, as launched: the Modal profile that went to
   * the child as MODAL_PROFILE, and the environment that went to `modal run`
   * as --env. Both null when the operator left the choice to the CLI's own
   * resolution — a different fact from an empty selection. Names, so DATA. */
  profile: string | null;
  environment: string | null;
  /** The Modal app this run created ("ap-…"), from the client's own "View run
   * at" url. Null while idle and until that line arrives. It is what `modal
   * app stop` takes — shown so an operator can stop a run by hand. DATA. */
  app_id: string | null;
  /** The transport tuple AS LAUNCHED, null while idle. The SERVER's record of
   * what the running GPU was started with: half of it is the Portal wire
   * schema (a disagreement drops every packet in silence) and `task` steers
   * the policy. The drift warning compares the form against these, which is
   * what makes it survive a page reload and cover a GPU another tab started.
   * `s_min` is echoed for both engines but only reaches the wire for rtc. */
  task: string | null;
  horizon: number | null;
  fps: number | null;
  video_codec: string | null;
  s_min: number | null;
  /** Survives an idle transition: after a failure this is the most useful
   * thing left. A path — data. */
  log_path: string | null;
  started_at: number | null;
  elapsed_s: number;
  /** Backend prose — rendered verbatim, never translated. */
  message: string | null;
  hint: string | null;
  /** The `gpu.*` code behind a FAILED state ("gpu.unauthenticated",
   * "gpu.launch_failed"); null in every other state. Backend data — matched
   * on and shown verbatim, never translated. Branch on this, never on the
   * prose beside it. */
  code: string | null;
  /** The most recent output line. DATA: a container's own log text. */
  last_line: string | null;
  /** Seconds until the idle auto-stop. Null unless `ready` AND no remote run is
   * live — a busy GPU is not idle, and a countdown that kept ticking through a
   * live run would be a lie. */
  idle_stop_in_s: number | null;
}

export interface GpuStartRequest {
  engine: RemoteEngine;
  policy_hub_id: string;
  task: string;
  horizon: number;
  fps: number;
  video_codec: "H264" | "MJPEG";
  s_min: number;
  /** WHICH WORKSPACE PAYS. Empty means the `modal` CLI resolves it itself,
   * which is what an API client that never sends them gets. The UI is
   * deliberately explicit instead: it sends whatever is selected, always. */
  profile: string;
  environment: string;
}

/** One row of `modal profile list --json`. All three are DATA — a profile name
 * and a workspace name are identifiers, never prose. */
export interface ModalProfile {
  name: string;
  workspace: string;
  /** The machine-wide active profile. The picker's DEFAULT, never a
   * constraint: the Lab chooses per launch and never activates one. */
  active: boolean;
}

/** One row of `modal environment list --json`, for one profile's workspace. */
export interface ModalEnvironment {
  name: string;
  active: boolean;
}

export interface GpuTargets {
  profiles: ModalProfile[];
  /** The environments of `profile` below — not of every profile. */
  environments: ModalEnvironment[];
  /** Which profile the environments belong to. Null when they could not be
   * listed, which is what stops a stale list reading as current. */
  profile: string | null;
  /** Why the listing failed, or null. A failed listing is NOT a failed launch:
   * with no selection the CLI still resolves the target, so this costs the two
   * pickers and nothing else. `message` is backend prose — shown verbatim. */
  error: { code: string; message: string } | null;
}

interface GpuLaunchResult {
  started: boolean;
  message: string;
  gpu: GpuStatus;
}

export async function getGpuStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<GpuStatus> {
  return apiRequest<GpuStatus>(
    baseUrl,
    fetcher,
    "/api/v1/remote-inference/gpu",
    { signal, action: "Read GPU status" },
  );
}

export async function startGpu(
  baseUrl: string,
  fetcher: Fetcher,
  body: GpuStartRequest,
): Promise<GpuLaunchResult> {
  return apiRequest<GpuLaunchResult>(
    baseUrl,
    fetcher,
    "/api/v1/remote-inference/gpu/start",
    { method: "POST", body, action: "Start the GPU" },
  );
}

export async function getGpuTargets(
  baseUrl: string,
  fetcher: Fetcher,
  profile: string,
  signal?: AbortSignal,
): Promise<GpuTargets> {
  const query = profile ? `?profile=${encodeURIComponent(profile)}` : "";
  return apiRequest<GpuTargets>(
    baseUrl,
    fetcher,
    `/api/v1/remote-inference/gpu/targets${query}`,
    { signal, action: "List this machine's Modal targets" },
  );
}

export async function stopGpu(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<GpuStatus> {
  return apiRequest<GpuStatus>(
    baseUrl,
    fetcher,
    "/api/v1/remote-inference/gpu/stop",
    { method: "POST", action: "Stop the GPU" },
  );
}

/** A cold start moves through five phases in 1-3 minutes; 2s keeps the phase
 * strip honest without hammering a route that only reads memory. */
export const GPU_ACTIVE_POLL_MS = 2000;
/** Everything else. A ready GPU changes only when the idle auto-stop fires. */
export const GPU_IDLE_POLL_MS = 10000;

export interface UseGpuLauncher {
  status: GpuStatus | null;
  /** True while a start/stop request is in flight (distinct from the backend's
   * own `starting`/`stopping`, which outlive the request). */
  pending: boolean;
  /** The thrown error's own text, or null. Backend prose — shown as raised. */
  error: string | null;
  start: (body: GpuStartRequest) => Promise<void>;
  stop: () => Promise<void>;
  refresh: () => void;
  /** The request THIS tab last launched the GPU with, kept while that GPU is
   * up so the panel can warn when the form drifts away from it — the
   * transport knobs are half of a fingerprint the running server holds, and a
   * mismatch is a run that receives nothing, not an error. Null after a
   * reload or for a GPU started elsewhere, and cleared by a stop — which is
   * exactly why it is now only the FALLBACK: the status echoes the tuple the
   * server itself launched with, and that record survives both. */
  launched: GpuStartRequest | null;
}

/**
 * @param enabled poll while true. False leaves the last answer in place rather
 * than blanking it, so collapsing the form does not throw away what the
 * operator just read.
 */
export function useGpuLauncher(enabled: boolean): UseGpuLauncher {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [status, setStatus] = useState<GpuStatus | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [launched, setLaunched] = useState<GpuStartRequest | null>(null);
  const [nonce, setNonce] = useState(0);
  // Ref rather than state: the poll effect reads it to decide whether to keep
  // ticking, and putting it in the dep array would re-arm the interval on
  // every answer.
  const settled = useRef(false);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  const state = status?.state;
  // A settled failure (or a plain idle) changes only when the operator acts, so
  // polling it forever is pure noise. Any action below clears the flag.
  const quiet = state === "failed" || (state === "idle" && settled.current);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const next = await getGpuStatus(baseUrl, fetchWithHeaders);
        if (cancelled) return;
        setStatus(next);
        if (next.state === "failed" || next.state === "idle")
          settled.current = true;
      } catch {
        // Transient; the next tick retries. A failed poll must never blank a
        // panel the operator is reading mid-launch.
      }
    };
    void tick();
    if (quiet)
      return () => {
        cancelled = true;
      };
    const id = setInterval(
      () => void tick(),
      state === "starting" || state === "stopping"
        ? GPU_ACTIVE_POLL_MS
        : GPU_IDLE_POLL_MS,
    );
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [enabled, baseUrl, fetchWithHeaders, state, quiet, nonce]);

  const start = useCallback(
    async (body: GpuStartRequest) => {
      setPending(true);
      setError(null);
      settled.current = false;
      try {
        const result = await startGpu(baseUrl, fetchWithHeaders, body);
        setStatus(result.gpu);
        setLaunched(body);
      } catch (e) {
        // The backend's own coded refusal text (gpu.cli_missing names the
        // install line, gpu.launch_failed names the field or the tailnet).
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setPending(false);
        refresh();
      }
    },
    [baseUrl, fetchWithHeaders, refresh],
  );

  const stop = useCallback(async () => {
    setPending(true);
    setError(null);
    settled.current = false;
    try {
      setStatus(await stopGpu(baseUrl, fetchWithHeaders));
      setLaunched(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
      refresh();
    }
  }, [baseUrl, fetchWithHeaders, refresh]);

  return { status, pending, error, start, stop, refresh, launched };
}

/* -------------------------------------------------------------------------
 * WHICH WORKSPACE PAYS
 * ---------------------------------------------------------------------- */

/** Remembered per Lab (the API base is not part of the key, exactly as
 * useSelectedModel's is not): an operator works one machine at a time, and a
 * profile that followed them to a different Lab would be a profile that
 * machine may not even have. */
const PROFILE_KEY = "makermodslab.gpuModalProfile";
const ENVIRONMENT_KEY = "makermodslab.gpuModalEnvironment";

function read(key: string): string {
  try {
    return localStorage.getItem(key) ?? "";
  } catch {
    // Storage unavailable (private mode). In-memory state still works, and the
    // active target is the fallback either way.
    return "";
  }
}

function write(key: string, value: string): void {
  try {
    if (value) localStorage.setItem(key, value);
    else localStorage.removeItem(key);
  } catch {
    // Same: a preference that cannot be persisted is still a live selection.
  }
}

export interface UseGpuTargets {
  /** The listing, or null before the first answer. */
  targets: GpuTargets | null;
  /** The EFFECTIVE selection — what a launch will be sent with. Empty when
   * nothing could be listed, which is the backend's "let the CLI decide". */
  profile: string;
  environment: string;
  setProfile: (name: string) => void;
  setEnvironment: (name: string) => void;
}

/**
 * This machine's Modal profiles and the selected profile's environments.
 *
 * Two rules shape it, and both are about not lying about who pays:
 *
 *  - **A remembered value that no longer exists falls back to the active one,
 *    silently.** A stale localStorage entry (the profile was renamed, or this
 *    is a different machine) must not leave the panel showing a target that
 *    cannot be launched — and must not block Start GPU either.
 *  - **The environments are re-listed when the profile changes**, because
 *    `modal environment list` describes ONE profile's workspace. Showing the
 *    previous workspace's environments under a newly picked profile is exactly
 *    the mislead this feature exists to prevent.
 *
 * @param enabled fetch while true.
 */
export function useGpuTargets(enabled: boolean): UseGpuTargets {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [targets, setTargets] = useState<GpuTargets | null>(null);
  // What the LISTING is requested for. State (not derived) because it is an
  // input to the fetch; the displayed selection below is derived from it.
  const [wantedProfile, setWantedProfile] = useState<string>(() =>
    read(PROFILE_KEY),
  );
  const [wantedEnvironment, setWantedEnvironment] = useState<string>(() =>
    read(ENVIRONMENT_KEY),
  );

  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    void (async () => {
      try {
        setTargets(
          await getGpuTargets(
            baseUrl,
            fetchWithHeaders,
            wantedProfile,
            controller.signal,
          ),
        );
      } catch {
        // Transient. The listing is a convenience — leaving the last answer in
        // place beats blanking a panel the operator is reading, and Start GPU
        // never depended on it.
      }
    })();
    return () => controller.abort();
  }, [enabled, baseUrl, fetchWithHeaders, wantedProfile]);

  // The one place a stale remembered profile is dropped: the machine answered,
  // it has profiles, and none of them is the remembered one. Clearing it
  // re-runs the fetch against the active profile.
  useEffect(() => {
    if (!targets || !wantedProfile) return;
    if (targets.profiles.length === 0) return;
    if (targets.profiles.some((p) => p.name === wantedProfile)) return;
    write(PROFILE_KEY, "");
    setWantedProfile("");
  }, [targets, wantedProfile]);

  const profiles = targets?.profiles ?? [];
  const environments = targets?.environments ?? [];

  const activeProfile = profiles.find((p) => p.active)?.name ?? "";
  const activeEnvironment = environments.find((e) => e.active)?.name ?? "";

  // Derived rather than stored, so a remembered value that does not exist in
  // the CURRENT listing simply reads as the active one. This is what makes the
  // fallback silent for the environment too, with no second clearing effect:
  // switching profile switches workspace, and the old environment usually
  // isn't in the new one.
  const profile = profiles.some((p) => p.name === wantedProfile)
    ? wantedProfile
    : activeProfile;
  const environment = environments.some((e) => e.name === wantedEnvironment)
    ? wantedEnvironment
    : activeEnvironment;

  const setProfile = useCallback((name: string) => {
    write(PROFILE_KEY, name);
    setWantedProfile(name);
  }, []);

  const setEnvironment = useCallback((name: string) => {
    write(ENVIRONMENT_KEY, name);
    setWantedEnvironment(name);
  }, []);

  return { targets, profile, environment, setProfile, setEnvironment };
}
