import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  Boxes,
  Check,
  Loader2,
  Pause,
  Play,
  SkipBack,
  SkipForward,
  Trash2,
  VideoOff,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { useStudio } from "@/contexts/StudioContext";
import { useSelectedDataset } from "@/hooks/useSelectedDataset";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import DatasetInfoCard from "@/components/landing/DatasetInfoCard";
import JointPositionChart from "@/components/dialogs/JointPositionChart";
import DeleteConfirmDialog from "@/components/dialogs/DeleteConfirmDialog";
import { DeleteResolution } from "@/lib/deleteSemantics";
import {
  deleteDataset,
  deleteEpisode,
  EpisodeJointSeries,
  EpisodeSummary,
  episodeVideoUrl,
  getDatasetHubStatus,
  getDatasetInfo,
  getEpisodeJoints,
  HubStatus,
  listEpisodes,
} from "@/lib/replayApi";
import { ApiError } from "@/lib/apiClient";

export interface DatasetDetailDialogProps {
  repoId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called when an action navigates to the studio, so a parent surface that
   * would otherwise cover the studio (e.g. the library sheet) can close too. */
  onStudioAction?: () => void;
  /** Called after the dataset itself (not just an episode) is deleted, so a
   * parent list (LibrarySheet, CollectPanel) can refresh. Not called for a
   * "remove local copy" action that keeps the dataset listed as Hub-only. */
  onDeleted?: () => void;
  /** Post-recording review mode (set by CollectPanel right after a session
   * ends). While set: the dialog can't be dismissed by ESC / outside-click /
   * close button, and its footer shows Finalize instead of Train. onFinalize
   * fires when the user clicks Finalize. onDiscarded fires (IN ADDITION to
   * onDeleted, not instead of it) when the whole dataset gets deleted during
   * review — there's nothing left to finalize. */
  finalize?: {
    onFinalize: () => void;
    onDiscarded: () => void;
  };
}

// Best (cols, tileW, tileH) for `n` tiles inside a box of `boxW` x `boxH`:
// the search over column counts that a video-gallery layout needs so the
// camera grid keeps growing/reflowing as cameras are added or the window is
// resized, instead of a fixed N-up template.
function computeCameraLayout(
  n: number,
  boxW: number,
  boxH: number,
  gap = 8,
  aspect = 16 / 10,
) {
  let best = { cols: 1, tileW: boxW, tileH: boxW / aspect };
  let bestArea = -1;
  for (let cols = 1; cols <= n; cols++) {
    const rows = Math.ceil(n / cols);
    const cellW = (boxW - gap * (cols - 1)) / cols;
    const cellH = (boxH - gap * (rows - 1)) / rows;
    let tileW = cellW;
    let tileH = tileW / aspect;
    if (tileH > cellH) {
      tileH = cellH;
      tileW = tileH * aspect;
    }
    const area = tileW * tileH;
    if (area > bestArea) {
      bestArea = area;
      best = { cols, tileW, tileH };
    }
  }
  return best;
}

const fmtTime = (t: number) => `${t.toFixed(1)}s`;

/**
 * Episode viewer: a growable grid of camera feeds (one per dataset camera —
 * reflows on resize and as camera count changes, never a fixed 2-up) with
 * transport controls and a joint-position chart synced to the playhead. Real
 * <video> elements pointing at /datasets/episode-video (Range-request backed,
 * so scrubbing seeks without buffering the whole file); no local data means no
 * episode list, so this renders a locked "not viewable yet" state instead.
 */
const EpisodeViewer: React.FC<{
  repoId: string;
  cameras: string[];
  episodes: EpisodeSummary[];
  selectedEpisode: number;
  onSelectEpisode: (episodeIndex: number) => void;
}> = ({ repoId, cameras, episodes, selectedEpisode, onSelectEpisode }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [joints, setJoints] = useState<EpisodeJointSeries | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoErrors, setVideoErrors] = useState<Record<string, boolean>>({});

  const episode = episodes.find((e) => e.episode_index === selectedEpisode) ?? null;
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  // The camera whose onTimeUpdate drives currentTime/the end-of-episode
  // boundary check. Falls back off cameras[0] onto the next camera that
  // hasn't errored — if the "primary" camera fails to decode, its <video>
  // element is replaced by an error placeholder and never fires
  // onTimeUpdate, so pinning this to a fixed index would silently stall
  // playback tracking (auto-pause, the loop-to-start fix, the scrubber)
  // for the rest of the session.
  const primaryCamera = cameras.find((c) => !videoErrors[c]) ?? cameras[0];
  // v3.0 packs consecutive episodes into the same physical mp4 per camera —
  // this episode's slice starts at `from` within that (possibly shared)
  // file, not at 0. Every seek/play/boundary check goes through this.
  const offsetFor = useCallback(
    (camera: string) => episode?.video_offsets?.[camera]?.from ?? 0,
    [episode],
  );

  useEffect(() => {
    const controller = new AbortController();
    setJoints(null);
    getEpisodeJoints(baseUrl, fetchWithHeaders, repoId, selectedEpisode, controller.signal)
      .then(setJoints)
      .catch(() => {
        if (!controller.signal.aborted) setJoints(null);
      });
    return () => controller.abort();
  }, [baseUrl, fetchWithHeaders, repoId, selectedEpisode]);

  // Warm the backend's cache for the NEXT episode while this one plays, so
  // advancing often lands on an already-fetched chunk instead of repeating
  // the same first-time delay (see docs/superpowers/specs/2026-07-24-episode-
  // prefetch-design.md). A 1-byte Range request drives get_episode_video_path
  // all the way through — including a Hub hf_hub_download fetch for a
  // not-yet-cached chunk — without downloading video the user hasn't asked
  // for yet; FileResponse already supports Range requests (it's how the real
  // <video> tag seeks). The joints response is discarded — its fetch alone is
  // what warms the data chunk server-side. Best-effort: any failure here has
  // zero effect on the real fetch that runs when the user actually navigates
  // there, so errors are silently swallowed.
  useEffect(() => {
    const next = episodes[episodes.findIndex((e) => e.episode_index === selectedEpisode) + 1];
    if (!next) return;
    const controller = new AbortController();
    // Debounced: don't kick off a prefetch until the user has settled on
    // this episode for a moment. Without this, rapid navigation (arrow-key
    // taps, clicking through the list) fires — and immediately abandons via
    // the abort below — a prefetch for every episode passed through on the
    // way to wherever the user actually stops. The frontend abort doesn't
    // necessarily cancel an in-flight Hub download the backend already
    // started, so an unthrottled burst can leave several abandoned
    // hf_hub_download calls running server-side.
    const timer = window.setTimeout(() => {
      for (const camera of cameras) {
        fetchWithHeaders(episodeVideoUrl(baseUrl, repoId, next.episode_index, camera), {
          headers: { Range: "bytes=0-0" },
          signal: controller.signal,
        }).catch(() => {});
      }
      getEpisodeJoints(baseUrl, fetchWithHeaders, repoId, next.episode_index, controller.signal).catch(() => {});
    }, 400);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [baseUrl, fetchWithHeaders, repoId, episodes, cameras, selectedEpisode]);

  useEffect(() => {
    setPlaying(false);
    setCurrentTime(0);
    setVideoErrors({});
  }, [selectedEpisode]);

  const forEachVideo = useCallback((fn: (v: HTMLVideoElement) => void) => {
    Object.values(videoRefs.current).forEach((v) => v && fn(v));
  }, []);

  const handlePlayPause = useCallback(() => {
    if (playing) {
      forEachVideo((v) => v.pause());
      setPlaying(false);
    } else {
      // Parked at (or past) the episode's own end: the underlying mp4 keeps
      // going into the next episode's frames past this point (see the
      // v3.0-packing note on offsetFor above), so resuming here would spill
      // into that footage before the boundary check in
      // handlePrimaryTimeUpdate catches up. Loop back to this episode's
      // start instead.
      if (episode && currentTime >= episode.duration - 0.02) {
        for (const [camera, v] of Object.entries(videoRefs.current)) {
          if (v) v.currentTime = offsetFor(camera);
        }
        setCurrentTime(0);
      }
      forEachVideo((v) => {
        v.play().catch(() => {});
      });
      setPlaying(true);
    }
  }, [playing, forEachVideo, episode, currentTime, offsetFor]);

  // Drives currentTime/scrubber from the primary camera's raw (file-relative)
  // playback position, and is the one place that notices the episode's own
  // slice has ended — native `ended` usually doesn't fire here since the
  // underlying file usually keeps going into the next episode's frames.
  const handlePrimaryTimeUpdate = (rawTime: number) => {
    if (!episode) return;
    const offset = offsetFor(primaryCamera);
    const t = Math.max(0, Math.min(episode.duration, rawTime - offset));
    setCurrentTime(t);
    if (rawTime >= offset + episode.duration - 0.02) {
      forEachVideo((v) => v.pause());
      setPlaying(false);
    }
  };

  // Fallback for the (possible) case where this episode's slice is the
  // physical file's own end — e.g. the last episode in a chunk — and the
  // video's real duration falls short of `episode.duration` (independently
  // derived from length/fps, not from the file itself). Native `ended` fires
  // here before the timeupdate threshold above ever would, and without this
  // handler `playing` would stay stuck true forever since no further
  // timeupdate events arrive once playback has actually stopped.
  const handlePrimaryEnded = () => {
    forEachVideo((v) => v.pause());
    setPlaying(false);
  };

  const handleSeek = (t: number) => {
    const clamped = Math.max(0, Math.min(episode?.duration ?? 0, t));
    for (const [camera, v] of Object.entries(videoRefs.current)) {
      if (v) v.currentTime = offsetFor(camera) + clamped;
    }
    setCurrentTime(clamped);
  };

  const gridWrapRef = useRef<HTMLDivElement>(null);
  const [layout, setLayout] = useState({ cols: 1, tileW: 100, tileH: 60 });
  useEffect(() => {
    const el = gridWrapRef.current;
    if (!el) return;
    const recompute = () => {
      const rect = el.getBoundingClientRect();
      setLayout(
        computeCameraLayout(
          Math.max(1, cameras.length),
          Math.max(80, rect.width - 16),
          Math.max(80, rect.height - 16),
        ),
      );
    };
    recompute();
    const ro = new ResizeObserver(recompute);
    ro.observe(el);
    return () => ro.disconnect();
  }, [cameras.length]);

  const scrubFrac = episode?.duration ? currentTime / episode.duration : 0;

  const scrubRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);
  const seekFromClientX = (clientX: number) => {
    const rect = scrubRef.current?.getBoundingClientRect();
    if (!rect || !episode) return;
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    handleSeek(frac * episode.duration);
  };

  const gotoEpisode = useCallback(
    (delta: number) => {
      const i = episodes.findIndex((e) => e.episode_index === selectedEpisode);
      const next = episodes[i + delta];
      if (next) onSelectEpisode(next.episode_index);
    },
    [episodes, selectedEpisode, onSelectEpisode],
  );

  const currentIndex = episodes.findIndex((e) => e.episode_index === selectedEpisode);
  const hasPrev = currentIndex > 0;
  const hasNext = currentIndex >= 0 && currentIndex < episodes.length - 1;

  // Mirrors the guarded-window-listener convention in RecordingSessionDialog:
  // skip when a text field has focus (the info-card tag input in the side
  // panel), preventDefault on the keys we do handle so Space doesn't scroll
  // the page or double-fire a focused button's native click.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      // Ignore OS auto-repeat from a held key — holding Space shouldn't
      // rapid-toggle play/pause, and holding an arrow shouldn't blow through
      // the episode list faster than an actual keypress.
      if (e.repeat) return;
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
        return;
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        gotoEpisode(-1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        gotoEpisode(1);
      } else if (e.key === " " || e.code === "Space") {
        // A focused button already activates on Space natively, and its
        // onClick calls the same handler we'd call here (Prev/Play/Next/
        // episode rows) — don't hijack that, or tabbing to any button in
        // this dialog would silently turn Space into a play/pause toggle
        // instead of activating the focused control.
        if (target && target.tagName === "BUTTON") return;
        e.preventDefault();
        handlePlayPause();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [gotoEpisode, handlePlayPause]);

  if (cameras.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center rounded-md border border-border bg-muted/30 text-sm text-muted-foreground">
        This dataset has no camera data to view.
      </div>
    );
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div
        ref={gridWrapRef}
        className="relative flex-1 overflow-auto rounded-md border border-border bg-[#0c0f14]"
      >
        <div
          className="grid justify-center content-center gap-2 p-2"
          style={{
            gridTemplateColumns: `repeat(${layout.cols}, ${layout.tileW}px)`,
            gridAutoRows: `${layout.tileH}px`,
          }}
        >
          {cameras.map((camera) => (
            <div
              key={camera}
              className="relative overflow-hidden rounded border border-white/10 bg-black"
            >
              <span className="absolute left-1.5 top-1.5 z-10 rounded bg-black/50 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-zinc-100">
                {camera}
              </span>
              {videoErrors[camera] ? (
                <div className="flex h-full w-full flex-col items-center justify-center gap-1 px-3 text-center text-zinc-300">
                  <VideoOff className="h-4 w-4" />
                  <p className="text-[10px] leading-snug">
                    Can't decode this camera's video in this browser.
                  </p>
                </div>
              ) : (
                <video
                  key={`${repoId}:${selectedEpisode}:${camera}`}
                  ref={(el) => {
                    videoRefs.current[camera] = el;
                  }}
                  src={episodeVideoUrl(baseUrl, repoId, selectedEpisode, camera)}
                  className="h-full w-full object-cover"
                  muted
                  playsInline
                  onLoadedMetadata={(e) => {
                    e.currentTarget.currentTime = offsetFor(camera);
                  }}
                  onTimeUpdate={
                    camera === primaryCamera
                      ? (e) => handlePrimaryTimeUpdate(e.currentTarget.currentTime)
                      : undefined
                  }
                  onEnded={camera === primaryCamera ? handlePrimaryEnded : undefined}
                  onError={() =>
                    setVideoErrors((prev) => ({ ...prev, [camera]: true }))
                  }
                />
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="icon"
          className="h-8 w-8 shrink-0"
          onClick={() => gotoEpisode(-1)}
          disabled={!episode || !hasPrev}
          aria-label="Previous episode"
        >
          <SkipBack className="h-3.5 w-3.5" />
        </Button>
        <Button
          size="icon"
          className="h-8 w-8 shrink-0"
          onClick={handlePlayPause}
          disabled={!episode}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
        </Button>
        <Button
          variant="outline"
          size="icon"
          className="h-8 w-8 shrink-0"
          onClick={() => gotoEpisode(1)}
          disabled={!episode || !hasNext}
          aria-label="Next episode"
        >
          <SkipForward className="h-3.5 w-3.5" />
        </Button>
        <span className="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
          {episode ? `${fmtTime(currentTime)} / ${fmtTime(episode.duration)}` : "—"}
        </span>
        <div
          ref={scrubRef}
          className="relative h-5 flex-1 cursor-pointer"
          onPointerDown={(e) => {
            draggingRef.current = true;
            e.currentTarget.setPointerCapture(e.pointerId);
            forEachVideo((v) => v.pause());
            setPlaying(false);
            seekFromClientX(e.clientX);
          }}
          onPointerMove={(e) => {
            if (draggingRef.current) seekFromClientX(e.clientX);
          }}
          onPointerUp={(e) => {
            draggingRef.current = false;
            e.currentTarget.releasePointerCapture(e.pointerId);
          }}
        >
          <div className="absolute inset-y-0 my-auto h-1 w-full rounded-full bg-muted" />
          <div
            className="absolute inset-y-0 my-auto h-1 rounded-full bg-foreground"
            style={{ width: `${scrubFrac * 100}%` }}
          />
          <div
            className="absolute top-1/2 h-3 w-3 -translate-y-1/2 rounded-full bg-foreground shadow"
            style={{ left: `calc(${scrubFrac * 100}% - 6px)` }}
          />
        </div>
      </div>

      <JointPositionChart joints={joints} episode={episode} scrubFrac={scrubFrac} />
    </div>
  );
};

const DatasetDetailDialog: React.FC<DatasetDetailDialogProps> = ({
  repoId,
  open,
  onOpenChange,
  onStudioAction,
  onDeleted,
  finalize,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { openStudio } = useStudio();
  const { setSelectedDataset } = useSelectedDataset();
  const { toast } = useToast();

  const [episodes, setEpisodes] = useState<EpisodeSummary[] | null>(null);
  const [episodesLoading, setEpisodesLoading] = useState(true);
  const [cameras, setCameras] = useState<string[]>([]);
  const [selectedEpisode, setSelectedEpisode] = useState<number | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [deleteTarget, setDeleteTarget] = useState<EpisodeSummary | null>(null);
  const [deleteHubStatus, setDeleteHubStatus] = useState<HubStatus | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [pendingDatasetDelete, setPendingDatasetDelete] = useState(false);

  useEffect(() => {
    if (!repoId || !open) return;
    const controller = new AbortController();
    setEpisodesLoading(true);
    setEpisodes(null);
    setCameras([]);
    Promise.all([
      listEpisodes(baseUrl, fetchWithHeaders, repoId, controller.signal).catch(() => null),
      getDatasetInfo(baseUrl, fetchWithHeaders, repoId, controller.signal).catch(() => null),
    ]).then(([eps, info]) => {
      if (controller.signal.aborted) return;
      setEpisodes(eps);
      setCameras(info?.cameras ?? []);
      setSelectedEpisode(eps && eps.length > 0 ? eps[0].episode_index : null);
      setEpisodesLoading(false);
    });
    return () => controller.abort();
  }, [repoId, open, baseUrl, fetchWithHeaders, reloadKey]);

  useEffect(() => {
    if (!(deleteTarget || pendingDatasetDelete) || !repoId) {
      setDeleteHubStatus(null);
      return;
    }
    const controller = new AbortController();
    getDatasetHubStatus(baseUrl, fetchWithHeaders, repoId, controller.signal)
      .then(setDeleteHubStatus)
      // Fall back to a definitive (not null) status so the whole-dataset
      // confirm dialog below can still open on a fetch failure instead of
      // staying stuck forever — see the `deleteHubStatus !== null` gate.
      .catch(() =>
        setDeleteHubStatus({ repo_id: repoId, status: "local_only", url: null }),
      );
    return () => controller.abort();
  }, [deleteTarget, pendingDatasetDelete, repoId, baseUrl, fetchWithHeaders]);

  if (!repoId) return null;

  const isLastEpisode = episodes !== null && episodes.length === 1;

  const openDeleteConfirm = (ep: EpisodeSummary) => {
    setDeleteError(null);
    setDeleteTarget(ep);
  };

  const handleConfirmDelete = async () => {
    if (!deleteTarget) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      if (isLastEpisode) {
        const result = await deleteDataset(baseUrl, fetchWithHeaders, repoId);
        if (!result.success) {
          setDeleteError(result.message ?? "Something went wrong");
          return;
        }
        setDeleteTarget(null);
        onOpenChange(false);
        onDeleted?.();
        finalize?.onDiscarded();
        return;
      }
      await deleteEpisode(
        baseUrl,
        fetchWithHeaders,
        repoId,
        deleteTarget.episode_index,
      );
      setDeleteTarget(null);
      // Re-fetch rather than filtering the deleted row out of local state:
      // lerobot's delete_episodes renumbers every surviving episode to stay
      // contiguous from 0 (old episode 2 of [0,1,2] becomes episode 1 of
      // [0,1]), so carrying forward the OLD episode_index values on the
      // remaining rows would show one episode's data under another's label.
      // This also remounts DatasetInfoCard (its key includes reloadKey), so
      // it re-reads the now-stale episode/frame counts.
      setReloadKey((k) => k + 1);
    } catch (err) {
      setDeleteError(
        err instanceof ApiError ? (err.detail ?? err.message) : "Something went wrong",
      );
    } finally {
      setDeleting(false);
    }
  };

  const confirmDatasetDelete = async (resolution: DeleteResolution) => {
    if (!repoId) return;
    try {
      const result = await deleteDataset(baseUrl, fetchWithHeaders, repoId);
      if (!result.success) {
        setPendingDatasetDelete(false);
        toast({
          title: "Delete failed",
          description: result.message ?? "Something went wrong",
          variant: "destructive",
        });
        return;
      }
      setPendingDatasetDelete(false);
      if (resolution.clearsSelection) {
        onOpenChange(false);
        onDeleted?.();
        finalize?.onDiscarded();
      } else {
        setReloadKey((k) => k + 1);
      }
    } catch (e) {
      setPendingDatasetDelete(false);
      toast({
        title: "Delete failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handleTrain = () => {
    setSelectedDataset(repoId);
    openStudio("train", { train: { datasetRepoId: repoId } });
    onOpenChange(false);
    onStudioAction?.();
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        hideClose={!!finalize}
        onEscapeKeyDown={finalize ? (e) => e.preventDefault() : undefined}
        onPointerDownOutside={finalize ? (e) => e.preventDefault() : undefined}
        onInteractOutside={finalize ? (e) => e.preventDefault() : undefined}
        className="flex h-[85vh] max-w-6xl flex-col gap-0 overflow-hidden p-0"
      >
        <DialogHeader className="shrink-0 space-y-0 border-b border-border px-6 py-4 text-left">
          <p className="eyebrow">MakerMods Lab Dataset Viewer</p>
          <DialogTitle className="break-all pt-1 font-mono text-base font-semibold">
            {repoId}
          </DialogTitle>
          {finalize && (
            <p className="pt-1 text-sm text-muted-foreground">
              Review your recording — delete any bad episodes, then finalize.
            </p>
          )}
        </DialogHeader>

        <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_300px]">
          <div className="flex min-h-0 min-w-0 flex-col gap-3 p-4">
            {episodesLoading ? (
              <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
                Loading episodes…
              </div>
            ) : episodes && episodes.length > 0 && selectedEpisode != null ? (
              <EpisodeViewer
                repoId={repoId}
                cameras={cameras}
                episodes={episodes}
                selectedEpisode={selectedEpisode}
                onSelectEpisode={setSelectedEpisode}
              />
            ) : (
              <div className="flex flex-1 flex-col items-center justify-center gap-2 rounded-md border border-dashed border-border bg-muted/30 px-6 text-center">
                <VideoOff className="h-6 w-6 text-muted-foreground" />
                <p className="text-sm font-medium text-foreground">
                  {episodes && episodes.length === 0
                    ? "No episodes recorded yet"
                    : "No viewable footage yet"}
                </p>
                <p className="max-w-sm text-xs text-muted-foreground">
                  {episodes && episodes.length === 0
                    ? "Record at least one episode into this dataset to view its camera footage here."
                    : "A Hub dataset with video streams on demand here — a first-time view of a new episode may take a moment to fetch. This message means the dataset predates the viewer's format, or has no video to show."}
                </p>
              </div>
            )}
          </div>

          <div className="flex min-h-0 flex-col divide-y divide-border overflow-y-auto border-l border-border">
            <div className="flex min-h-0 flex-1 flex-col p-3">
              <p className="eyebrow mb-2">episodes {episodes ? `(${episodes.length})` : ""}</p>
              <div className="min-h-0 flex-1 overflow-y-auto">
                {episodes && episodes.length > 0 ? (
                  <div className="space-y-0.5">
                    {episodes.map((ep) => (
                      <ContextMenu key={ep.episode_index}>
                        <ContextMenuTrigger asChild>
                          <div
                            className={`group flex w-full items-center gap-1 rounded-md px-2 py-1.5 text-xs transition-colors hover:bg-accent ${
                              selectedEpisode === ep.episode_index
                                ? "border border-border bg-accent"
                                : "border border-transparent"
                            }`}
                          >
                            <button
                              type="button"
                              onClick={() => setSelectedEpisode(ep.episode_index)}
                              className="flex min-w-0 flex-1 items-center gap-2 text-left"
                            >
                              <span className="w-6 shrink-0 font-mono text-[11px] text-muted-foreground">
                                {String(ep.episode_index).padStart(2, "0")}
                              </span>
                              <span className="min-w-0 flex-1 truncate">
                                Episode {ep.episode_index}
                              </span>
                              <span className="shrink-0 font-mono text-[10.5px] tabular-nums text-muted-foreground">
                                {ep.duration.toFixed(1)}s
                              </span>
                            </button>
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                openDeleteConfirm(ep);
                              }}
                              aria-label={`Delete episode ${ep.episode_index}`}
                              title="Delete episode"
                              className="shrink-0 rounded p-1 text-muted-foreground opacity-0 hover:text-destructive group-hover:opacity-100"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </button>
                          </div>
                        </ContextMenuTrigger>
                        <ContextMenuContent>
                          <ContextMenuItem onSelect={() => openDeleteConfirm(ep)}>
                            <Trash2 className="mr-2 h-3.5 w-3.5" />
                            Delete episode
                          </ContextMenuItem>
                        </ContextMenuContent>
                      </ContextMenu>
                    ))}
                  </div>
                ) : (
                  <p className="px-1 text-xs leading-relaxed text-muted-foreground">
                    Episodes appear here once this dataset is downloaded to your
                    machine.
                  </p>
                )}
              </div>
            </div>

            <div className="p-3">
              <DatasetInfoCard
                key={reloadKey}
                repoId={repoId}
                canDelete
                onDelete={() => setPendingDatasetDelete(true)}
                onDownloaded={() => setReloadKey((k) => k + 1)}
              />
            </div>

            <div className="p-3">
              {finalize ? (
                <Button onClick={finalize.onFinalize} className="w-full gap-2">
                  <Check className="h-4 w-4" />
                  Finalize
                </Button>
              ) : (
                <Button onClick={handleTrain} className="w-full gap-2">
                  <Boxes className="h-4 w-4" />
                  Train a skill from this
                </Button>
              )}
            </div>
          </div>
        </div>

        <AlertDialog
          open={deleteTarget !== null}
          onOpenChange={(next) => {
            if (!next && !deleting) {
              setDeleteTarget(null);
              setDeleteError(null);
            }
          }}
        >
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>
                {isLastEpisode
                  ? "Delete the whole dataset?"
                  : `Delete episode ${deleteTarget?.episode_index}?`}
              </AlertDialogTitle>
              <AlertDialogDescription>
                {isLastEpisode ? (
                  <>
                    Episode {deleteTarget?.episode_index} is the only episode
                    in <span className="font-mono text-foreground">{repoId}</span>.
                    Deleting it deletes the whole dataset. This can't be undone.
                    {deleteHubStatus?.status === "on_hub" && (
                      <>
                        {" "}
                        This dataset is on the Hugging Face Hub — the Hub copy
                        will be unaffected and will still exist there.
                      </>
                    )}
                  </>
                ) : (
                  <>
                    This removes episode {deleteTarget?.episode_index} from{" "}
                    <span className="font-mono text-foreground">{repoId}</span>{" "}
                    and can't be undone.
                    {deleteHubStatus?.status === "on_hub" && (
                      <>
                        {" "}
                        This dataset is on the Hugging Face Hub — the Hub
                        copy will NOT be updated. Use "Upload to Hub" to push
                        the edited version.
                      </>
                    )}
                  </>
                )}
                {deleteError && (
                  <span className="mt-2 block text-destructive">{deleteError}</span>
                )}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
              <AlertDialogAction
                disabled={deleting}
                onClick={(e) => {
                  e.preventDefault();
                  handleConfirmDelete();
                }}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                {deleting ? (
                  <Loader2 className="mr-1 h-4 w-4 animate-spin" />
                ) : (
                  <Trash2 className="mr-1 h-4 w-4" />
                )}
                {deleting
                  ? "Deleting…"
                  : isLastEpisode
                    ? "Delete dataset"
                    : "Delete episode"}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        <DeleteConfirmDialog
          kind="dataset"
          // Wait for deleteHubStatus to resolve before opening: it starts
          // null the instant pendingDatasetDelete flips true, and treating
          // that as "local" would briefly show the permanent-delete framing
          // for a dataset that's actually on the Hub (only its local copy
          // would go) — the opposite of LibrarySheet's version of this same
          // dialog, which has the real source synchronously and never races.
          item={
            pendingDatasetDelete && deleteHubStatus !== null
              ? { source: deleteHubStatus.status === "on_hub" ? "both" : "local" }
              : null
          }
          label={repoId ?? undefined}
          onOpenChange={(o) => !o && setPendingDatasetDelete(false)}
          onConfirm={confirmDatasetDelete}
        />
      </DialogContent>
    </Dialog>
  );
};

export default DatasetDetailDialog;
